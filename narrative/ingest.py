"""Ingestion of the narrative profile: Alg. 2 and Alg. 3 of the paper with observers and claims.

Fast path, with no LLM call: the event of a turn gets its visibility from the
scene (`present`), its action and speech parts split by the turn's marks, and a
MENTIONS edge to its speaker with the role ACTOR; the rest is Alg. 2 as in the
magma profile, except that SIMILAR_TO looks only at earlier events.

Slow path: one call to the extended C.1 prompt gives, besides the paper's
fields, the role of each entity and the claims of the speech. Each claim
becomes a CLAIM node, a facet of the event (STATED), with the event's
visibility, its own vector and BM25 entry, MENTIONS edges to what it is about
and SIMILAR_TO edges to close earlier claims. Dates and causal links are the
paper's.
"""

from __future__ import annotations

import queue
import threading

from pydantic import BaseModel, Field, field_validator

from magma.config import DEFAULT, IngestConfig, TemporalConfig
from magma.embed import Embedder, Vector
from magma.graph import Edge, GraphStore, Node
from magma.index import LexicalIndex, VectorIndex
from magma.ingest import (
    EXTRACTION_SYSTEM_PROMPT,
    EventExtraction,
    FastPathIngestion,
    SlowPathConsolidation,
    Turn,
    entity_id,
    entity_key,
)
from magma.json_types import JSONObject, JSONValue
from magma.llm import ContextWindowExceeded, InvalidStructuredOutput, LLMClient, Message
from magma.schema import MagmaEdge, MagmaNode
from magma.temporal import TemporalTagger
from narrative.config import DEFAULT_NARRATIVE, NarrativeConfig
from narrative.scene import describe_audience, split_turn, visible_to
from narrative.schema import ClaimType, MentionRole, NarrativeEdge, NarrativeNode

AGENT_KIND = "agent"  # the only entity kind a mechanism reads: the speakers


class SharedWrites:
    """The writes both paths make, behind one lock: the paths run on different threads.

    In the magma profile only the fast path indexes and only the slow path
    creates entities. Here the fast path also creates the speaker's entity and
    the slow path also indexes claims, so entity creation and every search-then-add
    on the indexes take this lock.
    """

    def __init__(self, graph: GraphStore) -> None:
        self._graph = graph
        self.lock = threading.RLock()

    def ensure(self, name: str, kind: str | None = None) -> tuple[str, bool] | None:
        """The id of the entity named `name`, and whether it was created now; None for an empty name."""
        key = entity_key(name)
        if not key:
            return None
        node_id = entity_id(key)
        with self.lock:
            if not self._graph.has_node(node_id):
                attributes: JSONObject = {"kind": kind} if kind else {}
                self._graph.add_node(Node(id=node_id, type=MagmaNode.ENTITY, text=name.strip(), attributes=attributes))
                return node_id, True
            current = self._graph.node(node_id)
            if kind and current.attributes.get("kind") != kind:
                attributes = dict(current.attributes)
                attributes["kind"] = kind
                self._graph.replace_node(Node(id=node_id, type=current.type, text=current.text, attributes=attributes))
            return node_id, False


class NarrativeFastPath(FastPathIngestion):
    """Alg. 2 with the scene: visibility, action and speech, and the speaker as ACTOR."""

    def __init__(
        self,
        graph: GraphStore,
        vectors: VectorIndex,
        keywords: LexicalIndex,
        embedder: Embedder,
        consolidation_queue: queue.Queue[str],
        shared: SharedWrites,
        config: IngestConfig = DEFAULT.ingest,
        narrative: NarrativeConfig = DEFAULT_NARRATIVE,
    ) -> None:
        super().__init__(graph, vectors, keywords, embedder, consolidation_queue, config)
        self._shared = shared
        self._narrative = narrative
        self._event_ids = {node.id for node in graph.nodes(MagmaNode.EVENT)}

    def ingest(self, turn: Turn) -> Node:
        """Alg. 2, with the speaker's MENTIONS added before the event is queued for the slow path."""
        event = self._segment_event(turn)
        previous = self._last_event
        self._graph.add_node(event)
        if previous is not None:
            self._graph.add_edge(Edge(source=previous.id, target=event.id, type=MagmaEdge.NEXT))
        self._link_actor(event, turn)

        vector = self._embed(event)
        with self._shared.lock:
            self._link_similar_events(event, vector)
            self._vectors.add(event.id, vector)
            self._keywords.add(event.id, event.text)
        self._event_ids.add(event.id)

        self._queue.put(event.id)
        self._event_count += 1
        self._last_event = event
        return event

    def _segment_event(self, turn: Turn) -> Node:
        node = super()._segment_event(turn)
        parts = split_turn(turn, self._narrative)
        attributes: JSONObject = dict(node.attributes)
        attributes["action_text"] = parts.action_text
        attributes["speech_text"] = parts.speech_text
        return Node(
            id=node.id,
            type=node.type,
            text=node.text,
            timestamp=node.timestamp,
            attributes=attributes,
            visible_to=visible_to(turn, self._narrative),
        )

    def _link_actor(self, event: Node, turn: Turn) -> None:
        if turn.speaker == self._narrative.narrator:
            return
        found = self._shared.ensure(turn.speaker, kind=AGENT_KIND)
        if found is None:
            return
        actor_id, _ = found
        self._graph.add_edge(
            Edge(source=event.id, target=actor_id, type=MagmaEdge.MENTIONS, attributes={"role": MentionRole.ACTOR.value})
        )

    def _link_similar_events(self, event: Node, vector: Vector) -> None:
        """As in Alg. 2, among earlier events only: claims share the vector index but are not events."""
        if not self._event_ids:
            return
        closest = self._vectors.search(vector, self._config.semantic_edge_max_neighbors, frozenset(self._event_ids))
        for hit in closest:
            if hit.score > self._config.semantic_edge_threshold:
                self._graph.add_edge(Edge(source=event.id, target=hit.node_id, type=MagmaEdge.SIMILAR_TO, weight=hit.score))


# Prompt C.1 with two more fields, in the same style; still one call per event.
NARRATIVE_EXTRACTION_SYSTEM_PROMPT = (
    EXTRACTION_SYSTEM_PROMPT
    + """
- "roles": List of objects, one per entity, each with "entity" (a name from "entities") and "role": ACTOR (who acts or speaks), TARGET (who or what the action or speech is directed at), RECIPIENT (who receives something), INSTRUMENT (what is used), LOCATION (where it happens) or OTHER.
- "claims": List of objects, one per proposition the speaker asserts or denies in the Speech, each with "statement" (one self-contained sentence with names instead of pronouns; for a denial, the proposition the speaker says is false), "about" (names from "entities" the statement is about) and "type" (ASSERTION when the speaker says it is true, DENIAL when the speaker says it is false). Take claims only from the Speech, never from the Action. Greetings, questions and requests are not claims. Return an empty list when there is no Speech."""
)

NARRATIVE_EXTRACTION_INPUT = """Input Data:
- Speaker: {speaker}
- Audience: {audience}
- Action: {action}
- Speech: {speech}
- Context: {prev_summary}"""

NO_PART = "none"


class EntityRole(BaseModel):
    entity: str
    role: MentionRole = MentionRole.OTHER

    @field_validator("role", mode="before")
    @classmethod
    def _unknown_role_is_other(cls, value: object) -> object:
        text = str(value).strip().upper()
        return text if text in MentionRole.__members__ else MentionRole.OTHER


class ExtractedClaim(BaseModel):
    statement: str
    about: list[str] = Field(default_factory=list)
    type: ClaimType = ClaimType.ASSERTION

    @field_validator("type", mode="before")
    @classmethod
    def _unknown_type_is_assertion(cls, value: object) -> object:
        text = str(value).strip().upper()
        return text if text in ClaimType.__members__ else ClaimType.ASSERTION


class NarrativeExtraction(EventExtraction):
    """Output of the extended prompt C.1."""

    roles: list[EntityRole] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)


def claim_id(event_id: str, number: int) -> str:
    return f"{event_id}.c{number}"


class NarrativeConsolidation(SlowPathConsolidation):
    """Alg. 3 with the extended C.1: entity roles and the claims of the speech."""

    def __init__(
        self,
        graph: GraphStore,
        llm: LLMClient,
        tagger: TemporalTagger,
        consolidation_queue: queue.Queue[str],
        vectors: VectorIndex,
        keywords: LexicalIndex,
        embedder: Embedder,
        shared: SharedWrites,
        config: IngestConfig = DEFAULT.ingest,
        temporal: TemporalConfig = DEFAULT.temporal,
    ) -> None:
        super().__init__(graph, llm, tagger, consolidation_queue, config, temporal)
        self._vectors = vectors
        self._keywords = keywords
        self._embedder = embedder
        self._shared = shared
        self._claim_ids = [node.id for node in graph.nodes(NarrativeNode.CLAIM)]

    def _extract(self, event: Node, errors: list[str]) -> EventExtraction | None:
        speaker = str(event.attributes.get("speaker", ""))
        user_input = NARRATIVE_EXTRACTION_INPUT.format(
            speaker=speaker,
            audience=describe_audience(event.visible_to, speaker),
            action=_part(event, "action_text"),
            speech=_part(event, "speech_text"),
            prev_summary=self._previous_summary(event),
        )
        messages: list[Message] = [
            {"role": "system", "content": NARRATIVE_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
        try:
            return self._llm.chat_structured(messages, NarrativeExtraction, tag="extraction").value
        except (InvalidStructuredOutput, ContextWindowExceeded) as error:
            errors.append(f"extraction: {error}")
            return None

    def _link_extraction(self, event: Node, extraction: EventExtraction) -> tuple[list[str], list[str]]:
        if not isinstance(extraction, NarrativeExtraction):
            raise TypeError(f"the narrative slow path extracts with its own prompt, got {type(extraction).__name__}")
        narrative = extraction
        mentioned, created = self._link_entities_with_roles(event, narrative)
        if _part(event, "speech_text") != NO_PART:  # an action alone states nothing
            for number, claim in enumerate(narrative.claims, start=1):
                created.extend(self._add_claim(event, number, claim))
        return mentioned, created

    def _link_entities_with_roles(self, event: Node, extraction: NarrativeExtraction) -> tuple[list[str], list[str]]:
        role_of = {entity_key(item.entity): item.role for item in extraction.roles}
        mentioned: list[str] = []
        created: list[str] = []
        for name in extraction.entities:
            found = self._shared.ensure(name)
            if found is None or found[0] in mentioned:
                continue
            node_id, is_new = found
            if is_new:
                created.append(node_id)
            mentioned.append(node_id)
            if not self._graph.has_edge(event.id, node_id, MagmaEdge.MENTIONS):
                role = role_of.get(entity_key(name), MentionRole.OTHER)
                self._graph.add_edge(
                    Edge(source=event.id, target=node_id, type=MagmaEdge.MENTIONS, attributes={"role": role.value})
                )
        return mentioned, created

    def _add_claim(self, event: Node, number: int, claim: ExtractedClaim) -> list[str]:
        """The claim node with its STATED, MENTIONS and SIMILAR_TO edges; returns the entities created for it."""
        statement = claim.statement.strip()
        node_id = claim_id(event.id, number)
        if not statement or self._graph.has_node(node_id):  # an empty statement, or consolidated before
            return []
        speaker = str(event.attributes.get("speaker", ""))
        about: list[JSONValue] = [name.strip() for name in claim.about if name.strip()]
        audience: list[JSONValue] = [listener for listener in sorted(event.visible_to)]
        node = Node(
            id=node_id,
            type=NarrativeNode.CLAIM,
            text=statement,
            attributes={"speaker": speaker, "audience": audience, "about": about, "claim_type": claim.type.value},
            visible_to=event.visible_to,
        )
        self._graph.add_node(node)
        self._graph.add_edge(Edge(source=event.id, target=node_id, type=NarrativeEdge.STATED))

        created: list[str] = []
        for name in claim.about:
            found = self._shared.ensure(name)
            if found is None:
                continue
            entity, is_new = found
            if is_new:
                created.append(entity)
            if not self._graph.has_edge(node_id, entity, MagmaEdge.MENTIONS):
                self._graph.add_edge(Edge(source=node_id, target=entity, type=MagmaEdge.MENTIONS))

        vector: Vector = self._embedder.embed_documents([statement])[0]
        with self._shared.lock:
            self._link_similar_claims(node_id, vector)
            self._vectors.add(node_id, vector)
            self._keywords.add(node_id, statement)
        self._claim_ids.append(node_id)
        return created

    def _link_similar_claims(self, node_id: str, vector: Vector) -> None:
        """The rule of Alg. 2 among claims: a proposition said again is another claim, similar to the first."""
        if not self._claim_ids:
            return
        closest = self._vectors.search(vector, self._config.semantic_edge_max_neighbors, frozenset(self._claim_ids))
        for hit in closest:
            if hit.score > self._config.semantic_edge_threshold:
                self._graph.add_edge(Edge(source=node_id, target=hit.node_id, type=MagmaEdge.SIMILAR_TO, weight=hit.score))


def _part(event: Node, name: str) -> str:
    text = event.attributes.get(name)
    return text if isinstance(text, str) and text else NO_PART

