"""Memory evolution (§3.4): the fast path (Alg. 2) and the slow path (Alg. 3).

Fast path, with no LLM call: each turn becomes one event node. It is linked to
the previous event by NEXT (the temporal backbone), embedded and indexed
(vector and BM25), linked by SIMILAR_TO to the closest earlier events, and
queued for the slow path.

Slow path, with the LLM: for each queued event, the extraction prompt (C.1)
gives its entities, topic, facts, dates and summary. Entities become ENTITY
nodes linked by MENTIONS, dates are resolved against the event's timestamp, and
a second prompt finds which earlier events of the 2-hop neighborhood caused it
(CAUSES edges, kept above the confidence δ).
"""

from __future__ import annotations

import queue
import re
from dataclasses import dataclass, replace
from datetime import datetime

from pydantic import BaseModel, Field

from magma.config import DEFAULT, IngestConfig, Layer, TemporalConfig
from magma.embed import Embedder, Vector
from magma.graph import VISIBLE_TO_EVERYONE, Edge, GraphStore, Node
from magma.index import LexicalIndex, VectorIndex
from magma.json_types import JSONObject, JSONValue
from magma.llm import ContextWindowExceeded, InvalidStructuredOutput, LLMClient, Message
from magma.schema import MagmaEdge, MagmaNode
from magma.temporal import TemporalMention, TemporalTagger


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str
    timestamp: datetime
    # Who is in the scene when the turn happens. The paper has no observers, so the magma profile ignores it;
    # the narrative profile reads it as the turn's visibility.
    present: frozenset[str] = VISIBLE_TO_EVERYONE


class OutOfOrderTurn(ValueError):
    """A turn older than the last event: the temporal backbone only grows forward."""


class FastPathIngestion:
    def __init__(
        self,
        graph: GraphStore,
        vectors: VectorIndex,
        keywords: LexicalIndex,
        embedder: Embedder,
        consolidation_queue: queue.Queue[str],
        config: IngestConfig = DEFAULT.ingest,
    ) -> None:
        self._graph = graph
        self._vectors = vectors
        self._keywords = keywords
        self._embedder = embedder
        self._queue = consolidation_queue
        self._config = config
        events = graph.nodes(MagmaNode.EVENT)
        self._event_count = len(events)
        self._last_event = events[-1] if events else None

    def ingest(self, turn: Turn) -> Node:
        """Alg. 2: segment, extend the temporal backbone, index, link similar events, enqueue."""
        event = self._segment_event(turn)
        previous = self._last_event
        self._graph.add_node(event)
        if previous is not None:
            self._graph.add_edge(Edge(source=previous.id, target=event.id, type=MagmaEdge.NEXT))

        vector = self._embed(event)
        self._link_similar_events(event, vector)
        self._vectors.add(event.id, vector)
        self._keywords.add(event.id, event.text)

        self._queue.put(event.id)
        self._event_count += 1
        self._last_event = event
        return event

    def _segment_event(self, turn: Turn) -> Node:
        """One turn is one event. The speaker goes into the content, so retrieval and the QA prompt see who spoke."""
        if self._last_event is not None and self._last_event.timestamp is not None:
            if turn.timestamp < self._last_event.timestamp:
                raise OutOfOrderTurn(f"turn at {turn.timestamp} comes after an event at {self._last_event.timestamp}")
        return Node(
            id=f"e{self._event_count + 1}",
            type=MagmaNode.EVENT,
            text=f"{turn.speaker}: {turn.text}",
            timestamp=turn.timestamp,
            attributes={"speaker": turn.speaker},
        )

    def _embed(self, event: Node) -> Vector:
        vector: Vector = self._embedder.embed_documents([event.text])[0]
        return vector

    def _link_similar_events(self, event: Node, vector: Vector) -> None:
        """SIMILAR_TO to the closest earlier events with cos(v_i, v_j) > θ_sim, at most `max_neighbors` of them.

        Runs before the event's own vector is indexed, so the event never finds itself.
        """
        closest = self._vectors.search(vector, top_k=self._config.semantic_edge_max_neighbors)
        for hit in closest:
            if hit.score > self._config.semantic_edge_threshold:
                self._graph.add_edge(Edge(source=event.id, target=hit.node_id, type=MagmaEdge.SIMILAR_TO, weight=hit.score))


# Prompt C.1 of the paper (Appendix C), split into the instructions and the input data.
EXTRACTION_SYSTEM_PROMPT = """System Role: You are an automated Graph Memory Parser. Your task is to extract structured metadata from raw conversational logs to build a knowledge graph.

Instructions: Analyze the input and return ONLY a valid JSON object matching the specific schema below. Do not include markdown formatting.

Target Schema:
- "entities": List of proper nouns (People, Locations, Organizations).
- "topic": String (1-3 words representing the main theme).
- "relationships": List of strings describing interactions (e.g., "X researches Y").
- "semantic_facts": List of atomic facts preserving key information.
- "dates_mentioned": List of temporal strings (e.g., "next Friday", "2024-01-01").
- "summary": One-sentence summary preserving speaker attribution."""

EXTRACTION_INPUT = """Input Data:
- Speaker: {speaker}
- Text: {text}
- Context: {prev_summary}"""

NO_CONTEXT = "none"

# The paper gives no causal prompt: Alg. 3 only says the LLM infers causal links from the neighborhood.
# This one follows the style of C.1 and asks for the confidence that δ is compared with.
CAUSAL_SYSTEM_PROMPT = """System Role: You are an automated Graph Memory Parser. Your task is to find causal links between the events of a conversation to build a knowledge graph.

Input: a new event and earlier events from its neighborhood in the memory graph, each with its id and time.

Instructions: List the earlier events that are a cause of the new event: an event that led to it, made it happen, or gave a speaker the reason or motive for it. Give each one a confidence between 0 and 1 that the causal link holds. Use only ids from the list of earlier events, and return an empty list when none of them is a cause. Return ONLY a valid JSON object matching the schema below. Do not include markdown formatting.

Target Schema:
- "causes": List of objects, each with "event_id" (string), "confidence" (number from 0 to 1) and "reason" (one short sentence)."""


class EventExtraction(BaseModel):
    """Output of prompt C.1."""

    entities: list[str] = Field(default_factory=list)
    topic: str = ""
    relationships: list[str] = Field(default_factory=list)
    semantic_facts: list[str] = Field(default_factory=list)
    dates_mentioned: list[str] = Field(default_factory=list)
    summary: str = ""


class FoundCause(BaseModel):
    event_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class CausalInference(BaseModel):
    """Output of the causal prompt."""

    causes: list[FoundCause] = Field(default_factory=list)


@dataclass(frozen=True)
class Consolidation:
    """What the slow path did for one event."""

    event_id: str
    entities: list[str]  # ids of the entity nodes the event mentions
    new_entities: list[str]  # the ones created for this event
    resolved_dates: list[TemporalMention]
    unresolved_dates: list[str]
    neighborhood_events: int  # earlier events within the hops, before any limit
    candidates: list[str]  # event ids shown to the causal prompt, in prompt order
    causes: list[Edge]
    errors: list[str]


_POSSESSIVE = re.compile(r"['’]s?$")


def entity_key(name: str) -> str:
    """Two mentions are the same entity when their keys match: lower case, single spaces, no possessive.

    "Caroline's" and "caroline" are one entity; nicknames ("Mel" for "Melanie") are not merged.
    """
    collapsed = " ".join(name.casefold().split())
    return _POSSESSIVE.sub("", collapsed).strip()


def entity_id(key: str) -> str:
    return f"entity:{key}"


class SlowPathConsolidation:
    """Alg. 3: consolidates the queued events, one at a time and in queue order."""

    def __init__(
        self,
        graph: GraphStore,
        llm: LLMClient,
        tagger: TemporalTagger,
        consolidation_queue: queue.Queue[str],
        config: IngestConfig = DEFAULT.ingest,
        temporal: TemporalConfig = DEFAULT.temporal,
    ) -> None:
        self._graph = graph
        self._llm = llm
        self._tagger = tagger
        self._queue = consolidation_queue
        self._config = config
        self._temporal = temporal
        self._interrupted: str | None = None  # taken from the queue, but its consolidation raised

    def drain(self) -> list[Consolidation]:
        """Consolidate every queued event and return when the queue is empty.

        If the LLM stays unavailable, the error goes up and the event being consolidated
        is the first one of the next drain, so no event is skipped and the order is kept.
        Consolidating an event again is safe: existing entities and edges are not added twice.
        """
        done: list[Consolidation] = []
        while True:
            if self._interrupted is not None:
                event_id = self._interrupted
            else:
                try:
                    event_id = self._queue.get_nowait()
                except queue.Empty:
                    return done
            self._interrupted = event_id
            done.append(self.consolidate(event_id))
            self._interrupted = None
            self._queue.task_done()

    def consolidate(self, event_id: str) -> Consolidation:
        """Extraction (C.1), entities, dates, then the causal links from the 2-hop neighborhood."""
        event = self._graph.node(event_id)
        errors: list[str] = []
        attributes: JSONObject = dict(event.attributes)

        extraction = self._extract(event, errors)
        entities: list[str] = []
        new_entities: list[str] = []
        resolved: list[TemporalMention] = []
        unresolved: list[str] = []
        if extraction is not None:
            entities, new_entities = self._link_extraction(event, extraction)
            resolved, unresolved = self._resolve_dates(event, extraction.dates_mentioned)
            attributes.update(_extraction_attributes(extraction, resolved, unresolved))

        neighborhood = self._earlier_neighborhood(event)
        limit = self._config.causal_candidate_limit
        candidates = neighborhood if limit is None else neighborhood[:limit]
        causes = self._infer_causes(event, candidates, errors) if candidates else []

        if errors:
            attributes["consolidation_errors"] = _json_strings(errors)
        self._graph.replace_node(replace(event, attributes=attributes))
        return Consolidation(
            event_id=event.id,
            entities=entities,
            new_entities=new_entities,
            resolved_dates=resolved,
            unresolved_dates=unresolved,
            neighborhood_events=len(neighborhood),
            candidates=[candidate.id for candidate in candidates],
            causes=causes,
            errors=errors,
        )

    def _extract(self, event: Node, errors: list[str]) -> EventExtraction | None:
        speaker = str(event.attributes.get("speaker", ""))
        text = event.text.removeprefix(f"{speaker}: ")
        user_input = EXTRACTION_INPUT.format(speaker=speaker, text=text, prev_summary=self._previous_summary(event))
        messages: list[Message] = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
        try:
            return self._llm.chat_structured(messages, EventExtraction, tag="extraction").value
        except (InvalidStructuredOutput, ContextWindowExceeded) as error:
            errors.append(f"extraction: {error}")
            return None

    def _previous_summary(self, event: Node) -> str:
        """The `Context` of C.1: the summary of the event right before, which the queue order consolidated first."""
        for neighbor in self._graph.neighbors(event.id, Layer.TEMPORAL):
            if neighbor.is_outgoing or neighbor.edge.type != MagmaEdge.NEXT:
                continue
            summary = neighbor.node.attributes.get("summary")
            if isinstance(summary, str) and summary:
                return summary
        return NO_CONTEXT

    def _link_extraction(self, event: Node, extraction: EventExtraction) -> tuple[list[str], list[str]]:
        """What the extraction adds to the graph; here, the entities and their MENTIONS edges.

        Returns the ids of the entities the event mentions and of the ones created for it. A
        profile that extracts more (such as claims) extends this step and `_extract`.
        """
        return self._link_entities(event, extraction.entities)

    def _link_entities(self, event: Node, names: list[str]) -> tuple[list[str], list[str]]:
        """One ENTITY node per entity key, shared by every event that mentions it, and a MENTIONS edge to it."""
        surface_of_key: dict[str, str] = {}
        for name in names:
            key = entity_key(name)
            if key and key not in surface_of_key:
                surface_of_key[key] = name.strip()
        mentioned: list[str] = []
        created: list[str] = []
        for key, surface in surface_of_key.items():
            node_id = entity_id(key)
            if not self._graph.has_node(node_id):
                self._graph.add_node(Node(id=node_id, type=MagmaNode.ENTITY, text=surface))
                created.append(node_id)
            if not self._graph.has_edge(event.id, node_id, MagmaEdge.MENTIONS):
                self._graph.add_edge(Edge(source=event.id, target=node_id, type=MagmaEdge.MENTIONS))
            mentioned.append(node_id)
        return mentioned, created

    def _resolve_dates(self, event: Node, expressions: list[str]) -> tuple[list[TemporalMention], list[str]]:
        """Each expression becomes the window of its own size, read from the event's timestamp ("yesterday" is the day before)."""
        assert event.timestamp is not None  # every event is an anchor
        resolved: list[TemporalMention] = []
        unresolved: list[str] = []
        for expression in expressions:
            mention = self._tagger.resolve(expression, event.timestamp, self._temporal.mention_preference)
            if mention is None:
                unresolved.append(expression)
            else:
                resolved.append(mention)
        return resolved, unresolved

    def _earlier_neighborhood(self, event: Node) -> list[Node]:
        """N(n_t) of Alg. 3: earlier events within `neighborhood_hops`, closest hop first, then the most recent.

        Only earlier events can be causes, which also keeps the causal graph acyclic. Later
        events are not walked through either, so the neighborhood does not depend on how far
        the fast path has run ahead of the slow path.
        """
        position = {node.id: index for index, node in enumerate(self._graph.nodes(MagmaNode.EVENT))}
        own_position = position[event.id]
        hop_of = {event.id: 0}
        found: list[tuple[int, Node]] = []
        frontier = [event.id]
        for hop in range(1, self._config.neighborhood_hops + 1):
            next_frontier: list[str] = []
            for node_id in frontier:
                for neighbor in self._graph.neighbors(node_id):
                    node = neighbor.node
                    is_later_event = position.get(node.id, -1) > own_position
                    if node.id in hop_of or is_later_event:
                        continue
                    hop_of[node.id] = hop
                    next_frontier.append(node.id)
                    if node.type == MagmaNode.EVENT:
                        found.append((hop, node))
            frontier = next_frontier
        found.sort(key=lambda item: (item[0], -position[item[1].id]))
        return [node for _, node in found]

    def _infer_causes(self, event: Node, candidates: list[Node], errors: list[str]) -> list[Edge]:
        """CAUSES from each earlier event the LLM names, when its confidence is above δ."""
        listed = "\n".join(_event_line(candidate) for candidate in candidates)
        messages: list[Message] = [
            {"role": "system", "content": CAUSAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"New event:\n{_event_line(event)}\n\nEarlier events:\n{listed}"},
        ]
        try:
            inference = self._llm.chat_structured(messages, CausalInference, tag="causal").value
        except (InvalidStructuredOutput, ContextWindowExceeded) as error:
            errors.append(f"causal: {error}")
            return []
        allowed = {candidate.id for candidate in candidates}
        causes: list[Edge] = []
        for found in inference.causes:
            if found.event_id not in allowed or found.confidence <= self._config.causal_min_confidence:
                continue
            if self._graph.has_edge(found.event_id, event.id, MagmaEdge.CAUSES):
                continue  # the same cause listed twice
            edge = Edge(
                source=found.event_id,
                target=event.id,
                type=MagmaEdge.CAUSES,
                weight=found.confidence,
                attributes={"reason": found.reason},
            )
            self._graph.add_edge(edge)
            causes.append(edge)
        return causes


def _event_line(event: Node) -> str:
    assert event.timestamp is not None
    return f"[{event.id}] {event.timestamp:%Y-%m-%d %H:%M} | {event.text}"


def _extraction_attributes(extraction: EventExtraction, resolved: list[TemporalMention], unresolved: list[str]) -> JSONObject:
    """A_i of the event: what C.1 returned, with the dates as resolved windows."""
    dates: list[JSONValue] = []
    for mention in resolved:
        dates.append(
            {
                "text": mention.text,
                "start": mention.window.start.isoformat(),
                "end": mention.window.end.isoformat(),
                "granularity": mention.granularity.value,
            }
        )
    return {
        "entities": _json_strings(extraction.entities),
        "topic": extraction.topic,
        "relationships": _json_strings(extraction.relationships),
        "semantic_facts": _json_strings(extraction.semantic_facts),
        "dates_mentioned": dates,
        "unresolved_dates": _json_strings(unresolved),
        "summary": extraction.summary,
    }


def _json_strings(values: list[str]) -> list[JSONValue]:
    strings: list[JSONValue] = []
    strings.extend(values)
    return strings

