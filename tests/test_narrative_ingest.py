import json
import queue
import re
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np

from magma.bm25_index import BM25Index
from magma.config import DEFAULT
from magma.embed import Matrix, Vector, normalize
from magma.faiss_index import FaissVectorIndex
from magma.graph import VISIBLE_TO_EVERYONE, Node
from magma.ingest import CAUSAL_SYSTEM_PROMPT, Turn
from magma.llm import ChatRequest, ClientSettings, Completion, LLMClient
from magma.networkx_graph import NetworkXGraphStore
from magma.temporal import DateparserTagger
from narrative.ingest import (
    NARRATIVE_EXTRACTION_SYSTEM_PROMPT,
    NarrativeConsolidation,
    NarrativeExtraction,
    NarrativeFastPath,
    SharedWrites,
)
from narrative.schema import NARRATIVE_PROFILE, ClaimType, MentionRole

START = datetime(2024, 3, 1, 20, 0)
CORRIDOR = frozenset({"Dara"})
HALL = frozenset({"Lia", "Brom", "Ceres"})
_SPEECH = re.compile(r"- Speech: (.*)")
_ACTION = re.compile(r"- Action: (.*)")


class ScriptedBackend:
    """Answers the extended extraction prompt by the speech (or the action, when there is none); no causes."""

    def __init__(self, extractions: dict[str, object]) -> None:
        self.extractions = extractions
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        system, user = request["messages"][0]["content"], request["messages"][1]["content"]
        reply: object = {"causes": []}
        if system == NARRATIVE_EXTRACTION_SYSTEM_PROMPT:
            speech = _SPEECH.search(user)
            action = _ACTION.search(user)
            assert speech is not None and action is not None
            key = speech.group(1) if speech.group(1) != "none" else action.group(1)
            reply = self.extractions.get(key, {})
        else:
            assert system == CAUSAL_SYSTEM_PROMPT
        return Completion(text=json.dumps(reply), prompt_tokens=100, completion_tokens=20, reasoning_tokens=0)

    def extraction_prompts(self) -> list[str]:
        return [r["messages"][1]["content"] for r in self.requests if r["messages"][0]["content"] == NARRATIVE_EXTRACTION_SYSTEM_PROMPT]


class TableEmbedder:
    """Each text gets the vector of the first table key it contains; otherwise a vector of its own axis."""

    def __init__(self, table: dict[str, tuple[float, float, float]]) -> None:
        self._table = table

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        rows = [self._row(text) for text in texts]
        return normalize(np.array(rows, dtype=np.float32))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector

    def _row(self, text: str) -> tuple[float, float, float]:
        for key, row in self._table.items():
            if key in text:
                return row
        return (0.0, 0.0, 1.0)


class Paths:
    def __init__(self, extractions: dict[str, object], table: dict[str, tuple[float, float, float]] | None = None) -> None:
        self.graph = NetworkXGraphStore(NARRATIVE_PROFILE)
        self.vectors = FaissVectorIndex(3)
        self.keywords = BM25Index(DEFAULT.lexical_index)
        self.backend = ScriptedBackend(extractions)
        embedder = TableEmbedder(table or {})
        waiting: queue.Queue[str] = queue.Queue()
        shared = SharedWrites(self.graph)
        llm = LLMClient(ClientSettings(model="test-model"), backend=self.backend, cache_dir=None)
        self.fast = NarrativeFastPath(self.graph, self.vectors, self.keywords, embedder, waiting, shared)
        self.slow = NarrativeConsolidation(
            self.graph, llm, DateparserTagger(), waiting, self.vectors, self.keywords, embedder, shared
        )

    def ingest(self, speaker: str, text: str, minute: int, present: frozenset[str] = VISIBLE_TO_EVERYONE) -> Node:
        return self.fast.ingest(Turn(speaker, text, START + timedelta(minutes=minute), present))

    def mentions_of(self, source: str) -> dict[str, object]:
        return {
            neighbor.node.id: neighbor.edge.attributes.get("role")
            for neighbor in self.graph.neighbors(source)
            if neighbor.is_outgoing and neighbor.edge.type == "MENTIONS"
        }


BROM_SPEECH = "Nobody attacked me, I dropped it."
BROM_EXTRACTION: dict[str, object] = {
    "entities": ["Brom", "wand"],
    "topic": "the wand",
    "summary": "Brom says he dropped the wand.",
    "roles": [{"entity": "Brom", "role": "actor"}, {"entity": "wand", "role": "weapon"}],
    "claims": [
        {"statement": "Someone attacked Brom", "about": ["Brom"], "type": "denial"},
        {"statement": "Brom dropped the wand", "about": ["Brom", "wand"], "type": "ASSERTION"},
    ],
}


def test_fast_path_takes_visibility_parts_and_actor_from_the_scene() -> None:
    paths = Paths({})
    event = paths.ingest("Brom", f'*picks up the wand* "{BROM_SPEECH}"', 0, CORRIDOR)
    assert event.visible_to == frozenset({"Brom", "Dara"})
    assert event.text == f'Brom: *picks up the wand* "{BROM_SPEECH}"'  # the vector is still the whole turn's
    assert (event.attributes["action_text"], event.attributes["speech_text"]) == ("picks up the wand", BROM_SPEECH)
    assert paths.mentions_of(event.id) == {"entity:brom": "ACTOR"}
    assert paths.graph.node("entity:brom").attributes == {"kind": "agent"}


def test_narration_has_no_actor() -> None:
    paths = Paths({})
    event = paths.ingest("narrator", "Lia casts a disarming spell on Brom in the hall.", 0, HALL)
    assert event.visible_to == HALL
    assert paths.mentions_of(event.id) == {}


def test_slow_path_adds_claims_as_facets_of_the_event() -> None:
    paths = Paths({BROM_SPEECH: BROM_EXTRACTION})
    event = paths.ingest("Brom", f'*picks up the wand* "{BROM_SPEECH}"', 0, CORRIDOR)
    paths.slow.drain()

    prompt = paths.backend.extraction_prompts()[0]
    assert "- Audience: {Dara}" in prompt
    assert "- Action: picks up the wand" in prompt
    assert f"- Speech: {BROM_SPEECH}" in prompt

    stated = [n.node.id for n in paths.graph.neighbors(event.id) if n.is_outgoing and n.edge.type == "STATED"]
    assert stated == ["e1.c1", "e1.c2"]
    denial = paths.graph.node("e1.c1")
    assert denial.text == "Someone attacked Brom"
    assert denial.visible_to == event.visible_to  # a facet sees what its event sees
    assert denial.timestamp is None
    assert denial.attributes == {"speaker": "Brom", "audience": ["Brom", "Dara"], "about": ["Brom"], "claim_type": "DENIAL"}
    assert paths.mentions_of("e1.c2") == {"entity:brom": None, "entity:wand": None}
    # roles of the event's mentions: the actor from the fast path, an unknown role read as OTHER
    assert paths.mentions_of(event.id) == {"entity:brom": "ACTOR", "entity:wand": "OTHER"}
    assert paths.vectors.contains("e1.c1") and paths.keywords.contains("e1.c2")


def test_an_action_alone_states_no_claim() -> None:
    narration = "Lia casts a disarming spell on Brom in the hall."
    extraction: dict[str, object] = {"entities": ["Lia"], "claims": [{"statement": "Lia attacked Brom"}]}
    paths = Paths({narration: extraction})
    paths.ingest("narrator", narration, 0, HALL)
    paths.slow.drain()
    assert paths.graph.nodes("CLAIM") == []


def test_consolidating_an_event_again_adds_nothing_twice() -> None:
    paths = Paths({BROM_SPEECH: BROM_EXTRACTION})
    event = paths.ingest("Brom", f'"{BROM_SPEECH}"', 0, CORRIDOR)
    paths.slow.drain()
    nodes, edges = len(paths.graph.nodes()), len(paths.graph.edges())
    paths.slow.consolidate(event.id)
    assert (len(paths.graph.nodes()), len(paths.graph.edges())) == (nodes, edges)


def test_similar_edges_link_events_to_events_and_claims_to_claims() -> None:
    extraction: dict[str, object] = {"claims": [{"statement": "The key is in the drawer"}]}
    table = {"key": (1.0, 0.0, 0.0)}  # every text mentioning the key gets the same vector
    paths = Paths({"The key is in the drawer.": extraction, "Yes, the key is in the drawer.": extraction}, table)
    paths.ingest("Lia", '"The key is in the drawer."', 0)
    paths.slow.drain()
    paths.ingest("Ceres", '"Yes, the key is in the drawer."', 1)
    paths.slow.drain()

    similar = {(edge.source, edge.target) for edge in paths.graph.edges() if edge.type == "SIMILAR_TO"}
    assert similar == {("e2", "e1"), ("e2.c1", "e1.c1")}  # never an event with a claim


def test_unknown_values_from_the_llm_fall_back_to_the_default() -> None:
    parsed = NarrativeExtraction.model_validate(
        {"roles": [{"entity": "x", "role": "weapon"}], "claims": [{"statement": "s", "type": "maybe"}]}
    )
    assert parsed.roles[0].role is MentionRole.OTHER
    assert parsed.claims[0].type is ClaimType.ASSERTION
