import queue
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from magma.bm25_index import BM25Index
from magma.config import DEFAULT, Layer
from magma.embed import Matrix, Vector, normalize
from magma.faiss_index import FaissVectorIndex
from magma.graph import GraphStore
from magma.ingest import FastPathIngestion, OutOfOrderTurn, Turn
from magma.networkx_graph import NetworkXGraphStore
from magma.schema import MAGMA_PROFILE, MagmaEdge

START = datetime(2023, 5, 8, 13, 56)


class FakeEmbedder:
    """Vectors chosen by the test: each text is looked up in a table, so the cosines are known in advance."""

    def __init__(self, table: dict[str, tuple[float, float]]) -> None:
        self._table = table

    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return normalize(np.array([self._table[text] for text in texts], dtype=np.float32))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector


class Memory:
    def __init__(self, table: dict[str, tuple[float, float]], max_neighbors: int = 5, graph: GraphStore | None = None) -> None:
        self.graph = graph or NetworkXGraphStore(MAGMA_PROFILE)
        self.vectors = FaissVectorIndex(dimension=2)
        self.keywords = BM25Index()
        self.queue: queue.Queue[str] = queue.Queue()
        config = replace(DEFAULT.ingest, semantic_edge_threshold=0.6, semantic_edge_max_neighbors=max_neighbors)
        self.fast_path = FastPathIngestion(self.graph, self.vectors, self.keywords, FakeEmbedder(table), self.queue, config)

    def say(self, speaker: str, text: str, minute: int) -> str:
        return self.fast_path.ingest(Turn(speaker=speaker, text=text, timestamp=START + timedelta(minutes=minute))).id

    def queued(self) -> list[str]:
        ids: list[str] = []
        while not self.queue.empty():
            ids.append(self.queue.get_nowait())
        return ids


# e1 along x, e2 along y, e3 at 30 degrees from x: cos(e3, e1) = cos 30° ≈ 0.866 and cos(e3, e2) = cos 60° = 0.5.
TABLE = {
    "Melanie: I played the violin": (1.0, 0.0),
    "Caroline: The weather in Paris": (0.0, 1.0),
    "Melanie: Violin lessons again": (0.866, 0.5),
}


def three_turns() -> Memory:
    memory = Memory(TABLE)
    memory.say("Melanie", "I played the violin", 0)
    memory.say("Caroline", "The weather in Paris", 1)
    memory.say("Melanie", "Violin lessons again", 2)
    return memory


def test_each_turn_becomes_an_event_with_the_speaker() -> None:
    memory = three_turns()
    events = memory.graph.nodes()
    assert [event.id for event in events] == ["e1", "e2", "e3"]
    assert events[0].text == "Melanie: I played the violin"
    assert events[0].attributes == {"speaker": "Melanie"}
    assert events[2].timestamp == START + timedelta(minutes=2)


def test_events_form_the_temporal_backbone() -> None:
    memory = three_turns()
    temporal = [(edge.source, edge.target) for edge in memory.graph.edges() if edge.type == MagmaEdge.NEXT]
    assert temporal == [("e1", "e2"), ("e2", "e3")]


def test_similar_events_are_linked_above_the_threshold() -> None:
    memory = three_turns()
    similar = memory.graph.neighbors("e3", Layer.SEMANTIC)
    assert [n.node.id for n in similar] == ["e1"]  # cos(e3, e2) = 0.5 is below θ_sim = 0.6
    assert similar[0].edge.weight == pytest.approx(0.866, abs=1e-3)


def test_at_most_max_neighbors_similar_edges_per_new_event() -> None:
    same = (1.0, 0.0)
    memory = Memory({f"A: turn {i}": same for i in range(4)}, max_neighbors=2)
    for minute in range(4):
        memory.say("A", f"turn {minute}", minute)
    new_edges_of_e4 = [n.node.id for n in memory.graph.neighbors("e4", Layer.SEMANTIC) if n.is_outgoing]
    assert new_edges_of_e4 == ["e1", "e2"]  # equal scores keep insertion order


def test_events_are_indexed_and_queued_for_the_slow_path() -> None:
    memory = three_turns()
    assert memory.vectors.ids() == ["e1", "e2", "e3"]
    assert [hit.node_id for hit in memory.keywords.search("violin", top_k=5)] == ["e1", "e3"]
    assert memory.queued() == ["e1", "e2", "e3"]


def test_turn_older_than_the_last_event_is_rejected_before_any_change() -> None:
    memory = three_turns()
    with pytest.raises(OutOfOrderTurn):
        memory.say("Melanie", "I played the violin", -5)
    assert len(memory.graph.nodes()) == 3
    assert memory.queued() == ["e1", "e2", "e3"]


def test_turns_at_the_same_time_are_accepted() -> None:
    memory = Memory({"A: one": (1.0, 0.0), "B: two": (0.0, 1.0)})
    memory.say("A", "one", 0)
    memory.say("B", "two", 0)  # LoCoMo gives one timestamp per session
    assert [event.id for event in memory.graph.nodes()] == ["e1", "e2"]


def test_ingestion_continues_an_existing_graph() -> None:
    first = three_turns()
    table = {**TABLE, "Caroline: Back from Paris": (0.0, 1.0)}
    resumed = Memory(table, graph=first.graph)
    new_id = resumed.say("Caroline", "Back from Paris", 3)
    assert new_id == "e4"
    assert ("e3", "e4") in [(edge.source, edge.target) for edge in first.graph.edges() if edge.type == MagmaEdge.NEXT]
