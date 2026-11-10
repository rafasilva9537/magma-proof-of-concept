"""Unit tests for the retrieval measurement script (`bench/measure_retrieval.py`)."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from bench.benchmark import Question
from bench.locomo import find_sample, load_samples
from bench.measure_retrieval import event_ids_by_dia_id, load_memory, measure_question, summarize
from magma.bm25_index import BM25Index
from magma.config import Intent
from magma.embed import Matrix, Vector, normalize
from magma.faiss_index import FaissVectorIndex
from magma.graph import Edge, Node, save_graph
from magma.index import save_vectors
from magma.json_types import JSONValue
from magma.networkx_graph import NetworkXGraphStore
from magma.retrieve import NodeVectors, Retriever
from magma.router import RouterOutput
from magma.schema import MAGMA_PROFILE, MagmaNode
from magma.temporal import TimeWindow

QUERY = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def at_cosine(value: float) -> Vector:
    """A unit vector whose cosine with QUERY is `value`."""
    unit: Vector = normalize(np.array([[value, math.sqrt(1 - value * value), 0.0]], dtype=np.float32))[0]
    return unit


class TableEmbedder:
    def __init__(self, table: dict[str, Vector]) -> None:
        self._table = table

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return np.asarray([self._table[text] for text in texts], dtype=np.float32).reshape(len(texts), 3)

    def embed_query(self, text: str) -> Vector:
        return self._table[text]


class WordCounter:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


class ToyMemory:
    def __init__(self) -> None:
        self.graph = NetworkXGraphStore(MAGMA_PROFILE)
        self.vectors = FaissVectorIndex(3)
        self.keywords = BM25Index()

    def event(
        self,
        node_id: str,
        text: str,
        cosine_to_query: float,
        when: datetime,
        dates: Sequence[tuple[datetime, datetime]] = (),
    ) -> None:
        mentioned: list[JSONValue] = [
            {"text": "a date", "start": start.isoformat(), "end": end.isoformat(), "granularity": "YEAR"}
            for start, end in dates
        ]
        self.graph.add_node(
            Node(id=node_id, type=MagmaNode.EVENT, text=text, timestamp=when, attributes={"dates_mentioned": mentioned})
        )
        self.vectors.add(node_id, at_cosine(cosine_to_query))
        self.keywords.add(node_id, text)

    def entity(self, node_id: str, name: str) -> None:
        self.graph.add_node(Node(id=node_id, type=MagmaNode.ENTITY, text=name))

    def link(self, source: str, target: str, edge_type: str) -> None:
        self.graph.add_edge(Edge(source=source, target=target, type=edge_type))

SAMPLE = {
    "sample_id": "conv-x",
    "qa": [],
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1_date_time": "1:00 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hi!"},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "Hello."},
        ],
        "session_2_date_time": "2:00 pm on 9 May, 2023",
        "session_2": [{"speaker": "Caroline", "dia_id": "D2:1", "text": "Bye."}],
    },
}


def test_event_ids_follow_the_order_of_the_turns(tmp_path: Path) -> None:
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([SAMPLE]), encoding="utf-8")

    ids = event_ids_by_dia_id(find_sample(load_samples(path), "conv-x"))

    assert ids == {"D1:1": "e1", "D1:2": "e2", "D2:1": "e3"}


def camping_memory() -> ToyMemory:
    memory = ToyMemory()
    year_2022 = (datetime(2022, 1, 1), datetime(2023, 1, 1))
    memory.event("e1", "Melanie: we went camping last year", 0.8, when=datetime(2023, 8, 14), dates=[year_2022])
    memory.event("e2", "Caroline: nice", 0.1, when=datetime(2023, 8, 14))
    memory.entity("ent:melanie", "Melanie")
    memory.link("e1", "ent:melanie", "MENTIONS")
    memory.link("e1", "e2", "NEXT")
    return memory


def test_a_question_reports_where_its_evidence_went() -> None:
    memory = camping_memory()
    embedder = TableEmbedder({"Melanie": at_cosine(0.05)})
    question = Question(
        id="q1",
        conversation_id="conv-x",
        category="TEMPORAL",
        question="Where did Melanie camp in 2022?",
        gold="the beach",
        evidence=("D1:1", "D9:9"),
    )
    route = RouterOutput(
        question=question.question,
        intent=Intent.ENTITY,
        intent_reason="test",
        time_window=TimeWindow(start=datetime(2022, 1, 1), end=datetime(2023, 1, 1)),
        temporal_mentions=(),
        query_vector=QUERY,
        keywords=("melanie", "camp", "2022"),
    )
    retriever = Retriever.create(memory.graph, memory.vectors, memory.keywords, embedder, WordCounter())
    node_vectors = NodeVectors(memory.graph, memory.vectors, embedder)

    measure = measure_question(question, {"D1:1": "e1", "D1:2": "e2"}, route, retriever, memory.graph, node_vectors)
    report = summarize("conv-x", [measure], memory.graph)

    assert measure.evidence == ("e1",)
    assert measure.missing_evidence == ("D9:9",)
    assert (measure.window_events_tau, measure.window_events_dates) == (0, 1)  # only the mentioned date is in 2022
    assert (measure.evidence_in_window_tau, measure.evidence_in_window_dates) == (0, 1)
    assert measure.evidence_in_anchors == 1
    assert measure.dropped_entities == 1  # the name is less similar to the question than 0.15
    assert measure.entity_cosines == pytest.approx((0.05,))
    assert report.time_channel.with_tau_only == 0
    assert report.time_channel.with_tau_or_dates == 1
    assert report.overall.anchor_recall == 1.0
    assert report.drops.entity_share_below == 1.0
    assert report.drops.event_share_below == 0.5  # e2 has cosine 0.1


def test_a_saved_memory_loads_with_its_keyword_index(tmp_path: Path) -> None:
    memory = camping_memory()
    memory.event("e3", "Melanie: see you", 0.2, when=datetime(2023, 8, 14))  # BM25 needs a term in under half the texts
    save_graph(memory.graph, tmp_path / "graph.json")
    save_vectors(memory.vectors, tmp_path / "vectors.npz")

    loaded = load_memory(tmp_path, dimension=3)

    assert [node.id for node in loaded.graph.nodes()] == ["e1", "e2", "ent:melanie", "e3"]
    assert [hit.node_id for hit in loaded.keywords.search("camping", top_k=5)] == ["e1"]
    assert loaded.vectors.ids() == ["e1", "e2", "e3"]
