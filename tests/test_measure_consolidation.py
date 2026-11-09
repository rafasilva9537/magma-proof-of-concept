import json
import queue
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bench.locomo import LocomoSample
from bench.measure_consolidation import distribution, ingest_conversation, read_calls, summarize
from magma.bm25_index import BM25Index
from magma.config import DEFAULT
from magma.embed import Matrix, Vector, normalize
from magma.faiss_index import FaissVectorIndex
from magma.ingest import CAUSAL_SYSTEM_PROMPT, FastPathIngestion, SlowPathConsolidation
from magma.llm import ChatRequest, ClientSettings, Completion, LLMClient
from magma.networkx_graph import NetworkXGraphStore
from magma.schema import MAGMA_PROFILE
from magma.temporal import DateparserTagger

# Three turns in two sessions; orthogonal vectors, so the fast path adds no SIMILAR_TO edge.
SAMPLE = LocomoSample(
    sample_id="conv-x",
    qa=[],
    conversation={
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Melanie", "dia_id": "D1:1", "text": "I adopted a dog, Oscar."},
            {"speaker": "Caroline", "dia_id": "D1:2", "text": "Oscar is so cute!"},
        ],
        "session_2_date_time": "2:00 pm on 20 May, 2023",
        "session_2": [{"speaker": "Melanie", "dia_id": "D2:1", "text": "Took oscar to the vet yesterday."}],
    },
)
VECTORS = {
    "Melanie: I adopted a dog, Oscar.": (1.0, 0.0, 0.0),
    "Caroline: Oscar is so cute!": (0.0, 1.0, 0.0),
    "Melanie: Took oscar to the vet yesterday.": (0.0, 0.0, 1.0),
}


class TableEmbedder:
    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return normalize(np.array([VECTORS[text] for text in texts], dtype=np.float32))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector


class CannedBackend:
    """Every turn mentions Oscar; e3 was caused by e1 (rank 2 in its candidates) and e2 by e1 (rank 1)."""

    def complete(self, request: ChatRequest) -> Completion:
        system, user = request["messages"][0]["content"], request["messages"][1]["content"]
        reply: dict[str, object]
        if system != CAUSAL_SYSTEM_PROMPT:
            dates = ["yesterday"] if "yesterday" in user else []
            reply = {"entities": ["Oscar" if "Oscar" in user else "oscar"], "dates_mentioned": dates}
        elif user.startswith("New event:\n[e3]"):
            reply = {"causes": [{"event_id": "e1", "confidence": 0.85, "reason": "the dog adopted earlier"}]}
        else:
            reply = {"causes": [{"event_id": "e1", "confidence": 0.4, "reason": "below δ"}]}
        return Completion(text=json.dumps(reply), prompt_tokens=len(user), completion_tokens=10, reasoning_tokens=0)


def test_distribution_of_values() -> None:
    summary = distribution([1.0, 2.0, 3.0, 10.0])
    assert (summary.count, summary.mean, summary.median, summary.maximum) == (4, 4.0, 2.5, 10.0)
    assert distribution([]).count == 0


def test_report_of_a_small_conversation(tmp_path: Path) -> None:
    log_path = tmp_path / "llm_calls.jsonl"
    llm = LLMClient(ClientSettings(model="test-model"), backend=CannedBackend(), cache_dir=None, log_path=log_path)
    graph = NetworkXGraphStore(MAGMA_PROFILE)
    pending: queue.Queue[str] = queue.Queue()
    fast_path = FastPathIngestion(graph, FaissVectorIndex(3), BM25Index(), TableEmbedder(), pending)
    slow_path = SlowPathConsolidation(graph, llm, DateparserTagger(), pending)

    results = ingest_conversation(SAMPLE, fast_path, slow_path)
    report = summarize("conv-x", "test-model", DEFAULT.ingest, graph, results, read_calls(log_path), 1.0, 5, 7)

    assert report.turns == 3
    assert (report.entities, report.mentions_edges) == (1, 3)  # "Oscar" and "oscar" are one node
    assert [(m.key, m.forms) for m in report.merge_examples] == [("oscar", ["Oscar", "oscar"])]
    assert (report.date_expressions, report.resolved_dates) == (1, 1)
    # e1: nothing earlier; e2: e1 at 1 hop; e3: e2 at 1 hop, e1 at 2 hops (NEXT and through Oscar).
    assert [result.neighborhood_events for result in results] == [0, 1, 2]
    assert (report.causal_edges, report.events_with_cause) == (1, 1)
    assert report.cause_rank.median == 2.0
    assert [(s.rank, s.share) for s in report.cause_rank_within] == [(5, 1.0), (10, 1.0), (20, 1.0), (50, 1.0)]
    assert report.cause_turn_gap.median == 2.0
    assert [(c.name, c.count) for c in report.confidence] == [(">= 0.8", 1)]
    assert [(u.tag, u.calls) for u in report.usage] == [("causal", 2), ("extraction", 3)]
    assert [example.reason for example in report.examples] == ["the dog adopted earlier"]


def test_pilot_stops_after_the_first_turns() -> None:
    llm = LLMClient(ClientSettings(model="test-model"), backend=CannedBackend(), cache_dir=None)
    graph = NetworkXGraphStore(MAGMA_PROFILE)
    pending: queue.Queue[str] = queue.Queue()
    fast_path = FastPathIngestion(graph, FaissVectorIndex(3), BM25Index(), TableEmbedder(), pending)
    slow_path = SlowPathConsolidation(graph, llm, DateparserTagger(), pending)
    results = ingest_conversation(SAMPLE, fast_path, slow_path, max_turns=2)
    assert [result.event_id for result in results] == ["e1", "e2"]
