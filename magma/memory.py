"""The memory-system boundary and the MAGMA facade that implements it.

`MemorySystem` is the only thing the benchmark runner, the metrics and the
consoles know about a memory: turns go in with `add_turn`, `consolidate`
finishes the pending background work, and `query` answers a question. Every
architecture and every baseline implements it, so they run on the same
benchmark code. The base class keeps the rules every implementation must
follow: turns arrive in time order, and each answer carries its latency and
the LLM usage it took.

`MagmaMemory` assembles the fast path (Alg. 2), the slow path (Alg. 3) running
on a background thread, the retrieval pipeline (Stages 1–4 of Alg. 1) and the
adaptive QA synthesizer (Appendix C.2) over one graph and its indexes.
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from magma.bm25_index import BM25Index
from magma.config import DEFAULT, Config
from magma.embed import Embedder
from magma.faiss_index import FaissVectorIndex
from magma.graph import GraphStore, save_graph
from magma.index import save_vectors
from magma.ingest import FastPathIngestion, OutOfOrderTurn, SlowPathConsolidation, Turn
from magma.json_types import JSONObject, JSONValue
from magma.llm import LLMClient, Usage
from magma.networkx_graph import NetworkXGraphStore
from magma.qa import AdaptiveQASynthesizer, AnswerSynthesizer, SynthesizedAnswer
from magma.retrieve import Anchor, Retrieval, Retriever, TokenCounter, Visit
from magma.router import IntentClassifier, QueryRouter, Router, RouterOutput, build_default_classifier
from magma.schema import MAGMA_PROFILE
from magma.temporal import TemporalTagger


@dataclass(frozen=True)
class Reply:
    """What an implementation returns for a question: the answer and the trace of how it was found."""

    answer: str
    trace: JSONObject


@dataclass(frozen=True)
class MemoryAnswer:
    """A reply with what it cost: the LLM usage during the query and the wall-clock latency."""

    answer: str
    trace: JSONObject
    usage: Usage
    latency_s: float


class MemorySystem(ABC):
    """A memory that ingests a conversation turn by turn and answers questions about it."""

    def __init__(self) -> None:
        self._last_turn_time: datetime | None = None

    def add_turn(self, turn: Turn) -> None:
        """Ingest one turn. Turns must come in time order: memory is built as the conversation happens."""
        if self._last_turn_time is not None and turn.timestamp < self._last_turn_time:
            raise OutOfOrderTurn(f"turn at {turn.timestamp} comes after a turn at {self._last_turn_time}")
        self._last_turn_time = turn.timestamp
        self._add_turn(turn)

    @abstractmethod
    def consolidate(self) -> None:
        """Finish all pending background work (the slow path), so a question sees every turn already given."""

    def query(self, question: str, reference_time: datetime, viewer: str | None = None) -> MemoryAnswer:
        """Answer `question` as of `reference_time` (the "now" used to read relative dates in it).

        `viewer` is the character asking, for memories that know who saw what; `None` is the
        omniscient reader. A memory without observers answers the same for every viewer.
        """
        usage_before = self.usage()
        start = time.perf_counter()
        reply = self._answer(question, reference_time, viewer)
        latency_s = time.perf_counter() - start
        return MemoryAnswer(answer=reply.answer, trace=reply.trace, usage=self.usage().since(usage_before), latency_s=latency_s)

    @abstractmethod
    def usage(self) -> Usage:
        """Total LLM usage of this memory so far (ingestion and queries)."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Write what the memory holds, for inspection and for the regression snapshot."""

    def close(self) -> None:
        """Release background resources (such as consolidation threads), if any."""

    @abstractmethod
    def _add_turn(self, turn: Turn) -> None: ...

    @abstractmethod
    def _answer(self, question: str, reference_time: datetime, viewer: str | None) -> Reply: ...


class _ConsolidationWorker:
    """Runs `SlowPathConsolidation.drain()` on a background thread and propagates errors on `drain()`."""

    def __init__(self, slow_path: SlowPathConsolidation) -> None:
        self._slow_path = slow_path
        self._condition = threading.Condition()
        self._pending_work = False
        self._is_working = False
        self._stopped = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="magma-slow-path", daemon=True)
        self._thread.start()

    def notify(self) -> None:
        """Wake the background worker after a new event is queued."""
        with self._condition:
            self._pending_work = True
            self._condition.notify_all()

    def drain(self) -> None:
        """Wait until the queue is empty and no consolidation step is running; re-raise any worker failure."""
        with self._condition:
            if self._error is None:
                self._pending_work = True
                self._condition.notify_all()
            while (self._pending_work or self._is_working) and self._error is None:
                self._condition.wait()
            if self._error is not None:
                error = self._error
                self._error = None
                raise error

    def close(self) -> None:
        """Stop the background worker thread cleanly."""
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                while (not self._pending_work or self._error is not None) and not self._stopped:
                    self._condition.wait()
                if self._stopped:
                    return
                self._pending_work = False
                self._is_working = True
            try:
                self._slow_path.drain()
            except BaseException as error:
                with self._condition:
                    self._error = error
                    self._pending_work = False
                    self._is_working = False
                    self._condition.notify_all()
                if not isinstance(error, Exception):
                    raise
            else:
                with self._condition:
                    self._is_working = False
                    self._condition.notify_all()


class MagmaMemory(MemorySystem):
    """The MAGMA memory: one typed multigraph with its vector and lexical indexes."""

    def __init__(
        self,
        graph: GraphStore,
        fast_path: FastPathIngestion,
        slow_path: SlowPathConsolidation,
        vectors: FaissVectorIndex,
        router: QueryRouter,
        retriever: Retriever,
        synthesizer: AnswerSynthesizer,
        llm: LLMClient,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._fast_path = fast_path
        self._slow_path = slow_path
        self._vectors = vectors
        self._router = router
        self._retriever = retriever
        self._synthesizer = synthesizer
        self._llm = llm
        self._worker = _ConsolidationWorker(slow_path)

    @staticmethod
    def create(
        llm: LLMClient,
        embedder: Embedder,
        tagger: TemporalTagger,
        tokens: TokenCounter,
        config: Config = DEFAULT,
        classifier: IntentClassifier | None = None,
    ) -> MagmaMemory:
        """An empty memory with the in-memory backends (NetworkX graph, FAISS, BM25) and the `magma` profile."""
        graph = NetworkXGraphStore(MAGMA_PROFILE)
        vectors = FaissVectorIndex(embedder.dimension)
        keywords = BM25Index(config.lexical_index)
        consolidation_queue: queue.Queue[str] = queue.Queue()
        fast_path = FastPathIngestion(graph, vectors, keywords, embedder, consolidation_queue, config.ingest)
        slow_path = SlowPathConsolidation(graph, llm, tagger, consolidation_queue, config.ingest, config.temporal)
        intent_classifier = classifier or build_default_classifier(tagger, config.router, config.temporal)
        router = Router(embedder, tagger, intent_classifier, config.temporal)
        retriever = Retriever.create(graph, vectors, keywords, embedder, tokens, config)
        synthesizer = AdaptiveQASynthesizer(llm)
        return MagmaMemory(graph, fast_path, slow_path, vectors, router, retriever, synthesizer, llm)

    def consolidate(self) -> None:
        self._worker.drain()

    def close(self) -> None:
        """Stop the background consolidation thread."""
        self._worker.close()

    def usage(self) -> Usage:
        return self._llm.total()

    def save(self, directory: Path) -> None:
        save_graph(self._graph, directory / "graph.json")
        save_vectors(self._vectors, directory / "vectors.npz")

    def _add_turn(self, turn: Turn) -> None:
        self._fast_path.ingest(turn)
        self._worker.notify()

    def _answer(self, question: str, reference_time: datetime, viewer: str | None) -> Reply:
        """The paper's memory has no observers: every viewer gets the omniscient answer."""
        route = self._router.route(question, reference_time)
        retrieval = self._retriever.retrieve(route)
        synthesized = self._synthesizer.synthesize(route, retrieval.context)
        return Reply(answer=synthesized.answer, trace=build_trace(route, retrieval, synthesized))


def build_trace(route: RouterOutput, retrieval: Retrieval, synthesized: SynthesizedAnswer) -> JSONObject:
    """What the query stages produced, for `/why`, `answers.jsonl` and the regression snapshot."""
    window: JSONValue = (
        {"start": route.time_window.start.isoformat(), "end": route.time_window.end.isoformat()}
        if route.time_window is not None
        else None
    )
    keywords: list[JSONValue] = list(route.keywords)
    anchors: list[JSONValue] = [_anchor_trace(anchor) for anchor in retrieval.anchors.anchors]
    visits: list[JSONValue] = [_visit_trace(visit) for visit in retrieval.traversal.visits]
    dropped: JSONObject = {node_type: count for node_type, count in retrieval.traversal.dropped_by_node_type.items()}
    included: list[JSONValue] = list(retrieval.context.included)
    omitted: list[JSONValue] = list(retrieval.context.omitted)
    messages: list[JSONValue] = [
        {"role": message["role"], "content": message["content"]} for message in synthesized.messages
    ]
    return {
        "intent": route.intent.value,
        "intent_reason": route.intent_reason,
        "time_window": window,
        "keywords": keywords,
        "time_channel_size": retrieval.anchors.time_channel_size,
        "anchors": anchors,
        "visits": visits,
        "transitions_checked": retrieval.traversal.transitions_checked,
        "dropped_by_node_type": dropped,
        "included": included,
        "omitted": omitted,
        "context_tokens": retrieval.context.tokens,
        "context": retrieval.context.text,
        "dynamic_instruction": synthesized.dynamic_instruction,
        "messages": messages,
    }


def _anchor_trace(anchor: Anchor) -> JSONObject:
    ranks: JSONObject = {
        "vector": anchor.ranks.vector,
        "keyword": anchor.ranks.keyword,
        "time": anchor.ranks.time,
    }
    return {"node_id": anchor.node_id, "score": round(anchor.score, 6), "ranks": ranks}


def _visit_trace(visit: Visit) -> JSONObject:
    return {
        "node_id": visit.node_id,
        "depth": visit.depth,
        "score": round(visit.score, 6),
        "transition_score": round(visit.transition_score, 6) if visit.transition_score is not None else None,
        "parent_id": visit.parent_id,
        "edge_type": visit.edge_type,
        "similarity": round(visit.similarity, 6) if visit.similarity is not None else None,
    }
