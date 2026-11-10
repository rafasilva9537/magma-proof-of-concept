"""The memory-system boundary and the MAGMA facade that implements it.

`MemorySystem` is the only thing the benchmark runner, the metrics and the
consoles know about a memory: turns go in with `add_turn`, `consolidate`
finishes the pending background work, and `query` answers a question. Every
architecture and every baseline implements it, so they run on the same
benchmark code. The base class keeps the rules every implementation must
follow: turns arrive in time order, and each answer carries its latency and
the LLM usage it took.

`MagmaMemory` assembles the fast path (Alg. 2) and the slow path (Alg. 3) over
one graph and its indexes. Answering (Stages 1–4 of Alg. 1 and the QA prompt)
is not wired in yet: until then `query` raises.
"""

from __future__ import annotations

import queue
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
from magma.json_types import JSONObject
from magma.llm import LLMClient, Usage
from magma.networkx_graph import NetworkXGraphStore
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

    def query(self, question: str, reference_time: datetime) -> MemoryAnswer:
        """Answer `question` as of `reference_time` (the "now" used to read relative dates in it)."""
        usage_before = self.usage()
        start = time.perf_counter()
        reply = self._answer(question, reference_time)
        latency_s = time.perf_counter() - start
        return MemoryAnswer(answer=reply.answer, trace=reply.trace, usage=self.usage().since(usage_before), latency_s=latency_s)

    @abstractmethod
    def usage(self) -> Usage:
        """Total LLM usage of this memory so far (ingestion and queries)."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Write what the memory holds, for inspection and for the regression snapshot."""

    @abstractmethod
    def _add_turn(self, turn: Turn) -> None: ...

    @abstractmethod
    def _answer(self, question: str, reference_time: datetime) -> Reply: ...


class MagmaMemory(MemorySystem):
    """The MAGMA memory: one typed multigraph with its vector and lexical indexes."""

    def __init__(
        self,
        graph: GraphStore,
        fast_path: FastPathIngestion,
        slow_path: SlowPathConsolidation,
        vectors: FaissVectorIndex,
        llm: LLMClient,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._fast_path = fast_path
        self._slow_path = slow_path
        self._vectors = vectors
        self._llm = llm

    @staticmethod
    def create(llm: LLMClient, embedder: Embedder, tagger: TemporalTagger, config: Config = DEFAULT) -> MagmaMemory:
        """An empty memory with the in-memory backends (NetworkX graph, FAISS, BM25) and the `magma` profile."""
        graph = NetworkXGraphStore(MAGMA_PROFILE)
        vectors = FaissVectorIndex(embedder.dimension)
        consolidation_queue: queue.Queue[str] = queue.Queue()
        fast_path = FastPathIngestion(graph, vectors, BM25Index(), embedder, consolidation_queue, config.ingest)
        slow_path = SlowPathConsolidation(graph, llm, tagger, consolidation_queue, config.ingest, config.temporal)
        return MagmaMemory(graph, fast_path, slow_path, vectors, llm)

    def consolidate(self) -> None:
        self._slow_path.drain()

    def usage(self) -> Usage:
        return self._llm.total()

    def save(self, directory: Path) -> None:
        save_graph(self._graph, directory / "graph.json")
        save_vectors(self._vectors, directory / "vectors.npz")

    def _add_turn(self, turn: Turn) -> None:
        self._fast_path.ingest(turn)

    def _answer(self, question: str, reference_time: datetime) -> Reply:
        raise NotImplementedError("answering needs the retrieval stages and the QA prompt, which are not wired in yet")
