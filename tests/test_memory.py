from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
import threading

import numpy as np
import pytest

from magma.config import DEFAULT, LLMConfig
from magma.embed import Matrix, Vector, normalize
from magma.ingest import OutOfOrderTurn, Turn
from magma.llm import ChatRequest, ClientSettings, Completion, LLMClient, LLMResult, LLMUnavailable, TransientBackendError, Usage
from magma.memory import MagmaMemory, MemorySystem, Reply
from magma.qa import FACTUAL_INSTRUCTION, MULTI_HOP_INSTRUCTION
from magma.router import KeywordClassifier
from magma.temporal import DateparserTagger

START = datetime(2023, 5, 8, 13, 0)


class RecordingMemory(MemorySystem):
    """A memory that only records what it is given; each answer spends one call of 100 + 10 tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.turns: list[Turn] = []
        self._usage = Usage()

    def consolidate(self) -> None:
        pass

    def usage(self) -> Usage:
        copy = Usage()
        copy.merge(self._usage)
        return copy

    def save(self, directory: Path) -> None:
        pass

    def _add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def _answer(self, question: str, reference_time: datetime, viewer: str | None) -> Reply:
        self._usage.add(LLMResult(text="", prompt_tokens=100, completion_tokens=10, reasoning_tokens=0, latency_s=0.5, cached=False))
        return Reply(answer=f"answer to {question}", trace={"turns_seen": len(self.turns)})


def test_turns_must_come_in_time_order() -> None:
    memory = RecordingMemory()
    memory.add_turn(Turn("Caroline", "hi", START))
    memory.add_turn(Turn("Melanie", "same session", START))  # LoCoMo dates sessions, so equal times are fine
    with pytest.raises(OutOfOrderTurn):
        memory.add_turn(Turn("Caroline", "back in time", START - timedelta(minutes=1)))
    assert len(memory.turns) == 2


def test_query_reports_the_usage_of_that_query_only() -> None:
    memory = RecordingMemory()
    memory.add_turn(Turn("Caroline", "hi", START))
    memory.query("first?", START)
    answer = memory.query("second?", START)
    assert answer.answer == "answer to second?"
    assert answer.trace == {"turns_seen": 1}
    assert (answer.usage.calls, answer.usage.prompt_tokens, answer.usage.billed_completion_tokens) == (1, 100, 10)
    assert answer.latency_s >= 0.0


class ScriptedMemoryBackend:
    """Answers extraction/causal JSON prompts and free-text QA prompts, optionally blocking until released."""

    def __init__(self, qa_reply: str = "a support group", gate: threading.Event | None = None) -> None:
        self.qa_reply = qa_reply
        self._gate = gate
        self.started = threading.Event()
        self.requests: list[ChatRequest] = []
        self._lock = threading.Lock()

    def complete(self, request: ChatRequest) -> Completion:
        self.started.set()
        if self._gate is not None:
            self._gate.wait(timeout=5.0)
        with self._lock:
            self.requests.append(request)
        if "response_format" in request:
            return Completion(text="{}", prompt_tokens=50, completion_tokens=5, reasoning_tokens=0)
        return Completion(text=self.qa_reply, prompt_tokens=120, completion_tokens=4, reasoning_tokens=0)


class FailingThenRecoveringBackend:
    """Fails on the first call, then returns `{}` once `recover()` is called."""

    def __init__(self) -> None:
        self.should_fail = True
        self.calls = 0

    def complete(self, request: ChatRequest) -> Completion:
        del request
        self.calls += 1
        if self.should_fail:
            raise TransientBackendError("provider down")
        return Completion(text="{}", prompt_tokens=40, completion_tokens=4, reasoning_tokens=0)


class TwoAxisEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return normalize(np.array([(1.0, float(index)) for index, _ in enumerate(texts, start=1)], dtype=np.float32))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def magma_memory(backend: ScriptedMemoryBackend) -> MagmaMemory:
    llm = LLMClient(ClientSettings(model="test-model"), backend=backend, cache_dir=None)
    embedder = TwoAxisEmbedder()
    return MagmaMemory.create(llm, embedder, DateparserTagger(), embedder, classifier=KeywordClassifier())


def test_add_turn_does_not_block_on_background_consolidation_and_consolidate_drains(tmp_path: Path) -> None:
    gate = threading.Event()
    backend = ScriptedMemoryBackend(gate=gate)
    memory = magma_memory(backend)
    try:
        memory.add_turn(Turn("Caroline", "I went to a support group yesterday.", START))
        memory.add_turn(Turn("Melanie", "That sounds brave.", START))
        assert backend.started.wait(timeout=2.0)  # the background worker picked up the first event
        assert len(backend.requests) == 0  # add_turn returned while the slow path is still waiting on the LLM

        gate.set()
        memory.consolidate()
        # one extraction per event, and one causal prompt for the second event (the first has no earlier events)
        assert len(backend.requests) == 3
        assert memory.usage().calls == 3

        memory.save(tmp_path)
        assert (tmp_path / "graph.json").exists()
        assert (tmp_path / "vectors.npz").exists()
    finally:
        gate.set()
        memory.close()


def test_consolidate_propagates_worker_failure_and_resumes_the_interrupted_event() -> None:
    backend = FailingThenRecoveringBackend()
    one_attempt = DEFAULT.llm
    llm = LLMClient(
        ClientSettings(model="test-model"),
        backend=backend,
        cache_dir=None,
        config=LLMConfig(max_attempts=1),
        sleep=lambda _: None,
    )
    del one_attempt
    embedder = TwoAxisEmbedder()
    memory = MagmaMemory.create(llm, embedder, DateparserTagger(), embedder, classifier=KeywordClassifier())
    try:
        memory.add_turn(Turn("Caroline", "I went to a support group yesterday.", START))
        with pytest.raises(LLMUnavailable):
            memory.consolidate()

        backend.should_fail = False
        memory.consolidate()
        assert memory.usage().calls == 1
    finally:
        memory.close()


def test_magma_memory_answers_from_retrieved_events_and_returns_the_trace() -> None:
    backend = ScriptedMemoryBackend(qa_reply="Because she wanted community support")
    memory = magma_memory(backend)
    try:
        memory.add_turn(Turn("Caroline", "I felt isolated after moving.", START))
        memory.add_turn(Turn("Caroline", "So I joined a support group in May 2023.", START + timedelta(days=1)))
        memory.consolidate()

        answer = memory.query("Why did Caroline join the support group?", START + timedelta(days=2))

        assert answer.answer == "Because she wanted community support"
        assert (answer.usage.calls, answer.usage.prompt_tokens, answer.usage.completion_tokens) == (1, 120, 4)
        trace = answer.trace
        assert trace["intent"] == "WHY"
        assert trace["dynamic_instruction"] == MULTI_HOP_INSTRUCTION
        assert trace["included"] == ["e1", "e2"]
        assert trace["omitted"] == []
        assert "<t:2023-05-08 13:00> Caroline: I felt isolated after moving. <ref:e1>" in str(trace["context"])
        assert "<t:2023-05-09 13:00> Caroline: So I joined a support group in May 2023. <ref:e2>" in str(trace["context"])

        entity_answer = memory.query("Who joined a support group?", START + timedelta(days=2))
        assert entity_answer.trace["intent"] == "ENTITY"
        assert entity_answer.trace["dynamic_instruction"] == FACTUAL_INSTRUCTION
    finally:
        memory.close()
