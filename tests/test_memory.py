from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from magma.embed import Matrix, Vector, normalize
from magma.ingest import OutOfOrderTurn, Turn
from magma.llm import ChatRequest, ClientSettings, Completion, LLMClient, LLMResult, Usage
from magma.memory import MagmaMemory, MemorySystem, Reply
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

    def _answer(self, question: str, reference_time: datetime) -> Reply:
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


class EmptyRepliesBackend:
    """Answers every prompt (extraction and causal) with an empty JSON object, which both schemas accept."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        return Completion(text="{}", prompt_tokens=50, completion_tokens=5, reasoning_tokens=0)


class TwoAxisEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return normalize(np.array([(1.0, float(index)) for index, _ in enumerate(texts, start=1)], dtype=np.float32))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector


def magma_memory(backend: EmptyRepliesBackend) -> MagmaMemory:
    llm = LLMClient(ClientSettings(model="test-model"), backend=backend, cache_dir=None)
    return MagmaMemory.create(llm, TwoAxisEmbedder(), DateparserTagger())


def test_magma_memory_builds_the_graph_and_consolidates_on_demand(tmp_path: Path) -> None:
    backend = EmptyRepliesBackend()
    memory = magma_memory(backend)
    memory.add_turn(Turn("Caroline", "I went to a support group yesterday.", START))
    memory.add_turn(Turn("Melanie", "That sounds brave.", START))
    assert backend.requests == []  # the fast path never calls the LLM

    memory.consolidate()
    # one extraction per event, and one causal prompt for the second event (the first has no earlier events)
    assert len(backend.requests) == 3
    assert memory.usage().calls == 3

    memory.save(tmp_path)
    assert (tmp_path / "graph.json").exists()
    assert (tmp_path / "vectors.npz").exists()


def test_magma_memory_does_not_answer_yet() -> None:
    memory = magma_memory(EmptyRepliesBackend())
    with pytest.raises(NotImplementedError):
        memory.query("What did Caroline do?", START)
