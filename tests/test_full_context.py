"""C1 Full Context: the C.2 prompt over every turn, or over the turns the viewer saw."""

import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from baselines.full_context import FullContextMemory
from bench.runner import System, create_memory
from magma.config import Ablation
from magma.embed import Matrix, Vector, normalize
from magma.ingest import Turn
from magma.llm import ChatRequest, ClientSettings, Completion, LLMClient
from magma.memory import MagmaMemory
from magma.router import KeywordClassifier
from magma.temporal import DateparserTagger
from narrative.memory import NarrativeMemory

START = datetime(2024, 3, 1, 20, 0)
HALL = frozenset({"Lia", "Brom", "Ceres"})
CORRIDOR = frozenset({"Dara"})


class EchoBackend:
    """Answers with a fixed text and keeps every request, so a test can read the prompt."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        text = json.dumps({"entities": []}) if "response_format" in request else "scripted answer"
        return Completion(text=text, prompt_tokens=100, completion_tokens=3, reasoning_tokens=0)


class SameVectorEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return normalize(np.array([(1.0, 1.0) for _ in texts], dtype=np.float32))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def build(visible_only: bool, context_window: int | None = None) -> tuple[FullContextMemory, EchoBackend]:
    backend = EchoBackend()
    llm = LLMClient(ClientSettings(model="test-model", context_window=context_window), backend=backend, cache_dir=None)
    embedder = SameVectorEmbedder()
    memory = FullContextMemory.create(
        llm, embedder, DateparserTagger(), embedder, classifier=KeywordClassifier(), visible_only=visible_only
    )
    memory.add_turn(Turn("narrator", "Lia casts a disarming spell on Brom in the hall.", START, HALL))
    memory.add_turn(Turn("Brom", '"Nobody attacked me, I dropped the wand."', START + timedelta(minutes=1), CORRIDOR))
    memory.consolidate()
    return memory, backend


def test_every_turn_is_in_the_prompt_in_the_format_of_the_graph_memories() -> None:
    memory, backend = build(visible_only=False)
    answer = memory.query("What happened to the wand?", START + timedelta(hours=1), viewer="Dara")
    assert answer.answer == "scripted answer"
    assert answer.trace["included"] == ["e1", "e2"]  # without the visible filter, the viewer changes nothing
    user = backend.requests[-1]["messages"][1]["content"]
    assert "<t:2024-03-01 20:00> narrator: Lia casts a disarming spell on Brom in the hall. <ref:e1>" in user
    assert '<t:2024-03-01 20:01> Brom: "Nobody attacked me, I dropped the wand." <ref:e2>' in user
    assert backend.requests[-1]["messages"][0]["content"].startswith("System Role: You are a precision QA assistant")


def test_the_visible_variant_shows_only_what_the_viewer_saw() -> None:
    memory, _ = build(visible_only=True)
    later = START + timedelta(hours=1)
    assert memory.query("What happened?", later, viewer="Dara").trace["included"] == ["e2"]
    assert memory.query("What happened?", later, viewer="Brom").trace["included"] == ["e1", "e2"]  # his own turn
    assert memory.query("What happened?", later, viewer="Lia").trace["included"] == ["e1"]
    assert memory.query("What happened?", later).trace["included"] == ["e1", "e2"]  # the omniscient reader


def test_a_transcript_over_the_window_is_recorded_not_cut() -> None:
    memory, backend = build(visible_only=False, context_window=50)
    calls_before = len(backend.requests)
    answer = memory.query("What happened to the wand?", START + timedelta(hours=1))
    assert answer.answer == ""
    assert answer.trace["context_overflow"] is True
    assert len(backend.requests) == calls_before  # nothing was sent


def test_save_writes_the_transcript(tmp_path: Path) -> None:
    memory, _ = build(visible_only=False)
    memory.save(tmp_path / "conv")
    lines = (tmp_path / "conv" / "transcript.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and lines[1].endswith("<ref:e2>")


def test_the_runner_builds_the_chosen_system() -> None:
    llm = LLMClient(ClientSettings(model="test-model"), backend=EchoBackend(), cache_dir=None)
    embedder = SameVectorEmbedder()
    tagger = DateparserTagger()
    classifier = KeywordClassifier()
    expected = {
        System.MAGMA: MagmaMemory,
        System.NARRATIVE: NarrativeMemory,
        System.FULL_CONTEXT: FullContextMemory,
        System.FULL_CONTEXT_VISIBLE: FullContextMemory,
    }
    for system, memory_class in expected.items():
        memory = create_memory(system, None, llm, embedder, embedder, tagger, classifier)
        try:
            assert type(memory) is memory_class
        finally:
            memory.close()
    assert System.MAGMA.takes_ablation and not System.FULL_CONTEXT.takes_ablation
    ablated = create_memory(System.MAGMA, Ablation.NO_CAUSAL, llm, embedder, embedder, tagger, classifier)
    ablated.close()
