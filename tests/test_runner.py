import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bench.benchmark import Conversation, Question
from bench.judge import Judge
from bench.runner import (
    BudgetExceeded,
    CostMeter,
    Prices,
    RunMode,
    RunOutput,
    run_benchmark,
    select_questions,
    smallest_conversation,
    stratified_sample,
)
from magma.ingest import Turn
from magma.llm import ChatRequest, ClientSettings, Completion, LLMClient, LLMResult, Usage
from magma.memory import MemorySystem, Reply

START = datetime(2023, 5, 8, 13, 0)
END = datetime(2023, 7, 1, 9, 0)


class ScriptedMemory(MemorySystem):
    """Records the order of the calls; answers with the gold it was given, and each answer costs 1000 + 100 tokens."""

    def __init__(self, log: list[str], answers: dict[str, str]) -> None:
        super().__init__()
        self.log = log
        self.answers = answers
        self._usage = Usage()

    def consolidate(self) -> None:
        self.log.append("consolidate")
        self._usage.add(LLMResult(text="", prompt_tokens=10_000, completion_tokens=1_000, reasoning_tokens=0, latency_s=2.0, cached=False))

    def usage(self) -> Usage:
        copy = Usage()
        copy.merge(self._usage)
        return copy

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "saved.txt").write_text("ok", encoding="utf-8")

    def _add_turn(self, turn: Turn) -> None:
        self.log.append(f"turn {turn.text}")

    def _answer(self, question: str, reference_time: datetime) -> Reply:
        self.log.append(f"query {question} @ {reference_time:%Y-%m-%d}")
        self._usage.add(LLMResult(text="", prompt_tokens=1_000, completion_tokens=100, reasoning_tokens=0, latency_s=0.5, cached=False))
        return Reply(answer=self.answers.get(question, "Information not found"), trace={"intent": "ENTITY"})


class ScoreByMatchBackend:
    """A judge that gives 1.0 when the candidate repeats the gold and 0.0 otherwise."""

    def complete(self, request: ChatRequest) -> Completion:
        user = request["messages"][1]["content"]
        gold = user.split("| Gold: ")[1].split(" | Candidate: ")[0]
        candidate = user.split(" | Candidate: ")[1]
        score = 1.0 if gold == candidate else 0.0
        return Completion(text=json.dumps({"score": score, "reasoning": "match" if score else "no match"}), prompt_tokens=200, completion_tokens=20, reasoning_tokens=0)


def make_judge() -> Judge:
    return Judge(LLMClient(ClientSettings(model="judge-model"), backend=ScoreByMatchBackend(), cache_dir=None))


def question(number: int, category: str, conversation_id: str = "conv-a") -> Question:
    return Question(f"{conversation_id}-q{number}", conversation_id, category, f"question {number}?", f"gold {number}", ())


def conversation(conversation_id: str, turns: int, questions: list[Question]) -> Conversation:
    return Conversation(
        id=conversation_id,
        turns=tuple(Turn("Caroline", f"{conversation_id} turn {i}", START + timedelta(minutes=i)) for i in range(turns)),
        reference_time=END,
        questions=tuple(questions),
    )


def test_stratified_sample_keeps_the_category_proportions_and_the_order() -> None:
    questions = [question(i, "SINGLE_HOP") for i in range(60)] + [question(60 + i, "TEMPORAL") for i in range(30)]
    questions += [question(90 + i, "OPEN_DOMAIN") for i in range(10)]
    sample = stratified_sample(questions, 20, seed=7)
    assert Counter(q.category for q in sample) == {"SINGLE_HOP": 12, "TEMPORAL": 6, "OPEN_DOMAIN": 2}
    positions = [questions.index(q) for q in sample]
    assert positions == sorted(positions)


def test_stratified_sample_hands_the_remainder_to_the_largest_fractions() -> None:
    # 7 of 10 → exact quotas 3.5 (A), 2.1 (B), 1.4 (C): floors 3 + 2 + 1 = 6, the last one goes to A (0.5)
    questions = [question(i, "A") for i in range(5)] + [question(5 + i, "B") for i in range(3)] + [question(8 + i, "C") for i in range(2)]
    assert Counter(q.category for q in stratified_sample(questions, 7, seed=1)) == {"A": 4, "B": 2, "C": 1}


def test_the_same_seed_gives_the_same_questions_and_another_seed_others() -> None:
    conversations = [conversation("conv-a", 1, [question(i, "SINGLE_HOP") for i in range(50)])]
    first = select_questions(conversations, 10, seed=7)
    assert first == select_questions(conversations, 10, seed=7)
    assert first != select_questions(conversations, 10, seed=8)
    assert len(select_questions(conversations, None, seed=7)) == 50


def test_categories_filter_the_pool() -> None:
    conversations = [conversation("conv-a", 1, [question(1, "SINGLE_HOP"), question(2, "TEMPORAL")])]
    assert [q.id for q in select_questions(conversations, None, 7, ["TEMPORAL"])] == ["conv-a-q2"]


def test_smallest_conversation_by_turns() -> None:
    assert smallest_conversation([conversation("conv-a", 3, []), conversation("conv-b", 2, [])]).id == "conv-b"


def test_every_turn_and_the_slow_path_come_before_the_questions(tmp_path: Path) -> None:
    asked = [question(1, "SINGLE_HOP"), question(2, "TEMPORAL")]
    conversations = [conversation("conv-a", 2, asked)]
    log: list[str] = []
    summary = run_benchmark(
        conversations, asked, lambda: ScriptedMemory(log, {"question 1?": "gold 1"}), make_judge(),
        CostMeter(None, None, None), RunOutput(tmp_path), "test-run",
    )
    assert log == [
        "turn conv-a turn 0",
        "turn conv-a turn 1",
        "consolidate",
        "query question 1? @ 2023-07-01",  # asked as of the end of the conversation
        "query question 2? @ 2023-07-01",
    ]
    assert summary.questions_answered == 2
    assert summary.overall.judge == 0.5
    assert [(c.category, c.judge) for c in summary.by_category] == [("SINGLE_HOP", 1.0), ("TEMPORAL", 0.0)]
    assert summary.tokens_per_query == 1_100
    assert (tmp_path / "memories" / "conv-a" / "saved.txt").exists()

    answers = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert answers[0]["judge_score"] == 1.0
    assert answers[0]["f1"] == 1.0
    assert answers[0]["trace"] == {"intent": "ENTITY"}
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["questions_answered"] == 2


def test_each_conversation_gets_a_fresh_memory_and_only_its_questions(tmp_path: Path) -> None:
    conversations = [conversation("conv-a", 1, [question(1, "X")]), conversation("conv-b", 1, [question(1, "X", "conv-b")])]
    asked = [question(1, "X", "conv-b")]
    memories: list[list[str]] = []

    def new_memory() -> MemorySystem:
        memories.append([])
        return ScriptedMemory(memories[-1], {})

    run_benchmark(conversations, asked, new_memory, make_judge(), CostMeter(None, None, None), RunOutput(tmp_path), "test-run")
    assert memories == [["turn conv-b turn 0", "consolidate", "query question 1? @ 2023-07-01"]]  # conv-a has no question


def test_build_only_builds_every_conversation_and_asks_nothing(tmp_path: Path) -> None:
    conversations = [conversation("conv-a", 2, [question(1, "X")]), conversation("conv-b", 1, [])]
    log: list[str] = []
    summary = run_benchmark(
        conversations, [], lambda: ScriptedMemory(log, {}), None, CostMeter(None, None, None), RunOutput(tmp_path), "build", build_only=True
    )
    assert not any(entry.startswith("query") for entry in log)
    assert summary.conversations == ["conv-a", "conv-b"]
    assert summary.build_calls == 2
    builds = [json.loads(line) for line in (tmp_path / "builds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [build["turns"] for build in builds] == [2, 1]


def test_prices_count_only_the_calls_sent() -> None:
    usage = Usage()
    usage.add(LLMResult(text="", prompt_tokens=1_000_000, completion_tokens=100_000, reasoning_tokens=0, latency_s=1.0, cached=False))
    usage.add(LLMResult(text="", prompt_tokens=1_000_000, completion_tokens=100_000, reasoning_tokens=0, latency_s=1.0, cached=True))
    # only the first call is billed: 1 M input at US$ 0.10 + 0.1 M output at US$ 0.40
    assert Prices(0.10, 0.40).cost(usage) == pytest.approx(0.14)


def test_the_run_stops_once_the_cap_is_passed_and_keeps_what_it_did(tmp_path: Path) -> None:
    asked = [question(i, "X") for i in range(5)]
    meter = CostMeter(2.5, Prices(1_000.0, 0.0), Prices(0.0, 0.0))  # the build's 10k input tokens cost US$ 10
    summary = run_benchmark(
        [conversation("conv-a", 1, asked)], asked, lambda: ScriptedMemory([], {}), make_judge(), meter, RunOutput(tmp_path), "capped"
    )
    assert summary.stopped is not None and "cap" in summary.stopped
    assert summary.questions_answered == 0
    assert (tmp_path / "builds.jsonl").exists()  # the build that passed the cap is recorded


def test_the_report_pilot_stops_when_the_projection_passes_the_cap(tmp_path: Path) -> None:
    asked = [question(i, "X") for i in range(10)]
    # build: 10k input tokens at US$ 1/M = US$ 0.01; each question: 1k memory input = US$ 0.001
    # after a pilot of 2 questions: 0.01 + 0.002 spent, 8 more at 0.001 → ~US$ 0.02 projected, above a cap of 0.015
    meter = CostMeter(0.015, Prices(1.0, 0.0), Prices(0.0, 0.0))
    summary = run_benchmark(
        [conversation("conv-a", 1, asked)], asked, lambda: ScriptedMemory([], {}), make_judge(), meter, RunOutput(tmp_path), "pilot",
        mode=RunMode.REPORT, pilot_questions=2,
    )
    assert summary.questions_answered == 2
    assert summary.stopped is not None and summary.stopped.startswith("pilot")


def test_a_cap_needs_prices() -> None:
    with pytest.raises(ValueError):
        CostMeter(1.0, None, None)
