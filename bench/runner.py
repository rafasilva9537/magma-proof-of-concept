"""Benchmark runner: builds each memory from a conversation, asks the questions, judges the answers.

The runner knows memories only through `MemorySystem`, so every architecture
and baseline runs the same code on the same questions:

1. for each conversation, a new memory gets every turn in order (`add_turn`),
   then `consolidate` finishes the slow path; the build time, LLM usage and
   cost are recorded and the memory is saved;
2. each question is asked as of the end of the conversation (`query`); the
   answer is scored by the judge (prompt C.3) and by token F1 and BLEU-1;
3. the summary gives the scores per category and overall, and the costs of
   Tab. 3: build time, tokens per query and latency per query.

Two modes. `sample` (default) runs a small stratified sample for development:
by default the smallest conversation and 40 of its questions. `report` runs
every conversation and every question (or a fixed stratified sample, with
`--questions`), needs a cost cap, and starts with a pilot: after the first
questions it projects the cost of the rest and stops if the projection passes
the cap. In both modes the cap, when given, stops the run as soon as the money
spent passes it. Samples are stratified by question category, drawn with a
fixed seed, and do not depend on the memory under test, so every system gets
the same questions. `--build-only` builds the memories and asks nothing.

    python -m bench.runner --locomo MAGMA-original-repo/data/locomo10.json --build-only
    python -m bench.runner --locomo MAGMA-original-repo/data/locomo10.json --mode report --max-cost-usd 20 \\
        --llm-prices <input> <output> --judge-prices <input> <output>

The memory LLM is read from the LLM_* variables and the judge from JUDGE_*
(see .env.example); both are cached, so running the same command again costs
nothing. Everything goes to results/<run_id>/: the run settings, one record per
build and per answer (with its retrieval trace), the summary, the saved
memories and the raw log of every LLM call.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from bench.benchmark import Conversation, Question
from bench.judge import Judge
from bench.lexical_metrics import bleu1, token_f1
from bench.locomo import load_samples, to_conversation
from magma.config import DEFAULT
from magma.embed import InstructedEmbedder
from magma.json_types import JSONObject, JSONValue
from magma.llm import ClientSettings, LLMClient, Usage
from magma.memory import MagmaMemory, MemorySystem
from magma.router import build_default_classifier
from magma.temporal import DateparserTagger

SAMPLE_QUESTIONS = 40
PILOT_QUESTIONS = 20


class RunMode(StrEnum):
    SAMPLE = "SAMPLE"
    REPORT = "REPORT"


@dataclass(frozen=True)
class Prices:
    """USD per million tokens. Reasoning tokens are billed as output, and are part of the completion tokens."""

    input_per_million: float
    output_per_million: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.billed_prompt_tokens * self.input_per_million + usage.billed_completion_tokens * self.output_per_million
        ) / 1_000_000


class BudgetExceeded(RuntimeError):
    """The money spent, or the pilot's projection, passed the cost cap."""


class CostMeter:
    """Adds up what the calls sent to the providers cost (cached calls are free) and enforces the cap."""

    def __init__(self, cap_usd: float | None, memory_prices: Prices | None, judge_prices: Prices | None) -> None:
        if cap_usd is not None and (memory_prices is None or judge_prices is None):
            raise ValueError("a cost cap needs the prices of the memory LLM and of the judge")
        self._cap_usd = cap_usd
        self._memory_prices = memory_prices
        self._judge_prices = judge_prices
        self.spent_usd = 0.0
        self.question_spent_usd = 0.0

    @property
    def priced(self) -> bool:
        return self._memory_prices is not None and self._judge_prices is not None

    def charge_build(self, usage: Usage) -> float | None:
        return self._cost(usage, self._memory_prices)

    def charge_question(self, memory_usage: Usage, judge_usage: Usage) -> float | None:
        memory_cost = self._cost(memory_usage, self._memory_prices)
        judge_cost = self._cost(judge_usage, self._judge_prices)
        if memory_cost is None or judge_cost is None:
            return None
        self.question_spent_usd += memory_cost + judge_cost
        return memory_cost + judge_cost

    def check(self) -> None:
        """Stops the run once the money spent passes the cap; the record that passed it is already written."""
        if self._cap_usd is not None and self.spent_usd > self._cap_usd:
            raise BudgetExceeded(f"spent US$ {self.spent_usd:.4f}, above the cap of US$ {self._cap_usd:.2f}")

    def check_projection(self, answered: int, remaining: int) -> float | None:
        """After the pilot: what the run would cost at the pilot's cost per question; stops if that passes the cap."""
        if self._cap_usd is None or answered == 0:
            return None
        projected = self.spent_usd + self.question_spent_usd / answered * remaining
        if projected > self._cap_usd:
            raise BudgetExceeded(
                f"pilot: {answered} questions cost US$ {self.question_spent_usd:.4f}; the remaining {remaining} "
                f"would bring the run to ~US$ {projected:.2f}, above the cap of US$ {self._cap_usd:.2f}"
            )
        return projected

    def _cost(self, usage: Usage, prices: Prices | None) -> float | None:
        if prices is None:
            return None
        cost = prices.cost(usage)
        self.spent_usd += cost
        return cost


@dataclass(frozen=True)
class BuildRecord:
    conversation_id: str
    turns: int
    seconds: float  # wall clock of add_turn for every turn plus consolidate
    usage: Usage
    cost_usd: float | None


@dataclass(frozen=True)
class AnswerRecord:
    question_id: str
    conversation_id: str
    category: str
    question: str
    gold: str
    answer: str
    judge_score: float | None
    judge_reasoning: str
    judge_error: str | None
    f1: float
    bleu1: float
    latency_s: float
    usage: Usage  # memory LLM during the query
    judge_usage: Usage
    cost_usd: float | None
    trace: JSONObject


@dataclass(frozen=True)
class CategoryScore:
    category: str
    questions: int
    judged: int  # questions with a valid verdict; the judge mean is over these
    judge: float | None
    f1: float | None
    bleu1: float | None


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    mode: RunMode
    build_only: bool
    conversations: list[str]
    questions_planned: int
    questions_answered: int
    judge_errors: int
    overall: CategoryScore
    by_category: list[CategoryScore]
    build_hours: float  # wall clock, summed over conversations
    build_llm_hours: float  # latency of the LLM calls of the builds, cached ones counted with their original latency
    build_cached_calls: int
    build_calls: int
    tokens_per_query: float | None  # prompt + completion tokens of the memory LLM, mean over questions
    prompt_tokens_per_query: float | None
    latency_per_query_s: float | None  # wall clock, mean over queries whose LLM calls were all sent (none cached)
    uncached_queries: int
    llm_latency_per_query_s: float | None  # LLM part of the latency, cached calls with their original latency
    spent_usd: float | None
    stopped: str | None


class RunOutput:
    """The files of one run, in results/<run_id>/."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("builds.jsonl", "answers.jsonl", "summary.json"):
            (directory / name).unlink(missing_ok=True)  # the records of this run only

    def write_settings(self, settings: JSONObject) -> None:
        (self.directory / "run.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_build(self, record: BuildRecord) -> None:
        self._append("builds.jsonl", json.dumps(asdict(record), ensure_ascii=False))

    def append_answer(self, record: AnswerRecord) -> None:
        self._append("answers.jsonl", json.dumps(asdict(record), ensure_ascii=False))

    def write_summary(self, summary: RunSummary) -> None:
        text = json.dumps(asdict(summary), ensure_ascii=False, indent=2)
        (self.directory / "summary.json").write_text(text, encoding="utf-8")

    def memory_directory(self, conversation_id: str) -> Path:
        return self.directory / "memories" / conversation_id

    def _append(self, name: str, line: str) -> None:
        with (self.directory / name).open("a", encoding="utf-8") as file:
            file.write(line + "\n")


def stratified_sample(questions: Sequence[Question], size: int, seed: int) -> list[Question]:
    """`size` questions with the categories in the proportions of `questions` (largest remainder), in their order."""
    if size >= len(questions):
        return list(questions)
    by_category: dict[str, list[Question]] = {}
    for question in questions:
        by_category.setdefault(question.category, []).append(question)
    categories = sorted(by_category)
    exact = {category: size * len(by_category[category]) / len(questions) for category in categories}
    quota = {category: int(exact[category]) for category in categories}
    by_remainder = sorted(categories, key=lambda category: (-(exact[category] - quota[category]), category))
    for category in by_remainder[: size - sum(quota.values())]:
        quota[category] += 1

    rng = random.Random(seed)
    chosen: set[str] = set()
    for category in categories:
        chosen.update(question.id for question in rng.sample(by_category[category], quota[category]))
    return [question for question in questions if question.id in chosen]


def select_questions(
    conversations: Sequence[Conversation], count: int | None, seed: int, categories: Sequence[str] | None = None
) -> list[Question]:
    """The questions of the run: all of them, or a stratified sample; the same for every memory under test."""
    pool = [
        question
        for conversation in conversations
        for question in conversation.questions
        if categories is None or question.category in categories
    ]
    return pool if count is None else stratified_sample(pool, count, seed)


def build_memory(conversation: Conversation, memory: MemorySystem) -> tuple[float, Usage]:
    """Every turn in order, then the slow path; returns the wall-clock seconds and the LLM usage."""
    usage_before = memory.usage()
    start = time.perf_counter()
    for turn in conversation.turns:
        memory.add_turn(turn)
    memory.consolidate()
    return time.perf_counter() - start, memory.usage().since(usage_before)


def run_benchmark(
    conversations: Sequence[Conversation],
    questions: Sequence[Question],
    new_memory: Callable[[], MemorySystem],
    judge: Judge | None,
    meter: CostMeter,
    output: RunOutput,
    run_id: str,
    mode: RunMode = RunMode.SAMPLE,
    build_only: bool = False,
    pilot_questions: int = PILOT_QUESTIONS,
) -> RunSummary:
    """Builds a fresh memory per conversation and, unless `build_only`, asks its questions."""
    if not build_only and judge is None:
        raise ValueError("answers need a judge")
    builds: list[BuildRecord] = []
    answers: list[AnswerRecord] = []
    stopped: str | None = None
    try:
        for conversation in conversations:
            asked = [question for question in questions if question.conversation_id == conversation.id]
            if not build_only and not asked:
                continue
            memory = new_memory()
            try:
                seconds, usage = build_memory(conversation, memory)
                memory.save(output.memory_directory(conversation.id))
                build = BuildRecord(conversation.id, len(conversation.turns), round(seconds, 2), usage, meter.charge_build(usage))
                builds.append(build)
                output.append_build(build)
                meter.check()
                if build_only or judge is None:
                    continue
                for question in asked:
                    record = ask(memory, judge, meter, conversation, question)
                    answers.append(record)
                    output.append_answer(record)
                    meter.check()
                    if mode is RunMode.REPORT and len(answers) == pilot_questions:
                        meter.check_projection(len(answers), len(questions) - len(answers))
            finally:
                memory.close()
    except BudgetExceeded as error:
        stopped = str(error)

    summary = summarize(run_id, mode, build_only, conversations, len(questions), builds, answers, meter, stopped)
    output.write_summary(summary)
    return summary


def ask(memory: MemorySystem, judge: Judge, meter: CostMeter, conversation: Conversation, question: Question) -> AnswerRecord:
    answer = memory.query(question.question, conversation.reference_time)
    judge_before = judge.llm.total()
    grade = judge.grade(question.question, question.gold, answer.answer)
    judge_usage = judge.llm.total().since(judge_before)
    return AnswerRecord(
        question_id=question.id,
        conversation_id=question.conversation_id,
        category=question.category,
        question=question.question,
        gold=question.gold,
        answer=answer.answer,
        judge_score=grade.score,
        judge_reasoning=grade.reasoning,
        judge_error=grade.error,
        f1=round(token_f1(answer.answer, question.gold), 4),
        bleu1=round(bleu1(answer.answer, question.gold), 4),
        latency_s=round(answer.latency_s, 3),
        usage=answer.usage,
        judge_usage=judge_usage,
        cost_usd=meter.charge_question(answer.usage, judge_usage),
        trace=answer.trace,
    )


def summarize(
    run_id: str,
    mode: RunMode,
    build_only: bool,
    conversations: Sequence[Conversation],
    questions_planned: int,
    builds: Sequence[BuildRecord],
    answers: Sequence[AnswerRecord],
    meter: CostMeter,
    stopped: str | None,
) -> RunSummary:
    categories = sorted({answer.category for answer in answers})
    uncached = [answer for answer in answers if answer.usage.cached_calls == 0]
    return RunSummary(
        run_id=run_id,
        mode=mode,
        build_only=build_only,
        conversations=[build.conversation_id for build in builds],
        questions_planned=0 if build_only else questions_planned,
        questions_answered=len(answers),
        judge_errors=sum(1 for answer in answers if answer.judge_error is not None),
        overall=score("OVERALL", answers),
        by_category=[score(category, [answer for answer in answers if answer.category == category]) for category in categories],
        build_hours=round(sum(build.seconds for build in builds) / 3600, 4),
        build_llm_hours=round(sum(build.usage.latency_s for build in builds) / 3600, 4),
        build_cached_calls=sum(build.usage.cached_calls for build in builds),
        build_calls=sum(build.usage.calls for build in builds),
        tokens_per_query=_mean([answer.usage.prompt_tokens + answer.usage.completion_tokens for answer in answers]),
        prompt_tokens_per_query=_mean([answer.usage.prompt_tokens for answer in answers]),
        latency_per_query_s=_mean([answer.latency_s for answer in uncached]),
        uncached_queries=len(uncached),
        llm_latency_per_query_s=_mean([answer.usage.latency_s for answer in answers]),
        spent_usd=round(meter.spent_usd, 4) if meter.priced else None,
        stopped=stopped,
    )


def score(category: str, answers: Sequence[AnswerRecord]) -> CategoryScore:
    judged = [answer.judge_score for answer in answers if answer.judge_score is not None]
    return CategoryScore(
        category=category,
        questions=len(answers),
        judged=len(judged),
        judge=_mean(judged),
        f1=_mean([answer.f1 for answer in answers]),
        bleu1=_mean([answer.bleu1 for answer in answers]),
    )


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(np.array(values, dtype=np.float64))), 4)


def smallest_conversation(conversations: Sequence[Conversation]) -> Conversation:
    return min(conversations, key=lambda conversation: (len(conversation.turns), conversation.id))


def run_settings(
    arguments: argparse.Namespace, run_id: str, questions: Sequence[Question], llm: ClientSettings, judge: ClientSettings | None
) -> JSONObject:
    """What the run used, for the report: never the API keys."""
    settings: JSONObject = {
        "run_id": run_id,
        "command": " ".join(sys.argv),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "mode": RunMode(arguments.mode.upper()).value,
        "seed": arguments.seed,
        "build_only": arguments.build_only,
        "question_ids": [question.id for question in questions],
        "max_cost_usd": arguments.max_cost_usd,
        "llm_prices": arguments.llm_prices,
        "judge_prices": arguments.judge_prices,
        "embedding_model": DEFAULT.embedding.model_name,
        "intent_model": DEFAULT.router.intent_model,
        "llm": _client_settings(llm),
        "judge": _client_settings(judge) if judge is not None else None,
    }
    return settings


def _client_settings(settings: ClientSettings) -> JSONObject:
    extra_body: JSONValue = settings.extra_body
    return {"model": settings.model, "base_url": settings.base_url, "context_window": settings.context_window, "extra_body": extra_body}


def _git(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(["git", *arguments], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def print_summary(summary: RunSummary, output: Path) -> None:
    print(f"{summary.run_id}: {len(summary.conversations)} conversations built in {summary.build_hours:.3f} h "
          f"({summary.build_calls} LLM calls, {summary.build_cached_calls} from the cache)")
    if not summary.build_only:
        overall = summary.overall
        print(f"{summary.questions_answered}/{summary.questions_planned} questions; judge {overall.judge}, F1 {overall.f1}, "
              f"BLEU-1 {overall.bleu1}; {summary.judge_errors} invalid verdicts")
        for category in summary.by_category:
            print(f"  {category.category}: {category.questions} questions, judge {category.judge}, F1 {category.f1}")
        print(f"tokens/query {summary.tokens_per_query}, latency/query {summary.latency_per_query_s} s "
              f"over {summary.uncached_queries} uncached queries")
    if summary.spent_usd is not None:
        print(f"spent US$ {summary.spent_usd:.4f}")
    if summary.stopped is not None:
        print(f"STOPPED: {summary.stopped}")
    print(f"Results written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locomo", type=Path, required=True, help="path to locomo10.json")
    parser.add_argument("--mode", choices=["sample", "report"], default="sample")
    parser.add_argument("--conversations", nargs="+", default=None, help="sample ids; default: the smallest (sample) or all (report)")
    parser.add_argument("--questions", type=int, default=None, help=f"stratified sample size; default: {SAMPLE_QUESTIONS} (sample) or all (report)")
    parser.add_argument("--categories", nargs="+", default=None, help="only these categories, e.g. TEMPORAL SINGLE_HOP")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--build-only", action="store_true", help="build the memories and ask nothing")
    parser.add_argument("--max-cost-usd", type=float, default=None, help="cost cap; required in report mode")
    parser.add_argument("--pilot-questions", type=int, default=PILOT_QUESTIONS, help="questions before the projection (report mode)")
    parser.add_argument("--llm-prices", type=float, nargs=2, metavar=("INPUT", "OUTPUT"), default=None, help="memory LLM, USD per million tokens")
    parser.add_argument("--judge-prices", type=float, nargs=2, metavar=("INPUT", "OUTPUT"), default=None, help="judge, USD per million tokens")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    arguments = parser.parse_args()

    mode = RunMode(arguments.mode.upper())
    if mode is RunMode.REPORT and arguments.max_cost_usd is None:
        parser.error("report mode needs --max-cost-usd")
    if arguments.max_cost_usd is not None and (arguments.llm_prices is None or (arguments.judge_prices is None and not arguments.build_only)):
        parser.error("a cost cap needs --llm-prices and, unless --build-only, --judge-prices")

    load_dotenv()
    every_conversation = [to_conversation(sample) for sample in load_samples(arguments.locomo)]
    if arguments.conversations is not None:
        wanted = set(arguments.conversations)
        conversations = [conversation for conversation in every_conversation if conversation.id in wanted]
    elif mode is RunMode.SAMPLE:
        conversations = [smallest_conversation(every_conversation)]
    else:
        conversations = every_conversation
    count = arguments.questions if arguments.questions is not None else (SAMPLE_QUESTIONS if mode is RunMode.SAMPLE else None)
    questions = [] if arguments.build_only else select_questions(conversations, count, arguments.seed, arguments.categories)

    run_id = arguments.run_id or f"magma-{mode.value.lower()}-{'build' if arguments.build_only else 'qa'}-seed{arguments.seed}"
    output = RunOutput(arguments.results_dir / run_id)
    for log_name in ("llm_calls.jsonl", "judge_calls.jsonl"):
        (output.directory / log_name).unlink(missing_ok=True)

    llm = LLMClient.from_env("LLM", log_path=output.directory / "llm_calls.jsonl")
    judge = None if arguments.build_only else Judge(LLMClient.from_env("JUDGE", log_path=output.directory / "judge_calls.jsonl"))
    output.write_settings(run_settings(arguments, run_id, questions, llm.settings, judge.llm.settings if judge else None))

    llm_prices = Prices(*arguments.llm_prices) if arguments.llm_prices else None
    judge_prices = Prices(*arguments.judge_prices) if arguments.judge_prices else None
    if arguments.build_only and judge_prices is None:
        judge_prices = Prices(0.0, 0.0)  # no judge calls in a build-only run
    meter = CostMeter(arguments.max_cost_usd, llm_prices, judge_prices)

    embedder = InstructedEmbedder.from_config()
    tagger = DateparserTagger()
    classifier = build_default_classifier(tagger)

    def new_memory() -> MemorySystem:
        return MagmaMemory.create(llm, embedder, tagger, embedder, classifier=classifier)

    summary = run_benchmark(
        conversations, questions, new_memory, judge, meter, output, run_id, mode, arguments.build_only, arguments.pilot_questions
    )
    print_summary(summary, output.directory)
    if summary.stopped is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
