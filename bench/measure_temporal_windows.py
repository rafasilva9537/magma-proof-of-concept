"""Measure the temporal tagger on the LoCoMo questions: which questions get a time window [τs, τe].

For every question, the reference time is the date of the conversation's last
session (the questions are asked after the conversation). The report counts
the questions with a window, by LoCoMo category; the granularity of the
mentions; the candidates that the cue-word filter rejects; and the questions
where keeping only the most specific mentions changed the window. A random
sample of windows, next to the gold answer, is kept for checking by hand.

    python -m bench.measure_temporal_windows --locomo MAGMA-original-repo/data/locomo10.json

The report is written as JSON to runs/measurements/ and summarized on the screen.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from dateparser.search import search_dates

from bench.locomo import LocomoQuestion, LocomoSample, load_samples, sessions
from magma.config import DEFAULT, DatePreference
from magma.temporal import DateparserTagger, TemporalMention, TimeWindow, has_temporal_cue, query_window


@dataclass(frozen=True)
class CategoryCount:
    category: int
    questions: int
    with_window: int


@dataclass(frozen=True)
class NamedCount:
    name: str
    count: int


@dataclass(frozen=True)
class WindowExample:
    question: str
    window_start: str
    window_end: str  # exclusive
    gold_answer: str


@dataclass(frozen=True)
class QuestionResult:
    question: LocomoQuestion
    window: TimeWindow | None
    mentions: list[TemporalMention]
    rejected_candidates: list[str]
    is_narrowed_by_specificity: bool  # the window differs from the plain span of every mention
    tagger_seconds: float  # time of find + query_window only, without the extra candidate search


@dataclass(frozen=True)
class TemporalWindowReport:
    questions: int
    with_window: int
    tagger_seconds: float
    by_category: list[CategoryCount]
    mention_granularity: list[NamedCount]
    rejected_candidates: list[NamedCount]
    narrowed_by_specificity: int
    examples: list[WindowExample]


def reference_time(sample: LocomoSample, tagger: DateparserTagger) -> datetime:
    """The day of the last session, when the questions are asked."""
    last_session = sessions(sample)[-1]
    mention = tagger.resolve(last_session.date_time, datetime(2000, 1, 1), DatePreference.PAST)
    if mention is None:
        raise ValueError(f"cannot read the session date {last_session.date_time!r}")
    return mention.window.start


def measure_question(question: LocomoQuestion, reference: datetime, tagger: DateparserTagger) -> QuestionResult:
    text = question["question"]
    start = time.perf_counter()
    mentions = tagger.find(text, reference, DEFAULT.temporal.question_preference)
    window = query_window(mentions)
    elapsed = time.perf_counter() - start
    candidates = search_dates(text, languages=list(DEFAULT.temporal.languages), settings={"RELATIVE_BASE": reference})
    rejected = [candidate for candidate, _ in candidates or [] if not has_temporal_cue(candidate)]
    return QuestionResult(
        question=question,
        window=window,
        mentions=mentions,
        rejected_candidates=rejected,
        is_narrowed_by_specificity=window is not None and window != _plain_span(mentions),
        tagger_seconds=elapsed,
    )


def _plain_span(mentions: list[TemporalMention]) -> TimeWindow:
    return TimeWindow(
        start=min(mention.window.start for mention in mentions),
        end=max(mention.window.end for mention in mentions),
    )


def summarize(results: list[QuestionResult], sample_size: int, seed: int) -> TemporalWindowReport:
    totals: Counter[int] = Counter()
    windows: Counter[int] = Counter()
    granularity: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    for result in results:
        category = result.question["category"]
        totals[category] += 1
        if result.window is not None:
            windows[category] += 1
        for mention in result.mentions:
            granularity[mention.granularity.value] += 1
        for candidate in result.rejected_candidates:
            rejected[candidate.lower()] += 1
    with_window = [result for result in results if result.window is not None]
    chosen = random.Random(seed).sample(with_window, min(sample_size, len(with_window)))
    return TemporalWindowReport(
        questions=len(results),
        with_window=len(with_window),
        tagger_seconds=round(sum(result.tagger_seconds for result in results), 1),
        by_category=[CategoryCount(category, totals[category], windows[category]) for category in sorted(totals)],
        mention_granularity=[NamedCount(name, count) for name, count in granularity.most_common()],
        rejected_candidates=[NamedCount(name, count) for name, count in rejected.most_common()],
        narrowed_by_specificity=sum(1 for result in results if result.is_narrowed_by_specificity),
        examples=[_example(result) for result in chosen],
    )


def _example(result: QuestionResult) -> WindowExample:
    assert result.window is not None
    question = result.question
    gold = question.get("answer", question.get("adversarial_answer", ""))
    return WindowExample(
        question=question["question"],
        window_start=result.window.start.date().isoformat(),
        window_end=result.window.end.date().isoformat(),
        gold_answer=str(gold),
    )


def print_summary(report: TemporalWindowReport) -> None:
    print(f"{report.with_window} of {report.questions} questions get a time window (tagger: {report.tagger_seconds} s)")
    print("by category: " + ", ".join(f"{c.category}: {c.with_window}/{c.questions}" for c in report.by_category))
    print("mention granularity: " + ", ".join(f"{g.name} {g.count}" for g in report.mention_granularity))
    print("rejected by the cue filter: " + ", ".join(f"{r.name!r} {r.count}" for r in report.rejected_candidates))
    print(f"windows narrowed by keeping the most specific mentions: {report.narrowed_by_specificity}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locomo", type=Path, required=True, help="path to locomo10.json")
    parser.add_argument("--sample-size", type=int, default=25, help="windows kept for checking by hand")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("runs/measurements/temporal_windows.json"))
    arguments = parser.parse_args()

    tagger = DateparserTagger()
    results: list[QuestionResult] = []
    for sample in load_samples(arguments.locomo):
        reference = reference_time(sample, tagger)
        for question in sample["qa"]:
            results.append(measure_question(question, reference, tagger))
    report = summarize(results, arguments.sample_size, arguments.seed)

    print_summary(report)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written to {arguments.output}")


if __name__ == "__main__":
    main()
