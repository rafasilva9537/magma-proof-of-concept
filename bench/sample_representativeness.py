"""How a question sample of a run spreads over the LoCoMo dataset, against the whole dataset.

The runner samples questions stratified by category only. This compares the
sample with all the questions on three axes:
- category (the strata, so the shares should match up to rounding);
- conversation (not a stratum: the shares vary with the draw);
- position inside each conversation's question list, in quartiles: whether the
  sample covers the start, middle and end of the conversations, not one stretch.

For each axis: the count and share of every value in the dataset and in the
sample, and the largest gap between the two shares, in percentage points.

    python -m bench.sample_representativeness --locomo MAGMA-original-repo/data/locomo10.json \\
        --run results/magma-report-qa500-seed7/run.json

The report is written as JSON to runs/measurements/sample_representativeness.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from bench.benchmark import Conversation
from bench.locomo import load_samples, to_conversation

QUARTILES = 4


@dataclass(frozen=True)
class Share:
    value: str
    dataset: int
    sample: int
    dataset_share: float  # percent
    sample_share: float  # percent


@dataclass(frozen=True)
class Axis:
    name: str
    shares: list[Share]
    largest_gap_points: float  # max |sample share − dataset share|, in percentage points


@dataclass(frozen=True)
class Representativeness:
    dataset_questions: int
    sample_questions: int
    unknown_ids: list[str]  # ids of the run that are not in the dataset
    axes: list[Axis]


def quartile(index: int, count: int) -> int:
    """Quartile (0–3) of the index-th of `count` questions, by relative position in the list."""
    if count <= 1:
        return 0
    return min(int(index / (count - 1) * QUARTILES), QUARTILES - 1)


def compare(name: str, dataset_values: Sequence[str], sample_values: Sequence[str]) -> Axis:
    dataset_counts = Counter(dataset_values)
    sample_counts = Counter(sample_values)
    shares: list[Share] = []
    for value in sorted(dataset_counts):
        dataset_share = 100 * dataset_counts[value] / len(dataset_values)
        sample_share = 100 * sample_counts[value] / len(sample_values) if sample_values else 0.0
        shares.append(Share(value, dataset_counts[value], sample_counts[value], round(dataset_share, 1), round(sample_share, 1)))
    gap = max((abs(share.sample_share - share.dataset_share) for share in shares), default=0.0)
    return Axis(name, shares, round(gap, 1))


@dataclass(frozen=True)
class QuestionPlace:
    """Where a question sits on each axis."""

    category: str
    conversation: str
    position_quartile: str  # "Q1" to "Q4"


def representativeness(conversations: Sequence[Conversation], sample_ids: set[str]) -> Representativeness:
    dataset: list[QuestionPlace] = []
    sample: list[QuestionPlace] = []
    for conversation in conversations:
        count = len(conversation.questions)
        for index, question in enumerate(conversation.questions):
            place = QuestionPlace(question.category, conversation.id, f"Q{quartile(index, count) + 1}")
            dataset.append(place)
            if question.id in sample_ids:
                sample.append(place)
    known = {question.id for conversation in conversations for question in conversation.questions}
    return Representativeness(
        dataset_questions=len(dataset),
        sample_questions=len(sample),
        unknown_ids=sorted(sample_ids - known),
        axes=[
            compare("category", [place.category for place in dataset], [place.category for place in sample]),
            compare("conversation", [place.conversation for place in dataset], [place.conversation for place in sample]),
            compare(
                "position_quartile",
                [place.position_quartile for place in dataset],
                [place.position_quartile for place in sample],
            ),
        ],
    )


def print_report(report: Representativeness) -> None:
    print(f"dataset {report.dataset_questions} questions, sample {report.sample_questions}; unknown ids: {len(report.unknown_ids)}")
    for axis in report.axes:
        print(f"\n{axis.name} (largest gap {axis.largest_gap_points} points)")
        for share in axis.shares:
            print(f"  {share.value:<14} {share.dataset:>5} ({share.dataset_share:>4}%)  {share.sample:>4} ({share.sample_share:>4}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locomo", type=Path, required=True, help="path to locomo10.json")
    parser.add_argument("--run", type=Path, default=Path("results/magma-report-qa500-seed7/run.json"), help="run.json with question_ids")
    parser.add_argument("--out", type=Path, default=Path("runs/measurements/sample_representativeness.json"))
    arguments = parser.parse_args()

    conversations = [to_conversation(sample) for sample in load_samples(arguments.locomo)]
    settings = json.loads(arguments.run.read_text(encoding="utf-8"))
    sample_ids = {str(question_id) for question_id in settings["question_ids"]}
    report = representativeness(conversations, sample_ids)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    print(f"\nwritten {arguments.out}")


if __name__ == "__main__":
    main()
