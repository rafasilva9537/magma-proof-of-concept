"""Calibrate θ_sim, the cosine threshold of SIMILAR_TO edges, for the configured embedding model.

Cosine thresholds do not transfer between embedding models, so θ_sim is measured
on real conversations: the distribution of cosines between turns of the same
conversation, and, simulating the fast path, how many SIMILAR_TO edges each new
event would get under each candidate threshold.

    python -m bench.calibrate_semantic_threshold --locomo MAGMA-original-repo/data/locomo10.json

The report is written as JSON to runs/calibration/ and summarized on the screen.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bench.locomo import find_sample, load_samples, sessions, turn_text
from magma.config import DEFAULT
from magma.embed import Embedder, InstructedEmbedder, Matrix

PERCENTILES = (50.0, 75.0, 90.0, 95.0, 99.0, 99.9)
DEFAULT_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
EXAMPLE_BAND_WIDTH = 0.05
EXAMPLES_PER_BAND = 4
MIN_TURN_DISTANCE_IN_EXAMPLES = 4  # neighbouring turns are similar for trivial reasons; skip them


@dataclass(frozen=True)
class Percentile:
    percentile: float
    cosine: float


@dataclass(frozen=True)
class ThresholdEffect:
    threshold: float
    mean_new_edges: float  # SIMILAR_TO edges a new event gets, on average
    share_with_edge: float  # events that get at least one
    share_at_limit: float  # events that hit the neighbour limit


@dataclass(frozen=True)
class ExamplePair:
    cosine: float
    first: str
    second: str


@dataclass(frozen=True)
class ExampleBand:
    band_start: float  # pairs with band_start < cosine < band_start + EXAMPLE_BAND_WIDTH
    pairs: list[ExamplePair]


@dataclass(frozen=True)
class VariantReport:
    conversation: str
    with_speaker: bool
    turns: int
    percentiles: list[Percentile]
    effects: list[ThresholdEffect]
    examples: list[ExampleBand]


@dataclass(frozen=True)
class CalibrationReport:
    embedding_model: str
    max_neighbors: int
    seed: int
    variants: list[VariantReport]


def load_turn_texts(locomo_path: Path, sample_id: str, with_speaker: bool) -> list[str]:
    """Turns of one conversation in order, as the fast path stores them: "Speaker: [Image: caption] text"."""
    sample = find_sample(load_samples(locomo_path), sample_id)
    return [turn_text(turn, with_speaker) for session in sessions(sample) for turn in session.turns]


def pairwise_percentiles(cosines: Matrix) -> list[Percentile]:
    """Percentiles of the cosines between distinct turns (upper triangle of the matrix)."""
    upper = cosines[np.triu_indices(len(cosines), k=1)]
    return [Percentile(percentile, round(float(np.percentile(upper, percentile)), 3)) for percentile in PERCENTILES]


def threshold_effect(cosines: Matrix, threshold: float, max_neighbors: int) -> ThresholdEffect:
    """What the fast path would do: each turn looks at its `max_neighbors` closest earlier turns."""
    new_edges: list[int] = []
    for turn in range(1, len(cosines)):
        closest_earlier = np.sort(cosines[turn, :turn])[::-1][:max_neighbors]
        new_edges.append(int((closest_earlier > threshold).sum()))
    counts = np.array(new_edges)
    return ThresholdEffect(
        threshold=threshold,
        mean_new_edges=round(float(counts.mean()), 2),
        share_with_edge=round(float((counts > 0).mean()), 3),
        share_at_limit=round(float((counts == max_neighbors).mean()), 3),
    )


def example_band(cosines: Matrix, texts: list[str], band_start: float, rng: random.Random) -> ExampleBand:
    """A few random pairs of the band, to read by hand whether they share a topic or only a style."""
    band = np.argwhere((cosines > band_start) & (cosines < band_start + EXAMPLE_BAND_WIDTH))
    candidates = [(int(a), int(b)) for a, b in band if int(b) - int(a) >= MIN_TURN_DISTANCE_IN_EXAMPLES]
    chosen = rng.sample(candidates, min(EXAMPLES_PER_BAND, len(candidates)))
    pairs = [ExamplePair(cosine=round(float(cosines[a, b]), 3), first=texts[a], second=texts[b]) for a, b in chosen]
    return ExampleBand(band_start=band_start, pairs=pairs)


def measure(
    embedder: Embedder,
    texts: list[str],
    conversation: str,
    with_speaker: bool,
    thresholds: list[float],
    seed: int,
) -> VariantReport:
    vectors = embedder.embed_documents(texts)
    cosines: Matrix = vectors @ vectors.T  # vectors are normalized, so the inner product is the cosine
    max_neighbors = DEFAULT.ingest.semantic_edge_max_neighbors
    rng = random.Random(seed)
    return VariantReport(
        conversation=conversation,
        with_speaker=with_speaker,
        turns=len(texts),
        percentiles=pairwise_percentiles(cosines),
        effects=[threshold_effect(cosines, threshold, max_neighbors) for threshold in thresholds],
        examples=[example_band(cosines, texts, start, rng) for start in thresholds],
    )


def print_summary(report: VariantReport) -> None:
    variant = "Speaker: text" if report.with_speaker else "text only"
    print(f"\n{report.conversation} ({report.turns} turns, {variant})")
    print("  " + "  ".join(f"p{item.percentile:g} {item.cosine:.3f}" for item in report.percentiles))
    print("  threshold  new edges/event  with edge  at limit")
    for effect in report.effects:
        print(
            f"  {effect.threshold:9.2f}  {effect.mean_new_edges:15.2f}"
            f"  {effect.share_with_edge:9.1%}  {effect.share_at_limit:8.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locomo", type=Path, required=True, help="path to locomo10.json")
    parser.add_argument("--conversations", nargs="+", default=["conv-26", "conv-30"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--seed", type=int, default=7, help="seed for the example pairs")
    parser.add_argument("--output", type=Path, default=Path("runs/calibration/semantic_threshold.json"))
    arguments = parser.parse_args()

    embedder = InstructedEmbedder.from_config()
    reports: list[VariantReport] = []
    for conversation in arguments.conversations:
        for with_speaker in (True, False):
            texts = load_turn_texts(arguments.locomo, conversation, with_speaker)
            report = measure(embedder, texts, conversation, with_speaker, arguments.thresholds, arguments.seed)
            print_summary(report)
            reports.append(report)

    calibration = CalibrationReport(
        embedding_model=DEFAULT.embedding.model_name,
        max_neighbors=DEFAULT.ingest.semantic_edge_max_neighbors,
        seed=arguments.seed,
        variants=reports,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(asdict(calibration), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to {arguments.output}")


if __name__ == "__main__":
    main()
