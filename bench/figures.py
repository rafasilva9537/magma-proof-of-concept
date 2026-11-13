"""Figures of the results, for the slides, the poster and the article.

Each figure is drawn from the summary.json of finished runs, never typed by hand:
- locomo_by_category: judge score of MAGMA (A) and Full Context (C1) per LoCoMo
  category and overall;
- locomo_tokens: tokens of the memory LLM per question, A against C1;
- locomo_ablations: judge score of each finished ablation of A against the full A.

Every figure is written as SVG and PDF (vector: slides, poster, article) and as
PNG at 300 dpi. Text is in Portuguese, with a decimal comma. The figure number,
title and source line go in the document, not inside the image (ABNT NBR 14724).
figures.json lists, for each figure, the runs its numbers come from.

    python -m bench.figures

The figures are written to runs/figures/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict, cast

import matplotlib

matplotlib.use("Agg")  # files only: no window, works without a display

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from magma.config import Ablation  # noqa: E402

FORMATS = ("svg", "pdf", "png")
PNG_DPI = 300

# One color per system in every figure (validated for color-vision deficiency as an adjacent set).
SYSTEM_COLORS = {
    "MAGMA": "#2a78d6",
    "NARRATIVE": "#1baf7a",
    "FULL_CONTEXT": "#eb6834",
    "FULL_CONTEXT_VISIBLE": "#6250d6",
}
SYSTEM_LABELS = {
    "MAGMA": "MAGMA (A)",
    "NARRATIVE": "Narrativa (B)",
    "FULL_CONTEXT": "Contexto completo (C1)",
    "FULL_CONTEXT_VISIBLE": "Contexto completo visível",
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

CATEGORY_LABELS = {
    "SINGLE_HOP": "Um salto",
    "MULTI_HOP": "Vários saltos",
    "TEMPORAL": "Temporal",
    "OPEN_DOMAIN": "Domínio aberto",
    "ADVERSARIAL": "Adversarial",
}
ABLATION_LABELS = {
    Ablation.NO_ADAPTIVE_POLICY: "Sem política adaptativa",
    Ablation.NO_CAUSAL: "Sem grafo causal",
    Ablation.NO_TEMPORAL: "Sem grafo temporal",
    Ablation.NO_ENTITY: "Sem grafo de entidades",
    Ablation.ONLY_CAUSAL: "Só o grafo causal",
    Ablation.ONLY_TEMPORAL: "Só o grafo temporal",
    Ablation.ONLY_ENTITY: "Só o grafo de entidades",
}


class CategoryRecord(TypedDict):
    """One category of summary.json, as the runner writes it."""

    category: str
    questions: int
    judge: float | None


class SummaryRecord(TypedDict):
    """The fields of summary.json that the figures read."""

    run_id: str
    overall: CategoryRecord
    by_category: list[CategoryRecord]
    tokens_per_query: float | None


@dataclass(frozen=True)
class CategoryJudge:
    category: str
    questions: int
    judge: float


@dataclass(frozen=True)
class RunScores:
    run_id: str
    overall: float
    questions: int
    by_category: tuple[CategoryJudge, ...]
    tokens_per_query: float


@dataclass(frozen=True)
class AblationScore:
    ablation: Ablation
    run_id: str
    judge: float
    delta: float  # against the full system


@dataclass(frozen=True)
class FigureSource:
    figure: str
    runs: list[str]


# ---------------------------------------------------------------- reading the runs


def load_scores(run_directory: Path) -> RunScores:
    """The judge scores and tokens of a finished run, from its summary.json."""
    summary = cast(SummaryRecord, json.loads((run_directory / "summary.json").read_text(encoding="utf-8")))
    overall = summary["overall"]
    tokens = summary["tokens_per_query"]
    if overall["judge"] is None or tokens is None:
        raise ValueError(f"{run_directory} has no judged answers")
    categories = tuple(
        CategoryJudge(record["category"], record["questions"], record["judge"])
        for record in summary["by_category"]
        if record["judge"] is not None
    )
    return RunScores(summary["run_id"], overall["judge"], overall["questions"], categories, tokens)


def run_ablation(run_directory: Path) -> Ablation | None:
    """The ablation a run was made with, from its run.json; None for the full system."""
    settings = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    value = settings.get("ablation")
    return None if value is None else Ablation(str(value))


def ablation_scores(full: RunScores, run_directories: list[Path]) -> list[AblationScore]:
    """The finished ablation runs, best first, each with its difference to the full system."""
    scores = []
    for directory in run_directories:
        ablation = run_ablation(directory)
        if ablation is None or not (directory / "summary.json").exists():
            continue
        run = load_scores(directory)
        scores.append(AblationScore(ablation, run.run_id, run.overall, run.overall - full.overall))
    return sorted(scores, key=lambda score: score.judge, reverse=True)


# ---------------------------------------------------------------- formatting


def decimal_pt(value: float, places: int = 2) -> str:
    """A number with the Portuguese decimal comma and thousands dot: 0.738 -> '0,74'; 31630 -> '31.630'."""
    text = f"{value:,.{places}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def signed_pt(value: float) -> str:
    """A difference with its sign, in Portuguese: -0.0234 -> '−0,02'."""
    sign = "+" if value >= 0 else "−"
    return sign + decimal_pt(abs(value))


def category_order(run: RunScores) -> list[str]:
    """The categories in the order of CATEGORY_LABELS, keeping only those the run has."""
    present = {score.category for score in run.by_category}
    return [category for category in CATEGORY_LABELS if category in present]


# ---------------------------------------------------------------- drawing


def new_figure(width: float, height: float) -> tuple[Figure, Axes]:
    figure, axes = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(colors=TEXT_SECONDARY, length=0)
    return figure, axes


def save(figure: Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        figure.savefig(directory / f"{name}.{extension}", dpi=PNG_DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def draw_by_category(magma: RunScores, full_context: RunScores) -> Figure:
    """Grouped columns: one group per category plus the overall, A beside C1."""
    categories = category_order(magma)
    magma_scores = {score.category: score for score in magma.by_category}
    full_scores = {score.category: score.judge for score in full_context.by_category}
    labels = [f"{CATEGORY_LABELS[c]}\n(n = {magma_scores[c].questions})" for c in categories]
    labels.append(f"Geral\n(n = {magma.questions})")
    magma_values = [magma_scores[c].judge for c in categories] + [magma.overall]
    full_values = [full_scores[c] for c in categories] + [full_context.overall]

    figure, axes = new_figure(8.0, 4.2)
    width = 0.36
    gap = 0.02  # the surface gap between the two touching columns of a group
    positions = list(range(len(labels)))
    for offset, values, system in ((-1, magma_values, "MAGMA"), (1, full_values, "FULL_CONTEXT")):
        x_values = [p + offset * (width + gap) / 2 for p in positions]
        bars = axes.bar(x_values, values, width=width, color=SYSTEM_COLORS[system], label=SYSTEM_LABELS[system])
        axes.bar_label(bars, labels=[decimal_pt(v) for v in values], padding=2, fontsize=8, color=TEXT_PRIMARY)
    axes.set_xticks(positions, labels, fontsize=8.5, color=TEXT_PRIMARY)
    axes.set_ylim(0, 1.0)
    axes.set_ylabel("Nota do juiz (média, 0 a 1)", color=TEXT_SECONDARY, fontsize=9)
    axes.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0,25", "0,50", "0,75", "1"])
    axes.legend(frameon=False, loc="upper left", ncols=2, fontsize=9, bbox_to_anchor=(0, 1.12))
    return figure


def draw_tokens(magma: RunScores, full_context: RunScores) -> Figure:
    """Horizontal bars: tokens of the memory LLM per question."""
    systems = ["MAGMA", "FULL_CONTEXT"]
    values = [magma.tokens_per_query, full_context.tokens_per_query]
    figure, axes = new_figure(8.0, 2.0)
    positions = [1, 0]  # A on top
    bars = axes.barh(positions, values, height=0.5, color=[SYSTEM_COLORS[s] for s in systems])
    axes.bar_label(bars, labels=[decimal_pt(v, 0) for v in values], padding=4, fontsize=9, color=TEXT_PRIMARY)
    axes.set_yticks(positions, [SYSTEM_LABELS[s] for s in systems], fontsize=9.5, color=TEXT_PRIMARY)
    axes.set_xlabel("Tokens por pergunta (entrada + saída do LLM)", color=TEXT_SECONDARY, fontsize=9)
    axes.xaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.set_xlim(0, max(values) * 1.15)
    ticks = axes.get_xticks()
    axes.set_xticks(ticks, [decimal_pt(t, 0) for t in ticks])
    axes.set_xlim(0, max(values) * 1.15)
    ratio = full_context.tokens_per_query / magma.tokens_per_query
    axes.set_title(f"A usa {decimal_pt(ratio, 1)} vezes menos tokens por pergunta", loc="left", fontsize=10, color=TEXT_PRIMARY)
    return figure


def draw_ablations(full: RunScores, ablations: list[AblationScore]) -> Figure:
    """Horizontal bars of each ablation's judge score, with the full system as a reference line."""
    labels = [ABLATION_LABELS[score.ablation] for score in ablations]
    values = [score.judge for score in ablations]
    figure, axes = new_figure(8.0, 0.5 + 0.45 * (len(ablations) + 1))
    positions = list(range(len(ablations), 0, -1))
    bars = axes.barh(positions, values, height=0.55, color=SYSTEM_COLORS["MAGMA"], alpha=0.55)
    value_labels = [f"{decimal_pt(s.judge)} ({signed_pt(s.delta)})" for s in ablations]
    axes.bar_label(bars, labels=value_labels, padding=4, fontsize=8.5, color=TEXT_PRIMARY)
    full_bar = axes.barh([0], [full.overall], height=0.55, color=SYSTEM_COLORS["MAGMA"])
    axes.bar_label(full_bar, labels=[decimal_pt(full.overall)], padding=4, fontsize=8.5, color=TEXT_PRIMARY)
    axes.axvline(full.overall, color=TEXT_SECONDARY, linewidth=0.8)
    axes.set_yticks([*positions, 0], [*labels, "MAGMA completo (A)"], fontsize=9, color=TEXT_PRIMARY)
    axes.set_xlim(0, 1.0)
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0,25", "0,50", "0,75", "1"])
    axes.set_xlabel("Nota do juiz (média, 0 a 1); entre parênteses, a diferença para o A completo", color=TEXT_SECONDARY, fontsize=9)
    axes.xaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    return figure


# ---------------------------------------------------------------- entry point


def write_figures(magma_run: Path, full_context_run: Path, ablation_runs: list[Path], output: Path) -> list[FigureSource]:
    """Draw every figure whose runs are finished; return where each figure's numbers come from."""
    magma = load_scores(magma_run)
    full_context = load_scores(full_context_run)
    save(draw_by_category(magma, full_context), output, "locomo_by_category")
    save(draw_tokens(magma, full_context), output, "locomo_tokens")
    sources = [
        FigureSource("locomo_by_category", [magma.run_id, full_context.run_id]),
        FigureSource("locomo_tokens", [magma.run_id, full_context.run_id]),
    ]
    ablations = ablation_scores(magma, ablation_runs)
    if ablations:
        save(draw_ablations(magma, ablations), output, "locomo_ablations")
        sources.append(FigureSource("locomo_ablations", [magma.run_id, *(score.run_id for score in ablations)]))
    sources_text = json.dumps([asdict(source) for source in sources], ensure_ascii=False, indent=2)
    (output / "figures.json").write_text(sources_text, encoding="utf-8")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"), help="directory with one folder per run")
    parser.add_argument("--magma", default="magma-report-qa500-seed7", help="run id of the full MAGMA run")
    parser.add_argument("--full-context", default="full-context-report-qa500-seed7", help="run id of the C1 run")
    parser.add_argument("--ablations", default="magma-*-report-qa500-seed7", help="glob of the ablation run ids")
    parser.add_argument("--out", type=Path, default=Path("runs/figures"))
    arguments = parser.parse_args()

    ablation_runs = sorted(arguments.results.glob(arguments.ablations))
    sources = write_figures(arguments.results / arguments.magma, arguments.results / arguments.full_context, ablation_runs, arguments.out)
    for source in sources:
        print(f"{source.figure}: {', '.join(source.runs)}")
    print(f"\nwritten to {arguments.out}")


if __name__ == "__main__":
    main()
