import json
from pathlib import Path

import pytest

from bench.figures import (
    ablation_scores,
    decimal_pt,
    load_scores,
    signed_pt,
    write_figures,
)
from magma.config import Ablation


def write_run(directory: Path, run_id: str, overall: float, tokens: float, ablation: str | None = None) -> Path:
    run = directory / run_id
    run.mkdir(parents=True)
    categories = [
        {"category": "SINGLE_HOP", "questions": 3, "judged": 3, "judge": overall + 0.1, "f1": 0.5, "bleu1": 0.4},
        {"category": "ADVERSARIAL", "questions": 1, "judged": 1, "judge": overall - 0.2, "f1": 0.0, "bleu1": 0.0},
    ]
    summary = {
        "run_id": run_id,
        "overall": {"category": "OVERALL", "questions": 4, "judged": 4, "judge": overall, "f1": 0.4, "bleu1": 0.3},
        "by_category": categories,
        "tokens_per_query": tokens,
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    settings = {"run_id": run_id} if ablation is None else {"run_id": run_id, "ablation": ablation}
    (run / "run.json").write_text(json.dumps(settings), encoding="utf-8")
    return run


def test_decimal_pt_uses_the_comma_and_the_thousands_dot() -> None:
    assert decimal_pt(0.738) == "0,74"
    assert decimal_pt(31629.9, 0) == "31.630"
    assert decimal_pt(13.51, 1) == "13,5"
    assert signed_pt(-0.0234) == "−0,02"
    assert signed_pt(0.01) == "+0,01"


def test_load_scores_reads_the_judged_categories(tmp_path: Path) -> None:
    run = load_scores(write_run(tmp_path, "magma", 0.7, 2341.0))
    assert run.overall == 0.7
    assert run.questions == 4
    assert [score.category for score in run.by_category] == ["SINGLE_HOP", "ADVERSARIAL"]
    assert run.by_category[0].judge == pytest.approx(0.8)


def test_ablation_scores_skip_the_full_and_unfinished_runs_and_sort_best_first(tmp_path: Path) -> None:
    full = load_scores(write_run(tmp_path, "magma", 0.74, 2341.0))
    worse = write_run(tmp_path, "magma-no-causal", 0.70, 2000.0, "NO_CAUSAL")
    better = write_run(tmp_path, "magma-no-temporal", 0.76, 2000.0, "NO_TEMPORAL")
    unfinished = tmp_path / "magma-no-entity"
    unfinished.mkdir()
    (unfinished / "run.json").write_text(json.dumps({"ablation": "NO_ENTITY"}), encoding="utf-8")

    scores = ablation_scores(full, [tmp_path / "magma", worse, better, unfinished])

    assert [score.ablation for score in scores] == [Ablation.NO_TEMPORAL, Ablation.NO_CAUSAL]
    assert scores[1].delta == pytest.approx(0.70 - 0.74)


def test_write_figures_writes_every_format_and_the_sources(tmp_path: Path) -> None:
    magma = write_run(tmp_path, "magma", 0.74, 2341.0)
    full_context = write_run(tmp_path, "full-context", 0.81, 31630.0)
    ablation = write_run(tmp_path, "magma-no-causal", 0.70, 2000.0, "NO_CAUSAL")
    output = tmp_path / "figures"

    sources = write_figures(magma, full_context, [ablation], output)

    names = ["locomo_by_category", "locomo_tokens", "locomo_ablations"]
    assert [source.figure for source in sources] == names
    for name in names:
        for extension in ("svg", "pdf", "png"):
            assert (output / f"{name}.{extension}").stat().st_size > 0
    recorded = json.loads((output / "figures.json").read_text(encoding="utf-8"))
    assert recorded[2]["runs"] == ["magma", "magma-no-causal"]


def test_write_figures_without_ablations_skips_that_figure(tmp_path: Path) -> None:
    magma = write_run(tmp_path, "magma", 0.74, 2341.0)
    full_context = write_run(tmp_path, "full-context", 0.81, 31630.0)

    sources = write_figures(magma, full_context, [], tmp_path / "figures")

    assert [source.figure for source in sources] == ["locomo_by_category", "locomo_tokens"]
