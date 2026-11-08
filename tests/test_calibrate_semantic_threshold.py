import json
import random
from pathlib import Path

import numpy as np
import pytest

from bench.calibrate_semantic_threshold import (
    Percentile,
    example_band,
    load_turn_texts,
    pairwise_percentiles,
    threshold_effect,
)
from magma.embed import Matrix


def cosines(values: list[list[float]]) -> Matrix:
    return np.array(values, dtype=np.float32)


# Four turns. Earlier neighbours of each turn: t1 -> t0 (0.9); t2 -> t0 (0.2), t1 (0.7); t3 -> t0, t1, t2 (0.1, 0.8, 0.6).
FOUR_TURNS = cosines(
    [
        [1.0, 0.9, 0.2, 0.1],
        [0.9, 1.0, 0.7, 0.8],
        [0.2, 0.7, 1.0, 0.6],
        [0.1, 0.8, 0.6, 1.0],
    ]
)


def test_threshold_effect_counts_edges_to_earlier_turns() -> None:
    # Above 0.65: t1 -> 1 edge (0.9), t2 -> 1 (0.7), t3 -> 1 (0.8). Mean 1.0, every turn linked.
    effect = threshold_effect(FOUR_TURNS, threshold=0.65, max_neighbors=5)
    assert effect.mean_new_edges == 1.0
    assert effect.share_with_edge == 1.0
    assert effect.share_at_limit == 0.0


def test_threshold_effect_respects_the_neighbour_limit() -> None:
    # Above 0.0 with at most 2 neighbours: t1 -> 1, t2 -> 2, t3 -> 2. Mean 5/3; t2 and t3 hit the limit.
    effect = threshold_effect(FOUR_TURNS, threshold=0.0, max_neighbors=2)
    assert effect.mean_new_edges == pytest.approx(5 / 3, abs=0.01)
    assert effect.share_at_limit == pytest.approx(2 / 3, abs=0.001)


def test_percentiles_ignore_the_diagonal() -> None:
    # Distinct pairs: 0.9, 0.2, 0.1, 0.7, 0.8, 0.6 -> median 0.65; the 1.0 of the diagonal does not count.
    by_percentile = {item.percentile: item.cosine for item in pairwise_percentiles(FOUR_TURNS)}
    assert by_percentile[50.0] == pytest.approx(0.65, abs=1e-3)
    assert all(isinstance(item, Percentile) for item in pairwise_percentiles(FOUR_TURNS))


def test_example_band_skips_neighbouring_turns() -> None:
    texts = ["t0", "t1", "t2", "t3"]
    band = example_band(FOUR_TURNS, texts, band_start=0.85, rng=random.Random(7))
    assert band.pairs == []  # t0-t1 is in (0.85, 0.90), but turns less than 4 apart are skipped


def test_turns_are_read_in_session_order_with_captions(tmp_path: Path) -> None:
    locomo = [
        {
            "sample_id": "conv-x",
            "qa": [],
            "conversation": {
                "speaker_a": "Caroline",
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_2_date_time": "1:00 pm on 9 May, 2023",
                "session_2": [{"speaker": "Melanie", "dia_id": "D2:1", "text": "Back home."}],
                "session_1": [
                    {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hi!"},
                    {"speaker": "Melanie", "dia_id": "D1:2", "text": "Look.", "blip_caption": "a photo of a dog"},
                ],
            },
        }
    ]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(locomo), encoding="utf-8")
    assert load_turn_texts(path, "conv-x", with_speaker=True) == [
        "Caroline: Hi!",
        "Melanie: [Image: a photo of a dog] Look.",
        "Melanie: Back home.",
    ]
    assert load_turn_texts(path, "conv-x", with_speaker=False)[0] == "Hi!"
    with pytest.raises(KeyError):
        load_turn_texts(path, "conv-missing", with_speaker=True)
