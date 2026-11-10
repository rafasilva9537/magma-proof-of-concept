import math

import pytest

from bench.lexical_metrics import bleu1, normalized_tokens, token_f1


def test_normalization_drops_case_punctuation_and_articles() -> None:
    assert normalized_tokens("The Sunrise, at a lake!") == ["sunrise", "at", "lake"]


def test_token_f1_by_hand() -> None:
    # prediction: painted, sunrise, 2022 (3 tokens); gold: sunrise, in, 2022 (3 tokens); shared: sunrise, 2022
    # precision = recall = 2/3, so F1 = 2/3
    assert token_f1("Painted a sunrise 2022", "A sunrise in 2022") == pytest.approx(2 / 3)
    assert token_f1("Paris", "paris.") == 1.0
    assert token_f1("London", "Paris") == 0.0


def test_bleu1_penalizes_short_answers() -> None:
    # prediction: sunrise (1 token, matches); gold: sunrise, lake (2 tokens)
    # precision 1, brevity penalty exp(1 - 2/1) = e^-1
    assert bleu1("sunrise", "sunrise lake") == pytest.approx(math.exp(-1))
    # repeated tokens are clipped to their count in the gold: 1 of 2 counts
    assert bleu1("lake lake", "lake view") == pytest.approx(0.5)


def test_empty_texts() -> None:
    assert token_f1("", "") == 1.0
    assert token_f1("the", "Paris") == 0.0  # only an article: nothing left
    assert bleu1("", "Paris") == 0.0
