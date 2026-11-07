import dataclasses

import pytest

from magma.config import ABLATION_LAYERS, DEFAULT, INTENTS, LAYERS, Ablation, Config, Intent, Layer


def test_table6_values() -> None:
    assert DEFAULT.retrieval.rrf_k == 60
    assert DEFAULT.retrieval.top_k == 20
    assert DEFAULT.traversal.max_depth == 5
    assert DEFAULT.traversal.max_nodes == 200
    assert DEFAULT.traversal.drop_threshold == 0.15
    assert DEFAULT.traversal.lambda1 == 1.0
    assert 0.3 <= DEFAULT.traversal.lambda2 <= 0.7
    assert DEFAULT.llm.temperature == 0.0


def test_intent_weights_inside_table6_ranges() -> None:
    ranges: dict[Layer, tuple[float, float]] = {
        "ENT": (2.5, 6.0),
        "TEMP": (0.5, 4.0),
        "CAUSAL": (3.0, 5.0),
        "SEM": (2.5, 5.0),
    }
    for intent in INTENTS:
        row = DEFAULT.traversal.intent_weights.of(intent)
        for layer in LAYERS:
            low, high = ranges[layer]
            assert low <= row.of(layer) <= high, (intent, layer)
    assert 2.0 <= DEFAULT.retrieval.channel_weights.keyword <= 5.0


def strongest_layer(intent: Intent) -> Layer:
    weights = DEFAULT.layer_weights(intent)
    return max(LAYERS, key=weights.of)


def test_each_intent_favours_its_layer() -> None:
    assert strongest_layer("WHY") == "CAUSAL"
    assert strongest_layer("WHEN") == "TEMP"
    assert strongest_layer("ENTITY") == "ENT"


def test_config_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT.retrieval.top_k = 5  # type: ignore[misc]  # the point of the test
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT.traversal.intent_weights.why.causal = 0.0  # type: ignore[misc]  # the point of the test


def test_unknown_ablation_rejected() -> None:
    with pytest.raises(ValueError):
        Config(ablation="no_such_thing")  # type: ignore[arg-type]  # the point of the test


LAYER_ABLATIONS: list[Ablation] = [a for a in ABLATION_LAYERS if a != "no_adaptive_policy"]


@pytest.mark.parametrize("ablation", LAYER_ABLATIONS)
def test_layer_ablations_zero_removed_layers(ablation: Ablation) -> None:
    cfg = DEFAULT.with_ablation(ablation)
    for intent in INTENTS:
        weights = cfg.layer_weights(intent)
        for layer in LAYERS:
            if layer in cfg.active_layers:
                assert weights.of(layer) > 0
            else:
                assert weights.of(layer) == 0


def test_no_adaptive_policy_uses_same_weights_for_every_intent() -> None:
    cfg = DEFAULT.with_ablation("no_adaptive_policy")
    assert not cfg.adaptive
    assert cfg.layer_weights("WHY") == cfg.layer_weights("WHEN") == cfg.layer_weights("ENTITY")
    assert cfg.active_layers == LAYERS
    assert cfg.layer_weights("WHY").causal == pytest.approx((5.0 + 3.0 + 3.0) / 3)


def test_with_ablation_keeps_default_untouched() -> None:
    DEFAULT.with_ablation("only_causal")
    assert DEFAULT.ablation is None
    assert DEFAULT.active_layers == LAYERS
