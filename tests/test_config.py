import dataclasses

import pytest

from magma.config import DEFAULT, Ablation, Intent, Layer


def test_table6_values() -> None:
    assert DEFAULT.retrieval.rrf_k == 60
    assert DEFAULT.retrieval.top_k == 20
    assert DEFAULT.traversal.max_depth == 5
    assert DEFAULT.traversal.max_nodes == 200
    assert DEFAULT.traversal.drop_threshold == 0.15
    assert DEFAULT.traversal.lambda1 == 1.0
    assert 0.3 <= DEFAULT.traversal.lambda2 <= 0.7
    assert DEFAULT.llm.temperature == 0.0


def test_causal_prompt_shows_a_limited_number_of_candidates() -> None:
    # The builds of the benchmark depend on it: without a limit the prompt grows with the conversation.
    assert DEFAULT.ingest.neighborhood_hops == 2
    assert DEFAULT.ingest.causal_candidate_limit == 50


def test_intent_weights_inside_table6_ranges() -> None:
    ranges = {
        Layer.ENTITY: (2.5, 6.0),
        Layer.TEMPORAL: (0.5, 4.0),
        Layer.CAUSAL: (3.0, 5.0),
        Layer.SEMANTIC: (2.5, 5.0),
    }
    for intent in Intent:
        row = DEFAULT.traversal.intent_weights.of(intent)
        for layer in Layer:
            low, high = ranges[layer]
            assert low <= row.of(layer) <= high, (intent, layer)
    assert 2.0 <= DEFAULT.retrieval.channel_weights.keyword <= 5.0


def strongest_layer(intent: Intent) -> Layer:
    weights = DEFAULT.layer_weights(intent)
    return max(Layer, key=weights.of)


def test_each_intent_favours_its_layer() -> None:
    assert strongest_layer(Intent.WHY) is Layer.CAUSAL
    assert strongest_layer(Intent.WHEN) is Layer.TEMPORAL
    assert strongest_layer(Intent.ENTITY) is Layer.ENTITY


def test_enums_round_trip_through_text() -> None:
    assert Layer("CAUSAL") is Layer.CAUSAL
    assert Ablation("NO_CAUSAL") is Ablation.NO_CAUSAL
    assert f"{Layer.CAUSAL}" == "CAUSAL"


def test_unknown_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        Layer("CAUSL")
    with pytest.raises(ValueError):
        Ablation("no_such_thing")


def test_config_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT.retrieval.top_k = 5  # type: ignore[misc]  # the point of the test
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT.traversal.intent_weights.why.causal = 0.0  # type: ignore[misc]  # the point of the test


LAYER_ABLATIONS = [a for a in Ablation if a is not Ablation.NO_ADAPTIVE_POLICY]


@pytest.mark.parametrize("ablation", LAYER_ABLATIONS)
def test_layer_ablations_zero_removed_layers(ablation: Ablation) -> None:
    cfg = DEFAULT.with_ablation(ablation)
    for intent in Intent:
        weights = cfg.layer_weights(intent)
        for layer in Layer:
            if layer in cfg.active_layers:
                assert weights.of(layer) > 0
            else:
                assert weights.of(layer) == 0


def test_no_adaptive_policy_uses_same_weights_for_every_intent() -> None:
    cfg = DEFAULT.with_ablation(Ablation.NO_ADAPTIVE_POLICY)
    assert not cfg.adaptive
    assert cfg.layer_weights(Intent.WHY) == cfg.layer_weights(Intent.WHEN) == cfg.layer_weights(Intent.ENTITY)
    assert cfg.active_layers == tuple(Layer)
    assert cfg.layer_weights(Intent.WHY).causal == pytest.approx((5.0 + 3.0 + 3.0) / 3)


def test_with_ablation_keeps_default_untouched() -> None:
    DEFAULT.with_ablation(Ablation.ONLY_CAUSAL)
    assert DEFAULT.ablation is None
    assert DEFAULT.active_layers == tuple(Layer)
