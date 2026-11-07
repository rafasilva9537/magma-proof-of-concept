import dataclasses

import pytest

from magma.config import ABLATIONS, DEFAULT, INTENTS, LAYERS, Config


def test_table6_values():
    assert DEFAULT.retrieval.rrf_k == 60
    assert DEFAULT.retrieval.top_k == 20
    assert DEFAULT.traversal.max_depth == 5
    assert DEFAULT.traversal.max_nodes == 200
    assert DEFAULT.traversal.drop_threshold == 0.15
    assert DEFAULT.traversal.lambda1 == 1.0
    assert 0.3 <= DEFAULT.traversal.lambda2 <= 0.7
    assert DEFAULT.llm.temperature == 0.0


def test_intent_weights_inside_table6_ranges():
    ranges = {"ENT": (2.5, 6.0), "TEMP": (0.5, 4.0), "CAUSAL": (3.0, 5.0), "SEM": (2.5, 5.0)}
    for intent in INTENTS:
        weights = DEFAULT.traversal.intent_weights[intent]
        for layer in LAYERS:
            low, high = ranges[layer]
            assert low <= weights[layer] <= high, (intent, layer)
    assert 2.0 <= DEFAULT.retrieval.channel_weights["keyword"] <= 5.0


def test_each_intent_favours_its_layer():
    assert max(DEFAULT.layer_weights("WHY"), key=DEFAULT.layer_weights("WHY").get) == "CAUSAL"
    assert max(DEFAULT.layer_weights("WHEN"), key=DEFAULT.layer_weights("WHEN").get) == "TEMP"
    assert max(DEFAULT.layer_weights("ENTITY"), key=DEFAULT.layer_weights("ENTITY").get) == "ENT"


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT.retrieval.top_k = 5
    with pytest.raises(TypeError):
        DEFAULT.traversal.intent_weights["WHY"]["CAUSAL"] = 0.0


def test_unknown_ablation_rejected():
    with pytest.raises(ValueError):
        Config(ablation="no_such_thing")


@pytest.mark.parametrize("ablation", [a for a in ABLATIONS if a != "no_adaptive_policy"])
def test_layer_ablations_zero_removed_layers(ablation):
    cfg = DEFAULT.with_ablation(ablation)
    for intent in INTENTS:
        weights = cfg.layer_weights(intent)
        for layer in LAYERS:
            if layer in cfg.active_layers:
                assert weights[layer] > 0
            else:
                assert weights[layer] == 0


def test_no_adaptive_policy_uses_same_weights_for_every_intent():
    cfg = DEFAULT.with_ablation("no_adaptive_policy")
    assert not cfg.adaptive
    assert cfg.layer_weights("WHY") == cfg.layer_weights("WHEN") == cfg.layer_weights("ENTITY")
    assert cfg.active_layers == LAYERS


def test_with_ablation_keeps_default_untouched():
    DEFAULT.with_ablation("only_causal")
    assert DEFAULT.ablation is None
    assert DEFAULT.active_layers == LAYERS
