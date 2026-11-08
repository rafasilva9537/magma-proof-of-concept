import pytest

from magma.config import Layer
from magma.schema import (
    MAGMA_PROFILE,
    EdgeType,
    EpistemicEffect,
    InvalidProfile,
    MagmaEdge,
    MagmaNode,
    NodeRole,
    NodeType,
    Producer,
    UnknownType,
)

EVENTS = frozenset({MagmaNode.EVENT.value})


def test_magma_profile_has_one_edge_type_per_layer() -> None:
    for layer in Layer:
        assert len(MAGMA_PROFILE.edges_in_layer(layer)) == 1, layer
    assert {t.name for t in MAGMA_PROFILE.node_types} == {"EVENT", "ENTITY"}


def test_magma_profile_matches_the_paper() -> None:
    assert MAGMA_PROFILE.edge_type(MagmaEdge.NEXT).layer is Layer.TEMPORAL
    assert MAGMA_PROFILE.edge_type(MagmaEdge.SIMILAR_TO).layer is Layer.SEMANTIC
    assert MAGMA_PROFILE.edge_type(MagmaEdge.MENTIONS).layer is Layer.ENTITY
    assert MAGMA_PROFILE.edge_type(MagmaEdge.CAUSES).layer is Layer.CAUSAL
    assert not MAGMA_PROFILE.edge_type(MagmaEdge.SIMILAR_TO).is_directed
    assert MAGMA_PROFILE.node_type(MagmaNode.EVENT).role is NodeRole.ANCHOR
    assert MAGMA_PROFILE.node_type(MagmaNode.ENTITY).role is NodeRole.ABSTRACT
    assert all(t.effect is EpistemicEffect.NONE for t in MAGMA_PROFILE.edge_types)


def test_allowed_endpoints() -> None:
    assert MAGMA_PROFILE.allows("MENTIONS", "EVENT", "ENTITY")
    assert not MAGMA_PROFILE.allows("MENTIONS", "ENTITY", "EVENT")
    assert MAGMA_PROFILE.allows("SIMILAR_TO", "EVENT", "EVENT")


def test_unknown_types_raise() -> None:
    with pytest.raises(UnknownType):
        MAGMA_PROFILE.node_type("CLAIM")
    with pytest.raises(UnknownType):
        MAGMA_PROFILE.edge_type("STATED")


def test_edge_needs_exactly_one_of_layer_or_anchoring() -> None:
    with pytest.raises(InvalidProfile):
        EdgeType("BAD", "", EVENTS, EVENTS, True, Producer.RULE, "{source} {target}")
    with pytest.raises(InvalidProfile):
        EdgeType("BAD", "", EVENTS, EVENTS, True, Producer.RULE, "{source} {target}", layer=Layer.CAUSAL, is_anchoring=True)


def test_profile_rejects_undeclared_node_types_and_duplicates() -> None:
    to_claims = EdgeType("REFUTES", "", EVENTS, frozenset({"CLAIM"}), True, Producer.NLI, "{source} {target}", layer=Layer.CAUSAL)
    with pytest.raises(InvalidProfile):
        MAGMA_PROFILE.extend("broken", edge_types=(to_claims,))
    duplicate = MAGMA_PROFILE.node_types[0]
    with pytest.raises(InvalidProfile):
        type(MAGMA_PROFILE)("broken", (duplicate, duplicate), ())


CLAIM = NodeType("CLAIM", "Something a character stated.", NodeRole.FACET, is_embedded=True)
STATED = EdgeType(
    "STATED",
    "The speech event states the claim.",
    EVENTS,
    frozenset({"CLAIM"}),
    True,
    Producer.LLM_EXTRACT,
    "{source} states {target}",
    is_anchoring=True,
)


def test_facet_needs_an_anchoring_edge() -> None:
    with pytest.raises(InvalidProfile):
        MAGMA_PROFILE.extend("broken", node_types=(CLAIM,))


def test_extension_adds_types_without_touching_the_base_profile() -> None:
    invalidates = EdgeType(
        "INVALIDATES",
        "The fact refutes the claim.",
        EVENTS,
        frozenset({"CLAIM"}),
        True,
        Producer.NLI,
        "{source} refutes {target}",
        layer=Layer.CAUSAL,
        effect=EpistemicEffect.REFUTES,
    )
    extended = MAGMA_PROFILE.extend("narrative-like", node_types=(CLAIM,), edge_types=(STATED, invalidates))
    assert extended.node_type("CLAIM").role is NodeRole.FACET
    assert extended.edge_type("STATED").layer is None
    assert {t.name for t in extended.edges_in_layer(Layer.CAUSAL)} == {"CAUSES", "INVALIDATES"}
    assert len(MAGMA_PROFILE.edge_types) == 4


def test_extension_can_replace_a_type() -> None:
    weighted_causes = EdgeType(
        "CAUSES", "Cause.", EVENTS, EVENTS, True, Producer.LLM_EXTRACT, "{source} causes {target}",
        layer=Layer.CAUSAL, weight_multiplier=2.0,
    )
    extended = MAGMA_PROFILE.extend("custom", edge_types=(weighted_causes,))
    assert extended.edge_type("CAUSES").weight_multiplier == 2.0
    assert len(extended.edge_types) == 4


def test_phrase_describes_a_path_step() -> None:
    assert MAGMA_PROFILE.edge_type("CAUSES").describe("e12", "e15") == "e12 causes e15"
