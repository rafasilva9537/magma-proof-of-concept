from magma.config import Layer
from magma.schema import MAGMA_PROFILE, EpistemicEffect, NodeRole
from narrative.schema import NARRATIVE_PROFILE, NarrativeEdge, NarrativeNode


def test_narrative_profile_has_three_node_types_and_nine_edge_types() -> None:
    assert [node.name for node in NARRATIVE_PROFILE.node_types] == ["EVENT", "CLAIM", "ENTITY"]
    assert len(NARRATIVE_PROFILE.edge_types) == 9


def test_claim_is_an_embedded_facet_anchored_by_stated() -> None:
    claim = NARRATIVE_PROFILE.node_type(NarrativeNode.CLAIM)
    assert claim.role is NodeRole.FACET and claim.is_embedded
    stated = NARRATIVE_PROFILE.edge_type(NarrativeEdge.STATED)
    assert stated.is_anchoring and stated.layer is None


def test_new_edges_take_the_layer_of_their_meaning_and_declare_their_effect() -> None:
    expected = {
        NarrativeEdge.INVALIDATES: (Layer.CAUSAL, EpistemicEffect.REFUTES),
        NarrativeEdge.SUPPORTS: (Layer.CAUSAL, EpistemicEffect.SUPPORTS),
        NarrativeEdge.CONTRADICTS: (Layer.CAUSAL, EpistemicEffect.CONTRADICTS),
        NarrativeEdge.SUPERSEDES: (Layer.TEMPORAL, EpistemicEffect.RETRACTS),
    }
    for name, (layer, effect) in expected.items():
        edge = NARRATIVE_PROFILE.edge_type(name)
        assert (edge.layer, edge.effect) == (layer, effect)


def test_evidence_comes_only_from_facts_and_contradiction_only_between_claims() -> None:
    assert NARRATIVE_PROFILE.allows(NarrativeEdge.INVALIDATES, "EVENT", "CLAIM")
    assert not NARRATIVE_PROFILE.allows(NarrativeEdge.INVALIDATES, "CLAIM", "CLAIM")
    assert NARRATIVE_PROFILE.allows(NarrativeEdge.CONTRADICTS, "CLAIM", "CLAIM")
    assert not NARRATIVE_PROFILE.allows(NarrativeEdge.CONTRADICTS, "EVENT", "CLAIM")


def test_magma_profile_is_untouched() -> None:
    assert [edge.name for edge in MAGMA_PROFILE.edge_types] == ["NEXT", "SIMILAR_TO", "MENTIONS", "CAUSES"]
    assert MAGMA_PROFILE.edge_type("MENTIONS").sources == frozenset({"EVENT"})
