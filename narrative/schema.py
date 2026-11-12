"""The `narrative` profile: the magma profile plus claims, the anchoring edge and the epistemic edges.

A turn is still one EVENT, the observable fact that someone acted or spoke. What
a speaker asserts is a CLAIM, a facet of that event linked to it by STATED: it
can be false, so it never becomes a fact. INVALIDATES and SUPPORTS go from a
fact to a claim it refutes or confirms, CONTRADICTS links incompatible claims of
different speakers, and SUPERSEDES links a newer version to the one it replaces.
Each new edge sits in one of the paper's four layers and takes that layer's
weight in the beam search; the status of a claim is read from the declared
epistemic effect of its edges, never from their names.
"""

from __future__ import annotations

from enum import StrEnum

from magma.config import Layer
from magma.schema import (
    CAUSES,
    ENTITY,
    EVENT,
    MAGMA_PROFILE,
    NEXT,
    EdgeType,
    EpistemicEffect,
    MagmaEdge,
    MagmaNode,
    NodeRole,
    NodeType,
    Producer,
    Profile,
)


class NarrativeNode(StrEnum):
    CLAIM = "CLAIM"


class NarrativeEdge(StrEnum):
    STATED = "STATED"
    SUPERSEDES = "SUPERSEDES"
    INVALIDATES = "INVALIDATES"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"


class MentionRole(StrEnum):
    """The part an entity plays in the event or claim that mentions it (attribute `role` of MENTIONS)."""

    ACTOR = "ACTOR"
    TARGET = "TARGET"
    INSTRUMENT = "INSTRUMENT"
    LOCATION = "LOCATION"
    RECIPIENT = "RECIPIENT"
    OTHER = "OTHER"


class ClaimType(StrEnum):
    ASSERTION = "ASSERTION"  # the speaker says the statement is true
    DENIAL = "DENIAL"  # the speaker says the statement is false


class ClaimStatus(StrEnum):
    """The status of a claim as one viewer sees it, from the evidence that viewer witnessed."""

    UNVERIFIED = "UNVERIFIED"
    SUPPORTED = "SUPPORTED"
    CONTESTED = "CONTESTED"
    REFUTED = "REFUTED"
    RETRACTED = "RETRACTED"


CLAIM = NodeType(
    name=NarrativeNode.CLAIM,
    description="A proposition a speaker asserted or denied in one turn; it may be false.",
    role=NodeRole.FACET,
    is_embedded=True,
)

_EVENTS = frozenset({MagmaNode.EVENT.value})
_CLAIMS = frozenset({NarrativeNode.CLAIM.value})
_ENTITIES = frozenset({MagmaNode.ENTITY.value})
_EVENTS_AND_CLAIMS = _EVENTS | _CLAIMS

STATED = EdgeType(
    name=NarrativeEdge.STATED,
    description="The speech event states the claim.",
    sources=_EVENTS,
    targets=_CLAIMS,
    is_directed=True,
    producer=Producer.LLM_EXTRACT,
    phrase="{source} states {target}",
    is_anchoring=True,
)

SIMILAR_TO = EdgeType(
    name=MagmaEdge.SIMILAR_TO,
    description="Two events, or two claims, are about similar things (cosine similarity above the threshold).",
    sources=_EVENTS_AND_CLAIMS,
    targets=_EVENTS_AND_CLAIMS,
    is_directed=False,
    producer=Producer.RULE,
    phrase="{source} is similar to {target}",
    layer=Layer.SEMANTIC,
)

MENTIONS = EdgeType(
    name=MagmaEdge.MENTIONS,
    description="The event or claim mentions the entity, which plays the role given in the edge.",
    sources=_EVENTS_AND_CLAIMS,
    targets=_ENTITIES,
    is_directed=True,
    producer=Producer.LLM_EXTRACT,
    phrase="{source} mentions {target}",
    layer=Layer.ENTITY,
)

SUPERSEDES = EdgeType(
    name=NarrativeEdge.SUPERSEDES,
    description="The source is a newer version of the target: the same speaker changed the claim, or the world changed.",
    sources=_EVENTS_AND_CLAIMS,
    targets=_EVENTS_AND_CLAIMS,
    is_directed=True,
    producer=Producer.NLI,
    phrase="{source} supersedes {target}",
    layer=Layer.TEMPORAL,
    effect=EpistemicEffect.RETRACTS,
)

INVALIDATES = EdgeType(
    name=NarrativeEdge.INVALIDATES,
    description="What was observed in the event shows the claim is false.",
    sources=_EVENTS,
    targets=_CLAIMS,
    is_directed=True,
    producer=Producer.NLI,
    phrase="{source} refutes {target}",
    layer=Layer.CAUSAL,
    effect=EpistemicEffect.REFUTES,
)

SUPPORTS = EdgeType(
    name=NarrativeEdge.SUPPORTS,
    description="What was observed in the event confirms the claim.",
    sources=_EVENTS,
    targets=_CLAIMS,
    is_directed=True,
    producer=Producer.NLI,
    phrase="{source} confirms {target}",
    layer=Layer.CAUSAL,
    effect=EpistemicEffect.SUPPORTS,
)

CONTRADICTS = EdgeType(
    name=NarrativeEdge.CONTRADICTS,
    description="Two claims of different speakers cannot both be true.",
    sources=_CLAIMS,
    targets=_CLAIMS,
    is_directed=False,
    producer=Producer.NLI,
    phrase="{source} contradicts {target}",
    layer=Layer.CAUSAL,
    effect=EpistemicEffect.CONTRADICTS,
)

NARRATIVE_PROFILE: Profile = MAGMA_PROFILE.extend(
    "narrative",
    node_types=(EVENT, CLAIM, ENTITY),
    edge_types=(NEXT, SIMILAR_TO, MENTIONS, CAUSES, STATED, SUPERSEDES, INVALIDATES, SUPPORTS, CONTRADICTS),
)
