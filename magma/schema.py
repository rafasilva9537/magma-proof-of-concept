"""Type registry: which node and edge types exist, and what each one means to the mechanisms.

The graph, retrieval and linearization never test a type by name. They read the
declarations below (layer, producer, epistemic effect, role of the node), so a
new profile is a new list of declarations, not new code. Profiles in other
packages extend `MAGMA_PROFILE` with `Profile.extend`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from magma.config import Layer


class NodeRole(StrEnum):
    """How a node relates to time and to retrieval."""

    ANCHOR = "ANCHOR"  # has its own timestamp; the unit counted in the top-K anchors (an event)
    FACET = "FACET"  # belongs to exactly one anchor and inherits its timestamp and visibility
    ABSTRACT = "ABSTRACT"  # timeless, shared by many anchors (an entity)


class Producer(StrEnum):
    """Which part of the system creates edges of a type."""

    RULE = "RULE"  # deterministic code in the fast path
    LLM_EXTRACT = "LLM_EXTRACT"  # the slow-path LLM, from the type's description
    NLI = "NLI"  # the contradiction check
    DERIVED = "DERIVED"  # computed at query time, never stored


class EpistemicEffect(StrEnum):
    """What an edge does to the status of the claim it points to."""

    NONE = "NONE"
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    CONTRADICTS = "CONTRADICTS"
    RETRACTS = "RETRACTS"
    UPDATES = "UPDATES"


class MagmaNode(StrEnum):
    EVENT = "EVENT"
    ENTITY = "ENTITY"


class MagmaEdge(StrEnum):
    NEXT = "NEXT"
    SIMILAR_TO = "SIMILAR_TO"
    MENTIONS = "MENTIONS"
    CAUSES = "CAUSES"


class UnknownType(KeyError):
    pass


class InvalidProfile(ValueError):
    pass


@dataclass(frozen=True)
class NodeType:
    name: str
    description: str  # also shown to the extraction LLM
    role: NodeRole
    is_embedded: bool  # gets a vector and a BM25 entry


@dataclass(frozen=True)
class EdgeType:
    name: str
    description: str  # also shown to the extraction LLM
    sources: frozenset[str]  # node type names allowed at the tail
    targets: frozenset[str]  # node type names allowed at the head
    is_directed: bool
    producer: Producer
    phrase: str  # how a path step reads in a trace, with {source} and {target}
    layer: Layer | None = None  # weighted by w_Tq of this layer in the beam search (Eq. 5)
    is_anchoring: bool = False  # links a facet to its anchor: no layer, no weight, no hop cost
    effect: EpistemicEffect = EpistemicEffect.NONE
    weight_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.is_anchoring == (self.layer is not None):
            raise InvalidProfile(f"edge {self.name} must have either a layer or is_anchoring=True, not both")

    def describe(self, source: str, target: str) -> str:
        return self.phrase.format(source=source, target=target)


@dataclass(frozen=True)
class Profile:
    name: str
    node_types: tuple[NodeType, ...]
    edge_types: tuple[EdgeType, ...]

    def __post_init__(self) -> None:
        _check_unique(self.name, [t.name for t in self.node_types], "node")
        _check_unique(self.name, [t.name for t in self.edge_types], "edge")
        declared = {t.name for t in self.node_types}
        for edge in self.edge_types:
            unknown = (edge.sources | edge.targets) - declared
            if unknown:
                raise InvalidProfile(f"{self.name}: edge {edge.name} uses undeclared node types {sorted(unknown)}")
        for node in self.node_types:
            if node.role is NodeRole.FACET and not self._anchoring_edges_into(node.name):
                raise InvalidProfile(f"{self.name}: facet {node.name} has no anchoring edge from an anchor")

    def node_type(self, name: str) -> NodeType:
        for node in self.node_types:
            if node.name == name:
                return node
        raise UnknownType(f"{self.name} has no node type {name!r}")

    def edge_type(self, name: str) -> EdgeType:
        for edge in self.edge_types:
            if edge.name == name:
                return edge
        raise UnknownType(f"{self.name} has no edge type {name!r}")

    def edges_in_layer(self, layer: Layer) -> tuple[EdgeType, ...]:
        return tuple(edge for edge in self.edge_types if edge.layer is layer)

    def allows(self, edge_name: str, source_type: str, target_type: str) -> bool:
        edge = self.edge_type(edge_name)
        if source_type in edge.sources and target_type in edge.targets:
            return True
        return not edge.is_directed and source_type in edge.targets and target_type in edge.sources

    def extend(
        self,
        name: str,
        node_types: tuple[NodeType, ...] = (),
        edge_types: tuple[EdgeType, ...] = (),
    ) -> Profile:
        """A new profile with these types added; a type with an existing name replaces it."""
        new_node_names = {t.name for t in node_types}
        new_edge_names = {t.name for t in edge_types}
        kept_nodes = tuple(t for t in self.node_types if t.name not in new_node_names)
        kept_edges = tuple(t for t in self.edge_types if t.name not in new_edge_names)
        return Profile(name=name, node_types=kept_nodes + node_types, edge_types=kept_edges + edge_types)

    def _anchoring_edges_into(self, node_name: str) -> list[EdgeType]:
        anchors = {t.name for t in self.node_types if t.role is NodeRole.ANCHOR}
        return [
            edge
            for edge in self.edge_types
            if edge.is_anchoring and node_name in edge.targets and edge.sources <= anchors
        ]


def _check_unique(profile: str, names: list[str], kind: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise InvalidProfile(f"{profile}: {kind} type {name!r} declared twice")
        seen.add(name)


EVENT = NodeType(
    name=MagmaNode.EVENT,
    description="Something that happened in the conversation: one turn, with its speaker and timestamp.",
    role=NodeRole.ANCHOR,
    is_embedded=True,
)

ENTITY = NodeType(
    name=MagmaNode.ENTITY,
    description="A person, place, object or concept mentioned in one or more events.",
    role=NodeRole.ABSTRACT,
    is_embedded=False,
)

_EVENTS = frozenset({MagmaNode.EVENT.value})
_ENTITIES = frozenset({MagmaNode.ENTITY.value})

NEXT = EdgeType(
    name=MagmaEdge.NEXT,
    description="Chronological order: the target happened right after the source.",
    sources=_EVENTS,
    targets=_EVENTS,
    is_directed=True,
    producer=Producer.RULE,
    phrase="{source} is followed by {target}",
    layer=Layer.TEMPORAL,
)

SIMILAR_TO = EdgeType(
    name=MagmaEdge.SIMILAR_TO,
    description="The two events are about similar things (cosine similarity above the threshold).",
    sources=_EVENTS,
    targets=_EVENTS,
    is_directed=False,
    producer=Producer.RULE,
    phrase="{source} is similar to {target}",
    layer=Layer.SEMANTIC,
)

MENTIONS = EdgeType(
    name=MagmaEdge.MENTIONS,
    description="The event mentions the entity.",
    sources=_EVENTS,
    targets=_ENTITIES,
    is_directed=True,
    producer=Producer.LLM_EXTRACT,
    phrase="{source} mentions {target}",
    layer=Layer.ENTITY,
)

CAUSES = EdgeType(
    name=MagmaEdge.CAUSES,
    description="The source event is a cause or a reason for the target event.",
    sources=_EVENTS,
    targets=_EVENTS,
    is_directed=True,
    producer=Producer.LLM_EXTRACT,
    phrase="{source} causes {target}",
    layer=Layer.CAUSAL,
)

MAGMA_PROFILE = Profile(
    name="magma",
    node_types=(EVENT, ENTITY),
    edge_types=(NEXT, SIMILAR_TO, MENTIONS, CAUSES),
)
