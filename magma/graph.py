"""The memory graph: typed nodes and edges, the store interface and JSON persistence.

Every change is checked against a `Profile` from `schema.py`, so a store only
accepts the node and edge types its profile declares. The store keeps no
vectors: embeddings live in the vector index, keyed by node id.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from pydantic import TypeAdapter

from magma.config import Layer
from magma.json_types import JSONObject
from magma.schema import NodeRole, Profile

EVERYONE = "*"
VISIBLE_TO_EVERYONE = frozenset({EVERYONE})


class UnknownNode(KeyError):
    pass


class InvalidGraphChange(ValueError):
    """A node or edge the profile does not allow, or one that is already in the graph."""


@dataclass(frozen=True)
class Node:
    """A memory item, n_i = <c_i, τ_i, v_i, A_i> of Eq. 3 without v_i, which the vector index keeps."""

    id: str
    type: str
    text: str  # c_i
    timestamp: datetime | None = None  # τ_i; only anchors have their own
    attributes: JSONObject = field(default_factory=dict)  # A_i, filled by the slow path
    visible_to: frozenset[str] = VISIBLE_TO_EVERYONE

    def is_visible_to(self, viewer: str | None) -> bool:
        """`None` is the omniscient reader, who sees every node."""
        return viewer is None or EVERYONE in self.visible_to or viewer in self.visible_to


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    type: str
    weight: float = 1.0  # e.g. the cosine of a SIMILAR_TO edge, the confidence of a CAUSES edge
    attributes: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class Neighbor:
    node: Node
    edge: Edge
    is_outgoing: bool  # the edge goes from the node asked about to this neighbor


class GraphStore(ABC):
    """Base of every graph backend.

    The public methods check each change against the profile and then call the
    abstract storage methods, so no backend can skip the checks. A backend only
    writes the storage methods.
    """

    def __init__(self, profile: Profile) -> None:
        self._profile = profile

    @property
    def profile(self) -> Profile:
        return self._profile

    def add_node(self, node: Node) -> None:
        if self.has_node(node.id):
            raise InvalidGraphChange(f"node {node.id!r} already exists")
        _check_node(self._profile, node)
        self._store_node(node)

    def replace_node(self, node: Node) -> None:
        """Store a new version of an existing node (same id and type), keeping its edges."""
        current = self.node(node.id)
        if current.type != node.type:
            raise InvalidGraphChange(f"node {node.id!r} is {current.type} and cannot become {node.type}")
        _check_node(self._profile, node)
        self._store_node(node)

    def add_edge(self, edge: Edge) -> None:
        source = self.node(edge.source)
        target = self.node(edge.target)
        _check_edge(self._profile, edge, source, target)
        if self.has_edge(edge.source, edge.target, edge.type):
            raise InvalidGraphChange(f"{edge.type} edge {edge.source!r} -> {edge.target!r} already exists")
        self._store_edge(edge)

    def has_node(self, node_id: str) -> bool:
        return self._find_node(node_id) is not None

    def has_edge(self, source: str, target: str, edge_type: str) -> bool:
        """For an undirected type, an edge in either direction counts."""
        if self._has_edge_from(source, target, edge_type):
            return True
        is_directed = self._profile.edge_type(edge_type).is_directed
        return not is_directed and self._has_edge_from(target, source, edge_type)

    def node(self, node_id: str) -> Node:
        found = self._find_node(node_id)
        if found is None:
            raise UnknownNode(node_id)
        return found

    def nodes(self, node_type: str | None = None) -> list[Node]:
        """In insertion order, which for events is chronological order."""
        return [node for node in self._all_nodes() if node_type is None or node.type == node_type]

    def neighbors(self, node_id: str, layer: Layer | None = None) -> list[Neighbor]:
        """Nodes linked to `node_id` by an edge in either direction, optionally only edges of one layer."""
        self.node(node_id)
        found: list[Neighbor] = []
        for neighbor in self._incident(node_id):
            if layer is None or self._profile.edge_type(neighbor.edge.type).layer is layer:
                found.append(neighbor)
        return found

    @abstractmethod
    def edges(self) -> list[Edge]:
        """In insertion order."""

    @abstractmethod
    def _store_node(self, node: Node) -> None:
        """Insert the node, or overwrite the one with the same id. Already checked."""

    @abstractmethod
    def _store_edge(self, edge: Edge) -> None:
        """Insert the edge. Already checked."""

    @abstractmethod
    def _find_node(self, node_id: str) -> Node | None: ...

    @abstractmethod
    def _has_edge_from(self, source: str, target: str, edge_type: str) -> bool:
        """Only in the direction given."""

    @abstractmethod
    def _all_nodes(self) -> list[Node]:
        """In insertion order."""

    @abstractmethod
    def _incident(self, node_id: str) -> list[Neighbor]:
        """Every edge touching the node: outgoing first, then incoming, each in insertion order."""


def _check_node(profile: Profile, node: Node) -> None:
    node_type = profile.node_type(node.type)
    is_anchor = node_type.role is NodeRole.ANCHOR
    if is_anchor and node.timestamp is None:
        raise InvalidGraphChange(f"{node.type} node {node.id!r} is an anchor and needs a timestamp")
    if not is_anchor and node.timestamp is not None:
        raise InvalidGraphChange(f"{node.type} node {node.id!r} is not an anchor and cannot have its own timestamp")
    if not node.visible_to:
        raise InvalidGraphChange(f"node {node.id!r} is visible to no one")


def _check_edge(profile: Profile, edge: Edge, source: Node, target: Node) -> None:
    if edge.source == edge.target:
        raise InvalidGraphChange(f"{edge.type} edge from {edge.source!r} to itself")
    if not profile.allows(edge.type, source.type, target.type):
        raise InvalidGraphChange(f"{profile.name} does not allow {edge.type} from {source.type} to {target.type}")


class NodeRecord(TypedDict):
    id: str
    type: str
    text: str
    timestamp: str | None
    attributes: JSONObject
    visible_to: list[str]


class EdgeRecord(TypedDict):
    source: str
    target: str
    type: str
    weight: float
    attributes: JSONObject


class GraphFile(TypedDict):
    profile: str
    nodes: list[NodeRecord]
    edges: list[EdgeRecord]


_GRAPH_FILE = TypeAdapter(GraphFile)


def save_graph(store: GraphStore, path: Path) -> None:
    graph_file = GraphFile(
        profile=store.profile.name,
        nodes=[_node_to_record(node) for node in store.nodes()],
        edges=[_edge_to_record(edge) for edge in store.edges()],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(graph_file, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_graph(path: Path, store: GraphStore) -> None:
    """Fill an empty store from a file written by `save_graph` for the same profile."""
    if store.nodes():
        raise InvalidGraphChange("load_graph needs an empty store")
    graph_file = _GRAPH_FILE.validate_json(path.read_text(encoding="utf-8"))
    if graph_file["profile"] != store.profile.name:
        raise InvalidGraphChange(f"{path} holds a {graph_file['profile']} graph, the store uses {store.profile.name}")
    for node_record in graph_file["nodes"]:
        store.add_node(_node_from_record(node_record))
    for edge_record in graph_file["edges"]:
        store.add_edge(_edge_from_record(edge_record))


def _node_to_record(node: Node) -> NodeRecord:
    return NodeRecord(
        id=node.id,
        type=node.type,
        text=node.text,
        timestamp=node.timestamp.isoformat() if node.timestamp else None,
        attributes=node.attributes,
        visible_to=sorted(node.visible_to),
    )


def _node_from_record(record: NodeRecord) -> Node:
    timestamp = record["timestamp"]
    return Node(
        id=record["id"],
        type=record["type"],
        text=record["text"],
        timestamp=datetime.fromisoformat(timestamp) if timestamp else None,
        attributes=record["attributes"],
        visible_to=frozenset(record["visible_to"]),
    )


def _edge_to_record(edge: Edge) -> EdgeRecord:
    return EdgeRecord(
        source=edge.source,
        target=edge.target,
        type=edge.type,
        weight=edge.weight,
        attributes=edge.attributes,
    )


def _edge_from_record(record: EdgeRecord) -> Edge:
    return Edge(
        source=record["source"],
        target=record["target"],
        type=record["type"],
        weight=record["weight"],
        attributes=record["attributes"],
    )
