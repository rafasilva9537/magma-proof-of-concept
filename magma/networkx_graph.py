"""In-memory GraphStore on a NetworkX MultiDiGraph.

The only module that imports networkx. Each NetworkX edge is keyed by its type,
so two nodes can be linked by several edges as long as their types differ.
"""

from __future__ import annotations

import networkx as nx

from magma.config import Layer
from magma.graph import Edge, InvalidGraphChange, Neighbor, Node, UnknownNode, check_edge, check_node
from magma.schema import Profile


class NetworkXGraphStore:
    def __init__(self, profile: Profile) -> None:
        self._profile = profile
        self._graph = nx.MultiDiGraph()

    @property
    def profile(self) -> Profile:
        return self._profile

    def add_node(self, node: Node) -> None:
        if self.has_node(node.id):
            raise InvalidGraphChange(f"node {node.id!r} already exists")
        check_node(self._profile, node)
        self._graph.add_node(node.id, node=node)

    def replace_node(self, node: Node) -> None:
        current = self.node(node.id)
        if current.type != node.type:
            raise InvalidGraphChange(f"node {node.id!r} is {current.type} and cannot become {node.type}")
        check_node(self._profile, node)
        self._graph.nodes[node.id]["node"] = node

    def add_edge(self, edge: Edge) -> None:
        source = self.node(edge.source)
        target = self.node(edge.target)
        check_edge(self._profile, edge, source, target)
        if self.has_edge(edge.source, edge.target, edge.type):
            raise InvalidGraphChange(f"{edge.type} edge {edge.source!r} -> {edge.target!r} already exists")
        self._graph.add_edge(edge.source, edge.target, key=edge.type, edge=edge)

    def has_node(self, node_id: str) -> bool:
        found: bool = self._graph.has_node(node_id)
        return found

    def has_edge(self, source: str, target: str, edge_type: str) -> bool:
        if self._graph.has_edge(source, target, key=edge_type):
            return True
        is_directed = self._profile.edge_type(edge_type).is_directed
        found_reversed: bool = self._graph.has_edge(target, source, key=edge_type)
        return not is_directed and found_reversed

    def node(self, node_id: str) -> Node:
        if not self.has_node(node_id):
            raise UnknownNode(node_id)
        stored: Node = self._graph.nodes[node_id]["node"]
        return stored

    def nodes(self, node_type: str | None = None) -> list[Node]:
        found: list[Node] = []
        for _, stored in self._graph.nodes(data="node"):
            if node_type is None or stored.type == node_type:
                found.append(stored)
        return found

    def edges(self) -> list[Edge]:
        return [stored for _, _, stored in self._graph.edges(data="edge")]

    def neighbors(self, node_id: str, layer: Layer | None = None) -> list[Neighbor]:
        self.node(node_id)
        found: list[Neighbor] = []
        for _, target_id, edge in self._graph.out_edges(node_id, data="edge"):
            if self._is_in_layer(edge, layer):
                found.append(Neighbor(node=self.node(target_id), edge=edge, is_outgoing=True))
        for source_id, _, edge in self._graph.in_edges(node_id, data="edge"):
            if self._is_in_layer(edge, layer):
                found.append(Neighbor(node=self.node(source_id), edge=edge, is_outgoing=False))
        return found

    def _is_in_layer(self, edge: Edge, layer: Layer | None) -> bool:
        return layer is None or self._profile.edge_type(edge.type).layer is layer
