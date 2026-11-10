"""In-memory GraphStore on a NetworkX MultiDiGraph.

The only module that imports networkx. Each NetworkX edge is keyed by its type,
so two nodes can be linked by several edges as long as their types differ.
"""

from __future__ import annotations

import threading

import networkx as nx

from magma.graph import Edge, GraphStore, Neighbor, Node
from magma.schema import Profile


class NetworkXGraphStore(GraphStore):
    def __init__(self, profile: Profile) -> None:
        super().__init__(profile)
        self._graph = nx.MultiDiGraph()
        self._lock = threading.RLock()

    def edges(self) -> list[Edge]:
        with self._lock:
            return [stored for _, _, stored in self._graph.edges(data="edge")]

    def _store_node(self, node: Node) -> None:
        with self._lock:
            self._graph.add_node(node.id, node=node)

    def _store_edge(self, edge: Edge) -> None:
        with self._lock:
            self._graph.add_edge(edge.source, edge.target, key=edge.type, edge=edge)

    def _find_node(self, node_id: str) -> Node | None:
        with self._lock:
            if not self._graph.has_node(node_id):
                return None
            stored: Node = self._graph.nodes[node_id]["node"]
            return stored

    def _has_edge_from(self, source: str, target: str, edge_type: str) -> bool:
        with self._lock:
            found: bool = self._graph.has_edge(source, target, key=edge_type)
            return found

    def _all_nodes(self) -> list[Node]:
        with self._lock:
            return [stored for _, stored in self._graph.nodes(data="node")]

    def _incident(self, node_id: str) -> list[Neighbor]:
        with self._lock:
            found: list[Neighbor] = []
            for _, target_id, edge in self._graph.out_edges(node_id, data="edge"):
                found.append(Neighbor(node=self.node(target_id), edge=edge, is_outgoing=True))
            for source_id, _, edge in self._graph.in_edges(node_id, data="edge"):
                found.append(Neighbor(node=self.node(source_id), edge=edge, is_outgoing=False))
            return found
