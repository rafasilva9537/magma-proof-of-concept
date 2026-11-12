"""What one viewer can see: the graph and the indexes with every other node removed.

The magma retrieval stages run unchanged on these views, so the visibility
filter is hard by construction: an invisible node is never a hit in the vector
or keyword channel, never in the time channel, never a neighbor in the beam
search, and never evidence for a claim status. The views are read-only.
"""

from __future__ import annotations

from magma.embed import Vector
from magma.graph import Edge, GraphStore, Neighbor, Node
from magma.index import Hit, LexicalIndex, VectorIndex


class ReadOnlyView(TypeError):
    pass


class ViewerGraph(GraphStore):
    def __init__(self, graph: GraphStore, viewer: str) -> None:
        super().__init__(graph.profile)
        self._graph = graph
        self._viewer = viewer

    @property
    def viewer(self) -> str:
        return self._viewer

    def edges(self) -> list[Edge]:
        return [edge for edge in self._graph.edges() if self._sees(edge.source) and self._sees(edge.target)]

    def _store_node(self, node: Node) -> None:
        raise ReadOnlyView("a viewer's graph cannot be changed")

    def _store_edge(self, edge: Edge) -> None:
        raise ReadOnlyView("a viewer's graph cannot be changed")

    def _find_node(self, node_id: str) -> Node | None:
        if not self._graph.has_node(node_id):
            return None
        node = self._graph.node(node_id)
        return node if node.is_visible_to(self._viewer) else None

    def _has_edge_from(self, source: str, target: str, edge_type: str) -> bool:
        if not (self._sees(source) and self._sees(target)):
            return False
        return any(
            neighbor.is_outgoing and neighbor.node.id == target and neighbor.edge.type == edge_type
            for neighbor in self._graph.neighbors(source)
        )

    def _all_nodes(self) -> list[Node]:
        return [node for node in self._graph.nodes() if node.is_visible_to(self._viewer)]

    def _incident(self, node_id: str) -> list[Neighbor]:
        return [neighbor for neighbor in self._graph.neighbors(node_id) if neighbor.node.is_visible_to(self._viewer)]

    def _sees(self, node_id: str) -> bool:
        return self._find_node(node_id) is not None


def visible_ids(view: ViewerGraph) -> frozenset[str]:
    return frozenset(node.id for node in view.nodes())


class MaskedVectorIndex(VectorIndex):
    """The vector index searched only among the ids a viewer can see."""

    def __init__(self, index: VectorIndex, visible: frozenset[str]) -> None:
        super().__init__(index.dimension)
        self._index = index
        self._visible = visible

    def contains(self, node_id: str) -> bool:
        return node_id in self._visible and self._index.contains(node_id)

    def vector(self, node_id: str) -> Vector:
        if node_id not in self._visible:
            raise KeyError(node_id)
        return self._index.vector(node_id)

    def ids(self) -> list[str]:
        return [node_id for node_id in self._index.ids() if node_id in self._visible]

    def _store(self, node_id: str, vector: Vector) -> None:
        raise ReadOnlyView("a viewer's index cannot be changed")

    def _search(self, query: Vector, top_k: int, allowed: frozenset[str] | None) -> list[Hit]:
        mask = self._visible if allowed is None else allowed & self._visible
        return self._index.search(query, top_k, mask)


class MaskedLexicalIndex(LexicalIndex):
    """The BM25 index searched only among the ids a viewer can see."""

    def __init__(self, index: LexicalIndex, visible: frozenset[str]) -> None:
        self._index = index
        self._visible = visible

    def contains(self, node_id: str) -> bool:
        return node_id in self._visible and self._index.contains(node_id)

    def _store(self, node_id: str, terms: list[str]) -> None:
        raise ReadOnlyView("a viewer's index cannot be changed")

    def _search(self, terms: list[str], top_k: int, allowed: frozenset[str] | None) -> list[Hit]:
        mask = self._visible if allowed is None else allowed & self._visible
        # The terms are already tokenized; joined again, they tokenize to themselves.
        return self._index.search(" ".join(terms), top_k, mask)
