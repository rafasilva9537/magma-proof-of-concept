"""Fast path of memory evolution (Alg. 2): a turn becomes an event, with no LLM call.

Each turn becomes one event node. It is linked to the previous event by NEXT
(the temporal backbone), embedded and indexed (vector and BM25), linked by
SIMILAR_TO to the closest earlier events, and queued for the slow path, which
adds the entity and causal structure later.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from datetime import datetime

from magma.config import DEFAULT, IngestConfig
from magma.embed import Embedder, Vector
from magma.graph import Edge, GraphStore, Node
from magma.index import LexicalIndex, VectorIndex
from magma.schema import MagmaEdge, MagmaNode


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str
    timestamp: datetime


class OutOfOrderTurn(ValueError):
    """A turn older than the last event: the temporal backbone only grows forward."""


class FastPathIngestion:
    def __init__(
        self,
        graph: GraphStore,
        vectors: VectorIndex,
        keywords: LexicalIndex,
        embedder: Embedder,
        consolidation_queue: queue.Queue[str],
        config: IngestConfig = DEFAULT.ingest,
    ) -> None:
        self._graph = graph
        self._vectors = vectors
        self._keywords = keywords
        self._embedder = embedder
        self._queue = consolidation_queue
        self._config = config
        events = graph.nodes(MagmaNode.EVENT)
        self._event_count = len(events)
        self._last_event = events[-1] if events else None

    def ingest(self, turn: Turn) -> Node:
        """Alg. 2: segment, extend the temporal backbone, index, link similar events, enqueue."""
        event = self._segment_event(turn)
        previous = self._last_event
        self._graph.add_node(event)
        if previous is not None:
            self._graph.add_edge(Edge(source=previous.id, target=event.id, type=MagmaEdge.NEXT))

        vector = self._embed(event)
        self._link_similar_events(event, vector)
        self._vectors.add(event.id, vector)
        self._keywords.add(event.id, event.text)

        self._queue.put(event.id)
        self._event_count += 1
        self._last_event = event
        return event

    def _segment_event(self, turn: Turn) -> Node:
        """One turn is one event. The speaker goes into the content, so retrieval and the QA prompt see who spoke."""
        if self._last_event is not None and self._last_event.timestamp is not None:
            if turn.timestamp < self._last_event.timestamp:
                raise OutOfOrderTurn(f"turn at {turn.timestamp} comes after an event at {self._last_event.timestamp}")
        return Node(
            id=f"e{self._event_count + 1}",
            type=MagmaNode.EVENT,
            text=f"{turn.speaker}: {turn.text}",
            timestamp=turn.timestamp,
            attributes={"speaker": turn.speaker},
        )

    def _embed(self, event: Node) -> Vector:
        vector: Vector = self._embedder.embed_documents([event.text])[0]
        return vector

    def _link_similar_events(self, event: Node, vector: Vector) -> None:
        """SIMILAR_TO to the closest earlier events with cos(v_i, v_j) > θ_sim, at most `max_neighbors` of them.

        Runs before the event's own vector is indexed, so the event never finds itself.
        """
        closest = self._vectors.search(vector, top_k=self._config.semantic_edge_max_neighbors)
        for hit in closest:
            if hit.score > self._config.semantic_edge_threshold:
                self._graph.add_edge(Edge(source=event.id, target=hit.node_id, type=MagmaEdge.SIMILAR_TO, weight=hit.score))
