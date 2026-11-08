"""VectorIndex on FAISS `IndexFlatIP`: exact search, inner product over normalized vectors.

The only module that imports faiss. FAISS numbers vectors by insertion
position; this class keeps the mapping between positions and node ids.
"""

from __future__ import annotations

import faiss
import numpy as np

from magma.embed import Vector
from magma.index import Hit, VectorIndex


class FaissVectorIndex(VectorIndex):
    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        self._index = faiss.IndexFlatIP(dimension)
        self._ids: list[str] = []
        self._positions: dict[str, int] = {}

    def contains(self, node_id: str) -> bool:
        return node_id in self._positions

    def vector(self, node_id: str) -> Vector:
        stored = np.empty(self.dimension, dtype=np.float32)
        self._index.reconstruct(self._positions[node_id], stored)
        return stored

    def ids(self) -> list[str]:
        return list(self._ids)

    def _store(self, node_id: str, vector: Vector) -> None:
        self._positions[node_id] = len(self._ids)
        self._ids.append(node_id)
        self._index.add(vector.reshape(1, -1))

    def _search(self, query: Vector, top_k: int, allowed: frozenset[str] | None) -> list[Hit]:
        parameters = None
        if allowed is not None:
            allowed_positions = sorted(self._positions[node_id] for node_id in allowed if node_id in self._positions)
            if not allowed_positions:
                return []
            parameters = faiss.SearchParameters()
            parameters.sel = faiss.IDSelectorBatch(np.array(allowed_positions, dtype=np.int64))
        k = min(top_k, len(self._ids))
        if k == 0:
            return []
        scores, positions = self._index.search(query.reshape(1, -1), k, params=parameters)
        hits: list[Hit] = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:  # FAISS pads with -1 when fewer vectors are allowed than were asked for
                continue
            hits.append(Hit(node_id=self._ids[position], score=float(score)))
        return hits
