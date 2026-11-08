"""LexicalIndex on rank_bm25 (Okapi BM25).

The only module that imports rank_bm25. Its model is built from the whole
corpus at once, so it is rebuilt on the first search after new documents
arrive, not on every add.
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from magma.config import DEFAULT, LexicalIndexConfig
from magma.index import Hit, LexicalIndex


class BM25Index(LexicalIndex):
    def __init__(self, config: LexicalIndexConfig = DEFAULT.lexical_index) -> None:
        self._config = config
        self._ids: list[str] = []
        self._positions: dict[str, int] = {}
        self._documents: list[list[str]] = []
        self._model: BM25Okapi | None = None

    def contains(self, node_id: str) -> bool:
        return node_id in self._positions

    def _store(self, node_id: str, terms: list[str]) -> None:
        self._positions[node_id] = len(self._ids)
        self._ids.append(node_id)
        self._documents.append(terms)
        self._model = None

    def _search(self, terms: list[str], top_k: int, allowed: frozenset[str] | None) -> list[Hit]:
        if not any(self._documents):
            return []
        if allowed is None:
            positions = list(range(len(self._ids)))
        else:
            positions = sorted(self._positions[node_id] for node_id in allowed if node_id in self._positions)
        if not positions:
            return []
        scores = self._fitted_model().get_batch_scores(terms, positions)
        ranked = sorted(zip(positions, scores, strict=True), key=lambda pair: (-pair[1], pair[0]))
        return [Hit(node_id=self._ids[position], score=float(score)) for position, score in ranked[:top_k]]

    def _fitted_model(self) -> BM25Okapi:
        if self._model is None:
            self._model = BM25Okapi(self._documents, k1=self._config.k1, b=self._config.b, epsilon=self._config.epsilon)
        return self._model
