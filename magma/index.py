"""Search indexes over node ids: dense vectors (semantic channel) and BM25 (lexical channel).

Both take an optional set of allowed ids, applied before ranking, so a reader
never gets a hit they cannot see. Like the graph store, each index is an
abstract base: the public methods hold the rules and the backends only store
and rank.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from magma.embed import Vector

# Words too common to say what a text is about. BM25 already gives them a low
# weight; dropping them keeps a question like "what did she say" from matching
# every turn through "what", "did" and "she".
STOPWORDS = frozenset(
    """
    a about after again all am an and any are as at be been before being but by can could did do does
    doing don for from had has have having he her here hers him his how i if in into is it its just me
    my no nor not of off on once only or other our ours out over own s same she should so some such t
    than that the their theirs them then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your yours
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased words without stopwords. Used for stored texts and for queries alike."""
    return [word for word in re.findall(r"\w+", text.lower()) if word not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    node_id: str
    score: float


class InvalidIndexChange(ValueError):
    pass


class VectorIndex(ABC):
    """Exact nearest-neighbour search by inner product, which for normalized vectors is the cosine."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(self, node_id: str, vector: Vector) -> None:
        if self.contains(node_id):
            raise InvalidIndexChange(f"node {node_id!r} already has a vector")
        self._check_shape(vector)
        self._store(node_id, vector.astype(np.float32))

    def search(self, query: Vector, top_k: int, allowed: frozenset[str] | None = None) -> list[Hit]:
        """The `top_k` stored vectors closest to `query`, best first; `allowed=None` means every node."""
        self._check_shape(query)
        if top_k <= 0 or (allowed is not None and not allowed):
            return []
        return self._search(query.astype(np.float32), top_k, allowed)

    def _check_shape(self, vector: Vector) -> None:
        if vector.shape != (self._dimension,):
            raise InvalidIndexChange(f"vector of shape {vector.shape}, the index holds dimension {self._dimension}")

    @abstractmethod
    def contains(self, node_id: str) -> bool: ...

    @abstractmethod
    def vector(self, node_id: str) -> Vector:
        """The stored vector; raises KeyError for an unknown id."""

    @abstractmethod
    def ids(self) -> list[str]:
        """In insertion order."""

    @abstractmethod
    def _store(self, node_id: str, vector: Vector) -> None: ...

    @abstractmethod
    def _search(self, query: Vector, top_k: int, allowed: frozenset[str] | None) -> list[Hit]: ...


class LexicalIndex(ABC):
    """Keyword search: only nodes that share at least one term with the query are hits."""

    def add(self, node_id: str, text: str) -> None:
        if self.contains(node_id):
            raise InvalidIndexChange(f"node {node_id!r} is already indexed")
        self._store(node_id, tokenize(text))

    def search(self, query: str, top_k: int, allowed: frozenset[str] | None = None) -> list[Hit]:
        """The `top_k` best-scoring nodes for the words of `query`, best first."""
        terms = tokenize(query)
        if not terms or top_k <= 0 or (allowed is not None and not allowed):
            return []
        ranked = self._search(terms, top_k, allowed)
        return [hit for hit in ranked if hit.score > 0]

    @abstractmethod
    def contains(self, node_id: str) -> bool: ...

    @abstractmethod
    def _store(self, node_id: str, terms: list[str]) -> None: ...

    @abstractmethod
    def _search(self, terms: list[str], top_k: int, allowed: frozenset[str] | None) -> list[Hit]:
        """Best first, ties in insertion order."""


def save_vectors(index: VectorIndex, path: Path) -> None:
    ids = index.ids()
    matrix = np.zeros((len(ids), index.dimension), dtype=np.float32)
    for row, node_id in enumerate(ids):
        matrix[row] = index.vector(node_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(file, ids=np.array(ids, dtype=np.str_), vectors=matrix)


def load_vectors(path: Path, index: VectorIndex) -> None:
    """Fill an empty index from a file written by `save_vectors`."""
    if index.ids():
        raise InvalidIndexChange("load_vectors needs an empty index")
    with np.load(path, allow_pickle=False) as saved:
        ids = [str(node_id) for node_id in saved["ids"]]
        vectors = saved["vectors"]
    for node_id, vector in zip(ids, vectors, strict=True):
        index.add(node_id, vector)
