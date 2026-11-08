import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from magma.bm25_index import BM25Index
from magma.embed import Vector, normalize
from magma.faiss_index import FaissVectorIndex
from magma.index import InvalidIndexChange, LexicalIndex, VectorIndex, load_vectors, save_vectors, tokenize

VECTOR_FACTORIES: list[Callable[[int], VectorIndex]] = [FaissVectorIndex]
LEXICAL_FACTORIES: list[Callable[[], LexicalIndex]] = [BM25Index]


@pytest.fixture(params=VECTOR_FACTORIES)
def make_vectors(request: pytest.FixtureRequest) -> Callable[[int], VectorIndex]:
    factory: Callable[[int], VectorIndex] = request.param
    return factory


@pytest.fixture(params=LEXICAL_FACTORIES)
def lexical(request: pytest.FixtureRequest) -> LexicalIndex:
    factory: Callable[[], LexicalIndex] = request.param
    return factory()


def unit(*values: float) -> Vector:
    vector: Vector = normalize(np.array([values], dtype=np.float32))[0]
    return vector


def three_directions(make_vectors: Callable[[int], VectorIndex]) -> VectorIndex:
    """e1 along x, e2 between x and y, e3 along y."""
    index = make_vectors(2)
    index.add("e1", unit(1, 0))
    index.add("e2", unit(1, 1))
    index.add("e3", unit(0, 1))
    return index


def test_vector_search_ranks_by_cosine(make_vectors: Callable[[int], VectorIndex]) -> None:
    index = three_directions(make_vectors)
    hits = index.search(unit(1, 0), top_k=3)
    assert [hit.node_id for hit in hits] == ["e1", "e2", "e3"]
    assert [hit.score for hit in hits] == pytest.approx([1.0, math.sqrt(0.5), 0.0], abs=1e-6)


def test_vector_search_respects_top_k(make_vectors: Callable[[int], VectorIndex]) -> None:
    index = three_directions(make_vectors)
    assert [hit.node_id for hit in index.search(unit(0, 1), top_k=1)] == ["e3"]
    assert len(index.search(unit(0, 1), top_k=10)) == 3
    assert index.search(unit(0, 1), top_k=0) == []
    assert make_vectors(2).search(unit(0, 1), top_k=5) == []


def test_vector_search_only_returns_allowed_ids(make_vectors: Callable[[int], VectorIndex]) -> None:
    index = three_directions(make_vectors)
    hits = index.search(unit(1, 0), top_k=3, allowed=frozenset({"e3", "e2", "not-indexed"}))
    assert [hit.node_id for hit in hits] == ["e2", "e3"]
    assert index.search(unit(1, 0), top_k=3, allowed=frozenset()) == []
    assert index.search(unit(1, 0), top_k=3, allowed=frozenset({"not-indexed"})) == []


def test_vector_index_checks_ids_and_shapes(make_vectors: Callable[[int], VectorIndex]) -> None:
    index = three_directions(make_vectors)
    with pytest.raises(InvalidIndexChange):
        index.add("e1", unit(0, 1))
    with pytest.raises(InvalidIndexChange):
        index.add("e4", unit(1, 0, 0))
    with pytest.raises(InvalidIndexChange):
        index.search(unit(1, 0, 0), top_k=1)
    with pytest.raises(KeyError):
        index.vector("missing")


def test_vectors_are_read_back_by_id(make_vectors: Callable[[int], VectorIndex]) -> None:
    index = three_directions(make_vectors)
    np.testing.assert_allclose(index.vector("e2"), unit(1, 1))
    assert index.ids() == ["e1", "e2", "e3"]
    assert index.contains("e3") and not index.contains("e4")


def test_vectors_save_and_load(make_vectors: Callable[[int], VectorIndex], tmp_path: Path) -> None:
    index = three_directions(make_vectors)
    path = tmp_path / "vectors" / "conversation.npz"
    save_vectors(index, path)

    loaded = make_vectors(2)
    load_vectors(path, loaded)
    assert loaded.ids() == index.ids()
    for node_id in index.ids():
        np.testing.assert_array_equal(loaded.vector(node_id), index.vector(node_id))
    with pytest.raises(InvalidIndexChange):
        load_vectors(path, loaded)


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    assert tokenize("What did Melanie's sister play at the Épsilon-4 show?") == [
        "melanie",
        "sister",
        "play",
        "épsilon",
        "4",
        "show",
    ]
    assert tokenize("what did she say") == ["say"]


def small_corpus(lexical: LexicalIndex) -> None:
    lexical.add("e1", "Melanie plays the violin")
    lexical.add("e2", "Melanie bought a clarinet")
    lexical.add("e3", "The weather in Paris")


def test_bm25_score_matches_the_formula(lexical: LexicalIndex) -> None:
    small_corpus(lexical)
    # After stopwords: e1 = melanie plays violin (3 terms), e2 = melanie bought clarinet (3), e3 = weather paris (2).
    # "violin" is in n = 1 of N = 3 documents, f = 1 in e1, |e1| = 3, avgdl = 8/3, k1 = 1.5, b = 0.75.
    idf = math.log((3 - 1 + 0.5) / (1 + 0.5))
    length_factor = 1 - 0.75 + 0.75 * 3 / (8 / 3)
    expected = idf * 1 * (1.5 + 1) / (1 + 1.5 * length_factor)
    hits = lexical.search("violin", top_k=5)
    assert [hit.node_id for hit in hits] == ["e1"]
    assert hits[0].score == pytest.approx(expected)


def test_bm25_ranks_rare_terms_and_skips_documents_without_overlap(lexical: LexicalIndex) -> None:
    small_corpus(lexical)
    hits = lexical.search("Melanie violin", top_k=5)
    assert [hit.node_id for hit in hits] == ["e1", "e2"]
    assert hits[0].score > hits[1].score > 0


def test_bm25_term_in_most_documents_keeps_a_small_positive_weight(lexical: LexicalIndex) -> None:
    small_corpus(lexical)
    # "melanie" is in 2 of 3 documents, so its raw IDF is negative; the epsilon floor keeps it positive.
    hits = lexical.search("melanie", top_k=5)
    assert [hit.node_id for hit in hits] == ["e1", "e2"]  # equal scores keep insertion order
    assert hits[0].score == pytest.approx(hits[1].score)
    assert hits[0].score > 0


def test_bm25_only_returns_allowed_ids(lexical: LexicalIndex) -> None:
    small_corpus(lexical)
    assert [hit.node_id for hit in lexical.search("melanie", top_k=5, allowed=frozenset({"e2"}))] == ["e2"]
    assert lexical.search("melanie", top_k=5, allowed=frozenset()) == []


def test_bm25_sees_documents_added_after_a_search(lexical: LexicalIndex) -> None:
    small_corpus(lexical)
    assert lexical.search("guitar", top_k=5) == []
    lexical.add("e4", "Caroline tunes her guitar")
    assert [hit.node_id for hit in lexical.search("guitar", top_k=5)] == ["e4"]


def test_bm25_edge_cases(lexical: LexicalIndex) -> None:
    assert lexical.search("violin", top_k=5) == []  # empty index
    lexical.add("e1", "and the of")
    assert lexical.search("violin", top_k=5) == []  # only stopwords indexed
    lexical.add("e2", "Melanie bought a clarinet")
    assert lexical.search("what did she", top_k=5) == []  # only stopwords asked
    assert lexical.search("clarinet", top_k=0) == []
    with pytest.raises(InvalidIndexChange):
        lexical.add("e2", "again")
