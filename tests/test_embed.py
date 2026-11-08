import logging
import os

import numpy as np
import pytest

from magma.embed import InstructedEmbedder, Matrix, cosine, normalize


class FakeEncoder:
    """Bag-of-letters vectors: deterministic, no model download. Records the prompts it receives."""

    dimension = 26
    max_tokens = 8

    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode(self, texts: list[str], prompt: str | None) -> Matrix:
        self.prompts.append(prompt)
        rows = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for char in text.lower():
                if "a" <= char <= "z":
                    rows[i, ord(char) - ord("a")] += 1.0
        return rows


def make_embedder() -> tuple[InstructedEmbedder, FakeEncoder]:
    encoder = FakeEncoder()
    return InstructedEmbedder(encoder, "Find the answer"), encoder


def test_instruction_goes_on_queries_only() -> None:
    embedder, encoder = make_embedder()
    embedder.embed_documents(["Melanie plays the violin"])
    embedder.embed_query("What does Melanie play?")
    assert encoder.prompts == [None, "Instruct: Find the answer\nQuery:"]


def test_vectors_are_unit_length() -> None:
    embedder, _ = make_embedder()
    docs = embedder.embed_documents(["violin", "clarinet and violin"])
    query = embedder.embed_query("violin")
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
    assert np.linalg.norm(query) == pytest.approx(1.0)
    assert docs.dtype == np.float32
    assert docs.shape == (2, 26)


def test_similar_texts_score_higher() -> None:
    embedder, _ = make_embedder()
    violin, war = embedder.embed_documents(["violin", "war"])
    query = embedder.embed_query("violins")
    assert cosine(query, violin) > cosine(query, war)


def test_empty_batch_returns_empty_matrix() -> None:
    embedder, encoder = make_embedder()
    assert embedder.embed_documents([]).shape == (0, 26)
    assert encoder.prompts == []


def test_long_text_triggers_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    embedder, _ = make_embedder()
    with caplog.at_level(logging.WARNING, logger="magma.embed"):
        embedder.embed_documents(["one two three"])
        assert not caplog.records
        embedder.embed_documents(["word " * 20])
    assert "keeps only the first 8" in caplog.text


def test_normalize_keeps_zero_rows() -> None:
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])


@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="downloads the embedding model; set RUN_MODEL_TESTS=1")
def test_real_model_ranks_paraphrase_above_unrelated() -> None:
    embedder = InstructedEmbedder.from_config()
    docs = embedder.embed_documents(
        [
            "Melanie said she has been playing the violin since she was a kid.",
            "The weather in Paris was rainy all week.",
        ]
    )
    query = embedder.embed_query("Which instrument does Melanie play?")
    assert embedder.dimension == docs.shape[1]
    assert cosine(query, docs[0]) > cosine(query, docs[1])
