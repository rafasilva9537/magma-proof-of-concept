"""Text embeddings: the vector v_i of each memory node and the query vector v_q.

Vectors are L2-normalized, so the inner product of two vectors is their cosine similarity.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from magma.config import DEFAULT, EmbeddingConfig

logger = logging.getLogger(__name__)

Vector = NDArray[np.float32]  # shape (dimension,)
Matrix = NDArray[np.float32]  # shape (number of texts, dimension)


class Embedder(Protocol):
    """What the rest of the system needs from an embedding model."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Vector: ...


class TextEncoder(Protocol):
    """A raw embedding model: turns texts into vectors, optionally with a prompt prefix."""

    @property
    def dimension(self) -> int: ...

    @property
    def max_tokens(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...

    def encode(self, texts: list[str], prompt: str | None) -> Matrix: ...


class SentenceTransformerEncoder:
    """Runs a sentence-transformers model in process (no HTTP)."""

    def __init__(self, model_name: str, batch_size: int):
        import torch
        from sentence_transformers import SentenceTransformer

        # The checkpoint loads as bfloat16, which most CPUs run much slower than float32.
        self._model = SentenceTransformer(model_name, model_kwargs={"dtype": torch.float32})
        self._batch_size = batch_size
        logger.info("loaded %s: dimension %d, max %d tokens", model_name, self.dimension, self.max_tokens)

    @property
    def dimension(self) -> int:
        size = self._model.get_embedding_dimension()
        if size is None:
            raise RuntimeError("embedding model does not report its dimension")
        return int(size)

    @property
    def max_tokens(self) -> int:
        limit = self._model.max_seq_length
        if limit is None:
            raise RuntimeError("embedding model does not report its maximum sequence length")
        return int(limit)

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer(text)["input_ids"])

    def encode(self, texts: list[str], prompt: str | None) -> Matrix:
        vectors = self._model.encode(texts, prompt=prompt, batch_size=self._batch_size, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


class InstructedEmbedder:
    """Embedder for instruction-aware models such as Qwen3-Embedding.

    Queries get the prefix "Instruct: <task>\\nQuery:"; stored texts are embedded as they are.
    Texts longer than the model's window are cut silently by the model, so this class warns about them.
    """

    def __init__(self, encoder: TextEncoder, query_instruction: str):
        self._encoder = encoder
        self._query_prompt = f"Instruct: {query_instruction}\nQuery:"

    @staticmethod
    def from_config(config: EmbeddingConfig = DEFAULT.embedding) -> InstructedEmbedder:
        encoder = SentenceTransformerEncoder(config.model_name, config.batch_size)
        return InstructedEmbedder(encoder, config.query_instruction)

    @property
    def dimension(self) -> int:
        return self._encoder.dimension

    @property
    def query_prompt(self) -> str:
        return self._query_prompt

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        batch = list(texts)
        if not batch:
            return np.zeros((0, self.dimension), dtype=np.float32)
        for text in batch:
            self._warn_if_truncated(text)
        return normalize(self._encoder.encode(batch, prompt=None))

    def embed_query(self, text: str) -> Vector:
        self._warn_if_truncated(text)
        matrix = normalize(self._encoder.encode([text], prompt=self._query_prompt))
        query_vector: Vector = matrix[0]
        return query_vector

    def _warn_if_truncated(self, text: str) -> None:
        size = self._encoder.count_tokens(text)
        if size > self._encoder.max_tokens:
            logger.warning(
                "text has %d tokens; the embedding model keeps only the first %d: %.60r",
                size,
                self._encoder.max_tokens,
                text,
            )


def normalize(vectors: Matrix) -> Matrix:
    """Scale each row to unit length; zero rows stay zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return (vectors / safe).astype(np.float32)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two normalized vectors."""
    return float(np.dot(a, b))
