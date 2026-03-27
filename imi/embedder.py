"""Embedding adapter — generates vector representations of text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Protocol for any embedding backend."""

    @property
    def dimensions(self) -> int: ...

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class SentenceTransformerEmbedder:
    """Local embedding via sentence-transformers."""

    model_name: str = "all-MiniLM-L6-v2"
    _model: object = field(default=None, repr=False)
    _dims: int = field(default=0, repr=False)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dims = self._model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        self._load()
        return self._dims

    def embed(self, text: str) -> np.ndarray:
        self._load()
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self._load()
        return self._model.encode(texts, normalize_embeddings=True)
