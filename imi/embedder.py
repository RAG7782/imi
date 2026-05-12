"""Embedding adapter — generates vector representations of text.

Backends:
  - SentenceTransformerEmbedder: local via sentence-transformers (default)
  - OllamaEmbedder: via Ollama API (all-minilm, nomic-embed-text, etc.)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


def create_embedder_from_env() -> "Embedder":
    """Create the default embedder from environment configuration.

    Defaults remain local sentence-transformers for backwards compatibility.
    Set ``IMI_EMBEDDER_PROVIDER=ollama`` to use Ollama's local embedding API.
    """
    provider = os.getenv("IMI_EMBEDDER_PROVIDER", "sentence-transformers").strip().lower()
    if provider in {"ollama", "ollama-local"}:
        return OllamaEmbedder(
            model_name=os.getenv("IMI_EMBEDDER_MODEL", "all-minilm"),
            base_url=os.getenv("IMI_OLLAMA_BASE_URL", os.getenv("OLLAMA_BASE_URL", "")),
            max_chars=int(os.getenv("IMI_EMBEDDER_MAX_CHARS", "400")),
        )
    if provider in {"sentence-transformers", "sentence_transformers", "local", "st"}:
        return SentenceTransformerEmbedder(
            model_name=os.getenv("IMI_EMBEDDER_MODEL", "all-MiniLM-L6-v2")
        )
    raise ValueError(
        "Unsupported IMI_EMBEDDER_PROVIDER="
        f"{provider!r}; expected 'sentence-transformers' or 'ollama'"
    )


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


@dataclass
class OllamaEmbedder:
    """Embedding via Ollama API — no sentence-transformers dependency needed.

    Uses Ollama's /api/embed endpoint. Default model: all-minilm (384d).
    Note: all-minilm has a practical input limit of ~400 chars.

    Usage::

        embedder = OllamaEmbedder()  # defaults to localhost:11434
        vec = embedder.embed("hello world")

        # VPS alan:
        embedder = OllamaEmbedder(base_url="http://100.73.123.8:11434")
    """

    model_name: str = "all-minilm"
    base_url: str = ""
    _dims: int = field(default=0, repr=False)
    max_chars: int = 400

    def __post_init__(self):
        if not self.base_url:
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Strip /v1 suffix if present (Ollama native API doesn't use it).
        self.base_url = self.base_url.rstrip("/").removesuffix("/v1")

    def _request_embed(self, input_data: str | list[str]) -> list[list[float]]:
        """Call Ollama /api/embed endpoint."""
        payload = json.dumps(
            {
                "model": self.model_name,
                "input": input_data,
            }
        ).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise RuntimeError(f"Ollama returned no embeddings for model {self.model_name}")

        if self._dims == 0:
            self._dims = len(embeddings[0])

        return embeddings

    def _truncate(self, text: str) -> str:
        """Truncate to max_chars to avoid Ollama errors on long input."""
        if len(text) > self.max_chars:
            logger.debug("OllamaEmbedder: truncating %d chars to %d", len(text), self.max_chars)
            return text[: self.max_chars]
        return text

    @property
    def dimensions(self) -> int:
        if self._dims == 0:
            # Probe with a short string to discover dimensions
            self._request_embed("hello")
        return self._dims

    def embed(self, text: str) -> np.ndarray:
        text = self._truncate(text)
        vecs = self._request_embed(text)
        arr = np.array(vecs[0], dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        texts = [self._truncate(t) for t in texts]
        vecs = self._request_embed(texts)
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        arr /= norms
        return arr
