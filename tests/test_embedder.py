"""Tests for embedding provider selection."""

from __future__ import annotations

import json

import numpy as np

from imi.embedder import OllamaEmbedder, SentenceTransformerEmbedder, create_embedder_from_env
from imi.space import IMISpace


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"embeddings": [[3.0, 4.0, 0.0]]}).encode()


def test_create_embedder_defaults_to_sentence_transformer(monkeypatch):
    monkeypatch.delenv("IMI_EMBEDDER_PROVIDER", raising=False)
    embedder = create_embedder_from_env()
    assert isinstance(embedder, SentenceTransformerEmbedder)


def test_create_embedder_uses_ollama_env(monkeypatch):
    monkeypatch.setenv("IMI_EMBEDDER_PROVIDER", "ollama")
    monkeypatch.setenv("IMI_EMBEDDER_MODEL", "all-minilm")
    monkeypatch.setenv("IMI_OLLAMA_BASE_URL", "http://ollama.local:11434/v1")
    monkeypatch.setenv("IMI_EMBEDDER_MAX_CHARS", "12")

    embedder = create_embedder_from_env()

    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model_name == "all-minilm"
    assert embedder.base_url == "http://ollama.local:11434"
    assert embedder.max_chars == 12


def test_ollama_embedder_normalizes_vectors(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode())
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    embedder = OllamaEmbedder(
        model_name="all-minilm",
        base_url="http://localhost:11434/v1",
        max_chars=5,
    )

    vector = embedder.embed("abcdefghij")

    assert captured["url"] == "http://localhost:11434/api/embed"
    assert captured["payload"] == {"model": "all-minilm", "input": "abcde"}
    np.testing.assert_allclose(vector, np.array([0.6, 0.8, 0.0], dtype=np.float32))


def test_space_default_embedder_honors_env(monkeypatch):
    monkeypatch.setenv("IMI_EMBEDDER_PROVIDER", "ollama")
    monkeypatch.setenv("IMI_EMBEDDER_MODEL", "all-minilm")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    space = IMISpace()

    assert isinstance(space.embedder, OllamaEmbedder)
