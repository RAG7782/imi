"""Tests for the MCP server tools."""

import json

import numpy as np
import pytest

import imi.mcp_server as mcp_mod
from imi.mcp_server import (
    im_drm,
    im_enc,
    im_glnk,
    im_nav,
    im_sact,
    im_sts,
)
from imi.space import IMISpace


class DummyEmbedder:
    def embed(self, text):
        seed = abs(hash(text)) % (2**32)
        emb = np.random.default_rng(seed).normal(size=384).astype(np.float32)
        return emb / np.linalg.norm(emb)


class DummyLLM:
    def generate(self, system, prompt, max_tokens=1024, temperature=None):
        lowered = system.lower()
        if "valid json" in lowered and "salience" in lowered:
            return '{"salience": 0.5, "valence": 0.0, "arousal": 0.5}'
        if "json array" in lowered:
            return "[]"
        return prompt[: max(1, min(max_tokens, 80))]


@pytest.fixture(autouse=True)
def reset_space(tmp_path, monkeypatch):
    """Reset global space and use temp db before each test."""
    monkeypatch.setenv("IMI_DB", str(tmp_path / "test.db"))
    from imi.integrations.fcm_security import SecureFCMBridge

    monkeypatch.setattr(SecureFCMBridge, "emit_encode", lambda self, node: None)
    mcp_mod._space = IMISpace.from_sqlite(
        tmp_path / "test.db",
        embedder=DummyEmbedder(),
        llm=DummyLLM(),
    )
    mcp_mod._space_loaded_at = 10**12
    yield
    mcp_mod._space = None


class TestImiStats:
    def test_stats_empty(self):
        result = json.loads(im_sts())
        assert result["total_memories"] == 0
        assert result["graph"]["total_edges"] == 0

    def test_stats_after_encode(self):
        im_enc("DNS failure at 03:00")
        result = json.loads(im_sts())
        assert result["episodic_count"] == 1


class TestImiEncode:
    def test_encode_basic(self):
        result = json.loads(im_enc("Auth cascade across services"))
        assert result["id"]
        assert result["summary"]
        assert result["mass"] > 0

    def test_encode_with_tags(self):
        result = json.loads(im_enc("Database connection timeout", tags="db,timeout", source="test"))
        assert "db" in result["tags"]
        assert "timeout" in result["tags"]


class TestImiNavigate:
    def test_navigate_empty(self):
        result = json.loads(im_nav("auth failures"))
        assert result["hits"] == 0
        assert result["intent"] in ("TEMPORAL", "EXPLORATORY", "ACTION", "DEFAULT")

    def test_navigate_finds_memory(self):
        im_enc("DNS failure caused authentication outage")
        result = json.loads(im_nav("DNS issues"))
        assert result["hits"] >= 1
        assert result["memories"][0]["score"] > 0

    def test_navigate_adaptive_temporal(self):
        im_enc("recent auth failure event")
        result = json.loads(im_nav("recent failures"))
        assert result["intent"] == "TEMPORAL"
        assert result["relevance_weight_used"] == 0.15

    def test_navigate_adaptive_exploratory(self):
        result = json.loads(im_nav("find all incidents"))
        assert result["intent"] == "EXPLORATORY"
        assert result["relevance_weight_used"] == 0.0


class TestImiDream:
    def test_dream_empty(self):
        result = json.loads(im_drm())
        assert result["nodes_processed"] == 0


class TestImiSearchActions:
    def test_search_actions_empty(self):
        result = json.loads(im_sact("restart"))
        assert result["results"] == []


class TestImiGraphLink:
    def test_graph_link(self):
        r1 = json.loads(im_enc("DNS failure"))
        r2 = json.loads(im_enc("Auth cascade"))
        result = json.loads(im_glnk(r1["id"], r2["id"], "causal", "caused"))
        assert result["status"] == "ok"
        assert result["total_edges"] >= 1
