"""Tests for the MCP server tools."""

import json
import pytest

import imi.mcp_server as mcp_mod
from imi.mcp_server import (
    imi_encode,
    imi_navigate,
    imi_dream,
    imi_search_actions,
    imi_stats,
    imi_graph_link,
)


@pytest.fixture(autouse=True)
def reset_space(tmp_path, monkeypatch):
    """Reset global space and use temp db before each test."""
    mcp_mod._space = None
    monkeypatch.setenv("IMI_DB", str(tmp_path / "test.db"))
    yield
    mcp_mod._space = None


class TestImiStats:
    def test_stats_empty(self):
        result = json.loads(imi_stats())
        assert result["total_memories"] == 0
        assert result["graph"]["total_edges"] == 0

    def test_stats_after_encode(self):
        imi_encode("DNS failure at 03:00")
        result = json.loads(imi_stats())
        assert result["episodic_count"] == 1


class TestImiEncode:
    def test_encode_basic(self):
        result = json.loads(imi_encode("Auth cascade across services"))
        assert result["id"]
        assert result["summary"]
        assert result["mass"] > 0

    def test_encode_with_tags(self):
        result = json.loads(imi_encode(
            "Database connection timeout", tags="db,timeout", source="test"
        ))
        assert "db" in result["tags"]
        assert "timeout" in result["tags"]


class TestImiNavigate:
    def test_navigate_empty(self):
        result = json.loads(imi_navigate("auth failures"))
        assert result["hits"] == 0
        assert result["intent"] in ("TEMPORAL", "EXPLORATORY", "ACTION", "DEFAULT")

    def test_navigate_finds_memory(self):
        imi_encode("DNS failure caused authentication outage")
        result = json.loads(imi_navigate("DNS issues"))
        assert result["hits"] >= 1
        assert result["memories"][0]["score"] > 0

    def test_navigate_adaptive_temporal(self):
        imi_encode("recent auth failure event")
        result = json.loads(imi_navigate("recent failures"))
        assert result["intent"] == "TEMPORAL"
        assert result["relevance_weight_used"] == 0.15

    def test_navigate_adaptive_exploratory(self):
        result = json.loads(imi_navigate("find all incidents"))
        assert result["intent"] == "EXPLORATORY"
        assert result["relevance_weight_used"] == 0.0


class TestImiDream:
    def test_dream_empty(self):
        result = json.loads(imi_dream())
        assert result["nodes_processed"] == 0


class TestImiSearchActions:
    def test_search_actions_empty(self):
        result = json.loads(imi_search_actions("restart"))
        assert result["results"] == []


class TestImiGraphLink:
    def test_graph_link(self):
        r1 = json.loads(imi_encode("DNS failure"))
        r2 = json.loads(imi_encode("Auth cascade"))
        result = json.loads(imi_graph_link(r1["id"], r2["id"], "causal", "caused"))
        assert result["status"] == "ok"
        assert result["total_edges"] >= 1
