"""Tests for the FastAPI REST API."""

import pytest
from fastapi.testclient import TestClient

from imi.api import app


@pytest.fixture(autouse=True)
def reset_space(tmp_path, monkeypatch):
    """Reset global space and use temp db before each test."""
    import imi.api

    imi.api._space = None
    monkeypatch.setenv("IMI_DB", str(tmp_path / "test.db"))
    yield
    imi.api._space = None


@pytest.fixture
def client():
    return TestClient(app)


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestStats:
    def test_stats_empty(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_memories"] == 0
        assert data["graph"]["total_edges"] == 0


class TestEncode:
    def test_encode_basic(self, client):
        r = client.post(
            "/encode",
            json={
                "experience": "DNS failure at 03:00 caused auth cascade",
                "tags": ["dns", "auth"],
                "source": "test",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["id"]
        assert data["summary"]
        assert "dns" in data["tags"]
        assert data["total_memories"] == 1

    def test_encode_minimal(self, client):
        r = client.post("/encode", json={"experience": "simple event"})
        assert r.status_code == 200
        assert r.json()["id"]


class TestNavigate:
    def test_navigate_empty(self, client):
        r = client.post("/navigate", json={"query": "auth failures"})
        assert r.status_code == 200
        assert r.json()["hits"] == 0

    def test_navigate_with_data(self, client):
        client.post("/encode", json={"experience": "DNS failure caused outage"})
        r = client.post("/navigate", json={"query": "DNS issues", "top_k": 5})
        assert r.status_code == 200
        data = r.json()
        assert data["hits"] >= 1
        assert data["intent"] in ("TEMPORAL", "EXPLORATORY", "ACTION", "DEFAULT")

    def test_navigate_adaptive_rw(self, client):
        client.post("/encode", json={"experience": "recent auth failure"})
        r = client.post("/navigate", json={"query": "recent failures"})
        data = r.json()
        assert data["intent"] == "TEMPORAL"
        assert data["relevance_weight_used"] == 0.15


class TestDream:
    def test_dream_empty(self, client):
        r = client.post("/dream")
        assert r.status_code == 200
        data = r.json()
        assert data["nodes_processed"] == 0


class TestSearchActions:
    def test_search_actions_empty(self, client):
        r = client.post("/search-actions", json={"action_query": "restart"})
        assert r.status_code == 200
        assert r.json()["results"] == []


class TestGraphLink:
    def test_graph_link(self, client):
        # Encode two memories
        r1 = client.post("/encode", json={"experience": "DNS failure"})
        r2 = client.post("/encode", json={"experience": "Auth cascade"})
        id1 = r1.json()["id"]
        id2 = r2.json()["id"]

        r = client.post(
            "/graph/link",
            json={
                "source_id": id1,
                "target_id": id2,
                "edge_type": "causal",
                "label": "caused",
            },
        )
        assert r.status_code == 200
        assert r.json()["total_edges"] >= 1
