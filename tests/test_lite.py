"""Tests for IMI Lite-B (ZoomRAG wrapper over ChromaDB)."""

from __future__ import annotations

import pytest

from imi.lite import ZoomRAG


@pytest.fixture
def zr():
    return ZoomRAG()


SAMPLE_MEMORIES = [
    {
        "text": "OAuth token refresh failed silently causing 401 cascade across microservices",
        "orbital": "auth token failure",
        "medium": "OAuth token refresh failed silently, 401 cascade",
        "detailed": "OAuth token refresh failed silently causing 401 cascade across all microservices. Root cause: refresh token rotation race condition.",
        "seed": "oauth refresh silent fail → 401 cascade → race condition in token rotation",
        "affordances": [
            {"action": "add retry logic to token refresh", "confidence": 0.9, "conditions": "when using OAuth", "domain": "auth"},
            {"action": "monitor 401 error rates", "confidence": 0.8, "conditions": "always", "domain": "monitoring"},
        ],
        "tags": ["auth", "incident"],
    },
    {
        "text": "PostgreSQL vacuum blocked by analytics query, table bloat 400GB",
        "orbital": "postgres vacuum blocked",
        "medium": "PostgreSQL vacuum blocked by analytics query, 400GB bloat",
        "detailed": "PostgreSQL vacuum job blocked by long-running analytics query. Table bloat reached 400GB before detection.",
        "seed": "pg vacuum blocked → analytics query → 400GB table bloat",
        "affordances": [
            {"action": "set statement_timeout for analytics queries", "confidence": 0.85, "conditions": "analytics workloads", "domain": "database"},
        ],
        "tags": ["database", "incident"],
    },
    {
        "text": "Kubernetes pod OOM killed: memory limit 256Mi but Java heap needed 512Mi",
        "orbital": "k8s OOM kill",
        "medium": "K8s pod OOM killed: memory limit 256Mi, Java heap 512Mi",
        "detailed": "Pod repeatedly OOM killed. Memory limit was 256Mi but Java heap alone needed 512Mi. JVM flags were not aligned with k8s resource limits.",
        "seed": "pod OOM → 256Mi limit vs 512Mi heap → JVM flags misaligned",
        "affordances": [
            {"action": "align JVM heap flags with k8s resource limits", "confidence": 0.95, "conditions": "Java on k8s", "domain": "infrastructure"},
            {"action": "add OOM kill alerting", "confidence": 0.7, "conditions": "always", "domain": "monitoring"},
        ],
        "tags": ["kubernetes", "incident"],
    },
]


class TestZoomRAG:
    def test_ingest_and_count(self, zr):
        for mem in SAMPLE_MEMORIES:
            zr.ingest(
                mem["text"],
                summary_orbital=mem["orbital"],
                summary_medium=mem["medium"],
                summary_detailed=mem["detailed"],
                seed=mem["seed"],
                affordances=mem["affordances"],
                tags=mem["tags"],
            )
        assert zr.count == 3

    def test_search_zoom_levels(self, zr):
        for mem in SAMPLE_MEMORIES:
            zr.ingest(mem["text"], summary_orbital=mem["orbital"],
                      summary_medium=mem["medium"], summary_detailed=mem["detailed"],
                      seed=mem["seed"], tags=mem["tags"])

        # Orbital zoom should return short content
        results = zr.search("authentication token", zoom="orbital")
        assert len(results) > 0
        assert len(results[0]["content"]) <= 50

        # Detailed zoom should return longer content
        results_d = zr.search("authentication token", zoom="detailed")
        assert len(results_d) > 0
        assert len(results_d[0]["content"]) > len(results[0]["content"])

    def test_search_relevance(self, zr):
        for mem in SAMPLE_MEMORIES:
            zr.ingest(mem["text"], summary_medium=mem["medium"], tags=mem["tags"])

        results = zr.search("database vacuum bloat", zoom="medium")
        assert results[0]["id"] is not None
        # Top result should be database-related
        assert "postgres" in results[0]["content"].lower() or "vacuum" in results[0]["content"].lower()

    def test_search_actions(self, zr):
        for mem in SAMPLE_MEMORIES:
            zr.ingest(mem["text"], affordances=mem["affordances"], tags=mem["tags"])

        actions = zr.search_actions("fix memory issues in kubernetes")
        assert len(actions) > 0
        # Top action should be k8s/JVM related
        top = actions[0]
        assert "action" in top
        assert "score" in top
        assert top["score"] > 0

    def test_search_actions_empty(self):
        fresh_zr = ZoomRAG()
        fresh_zr.ingest("some text without affordances")
        actions = fresh_zr.search_actions("anything")
        assert actions == []

    def test_fallback_zoom_levels(self, zr):
        """When no summaries provided, uses text truncation."""
        long_text = "A" * 500
        zr.ingest(long_text)
        results = zr.search("A" * 10, zoom="orbital")
        assert len(results) > 0
        assert len(results[0]["content"]) == 40  # truncated

    def test_custom_node_id(self, zr):
        nid = zr.ingest("test memory", node_id="custom_123")
        assert nid == "custom_123"
        results = zr.search("test")
        assert results[0]["id"] == "custom_123"

    def test_tags_roundtrip(self, zr):
        zr.ingest("test", tags=["tag1", "tag2"])
        results = zr.search("test")
        assert results[0]["tags"] == ["tag1", "tag2"]
