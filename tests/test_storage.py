"""Tests for IMI storage backends.

JSONBackend tests run without any infrastructure.
TimescaleDBBackend tests require docker-compose up and are marked @integration.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from imi.events import ENCODE, CONSOLIDATE, MemoryEvent
from imi.node import MemoryNode
from imi.storage import JSONBackend, TimescaleDBBackend
from imi.temporal import TemporalContext

TSDB_CONN = "postgresql://imi:imi_dev@localhost:5433/imi"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node(i: int = 0, **overrides) -> MemoryNode:
    """Create a minimal but realistic MemoryNode for testing."""
    emb = np.random.randn(384).astype(np.float32)
    emb /= np.linalg.norm(emb)
    defaults = dict(
        id=f"test_{i:04d}",
        seed=f"Test memory node {i}",
        summary_orbital=f"test {i}",
        summary_medium=f"test node {i} medium",
        summary_detailed=f"test node {i} detailed summary",
        embedding=emb,
        tags=["test", f"batch_{i // 10}"],
        source="test",
        mass=1.0,
        created_at=time.time() - (i * 3600),  # spread over hours
    )
    defaults.update(overrides)
    return MemoryNode(**defaults)


# ---------------------------------------------------------------------------
# JSONBackend Tests
# ---------------------------------------------------------------------------


class TestJSONBackend:
    @pytest.fixture
    def backend(self, tmp_path):
        b = JSONBackend(tmp_path / "imi_test")
        b.setup()
        return b

    def test_put_get_node(self, backend):
        node = make_node(0)
        backend.put_node("episodic", node)
        loaded = backend.get_node("episodic", node.id)
        assert loaded is not None
        assert loaded.id == node.id
        assert loaded.seed == node.seed
        assert loaded.tags == node.tags

    def test_put_nodes_get_all(self, backend):
        nodes = [make_node(i) for i in range(5)]
        backend.put_nodes("episodic", nodes)
        loaded = backend.get_all_nodes("episodic")
        assert len(loaded) == 5
        assert {n.id for n in loaded} == {n.id for n in nodes}

    def test_remove_node(self, backend):
        nodes = [make_node(i) for i in range(3)]
        backend.put_nodes("episodic", nodes)
        backend.remove_node("episodic", nodes[1].id)
        loaded = backend.get_all_nodes("episodic")
        assert len(loaded) == 2
        assert nodes[1].id not in {n.id for n in loaded}

    def test_separate_stores(self, backend):
        ep = make_node(0, id="ep_001")
        sem = make_node(1, id="sem_001")
        backend.put_node("episodic", ep)
        backend.put_node("semantic", sem)
        assert len(backend.get_all_nodes("episodic")) == 1
        assert len(backend.get_all_nodes("semantic")) == 1
        assert backend.get_node("episodic", "sem_001") is None
        assert backend.get_node("semantic", "ep_001") is None

    def test_anchors(self, backend):
        anchors = {
            "node_001": [{"type": "fact", "reference": "sky is blue"}],
            "node_002": [
                {"type": "file", "reference": "/tmp/test.py"},
                {"type": "date", "reference": "2026-03-27"},
            ],
        }
        backend.put_anchors(anchors)
        loaded = backend.get_anchors()
        assert loaded == anchors

    def test_temporal(self, backend):
        contexts = {
            "node_001": TemporalContext(
                timestamp=time.time(),
                session_id="session_1",
                sequence_pos=1,
                temporal_neighbors=["node_002"],
            ),
            "node_002": TemporalContext(
                timestamp=time.time() + 60,
                session_id="session_1",
                sequence_pos=2,
                temporal_neighbors=["node_001"],
            ),
        }
        backend.put_temporal(contexts)
        loaded = backend.get_temporal()
        assert len(loaded) == 2
        assert loaded["node_001"].session_id == "session_1"
        assert loaded["node_002"].temporal_neighbors == ["node_001"]

    def test_events(self, backend):
        evt = MemoryEvent(
            event_type=ENCODE,
            node_id="node_001",
            store_name="episodic",
            metadata={"tags": ["test"]},
        )
        backend.log_event(evt)
        backend.log_event(
            MemoryEvent(
                event_type=CONSOLIDATE,
                node_id="pattern_001",
                store_name="semantic",
            )
        )
        events = backend.query_events()
        assert len(events) == 2
        assert events[0].event_type == ENCODE

        filtered = backend.query_events(event_type=CONSOLIDATE)
        assert len(filtered) == 1

    def test_export_import_roundtrip(self, backend, tmp_path):
        nodes = [make_node(i) for i in range(3)]
        backend.put_nodes("episodic", nodes)
        backend.put_nodes("semantic", [make_node(10, id="sem_001")])
        backend.put_anchors({"node_001": [{"type": "fact", "reference": "test"}]})
        backend.put_temporal(
            {"test_0000": TemporalContext(timestamp=time.time(), session_id="s1")}
        )

        data = backend.export_all()

        backend2 = JSONBackend(tmp_path / "imi_test2")
        backend2.setup()
        backend2.import_all(data)

        assert len(backend2.get_all_nodes("episodic")) == 3
        assert len(backend2.get_all_nodes("semantic")) == 1
        assert "node_001" in backend2.get_anchors()

    def test_embedding_fidelity(self, backend):
        """Verify embeddings survive serialize/deserialize unchanged."""
        node = make_node(0)
        original_emb = node.embedding.copy()
        backend.put_node("episodic", node)
        loaded = backend.get_node("episodic", node.id)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded.embedding, original_emb, decimal=6)

    def test_query_by_time_range(self, backend):
        now = time.time()
        nodes = [
            make_node(0, id="old", created_at=now - 86400 * 7),
            make_node(1, id="recent", created_at=now - 3600),
            make_node(2, id="very_old", created_at=now - 86400 * 30),
        ]
        backend.put_nodes("episodic", nodes)
        # Query last 2 days
        result = backend.query_by_time_range(now - 86400 * 2, now, "episodic")
        assert len(result) == 1
        assert result[0].id == "recent"

    def test_query_by_session(self, backend):
        backend.put_temporal(
            {
                "n1": TemporalContext(timestamp=time.time(), session_id="s1"),
                "n2": TemporalContext(timestamp=time.time(), session_id="s2"),
                "n3": TemporalContext(timestamp=time.time(), session_id="s1"),
            }
        )
        result = backend.query_by_session("s1")
        assert set(result) == {"n1", "n3"}


# ---------------------------------------------------------------------------
# TimescaleDB Backend Tests
# ---------------------------------------------------------------------------


def _tsdb_available() -> bool:
    try:
        import psycopg
        with psycopg.connect(TSDB_CONN, autocommit=True) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture
def tsdb_backend():
    if not _tsdb_available():
        pytest.skip("TimescaleDB not available at localhost:5433")
    backend = TimescaleDBBackend(TSDB_CONN)
    backend.setup()
    # Clean test data
    with backend.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_nodes")
            cur.execute("DELETE FROM memory_events")
            cur.execute("DELETE FROM temporal_contexts")
            cur.execute("DELETE FROM anchors")
        conn.commit()
    yield backend
    backend.close()


class TestTimescaleDBBackend:
    def test_put_get_node(self, tsdb_backend):
        node = make_node(0)
        tsdb_backend.put_node("episodic", node)
        loaded = tsdb_backend.get_node("episodic", node.id)
        assert loaded is not None
        assert loaded.id == node.id
        assert loaded.seed == node.seed

    def test_put_nodes_get_all(self, tsdb_backend):
        nodes = [make_node(i) for i in range(5)]
        tsdb_backend.put_nodes("episodic", nodes)
        loaded = tsdb_backend.get_all_nodes("episodic")
        assert len(loaded) == 5

    def test_remove_node(self, tsdb_backend):
        nodes = [make_node(i) for i in range(3)]
        tsdb_backend.put_nodes("episodic", nodes)
        tsdb_backend.remove_node("episodic", nodes[1].id)
        loaded = tsdb_backend.get_all_nodes("episodic")
        assert len(loaded) == 2

    def test_node_versioning(self, tsdb_backend):
        node = make_node(0)
        tsdb_backend.put_node("episodic", node)
        # Mutate and save again
        node.access_count = 5
        node.summary_medium = "updated medium"
        tsdb_backend.put_node("episodic", node)
        # Get latest
        latest = tsdb_backend.get_node("episodic", node.id)
        assert latest.access_count == 5
        assert latest.summary_medium == "updated medium"
        # Get history
        history = tsdb_backend.get_node_history("episodic", node.id)
        assert len(history) == 2

    def test_embedding_fidelity(self, tsdb_backend):
        node = make_node(0)
        original_emb = node.embedding.copy()
        tsdb_backend.put_node("episodic", node)
        loaded = tsdb_backend.get_node("episodic", node.id)
        assert loaded is not None
        np.testing.assert_array_almost_equal(
            loaded.embedding, original_emb, decimal=5
        )

    def test_time_range_query(self, tsdb_backend):
        now = time.time()
        nodes = [
            make_node(0, id="old", created_at=now - 86400 * 7),
            make_node(1, id="recent", created_at=now - 3600),
            make_node(2, id="very_old", created_at=now - 86400 * 30),
        ]
        tsdb_backend.put_nodes("episodic", nodes)
        result = tsdb_backend.query_by_time_range(now - 86400 * 2, now, "episodic")
        assert len(result) == 1
        assert result[0].id == "recent"

    def test_events(self, tsdb_backend):
        tsdb_backend.log_event(
            MemoryEvent(event_type=ENCODE, node_id="n1", store_name="episodic")
        )
        tsdb_backend.log_event(
            MemoryEvent(event_type=CONSOLIDATE, node_id="p1", store_name="semantic")
        )
        events = tsdb_backend.query_events()
        assert len(events) == 2
        filtered = tsdb_backend.query_events(event_type=ENCODE)
        assert len(filtered) == 1

    def test_temporal(self, tsdb_backend):
        contexts = {
            "n1": TemporalContext(
                timestamp=time.time(),
                session_id="s1",
                sequence_pos=1,
                temporal_neighbors=["n2"],
            ),
            "n2": TemporalContext(
                timestamp=time.time() + 60,
                session_id="s1",
                sequence_pos=2,
                temporal_neighbors=["n1"],
            ),
        }
        tsdb_backend.put_temporal(contexts)
        loaded = tsdb_backend.get_temporal()
        assert len(loaded) == 2
        assert loaded["n1"].session_id == "s1"

    def test_session_query(self, tsdb_backend):
        tsdb_backend.put_temporal(
            {
                "n1": TemporalContext(timestamp=time.time(), session_id="s1"),
                "n2": TemporalContext(timestamp=time.time(), session_id="s2"),
                "n3": TemporalContext(timestamp=time.time(), session_id="s1"),
            }
        )
        result = tsdb_backend.query_by_session("s1")
        assert set(result) == {"n1", "n3"}

    def test_export_import(self, tsdb_backend):
        nodes = [make_node(i) for i in range(3)]
        tsdb_backend.put_nodes("episodic", nodes)
        tsdb_backend.put_anchors(
            {"test_0000": [{"type": "fact", "reference": "test"}]}
        )
        data = tsdb_backend.export_all()
        assert len(data["episodic"]) == 3
        assert "test_0000" in data["anchors"]
