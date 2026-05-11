"""Tests for IMI storage backends.

JSONBackend and SQLiteBackend tests run without any infrastructure.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from imi.events import CONSOLIDATE, ENCODE, MemoryEvent
from imi.graph import EdgeType
from imi.node import MemoryNode
from imi.reconsolidate import ReconsolidationEvent
from imi.space import IMISpace
from imi.storage import JSONBackend, SQLiteBackend
from imi.temporal import TemporalContext

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


class DummyEmbedder:
    def embed(self, text: str):
        emb = np.ones(384, dtype=np.float32)
        return emb / np.linalg.norm(emb)


class DummyLLM:
    def generate(self, system: str, prompt: str, max_tokens: int = 1024, temperature=None):
        return prompt[:max_tokens]


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
# SQLite Backend Tests
# ---------------------------------------------------------------------------


class TestSQLiteBackend:
    @pytest.fixture
    def backend(self, tmp_path):
        b = SQLiteBackend(tmp_path / "imi_test.db")
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
        filtered = backend.query_events(event_type=CONSOLIDATE)
        assert len(filtered) == 1

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

    def test_node_versioning(self, backend):
        node = make_node(0)
        backend.put_node("episodic", node)
        node.access_count = 5
        node.summary_medium = "updated medium"
        backend.put_node("episodic", node)
        latest = backend.get_node("episodic", node.id)
        assert latest.access_count == 5
        assert latest.summary_medium == "updated medium"
        history = backend.get_node_history("episodic", node.id)
        assert len(history) == 2

    def test_export_import_roundtrip(self, backend, tmp_path):
        nodes = [make_node(i) for i in range(3)]
        backend.put_nodes("episodic", nodes)
        backend.put_nodes("semantic", [make_node(10, id="sem_001")])
        backend.put_anchors({"node_001": [{"type": "fact", "reference": "test"}]})
        backend.put_temporal(
            {"test_0000": TemporalContext(timestamp=time.time(), session_id="s1")}
        )
        data = backend.export_all()

        backend2 = SQLiteBackend(tmp_path / "imi_test2.db")
        backend2.setup()
        backend2.import_all(data)
        assert len(backend2.get_all_nodes("episodic")) == 3
        assert len(backend2.get_all_nodes("semantic")) == 1
        assert "node_001" in backend2.get_anchors()
        backend2.close()

    def test_fts_search(self, backend):
        """Test FTS5 full-text search on seeds/summaries."""
        nodes = [
            make_node(0, id="deploy_node", seed="kubernetes deployment failed with OOM"),
            make_node(1, id="cache_node", seed="redis cache invalidation caused stale data"),
            make_node(2, id="deploy2_node", seed="deployment rollback after memory spike"),
        ]
        for n in nodes:
            backend.put_node("episodic", n)
        results = backend.search_fts("deployment")
        node_ids = [r[0] for r in results]
        assert "deploy_node" in node_ids
        assert "deploy2_node" in node_ids
        assert "cache_node" not in node_ids

    def test_wal_mode(self, backend):
        """Verify WAL mode is active."""
        conn = backend._get_conn()
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"


class TestIMISpacePersistence:
    def test_dirty_tracking_persists_only_marked_nodes(self, tmp_path):
        db_path = tmp_path / "space.db"
        space = IMISpace.from_sqlite(db_path, embedder=DummyEmbedder(), llm=DummyLLM())
        node = make_node(0, id="dirty_node")
        space.episodic.add(node)

        conn = space.backend._get_conn()
        assert conn.execute("SELECT count(*) FROM memory_nodes").fetchone()[0] == 1

        node.summary_medium = "updated via dirty tracking"
        space.mark_dirty("episodic", node.id)
        space.save()
        assert conn.execute("SELECT count(*) FROM memory_nodes").fetchone()[0] == 2

        space.save()
        assert conn.execute("SELECT count(*) FROM memory_nodes").fetchone()[0] == 2

    def test_graph_reconsolidation_and_annealing_roundtrip(self, tmp_path):
        db_path = tmp_path / "space.db"
        space = IMISpace.from_sqlite(db_path, embedder=DummyEmbedder(), llm=DummyLLM())
        space.graph.add_edge("a", "b", EdgeType.CAUSAL, weight=0.7, label="test")
        space.reconsolidation_log.append(ReconsolidationEvent(
            node_id="a",
            timestamp=123.0,
            context="ctx",
            changes=["changed"],
            previous_orbital="old",
            new_orbital="new",
        ))
        space.annealing.iteration = 3
        space.annealing.energy_history = [1.0, 0.8, 0.7]
        space.annealing.converged = True
        space.save()

        loaded = IMISpace.from_sqlite(db_path, embedder=DummyEmbedder(), llm=DummyLLM())
        assert loaded.graph.stats()["total_edges"] == 2
        assert len(loaded.reconsolidation_log) == 1
        assert loaded.reconsolidation_log[0].changes == ["changed"]
        assert loaded.annealing.iteration == 3
        assert loaded.annealing.energy_history == [1.0, 0.8, 0.7]
        assert loaded.annealing.converged is True
