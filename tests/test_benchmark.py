"""IMI-specific benchmark harness.

Measures real IMI storage operations, not generic ops/s.
Compares JSONBackend vs TimescaleDBBackend on the same workloads.

Usage:
    pytest tests/test_benchmark.py -v -s
    python tests/test_benchmark.py          # standalone mode
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from imi.events import MemoryEvent
from imi.node import MemoryNode
from imi.storage import JSONBackend, StorageBackend, TimescaleDBBackend
from imi.temporal import TemporalContext

TSDB_CONN = "postgresql://imi:imi_dev@localhost:5433/imi"


def make_node(i: int) -> MemoryNode:
    """Create a realistic MemoryNode for benchmarking."""
    emb = np.random.randn(384).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return MemoryNode(
        id=f"bench_{i:06d}",
        seed=f"Benchmark memory node {i} with representative content length "
        f"that simulates a real seed of approximately 80 tokens",
        summary_orbital=f"bench {i}",
        summary_medium=f"benchmark node {i} medium summary with more detail",
        summary_detailed=(
            f"benchmark node {i} detailed summary with technical content "
            f"that approximates the real 100-token detailed view"
        ),
        embedding=emb,
        tags=["benchmark", f"batch_{i // 100}"],
        source="benchmark",
        mass=0.5 + (i % 10) * 0.05,
        created_at=time.time() - (i * 60),  # 1 minute apart
    )


class IMIBenchmarkHarness:
    """Benchmark harness measuring real IMI storage patterns."""

    def __init__(self, backend: StorageBackend, label: str = "") -> None:
        self.backend = backend
        self.label = label
        self.results: dict[str, float | int] = {}

    def bench_encode_persist(self, n: int = 100) -> float:
        """Single-node INSERT latency (the hot path)."""
        nodes = [make_node(i) for i in range(n)]
        t0 = time.perf_counter()
        for node in nodes:
            self.backend.put_node("episodic", node)
        elapsed = time.perf_counter() - t0
        self.results["encode_persist_total_ms"] = elapsed * 1000
        self.results["encode_persist_avg_ms"] = (elapsed / n) * 1000
        self.results["encode_persist_ops_per_s"] = n / elapsed
        return elapsed

    def bench_bulk_save(self, n: int = 1000) -> float:
        """Full save (all nodes at once, like IMISpace.save())."""
        nodes = [make_node(i + 10000) for i in range(n)]
        t0 = time.perf_counter()
        self.backend.put_nodes("semantic", nodes)
        elapsed = time.perf_counter() - t0
        self.results["bulk_save_ms"] = elapsed * 1000
        self.results["bulk_save_nodes"] = n
        self.results["bulk_save_nodes_per_s"] = n / elapsed
        return elapsed

    def bench_load_all(self) -> float:
        """Reconstruct all nodes (like IMISpace.load())."""
        t0 = time.perf_counter()
        nodes = self.backend.get_all_nodes("episodic")
        elapsed = time.perf_counter() - t0
        self.results["load_all_ms"] = elapsed * 1000
        self.results["load_all_count"] = len(nodes)
        return elapsed

    def bench_time_range_query(self, window_hours: float = 24.0) -> float:
        """Time-range query (like navigate_temporal)."""
        now = time.time()
        t0 = time.perf_counter()
        nodes = self.backend.query_by_time_range(
            now - window_hours * 3600, now, "episodic"
        )
        elapsed = time.perf_counter() - t0
        self.results["time_range_ms"] = elapsed * 1000
        self.results["time_range_hits"] = len(nodes)
        return elapsed

    def bench_event_log(self, n: int = 500) -> float:
        """Event logging throughput (dreaming cycle)."""
        t0 = time.perf_counter()
        for i in range(n):
            self.backend.log_event(
                MemoryEvent(
                    event_type="consolidate",
                    node_id=f"node_{i % 100:04d}",
                    store_name="semantic",
                    metadata={"source_count": 3, "strength": 0.8},
                )
            )
        elapsed = time.perf_counter() - t0
        self.results["event_log_total_ms"] = elapsed * 1000
        self.results["event_log_avg_ms"] = (elapsed / n) * 1000
        self.results["event_log_ops_per_s"] = n / elapsed
        return elapsed

    def bench_node_history(self) -> float:
        """Version history retrieval (after reconsolidation)."""
        node = make_node(99999)
        for v in range(10):
            node.access_count = v
            node.summary_medium = f"version {v} of this memory"
            self.backend.put_node("episodic", node)

        t0 = time.perf_counter()
        history = self.backend.get_node_history("episodic", node.id)
        elapsed = time.perf_counter() - t0
        self.results["node_history_ms"] = elapsed * 1000
        self.results["node_history_versions"] = len(history)
        return elapsed

    def bench_fidelity(self) -> bool:
        """Verify serialize/deserialize preserves data exactly."""
        node = make_node(88888)
        original_dict = node.to_dict()
        self.backend.put_node("episodic", node)
        loaded = self.backend.get_node("episodic", node.id)
        if loaded is None:
            self.results["fidelity"] = 0
            return False

        loaded_dict = loaded.to_dict()

        # Compare all fields except embedding (float precision)
        for key in original_dict:
            if key == "embedding":
                orig_emb = np.array(original_dict["embedding"], dtype=np.float32)
                load_emb = np.array(loaded_dict["embedding"], dtype=np.float32)
                if not np.allclose(orig_emb, load_emb, atol=1e-5):
                    self.results["fidelity"] = 0
                    self.results["fidelity_fail_field"] = "embedding"
                    return False
            elif original_dict[key] != loaded_dict.get(key):
                self.results["fidelity"] = 0
                self.results["fidelity_fail_field"] = key
                return False

        self.results["fidelity"] = 1
        return True

    def run_all(self) -> dict[str, float | int]:
        """Run all benchmarks and return results."""
        self.bench_encode_persist()
        self.bench_bulk_save()
        self.bench_load_all()
        self.bench_time_range_query()
        self.bench_event_log()
        self.bench_node_history()
        self.bench_fidelity()
        return self.results

    def print_report(self) -> None:
        """Pretty-print benchmark results."""
        print(f"\n{'=' * 60}")
        print(f"  IMI Benchmark: {self.label}")
        print(f"{'=' * 60}")
        for key, value in sorted(self.results.items()):
            if isinstance(value, float):
                print(f"  {key:40s} {value:>12.2f}")
            else:
                print(f"  {key:40s} {value:>12}")
        print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------


@pytest.fixture
def json_harness(tmp_path):
    backend = JSONBackend(tmp_path / "bench_json")
    backend.setup()
    return IMIBenchmarkHarness(backend, "JSONBackend")


def _tsdb_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(TSDB_CONN, autocommit=True) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture
def tsdb_harness():
    if not _tsdb_available():
        pytest.skip("TimescaleDB not available")
    backend = TimescaleDBBackend(TSDB_CONN)
    backend.setup()
    with backend.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_nodes")
            cur.execute("DELETE FROM memory_events")
            cur.execute("DELETE FROM temporal_contexts")
            cur.execute("DELETE FROM anchors")
        conn.commit()
    yield IMIBenchmarkHarness(backend, "TimescaleDBBackend")
    backend.close()


def test_benchmark_json(json_harness):
    results = json_harness.run_all()
    json_harness.print_report()
    assert results["fidelity"] == 1


def test_benchmark_tsdb(tsdb_harness):
    results = tsdb_harness.run_all()
    tsdb_harness.print_report()
    assert results["fidelity"] == 1


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("Running IMI Storage Benchmarks...\n")

    # JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        json_backend = JSONBackend(Path(tmpdir) / "bench")
        json_backend.setup()
        json_h = IMIBenchmarkHarness(json_backend, "JSONBackend")
        json_results = json_h.run_all()
        json_h.print_report()

    # TimescaleDB
    if _tsdb_available():
        tsdb_backend = TimescaleDBBackend(TSDB_CONN)
        tsdb_backend.setup()
        with tsdb_backend.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memory_nodes")
                cur.execute("DELETE FROM memory_events")
                cur.execute("DELETE FROM temporal_contexts")
                cur.execute("DELETE FROM anchors")
            conn.commit()
        tsdb_h = IMIBenchmarkHarness(tsdb_backend, "TimescaleDBBackend")
        tsdb_results = tsdb_h.run_all()
        tsdb_h.print_report()
        tsdb_backend.close()

        # Comparison
        print("\n" + "=" * 60)
        print("  COMPARISON: JSON vs TimescaleDB")
        print("=" * 60)
        for key in sorted(json_results.keys()):
            if key in tsdb_results and isinstance(json_results[key], (int, float)):
                j = json_results[key]
                t = tsdb_results[key]
                if j > 0 and isinstance(j, float):
                    ratio = t / j
                    winner = "JSON" if ratio > 1 else "TSDB"
                    print(f"  {key:40s}  JSON={j:>10.2f}  TSDB={t:>10.2f}  ({winner})")
        print("=" * 60)
    else:
        print("TimescaleDB not available — skipping TSDB benchmark.")
        print("Run: docker compose up -d   (from imi/ directory)")
