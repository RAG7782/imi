"""Tests for the Continuity Canary (silent-drift detector).

Spec: ~/experimentos/specs/2026-06-14-imi-hmem-positional-index.md (CANÁRIO)

The canary itself must not regress — a broken detector is worse than none
(it reports "ok" while drift happens). These tests use a tiny in-memory
IMISpace with known nodes, so they run offline (no Ollama, no production DB).
"""

import pytest

from imi.canary import Anchor, CanaryReport, run_canary, load_anchors


@pytest.fixture
def space_with_known_nodes(tmp_path):
    """A small SQLite-backed space with two distinctively-tokened nodes."""
    from imi.space import IMISpace
    from imi.node import MemoryNode

    space = IMISpace.from_sqlite(str(tmp_path / "canary_test.db"))
    # Two nodes with rare, distinctive tokens in their text.
    n1 = MemoryNode(id="alpha0000001", seed="zzqxuniquetokenalpha experiment one")
    n2 = MemoryNode(id="beta00000002", seed="wwvyuniquetokenbeta experiment two")
    # VectorStore.add persists through the attached backend (FTS index included).
    space.episodic.add(n1)
    space.episodic.add(n2)
    return space


class TestCanaryReport:
    def test_empty_anchors_is_unavailable(self):
        report = run_canary(space=None, anchors=[])
        assert report.status == "unavailable"
        assert "no anchors" in report.reason

    def test_report_hit_rate(self):
        r = CanaryReport(total=4, hits=3)
        assert r.hit_rate == 0.75


class TestCanaryRun:
    def test_all_anchors_hit_is_ok(self, space_with_known_nodes):
        anchors = [
            Anchor(token="zzqxuniquetokenalpha", expected_id="alpha0000001"),
            Anchor(token="wwvyuniquetokenbeta", expected_id="beta00000002"),
        ]
        report = run_canary(space_with_known_nodes, anchors, top_k=5)
        assert report.status == "ok"
        assert report.hits == 2
        assert report.misses == []

    def test_missing_target_is_drift(self, space_with_known_nodes):
        """An anchor whose expected_id is wrong → drift (the LOUD signal)."""
        anchors = [
            Anchor(token="zzqxuniquetokenalpha", expected_id="WRONGID00000"),
        ]
        report = run_canary(space_with_known_nodes, anchors, top_k=5)
        assert report.status == "drift"
        assert len(report.misses) == 1
        assert report.misses[0]["expected_id"] == "WRONGID00000"


class TestAnchorPersistence:
    def test_load_anchors_absent_file_returns_empty(self, tmp_path):
        assert load_anchors(tmp_path / "nonexistent.json") == []

    def test_load_anchors_roundtrip(self, tmp_path):
        import json
        p = tmp_path / "anchors.json"
        p.write_text(json.dumps([{"token": "t1", "expected_id": "id1", "note": "n"}]))
        anchors = load_anchors(p)
        assert len(anchors) == 1
        assert anchors[0].token == "t1"
        assert anchors[0].expected_id == "id1"
