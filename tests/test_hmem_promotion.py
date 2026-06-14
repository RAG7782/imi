"""Tests for H-MEM tree promotion in consolidation (maintain.py §3.4).

Spec: ~/experimentos/specs/2026-06-14-imi-hmem-positional-index.md (§3.4, CHECK-1/2)

Promotion is the writer that POPULATES the tree recursive_retrieve() descends.
These tests verify the writer's invariants against fakes — no Ollama, no
Anthropic (the disabled-org trap from the prior session), no real DB:
  - a consolidated cluster's pattern node becomes a Trace-layer index node
    (layer=2) with child_ptrs = member ids;
  - each member gets parent_id = the pattern id (CHECK-1 bidirectional);
  - acyclicity holds by construction (index→member only);
  - the dirty_sink is called for every re-parented member (silent-write-loss
    guard — the new parent_id must survive a backend save);
  - the flag gates everything: OFF means zero tree mutation (default).
"""

import os

import numpy as np
import pytest

from imi.maintain import consolidate
from imi.node import MemoryNode
from imi.store import VectorStore


class _FakeEmbedder:
    """Deterministic 2-D embedder: never touches Ollama."""

    def embed(self, text: str) -> np.ndarray:
        # Stable pseudo-vector from the text length — enough for the store search.
        v = np.array([1.0, float(len(text) % 7)], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v


class _FakeLLM:
    """Returns a fixed pattern string: never touches Anthropic/phi4."""

    def generate(self, system="", prompt="", max_tokens=0, **_) -> str:
        return "padrão emergente sintético"


def _ep(nid, vec, salience=0.9):
    from imi.affect import AffectiveTag

    n = MemoryNode(id=nid, seed=f"seed {nid}", summary_medium=f"sum {nid}")
    n.embedding = np.array(vec, dtype=np.float32)
    n.affect = AffectiveTag(salience=salience)
    return n


@pytest.fixture(autouse=True)
def _promote_on(monkeypatch):
    """Enable promotion for these tests (default is OFF in production)."""
    monkeypatch.setenv("IMI_HMEM_PROMOTE", "1")
    # Keep cluster under the LLM-synthesis threshold so _FakeLLM is enough.
    monkeypatch.setenv("IMI_CONSOLIDATION_MIN_CLUSTER", "99")


def test_cluster_becomes_index_node():
    """A consolidated cluster yields a Trace-layer pattern node with child_ptrs."""
    members = [_ep("ep1", [1.0, 0.0]), _ep("ep2", [1.0, 0.0])]
    semantic = VectorStore()
    consolidate([members], semantic, _FakeEmbedder(), _FakeLLM())

    assert len(semantic.nodes) == 1
    idx = semantic.nodes[0]
    assert idx.layer == 2, "pattern node must be promoted to Trace layer (L2)"
    assert set(idx.child_ptrs) == {"ep1", "ep2"}


def test_members_get_parent_id():
    """CHECK-1: each member's parent_id == the index node whose child_ptrs holds it."""
    members = [_ep("a", [1.0, 0.0]), _ep("b", [1.0, 0.0])]
    semantic = VectorStore()
    consolidate([members], semantic, _FakeEmbedder(), _FakeLLM())

    idx = semantic.nodes[0]
    for m in members:
        assert m.parent_id == idx.id
        assert m.id in idx.child_ptrs  # bidirectional consistency


def test_dirty_sink_called_for_every_member():
    """Silent-write-loss guard: re-parented members must be flagged dirty."""
    members = [_ep("x", [1.0, 0.0]), _ep("y", [1.0, 0.0])]
    semantic = VectorStore()
    marked = []
    consolidate([members], semantic, _FakeEmbedder(), _FakeLLM(),
                dirty_sink=lambda n: marked.append(n.id))

    assert set(marked) >= {"x", "y"}, "both members must be marked dirty for save"


def test_acyclic_by_construction():
    """Members never receive child_ptrs — no cycle can form (writer's guarantee)."""
    members = [_ep("m1", [1.0, 0.0]), _ep("m2", [1.0, 0.0])]
    semantic = VectorStore()
    consolidate([members], semantic, _FakeEmbedder(), _FakeLLM())

    for m in members:
        assert m.child_ptrs == [], "an episode (leaf) must have no children"
    # The only downward edge is index → members; members point up via parent_id.


def test_flag_off_means_no_mutation(monkeypatch):
    """Default OFF: consolidation runs but writes NO tree pointers."""
    monkeypatch.setenv("IMI_HMEM_PROMOTE", "0")
    members = [_ep("p", [1.0, 0.0]), _ep("q", [1.0, 0.0])]
    semantic = VectorStore()
    consolidate([members], semantic, _FakeEmbedder(), _FakeLLM())

    idx = semantic.nodes[0]
    assert idx.layer == 3, "no promotion → pattern stays at default Episode layer"
    assert idx.child_ptrs == []
    assert all(m.parent_id is None for m in members)


def test_promotion_survives_save_and_reload(tmp_path):
    """Silent-write-loss regression: the promoted pattern node's layer/child_ptrs
    AND members' parent_id must survive a real backend save → reload.

    Found 2026-06-14 by the Passo 7 harness: the fresh-pattern branch wired the
    tree in memory but never re-marked the pattern node dirty, so save()'s
    dirty-only path persisted it at layer=3 (its add()-time value) and dropped the
    promotion. A copy with index_nodes=20 reloaded with index_nodes=0. This test
    is the net that makes that failure loud (it uses a REAL SQLite backend — an
    in-memory VectorStore would never have caught it).
    """
    import numpy as np

    from imi.affect import AffectiveTag
    from imi.space import IMISpace

    os.environ["IMI_HMEM_PROMOTE"] = "1"
    db = tmp_path / "promote_persist.db"
    space = IMISpace.from_sqlite(str(db))

    # Two near-identical episodes so find_clusters groups them.
    for nid in ("e1", "e2", "e3"):
        n = MemoryNode(id=nid, seed=f"seed {nid}", summary_medium=f"sum {nid}")
        n.embedding = np.array([1.0, 0.0], dtype=np.float32)
        n.affect = AffectiveTag(salience=0.9)
        space.episodic.add(n)

    from imi.maintain import find_clusters

    clusters = find_clusters(space.episodic, similarity_threshold=0.5)
    consolidate(clusters, space.semantic, space.embedder, _FakeLLM(),
                dirty_sink=space.mark_node_dirty)
    space.save()

    # Fresh reload from the same DB file — the real round-trip.
    reloaded = IMISpace.from_sqlite(str(db))
    idx = [n for n in reloaded.semantic.nodes if n.layer < 3 and n.child_ptrs]
    reparented = [n for n in reloaded.episodic.nodes if n.parent_id]
    assert idx, "promoted index node must survive reload (layer<3 + child_ptrs)"
    assert reparented, "re-parented episodes must survive reload (parent_id set)"
    # The index node's children must still resolve to the reparented episodes.
    assert set(idx[0].child_ptrs) == {n.id for n in reparented}
