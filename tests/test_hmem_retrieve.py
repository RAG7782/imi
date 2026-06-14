"""Tests for H-MEM recursive retrieval (hmem_retrieve.py).

Spec: ~/experimentos/specs/2026-06-14-imi-hmem-positional-index.md (§3.3, §4)

These tests guard the recursive-descent invariants against a SYNTHETIC tree —
no DB writes, no embedder, no Ollama. Each test maps to a numbered pre-mortem
guard or a spec ASSERT/CHECK. They are the non-regression net (ASSERT-4) and
must run in CI, not once (pre-mortem cause #10).

Tree under test (a tiny 3-layer H-MEM):

    dom0 (L0 Domain) ── child_ptrs ─▶ cat0 (L1) ──▶ ep_a (L3), ep_b (L3)
                                       cat1 (L1) ──▶ ep_c (L3)
    orphan_x (L3, nobody's child)         ← must STILL be searchable (ASSERT-6)

Embeddings are hand-set unit vectors so sim(q, node) is fully deterministic —
the test never depends on a real embedder.
"""

import numpy as np
import pytest

from imi.hmem_retrieve import recursive_retrieve
from imi.node import MemoryNode


class _FakeStore:
    """Minimal VectorStore stand-in: just holds nodes and a .get(id)."""

    def __init__(self, nodes):
        self.nodes = nodes

    def get(self, node_id):
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None


def _node(nid, layer, vec, child_ptrs=None, parent_id=None):
    n = MemoryNode(id=nid, layer=layer, seed=nid)
    n.embedding = np.array(vec, dtype=np.float32)
    n.child_ptrs = child_ptrs or []
    n.parent_id = parent_id
    return n


@pytest.fixture
def tree_store():
    """A populated 3-layer tree + one orphan, all with deterministic vectors."""
    # 2-D unit vectors: ep_a points at [1,0], ep_c at [0,1].
    dom0 = _node("dom0", 0, [0.9, 0.1], child_ptrs=["cat0", "cat1"])
    cat0 = _node("cat0", 1, [0.95, 0.05], child_ptrs=["ep_a", "ep_b"], parent_id="dom0")
    cat1 = _node("cat1", 1, [0.1, 0.9], child_ptrs=["ep_c"], parent_id="dom0")
    ep_a = _node("ep_a", 3, [1.0, 0.0], parent_id="cat0")
    ep_b = _node("ep_b", 3, [0.8, 0.2], parent_id="cat0")
    ep_c = _node("ep_c", 3, [0.0, 1.0], parent_id="cat1")
    orphan_x = _node("orphan_x", 3, [0.7, 0.7])  # no parent, no children
    return _FakeStore([dom0, cat0, cat1, ep_a, ep_b, ep_c, orphan_x])


def test_descends_to_correct_episode(tree_store):
    """Query aligned with ep_a's branch returns ep_a at rank 1, via the tree."""
    q = np.array([1.0, 0.0], dtype=np.float32)
    res = recursive_retrieve(q, [tree_store], k_final=3)
    assert res.hits, "must return at least one hit"
    assert res.hits[0].node.id == "ep_a"
    assert res.hits[0].via == "tree"
    assert res.layers_descended >= 2  # Domain → Category → Episode


def test_confidence_is_surfaced(tree_store):
    """ASSERT-5: every hit carries a confidence weight in [0,1], not just rank."""
    q = np.array([0.0, 1.0], dtype=np.float32)
    res = recursive_retrieve(q, [tree_store], k_final=3)
    assert all(0.0 <= h.confidence <= 1.0 for h in res.hits)
    # ep_c is the [0,1] branch — should win this query.
    assert res.hits[0].node.id == "ep_c"


def test_orphan_is_never_lost(tree_store):
    """ASSERT-6: a query nearest the orphan still retrieves it (flat pool)."""
    q = np.array([0.7, 0.7], dtype=np.float32)  # exactly orphan_x
    res = recursive_retrieve(q, [tree_store], k_final=5)
    ids = [h.node.id for h in res.hits]
    assert "orphan_x" in ids
    assert res.orphan_pool_size >= 1
    orphan_hit = next(h for h in res.hits if h.node.id == "orphan_x")
    assert orphan_hit.via == "orphan"


def test_empty_tree_orphan_pool_serves_everything():
    """Pre-migration reality: all nodes layer=3, no child_ptrs → all are orphans.

    The whole search must still work (this is the production state TODAY).
    """
    nodes = [_node(f"ep{i}", 3, [1.0, float(i) / 10]) for i in range(5)]
    store = _FakeStore(nodes)
    q = np.array([1.0, 0.0], dtype=np.float32)
    res = recursive_retrieve(q, [store], k_final=3)
    assert len(res.hits) == 3
    assert res.tree_nodes_visited == 0  # no tree was walked
    assert res.orphan_pool_size == 5
    assert all(h.via == "orphan" for h in res.hits)


def test_broken_pointer_becomes_leaf(tree_store):
    """CHECK-3: a child_ptr to a missing node is dropped + counted, never crashes."""
    # Point cat0 at a non-existent child id.
    cat0 = tree_store.get("cat0")
    cat0.child_ptrs = ["ep_a", "ghost_node", "ep_b"]
    q = np.array([1.0, 0.0], dtype=np.float32)
    res = recursive_retrieve(q, [tree_store], k_final=3)  # must not raise
    assert res.broken_ptrs == 1
    assert res.hits[0].node.id == "ep_a"  # real children still reached


def test_cycle_does_not_hang(tree_store):
    """CHECK-2: an accidental cycle in the tree terminates via the visited-set."""
    # Make ep_a point back up to cat0 — a cycle cat0 → ep_a → cat0.
    ep_a = tree_store.get("ep_a")
    ep_a.layer = 1  # pretend it's an index node so the descent tries its children
    ep_a.child_ptrs = ["cat0"]
    q = np.array([1.0, 0.0], dtype=np.float32)
    res = recursive_retrieve(q, [tree_store], k_final=3)  # must terminate
    # If the visited-set works, every id appears at most once in the descent.
    assert res.tree_nodes_visited <= 7  # total nodes; no infinite revisits


def test_k_topo_wider_than_k_final(monkeypatch):
    """ASSERT-7: the Domain-layer beam is k_topo = 3× k_final by default."""
    from imi import hmem_retrieve

    # Build 6 domain nodes; with k_final=2, k_topo=6 keeps all of them in the beam.
    doms = []
    for i in range(6):
        d = _node(f"dom{i}", 0, [1.0, float(i)], child_ptrs=[f"ep{i}"])
        ep = _node(f"ep{i}", 3, [1.0, float(i)], parent_id=f"dom{i}")
        doms.extend([d, ep])
    store = _FakeStore(doms)
    q = np.array([1.0, 5.0], dtype=np.float32)  # closest to dom5/ep5
    res = recursive_retrieve(q, [store], k_final=2)
    # ep5 is reachable only if dom5 survived the topo beam (k_topo=6 ≥ 6 doms).
    assert any(h.node.id == "ep5" for h in res.hits)


def test_collapsed_net_recovers_strong_leaf_under_weak_centroid():
    """RAPTOR-style safety net: a strong leaf whose parent centroid matches the
    query WEAKLY must still be retrieved (its branch is pruned at the top otherwise).

    Mechanism found 2026-06-14 on the real store: target leaf sim=0.67 to query but
    parent centroid sim=0.07 → parent ranked #51, branch pruned, leaf lost. The
    collapsed-tree blend sweeps in-tree leaves directly so the leaf is recovered.
    """
    # A weak centroid pointing at a strong leaf. The centroid embedding is far from
    # the query [1,0]; the leaf is exactly the query.
    weak_idx = _node("weakidx", 0, [0.0, 1.0], child_ptrs=["strongleaf"])  # centroid ⊥ query
    strong_leaf = _node("strongleaf", 3, [1.0, 0.0], parent_id="weakidx")
    # Enough decoy index nodes (all matching the query better than weak_idx) to push
    # weak_idx OUT of the k_topo beam (k_final=2 → k_topo=6), so its branch is pruned
    # at the top and the strong leaf can ONLY come back via the collapsed net.
    decoys = [_node(f"d{i}", 0, [0.5, 0.5], child_ptrs=[f"de{i}"]) for i in range(20)]
    decoy_eps = [_node(f"de{i}", 3, [0.4, 0.5], parent_id=f"d{i}") for i in range(20)]
    store = _FakeStore([weak_idx, strong_leaf] + decoys + decoy_eps)

    q = np.array([1.0, 0.0], dtype=np.float32)  # matches strong_leaf, NOT weak_idx
    res = recursive_retrieve(q, [store], k_final=2)
    ids = [h.node.id for h in res.hits]
    assert "strongleaf" in ids, "strong leaf under a weak centroid must be recovered"
    hit = next(h for h in res.hits if h.node.id == "strongleaf")
    assert hit.via == "collapsed", "should be recovered via the collapsed-tree net"


def test_embedding_less_node_sorts_last_not_dropped():
    """A node with no embedding gets sim=-1.0 (sorts last) but is not a crash."""
    n_ok = _node("ok", 3, [1.0, 0.0])
    n_bad = MemoryNode(id="bad", layer=3, seed="bad")  # embedding stays None
    store = _FakeStore([n_ok, n_bad])
    q = np.array([1.0, 0.0], dtype=np.float32)
    res = recursive_retrieve(q, [store], k_final=2)  # must not raise
    assert res.hits[0].node.id == "ok"
