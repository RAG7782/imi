"""Tests for graph layer, adaptive rw, and causal detection."""

from __future__ import annotations

import numpy as np

from imi.adaptive import AdaptiveRW, QueryIntent
from imi.causal import auto_link_causal, detect_causal_candidates
from imi.graph import EdgeType, MemoryGraph
from imi.node import MemoryNode
from imi.store import VectorStore


def make_node(idx: int, domain: str = "test", seed: str = "") -> MemoryNode:
    emb = np.random.RandomState(idx).randn(384).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return MemoryNode(
        id=f"{domain}_{idx:02d}",
        seed=seed or f"node {idx} in {domain}",
        embedding=emb,
        tags=[domain],
        source="test",
    )


# ---------------------------------------------------------------------------
# MemoryGraph tests
# ---------------------------------------------------------------------------


class TestMemoryGraph:
    def test_add_edge(self):
        g = MemoryGraph()
        g.add_edge("a", "b", EdgeType.CAUSAL, weight=0.8, label="test")
        assert g.stats()["total_edges"] == 2  # bidirectional

    def test_neighbors(self):
        g = MemoryGraph()
        g.add_edge("a", "b", EdgeType.CAUSAL)
        g.add_edge("a", "c", EdgeType.SIMILAR)
        neighbors = g.neighbors("a")
        target_ids = {nid for nid, _ in neighbors}
        assert "b" in target_ids
        assert "c" in target_ids

    def test_neighbors_filter_type(self):
        g = MemoryGraph()
        g.add_edge("a", "b", EdgeType.CAUSAL)
        g.add_edge("a", "c", EdgeType.SIMILAR)
        causal = g.neighbors("a", edge_type=EdgeType.CAUSAL)
        # Bidirectional: out(a→b) + in(b→a) = 2 edges, both target "b"
        target_ids = {nid for nid, _ in causal}
        assert "b" in target_ids
        assert "c" not in target_ids

    def test_expand_1hop(self):
        g = MemoryGraph()
        g.add_edge("a", "b", EdgeType.CAUSAL, weight=0.8)
        g.add_edge("b", "c", EdgeType.CAUSAL, weight=0.8)
        activation = g.expand(["a"], hops=1)
        assert "a" in activation
        assert "b" in activation
        assert "c" not in activation  # 2 hops away

    def test_expand_2hop(self):
        g = MemoryGraph()
        g.add_edge("a", "b", EdgeType.CAUSAL, weight=0.8)
        g.add_edge("b", "c", EdgeType.CAUSAL, weight=0.8)
        activation = g.expand(["a"], hops=2)
        assert "c" in activation

    def test_remove_edges(self):
        g = MemoryGraph()
        g.add_edge("a", "b", EdgeType.CAUSAL)
        g.add_edge("a", "c", EdgeType.SIMILAR)
        removed = g.remove_edges("a")
        assert removed > 0
        assert g.stats()["total_edges"] == 0

    def test_serialization(self):
        g = MemoryGraph()
        g.add_edge("a", "b", EdgeType.CAUSAL, weight=0.7, label="test")
        data = g.to_dict()
        g2 = MemoryGraph.from_dict(data)
        assert g2.stats()["total_edges"] == len(data)

    def test_auto_link_similar(self):
        store = VectorStore()
        # Create nodes with similar embeddings
        n1 = make_node(0, "auth", "token refresh failure")
        n2 = make_node(0, "auth", "token refresh failure")  # same embedding
        n2.id = "auth_01"
        store.add(n1)
        store.add(n2)

        g = MemoryGraph()
        count = g.auto_link_similar(store, threshold=0.99)
        assert count >= 1

    def test_search_with_expansion(self):
        store = VectorStore()
        nodes = [make_node(i, "test") for i in range(10)]
        for n in nodes:
            store.add(n)

        g = MemoryGraph()
        g.add_edge(nodes[0].id, nodes[5].id, EdgeType.CAUSAL, weight=0.8)

        # Search for node[0] should also surface node[5] via graph
        results = g.search_with_expansion(
            store, nodes[0].embedding, top_k=5, hops=1, graph_weight=0.3)
        ids = [n.id for n, s in results]
        assert nodes[0].id in ids


# ---------------------------------------------------------------------------
# AdaptiveRW tests
# ---------------------------------------------------------------------------


class TestAdaptiveRW:
    def test_temporal_intent(self):
        arw = AdaptiveRW()
        assert arw.classify_intent("recent auth failures") == QueryIntent.TEMPORAL
        assert arw.classify_intent("latest database issues") == QueryIntent.TEMPORAL
        assert arw.classify_intent("what just happened") == QueryIntent.TEMPORAL

    def test_exploratory_intent(self):
        arw = AdaptiveRW()
        assert arw.classify_intent("find all cert incidents") == QueryIntent.EXPLORATORY
        assert arw.classify_intent("list every failure") == QueryIntent.EXPLORATORY

    def test_action_intent(self):
        arw = AdaptiveRW()
        assert arw.classify_intent("how to prevent DNS outages") == QueryIntent.ACTION
        assert arw.classify_intent("fix migration issues") == QueryIntent.ACTION

    def test_default_intent(self):
        arw = AdaptiveRW()
        assert arw.classify_intent("auth token failures") == QueryIntent.DEFAULT

    def test_rw_values(self):
        arw = AdaptiveRW()
        assert arw.classify("recent failures") == 0.15
        assert arw.classify("find all incidents") == 0.00
        assert arw.classify("how to fix this") == 0.05
        assert arw.classify("database issues") == 0.10

    def test_custom_rw_override(self):
        arw = AdaptiveRW(intent_rw={QueryIntent.TEMPORAL: 0.20})
        assert arw.classify("recent failures") == 0.20


# ---------------------------------------------------------------------------
# Causal detection tests
# ---------------------------------------------------------------------------


class TestCausalDetection:
    def test_detect_cross_domain(self):
        store = VectorStore()
        # Create nodes in different domains with similar content
        n1 = make_node(42, "auth", "connection timeout during authentication")
        n2 = make_node(42, "network", "connection timeout during authentication")
        n2.id = "network_00"
        store.add(n1)
        store.add(n2)

        candidates = detect_causal_candidates(n1, store, threshold=0.5, cross_domain_only=True)
        # n2 is same embedding, different domain → should be candidate
        assert any(c.target_id == "network_00" for c in candidates)

    def test_same_domain_filtered(self):
        store = VectorStore()
        n1 = make_node(42, "auth")
        n2 = make_node(42, "auth")
        n2.id = "auth_99"
        store.add(n1)
        store.add(n2)

        candidates = detect_causal_candidates(n1, store, threshold=0.5, cross_domain_only=True)
        assert not any(c.target_id == "auth_99" for c in candidates)

    def test_auto_link_no_llm(self):
        store = VectorStore()
        n1 = make_node(42, "auth", "auth timeout")
        n2 = make_node(42, "network", "auth timeout")
        n2.id = "network_00"
        store.add(n1)
        store.add(n2)

        g = MemoryGraph()
        added = auto_link_causal(n1, store, g, threshold=0.5, max_edges=2, llm=None)
        assert added >= 1
        assert g.stats()["total_edges"] > 0
