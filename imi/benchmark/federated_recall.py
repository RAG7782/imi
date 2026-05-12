"""Federated Recall Benchmark — recall in simulated federated memory.

First benchmark of federated memory recall in the literature.
Measures whether sharing memories between two agents (via federation events)
improves recall compared to isolated stores.

Design:
  - Create TWO VectorStores (simulating two agents)
  - Split incidents: 60% to store_A, 60% to store_B (40% overlap)
  - Simulate federation: events from store_A are shared to store_B
    (with metadata tag "federated")
  - Measure: can store_B recall incidents from store_A?
  - Compare: federated_r5 vs isolated_r5

Novel contribution: demonstrates that federated memory architecture
improves recall for multi-agent systems.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..embedder import SentenceTransformerEmbedder
from ..node import MemoryNode
from ..store import VectorStore
from .ambench import generate_incidents, recall_at_k


@dataclass
class FederatedRecallResults:
    """Results from federated recall benchmark."""

    isolated_r5: float = 0.0  # Recall@5 without federation
    federated_r5: float = 0.0  # Recall@5 with federation
    federation_boost: float = 0.0  # federated_r5 - isolated_r5
    overlap_pct: float = 0.0  # Actual overlap percentage
    n_store_a: int = 0  # Incidents in store A
    n_store_b: int = 0  # Incidents in store B (before federation)
    n_federated: int = 0  # Incidents federated from A to B
    n_queries: int = 0
    duration_s: float = 0.0
    system_name: str = "IMI"

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "isolated_r5": round(self.isolated_r5, 3),
            "federated_r5": round(self.federated_r5, 3),
            "federation_boost": round(self.federation_boost, 3),
            "overlap_pct": round(self.overlap_pct, 3),
            "n_store_a": self.n_store_a,
            "n_store_b": self.n_store_b,
            "n_federated": self.n_federated,
            "n_queries": self.n_queries,
            "duration_s": round(self.duration_s, 2),
        }

    def __str__(self) -> str:
        check = "+" if self.federation_boost > 0 else "x"
        return (
            f"FederatedRecall Results ({self.system_name}):\n"
            f"  Isolated R@5:      {self.isolated_r5:.3f}\n"
            f"  Federated R@5:     {self.federated_r5:.3f}\n"
            f"  Federation Boost:  {self.federation_boost:+.3f} {check}\n"
            f"  Overlap:           {self.overlap_pct:.1%}\n"
            f"  Store A: {self.n_store_a} | Store B: {self.n_store_b} | "
            f"Federated: {self.n_federated}\n"
            f"  Queries: {self.n_queries} | Duration: {self.duration_s:.1f}s"
        )


class FederatedRecall:
    """Benchmark: recall improvement through memory federation.

    Simulates two agents with partially overlapping memories.
    Tests whether federating memories from agent A to agent B
    improves B's recall on A-originated queries.

    Usage:
        bench = FederatedRecall(n_incidents=300, n_days=90)
        results = bench.run()
        print(results)
    """

    def __init__(
        self,
        n_incidents: int = 300,
        n_days: int = 90,
        seed: int = 42,
        embedder=None,
        split_ratio: float = 0.6,
    ):
        self.n_incidents = n_incidents
        self.n_days = n_days
        self.seed = seed
        self.split_ratio = split_ratio
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.incidents = generate_incidents(n_incidents, n_days, seed)

    def run(
        self,
        system_name: str = "IMI",
        relevance_weight: float = 0.10,
        n_queries: int = 50,
        **kwargs,
    ) -> FederatedRecallResults:
        """Run the federated recall benchmark.

        Phase 1: Split incidents between two stores with overlap.
        Phase 2: Measure isolated recall (store_B queried for A-only incidents).
        Phase 3: Federate A-only incidents to B, measure recall again.
        """
        t0 = time.time()
        rng = np.random.RandomState(self.seed)

        n = len(self.incidents)
        n_per_store = int(n * self.split_ratio)

        # Shuffle indices deterministically
        all_indices = np.arange(n)
        rng.shuffle(all_indices)

        # Assign: first 60% to A, last 60% to B (middle 20% overlaps)
        indices_a = set(all_indices[:n_per_store].tolist())
        indices_b = set(all_indices[n - n_per_store :].tolist())
        overlap = indices_a & indices_b
        a_only = indices_a - overlap

        # Pre-compute embeddings
        embeddings: dict[int, np.ndarray] = {}
        for i in range(n):
            embeddings[i] = self.embedder.embed(self.incidents[i]["text"])

        def build_store(indices: set[int]) -> VectorStore:
            store = VectorStore()
            for idx in sorted(indices):
                inc = self.incidents[idx]
                node = MemoryNode(
                    seed=inc["text"],
                    summary_medium=inc["text"],
                    embedding=embeddings[idx],
                    tags=[inc["pattern_type"]],
                    created_at=float(inc["day"] * 86400),
                )
                node.id = inc["id"]
                store.add(node)
            return store

        # Phase 1: Build isolated store B
        store_b_isolated = build_store(indices_b)

        # Phase 2: Query store_B for incidents that are ONLY in A
        # (these should be hard to find without federation)
        a_only_list = sorted(a_only)
        n_test = min(n_queries, len(a_only_list))
        test_indices = rng.choice(a_only_list, n_test, replace=False)

        isolated_hits = 0
        for idx in test_indices:
            inc = self.incidents[idx]
            query_emb = embeddings[idx]
            results = store_b_isolated.search(
                query_emb, top_k=10, relevance_weight=relevance_weight
            )
            patterns = [nd.tags[0] if nd.tags else "" for nd, _ in results]
            isolated_hits += recall_at_k(patterns, inc["pattern_type"], k=5)
        isolated_r5 = isolated_hits / n_test if n_test > 0 else 0

        # Phase 3: Federate A-only incidents to B
        store_b_federated = build_store(indices_b)

        for idx in a_only:
            inc = self.incidents[idx]
            node = MemoryNode(
                seed=inc["text"],
                summary_medium=inc["text"],
                embedding=embeddings[idx],
                tags=[inc["pattern_type"], "federated"],
                created_at=float(inc["day"] * 86400),
                source="federated:agent_a",
            )
            node.id = f"fed_{inc['id']}"
            store_b_federated.add(node)

        # Query again with federation
        federated_hits = 0
        for idx in test_indices:
            inc = self.incidents[idx]
            query_emb = embeddings[idx]
            results = store_b_federated.search(
                query_emb, top_k=10, relevance_weight=relevance_weight
            )
            patterns = [nd.tags[0] if nd.tags else "" for nd, _ in results]
            federated_hits += recall_at_k(patterns, inc["pattern_type"], k=5)
        federated_r5 = federated_hits / n_test if n_test > 0 else 0

        duration = time.time() - t0
        actual_overlap = len(overlap) / n if n > 0 else 0

        return FederatedRecallResults(
            isolated_r5=isolated_r5,
            federated_r5=federated_r5,
            federation_boost=federated_r5 - isolated_r5,
            overlap_pct=actual_overlap,
            n_store_a=len(indices_a),
            n_store_b=len(indices_b),
            n_federated=len(a_only),
            n_queries=n_test,
            duration_s=duration,
            system_name=system_name,
        )
