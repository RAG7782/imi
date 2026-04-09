"""Tiered Recall Benchmark — measures recall at each tier level.

Tests whether the L0-L3 tiering VIEW preserves retrieval quality
while reducing token cost. Core hypothesis: L0+L1 should handle
>=90% of queries that full access handles correctly.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..node import MemoryNode, AffectiveTag
from ..store import VectorStore
from ..embedder import SentenceTransformerEmbedder
from ..tiering import generate_l1, compute_tier, apply_tiering
from .ambench import generate_incidents, recall_at_k


@dataclass
class TieredRecallResults:
    """Results from tiered recall benchmark."""
    full_r5: float = 0.0          # Recall@5 with full access (baseline)
    l1_coverage: float = 0.0      # % of queries answerable from L1 facts alone
    l2_r5: float = 0.0            # Recall@5 with L2 filtered search
    l3_r5: float = 0.0            # Recall@5 with L3 deep search (=full)
    tier_ratio: float = 0.0       # l1_coverage / full_r5 (target: >=0.90)
    n_queries: int = 0
    duration_s: float = 0.0
    system_name: str = "IMI"

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "full_r5": round(self.full_r5, 3),
            "l1_coverage": round(self.l1_coverage, 3),
            "l2_r5": round(self.l2_r5, 3),
            "l3_r5": round(self.l3_r5, 3),
            "tier_ratio": round(self.tier_ratio, 3),
            "n_queries": self.n_queries,
            "duration_s": round(self.duration_s, 2),
        }

    def __str__(self) -> str:
        check = "+" if self.tier_ratio >= 0.90 else "x"
        return (
            f"TieredRecall Results ({self.system_name}):\n"
            f"  Full R@5 (baseline): {self.full_r5:.3f}\n"
            f"  L1 Coverage:         {self.l1_coverage:.3f}\n"
            f"  L2 R@5:              {self.l2_r5:.3f}\n"
            f"  L3 R@5:              {self.l3_r5:.3f}\n"
            f"  Tier Ratio (L1/Full): {self.tier_ratio:.3f} {check} (target >=0.90)\n"
            f"  Queries: {self.n_queries} | Duration: {self.duration_s:.1f}s"
        )


class TieredRecall:
    """Benchmark: recall quality at each tier level."""

    def __init__(self, n_incidents: int = 300, n_days: int = 90, seed: int = 42, embedder=None):
        self.n_incidents = n_incidents
        self.n_days = n_days
        self.seed = seed
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.incidents = generate_incidents(n_incidents, n_days, seed)

    def run(self, system_name: str = "IMI", relevance_weight: float = 0.10, eval_every: int = 30) -> TieredRecallResults:
        t0 = time.time()
        store = VectorStore()
        ground_truth = {}

        full_hits = 0
        l1_hits = 0
        l2_hits = 0
        n_evals = 0

        for i, incident in enumerate(self.incidents):
            emb = self.embedder.embed(incident["text"])
            node = MemoryNode(
                seed=incident["text"],
                summary_orbital=incident["text"][:40],
                summary_medium=incident["text"][:120],
                summary_detailed=incident["text"],
                embedding=emb,
                tags=[incident["pattern_type"]],
                affect=AffectiveTag(
                    salience=0.8 if incident["severity"] in ("high", "critical") else 0.4,
                    valence=0.3,
                    arousal=0.6,
                ),
                mass=1.5 if incident["severity"] in ("high", "critical") else 1.0,
                created_at=float(incident["day"] * 86400),
            )
            node.id = incident["id"]
            node.access_count = max(1, i % 5)  # Simulate varying access
            store.add(node)
            ground_truth[node.id] = incident["pattern_type"]

            if i > 0 and i % eval_every == 0:
                query_emb = emb  # Use current incident as query
                target = incident["pattern_type"]

                # L3/Full: search all nodes
                full_results = store.search(query_emb, top_k=10, relevance_weight=relevance_weight)
                full_patterns = [n.tags[0] if n.tags else "" for n, _ in full_results]
                r5_full = recall_at_k(full_patterns, target, k=5)

                # L2: filtered search (only nodes with matching tags or high relevance)
                # Simulate by searching with higher relevance_weight (prioritizes relevant nodes)
                l2_results = store.search(query_emb, top_k=5, relevance_weight=min(0.3, relevance_weight * 3))
                l2_patterns = [n.tags[0] if n.tags else "" for n, _ in l2_results]
                r5_l2 = recall_at_k(l2_patterns, target, k=5)

                # L1: check if target pattern is in hot facts
                l1 = generate_l1(list(store.nodes))
                l1_tags = set()
                for fact in l1.facts:
                    l1_tags.update(fact.get("tags", []))
                l1_hit = 1.0 if target in l1_tags else 0.0

                full_hits += r5_full
                l2_hits += r5_l2
                l1_hits += l1_hit
                n_evals += 1

        duration = time.time() - t0

        full_r5 = full_hits / n_evals if n_evals > 0 else 0
        l1_cov = l1_hits / n_evals if n_evals > 0 else 0
        l2_r5 = l2_hits / n_evals if n_evals > 0 else 0

        return TieredRecallResults(
            full_r5=full_r5,
            l1_coverage=l1_cov,
            l2_r5=l2_r5,
            l3_r5=full_r5,  # L3 = full access
            tier_ratio=l1_cov / full_r5 if full_r5 > 0 else 0,
            n_queries=n_evals,
            duration_s=duration,
            system_name=system_name,
        )
