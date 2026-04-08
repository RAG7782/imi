"""LongMemEval Adapted — long-term recall over extended conversation simulations.

Measures recall quality across different time horizons, comparable to
MemPalace's reported 96.6% overall recall.

Design:
  - Generate 500 incidents across 180 days (longer than AMBench's 90)
  - Add incidents one by one, simulating a conversation
  - At checkpoints, query for incidents from different time ranges
  - Measure recall separately: recent (0-30 days), mid (30-90), old (90-180)
  - Overall = weighted average (recent=40%, mid=35%, old=25%)

Meta target: overall >= 96.6% to match MemPalace.
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
class LongMemEvalResults:
    """Results from LongMemEval benchmark."""
    recent_r5: float = 0.0      # Recall@5 for last 30 days
    mid_r5: float = 0.0         # Recall@5 for 30-90 days
    old_r5: float = 0.0         # Recall@5 for 90-180 days
    overall_r5: float = 0.0     # Weighted average
    n_incidents: int = 0
    n_queries: int = 0
    n_days: int = 0
    duration_s: float = 0.0
    system_name: str = "IMI"

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "recent_r5": round(self.recent_r5, 3),
            "mid_r5": round(self.mid_r5, 3),
            "old_r5": round(self.old_r5, 3),
            "overall_r5": round(self.overall_r5, 3),
            "n_incidents": self.n_incidents,
            "n_queries": self.n_queries,
            "n_days": self.n_days,
            "duration_s": round(self.duration_s, 2),
        }

    def __str__(self) -> str:
        check = "+" if self.overall_r5 >= 0.966 else "x"
        return (
            f"LongMemEval Results ({self.system_name}):\n"
            f"  Recent R@5 (0-30d):    {self.recent_r5:.3f}\n"
            f"  Mid R@5 (30-90d):      {self.mid_r5:.3f}\n"
            f"  Old R@5 (90-180d):     {self.old_r5:.3f}\n"
            f"  Overall R@5:           {self.overall_r5:.3f} {check} (target >=0.966)\n"
            f"  Incidents: {self.n_incidents} over {self.n_days} days\n"
            f"  Queries: {self.n_queries} | Duration: {self.duration_s:.1f}s"
        )


class LongMemEval:
    """Benchmark: long-term recall over extended timelines.

    Simulates 180 days of memory accumulation and measures
    whether recall degrades for older memories.

    Usage:
        bench = LongMemEval(n_incidents=500, n_days=180)
        results = bench.run()
        print(results)
    """

    def __init__(
        self,
        n_incidents: int = 500,
        n_days: int = 180,
        seed: int = 42,
        embedder=None,
    ):
        self.n_incidents = n_incidents
        self.n_days = n_days
        self.seed = seed
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.incidents = generate_incidents(n_incidents, n_days, seed)

    def run(
        self,
        system_name: str = "IMI",
        relevance_weight: float = 0.10,
        n_queries_per_bucket: int = 20,
        **kwargs,
    ) -> LongMemEvalResults:
        """Run the long-term recall evaluation.

        Phase 1: Ingest all incidents into the store.
        Phase 2: Query for incidents from each time bucket.
        """
        t0 = time.time()
        store = VectorStore()
        rng = np.random.RandomState(self.seed)

        # Phase 1: Ingest all incidents
        incident_by_id: dict[str, dict] = {}
        for incident in self.incidents:
            emb = self.embedder.embed(incident["text"])
            node = MemoryNode(
                seed=incident["text"],
                summary_medium=incident["text"],
                embedding=emb,
                tags=[incident["pattern_type"]],
                created_at=float(incident["day"] * 86400),
            )
            node.id = incident["id"]
            store.add(node)
            incident_by_id[incident["id"]] = incident

        # Phase 2: Bucket incidents by time
        max_day = max(inc["day"] for inc in self.incidents)

        # Define buckets relative to the latest day
        # Recent: last 30 days, Mid: 30-90 days ago, Old: 90-180 days ago
        recent_cutoff = max_day - 30
        mid_cutoff = max_day - 90
        old_cutoff = max_day - 180

        recent_incidents = [
            inc for inc in self.incidents if inc["day"] >= recent_cutoff
        ]
        mid_incidents = [
            inc for inc in self.incidents
            if mid_cutoff <= inc["day"] < recent_cutoff
        ]
        old_incidents = [
            inc for inc in self.incidents
            if old_cutoff <= inc["day"] < mid_cutoff
        ]

        def eval_bucket(bucket: list[dict], n_queries: int) -> float:
            """Evaluate recall on a sample from a time bucket."""
            if not bucket:
                return 0.0
            n_sample = min(n_queries, len(bucket))
            indices = rng.choice(len(bucket), n_sample, replace=False)
            hits = 0
            for idx in indices:
                inc = bucket[idx]
                query_emb = self.embedder.embed(inc["text"])
                results = store.search(
                    query_emb, top_k=10, relevance_weight=relevance_weight
                )
                patterns = [n.tags[0] if n.tags else "" for n, _ in results]
                hits += recall_at_k(patterns, inc["pattern_type"], k=5)
            return hits / n_sample

        recent_r5 = eval_bucket(recent_incidents, n_queries_per_bucket)
        mid_r5 = eval_bucket(mid_incidents, n_queries_per_bucket)
        old_r5 = eval_bucket(old_incidents, n_queries_per_bucket)

        # Count total queries actually run
        total_queries = sum(
            min(n_queries_per_bucket, len(b))
            for b in [recent_incidents, mid_incidents, old_incidents]
        )

        # Weighted average: recent=40%, mid=35%, old=25%
        overall_r5 = 0.40 * recent_r5 + 0.35 * mid_r5 + 0.25 * old_r5

        duration = time.time() - t0

        return LongMemEvalResults(
            recent_r5=recent_r5,
            mid_r5=mid_r5,
            old_r5=old_r5,
            overall_r5=overall_r5,
            n_incidents=self.n_incidents,
            n_queries=total_queries,
            n_days=self.n_days,
            duration_s=duration,
            system_name=system_name,
        )
