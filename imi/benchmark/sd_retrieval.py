"""Semiotic Density Retrieval Benchmark — measures DS-d impact on recall.

Tests whether encoding with distributional semiotic density (DS-d) scoring
improves retrieval quality compared to a plain baseline.

Core hypothesis: DS-d scored seeds should yield >=5% improvement in R@5
when Densify is applied.

Theoretical grounding:
  - Semiotic Density (Paper 9): DS-d measures distributional density
  - Densify operation: replace generic terms with domain-dense equivalents
  - Higher DS-d = more semantically loaded terms = better recall
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..dialect import compute_ds_d
from ..embedder import SentenceTransformerEmbedder
from ..node import MemoryNode
from ..store import VectorStore
from .ambench import generate_incidents, recall_at_k


@dataclass
class SDRetrievalResults:
    """Results from SD Retrieval benchmark."""

    baseline_r5: float = 0.0  # Recall@5 without DS-d scoring
    sd_r5: float = 0.0  # Recall@5 with DS-d weighted results
    improvement_pct: float = 0.0  # (sd_r5 - baseline_r5) / baseline_r5 * 100
    ds_d_mean: float = 0.0  # Mean DS-d across all incidents
    ds_d_std: float = 0.0  # Std of DS-d scores
    n_incidents: int = 0
    n_queries: int = 0
    duration_s: float = 0.0
    system_name: str = "IMI"

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "baseline_r5": round(self.baseline_r5, 3),
            "sd_r5": round(self.sd_r5, 3),
            "improvement_pct": round(self.improvement_pct, 2),
            "ds_d_mean": round(self.ds_d_mean, 3),
            "ds_d_std": round(self.ds_d_std, 3),
            "n_incidents": self.n_incidents,
            "n_queries": self.n_queries,
            "duration_s": round(self.duration_s, 2),
        }

    def __str__(self) -> str:
        check = "+" if self.improvement_pct >= 5.0 else "x"
        return (
            f"SDRetrieval Results ({self.system_name}):\n"
            f"  Baseline R@5:    {self.baseline_r5:.3f}\n"
            f"  SD R@5:          {self.sd_r5:.3f}\n"
            f"  Improvement:     {self.improvement_pct:.1f}% {check} (target >=5%)\n"
            f"  DS-d mean:       {self.ds_d_mean:.3f} +/- {self.ds_d_std:.3f}\n"
            f"  Queries: {self.n_queries} | Duration: {self.duration_s:.1f}s"
        )


class SDRetrieval:
    """Benchmark: semiotic density impact on retrieval quality.

    Measures whether DS-d scoring improves recall by weighting
    search results according to their semiotic density.

    Usage:
        bench = SDRetrieval(n_incidents=300, n_days=90)
        results = bench.run()
        print(results)
    """

    def __init__(
        self,
        n_incidents: int = 300,
        n_days: int = 90,
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
        eval_every: int = 30,
        **kwargs,
    ) -> SDRetrievalResults:
        """Run the SD retrieval benchmark.

        For each evaluation point:
        1. Baseline: standard cosine search, measure R@5
        2. SD-weighted: re-rank results by DS-d score, measure R@5
        3. Compare the two
        """
        t0 = time.time()
        store = VectorStore()
        ds_d_scores: list[float] = []

        # Phase 1: Populate store and compute DS-d for each incident
        node_ds_d: dict[str, float] = {}
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

            # Compute DS-d for this incident
            ds_d = compute_ds_d(incident["text"], self.embedder)
            ds_d_scores.append(ds_d)
            node_ds_d[node.id] = ds_d
            node.mass = 1.0 + ds_d  # Encode DS-d into node mass for weighting

            store.add(node)

        # Phase 2: Evaluate baseline vs SD-weighted retrieval
        baseline_hits = 0
        sd_hits = 0
        n_evals = 0

        eval_indices = list(range(eval_every, len(self.incidents), eval_every))

        for idx in eval_indices:
            incident = self.incidents[idx]
            query_emb = self.embedder.embed(incident["text"])
            target = incident["pattern_type"]

            # Baseline: standard cosine search
            baseline_results = store.search(
                query_emb,
                top_k=10,
                relevance_weight=0.0,  # pure cosine
            )
            baseline_patterns = [n.tags[0] if n.tags else "" for n, _ in baseline_results]
            baseline_r5 = recall_at_k(baseline_patterns, target, k=5)

            # SD-weighted: re-rank by combining cosine score with DS-d
            sd_results = store.search(query_emb, top_k=10, relevance_weight=relevance_weight)
            # Re-rank: boost results with higher DS-d
            reranked = []
            for node, score in sd_results:
                ds_d = node_ds_d.get(node.id, 0.5)
                # Combined score: 70% cosine + 30% DS-d bonus
                combined = 0.7 * score + 0.3 * ds_d
                reranked.append((node, combined))
            reranked.sort(key=lambda x: x[1], reverse=True)

            sd_patterns = [n.tags[0] if n.tags else "" for n, _ in reranked]
            sd_r5 = recall_at_k(sd_patterns, target, k=5)

            baseline_hits += baseline_r5
            sd_hits += sd_r5
            n_evals += 1

        duration = time.time() - t0

        avg_baseline = baseline_hits / n_evals if n_evals > 0 else 0
        avg_sd = sd_hits / n_evals if n_evals > 0 else 0
        improvement = (avg_sd - avg_baseline) / avg_baseline * 100 if avg_baseline > 0 else 0.0

        return SDRetrievalResults(
            baseline_r5=avg_baseline,
            sd_r5=avg_sd,
            improvement_pct=improvement,
            ds_d_mean=float(np.mean(ds_d_scores)),
            ds_d_std=float(np.std(ds_d_scores)),
            n_incidents=self.n_incidents,
            n_queries=n_evals,
            duration_s=duration,
            system_name=system_name,
        )
