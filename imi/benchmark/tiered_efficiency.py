"""Tiered Efficiency Benchmark — measures token cost per quality unit.

Core hypothesis: L0+L1 should use <=200 tokens for >=90% of cases.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..node import MemoryNode, AffectiveTag
from ..store import VectorStore
from ..embedder import SentenceTransformerEmbedder
from ..tiering import L0Identity, generate_l1
from .ambench import generate_incidents


@dataclass
class TieredEfficiencyResults:
    l0_tokens: int = 0
    l1_tokens: int = 0
    l0_l1_tokens: int = 0
    l2_avg_tokens: float = 0.0
    l3_avg_tokens: float = 0.0
    pct_under_200: float = 0.0    # % of L0+L1 generations under 200 tokens
    pct_under_500: float = 0.0    # % of L2 retrievals under 500 tokens
    n_sessions: int = 0
    duration_s: float = 0.0
    system_name: str = "IMI"

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "l0_tokens": self.l0_tokens,
            "l1_tokens": self.l1_tokens,
            "l0_l1_tokens": self.l0_l1_tokens,
            "l2_avg_tokens": round(self.l2_avg_tokens, 1),
            "l3_avg_tokens": round(self.l3_avg_tokens, 1),
            "pct_under_200": round(self.pct_under_200, 3),
            "pct_under_500": round(self.pct_under_500, 3),
            "n_sessions": self.n_sessions,
            "duration_s": round(self.duration_s, 2),
        }

    def __str__(self) -> str:
        check_200 = "+" if self.l0_l1_tokens <= 200 else "x"
        check_90 = "+" if self.pct_under_200 >= 0.90 else "x"
        return (
            f"TieredEfficiency Results ({self.system_name}):\n"
            f"  L0 tokens:     {self.l0_tokens}\n"
            f"  L1 tokens:     {self.l1_tokens}\n"
            f"  L0+L1 tokens:  {self.l0_l1_tokens} {check_200} (target <=200)\n"
            f"  L2 avg tokens: {self.l2_avg_tokens:.0f}\n"
            f"  L3 avg tokens: {self.l3_avg_tokens:.0f}\n"
            f"  Under 200 tok: {self.pct_under_200:.1%} {check_90} (target >=90%)\n"
            f"  Sessions: {self.n_sessions} | Duration: {self.duration_s:.1f}s"
        )


class TieredEfficiency:
    """Benchmark: token economy across tiers."""

    def __init__(self, n_incidents: int = 300, n_days: int = 90, seed: int = 42, n_sessions: int = 50, embedder=None):
        self.n_incidents = n_incidents
        self.n_days = n_days
        self.seed = seed
        self.n_sessions = n_sessions
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.incidents = generate_incidents(n_incidents, n_days, seed)

    def run(self, system_name: str = "IMI") -> TieredEfficiencyResults:
        t0 = time.time()
        store = VectorStore()

        # Populate store
        for incident in self.incidents:
            emb = self.embedder.embed(incident["text"])
            node = MemoryNode(
                seed=incident["text"],
                summary_orbital=incident["text"][:40],
                summary_medium=incident["text"][:120],
                summary_detailed=incident["text"],
                embedding=emb,
                tags=[incident["pattern_type"]],
                affect=AffectiveTag(salience=0.5, valence=0.3, arousal=0.4),
                created_at=float(incident["day"] * 86400),
            )
            node.id = incident["id"]
            store.add(node)

        # Simulate sessions
        l0_tokens_list = []
        l1_tokens_list = []
        l0l1_tokens_list = []
        l2_tokens_list = []
        l3_tokens_list = []

        l0 = L0Identity(agent_name="Benchmark Agent", domain="sre")

        rng = np.random.RandomState(self.seed)

        for _ in range(self.n_sessions):
            # L0: identity (constant)
            l0_tok = l0.token_estimate()
            l0_tokens_list.append(l0_tok)

            # L1: generate hot facts
            l1 = generate_l1(list(store.nodes), max_facts=7, max_affordances=3)
            l1_tok = l1.token_estimate()
            l1_tokens_list.append(l1_tok)
            l0l1_tokens_list.append(l0_tok + l1_tok)

            # L2: simulate filtered search (top-5 results, medium summaries)
            query_idx = rng.randint(0, len(self.incidents))
            query_emb = self.embedder.embed(self.incidents[query_idx]["text"])
            l2_results = store.search(query_emb, top_k=5, relevance_weight=0.1)
            l2_text = "\n".join(n.summary_medium for n, _ in l2_results if n.summary_medium)
            l2_tokens_list.append(len(l2_text) // 4)

            # L3: full search (top-20, detailed summaries + seeds)
            l3_results = store.search(query_emb, top_k=20, relevance_weight=0.1)
            l3_text = "\n".join(
                f"{n.summary_detailed}\n{n.seed}" for n, _ in l3_results
                if n.summary_detailed or n.seed
            )
            l3_tokens_list.append(len(l3_text) // 4)

        duration = time.time() - t0

        return TieredEfficiencyResults(
            l0_tokens=int(np.mean(l0_tokens_list)),
            l1_tokens=int(np.mean(l1_tokens_list)),
            l0_l1_tokens=int(np.mean(l0l1_tokens_list)),
            l2_avg_tokens=float(np.mean(l2_tokens_list)),
            l3_avg_tokens=float(np.mean(l3_tokens_list)),
            pct_under_200=sum(1 for t in l0l1_tokens_list if t <= 200) / len(l0l1_tokens_list),
            pct_under_500=sum(1 for t in l2_tokens_list if t <= 500) / len(l2_tokens_list),
            n_sessions=self.n_sessions,
            duration_s=duration,
            system_name=system_name,
        )
