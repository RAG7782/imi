"""Cross-Session Recall Benchmark — measures recall persistence across sessions.

Simulates N sessions with dream/consolidation cycles between them.
Core hypothesis: recall >=85% after 30 sessions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..embedder import SentenceTransformerEmbedder
from ..maintain import run_maintenance
from ..node import AffectiveTag, MemoryNode
from ..store import VectorStore
from .ambench import generate_incidents, recall_at_k


@dataclass
class CrossSessionResults:
    initial_r5: float = 0.0
    final_r5: float = 0.0
    retention_rate: float = 0.0  # final_r5 / initial_r5
    r5_per_session: list = None  # Recall at each checkpoint
    n_sessions: int = 0
    n_dreams: int = 0
    patterns_consolidated: int = 0
    nodes_pruned: int = 0
    duration_s: float = 0.0
    system_name: str = "IMI"

    def __post_init__(self):
        if self.r5_per_session is None:
            self.r5_per_session = []

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "initial_r5": round(self.initial_r5, 3),
            "final_r5": round(self.final_r5, 3),
            "retention_rate": round(self.retention_rate, 3),
            "n_sessions": self.n_sessions,
            "n_dreams": self.n_dreams,
            "patterns_consolidated": self.patterns_consolidated,
            "nodes_pruned": self.nodes_pruned,
            "duration_s": round(self.duration_s, 2),
        }

    def __str__(self) -> str:
        check = "+" if self.retention_rate >= 0.85 else "x"
        return (
            f"CrossSession Results ({self.system_name}):\n"
            f"  Initial R@5:      {self.initial_r5:.3f}\n"
            f"  Final R@5:        {self.final_r5:.3f}\n"
            f"  Retention Rate:   {self.retention_rate:.3f} {check} (target >=0.85)\n"
            f"  Sessions: {self.n_sessions} | Dreams: {self.n_dreams}\n"
            f"  Consolidated: {self.patterns_consolidated} | Pruned: {self.nodes_pruned}\n"
            f"  Duration: {self.duration_s:.1f}s"
        )


class CrossSession:
    """Benchmark: recall persistence through dream cycles."""

    def __init__(
        self,
        n_incidents: int = 100,
        n_days: int = 30,
        n_sessions: int = 30,
        seed: int = 42,
        embedder=None,
    ):
        self.n_incidents = n_incidents
        self.n_days = n_days
        self.n_sessions = n_sessions
        self.seed = seed
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.incidents = generate_incidents(n_incidents, n_days, seed)

    def _eval_recall(
        self, store: VectorStore, test_incidents: list[dict], rw: float = 0.10
    ) -> float:
        """Evaluate recall on test set."""
        hits = 0
        for inc in test_incidents:
            emb = self.embedder.embed(inc["text"])
            results = store.search(emb, top_k=10, relevance_weight=rw)
            patterns = [n.tags[0] if n.tags else "" for n, _ in results]
            hits += recall_at_k(patterns, inc["pattern_type"], k=5)
        return hits / len(test_incidents) if test_incidents else 0

    def run(self, system_name: str = "IMI", relevance_weight: float = 0.10) -> CrossSessionResults:
        t0 = time.time()

        # Split: 70% train, 30% test
        split = int(len(self.incidents) * 0.7)
        train = self.incidents[:split]
        test = self.incidents[split:]

        episodic = VectorStore()
        semantic = VectorStore()

        # Ingest training incidents
        for inc in train:
            emb = self.embedder.embed(inc["text"])
            node = MemoryNode(
                seed=inc["text"],
                summary_orbital=inc["text"][:40],
                summary_medium=inc["text"][:120],
                embedding=emb,
                tags=[inc["pattern_type"]],
                affect=AffectiveTag(salience=0.5, valence=0.3, arousal=0.4),
                created_at=float(inc["day"] * 86400),
            )
            node.id = inc["id"]
            episodic.add(node)

        # Measure initial recall (before any dreams)
        initial_r5 = self._eval_recall(episodic, test, relevance_weight)

        # Simulate sessions with dream cycles
        r5_history = [initial_r5]
        total_consolidated = 0
        total_pruned = 0

        for session in range(self.n_sessions):
            # Dream (consolidation cycle)
            report = run_maintenance(
                episodic,
                semantic,
                self.embedder,
                similarity_threshold=0.45,
                budget=50,
            )
            total_consolidated += report.consolidated
            total_pruned += report.pruned

            # Eval recall on combined stores
            # Merge episodic + semantic for search
            combined = VectorStore()
            for node in episodic.nodes:
                combined.add(node)
            for node in semantic.nodes:
                combined.add(node)

            r5 = self._eval_recall(combined, test, relevance_weight)
            r5_history.append(r5)

        final_r5 = r5_history[-1]
        duration = time.time() - t0

        return CrossSessionResults(
            initial_r5=initial_r5,
            final_r5=final_r5,
            retention_rate=final_r5 / initial_r5 if initial_r5 > 0 else 0,
            r5_per_session=r5_history,
            n_sessions=self.n_sessions,
            n_dreams=self.n_sessions,
            patterns_consolidated=total_consolidated,
            nodes_pruned=total_pruned,
            duration_s=duration,
            system_name=system_name,
        )
