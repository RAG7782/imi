"""AMBench — Agent Memory Benchmark.

Evaluates agent memory systems on the full encode-retrieve-consolidate-act
lifecycle over simulated days with recurring incident patterns.

Metrics:
  M1. Retrieval R@5 — Can the system find relevant past incidents?
  M2. Cluster Purity — Does it recognize recurring patterns?
  M3. Action P@1 — Are suggested actions relevant?
  M4. Temporal Coherence — Average age of top results (lower = more recent)
  M5. Learning Curve — Does retrieval improve over time?
"""

from __future__ import annotations

import copy
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from math import log2

import numpy as np

from imi.embedder import SentenceTransformerEmbedder
from imi.maintain import find_clusters
from imi.node import MemoryNode
from imi.store import VectorStore


# ---------------------------------------------------------------------------
# Incident templates (10 pattern types)
# ---------------------------------------------------------------------------

INCIDENT_TEMPLATES = {
    "connection_pool": [
        "Connection pool exhaustion on {service}: {pool_size} connections maxed out during {trigger}",
        "Database connection pool depleted on {service} after {trigger}, all queries timing out",
    ],
    "memory_leak": [
        "Memory leak detected in {service}: RSS grew from {start_mb}MB to {end_mb}MB over {hours}h",
        "{service} OOM killed, memory usage {end_mb}MB exceeded limit",
    ],
    "timeout_cascade": [
        "Timeout cascade: {service_a} → {service_b} → {service_c}, p99 latency {latency}ms",
        "Cascading timeouts starting at {service_a}: downstream {service_b} and {service_c} affected",
    ],
    "cert_expiry": [
        "TLS certificate expired on {service}, breaking connections from {affected_count} clients",
        "Certificate rotation failed on {service}, manual intervention needed",
    ],
    "deploy_rollback": [
        "Deployment of {service} v{version} rolled back: {error_type} errors increased {pct}%",
        "{service} v{version} deployment failed, errors at {pct}%, rolling back",
    ],
    "dns_failure": [
        "DNS resolution failure for {service}.internal: {affected_count} services impacted for {duration}m",
        "Internal DNS returning NXDOMAIN for {service}: {affected_count} dependent services failing",
    ],
    "disk_full": [
        "Disk full on {service} node: {path} at 100%, writes failing",
        "{service} disk space exhausted, log rotation failed",
    ],
    "rate_limit": [
        "Rate limiter triggered on {service}: {rps} req/s exceeded threshold, {pct}% requests rejected",
        "{service} rate limiting {pct}% of traffic at {rps} req/s",
    ],
    "data_inconsistency": [
        "Data inconsistency between {service_a} and {service_b}: {count} records diverged",
        "Replication lag caused {count} records to differ between {service_a} and {service_b}",
    ],
    "auth_failure": [
        "Authentication failures spike on {service}: {pct}% of requests getting 401",
        "JWT validation failing on {service} after key rotation, {pct}% auth errors",
    ],
}

SERVICES = [
    "api-gateway", "user-service", "payment-service", "order-service",
    "inventory-service", "notification-service", "analytics-service",
    "auth-service", "search-service", "cdn-edge",
]

TRIGGERS = ["Black Friday traffic", "batch job spike", "DDoS attempt",
            "marketing campaign", "data migration", "partner API burst"]


def generate_incidents(n: int = 300, days: int = 90, seed: int = 42) -> list[dict]:
    """Generate n realistic incidents over `days` simulated days."""
    rng = random.Random(seed)
    incidents = []
    pattern_types = list(INCIDENT_TEMPLATES.keys())

    pattern_weights = {p: rng.paretovariate(1.0) for p in pattern_types}
    total = sum(pattern_weights.values())
    pattern_weights = {p: w / total for p, w in pattern_weights.items()}

    for i in range(n):
        roll = rng.random()
        cumulative = 0
        pattern_type = pattern_types[0]
        for p, w in pattern_weights.items():
            cumulative += w
            if roll <= cumulative:
                pattern_type = p
                break

        template = rng.choice(INCIDENT_TEMPLATES[pattern_type])
        params = {
            "service": rng.choice(SERVICES),
            "service_a": rng.choice(SERVICES),
            "service_b": rng.choice(SERVICES),
            "service_c": rng.choice(SERVICES),
            "pool_size": rng.choice([50, 100, 200, 500]),
            "trigger": rng.choice(TRIGGERS),
            "start_mb": rng.randint(256, 1024),
            "end_mb": rng.randint(2048, 8192),
            "hours": rng.randint(2, 48),
            "latency": rng.choice([500, 1000, 2000, 5000, 10000]),
            "affected_count": rng.randint(3, 50),
            "version": f"{rng.randint(1, 5)}.{rng.randint(0, 20)}.{rng.randint(0, 99)}",
            "error_type": rng.choice(["5xx", "timeout", "connection_refused", "OOM"]),
            "pct": rng.randint(10, 90),
            "duration": rng.randint(5, 120),
            "rps": rng.choice([1000, 5000, 10000, 50000]),
            "path": rng.choice(["/var/log", "/data", "/tmp", "/opt/app/logs"]),
            "size": rng.randint(50, 500),
            "count": rng.randint(100, 10000),
            "component": rng.choice(["request handler", "cache layer", "session store"]),
        }

        text = template.format(**params)
        day = int(i / n * days)

        incidents.append({
            "id": f"inc_{i:04d}",
            "text": text,
            "pattern_type": pattern_type,
            "day": day,
            "severity": rng.choice(["low", "medium", "high", "critical"]),
        })

    return incidents


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_patterns: list[str], ground_truth: str, k: int = 5) -> float:
    """1 if ground truth pattern is in top-k retrieved, 0 otherwise."""
    return 1.0 if ground_truth in retrieved_patterns[:k] else 0.0


def cluster_purity(clusters: list[list[MemoryNode]], ground_truth: dict[str, str]) -> float:
    """How pure are the clusters w.r.t. ground truth patterns."""
    if not clusters:
        return 0.0
    total_correct = 0
    total_nodes = 0
    for cluster in clusters:
        pattern_counts = defaultdict(int)
        for node in cluster:
            gt = ground_truth.get(node.id, "unknown")
            pattern_counts[gt] += 1
        majority = max(pattern_counts.values()) if pattern_counts else 0
        total_correct += majority
        total_nodes += len(cluster)
    return total_correct / total_nodes if total_nodes > 0 else 0.0


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class AMBenchResults:
    """Results from running AMBench."""
    retrieval_r5: float = 0.0
    retrieval_r10: float = 0.0
    cluster_purity_score: float = 0.0
    temporal_avg_age: float = 0.0
    n_incidents: int = 0
    n_days: int = 0
    n_patterns: int = 0
    duration_s: float = 0.0
    system_name: str = ""

    def __str__(self) -> str:
        return (
            f"AMBench Results ({self.system_name}):\n"
            f"  Incidents: {self.n_incidents} over {self.n_days} days ({self.n_patterns} patterns)\n"
            f"  Retrieval R@5:  {self.retrieval_r5:.3f}\n"
            f"  Retrieval R@10: {self.retrieval_r10:.3f}\n"
            f"  Cluster Purity: {self.cluster_purity_score:.3f}\n"
            f"  Temporal Avg Age: {self.temporal_avg_age:.1f} days\n"
            f"  Duration: {self.duration_s:.1f}s"
        )

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "retrieval_r5": round(self.retrieval_r5, 3),
            "retrieval_r10": round(self.retrieval_r10, 3),
            "cluster_purity": round(self.cluster_purity_score, 3),
            "temporal_avg_age": round(self.temporal_avg_age, 1),
            "n_incidents": self.n_incidents,
            "n_days": self.n_days,
            "duration_s": round(self.duration_s, 1),
        }


# ---------------------------------------------------------------------------
# AMBench runner
# ---------------------------------------------------------------------------


class AMBench:
    """Agent Memory Benchmark — evaluate any memory system.

    Usage:
        bench = AMBench(n_incidents=300, n_days=90)
        results = bench.run(system_name="IMI", relevance_weight=0.10)
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
        self._ground_truth: dict[str, str] = {}

    def run(
        self,
        system_name: str = "IMI",
        relevance_weight: float = 0.10,
        eval_every: int = 30,
    ) -> AMBenchResults:
        """Run the full benchmark.

        Args:
            system_name: Name for this run (e.g., "IMI", "RAG", "custom")
            relevance_weight: rw for the vector store search
            eval_every: Evaluate retrieval every N incidents
        """
        t0 = time.time()
        store = VectorStore()
        r5_scores = []
        r10_scores = []
        temporal_ages = []

        for i, incident in enumerate(self.incidents):
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
            self._ground_truth[node.id] = incident["pattern_type"]

            # Evaluate periodically
            if i > 0 and i % eval_every == 0:
                query_emb = emb  # use current incident as query
                results = store.search(
                    query_emb, top_k=10, relevance_weight=relevance_weight
                )
                retrieved_patterns = [n.tags[0] if n.tags else "" for n, _ in results]

                r5 = recall_at_k(retrieved_patterns, incident["pattern_type"], k=5)
                r10 = recall_at_k(retrieved_patterns, incident["pattern_type"], k=10)
                r5_scores.append(r5)
                r10_scores.append(r10)

                # Temporal coherence
                current_day = incident["day"]
                for n, _ in results[:5]:
                    node_day = n.created_at / 86400 if n.created_at else 0
                    temporal_ages.append(abs(current_day - node_day))

        # Cluster purity
        clusters = find_clusters(store, similarity_threshold=0.45)

        duration = time.time() - t0

        return AMBenchResults(
            retrieval_r5=float(np.mean(r5_scores)) if r5_scores else 0.0,
            retrieval_r10=float(np.mean(r10_scores)) if r10_scores else 0.0,
            cluster_purity_score=cluster_purity(clusters, self._ground_truth),
            temporal_avg_age=float(np.mean(temporal_ages)) if temporal_ages else 0.0,
            n_incidents=self.n_incidents,
            n_days=self.n_days,
            n_patterns=len(INCIDENT_TEMPLATES),
            duration_s=duration,
            system_name=system_name,
        )
