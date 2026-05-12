"""AMBench — Agent Memory Benchmark.

Evaluates agent memory systems on the full encode-retrieve-consolidate-act
lifecycle over simulated days with recurring incident patterns.

Metrics:
  M1. Retrieval R@5         — Can the system find relevant past incidents?
  M2. Retrieval R@10        — Broader recall
  M3. Cluster Purity        — Does it recognize recurring patterns?
  M4. Temporal Coherence    — Average age of top results (lower = more recent)
  M5. Task Success Rate     — Does retrieval enable correct action selection?
  M6. Pattern Extraction    — Classification accuracy of unsupervised clusters
  M7. Quarterly R@5         — Temporal degradation across Q1/Q2/Q3/Q4 (365-day)

Default: 600 incidents over 365 days (full cognitive evaluation).
Quick mode: n_days=90, n_incidents=300 (backward compatible).
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

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
        "Connection pool exhaustion on {service}: {pool_size} connections maxed out "
        "during {trigger}",
        "Database connection pool depleted on {service} after {trigger}, all queries timing out",
    ],
    "memory_leak": [
        "Memory leak detected in {service}: RSS grew from {start_mb}MB to {end_mb}MB over {hours}h",
        "{service} OOM killed, memory usage {end_mb}MB exceeded limit",
    ],
    "timeout_cascade": [
        "Timeout cascade: {service_a} → {service_b} → {service_c}, p99 latency {latency}ms",
        "Cascading timeouts starting at {service_a}: downstream {service_b} "
        "and {service_c} affected",
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
        "DNS resolution failure for {service}.internal: {affected_count} services "
        "impacted for {duration}m",
        "Internal DNS returning NXDOMAIN for {service}: {affected_count} dependent "
        "services failing",
    ],
    "disk_full": [
        "Disk full on {service} node: {path} at 100%, writes failing",
        "{service} disk space exhausted, log rotation failed",
    ],
    "rate_limit": [
        "Rate limiter triggered on {service}: {rps} req/s exceeded threshold, "
        "{pct}% requests rejected",
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
    "api-gateway",
    "user-service",
    "payment-service",
    "order-service",
    "inventory-service",
    "notification-service",
    "analytics-service",
    "auth-service",
    "search-service",
    "cdn-edge",
]

TRIGGERS = [
    "Black Friday traffic",
    "batch job spike",
    "DDoS attempt",
    "marketing campaign",
    "data migration",
    "partner API burst",
]

# Canonical remediation actions per incident pattern (ground truth for task success)
REMEDIATION_ACTIONS = {
    "connection_pool": "increase connection pool size and add circuit breaker",
    "memory_leak": "restart service, profile heap, patch memory leak in code",
    "timeout_cascade": "add timeout bulkheads, increase downstream timeout budget",
    "cert_expiry": "rotate TLS certificate, add expiry monitoring alert",
    "deploy_rollback": "rollback deployment, investigate error spike root cause",
    "dns_failure": "flush DNS cache, verify DNS server health, check /etc/resolv.conf",
    "disk_full": "free disk space, fix log rotation, add disk alert at 80%",
    "rate_limit": "scale service horizontally, tune rate limit thresholds",
    "data_inconsistency": "pause writes, run reconciliation job, fix replication lag",
    "auth_failure": "redeploy auth service, verify key rotation procedure",
}


def generate_incidents(n: int = 600, days: int = 365, seed: int = 42) -> list[dict]:
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

        incidents.append(
            {
                "id": f"inc_{i:04d}",
                "text": text,
                "pattern_type": pattern_type,
                "day": day,
                "severity": rng.choice(["low", "medium", "high", "critical"]),
                "expected_action": REMEDIATION_ACTIONS[pattern_type],
            }
        )

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


def task_success_rate(
    retrieved_nodes: list[MemoryNode],
    ground_truth_pattern: str,
    expected_action: str,
    ground_truth: dict[str, str],
    k: int = 5,
) -> float:
    """M5: Task Success Rate — does retrieval enable correct action selection?

    Success requires both:
      (a) Correct pattern retrieved (pattern_hit)
      (b) Expected action present in retrieved affordances (action_match)

    Returns average of (a) and (b) — range [0, 1].
    """
    top_k = retrieved_nodes[:k]
    retrieved_patterns = [ground_truth.get(n.id, "") for n in top_k]
    pattern_hit = 1.0 if ground_truth_pattern in retrieved_patterns else 0.0

    # Check if any retrieved node has an affordance matching expected action keywords
    action_keywords = set(expected_action.lower().split())
    action_match = 0.0
    for node in top_k:
        for affordance in node.affordances:
            aff_words = (
                set(affordance.action.lower().split()) if hasattr(affordance, "action") else set()
            )
            if len(action_keywords & aff_words) >= 2:  # at least 2 keyword overlap
                action_match = 1.0
                break
        if action_match:
            break

    return (pattern_hit + action_match) / 2.0


def pattern_extraction_accuracy(
    clusters: list[list[MemoryNode]],
    ground_truth: dict[str, str],
) -> float:
    """M6: Pattern Extraction Accuracy — classification accuracy of unsupervised clusters.

    Distinct from cluster_purity:
    - cluster_purity: % of majority pattern within each cluster
    - pattern_extraction_accuracy: % of all clustered nodes correctly classified
      (where 'correct' = node assigned to a cluster whose majority = node's ground truth)

    This measures whether the system learns to *categorize* incidents correctly,
    not just whether each cluster is internally pure.
    """
    if not clusters:
        return 0.0

    # Determine majority label for each cluster
    cluster_labels: list[str] = []
    for cluster in clusters:
        counts: dict[str, int] = defaultdict(int)
        for node in cluster:
            counts[ground_truth.get(node.id, "unknown")] += 1
        cluster_labels.append(max(counts, key=lambda k: counts[k]))

    # Count correctly classified nodes
    correct = 0
    total = 0
    for cluster, label in zip(clusters, cluster_labels):
        for node in cluster:
            total += 1
            if ground_truth.get(node.id, "") == label:
                correct += 1

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class AMBenchResults:
    """Results from running AMBench."""

    # Core retrieval
    retrieval_r5: float = 0.0
    retrieval_r10: float = 0.0
    cluster_purity_score: float = 0.0
    temporal_avg_age: float = 0.0
    # Cognitive metrics (M5-M7)
    task_success: float = 0.0
    pattern_extraction_acc: float = 0.0
    quarterly_r5: dict[str, float] = field(default_factory=dict)  # Q1..Q4
    # Meta
    n_incidents: int = 0
    n_days: int = 0
    n_patterns: int = 0
    duration_s: float = 0.0
    system_name: str = ""

    def __str__(self) -> str:
        quarterly = ""
        if self.quarterly_r5:
            qs = "  ".join(f"{q}: {v:.3f}" for q, v in sorted(self.quarterly_r5.items()))
            quarterly = f"\n  Quarterly R@5:  {qs}"
        return (
            f"AMBench Results ({self.system_name}):\n"
            f"  Incidents: {self.n_incidents} over {self.n_days} days "
            f"({self.n_patterns} patterns)\n"
            f"  Retrieval R@5:        {self.retrieval_r5:.3f}\n"
            f"  Retrieval R@10:       {self.retrieval_r10:.3f}\n"
            f"  Cluster Purity:       {self.cluster_purity_score:.3f}\n"
            f"  Pattern Extraction:   {self.pattern_extraction_acc:.3f}\n"
            f"  Task Success Rate:    {self.task_success:.3f}\n"
            f"  Temporal Avg Age:     {self.temporal_avg_age:.1f} days"
            f"{quarterly}\n"
            f"  Duration: {self.duration_s:.1f}s"
        )

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "retrieval_r5": round(self.retrieval_r5, 3),
            "retrieval_r10": round(self.retrieval_r10, 3),
            "cluster_purity": round(self.cluster_purity_score, 3),
            "pattern_extraction_acc": round(self.pattern_extraction_acc, 3),
            "task_success": round(self.task_success, 3),
            "temporal_avg_age": round(self.temporal_avg_age, 1),
            "quarterly_r5": {q: round(v, 3) for q, v in self.quarterly_r5.items()},
            "n_incidents": self.n_incidents,
            "n_days": self.n_days,
            "duration_s": round(self.duration_s, 1),
        }


# ---------------------------------------------------------------------------
# AMBench runner
# ---------------------------------------------------------------------------


class AMBench:
    """Agent Memory Benchmark — evaluate any memory system.

    Default (full cognitive evaluation):
        bench = AMBench()  # 600 incidents, 365 days
        results = bench.run(system_name="IMI")
        print(results)

    Quick mode (backward compatible):
        bench = AMBench(n_incidents=300, n_days=90)
        results = bench.run(system_name="IMI-quick")
    """

    def __init__(
        self,
        n_incidents: int = 600,
        n_days: int = 365,
        seed: int = 42,
        embedder=None,
    ):
        self.n_incidents = n_incidents
        self.n_days = n_days
        self.seed = seed
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.incidents = generate_incidents(n_incidents, n_days, seed)
        self._ground_truth: dict[str, str] = {}
        self._incident_map: dict[str, dict] = {}  # id → full incident (for task success)

    def run(
        self,
        system_name: str = "IMI",
        relevance_weight: float = 0.10,
        eval_every: int = 30,
    ) -> AMBenchResults:
        """Run the full benchmark with all cognitive metrics.

        Args:
            system_name: Name for this run (e.g., "IMI", "RAG", "custom")
            relevance_weight: rw for the vector store search
            eval_every: Evaluate retrieval every N incidents
        """
        t0 = time.time()
        store = VectorStore()
        r5_scores: list[float] = []
        r10_scores: list[float] = []
        temporal_ages: list[float] = []
        task_scores: list[float] = []

        # Quarterly tracking: Q1=days 0-90, Q2=91-181, Q3=182-272, Q4=273-365
        quarter_size = max(1, self.n_days // 4)
        quarterly_r5: dict[str, list[float]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}

        def _quarter(day: int) -> str:
            q = min(3, day // quarter_size)
            return ["Q1", "Q2", "Q3", "Q4"][q]

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
            self._incident_map[node.id] = incident

            # Evaluate periodically
            if i > 0 and i % eval_every == 0:
                query_emb = emb
                results = store.search(query_emb, top_k=10, relevance_weight=relevance_weight)
                retrieved_nodes = [n for n, _ in results]
                retrieved_patterns = [n.tags[0] if n.tags else "" for n in retrieved_nodes]

                r5 = recall_at_k(retrieved_patterns, incident["pattern_type"], k=5)
                r10 = recall_at_k(retrieved_patterns, incident["pattern_type"], k=10)
                r5_scores.append(r5)
                r10_scores.append(r10)

                # Quarterly R@5
                q = _quarter(incident["day"])
                quarterly_r5[q].append(r5)

                # Temporal coherence
                current_day = incident["day"]
                for n in retrieved_nodes[:5]:
                    node_day = n.created_at / 86400 if n.created_at else 0
                    temporal_ages.append(abs(current_day - node_day))

                # Task success (M5)
                ts = task_success_rate(
                    retrieved_nodes,
                    incident["pattern_type"],
                    incident["expected_action"],
                    self._ground_truth,
                    k=5,
                )
                task_scores.append(ts)

        # Cluster purity + pattern extraction (M3, M6)
        clusters = find_clusters(store, similarity_threshold=0.45)
        purity = cluster_purity(clusters, self._ground_truth)
        extraction_acc = pattern_extraction_accuracy(clusters, self._ground_truth)

        # Quarterly averages (skip empty quarters for short runs)
        quarterly_avg = {q: float(np.mean(scores)) for q, scores in quarterly_r5.items() if scores}

        duration = time.time() - t0

        return AMBenchResults(
            retrieval_r5=float(np.mean(r5_scores)) if r5_scores else 0.0,
            retrieval_r10=float(np.mean(r10_scores)) if r10_scores else 0.0,
            cluster_purity_score=purity,
            temporal_avg_age=float(np.mean(temporal_ages)) if temporal_ages else 0.0,
            task_success=float(np.mean(task_scores)) if task_scores else 0.0,
            pattern_extraction_acc=extraction_acc,
            quarterly_r5=quarterly_avg,
            n_incidents=self.n_incidents,
            n_days=self.n_days,
            n_patterns=len(INCIDENT_TEMPLATES),
            duration_s=duration,
            system_name=system_name,
        )
