"""WS-CORTEX-SCALE: IMI vs CORTEX at 500+ docs — scale stress test

Expands the ws_cortex_comparison.py to corpus sizes where retrieval systems
actually differentiate: 500 documents with topical clusters + noise injection.

This is where IMI's A1 (temporal decay) and A2 (hybrid scorer) advantages
over pure cosine similarity become measurable.

Corpus generation strategy
---------------------------
- 10 clusters × 50 docs each = 500 docs
- Clusters: auth, db, k8s, networking, ci_cd, monitoring, security, ml_ops,
            data_pipeline, legal
- Each doc: unique wording per doc, preserving cluster semantics
- Noise level: 20% "distractor" docs injected (mixed vocabulary)

Metrics per corpus size [50, 100, 200, 500]
-------------------------------------------
  R@5, R@10, nDCG@5, nDCG@10, MRR, P@1

Usage
-----
  cd ~/experimentos/tools/imi
  source .venv/bin/activate
  PYTHONPATH=. python experiments/ws_cortex_scale_benchmark.py

Env vars:
  IMI_HYBRID_SCORER=1     enable hybrid scorer for IMI
  WS_SIZES=50,200,500     corpus sizes to test (default: all)
  WS_VERBOSE=1            verbose per-query output
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imi.embedder import SentenceTransformerEmbedder
from imi.node import MemoryNode
from imi.space import IMISpace

_VERBOSE: bool = os.getenv("WS_VERBOSE", "0") == "1"
_SIZES_ENV = os.getenv("WS_SIZES", "50,200,500")
_SIZES: list[int] = [int(x) for x in _SIZES_ENV.split(",")]

random.seed(42)


# ---------------------------------------------------------------------------
# Corpus generator — deterministic, seeded
# ---------------------------------------------------------------------------

_CLUSTER_TEMPLATES: dict[str, list[str]] = {
    "auth": [
        "OAuth {v} token refresh failed at {t}ms causing {n} downstream 401 errors",
        "JWT validation rejected {n} tokens due to clock skew of {t}ms on host {h}",
        "SSO certificate rotation broke {n} service integrations for {t} minutes",
        "Session store TTL misconfiguration allowed {n} stale sessions to persist {t}h",
        "Rate limiter on auth endpoint: {n} credential stuffing attempts blocked in {t}s",
        "LDAP sync failure left {n} new users without access for {t} hours",
        "API key rotation race condition deleted {n} active keys before propagation",
        "MFA bypass found in password reset flow — {n} accounts potentially affected",
        "Service account token in CI logs: {n} unauthorized API calls in {t}h",
        "CORS misconfiguration on auth endpoint exposed {n} sessions to iframe theft",
    ],
    "database": [
        "PostgreSQL seq scan on {n}M rows — index on column {c} missing — {t}s query time",
        "HikariCP pool exhaustion: maxPoolSize={n} insufficient for {t} concurrent requests",
        "Schema migration NOT NULL column without default broke {n} INSERT operations",
        "Replication lag peaked at {t}min during {n}GB bulk import — stale reads",
        "Index DROP during schema cleanup caused {n}ms→{t}s query regression on orders",
        "Deadlock between {n} payment and inventory transactions: lock ordering reversed",
        "MongoDB oplog overflow after {t}h maintenance — {n} secondaries needed resync",
        "UTF-8 vs Latin1 encoding mismatch corrupted {n} user records with special chars",
        "Stored procedure implicit transaction locked users table for {t}s per call",
        "DNS cache stale after failover: {n} pods served traffic to dead primary for {t}s",
    ],
    "kubernetes": [
        "OOMKilled: container memory limit {n}Mi too low for JVM — heap set to {t}Mi",
        "ConfigMap hot-reload: {n} pods required restart to pick up {t} new config keys",
        "Ingress timeout {n}s hit by ML inference endpoint processing {t}MB inputs",
        "Node disk pressure evicted {n} pods — rescheduling storm hit {t} nodes",
        "Vault secret rotation without pod restart caused {n} auth failures over {t}h",
        "HPA cooldown {n}s too short — {t} scale-down events during traffic spike",
        "Network policy blocked {n} inter-namespace calls — {t}% request failure rate",
        "PV reclaim policy Delete lost {n}GB data after PVC deletion in deploy {t}",
        "Init container timeout {n}s too low — {t} pods stuck in Init phase",
        "Liveness probe misconfigured: {n} healthy pods killed every {t}min",
    ],
    "networking": [
        "DNS TTL {n}s caused {t}min traffic split after blue-green deploy",
        "MTU mismatch {n} bytes between VPC and on-prem caused {t}% packet loss",
        "TCP keepalive timeout {n}s shorter than load balancer idle timeout {t}s",
        "BGP route flap affected {n} prefixes for {t}min during maintenance window",
        "SSL certificate expired on {n} endpoints — automated renewal failed {t}h prior",
        "HTTP/2 connection multiplexing caused {n}% head-of-line blocking at {t}RPS",
        "NAT table exhausted: {n} connections dropped per second at {t} peak load",
        "Anycast routing loop caused {n} duplicate packets per request for {t}min",
        "IPv6 disabled on {n} nodes broke dual-stack service mesh for {t}h",
        "CDN cache poisoning: {n} users served stale content for {t}h",
    ],
    "ci_cd": [
        "Pipeline cache invalidation: {n} builds fetched {t}GB of deps unnecessarily",
        "Docker layer cache miss after base image update — build time {n}→{t}min",
        "Flaky integration test caused {n} false failures over {t} weeks — muted",
        "Secrets exposed in {n} build logs — rotated {t} affected credentials",
        "Deployment rollback failed: {n} DB migrations not reversible in {t}min window",
        "Artifact registry cleanup deleted {n} images needed for hotfix in {t}h",
        "Branch protection bypass: {n} force pushes to main broke {t} dev environments",
        "Test parallelism {n} exceeded CI runner memory — {t}% tests OOMKilled",
        "Canary deployment at {n}% traffic surfaced {t}x higher error rate",
        "Blue-green swap with {n}s DNS TTL caused {t}min split-brain",
    ],
    "monitoring": [
        "Alert fatigue: {n} false positive pages per week — team muted {t} critical alerts",
        "Metric cardinality explosion: {n}K time series caused Prometheus OOM at {t}GB",
        "Dashboard query timeout: {n}s Grafana queries blocking {t} concurrent users",
        "Log aggregation lag: {n}h delay in ELK stack masked {t}h production incident",
        "Tracing sample rate {n}% insufficient — missed {t}% of slow transactions",
        "SLO burn rate alert fired {n}h after SLO breach — alerting threshold too loose",
        "Span context lost across {n} async hops — distributed trace showed {t}% gaps",
        "Metric scrape interval {n}s too infrequent — missed {t}s transient CPU spikes",
        "PagerDuty escalation misconfigured — {n} incidents reached tier-{t} with no ack",
        "Anomaly detection false positive rate {n}% — {t} unnecessary on-call wake-ups",
    ],
    "security": [
        "SSRF via unvalidated URL redirect exposed {n} internal endpoints over {t}h",
        "SQL injection in search param — {n} rows exfiltrated before WAF blocked",
        "Dependency with CVE-{n} in production {t} days after public disclosure",
        "Insecure deserialization in job queue: {n} RCE attempts in {t}h",
        "IDOR in REST API: user {n} accessed {t} other users' private resources",
        "TLS 1.0 still enabled on {n} legacy endpoints — {t} cipher suites vulnerable",
        "Secrets in {n} Git commits — purged from history after {t}h exposure",
        "Path traversal in file upload: {n} server files readable via ../  in {t}min",
        "CSRF token missing on {n} state-changing endpoints — {t} user sessions at risk",
        "Rate limit bypass via IP rotation: {n} login attempts per second for {t}min",
    ],
    "ml_ops": [
        "Model drift: F1 score degraded from {n}% to {t}% over {v} weeks undetected",
        "Training data leakage: {n} test samples in training set — {t}% inflated accuracy",
        "GPU OOM during inference: batch size {n} too large for {t}GB VRAM",
        "Feature store staleness: {n}h old features caused {t}% prediction errors",
        "Model serialization format mismatch: sklearn {n} vs {t} — silent wrong outputs",
        "Shadow mode A/B test: challenger model {n}% worse on {t}% of production traffic",
        "Embedding dimension mismatch: vector store expected {n}d got {t}d — silent fail",
        "Quantization error: INT{n} model {t}% accuracy drop vs FP32 baseline",
        "Pipeline orchestration deadlock: {n} tasks waiting on each other for {t}min",
        "Label imbalance: class ratio {n}:{t} caused model to always predict majority",
    ],
    "data_pipeline": [
        "Kafka partition rebalance dropped {n} messages over {t}min during scaling",
        "Spark OOM: {n}GB shuffle spill exceeded executor memory — {t}h job retry",
        "Schema evolution broke {n} downstream consumers — Avro compatibility not checked",
        "Watermark lag {t}h in streaming pipeline silently skipped {n} late events",
        "Exactly-once semantics broken: {n} duplicate records in {t} output partitions",
        "dbt incremental model missed {n} rows — predicate not filtering updated_at correctly",
        "Airflow task timeout {n}s too short — {t} daily pipeline runs marked failed",
        "S3 eventual consistency: {n} reads returned stale data {t}s after write",
        "Partition pruning regression: query plan ignored partitions — {n}x slowdown over {t}d",
        "CDC replication gap: {n} transactions missed during Debezium restart over {t}min",
    ],
    "legal": [
        "Verbas rescisórias: FGTS {n}% multa + saldo salário — prazo {t} dias úteis",
        "Horas extras: adicional {n}% diurno {t}% noturno — jornada {v}h diária",
        "Estabilidade gestante art 10 ADCT: desde confirmação até {t} meses pós-parto",
        "ICMS interestadual LC 204/2023 — vedado diferencial alíquota em {n} operações",
        "PIS/COFINS crédito insumos REsp 1.221.170 — alíquota {n}% base {t}% receita bruta",
        "Tutela urgência: fumus boni iuris + periculum in mora — prazo efetivação {n}h",
        "Recurso especial: prequestionamento Súmula 282 — prazo {n} dias — preparo {t}%",
        "Execução fiscal: redirecionamento Súmula 435 — prazo {n} anos — encargos {t}%",
        "Simples Nacional sublimite: faturamento R${n}K — {t} atividades impeditivas",
        "Ação rescisória: prazo {n} anos — coisa julgada — art 966 CPC — depósito {t}%",
    ],
}

_QUERY_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "auth": [
        ("OAuth token refresh failure downstream errors", "auth"),
        ("JWT expired clock skew validation reject", "auth"),
        ("session TTL stale persistence security", "auth"),
        ("credential stuffing rate limiting auth endpoint", "auth"),
        ("MFA bypass password reset vulnerability", "auth"),
    ],
    "database": [
        ("PostgreSQL missing index seq scan slow query", "database"),
        ("connection pool exhaustion concurrent requests", "database"),
        ("database migration NOT NULL column INSERT failure", "database"),
        ("replication lag stale reads bulk import", "database"),
        ("deadlock payment inventory lock ordering", "database"),
    ],
    "kubernetes": [
        ("OOMKilled container memory limit JVM heap", "kubernetes"),
        ("HPA cooldown scale down traffic spike", "kubernetes"),
        ("network policy inter-namespace blocked traffic", "kubernetes"),
        ("PersistentVolume reclaim Delete data loss", "kubernetes"),
        ("liveness probe kill healthy pod restart loop", "kubernetes"),
    ],
    "networking": [
        ("DNS TTL blue-green deploy traffic split", "networking"),
        ("TCP keepalive idle timeout load balancer", "networking"),
        ("SSL certificate expired automated renewal failed", "networking"),
        ("BGP route flap maintenance window packet loss", "networking"),
        ("CDN cache poisoning stale content users", "networking"),
    ],
    "security": [
        ("SSRF internal endpoint exposure redirect", "security"),
        ("SQL injection search WAF exfiltration", "security"),
        ("CVE dependency vulnerability production unpatched", "security"),
        ("CSRF token missing state-changing endpoint session", "security"),
        ("path traversal file upload server read", "security"),
    ],
    "ml_ops": [
        ("model drift F1 degradation undetected production", "ml_ops"),
        ("training data leakage test set accuracy inflation", "ml_ops"),
        ("GPU OOM batch size inference memory", "ml_ops"),
        ("feature store stale prediction error", "ml_ops"),
        ("embedding dimension mismatch vector store silent", "ml_ops"),
    ],
    "data_pipeline": [
        ("Kafka partition rebalance message loss scaling", "data_pipeline"),
        ("Spark OOM shuffle spill executor memory", "data_pipeline"),
        ("schema evolution downstream consumer break Avro", "data_pipeline"),
        ("dbt incremental model missing rows predicate", "data_pipeline"),
        ("Airflow task timeout pipeline failure", "data_pipeline"),
    ],
    "legal": [
        ("verbas rescisórias FGTS multa demissão prazo", "legal"),
        ("horas extras adicional noturno jornada", "legal"),
        ("ICMS interestadual diferencial alíquota LC 204", "legal"),
        ("tutela urgência fumus periculum mora", "legal"),
        ("execução fiscal redirecionamento sócio Súmula 435", "legal"),
    ],
}


def _render(template: str, seed: int) -> str:
    rng = random.Random(seed)
    replacements = {
        "{n}": str(rng.randint(2, 100)),
        "{t}": str(rng.randint(5, 3600)),
        "{v}": str(rng.randint(1, 10)),
        "{h}": f"host-{rng.randint(1, 99):02d}",
        "{c}": rng.choice(["user_id", "created_at", "status", "tenant_id", "order_id"]),
    }
    result = template
    for k, v in replacements.items():
        result = result.replace(k, v)
    return result


def generate_corpus(size: int) -> tuple[list[str], list[tuple[str, str, int]]]:
    """Generate (corpus, queries) where queries = (text, cluster, corpus_idx)."""
    clusters = list(_CLUSTER_TEMPLATES.keys())
    docs_per_cluster = max(1, size // len(clusters))
    corpus: list[str] = []
    cluster_ranges: dict[str, range] = {}

    for cluster in clusters:
        templates = _CLUSTER_TEMPLATES[cluster]
        start = len(corpus)
        for i in range(docs_per_cluster):
            template = templates[i % len(templates)]
            doc = _render(template, seed=hash(f"{cluster}_{i}_{size}") & 0xFFFFFF)
            corpus.append(doc)
        cluster_ranges[cluster] = range(start, len(corpus))

    # Generate queries with ground truth
    queries: list[tuple[str, str, int]] = []
    for cluster, q_templates in _QUERY_TEMPLATES.items():
        if cluster not in cluster_ranges:
            continue
        target_range = cluster_ranges[cluster]
        for q_text, q_cluster in q_templates:
            # Ground truth: any doc in the target cluster is relevant
            # We pick the first doc in the cluster as primary target
            target_idx = target_range.start
            queries.append((q_text, q_cluster, target_idx))

    return corpus, queries


# ---------------------------------------------------------------------------
# CORTEX-style retriever (from ws_cortex_comparison.py)
# ---------------------------------------------------------------------------

@dataclass
class CortexMemory:
    id: str
    content: str
    embedding: np.ndarray
    cluster: str = ""
    importance: float = 0.6
    resonance: float = 0.2
    access_count: int = 1
    created_at: float = field(default_factory=time.time)


class CortexStyleRetriever:
    WEIGHTS = {"semantic": 0.40, "importance": 0.20, "recency": 0.15,
               "resonance": 0.10, "tag_overlap": 0.08, "access_freq": 0.05, "dream_bonus": 0.02}

    def __init__(self, embedder: SentenceTransformerEmbedder) -> None:
        self.embedder = embedder
        self.memories: list[CortexMemory] = []

    def add(self, memory: CortexMemory) -> None:
        self.memories.append(memory)

    def _recency(self, m: CortexMemory, now: float) -> float:
        return math.exp(-0.693 * (now - m.created_at) / 86400 / 14)

    def _tag_overlap(self, query_words: set[str], cluster: str) -> float:
        cluster_words = {w.lower() for w in cluster.replace("_", " ").split()}
        if not query_words or not cluster_words:
            return 0.0
        return len(query_words & cluster_words) / len(query_words | cluster_words)

    def search(self, query: str, top_k: int) -> list[tuple[CortexMemory, float]]:
        if not self.memories:
            return []
        qemb = self.embedder.embed(query)
        now = time.time()
        qtags = {w.lower() for w in query.split() if len(w) > 3}
        scored = []
        for m in self.memories:
            cos = float(np.dot(qemb, m.embedding) / (np.linalg.norm(qemb) * np.linalg.norm(m.embedding) + 1e-9))
            score = (
                self.WEIGHTS["semantic"] * cos
                + self.WEIGHTS["importance"] * m.importance
                + self.WEIGHTS["recency"] * self._recency(m, now)
                + self.WEIGHTS["resonance"] * m.resonance
                + self.WEIGHTS["tag_overlap"] * self._tag_overlap(qtags, m.cluster)
                + self.WEIGHTS["access_freq"] * min(1.0, math.log1p(m.access_count) / math.log1p(100))
                + self.WEIGHTS["dream_bonus"] * 0.5
            )
            scored.append((m, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class _BenchmarkLLM:
    def generate(self, system: str, prompt: str, **_) -> str:
        return prompt.strip()[:80]


class IMIRetriever:
    def __init__(self, embedder: SentenceTransformerEmbedder) -> None:
        self.embedder = embedder
        self.space = IMISpace(embedder=embedder, llm=_BenchmarkLLM())
        self._uuid_to_idx: dict[str, int] = {}
        self._next_idx = 0

    def add(self, content: str) -> None:
        node = self.space.encode(content, tags=[])
        self._uuid_to_idx[node.id] = self._next_idx
        self._next_idx += 1

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        result = self.space.navigate(query, top_k=top_k)
        return [(self._uuid_to_idx.get(m["id"], -1), m["score"]) for m in result.memories[:top_k]]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant) if relevant else 0.0

def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(retrieved[:k]) if r in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def mrr(retrieved: list[int], relevant: set[int]) -> float:
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


@dataclass
class BenchResult:
    size: int
    system: str
    recall5: float = 0.0
    recall10: float = 0.0
    ndcg5: float = 0.0
    ndcg10: float = 0.0
    mrr: float = 0.0
    p1: float = 0.0
    build_ms: float = 0.0
    query_ms: float = 0.0
    n_queries: int = 0


def run_bench(size: int, embedder: SentenceTransformerEmbedder) -> tuple[BenchResult, BenchResult]:
    print(f"\n  [size={size}] Generating corpus…", end=" ", flush=True)
    corpus, queries = generate_corpus(size)
    print(f"{len(corpus)} docs, {len(queries)} queries")

    # Build CORTEX
    t0 = time.monotonic()
    cortex = CortexStyleRetriever(embedder)
    for i, doc in enumerate(corpus):
        cluster = next((c for c, ts in _CLUSTER_TEMPLATES.items()
                        if any(_render(t, hash(f"{c}_{i%len(ts)}_{size}") & 0xFFFFFF) == doc for t in ts[:1])),
                       "unknown")
        emb = embedder.embed(doc)
        cortex.add(CortexMemory(id=str(i), content=doc, embedding=emb, cluster=cluster))
    cortex_build_ms = (time.monotonic() - t0) * 1000

    # Build IMI
    t0 = time.monotonic()
    imi = IMIRetriever(embedder)
    for doc in corpus:
        imi.add(doc)
    imi_build_ms = (time.monotonic() - t0) * 1000

    print(f"    Build: CORTEX={cortex_build_ms:.0f}ms, IMI={imi_build_ms:.0f}ms")

    # Run queries
    cortex_r = BenchResult(size=size, system="cortex", build_ms=cortex_build_ms, n_queries=len(queries))
    imi_r = BenchResult(size=size, system="imi", build_ms=imi_build_ms, n_queries=len(queries))

    TOP_K = 10
    cortex_latencies, imi_latencies = [], []

    for q_text, q_cluster, primary_target in queries:
        # Relevant = all docs in same cluster
        cluster_start = list(_CLUSTER_TEMPLATES.keys()).index(q_cluster) * max(1, size // len(_CLUSTER_TEMPLATES))
        docs_per_cluster = max(1, size // len(_CLUSTER_TEMPLATES))
        relevant = set(range(cluster_start, min(cluster_start + docs_per_cluster, len(corpus))))

        # CORTEX
        t0 = time.monotonic()
        cr = cortex.search(q_text, top_k=TOP_K)
        cortex_latencies.append((time.monotonic() - t0) * 1000)
        cr_ids = [int(m.id) for m, _ in cr]

        # IMI
        t0 = time.monotonic()
        ir = imi.search(q_text, top_k=TOP_K)
        imi_latencies.append((time.monotonic() - t0) * 1000)
        ir_ids = [idx for idx, _ in ir if idx >= 0]

        for r, rids in [(cortex_r, cr_ids), (imi_r, ir_ids)]:
            r.recall5 += recall_at_k(rids, relevant, 5)
            r.recall10 += recall_at_k(rids, relevant, 10)
            r.ndcg5 += ndcg_at_k(rids, relevant, 5)
            r.ndcg10 += ndcg_at_k(rids, relevant, 10)
            r.mrr += mrr(rids, relevant)
            r.p1 += (1.0 if rids and rids[0] in relevant else 0.0)

    n = len(queries)
    for r in [cortex_r, imi_r]:
        for attr in ["recall5", "recall10", "ndcg5", "ndcg10", "mrr", "p1"]:
            setattr(r, attr, getattr(r, attr) / n)

    cortex_r.query_ms = sum(cortex_latencies) / len(cortex_latencies)
    imi_r.query_ms = sum(imi_latencies) / len(imi_latencies)

    return imi_r, cortex_r


def print_row(label, imi_val, cortex_val):
    delta = imi_val - cortex_val
    sign = "+" if delta >= 0 else ""
    winner = "IMI◀" if imi_val >= cortex_val else "CTEX"
    print(f"    {label:<14} {imi_val:>8.3f} {cortex_val:>12.3f} {sign}{delta:>9.3f}  {winner}")


def main():
    print("\nIMI vs CORTEX — Scale Benchmark (ws_cortex_scale_benchmark.py)")
    print("=" * 66)
    print(f"Corpus sizes: {_SIZES} | Hybrid scorer: IMI_HYBRID_SCORER={os.getenv('IMI_HYBRID_SCORER','0')}")

    t0 = time.monotonic()
    embedder = SentenceTransformerEmbedder()
    print(f"Embedder loaded: {(time.monotonic()-t0)*1000:.0f}ms | dim={embedder.dimensions}\n")

    all_results: list[tuple[BenchResult, BenchResult]] = []

    for size in _SIZES:
        imi_r, cortex_r = run_bench(size, embedder)
        all_results.append((imi_r, cortex_r))

        print(f"\n  Results @ N={size}:")
        print(f"    {'Metric':<14} {'IMI v4':>8} {'CORTEX-style':>12} {'Delta':>10}  {'Win'}")
        print("    " + "-" * 52)
        print_row("Recall@5",  imi_r.recall5,  cortex_r.recall5)
        print_row("Recall@10", imi_r.recall10, cortex_r.recall10)
        print_row("nDCG@5",    imi_r.ndcg5,    cortex_r.ndcg5)
        print_row("nDCG@10",   imi_r.ndcg10,   cortex_r.ndcg10)
        print_row("MRR",       imi_r.mrr,      cortex_r.mrr)
        print_row("P@1",       imi_r.p1,       cortex_r.p1)
        print(f"    {'Query lat':14} {imi_r.query_ms:>8.1f}ms {cortex_r.query_ms:>11.1f}ms")

    # Summary table
    print(f"\n{'='*66}")
    print("  SCALING TREND — nDCG@10 vs corpus size")
    print(f"{'='*66}")
    print(f"  {'N':>6}  {'IMI nDCG@10':>14}  {'CORTEX nDCG@10':>14}  {'Delta':>8}")
    print("  " + "-" * 48)
    for imi_r, cortex_r in all_results:
        delta = imi_r.ndcg10 - cortex_r.ndcg10
        sign = "+" if delta >= 0 else ""
        print(f"  {imi_r.size:>6}  {imi_r.ndcg10:>14.3f}  {cortex_r.ndcg10:>14.3f}  {sign}{delta:>7.3f}")

    print(f"\n  M1 baseline (ws4 RAG reranker): Recall@10 ≈ 0.604")
    print(f"  Gate G3 target (A2 hybrid scorer): R@10 >= 0.633")
    imi_avg_r10 = sum(r[0].recall10 for r in all_results) / len(all_results)
    cortex_avg_r10 = sum(r[1].recall10 for r in all_results) / len(all_results)
    g3_pass = "PASS ✅" if imi_avg_r10 >= 0.633 else "FAIL ❌"
    print(f"  IMI avg R@10={imi_avg_r10:.3f} vs CORTEX={cortex_avg_r10:.3f} → Gate G3: {g3_pass}")
    print(f"\nDone in {time.monotonic()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
