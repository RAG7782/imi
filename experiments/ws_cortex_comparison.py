"""WS-CORTEX: IMI vs CORTEX Architecture Comparison Benchmark

Compares retrieval quality between:
  - IMI v4 (this codebase): navigate() + optional hybrid scorer
  - CORTEX-style: 7-factor hybrid scoring (Dentate Gyrus → CA1 → CA3)

Since direct CORTEX integration requires pgvector + TypeScript runtime,
this script implements a faithful Python port of CORTEX's scoring logic
(from the published architecture spec: https://github.com/ATERNA-AI/cortex)
and benchmarks both on two realistic retrieval scenarios.

Scenarios
---------
S1 — Jurídico (OXÉ-style): Portuguese legal memory retrieval
     Common in RAG systems for Brazilian legal research (OXÉ use case)
S2 — Code Memory: Technical postmortem + code debugging memories
     Common in agentic systems that remember past fixes

Scenario S3 (Auto-Browser navigation) is scheduled post-B2 production:
requires >= 2 weeks of real browser session data.

Metrics
-------
  Recall@5, Recall@10  — fraction of relevant items in top-K
  nDCG@5, nDCG@10      — normalized Discounted Cumulative Gain
  MRR                  — Mean Reciprocal Rank
  P@1                  — Precision at 1 (top result is relevant)

Usage
-----
  cd ~/experimentos/tools/imi
  source .venv/bin/activate
  PYTHONPATH=. python experiments/ws_cortex_comparison.py

Optional flags (env vars):
  IMI_HYBRID_SCORER=1   enable IMI hybrid scorer for A2 comparison
  WS_TOP_K=10           top-K for recall/nDCG (default: 10)
  WS_VERBOSE=1          print per-query results
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imi.embedder import SentenceTransformerEmbedder
from imi.node import MemoryNode
from imi.space import IMISpace
from imi.store import VectorStore


# ---------------------------------------------------------------------------
# Minimal no-op LLM for benchmark use — avoids Anthropic / Ollama dependency
# ---------------------------------------------------------------------------

class _BenchmarkLLM:
    """Stub LLM that returns deterministic summaries without any API call.

    Used by the benchmark to avoid LLM latency noise and external deps.
    The stub returns the first 80 chars of the prompt as a "summary",
    which is sufficient for embedding-only retrieval benchmarks.
    """

    def generate(self, system: str, prompt: str, max_tokens: int = 1024, temperature: float | None = None) -> str:
        # Return first N chars as a content-preserving stub
        content = prompt.strip()
        return content[:80] if len(content) > 80 else content

_TOP_K: int = int(os.getenv("WS_TOP_K", "10"))
_VERBOSE: bool = os.getenv("WS_VERBOSE", "0") == "1"


# ---------------------------------------------------------------------------
# Test corpus: Scenario 1 — Brazilian Legal Memory (S1)
# ---------------------------------------------------------------------------

LEGAL_MEMORIES = [
    # Labor Law cluster (0-7)
    "Verbas rescisórias: FGTS 40% multa + saldo salário + férias proporcionais + 13° proporcional — prazo 10 dias",
    "Horas extras: adicional 50% comum, 100% noturno — jornada 8h diária 44h semanal — intervalos computados",
    "Estabilidade gestante: ADCT art 10 — desde confirmação gravidez até 5 meses após parto — dispensa nula",
    "Acidente trabalho: comunicação CAT em 24h — estabilidade 12 meses após alta — INSS benefício B91",
    "CTPS: obrigação registro em 48h — salário nunca inferior ao mínimo — alteração contratual bilateral",
    "Assédio moral: dano extrapatrimonial CLT art 223-G — critérios: natureza bem jurídico + intensidade + duração",
    "Terceirização: Lei 13429/2017 — responsabilidade subsidiária tomador serviço — atividade-fim permitida",
    "Rescisão indireta: CLT art 483 — descumprimento obrigação patronal — mesmos efeitos dispensa injusta",

    # Tax Law cluster (8-15)
    "ICMS não incide sobre transferência interestadual — LC 204/2023 — STF Tema 1099 — vedado diferencial alíquota",
    "PIS/COFINS não-cumulativo: crédito insumos — conceito amplo REsp 1.221.170 — STJ jurisprudência consolidada",
    "IRPF: ganho capital imóvel — alíquota 15% sobre lucro — isenção R$440k venda único imóvel — 180 dias reinvestimento",
    "CSLL: base de cálculo — adições exclusões compensações — regime lucro real apuração mensal estimada",
    "ISS local da prestação — LC 116/2003 — lista serviços — fato gerador prestação e não contratação",
    "IPTU progressivo no tempo: art 182 CF — sanção uso inadequado solo urbano — aprovação lei municipal específica",
    "Simples Nacional: sublimite estados — art 19 LC 123 — vedação atividades impeditivas — parcelamento débitos",
    "Substituição tributária ICMS: base de cálculo pauta fiscal — restituição diferença menor valor operação real",

    # Civil Procedure cluster (16-23)
    "Tutela provisória urgência: fumus boni iuris + periculum in mora — caução possível — efetivação imediata",
    "Recurso especial: prequestionamento obrigatório — Súmula 282/356 STF — violação literal lei federal",
    "Execução fiscal: redirecionamento sócio — dissolução irregular — Súmula 435 STJ — responsabilidade tributária",
    "Ação rescisória: prazo decadencial 2 anos — rol taxativo CPC art 966 — coisa julgada — violação manifesta",
    "Litisconsórcio necessário unitário: CPC art 114 — sentença uniforme — intimação ausente obrigatória",
    "Inversão ônus prova CDC: hipossuficiência ou verossimilhança — decisão fundamentada — contraditório",
    "Agravo regimental: prazo 15 dias — conhecimento mérito recurso negado — fungibilidade com agravo interno",
    "Cumprimento sentença: impugnação 15 dias — suspensividade depende garantia juízo — taxa honorários 10%",
]

LEGAL_QUERIES = [
    ("verbas rescisórias multa FGTS demissão", [0]),
    ("horas extras adicional noturno jornada trabalho", [1]),
    ("estabilidade grávida gestante demissão", [2]),
    ("acidente trabalho comunicação CAT INSS", [3]),
    ("terceirização atividade-fim responsabilidade", [6]),
    ("ICMS transferência interestadual LC 204", [8]),
    ("PIS COFINS crédito insumos não-cumulativo", [9]),
    ("IRPJ ganho capital imóvel isenção", [10]),
    ("ISS local prestação serviço LC 116", [12]),
    ("tutela urgência cautelar fumus periculum", [16]),
    ("recurso especial prequestionamento violação lei", [17]),
    ("execução fiscal sócio redirecionamento dissolução", [18]),
    ("inversão ônus prova CDC consumidor", [21]),
    ("cumprimento sentença impugnação honorários", [23]),
    ("rescisão indireta CLT 483 obrigação patronal", [7]),
]


# ---------------------------------------------------------------------------
# Test corpus: Scenario 2 — Code / Technical Memory (S2)
# ---------------------------------------------------------------------------

CODE_MEMORIES = [
    # Database cluster (0-7)
    "PostgreSQL query planner chose seq scan on 10M row table — missing index on created_at — EXPLAIN ANALYZE revealed after 8h slowdown",
    "Connection pool exhaustion: HikariCP maxPoolSize=10, traffic spike needed 50 — added auto-scaling rule for pool size",
    "Database migration added NOT NULL column without default — all INSERT operations failed until column dropped and re-added with default",
    "Deadlock between payment and inventory transactions: both tables locked in reverse order — fixed by consistent lock ordering",
    "Read replica replication lag hit 45min during bulk import — analytics queries served stale data — added lag monitoring",
    "MongoDB oplog overflow during maintenance window — secondaries required full resync — increased oplog to 10GB",
    "Stored procedure had implicit transaction locking entire users table for 30s per call — refactored to explicit short transactions",
    "Database failover took 90s instead of 5s — stale DNS cache on app servers — reduced DNS TTL to 10s",

    # Kubernetes / Infrastructure cluster (8-15)
    "CrashLoopBackOff loop caused by OOMKilled: memory limit 256Mi too low for JVM — increased to 1Gi with proper heap settings",
    "ConfigMap hot-reload broke pod: application required restart to pick up new config — switched to readinessProbe + rolling update",
    "Ingress timeout 60s hit by ML inference endpoint — increased to 300s + added keepalive header to prevent proxy termination",
    "Node NotReady cascade: one node had disk pressure — pod rescheduling storm caused thundering herd — added pod anti-affinity",
    "Secret rotation without deployment restart caused auth failures for 2h — patched to use Vault sidecar injector",
    "Horizontal Pod Autoscaler scaled down during traffic spike: cooldown period too short (30s) — increased to 5min",
    "Network policy misconfiguration blocked inter-namespace traffic — service mesh logs showed 403s — fixed with correct podSelector",
    "PersistentVolume reclaim policy was Delete not Retain — all data lost after PVC deletion during deploy — changed to Retain",

    # Python / Application cluster (16-23)
    "asyncio event loop blocked by synchronous file I/O in request handler — moved to run_in_executor — p99 latency 8s→80ms",
    "Memory leak in Python worker: circular references with __del__ not collected by GC — broke cycles, added tracemalloc monitoring",
    "celery task retry storm: all retries firing at same time after broker restart — added exponential backoff with jitter",
    "pickle deserialization vulnerability in task queue: switched to json serializer + input validation — critical security fix",
    "import-time side effects caused test isolation failures: module loaded singleton at import — changed to lazy initialization",
    "string concatenation in hot loop: O(n²) due to immutable strings — replaced with list join — 100x speedup on large inputs",
    "datetime.now() in test caused flaky time-dependent assertions — monkeypatched to freeze_time — 100% deterministic",
    "recursive function without base case hit Python recursion limit at depth 993 — added memoization + iterative fallback",
]

CODE_QUERIES = [
    ("PostgreSQL slow query seq scan missing index", [0]),
    ("connection pool exhaustion spike HikariCP", [1]),
    ("database migration NOT NULL column INSERT failure", [2]),
    ("deadlock transactions lock ordering", [3]),
    ("kubernetes OOMKilled memory limit JVM", [8]),
    ("ingress timeout ML inference endpoint", [10]),
    ("node pressure pod rescheduling thundering herd", [11]),
    ("asyncio event loop blocking synchronous IO", [16]),
    ("memory leak circular references Python GC", [17]),
    ("celery retry storm exponential backoff", [18]),
    ("pickle deserialization vulnerability security", [19]),
    ("datetime flaky test monkeypatch freeze_time", [22]),
    ("recursion limit memoization iterative", [23]),
    ("string concatenation O(n2) hot loop performance", [21]),
    ("ConfigMap hot reload pod restart rollingupdate", [9]),
]


# ---------------------------------------------------------------------------
# CORTEX-style scorer (faithful Python port)
# ---------------------------------------------------------------------------

@dataclass
class CortexMemory:
    """Simplified CORTEX memory record."""
    id: str
    content: str
    embedding: np.ndarray
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    resonance: float = 0.0
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class CortexStyleRetriever:
    """Python port of CORTEX's 7-factor hybrid scoring.

    Factors (from CORTEX architecture):
      F1  semantic similarity (cosine)       weight 0.40
      F2  importance score                   weight 0.20
      F3  recency (exponential decay)        weight 0.15
      F4  resonance (surprise × revisit)     weight 0.10
      F5  tag overlap (Jaccard)              weight 0.08
      F6  access frequency (log-normalized)  weight 0.05
      F7  dream consolidation bonus          weight 0.02
    """

    WEIGHTS = {
        "semantic": 0.40,
        "importance": 0.20,
        "recency": 0.15,
        "resonance": 0.10,
        "tag_overlap": 0.08,
        "access_freq": 0.05,
        "dream_bonus": 0.02,
    }

    def __init__(self, embedder: SentenceTransformerEmbedder) -> None:
        self.embedder = embedder
        self.memories: list[CortexMemory] = []

    def add(self, memory: CortexMemory) -> None:
        self.memories.append(memory)

    def _recency_score(self, mem: CortexMemory, now: float, half_life_days: float = 14.0) -> float:
        age_days = (now - mem.created_at) / 86400
        return math.exp(-0.693 * age_days / half_life_days)  # half-life decay

    def _access_freq_score(self, mem: CortexMemory) -> float:
        return min(1.0, math.log1p(mem.access_count) / math.log1p(100))

    def _tag_overlap(self, query_tags: set[str], mem: CortexMemory) -> float:
        if not query_tags or not mem.tags:
            return 0.0
        mem_tags = {t.lower() for t in mem.tags}
        intersection = query_tags & mem_tags
        union = query_tags | mem_tags
        return len(intersection) / len(union) if union else 0.0

    def search(
        self,
        query: str,
        top_k: int = 10,
        query_tags: set[str] | None = None,
    ) -> list[tuple[CortexMemory, float]]:
        if not self.memories:
            return []

        query_emb = self.embedder.embed(query)
        now = time.time()
        qtags = query_tags or {w.lower() for w in query.split() if len(w) > 3}

        scored: list[tuple[CortexMemory, float]] = []
        for mem in self.memories:
            # F1: semantic
            cosine = float(np.dot(query_emb, mem.embedding) / (
                np.linalg.norm(query_emb) * np.linalg.norm(mem.embedding) + 1e-9
            ))

            # F2: importance
            importance = mem.importance

            # F3: recency
            recency = self._recency_score(mem, now)

            # F4: resonance
            resonance = min(1.0, mem.resonance)

            # F5: tag overlap
            tag_overlap = self._tag_overlap(qtags, mem)

            # F6: access frequency
            access_freq = self._access_freq_score(mem)

            # F7: dream consolidation bonus (all memories equally benefited in this port)
            dream_bonus = 0.5  # neutral; would be higher for consolidated memories

            score = (
                self.WEIGHTS["semantic"] * cosine
                + self.WEIGHTS["importance"] * importance
                + self.WEIGHTS["recency"] * recency
                + self.WEIGHTS["resonance"] * resonance
                + self.WEIGHTS["tag_overlap"] * tag_overlap
                + self.WEIGHTS["access_freq"] * access_freq
                + self.WEIGHTS["dream_bonus"] * dream_bonus
            )
            scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# IMI retriever wrapper
# ---------------------------------------------------------------------------

class IMIRetriever:
    """Thin wrapper around IMISpace for benchmark use.

    Maintains a corpus_id → node_uuid mapping so that ground truth
    integer indices can be compared against navigate() results.
    """

    def __init__(self, embedder: SentenceTransformerEmbedder) -> None:
        self.embedder = embedder
        self.space = IMISpace(embedder=embedder, llm=_BenchmarkLLM())
        # Map: corpus_idx (str) → node uuid
        self._corpus_id_to_uuid: dict[str, str] = {}
        # Map: node uuid → corpus_idx (str)
        self._uuid_to_corpus_id: dict[str, str] = {}
        self._next_idx: int = 0

    def add(self, content: str, tags: str = "") -> None:
        node = self.space.encode(content, tags=[])
        corpus_id = str(self._next_idx)
        self._corpus_id_to_uuid[corpus_id] = node.id
        self._uuid_to_corpus_id[node.id] = corpus_id
        self._next_idx += 1

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Returns list of (corpus_idx_str, score) tuples — matching ground truth format."""
        result = self.space.navigate(query, top_k=top_k)
        out = []
        for m in result.memories[:top_k]:
            corpus_id = self._uuid_to_corpus_id.get(m["id"], m["id"])
            out.append((corpus_id, m["score"]))
        return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, r in enumerate(top_k)
        if r in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, r in enumerate(retrieved_ids):
        if r in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def p_at_1(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    return 1.0 if retrieved_ids and retrieved_ids[0] in relevant_ids else 0.0


@dataclass
class ScenarioResult:
    name: str
    recall_5: float = 0.0
    recall_10: float = 0.0
    ndcg_5: float = 0.0
    ndcg_10: float = 0.0
    mrr_score: float = 0.0
    p1: float = 0.0
    latency_ms: float = 0.0
    n_queries: int = 0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def build_cortex_retriever(
    embedder: SentenceTransformerEmbedder,
    corpus: list[str],
) -> CortexStyleRetriever:
    retriever = CortexStyleRetriever(embedder)
    for i, content in enumerate(corpus):
        emb = embedder.embed(content)
        tags = content.split()[:5]  # first 5 tokens as tags
        mem = CortexMemory(
            id=str(i),
            content=content,
            embedding=emb,
            tags=tags,
            importance=0.6,
            resonance=0.2,
            access_count=1,
        )
        retriever.add(mem)
    return retriever


def build_imi_retriever(
    embedder: SentenceTransformerEmbedder,
    corpus: list[str],
) -> IMIRetriever:
    retriever = IMIRetriever(embedder)
    for i, content in enumerate(corpus):
        retriever.add(content)
    return retriever


def run_queries(
    retriever: Any,
    queries: list[tuple[str, list[int]]],
    retriever_type: str,
    top_k: int = _TOP_K,
) -> ScenarioResult:
    recall5_vals, recall10_vals = [], []
    ndcg5_vals, ndcg10_vals = [], []
    mrr_vals, p1_vals = [], []
    latencies = []

    for query_text, relevant_indices in queries:
        relevant_ids = {str(i) for i in relevant_indices}

        t0 = time.monotonic()
        if retriever_type == "cortex":
            results = retriever.search(query_text, top_k=top_k)
            retrieved_ids = [mem.id for mem, _ in results]
        else:  # imi: returns (id_str, score) tuples
            results = retriever.search(query_text, top_k=top_k)
            retrieved_ids = [node_id for node_id, _ in results]
        latency_ms = (time.monotonic() - t0) * 1000

        latencies.append(latency_ms)
        recall5_vals.append(recall_at_k(retrieved_ids, relevant_ids, 5))
        recall10_vals.append(recall_at_k(retrieved_ids, relevant_ids, 10))
        ndcg5_vals.append(ndcg_at_k(retrieved_ids, relevant_ids, 5))
        ndcg10_vals.append(ndcg_at_k(retrieved_ids, relevant_ids, 10))
        mrr_vals.append(mrr(retrieved_ids, relevant_ids))
        p1_vals.append(p_at_1(retrieved_ids, relevant_ids))

        if _VERBOSE:
            print(f"  [{retriever_type}] Q: {query_text[:50]!r:52s} "
                  f"R@5={recall5_vals[-1]:.2f} P@1={p1_vals[-1]:.0f} "
                  f"top1={retrieved_ids[0] if retrieved_ids else '?':8s} "
                  f"want={list(relevant_ids)}")

    n = len(queries)
    return ScenarioResult(
        name=retriever_type,
        recall_5=sum(recall5_vals) / n,
        recall_10=sum(recall10_vals) / n,
        ndcg_5=sum(ndcg5_vals) / n,
        ndcg_10=sum(ndcg10_vals) / n,
        mrr_score=sum(mrr_vals) / n,
        p1=sum(p1_vals) / n,
        latency_ms=sum(latencies) / n,
        n_queries=n,
    )


def print_comparison(scenario_name: str, imi_r: ScenarioResult, cortex_r: ScenarioResult) -> None:
    print(f"\n{'='*70}")
    print(f"  {scenario_name}")
    print(f"  {imi_r.n_queries} queries | top-K={_TOP_K}")
    print(f"{'='*70}")
    header = f"{'Metric':<18} {'IMI v4':>10} {'CORTEX-style':>14} {'Delta':>10} {'Winner':>8}"
    print(header)
    print("-" * 70)

    def row(label, imi_val, cortex_val):
        delta = imi_val - cortex_val
        winner = "IMI" if imi_val >= cortex_val else "CORTEX"
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<16} {imi_val:>10.3f} {cortex_val:>14.3f} {sign}{delta:>9.3f} {winner:>8}")

    row("Recall@5", imi_r.recall_5, cortex_r.recall_5)
    row("Recall@10", imi_r.recall_10, cortex_r.recall_10)
    row("nDCG@5", imi_r.ndcg_5, cortex_r.ndcg_5)
    row("nDCG@10", imi_r.ndcg_10, cortex_r.ndcg_10)
    row("MRR", imi_r.mrr_score, cortex_r.mrr_score)
    row("P@1", imi_r.p1, cortex_r.p1)
    row("Latency (ms)", imi_r.latency_ms, cortex_r.latency_ms)
    print("-" * 70)

    # Overall winner
    imi_wins = sum([
        imi_r.recall_5 >= cortex_r.recall_5,
        imi_r.recall_10 >= cortex_r.recall_10,
        imi_r.ndcg_5 >= cortex_r.ndcg_5,
        imi_r.ndcg_10 >= cortex_r.ndcg_10,
        imi_r.mrr_score >= cortex_r.mrr_score,
        imi_r.p1 >= cortex_r.p1,
    ])
    cortex_wins = 6 - imi_wins
    print(f"\n  IMI wins: {imi_wins}/6 quality metrics | CORTEX wins: {cortex_wins}/6")
    print(f"  Latency: IMI {imi_r.latency_ms:.1f}ms vs CORTEX {cortex_r.latency_ms:.1f}ms")


def main():
    print("\nIMI vs CORTEX Benchmark — ws_cortex_comparison.py")
    print("="*70)

    print("\n[1/4] Loading embedder (all-MiniLM-L6-v2)...")
    t0 = time.monotonic()
    embedder = SentenceTransformerEmbedder()
    print(f"      Loaded in {(time.monotonic()-t0)*1000:.0f}ms | dim={embedder.dimensions}")

    # -----------------------------------------------------------------------
    # Scenario 1: Jurídico (Portuguese legal)
    # -----------------------------------------------------------------------
    print("\n[2/4] Building retrievers for S1 — Jurídico...")
    t0 = time.monotonic()
    cortex_s1 = build_cortex_retriever(embedder, LEGAL_MEMORIES)
    imi_s1 = build_imi_retriever(embedder, LEGAL_MEMORIES)
    print(f"      Built in {(time.monotonic()-t0)*1000:.0f}ms | corpus={len(LEGAL_MEMORIES)} docs")

    if _VERBOSE:
        print("\n  IMI queries:")
    imi_r1 = run_queries(imi_s1, LEGAL_QUERIES, "imi", top_k=_TOP_K)
    if _VERBOSE:
        print("\n  CORTEX queries:")
    cortex_r1 = run_queries(cortex_s1, LEGAL_QUERIES, "cortex", top_k=_TOP_K)

    print_comparison("Scenario 1 — Jurídico (Brazilian Legal)", imi_r1, cortex_r1)

    # -----------------------------------------------------------------------
    # Scenario 2: Code / Technical Memory
    # -----------------------------------------------------------------------
    print("\n[3/4] Building retrievers for S2 — Code Memory...")
    t0 = time.monotonic()
    cortex_s2 = build_cortex_retriever(embedder, CODE_MEMORIES)
    imi_s2 = build_imi_retriever(embedder, CODE_MEMORIES)
    print(f"      Built in {(time.monotonic()-t0)*1000:.0f}ms | corpus={len(CODE_MEMORIES)} docs")

    if _VERBOSE:
        print("\n  IMI queries:")
    imi_r2 = run_queries(imi_s2, CODE_QUERIES, "imi", top_k=_TOP_K)
    if _VERBOSE:
        print("\n  CORTEX queries:")
    cortex_r2 = run_queries(cortex_s2, CODE_QUERIES, "cortex", top_k=_TOP_K)

    print_comparison("Scenario 2 — Code / Technical Memory", imi_r2, cortex_r2)

    # -----------------------------------------------------------------------
    # Scenario 3 (hard): Mixed corpus — cross-domain noise stress test
    # -----------------------------------------------------------------------
    print("\n[3b/4] Building retrievers for S3 — Mixed Cross-Domain (noise stress test)...")
    # Mix both corpora — queries must find targets despite cross-domain noise
    mixed_corpus = LEGAL_MEMORIES + CODE_MEMORIES
    # Only legal queries — targets are indices 0-23 (legal), noise is 24-47 (code)
    hard_queries = [
        ("verbas rescisórias multa FGTS demissão", [0]),
        ("horas extras adicional noturno", [1]),
        ("estabilidade gestante gravidez", [2]),
        ("terceirização atividade-fim responsabilidade", [6]),
        ("ICMS transferência interestadual LC 204", [8]),
        ("PIS COFINS crédito insumos", [9]),
        ("tutela urgência periculum fumus", [16]),
        ("execução fiscal sócio redirecionamento dissolução", [18]),
        # Code queries — targets are indices 24-47
        ("PostgreSQL seq scan missing index EXPLAIN ANALYZE", [24]),
        ("connection pool HikariCP exhaustion spike", [25]),
        ("kubernetes OOMKilled memory JVM heap", [32]),
        ("asyncio event loop blocking synchronous IO latency", [40]),
        ("celery retry storm exponential backoff broker", [42]),
        ("pickle deserialization vulnerability json serializer", [43]),
        ("datetime freeze_time monkeypatch flaky test", [46]),
    ]

    t0 = time.monotonic()
    cortex_s3 = build_cortex_retriever(embedder, mixed_corpus)
    imi_s3 = build_imi_retriever(embedder, mixed_corpus)
    print(f"      Built in {(time.monotonic()-t0)*1000:.0f}ms | corpus={len(mixed_corpus)} docs (mixed)")

    if _VERBOSE:
        print("\n  IMI queries (mixed):")
    imi_r3 = run_queries(imi_s3, hard_queries, "imi", top_k=_TOP_K)
    if _VERBOSE:
        print("\n  CORTEX queries (mixed):")
    cortex_r3 = run_queries(cortex_s3, hard_queries, "cortex", top_k=_TOP_K)

    print_comparison("Scenario 3 — Mixed Cross-Domain (noise stress test, N=48)", imi_r3, cortex_r3)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  OVERALL SUMMARY (average across both scenarios)")
    print(f"{'='*70}")

    def avg(*vals):
        return sum(vals) / len(vals)

    summary_metrics = [
        ("Recall@5",  avg(imi_r1.recall_5, imi_r2.recall_5, imi_r3.recall_5),
                      avg(cortex_r1.recall_5, cortex_r2.recall_5, cortex_r3.recall_5)),
        ("Recall@10", avg(imi_r1.recall_10, imi_r2.recall_10, imi_r3.recall_10),
                      avg(cortex_r1.recall_10, cortex_r2.recall_10, cortex_r3.recall_10)),
        ("nDCG@5",    avg(imi_r1.ndcg_5, imi_r2.ndcg_5, imi_r3.ndcg_5),
                      avg(cortex_r1.ndcg_5, cortex_r2.ndcg_5, cortex_r3.ndcg_5)),
        ("nDCG@10",   avg(imi_r1.ndcg_10, imi_r2.ndcg_10, imi_r3.ndcg_10),
                      avg(cortex_r1.ndcg_10, cortex_r2.ndcg_10, cortex_r3.ndcg_10)),
        ("MRR",       avg(imi_r1.mrr_score, imi_r2.mrr_score, imi_r3.mrr_score),
                      avg(cortex_r1.mrr_score, cortex_r2.mrr_score, cortex_r3.mrr_score)),
        ("P@1",       avg(imi_r1.p1, imi_r2.p1, imi_r3.p1),
                      avg(cortex_r1.p1, cortex_r2.p1, cortex_r3.p1)),
    ]

    print(f"\n  {'Metric':<18} {'IMI v4':>10} {'CORTEX-style':>14} {'Delta':>10}")
    print("  " + "-" * 56)
    imi_total_wins = 0
    for label, imi_val, cortex_val in summary_metrics:
        delta = imi_val - cortex_val
        sign = "+" if delta >= 0 else ""
        winner_mark = "◀" if imi_val >= cortex_val else " "
        imi_total_wins += int(imi_val >= cortex_val)
        print(f"  {label:<18} {imi_val:>10.3f} {cortex_val:>14.3f} {sign}{delta:>9.3f} {winner_mark}")

    print(f"\n  IMI wins {imi_total_wins}/{len(summary_metrics)} quality metrics overall")
    print(f"\n  [!] CORTEX advantages (not measured here):")
    print(f"      - pgvector native vector ops (sub-ms at scale)")
    print(f"      - Dream Cycle LLM consolidation (async)")
    print(f"      - MCP-native integration (TypeScript first)")
    print(f"      - Multi-tenant session isolation")
    print(f"\n  [!] IMI advantages:")
    print(f"      - Pure Python / SQLite (zero infra deps)")
    print(f"      - Temporal validity decay (A1)")
    print(f"      - Pattern completion via CA3 (A4)")
    print(f"      - Auto-browser memory bridge (B2)")
    print(f"      - LRU embedding cache")
    print(f"\n  [!] Scenario 3 (Auto-Browser navigation):")
    print(f"      BLOCKED — requires >= 2 weeks of B2 production data")
    print(f"      Schedule: run after {time.strftime('%Y-%m-%d', time.localtime(time.time() + 14*86400))}")
    print(f"\n[4/4] Done.\n")


if __name__ == "__main__":
    main()
