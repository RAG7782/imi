"""WS4 Q4: IMI navigate() vs ChromaDB + Reranker

Tests whether a simple RAG system with reranking matches or beats
IMI's navigate() for retrieval quality.

Setup:
- 50 realistic postmortem-style memories with known ground truth
- 20 queries with expected relevant memory IDs
- Compare: ChromaDB cosine, ChromaDB + embedding rerank, IMI navigate()
- Metrics: Recall@5, Recall@10, nDCG@5, MRR

No external reranker API needed — we use embedding-based reranking
(which is what Cohere/Voyage do under the hood, minus the fine-tuning).

Usage:
    source .venv/bin/activate && PYTHONPATH=. python experiments/ws4_imi_vs_rag_reranker.py
"""

from __future__ import annotations

import time
from math import log2

import chromadb
import numpy as np

from imi.embedder import SentenceTransformerEmbedder
from imi.maintain import find_clusters
from imi.node import MemoryNode
from imi.store import VectorStore


# ---------------------------------------------------------------------------
# Test data: realistic postmortem-style memories with ground truth clusters
# ---------------------------------------------------------------------------

MEMORIES = [
    # Auth cluster (IDs 0-9)
    "OAuth token refresh failed silently, causing 401 cascade across all microservices",
    "JWT validation library had a bug where expired tokens passed verification in clock-skewed environments",
    "SSO integration broke when IdP rotated their signing certificate without notification",
    "Session store migration from Redis to DynamoDB lost TTL settings, sessions never expired",
    "Rate limiter on auth endpoint was per-pod not per-user, allowing credential stuffing at scale",
    "LDAP sync job crashed silently, new employees couldn't log in for 48 hours",
    "API key rotation script had a race condition: old keys deleted before new keys propagated",
    "Multi-factor authentication bypass discovered in password reset flow",
    "Service account token leaked in CI logs, used for lateral movement across clusters",
    "CORS misconfiguration allowed credential theft via malicious iframe on partner site",

    # Database cluster (IDs 10-19)
    "PostgreSQL vacuum job blocked by long-running analytics query, table bloat reached 400GB",
    "Connection pool exhaustion during Black Friday: HikariCP max was 10, needed 50",
    "Database migration added NOT NULL column without default, broke all INSERT operations",
    "Replication lag hit 45 minutes during bulk import, read replicas served stale data",
    "Index on orders table was accidentally dropped during schema cleanup, queries went from 2ms to 12s",
    "Deadlock between payment and inventory transactions caused 5-minute checkout failures",
    "MongoDB oplog overflow during maintenance window, secondaries required full resync",
    "Character encoding mismatch between app (UTF-8) and database (Latin1) corrupted user names",
    "Stored procedure had implicit transaction that locked entire users table for 30 seconds per call",
    "Database failover took 90 seconds instead of expected 5 due to stale DNS cache",

    # Kubernetes/Infra cluster (IDs 20-29)
    "Pod OOM killed repeatedly: memory limit was 256Mi but Java heap alone needed 512Mi",
    "Rolling deployment got stuck: readiness probe checked /health but app served 200 before DB connected",
    "Horizontal pod autoscaler thrashed between 2 and 20 replicas due to metric collection delay",
    "Node drain during maintenance evicted all pods simultaneously because PDB was misconfigured",
    "Kubernetes DNS intermittent failures: ndots:5 default caused excessive DNS lookups",
    "ConfigMap update didn't trigger pod restart, stale config served for 6 hours",
    "Ingress controller ran out of file descriptors during traffic spike, dropped new connections",
    "PersistentVolume reclaim policy was Delete instead of Retain, lost production data on PVC deletion",
    "CronJob backoffLimit was 0, single failure permanently disabled the scheduled task",
    "Service mesh sidecar injection broke gRPC streaming: Envoy proxy had 15s idle timeout",

    # Monitoring/Alerting cluster (IDs 30-39)
    "PagerDuty alert fatigue: 200 alerts/day, team stopped responding, real outage missed",
    "Grafana dashboard showed 99.9% uptime but measured at load balancer, not end-user",
    "Log pipeline dropped 30% of events during peak: Kafka partition count was too low",
    "Synthetic monitoring showed green but real users had 5s TTFB due to CDN cache miss pattern",
    "Alert threshold set to 500ms p99 but baseline was already 480ms, alerted every 10 minutes",
    "Distributed tracing lost spans at service boundaries: different OpenTelemetry SDK versions",
    "Metrics cardinality explosion from logging user IDs as labels, Prometheus OOM",
    "Health check endpoint returned 200 even when downstream dependencies were failing",
    "Error rate alert used rate() over 1m window, missed slow-burn degradation over 30 minutes",
    "On-call escalation policy had a gap: primary and secondary were the same person",

    # Network cluster (IDs 40-49)
    "TCP connection timeout set to 30s but upstream proxy had 15s timeout, silent request drops",
    "DNS TTL was 300s but application cached DNS indefinitely, failover to new IP took restart",
    "TLS certificate expired on internal service: team only monitored external certificates",
    "Load balancer health check interval was 30s, unhealthy backend served traffic for 60s total",
    "MTU mismatch between VPN and VPC caused silent packet drops for payloads over 1400 bytes",
    "IPv6 dual-stack rollout broke legacy service that only bound to 0.0.0.0 not [::]",
    "CDN cache key didn't include Accept-Language header, Spanish users got English responses",
    "WebSocket connections killed every 60s by intermediate proxy with HTTP keep-alive timeout",
    "Geo-routing sent Asian traffic to US-East during failover: latency went from 50ms to 400ms",
    "BGP route leak from provider caused 20% of traffic to take a 200ms longer path for 4 hours",
]

# Ground truth: which memories are relevant for each query
QUERIES = [
    {
        "query": "authentication failures and token problems",
        "relevant": [0, 1, 2, 6, 8],
    },
    {
        "query": "database performance and query optimization issues",
        "relevant": [10, 11, 14, 15, 18],
    },
    {
        "query": "kubernetes pod failures and resource management",
        "relevant": [20, 21, 22, 23, 27],
    },
    {
        "query": "monitoring blind spots and false positives",
        "relevant": [30, 33, 34, 37, 38],
    },
    {
        "query": "network timeout and connection issues",
        "relevant": [40, 41, 43, 47, 49],
    },
    {
        "query": "security vulnerabilities in authentication",
        "relevant": [4, 7, 8, 9],
    },
    {
        "query": "data loss and data corruption incidents",
        "relevant": [17, 27, 3],
    },
    {
        "query": "deployment and configuration management failures",
        "relevant": [21, 25, 28, 12],
    },
    {
        "query": "capacity planning and resource exhaustion",
        "relevant": [10, 11, 20, 26, 36],
    },
    {
        "query": "DNS and name resolution problems",
        "relevant": [19, 24, 41],
    },
    {
        "query": "certificate and TLS issues",
        "relevant": [2, 42],
    },
    {
        "query": "incidents caused by race conditions or timing",
        "relevant": [6, 15, 22, 38],
    },
    {
        "query": "silent failures that went undetected",
        "relevant": [0, 5, 25, 33, 37, 40],
    },
    {
        "query": "problems during traffic spikes or peak load",
        "relevant": [4, 11, 26, 32],
    },
    {
        "query": "misconfigured limits thresholds and defaults",
        "relevant": [20, 24, 28, 34, 40, 43],
    },
    {
        "query": "incidents involving third party integrations",
        "relevant": [2, 9, 49],
    },
    {
        "query": "cache related incidents",
        "relevant": [19, 33, 41, 46],
    },
    {
        "query": "on-call and incident response process failures",
        "relevant": [30, 39],
    },
    {
        "query": "replication and data consistency issues",
        "relevant": [13, 16],
    },
    {
        "query": "cross-cutting incidents spanning auth and network",
        "relevant": [0, 8, 9, 40, 42],
    },
]


# ---------------------------------------------------------------------------
# Systems under test
# ---------------------------------------------------------------------------


def build_chromadb(embedder: SentenceTransformerEmbedder) -> chromadb.Collection:
    """Build ChromaDB collection with the same memories."""
    client = chromadb.Client()
    collection = client.create_collection(
        name="postmortems",
        metadata={"hnsw:space": "cosine"},
    )
    embeddings = [embedder.embed(m).tolist() for m in MEMORIES]
    collection.add(
        ids=[str(i) for i in range(len(MEMORIES))],
        documents=MEMORIES,
        embeddings=embeddings,
    )
    return collection


def build_imi_store(embedder: SentenceTransformerEmbedder) -> VectorStore:
    """Build IMI VectorStore with the same memories."""
    store = VectorStore()
    for i, mem in enumerate(MEMORIES):
        emb = embedder.embed(mem)
        node = MemoryNode(
            id=str(i),
            seed=mem,
            summary_orbital=mem[:30],
            summary_medium=mem[:80],
            summary_detailed=mem,
            embedding=emb,
            tags=[f"cluster_{i // 10}"],
            source="postmortem",
            created_at=time.time() - i * 3600,
        )
        store.add(node)
    return store


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------


def retrieve_chromadb(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[str]:
    """ChromaDB cosine retrieval."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results["ids"][0]


def retrieve_chromadb_reranked(
    collection: chromadb.Collection,
    query_embedding: list[float],
    embedder: SentenceTransformerEmbedder,
    query_text: str,
    top_k: int = 10,
    initial_k: int = 30,
) -> list[str]:
    """ChromaDB retrieval + embedding-based reranking.

    Stage 1: Retrieve top-initial_k by cosine
    Stage 2: Re-embed query with document context, re-score
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(initial_k, len(MEMORIES)),
        include=["documents", "embeddings"],
    )

    docs = results["documents"][0]
    ids = results["ids"][0]
    embs = results["embeddings"][0]

    # Rerank: score each doc embedding against query embedding more carefully
    # Simulate a reranker by using cross-attention proxy:
    # embed(query + doc_snippet) and compare to embed(doc)
    query_emb = np.array(query_embedding, dtype=np.float32)
    scores = []
    for doc, emb in zip(docs, embs):
        doc_emb = np.array(emb, dtype=np.float32)
        # Base cosine
        cosine = float(np.dot(query_emb, doc_emb))
        # Boost: embed the combined query+snippet and compare
        combined = f"{query_text}: {doc[:100]}"
        combined_emb = embedder.embed(combined)
        cross_score = float(np.dot(combined_emb, doc_emb))
        scores.append(0.5 * cosine + 0.5 * cross_score)

    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:top_k]]


def retrieve_imi(
    store: VectorStore,
    query_embedding: np.ndarray,
    top_k: int = 10,
    relevance_weight: float = 0.3,
) -> list[str]:
    """IMI VectorStore retrieval with relevance weighting."""
    results = store.search(query_embedding, top_k=top_k, relevance_weight=relevance_weight)
    return [node.id for node, score in results]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], relevant: list[int], k: int) -> float:
    """Fraction of relevant docs found in top-k."""
    relevant_set = {str(r) for r in relevant}
    found = sum(1 for r in retrieved[:k] if r in relevant_set)
    return found / len(relevant_set) if relevant_set else 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain."""
    relevant_set = {str(r) for r in relevant}

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / log2(i + 2)  # i+2 because position is 1-indexed

    # Ideal DCG
    idcg = sum(1.0 / log2(i + 2) for i in range(min(len(relevant_set), k)))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved: list[str], relevant: list[int]) -> float:
    """Mean Reciprocal Rank — position of first relevant result."""
    relevant_set = {str(r) for r in relevant}
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("  WS4 Q4: IMI navigate() vs ChromaDB vs ChromaDB + Reranker")
    print("=" * 80)

    print("\nInitializing embedder...")
    embedder = SentenceTransformerEmbedder()

    print("Building systems...")
    collection = build_chromadb(embedder)
    imi_store = build_imi_store(embedder)

    print(f"Dataset: {len(MEMORIES)} memories, {len(QUERIES)} queries\n")

    # Run all systems
    systems = {
        "ChromaDB": [],
        "ChromaDB+Rerank": [],
        "IMI navigate()": [],
    }

    for q in QUERIES:
        query_emb = embedder.embed(q["query"])

        systems["ChromaDB"].append(
            retrieve_chromadb(collection, query_emb.tolist(), top_k=10)
        )
        systems["ChromaDB+Rerank"].append(
            retrieve_chromadb_reranked(
                collection, query_emb.tolist(), embedder, q["query"], top_k=10
            )
        )
        systems["IMI navigate()"].append(
            retrieve_imi(imi_store, query_emb, top_k=10)
        )

    # Compute metrics
    print(f"{'System':<20} {'Recall@5':>10} {'Recall@10':>10} {'nDCG@5':>10} {'MRR':>10}")
    print("-" * 62)

    for sys_name, all_retrieved in systems.items():
        r5_scores = []
        r10_scores = []
        ndcg_scores = []
        mrr_scores = []

        for i, retrieved in enumerate(all_retrieved):
            relevant = QUERIES[i]["relevant"]
            r5_scores.append(recall_at_k(retrieved, relevant, 5))
            r10_scores.append(recall_at_k(retrieved, relevant, 10))
            ndcg_scores.append(ndcg_at_k(retrieved, relevant, 5))
            mrr_scores.append(mrr(retrieved, relevant))

        avg_r5 = np.mean(r5_scores)
        avg_r10 = np.mean(r10_scores)
        avg_ndcg = np.mean(ndcg_scores)
        avg_mrr = np.mean(mrr_scores)

        print(f"  {sys_name:<18} {avg_r5:>9.3f} {avg_r10:>10.3f} {avg_ndcg:>10.3f} {avg_mrr:>10.3f}")

    print("-" * 62)

    # Per-query breakdown for interesting cases
    print("\nPer-query Recall@5 (interesting divergences):")
    print(f"{'Query':<50} {'Chroma':>7} {'Rerank':>7} {'IMI':>7}")
    print("-" * 75)
    for i, q in enumerate(QUERIES):
        r5_chroma = recall_at_k(systems["ChromaDB"][i], q["relevant"], 5)
        r5_rerank = recall_at_k(systems["ChromaDB+Rerank"][i], q["relevant"], 5)
        r5_imi = recall_at_k(systems["IMI navigate()"][i], q["relevant"], 5)

        # Only show if there's meaningful divergence
        if max(r5_chroma, r5_rerank, r5_imi) - min(r5_chroma, r5_rerank, r5_imi) > 0.1:
            marker = " ← DIVERGENT"
        else:
            marker = ""
        print(f"  {q['query'][:48]:<48} {r5_chroma:>6.2f} {r5_rerank:>7.2f} {r5_imi:>7.2f}{marker}")

    # Verdict
    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)

    chroma_r5 = np.mean([recall_at_k(systems["ChromaDB"][i], QUERIES[i]["relevant"], 5) for i in range(len(QUERIES))])
    rerank_r5 = np.mean([recall_at_k(systems["ChromaDB+Rerank"][i], QUERIES[i]["relevant"], 5) for i in range(len(QUERIES))])
    imi_r5 = np.mean([recall_at_k(systems["IMI navigate()"][i], QUERIES[i]["relevant"], 5) for i in range(len(QUERIES))])

    if imi_r5 > rerank_r5 + 0.05:
        print(f"  IMI navigate() WINS (Recall@5: {imi_r5:.3f} vs Reranker: {rerank_r5:.3f})")
        print(f"  → Relevance weighting provides measurable advantage over pure embedding similarity")
    elif rerank_r5 > imi_r5 + 0.05:
        print(f"  ChromaDB+Reranker WINS (Recall@5: {rerank_r5:.3f} vs IMI: {imi_r5:.3f})")
        print(f"  → Affordances and relevance weighting need rethinking")
    else:
        print(f"  TIE (ChromaDB: {chroma_r5:.3f}, Reranker: {rerank_r5:.3f}, IMI: {imi_r5:.3f})")
        print(f"  → No system has clear advantage at this scale")
        print(f"  → IMI's value may be in zoom/affordance features, not raw retrieval")


if __name__ == "__main__":
    main()
