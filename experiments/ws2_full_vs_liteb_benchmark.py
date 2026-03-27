"""WS2 Benchmark: IMI Full vs IMI Lite-B vs RAG Puro

Compares three systems on the same 50-postmortem dataset from Q4:
1. RAG Puro (ChromaDB cosine)
2. IMI Lite-B (ZoomRAG: zoom + affordances over ChromaDB)
3. IMI Full (VectorStore + relevance weighting)

Metrics: Recall@5, Recall@10, nDCG@5, MRR, tokens consumed per zoom level

Usage:
    source .venv/bin/activate && PYTHONPATH=. python experiments/ws2_full_vs_liteb_benchmark.py
"""

from __future__ import annotations

import time
from math import log2

import chromadb
import numpy as np

from imi.embedder import SentenceTransformerEmbedder
from imi.lite import ZoomRAG
from imi.node import MemoryNode
from imi.store import VectorStore

# Reuse dataset from Q4
from experiments.ws4_imi_vs_rag_reranker import MEMORIES, QUERIES


# ---------------------------------------------------------------------------
# Metrics (same as Q4)
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant: list[int], k: int) -> float:
    relevant_set = {str(r) for r in relevant}
    found = sum(1 for r in retrieved_ids[:k] if r in relevant_set)
    return found / len(relevant_set) if relevant_set else 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant: list[int], k: int) -> float:
    relevant_set = {str(r) for r in relevant}
    dcg = sum(1.0 / log2(i + 2) for i, rid in enumerate(retrieved_ids[:k]) if rid in relevant_set)
    idcg = sum(1.0 / log2(i + 2) for i in range(min(len(relevant_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved_ids: list[str], relevant: list[int]) -> float:
    relevant_set = {str(r) for r in relevant}
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Build systems
# ---------------------------------------------------------------------------

def build_rag_pure(embedder):
    client = chromadb.Client()
    col = client.create_collection("rag_pure_bench", metadata={"hnsw:space": "cosine"})
    embeddings = [embedder.embed(m).tolist() for m in MEMORIES]
    col.add(
        ids=[str(i) for i in range(len(MEMORIES))],
        documents=MEMORIES,
        embeddings=embeddings,
    )
    return col


def build_liteb(embedder):
    zr = ZoomRAG(embedder=embedder, collection_prefix="liteb_bench")
    for i, mem in enumerate(MEMORIES):
        # Simulate LLM-generated zoom levels with truncation
        zr.ingest(
            mem,
            summary_orbital=mem[:30],
            summary_medium=mem[:80],
            summary_detailed=mem,
            seed=mem,
            node_id=str(i),
            tags=[f"cluster_{i // 10}"],
        )
    return zr


def build_imi_full(embedder):
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
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_rag(col, query_emb, top_k=10):
    results = col.query(query_embeddings=[query_emb.tolist()], n_results=top_k)
    return results["ids"][0]


def retrieve_liteb(zr, query, top_k=10):
    results = zr.search(query, zoom="medium", top_k=top_k)
    return [r["id"] for r in results]


def retrieve_imi(store, query_emb, top_k=10):
    results = store.search(query_emb, top_k=top_k, relevance_weight=0.3)
    return [node.id for node, score in results]


# ---------------------------------------------------------------------------
# Token budget analysis
# ---------------------------------------------------------------------------

TOKEN_ESTIMATES = {
    "orbital": 10,
    "medium": 40,
    "detailed": 100,
    "full": 200,
}


def tokens_for_zoom(n_results: int, zoom: str) -> int:
    return n_results * TOKEN_ESTIMATES.get(zoom, 40)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("  WS2: IMI Full vs IMI Lite-B vs RAG Puro")
    print("=" * 80)

    embedder = SentenceTransformerEmbedder()

    print("\nBuilding systems...")
    rag_col = build_rag_pure(embedder)
    liteb = build_liteb(embedder)
    imi_store = build_imi_full(embedder)

    print(f"Dataset: {len(MEMORIES)} memories, {len(QUERIES)} queries\n")

    # Retrieval comparison
    systems = {}
    for name, retrieve_fn in [
        ("RAG Puro", lambda q: retrieve_rag(rag_col, embedder.embed(q["query"]))),
        ("Lite-B", lambda q: retrieve_liteb(liteb, q["query"])),
        ("IMI Full", lambda q: retrieve_imi(imi_store, embedder.embed(q["query"]))),
    ]:
        results = [retrieve_fn(q) for q in QUERIES]
        systems[name] = results

    # Metrics table
    print(f"{'System':<15} {'Recall@5':>10} {'Recall@10':>10} {'nDCG@5':>10} {'MRR':>10}")
    print("-" * 58)

    for name, all_retrieved in systems.items():
        r5 = np.mean([recall_at_k(all_retrieved[i], QUERIES[i]["relevant"], 5) for i in range(len(QUERIES))])
        r10 = np.mean([recall_at_k(all_retrieved[i], QUERIES[i]["relevant"], 10) for i in range(len(QUERIES))])
        ndcg = np.mean([ndcg_at_k(all_retrieved[i], QUERIES[i]["relevant"], 5) for i in range(len(QUERIES))])
        m = np.mean([mrr(all_retrieved[i], QUERIES[i]["relevant"]) for i in range(len(QUERIES))])
        print(f"  {name:<13} {r5:>9.3f} {r10:>10.3f} {ndcg:>10.3f} {m:>10.3f}")

    print("-" * 58)

    # Token budget comparison
    print("\nToken budget per query (top-5 results):")
    print(f"{'Zoom Level':<12} {'Tokens':>8}  {'Use Case'}")
    print("-" * 50)
    for zoom, tok in TOKEN_ESTIMATES.items():
        total = tokens_for_zoom(5, zoom)
        use = {
            "orbital": "Quick scan / triage",
            "medium": "Standard context injection",
            "detailed": "Deep analysis",
            "full": "Full reconstruction (IMI only)",
        }[zoom]
        systems_avail = "All" if zoom != "full" else "IMI Full only"
        print(f"  {zoom:<10} {total:>6} tok  {use} [{systems_avail}]")

    # What Lite-B can/can't do
    print("\n" + "=" * 80)
    print("  FEATURE COMPARISON")
    print("=" * 80)
    features = [
        ("Vector retrieval", "Yes", "Yes", "Yes"),
        ("Multi-zoom levels", "No", "Yes", "Yes"),
        ("Affordance search", "No", "Yes", "Yes"),
        ("Relevance weighting (recency/freq)", "No", "No", "Yes"),
        ("Affect modulation", "No", "No", "Yes"),
        ("Surprise boost", "No", "No", "Yes"),
        ("Temporal navigation", "No", "No", "Yes"),
        ("Consolidation/dreaming", "No", "No", "Yes"),
        ("Full reconstruction (seed→LLM)", "No", "No", "Yes"),
        ("FTS5 hybrid search", "No", "No", "Yes (SQLite)"),
        ("Zero external deps", "No (Chroma)", "No (Chroma)", "Yes"),
        ("Lines of code", "~10", "~160", "~2000+"),
    ]
    print(f"  {'Feature':<38} {'RAG':>6} {'Lite-B':>8} {'Full':>8}")
    print("  " + "-" * 62)
    for feat, rag, lite, full in features:
        print(f"  {feat:<38} {rag:>6} {lite:>8} {full:>8}")

    # Verdict
    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)

    rag_r5 = np.mean([recall_at_k(systems["RAG Puro"][i], QUERIES[i]["relevant"], 5) for i in range(len(QUERIES))])
    lite_r5 = np.mean([recall_at_k(systems["Lite-B"][i], QUERIES[i]["relevant"], 5) for i in range(len(QUERIES))])
    imi_r5 = np.mean([recall_at_k(systems["IMI Full"][i], QUERIES[i]["relevant"], 5) for i in range(len(QUERIES))])

    print(f"\n  Retrieval quality: RAG={rag_r5:.3f}, Lite-B={lite_r5:.3f}, Full={imi_r5:.3f}")
    if abs(lite_r5 - imi_r5) < 0.05:
        print(f"  → Lite-B matches Full on raw retrieval")
        print(f"  → Full's value is in features Lite-B can't replicate:")
        print(f"    - Relevance weighting (recency/frequency/affect/surprise)")
        print(f"    - Consolidation/dreaming (episodic→semantic)")
        print(f"    - Full reconstruction via seed+context")
        print(f"    - Temporal navigation")
    else:
        delta = (imi_r5 - lite_r5) / lite_r5 * 100
        print(f"  → Full beats Lite-B by {delta:.1f}% on Recall@5")

    print(f"\n  Recommendation: Lite-B as entry point, Full for production agents")


if __name__ == "__main__":
    main()
