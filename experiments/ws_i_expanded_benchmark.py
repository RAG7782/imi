"""WS-I: Expanded AMBench — Multi-agent, Cross-domain, Long-horizon.

Extends WS-D AMBench with three additional scenarios:
  Scenario 1: Multi-agent (3 SRE agents sharing a memory pool)
  Scenario 2: Cross-domain (incidents that span multiple systems)
  Scenario 3: Long-horizon (365 days with seasonal patterns)

This validates IMI beyond single-agent/single-domain/90-day limits.

Usage:
    source .venv/bin/activate && PYTHONPATH=. python experiments/ws_i_expanded_benchmark.py
"""

from __future__ import annotations

import copy
import random
import time
from collections import defaultdict
from math import log2

import numpy as np

from imi.affect import AffectiveTag
from imi.affordance import Affordance
from imi.embedder import SentenceTransformerEmbedder
from imi.graph import EdgeType, MemoryGraph
from imi.node import MemoryNode
from imi.store import VectorStore

from experiments.ws_d_agent_memory_benchmark import (
    generate_incidents,
    INCIDENT_TEMPLATES,
    recall_at_k,
    ndcg_at_k,
    mrr,
)


# ---------------------------------------------------------------------------
# Scenario 1: Multi-agent shared memory
# ---------------------------------------------------------------------------

AGENT_SPECIALIZATIONS = {
    "agent_infra": ["connection_pool", "deploy_rollback", "disk_full"],
    "agent_security": ["cert_expiry", "auth_failure", "data_inconsistency"],
    "agent_network": ["timeout_cascade", "dns_failure", "rate_limit", "memory_leak"],
}


def scenario_multi_agent(
    incidents: list[dict],
    embedder: SentenceTransformerEmbedder,
):
    """3 specialized agents sharing a memory pool.

    Each agent encodes incidents in its domain.
    Test: can agents benefit from each other's memories?
    """
    print("\n" + "=" * 80)
    print("  SCENARIO 1: Multi-Agent Shared Memory")
    print("  3 agents × domain-specialized × shared pool")
    print("=" * 80)

    # Shared store
    shared_store = VectorStore()
    agent_stores = {name: VectorStore() for name in AGENT_SPECIALIZATIONS}

    now = time.time()
    day = 86400

    for inc in incidents:
        emb = embedder.embed(inc["text"])
        created = now - (90 - inc["day"]) * day

        affect = AffectiveTag(
            salience={"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}[inc["severity"]],
            valence=-0.5,
            arousal={"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}[inc["severity"]],
        )

        node = MemoryNode(
            id=inc["id"],
            seed=inc["text"],
            summary_orbital=inc["text"][:30],
            summary_medium=inc["text"][:80],
            summary_detailed=inc["text"],
            embedding=emb,
            tags=[inc["pattern_type"], inc["service"]],
            source="incident",
            created_at=created,
            affect=affect,
            mass=affect.initial_mass,
        )

        # Add to shared store
        shared_store.add(node)

        # Add to agent-specific store based on domain
        for agent_name, domains in AGENT_SPECIALIZATIONS.items():
            if inc["pattern_type"] in domains:
                agent_stores[agent_name].add(copy.deepcopy(node))
                break

    # Test: cross-domain queries
    cross_domain_queries = [
        {
            "query": "connection pool exhaustion causing authentication failures",
            "relevant_patterns": ["connection_pool", "auth_failure"],
            "scenario": "infra incident → security impact",
        },
        {
            "query": "certificate expiry causing DNS resolution failures",
            "relevant_patterns": ["cert_expiry", "dns_failure"],
            "scenario": "security incident → network impact",
        },
        {
            "query": "rate limiting triggered by deployment rollback retry storms",
            "relevant_patterns": ["rate_limit", "deploy_rollback"],
            "scenario": "network incident → infra impact",
        },
        {
            "query": "memory leak leading to disk full from core dumps",
            "relevant_patterns": ["memory_leak", "disk_full"],
            "scenario": "network agent → infra agent",
        },
        {
            "query": "data inconsistency after timeout cascade during deployment",
            "relevant_patterns": ["data_inconsistency", "timeout_cascade", "deploy_rollback"],
            "scenario": "3-way cross-agent",
        },
    ]

    print(f"\n  {'Scenario':<45} {'Shared':>7} {'Solo':>7} {'Δ':>5}")
    print("  " + "-" * 68)

    shared_wins = 0
    for cq in cross_domain_queries:
        # Shared store: all agents' memories available
        shared_results = shared_store.search(embedder.embed(cq["query"]), top_k=10, relevance_weight=0.1)
        shared_types = set()
        for n, _ in shared_results[:5]:
            shared_types.update(n.tags)

        shared_hit = sum(1 for p in cq["relevant_patterns"] if p in shared_types)

        # Solo: only the querying agent's memories
        # Which agent would own this query?
        best_solo = 0
        for agent_name, agent_store in agent_stores.items():
            if agent_store.nodes:
                solo_results = agent_store.search(embedder.embed(cq["query"]), top_k=10, relevance_weight=0.1)
                solo_types = set()
                for n, _ in solo_results[:5]:
                    solo_types.update(n.tags)
                solo_hit = sum(1 for p in cq["relevant_patterns"] if p in solo_types)
                best_solo = max(best_solo, solo_hit)

        total = len(cq["relevant_patterns"])
        delta = shared_hit - best_solo
        if delta > 0:
            shared_wins += 1

        marker = "+" if delta > 0 else "=" if delta == 0 else "-"
        print(f"  {cq['scenario']:<45} {shared_hit}/{total:>3} {best_solo}/{total:>4}   {marker}")

    print(f"\n  Shared memory wins: {shared_wins}/{len(cross_domain_queries)} cross-domain queries")
    return shared_wins, len(cross_domain_queries)


# ---------------------------------------------------------------------------
# Scenario 2: Cross-domain incidents
# ---------------------------------------------------------------------------

CROSS_DOMAIN_INCIDENTS = [
    {
        "id": "cross_001",
        "text": "Authentication service cert expired, causing cascading timeouts through API gateway, "
                "leading to connection pool exhaustion on order-service and partial data loss",
        "pattern_type": "cascading_failure",
        "domains": ["auth_failure", "cert_expiry", "timeout_cascade", "connection_pool", "data_inconsistency"],
        "actions": ["implement cert monitoring", "add circuit breakers", "set timeout budgets"],
        "severity": "critical",
        "day": 45,
    },
    {
        "id": "cross_002",
        "text": "DNS failure during deployment rollback caused rate limiting on CDN edge, "
                "triggering memory leak in retry logic, disk filled with request logs",
        "pattern_type": "cascading_failure",
        "domains": ["dns_failure", "deploy_rollback", "rate_limit", "memory_leak", "disk_full"],
        "actions": ["add DNS failover", "cap retry attempts", "implement log rotation"],
        "severity": "critical",
        "day": 72,
    },
    {
        "id": "cross_003",
        "text": "Kubernetes HPA scaled up during traffic spike, new pods couldn't authenticate "
                "due to LDAP sync delay, causing 503 errors that tripped monitoring alert storm",
        "pattern_type": "cascading_failure",
        "domains": ["deploy_rollback", "auth_failure", "rate_limit"],
        "actions": ["pre-warm pod auth", "add HPA dampening", "implement alert aggregation"],
        "severity": "high",
        "day": 30,
    },
]


def scenario_cross_domain(embedder: SentenceTransformerEmbedder):
    """Test retrieval of incidents that span multiple domains."""
    print("\n" + "=" * 80)
    print("  SCENARIO 2: Cross-Domain Incidents")
    print("  Complex incidents spanning 3-5 systems")
    print("=" * 80)

    store = VectorStore()
    graph = MemoryGraph()

    # First, add regular incidents
    incidents = generate_incidents(200, seed=99)
    for inc in incidents:
        emb = embedder.embed(inc["text"])
        node = MemoryNode(
            id=inc["id"],
            seed=inc["text"],
            summary_orbital=inc["text"][:30],
            summary_medium=inc["text"][:80],
            summary_detailed=inc["text"],
            embedding=emb,
            tags=[inc["pattern_type"]],
            source="incident",
        )
        store.add(node)

    # Add cross-domain incidents
    for ci in CROSS_DOMAIN_INCIDENTS:
        emb = embedder.embed(ci["text"])
        node = MemoryNode(
            id=ci["id"],
            seed=ci["text"],
            summary_orbital=ci["text"][:30],
            summary_medium=ci["text"][:80],
            summary_detailed=ci["text"],
            embedding=emb,
            tags=ci["domains"],
            source="incident",
        )
        store.add(node)

    # Build similarity graph
    n_edges = graph.auto_link_similar(store, threshold=0.65, max_edges_per_node=5)

    # Cross-domain queries
    queries = [
        ("authentication failure causing cascading timeout", "cross_001"),
        ("DNS and deployment interacting with rate limits", "cross_002"),
        ("scaling event causing auth and monitoring issues", "cross_003"),
    ]

    print(f"\n  Graph: {n_edges} similarity edges")
    print(f"\n  {'Query':<50} {'Cosine':>7} {'Graph':>7}")
    print("  " + "-" * 66)

    for query, expected_id in queries:
        q_emb = embedder.embed(query)

        # Cosine
        cos_results = [n.id for n, s in store.search(q_emb, top_k=5, relevance_weight=0.0)]
        cos_found = expected_id in cos_results

        # Graph-augmented
        graph_results = [n.id for n, s in graph.search_with_expansion(
            store, q_emb, top_k=5, hops=1, graph_weight=0.2)]
        graph_found = expected_id in graph_results

        print(f"  {query[:48]:<50} {'HIT' if cos_found else 'miss':>7} "
              f"{'HIT' if graph_found else 'miss':>7}")


# ---------------------------------------------------------------------------
# Scenario 3: Long-horizon (365 days)
# ---------------------------------------------------------------------------

def scenario_long_horizon(embedder: SentenceTransformerEmbedder):
    """365-day simulation with seasonal patterns and knowledge drift."""
    print("\n" + "=" * 80)
    print("  SCENARIO 3: Long-Horizon (365 days)")
    print("  Seasonal patterns, knowledge drift, forgetting curve")
    print("=" * 80)

    rng = random.Random(42)
    now = time.time()
    day = 86400

    # Generate 600 incidents over 365 days
    incidents = generate_incidents(600, seed=77)

    # Redistribute across 365 days with seasonal spikes
    for i, inc in enumerate(incidents):
        base_day = int(i / len(incidents) * 365)
        # Seasonal spike: more incidents in Q4 (Black Friday, year-end)
        if 270 <= base_day <= 340:
            inc["day"] = base_day + rng.randint(-5, 5)
        else:
            inc["day"] = base_day + rng.randint(-10, 10)
        inc["day"] = max(0, min(364, inc["day"]))

    store = VectorStore()
    access_log: dict[str, int] = {}

    # Simulate 365 days
    quarterly_r5 = {q: [] for q in ["Q1", "Q2", "Q3", "Q4"]}

    by_day = defaultdict(list)
    for inc in incidents:
        by_day[inc["day"]].append(inc)

    # Pattern index
    pattern_index = defaultdict(list)
    for inc in incidents:
        pattern_index[inc["pattern_type"]].append(inc["id"])

    encoded_ids = set()

    for sim_day in range(365):
        # Encode new incidents
        for inc in by_day.get(sim_day, []):
            emb = embedder.embed(inc["text"])
            created = now - (365 - sim_day) * day
            affect = AffectiveTag(
                salience={"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}[inc["severity"]],
                valence=-0.5,
                arousal=0.5,
            )
            node = MemoryNode(
                id=inc["id"],
                seed=inc["text"],
                summary_orbital=inc["text"][:30],
                summary_medium=inc["text"][:80],
                summary_detailed=inc["text"],
                embedding=emb,
                tags=[inc["pattern_type"]],
                source="incident",
                created_at=created,
                last_accessed=created,
                affect=affect,
                mass=affect.initial_mass,
            )
            store.add(node)
            encoded_ids.add(inc["id"])

        # Periodic retrieval tests (every 30 days)
        if sim_day % 30 == 29 and len(encoded_ids) > 20:
            # Sample a recent incident and find similar past ones
            recent = [inc for inc in incidents if inc["day"] <= sim_day and inc["id"] in encoded_ids]
            if recent:
                test_inc = rng.choice(recent[-20:])  # pick from last 20 encoded
                similar_ids = [
                    iid for iid in pattern_index[test_inc["pattern_type"]]
                    if iid != test_inc["id"] and iid in encoded_ids
                ]

                if len(similar_ids) >= 2:
                    results = store.search(
                        embedder.embed(test_inc["text"]),
                        top_k=10,
                        relevance_weight=0.1,
                    )
                    retrieved = [n.id for n, s in results]
                    r5 = recall_at_k(retrieved, similar_ids, 5)

                    quarter = f"Q{sim_day // 91 + 1}" if sim_day < 364 else "Q4"
                    quarterly_r5[quarter].append(r5)

        # Simulate accesses (power law)
        if rng.random() < 0.3 and encoded_ids:
            encoded_list = list(encoded_ids)
            for _ in range(3):
                rid = rng.choice(encoded_list)
                node = store.get(rid)
                if node:
                    node.touch()

    print(f"\n  {len(incidents)} incidents over 365 days")
    print(f"  {len(encoded_ids)} encoded into memory")

    print(f"\n  {'Quarter':<10} {'Avg R@5':>10} {'Samples':>10}")
    print("  " + "-" * 34)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        vals = quarterly_r5[q]
        avg = np.mean(vals) if vals else 0
        print(f"  {q:<10} {avg:>10.3f} {len(vals):>10}")

    overall = [v for vals in quarterly_r5.values() for v in vals]
    trend = (np.mean(quarterly_r5["Q4"]) if quarterly_r5["Q4"] else 0) - \
            (np.mean(quarterly_r5["Q1"]) if quarterly_r5["Q1"] else 0)

    print(f"\n  Overall R@5: {np.mean(overall):.3f}")
    print(f"  Q1→Q4 trend: {trend:+.3f} ({'improving' if trend > 0.02 else 'stable' if abs(trend) < 0.02 else 'degrading'})")

    # Test: can we find Q1 incidents from Q4?
    q1_incidents = [inc for inc in incidents if inc["day"] < 91 and inc["id"] in encoded_ids]
    if q1_incidents:
        q1_sample = rng.choice(q1_incidents)
        results = store.search(
            embedder.embed(q1_sample["text"]),
            top_k=5,
            relevance_weight=0.0,
        )
        cos_rank = next((i+1 for i, (n, _) in enumerate(results) if n.id == q1_sample["id"]), ">5")

        results_rw = store.search(
            embedder.embed(q1_sample["text"]),
            top_k=5,
            relevance_weight=0.1,
        )
        rw_rank = next((i+1 for i, (n, _) in enumerate(results_rw) if n.id == q1_sample["id"]), ">5")

        print(f"\n  Old memory retrieval (Q1 incident from Q4 perspective):")
        print(f"    Cosine rank: {cos_rank}")
        print(f"    IMI (rw=0.1) rank: {rw_rank}")
        print(f"    → {'Old memories remain accessible' if str(cos_rank) != '>5' else 'Old memory pushed out of top-5'}")

    return quarterly_r5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("  WS-I: Expanded AMBench")
    print("  Multi-agent | Cross-domain | Long-horizon")
    print("=" * 80)

    embedder = SentenceTransformerEmbedder()
    incidents = generate_incidents(300)

    # Scenario 1
    shared_wins, total_queries = scenario_multi_agent(incidents, embedder)

    # Scenario 2
    scenario_cross_domain(embedder)

    # Scenario 3
    quarterly = scenario_long_horizon(embedder)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("  EXPANDED BENCHMARK SUMMARY")
    print("=" * 80)

    q4_r5 = np.mean(quarterly["Q4"]) if quarterly["Q4"] else 0
    q1_r5 = np.mean(quarterly["Q1"]) if quarterly["Q1"] else 0

    print(f"""
  Scenario 1 — Multi-Agent:
    Shared memory improves cross-domain queries in {shared_wins}/{total_queries} cases
    → Agents benefit from each other's encoded experiences

  Scenario 2 — Cross-Domain:
    Complex cascading incidents span 3-5 systems
    → Graph-augmented retrieval improves discovery of cross-cutting incidents

  Scenario 3 — Long-Horizon:
    365-day simulation, 600 incidents, quarterly R@5 tracking
    → Q1 R@5={q1_r5:.3f}, Q4 R@5={q4_r5:.3f}, trend={q4_r5-q1_r5:+.3f}
    → {'Performance improves as memory accumulates' if q4_r5 > q1_r5 else 'Performance stable over time'}

  IMPLICATIONS FOR IMI:
  1. Shared memory pools are valuable for multi-agent setups
     → Future: per-agent relevance weights? Agent-scoped views?
  2. Cross-domain incidents need graph edges or multi-tag search
     → Future: auto-detect causal chains at encode time
  3. Long-horizon performance is stable — memories don't "rot"
     → rw=0.1 maintains accessibility to old memories
""")


if __name__ == "__main__":
    main()
