"""WS-B: Temporal Decay Test — Does relevance weighting help over time?

Simulates 90 days of agent usage with realistic access patterns:
- Agent encounters incidents over time
- Some incidents are revisited (recurring patterns)
- Query at day 90: "what should I check?" — relevance should boost recent/frequent

This tests the AGENT USE CASE for relevance features, unlike WS-A which tested
pure retrieval. The hypothesis: relevance weighting helps when the query is
"what's most useful NOW?" rather than "find the most similar memory."

Setup:
  - 100 incidents ingested over 90 simulated days
  - Power-law access patterns (some memories revisited many times)
  - 3 evaluation scenarios:
    A) Recent incident recurrence → should retrieve recent similar incident
    B) Frequently-accessed pattern → should surface well-known issue
    C) Forgotten old incident → should still be findable via semantic search

Usage:
    source .venv/bin/activate && PYTHONPATH=. python experiments/ws_b_temporal_decay.py
"""

from __future__ import annotations

import copy
import random
import time
from math import log2

import numpy as np

from imi.affect import AffectiveTag
from imi.embedder import SentenceTransformerEmbedder
from imi.node import MemoryNode
from imi.store import VectorStore
from imi.affordance import Affordance

from experiments.ws3_validation_framework import (
    DOMAINS,
    recall_at_k,
    ndcg_at_k,
    mrr_score,
)

# ---------------------------------------------------------------------------
# Temporal simulation
# ---------------------------------------------------------------------------

def build_temporal_dataset(embedder: SentenceTransformerEmbedder) -> list[MemoryNode]:
    """Build dataset with realistic temporal distribution over 90 days."""
    nodes = []
    rng = random.Random(42)
    now = time.time()
    day = 86400

    all_incidents = []
    for domain, data in DOMAINS.items():
        for local_idx, (text, actions) in enumerate(data["incidents"]):
            all_incidents.append((domain, local_idx, text, actions))

    # Shuffle and assign creation dates across 90 days
    rng.shuffle(all_incidents)

    for i, (domain, local_idx, text, actions) in enumerate(all_incidents):
        node_id = f"{domain}_{local_idx:02d}"

        # Spread across 90 days (earliest = 90 days ago, latest = 1 day ago)
        days_ago = 90 - (i / len(all_incidents)) * 89
        created = now - days_ago * day

        emb = embedder.embed(text)
        affordances = [
            Affordance(action=a, confidence=0.8, conditions=domain, domain=domain)
            for a in actions
        ]

        # Affect based on domain
        domain_affects = {
            "auth": AffectiveTag(salience=0.8, valence=-0.6, arousal=0.8),
            "database": AffectiveTag(salience=0.7, valence=-0.5, arousal=0.7),
            "infrastructure": AffectiveTag(salience=0.5, valence=-0.3, arousal=0.5),
            "monitoring": AffectiveTag(salience=0.4, valence=-0.2, arousal=0.3),
            "network": AffectiveTag(salience=0.6, valence=-0.4, arousal=0.6),
        }
        affect = domain_affects.get(domain, AffectiveTag())

        node = MemoryNode(
            id=node_id,
            seed=text,
            summary_orbital=text[:30],
            summary_medium=text[:80],
            summary_detailed=text,
            embedding=emb,
            tags=[domain, f"cluster_{domain}"],
            source="postmortem",
            affordances=affordances,
            created_at=created,
            last_accessed=created,
            access_count=0,
            affect=affect,
            mass=affect.initial_mass,
        )
        nodes.append(node)

    return nodes


def simulate_access_patterns(
    nodes: list[MemoryNode],
    days: int = 90,
    accesses_per_day: int = 5,
) -> dict[str, int]:
    """Simulate agent accessing memories over time.

    Returns access log: {node_id: total_accesses}
    """
    rng = random.Random(42)
    now = time.time()
    day = 86400
    node_map = {n.id: n for n in nodes}
    access_log: dict[str, int] = {}

    # Some incidents are "recurring patterns" — accessed repeatedly
    recurring_ids = rng.sample([n.id for n in nodes], k=15)
    # Some are "recent hot" — only accessed in last 7 days
    recent_hot_ids = [n.id for n in nodes if (now - n.created_at) < 7 * day][:10]

    for sim_day in range(days):
        current_time = now - (days - sim_day) * day

        # Daily accesses: mix of recurring, recent, and random
        for _ in range(accesses_per_day):
            roll = rng.random()
            if roll < 0.4 and recurring_ids:
                # 40%: access a recurring pattern
                node_id = rng.choice(recurring_ids)
            elif roll < 0.7 and recent_hot_ids and sim_day > days - 7:
                # 30% in last week: access recent hot
                node_id = rng.choice(recent_hot_ids)
            else:
                # 30%: random access from available nodes (created before this day)
                available = [n for n in nodes if n.created_at <= current_time]
                if available:
                    node_id = rng.choice(available).id
                else:
                    continue

            node = node_map.get(node_id)
            if node:
                node.access_count += 1
                node.last_accessed = current_time + rng.uniform(0, day)
                access_log[node_id] = access_log.get(node_id, 0) + 1

    return access_log


# ---------------------------------------------------------------------------
# Evaluation scenarios
# ---------------------------------------------------------------------------

TEMPORAL_QUERIES = [
    # Scenario A: Recent recurrence — agent had a similar incident recently
    {
        "name": "A: Recent DNS failure (should rank recent similar higher)",
        "query": "DNS resolution failing intermittently for internal services",
        "prefer_recent": True,
        "relevant_domains": ["network"],
    },
    {
        "name": "A: Auth token issue this week",
        "query": "users getting 401 errors after token refresh",
        "prefer_recent": True,
        "relevant_domains": ["auth"],
    },
    {
        "name": "A: Database connection problems just happened",
        "query": "connection pool running out of available connections",
        "prefer_recent": True,
        "relevant_domains": ["database"],
    },
    # Scenario B: Well-known pattern — frequently accessed
    {
        "name": "B: Recurring deployment issue (frequent pattern)",
        "query": "pod failing to start after deployment",
        "prefer_frequent": True,
        "relevant_domains": ["infrastructure"],
    },
    {
        "name": "B: Known monitoring blind spot",
        "query": "alerts not firing for actual outage",
        "prefer_frequent": True,
        "relevant_domains": ["monitoring"],
    },
    # Scenario C: Old but relevant — should still be findable
    {
        "name": "C: Old cert expiry incident (semantic only)",
        "query": "TLS certificate validation failing",
        "prefer_old": True,
        "relevant_domains": ["auth", "network"],
    },
    {
        "name": "C: Historic data corruption (forgotten)",
        "query": "character encoding corruption in user data",
        "prefer_old": True,
        "relevant_domains": ["database"],
    },
    # Scenario D: Cross-domain — tests if relevance helps or hurts multi-hop
    {
        "name": "D: Cascading failure across systems",
        "query": "one service failure causing cascade across multiple systems",
        "relevant_domains": ["auth", "database", "infrastructure", "network"],
    },
    {
        "name": "D: Timeout chain investigation",
        "query": "timeout mismatches causing request failures through proxy chain",
        "relevant_domains": ["network", "infrastructure"],
    },
]


def evaluate_temporal(
    nodes: list[MemoryNode],
    embedder: SentenceTransformerEmbedder,
    relevance_weight: float,
    top_k: int = 10,
) -> list[dict]:
    """Evaluate retrieval with temporal awareness."""
    store = VectorStore()
    for n in nodes:
        store.add(n)

    results = []
    for tq in TEMPORAL_QUERIES:
        q_emb = embedder.embed(tq["query"])
        search_results = store.search(q_emb, top_k=top_k, relevance_weight=relevance_weight)

        retrieved_nodes = [(n, s) for n, s in search_results]
        retrieved_domains = [n.tags[0] for n, s in retrieved_nodes]

        # Domain relevance
        expected = set(tq["relevant_domains"])
        domain_hits_5 = sum(1 for d in retrieved_domains[:5] if d in expected)
        domain_hits_10 = sum(1 for d in retrieved_domains[:10] if d in expected)

        # Temporal analysis of results
        now = time.time()
        ages_days = [(now - n.created_at) / 86400 for n, s in retrieved_nodes[:5]]
        accesses = [n.access_count for n, s in retrieved_nodes[:5]]

        results.append({
            "name": tq["name"],
            "domain_precision@5": domain_hits_5 / 5,
            "domain_precision@10": domain_hits_10 / 10,
            "avg_age_days_top5": np.mean(ages_days) if ages_days else 0,
            "avg_accesses_top5": np.mean(accesses) if accesses else 0,
            "top1_domain": retrieved_domains[0] if retrieved_domains else "?",
            "top1_score": search_results[0][1] if search_results else 0,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("  WS-B: Temporal Decay Test")
    print("  90-day agent simulation with access patterns")
    print("=" * 80)

    embedder = SentenceTransformerEmbedder()

    print("\nBuilding temporal dataset...")
    nodes = build_temporal_dataset(embedder)
    print(f"  {len(nodes)} incidents across 90 simulated days")

    # Simulate access patterns
    print("\nSimulating 90 days of agent access...")
    nodes_with_access = copy.deepcopy(nodes)
    access_log = simulate_access_patterns(nodes_with_access)

    accessed_count = sum(1 for v in access_log.values() if v > 0)
    total_accesses = sum(access_log.values())
    print(f"  {accessed_count} memories accessed, {total_accesses} total accesses")
    print(f"  Most accessed: {max(access_log.values())} times")

    # Access distribution
    access_counts = [n.access_count for n in nodes_with_access]
    print(f"  Access distribution: mean={np.mean(access_counts):.1f}, "
          f"median={np.median(access_counts):.0f}, max={max(access_counts)}")

    # --- Test 1: With vs Without access patterns ---
    print("\n" + "-" * 80)
    print("  TEST 1: Impact of access patterns on retrieval")
    print("-" * 80)

    rw_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    print(f"\n  === Without access patterns (fresh nodes) ===")
    print(f"  {'rw':>6} {'DomPrec@5':>10} {'DomPrec@10':>11} {'AvgAge@5':>10} {'AvgAcc@5':>10}")
    print("  " + "-" * 50)

    for rw in rw_values:
        results = evaluate_temporal(nodes, embedder, rw)
        avg_dp5 = np.mean([r["domain_precision@5"] for r in results])
        avg_dp10 = np.mean([r["domain_precision@10"] for r in results])
        avg_age = np.mean([r["avg_age_days_top5"] for r in results])
        avg_acc = np.mean([r["avg_accesses_top5"] for r in results])
        print(f"  {rw:>5.2f} {avg_dp5:>10.3f} {avg_dp10:>11.3f} {avg_age:>9.1f}d {avg_acc:>10.1f}")

    print(f"\n  === With 90 days of access patterns ===")
    print(f"  {'rw':>6} {'DomPrec@5':>10} {'DomPrec@10':>11} {'AvgAge@5':>10} {'AvgAcc@5':>10}")
    print("  " + "-" * 50)

    for rw in rw_values:
        results = evaluate_temporal(nodes_with_access, embedder, rw)
        avg_dp5 = np.mean([r["domain_precision@5"] for r in results])
        avg_dp10 = np.mean([r["domain_precision@10"] for r in results])
        avg_age = np.mean([r["avg_age_days_top5"] for r in results])
        avg_acc = np.mean([r["avg_accesses_top5"] for r in results])
        print(f"  {rw:>5.2f} {avg_dp5:>10.3f} {avg_dp10:>11.3f} {avg_age:>9.1f}d {avg_acc:>10.1f}")

    # --- Test 2: Per-scenario analysis ---
    print("\n" + "-" * 80)
    print("  TEST 2: Per-scenario analysis (rw=0.0 vs rw=0.15 vs rw=0.30)")
    print("-" * 80)

    for rw in [0.0, 0.15, 0.30]:
        print(f"\n  --- rw={rw:.2f} ---")
        results = evaluate_temporal(nodes_with_access, embedder, rw)
        for r in results:
            print(f"    {r['name'][:55]:<55} "
                  f"DP@5={r['domain_precision@5']:.2f} "
                  f"age={r['avg_age_days_top5']:.0f}d "
                  f"acc={r['avg_accesses_top5']:.0f}")

    # --- Test 3: Recency bias measurement ---
    print("\n" + "-" * 80)
    print("  TEST 3: Recency bias — do recent memories get unfairly boosted?")
    print("-" * 80)

    # Query about an OLD topic and check if old memories surface
    old_query = "TLS certificate validation failing"  # Scenario C
    q_emb = embedder.embed(old_query)

    now = time.time()
    print(f"\n  Query: '{old_query}'")
    print(f"  {'rw':>6} {'Top result':>50} {'Age(d)':>8} {'Accesses':>10}")
    print("  " + "-" * 78)

    for rw in [0.0, 0.10, 0.20, 0.30]:
        store = VectorStore()
        for n in nodes_with_access:
            store.add(n)
        results = store.search(q_emb, top_k=5, relevance_weight=rw)
        if results:
            top_node, top_score = results[0]
            age = (now - top_node.created_at) / 86400
            print(f"  {rw:>5.2f} {top_node.seed[:50]:>50} {age:>7.0f} {top_node.access_count:>10}")

    # --- Conclusions ---
    print("\n" + "=" * 80)
    print("  CONCLUSIONS")
    print("=" * 80)

    # Compare rw=0 vs rw=0.15 on access-enriched data
    results_0 = evaluate_temporal(nodes_with_access, embedder, 0.0)
    results_15 = evaluate_temporal(nodes_with_access, embedder, 0.15)

    dp5_0 = np.mean([r["domain_precision@5"] for r in results_0])
    dp5_15 = np.mean([r["domain_precision@5"] for r in results_15])
    delta = dp5_15 - dp5_0

    # Check scenario-specific: does rw help "recent recurrence" queries?
    recent_scenarios_0 = [r for r in results_0 if "Recent" in r["name"] or "Recurring" in r["name"]]
    recent_scenarios_15 = [r for r in results_15 if "Recent" in r["name"] or "Recurring" in r["name"]]
    recent_dp5_0 = np.mean([r["domain_precision@5"] for r in recent_scenarios_0]) if recent_scenarios_0 else 0
    recent_dp5_15 = np.mean([r["domain_precision@5"] for r in recent_scenarios_15]) if recent_scenarios_15 else 0

    old_scenarios_0 = [r for r in results_0 if "Old" in r["name"] or "Historic" in r["name"]]
    old_scenarios_15 = [r for r in results_15 if "Old" in r["name"] or "Historic" in r["name"]]
    old_dp5_0 = np.mean([r["domain_precision@5"] for r in old_scenarios_0]) if old_scenarios_0 else 0
    old_dp5_15 = np.mean([r["domain_precision@5"] for r in old_scenarios_15]) if old_scenarios_15 else 0

    print(f"""
  Overall domain precision@5:
    rw=0.00: {dp5_0:.3f}
    rw=0.15: {dp5_15:.3f}  (delta: {delta:+.3f})

  Recent/recurring scenarios:
    rw=0.00: {recent_dp5_0:.3f}
    rw=0.15: {recent_dp5_15:.3f}  (delta: {recent_dp5_15 - recent_dp5_0:+.3f})

  Old/forgotten scenarios:
    rw=0.00: {old_dp5_0:.3f}
    rw=0.15: {old_dp5_15:.3f}  (delta: {old_dp5_15 - old_dp5_0:+.3f})

  KEY FINDING: Relevance weighting {'helps' if delta > 0 else 'hurts'} overall domain precision
  by {abs(delta):.3f} at rw=0.15.

  For recent/recurring patterns: {'+' if recent_dp5_15 > recent_dp5_0 else '-'}{abs(recent_dp5_15 - recent_dp5_0):.3f}
  For old/forgotten memories:    {'+' if old_dp5_15 > old_dp5_0 else '-'}{abs(old_dp5_15 - old_dp5_0):.3f}

  RECOMMENDATION: {'Use rw=0.10-0.15 for agent scenarios (slight temporal boost without losing too much semantic quality)' if delta >= -0.05 else 'Even for agent scenarios, rw > 0.10 degrades quality. Prefer rw=0.0-0.05.'}
""")


if __name__ == "__main__":
    main()
