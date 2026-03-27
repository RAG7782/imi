"""WS4 Investigation: Clustering threshold 0.45 vs 0.80 vs 0.85

The dream() function in space.py uses 0.45, while find_clusters() defaults to 0.85
and run_maintenance() defaults to 0.80. This 2x gap likely produces garbage clusters
at 0.45 (grouping barely related memories) or misses valid clusters at 0.85.

This script:
1. Generates realistic memory nodes with known cluster structure
2. Runs find_clusters at thresholds from 0.30 to 0.95
3. Measures: precision (clusters are pure), recall (true groups found), cluster count
4. Determines the empirically optimal threshold

Usage:
    source .venv/bin/activate && python experiments/ws4_threshold_analysis.py
"""

from __future__ import annotations

import time

import numpy as np

from imi.maintain import find_clusters
from imi.node import MemoryNode
from imi.store import VectorStore


def make_clustered_nodes(
    n_clusters: int = 5,
    nodes_per_cluster: int = 6,
    noise_nodes: int = 10,
    dim: int = 384,
    intra_noise: float = 0.15,
) -> tuple[list[MemoryNode], dict[str, int]]:
    """Create nodes with known ground-truth cluster assignments.

    Each cluster shares a centroid direction. Intra-cluster noise controls
    how tight the cluster is (lower = tighter = higher cosine similarity).

    Returns (nodes, ground_truth) where ground_truth maps node_id -> cluster_id.
    Noise nodes get cluster_id = -1.
    """
    rng = np.random.default_rng(42)
    nodes = []
    ground_truth = {}

    topics = [
        ("kubernetes", "deployment", "pod", "container"),
        ("database", "query", "index", "postgres"),
        ("authentication", "token", "oauth", "session"),
        ("monitoring", "alert", "metric", "grafana"),
        ("network", "latency", "timeout", "dns"),
    ]

    for c in range(n_clusters):
        centroid = rng.standard_normal(dim).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        for i in range(nodes_per_cluster):
            noise = rng.standard_normal(dim).astype(np.float32) * intra_noise
            emb = centroid + noise
            emb /= np.linalg.norm(emb)

            topic = topics[c % len(topics)]
            node_id = f"c{c}_n{i}"
            node = MemoryNode(
                id=node_id,
                seed=f"Experience about {topic[0]}: {topic[i % len(topic)]} issue #{i}",
                summary_orbital=f"{topic[0]} {topic[1]}",
                summary_medium=f"{topic[0]} {topic[1]} problem with {topic[2]}",
                summary_detailed=f"Detailed: {topic[0]} had {topic[1]} issue related to {topic[2]} and {topic[3]}",
                embedding=emb,
                tags=[topic[0], topic[1]],
                source="experiment",
                mass=1.0,
                created_at=time.time() - (c * 3600 + i * 60),
            )
            nodes.append(node)
            ground_truth[node_id] = c

    # Noise nodes — random directions, not belonging to any cluster
    for i in range(noise_nodes):
        emb = rng.standard_normal(dim).astype(np.float32)
        emb /= np.linalg.norm(emb)
        node_id = f"noise_{i}"
        node = MemoryNode(
            id=node_id,
            seed=f"Random isolated memory {i}",
            summary_orbital=f"noise {i}",
            embedding=emb,
            tags=["noise"],
            source="experiment",
            created_at=time.time() - i * 120,
        )
        nodes.append(node)
        ground_truth[node_id] = -1

    return nodes, ground_truth


def evaluate_clustering(
    clusters: list[list[MemoryNode]],
    ground_truth: dict[str, int],
    n_true_clusters: int,
) -> dict[str, float]:
    """Evaluate clustering quality against ground truth.

    Metrics:
    - precision: fraction of cluster pairs that share ground truth label
    - recall: fraction of true cluster pairs that appear in same predicted cluster
    - purity: average dominant-label fraction per cluster
    - n_clusters: number of predicted clusters
    - noise_captured: noise nodes incorrectly clustered
    """
    # Precision: for each predicted cluster, what fraction of pairs are correct?
    total_pairs = 0
    correct_pairs = 0
    noise_in_clusters = 0

    for cluster in clusters:
        ids = [n.id for n in cluster]
        labels = [ground_truth.get(nid, -1) for nid in ids]
        noise_in_clusters += sum(1 for l in labels if l == -1)

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                total_pairs += 1
                if labels[i] == labels[j] and labels[i] != -1:
                    correct_pairs += 1

    precision = correct_pairs / total_pairs if total_pairs > 0 else 0.0

    # Purity: average of (dominant_label_count / cluster_size)
    purities = []
    for cluster in clusters:
        labels = [ground_truth.get(n.id, -1) for n in cluster]
        real_labels = [l for l in labels if l != -1]
        if real_labels:
            from collections import Counter
            most_common_count = Counter(real_labels).most_common(1)[0][1]
            purities.append(most_common_count / len(cluster))
        else:
            purities.append(0.0)
    purity = np.mean(purities) if purities else 0.0

    # Recall: what fraction of true same-cluster pairs are found?
    # Build predicted cluster assignment
    pred_assignment: dict[str, int] = {}
    for ci, cluster in enumerate(clusters):
        for n in cluster:
            pred_assignment[n.id] = ci

    true_pairs = 0
    found_pairs = 0
    all_ids = list(ground_truth.keys())
    for i in range(len(all_ids)):
        for j in range(i + 1, len(all_ids)):
            id_a, id_b = all_ids[i], all_ids[j]
            gt_a, gt_b = ground_truth[id_a], ground_truth[id_b]
            if gt_a == gt_b and gt_a != -1:
                true_pairs += 1
                if (
                    id_a in pred_assignment
                    and id_b in pred_assignment
                    and pred_assignment[id_a] == pred_assignment[id_b]
                ):
                    found_pairs += 1

    recall = found_pairs / true_pairs if true_pairs > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "purity": purity,
        "n_clusters": len(clusters),
        "noise_captured": noise_in_clusters,
        "total_clustered": sum(len(c) for c in clusters),
    }


def main():
    print("=" * 80)
    print("  WS4: Clustering Threshold Analysis")
    print("  dream() uses 0.45, find_clusters() default 0.85, run_maintenance() 0.80")
    print("=" * 80)

    # Real sentence embeddings (384d, normalized) have cosine similarity
    # ~0.70-0.90 for semantically related texts. intra_noise=0.05 simulates this.
    nodes, ground_truth = make_clustered_nodes(
        n_clusters=5, nodes_per_cluster=6, noise_nodes=10, intra_noise=0.05
    )
    n_true = 5
    print(f"\nDataset: {len(nodes)} nodes, {n_true} true clusters of 6, 10 noise nodes")

    # Print intra-cluster similarity stats
    print("\nIntra-cluster cosine similarity stats:")
    for c in range(n_true):
        cluster_nodes = [n for n in nodes if ground_truth.get(n.id) == c]
        embs = np.vstack([n.embedding for n in cluster_nodes])
        sims = embs @ embs.T
        # Upper triangle only
        mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
        intra_sims = sims[mask]
        print(f"  Cluster {c}: mean={intra_sims.mean():.3f}, "
              f"min={intra_sims.min():.3f}, max={intra_sims.max():.3f}")

    # Test thresholds
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                  0.75, 0.80, 0.85, 0.90, 0.95]

    print(f"\n{'Threshold':>10} {'Clusters':>9} {'Precision':>10} {'Recall':>8} "
          f"{'Purity':>8} {'Noise↓':>7} {'Clustered':>10}")
    print("-" * 72)

    store = VectorStore(nodes=nodes)
    best_f1 = 0
    best_threshold = 0

    for t in thresholds:
        clusters = find_clusters(store, similarity_threshold=t)
        metrics = evaluate_clustering(clusters, ground_truth, n_true)

        p, r = metrics["precision"], metrics["recall"]
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        marker = ""
        if t == 0.45:
            marker = " ← dream()"
        elif t == 0.80:
            marker = " ← run_maintenance()"
        elif t == 0.85:
            marker = " ← find_clusters()"

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

        print(f"  {t:>7.2f}   {metrics['n_clusters']:>7}   {p:>8.3f}   {r:>6.3f}   "
              f"{metrics['purity']:>6.3f}   {metrics['noise_captured']:>5}   "
              f"{metrics['total_clustered']:>8}{marker}")

    print("-" * 72)
    print(f"\n  Best F1 = {best_f1:.3f} at threshold = {best_threshold:.2f}")

    # Verdict
    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)
    if best_threshold >= 0.70:
        print(f"  0.45 is TOO LOW — produces impure clusters (garbage in, garbage out)")
        print(f"  Optimal threshold: {best_threshold:.2f}")
        print(f"  RECOMMENDATION: Change dream() default from 0.45 to {best_threshold:.2f}")
    elif best_threshold <= 0.50:
        print(f"  0.45 is approximately correct for this data distribution")
    else:
        print(f"  Optimal: {best_threshold:.2f} — between dream() and run_maintenance()")

    # Additional: what happens with different noise levels
    print("\n\nSensitivity to intra-cluster noise:")
    print(f"{'Noise σ':>10} {'Best threshold':>15} {'Best F1':>10} {'0.45 F1':>10} {'0.80 F1':>10}")
    print("-" * 60)
    for noise in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        nodes_n, gt_n = make_clustered_nodes(intra_noise=noise)
        store_n = VectorStore(nodes=nodes_n)
        best_f1_n = 0
        best_t_n = 0
        f1_045 = 0
        f1_080 = 0
        for t in thresholds:
            clusters_n = find_clusters(store_n, similarity_threshold=t)
            m = evaluate_clustering(clusters_n, gt_n, n_true)
            p, r = m["precision"], m["recall"]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1_n:
                best_f1_n = f1
                best_t_n = t
            if t == 0.45:
                f1_045 = f1
            if t == 0.80:
                f1_080 = f1
        delta = f1_045 - best_f1_n
        marker = " ← GOOD" if abs(delta) < 0.05 else " ← BAD" if delta < -0.1 else ""
        print(f"  {noise:>7.2f}   {best_t_n:>13.2f}   {best_f1_n:>8.3f}   "
              f"{f1_045:>8.3f}   {f1_080:>8.3f}{marker}")


if __name__ == "__main__":
    main()
