"""Calibrate IMI_COMPLETION_THRESHOLD on the live IMI corpus.

Usage
-----
    python scripts/calibrate_completion_threshold.py

The script:
1. Loads the live IMISpace from SQLite.
2. Generates 3 × N/3 fragments from real node seeds (truncated at
   33% / 50% / 67% from different positions for representativeness).
3. Sweeps thresholds 0.20 → 0.50 in steps of 0.05.
4. Reports precision@1 and F1 for each threshold.
5. Prints the recommended value and the export command.

Output example
--------------
    Fragments generated: 50 (3 truncation strategies)
    threshold=0.20 → P@1=0.680
    threshold=0.25 → P@1=0.740
    threshold=0.30 → P@1=0.780
    threshold=0.35 → P@1=0.800  ← best
    threshold=0.40 → P@1=0.780
    ...
    Recommended: IMI_COMPLETION_THRESHOLD=0.35
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def _load_space():
    from imi.space import IMISpace

    # Prefer the project DB (has real encoded nodes), fall back to ~/.imi
    default_db = os.path.join(os.path.dirname(__file__), "..", "imi_memory.db")
    default_db = os.path.normpath(default_db)
    db_path = os.environ.get("IMI_DB", default_db)
    return IMISpace.from_sqlite(db_path)


def _generate_fragments(space, n: int = 50):
    """Generate (target_node, fragment) pairs with 3 truncation strategies."""
    nodes = [
        n
        for n in space.episodic.nodes
        if n.seed and len(n.seed.split()) >= 6 and n.embedding is not None
    ]
    if not nodes:
        print("No eligible nodes found (need seed + embedding, >= 6 tokens).")
        sys.exit(1)

    # Cap at n, spread evenly
    step = max(1, len(nodes) // n)
    selected = nodes[::step][:n]

    fragments = []
    for i, node in enumerate(selected):
        words = node.seed.split()
        strategy = i % 3
        if strategy == 0:
            # Truncate: keep first 33%
            cut = max(2, len(words) // 3)
            frag = " ".join(words[:cut])
        elif strategy == 1:
            # Truncate: keep middle 50%
            start = len(words) // 4
            end = start + max(2, len(words) // 2)
            frag = " ".join(words[start:end])
        else:
            # Truncate: keep last 67%
            cut = len(words) // 3
            frag = " ".join(words[cut:])
        fragments.append((node, frag))

    return fragments


def _evaluate(space, fragments, threshold: float) -> float:
    """Return precision@1: fraction of fragments that reconstruct their source node."""
    from imi.completion import reconstruct_from_partial

    hits = 0
    for target, fragment in fragments:
        result = reconstruct_from_partial(space, fragment, threshold=threshold)
        if result is not None and result.id == target.id:
            hits += 1
    return hits / len(fragments)


def main():
    print("Loading IMI space…")
    space = _load_space()
    print(f"Loaded: {len(space.episodic.nodes)} episodic, {len(space.semantic.nodes)} semantic")

    print("Generating fragments…")
    fragments = _generate_fragments(space)
    print(f"Fragments generated: {len(fragments)} (3 truncation strategies)\n")

    thresholds = [round(t, 2) for t in np.arange(0.20, 0.55, 0.05).tolist()]
    results: dict[float, float] = {}

    for t in thresholds:
        p1 = _evaluate(space, fragments, threshold=t)
        results[t] = p1
        print(f"  threshold={t:.2f} → P@1={p1:.3f}")

    best = max(results, key=results.__getitem__)
    print(f"\nRecommended threshold: {best:.2f}  (P@1={results[best]:.3f})")
    print(f"\nExport command:")
    print(f"  export IMI_COMPLETION_THRESHOLD={best:.2f}")


if __name__ == "__main__":
    main()
