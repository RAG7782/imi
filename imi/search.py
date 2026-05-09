"""IMI Hybrid Scorer (A2) — 6-factor retrieval scoring.

Replaces the single cosine-similarity score used in navigate() when
IMI_HYBRID_SCORER=1 is set. Feature-flagged for instant rollback.

Factors
-------
F1  Semantic similarity  cosine(query_emb, node_emb)           weight 0.40
F2  Tag match            Jaccard over node.tags vs query_tags  weight 0.20
F3  Recency              half-life 20 days from created_at     weight 0.15
F4  Resonance            access_count (pre-computed, cached)   weight 0.10
F5  Salience             affect.salience                       weight 0.10
F6  Graph degree         pre-computed neighbour count / 10     weight 0.05

Design decisions
----------------
- F4 and F6 are PRE-COMPUTED on the node at write/access time, NOT at query
  time — so they add zero per-node overhead during navigate().
- BM25 was explicitly rejected: seeds are SDE-compressed (~80 token dense
  text); term-frequency assumptions of BM25 do not hold on this corpus.
- Weights sum to 1.0. Override via IMI_HYBRID_WEIGHTS env var (CSV of 6
  floats, e.g. "0.5,0.2,0.1,0.1,0.05,0.05").

Rollback
--------
    export IMI_HYBRID_SCORER=0   # instant revert, no migration needed
"""
from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from imi.node import MemoryNode

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

HYBRID_SCORER_ENABLED: bool = os.getenv("IMI_HYBRID_SCORER", "0") == "1"

# ---------------------------------------------------------------------------
# Weights (configurable)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = [0.40, 0.20, 0.15, 0.10, 0.10, 0.05]


def _load_weights() -> list[float]:
    raw = os.getenv("IMI_HYBRID_WEIGHTS", "")
    if raw:
        try:
            parts = [float(x) for x in raw.split(",")]
            if len(parts) == 6:
                total = sum(parts)
                return [p / total for p in parts]  # normalise to sum=1
        except ValueError:
            pass
    return _DEFAULT_WEIGHTS


_WEIGHTS: list[float] = _load_weights()


# ---------------------------------------------------------------------------
# Pre-compute helpers (called at write / access time, NOT at query time)
# ---------------------------------------------------------------------------

def update_cached_resonance(node: "MemoryNode") -> None:
    """Update node._cached_resonance from current access_count.

    Call inside MemoryNode.touch() or after encode.
    Normalised: access_count=10 → 1.0 (soft cap via min).
    """
    node._cached_resonance = min(1.0, node.access_count / 10.0)


def update_cached_graph_degree(node: "MemoryNode", degree: int) -> None:
    """Update node._cached_graph_degree after graph edge addition/removal.

    Call from MemoryGraph.add_edge() / remove_edge().
    Normalised: degree=10 → 1.0.
    """
    node._cached_graph_degree = min(1.0, degree / 10.0)


# ---------------------------------------------------------------------------
# Core hybrid scorer
# ---------------------------------------------------------------------------

def hybrid_score(
    node: "MemoryNode",
    query_embedding: np.ndarray,
    query_tags: set[str],
    now: float | None = None,
) -> float:
    """Compute 6-factor hybrid score for a node given a query.

    Parameters
    ----------
    node:            MemoryNode to score.
    query_embedding: L2-normalised query vector.
    query_tags:      Set of lowercase tag strings extracted from the query
                     (derived from the active tags in the query context or
                     passed explicitly from navigate()).
    now:             Current epoch (injected for testability; defaults to
                     time.time()).

    Returns
    -------
    float in [0, 1] — higher is more relevant.
    """
    if now is None:
        now = time.time()

    # F1 — Semantic (cosine; embeddings are L2-normalised by the embedder)
    if node.embedding is not None and query_embedding is not None:
        f1 = float(np.dot(node.embedding, query_embedding))
        f1 = max(0.0, min(1.0, f1))
    else:
        f1 = 0.0

    # F2 — Tag match (Jaccard: |intersection| / |union|)
    # When tags are absent on the node, fall back to soft token overlap on the
    # node seed (SDE text) vs query terms — preserves signal in tag-sparse corpora.
    node_tags = {t.lower() for t in node.tags} if node.tags else set()
    if query_tags and node_tags:
        union = query_tags | node_tags
        f2 = len(query_tags & node_tags) / len(union)
    elif query_tags and node.seed:
        # Soft fallback: token overlap on seed text
        seed_tokens = {w.lower() for w in node.seed.split() if len(w) > 3}
        if seed_tokens:
            overlap = query_tags & seed_tokens
            f2 = len(overlap) / len(query_tags) * 0.5  # half-weight for soft match
        else:
            f2 = 0.0
    else:
        f2 = 0.0

    # F3 — Recency (half-life 20 days from created_at)
    age_days = (now - node.created_at) / 86400
    f3 = 1.0 / (1.0 + 0.05 * age_days)

    # F4 — Resonance (pre-computed; fallback to live calc if attribute absent)
    f4 = getattr(node, "_cached_resonance", min(1.0, node.access_count / 10.0))

    # F5 — Salience (from affect)
    f5 = node.affect.salience if node.affect else 0.5

    # F6 — Graph degree (pre-computed; fallback to 0 if attribute absent)
    f6 = getattr(node, "_cached_graph_degree", 0.0)

    w = _WEIGHTS
    return w[0]*f1 + w[1]*f2 + w[2]*f3 + w[3]*f4 + w[4]*f5 + w[5]*f6
