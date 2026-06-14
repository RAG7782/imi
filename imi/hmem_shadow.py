"""hmem_shadow.py — shadow-mode divergence logger for H-MEM retrieval.

Spec: ~/experimentos/specs/2026-06-14-imi-hmem-positional-index.md (§4.5 steps 4 & 7)

PORQUÊ shadow-mode comes BEFORE flipping the default (pre-mortem §4.5):
  The dangerous failure of a memory system is silent: hierarchical returns
  something plausible but not the right thing, and nobody notices for weeks. So
  the hierarchical path runs ALONGSIDE the trusted flat path, returns the flat
  result to the caller (zero production risk), and logs where the two disagree.
  Only when the logged divergence is small enough — for long enough — do we flip.

PROMOTION CRITERION (spec §4.5 step 7, agreed 2026-06-14):
  flat → hierarchical default is allowed ONLY when, over 2 weeks:
    • top-1 divergence < 2%  (ASSERT-1 endurecido: ≥98% same top-1 on the REAL store)
    • canary hit-rate == 100%  (imi/canary.py, every boot)
  This module only PRODUCES the divergence data. It never flips the default —
  that is a human decision made later with this log in hand.

This module does NOT decide ranking. It observes. The caller always gets the
flat result; the shadow result is logged, not served.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from imi.hmem_retrieve import recursive_retrieve

if TYPE_CHECKING:
    from imi.node import MemoryNode

_SHADOW_LOG = Path(
    os.getenv("IMI_HMEM_SHADOW_LOG", str(Path.home() / ".imi" / "hmem_shadow.jsonl"))
)

# Shadow logging is independent of the retrieval flag: you want divergence data
# WHILE the default is still flat. Default OFF (no overhead unless opted in).
SHADOW_ENABLED: bool = os.getenv("IMI_HMEM_SHADOW", "0") == "1"


def _top1_id(pairs: list) -> str | None:
    """First node id from a [(node, score), ...] flat-result list."""
    return pairs[0][0].id if pairs else None


def shadow_compare(
    query: str,
    query_embedding: np.ndarray,
    flat_results: list,
    stores: list,
    k_final: int = 10,
) -> dict:
    """Run hierarchical beside an already-computed flat result; log divergence.

    Parameters
    ----------
    query           : raw query text (for the log; never used for ranking here).
    query_embedding : the SAME embedding the flat path used (apples-to-apples).
    flat_results    : the flat [(node, score), ...] the caller is about to serve.
    stores          : VectorStores to walk for the hierarchical pass.
    k_final         : k both sides compare at.

    Returns
    -------
    The divergence record (also appended to the JSONL log). The caller ignores
    the return value for ranking — the flat result is what gets served.
    """
    t0 = time.monotonic()
    hmem = recursive_retrieve(query_embedding, stores, k_final=k_final)
    hier_ms = (time.monotonic() - t0) * 1000

    flat_top1 = _top1_id(flat_results)
    hier_top1 = hmem.hits[0].node.id if hmem.hits else None
    flat_ids = [p[0].id for p in flat_results[:k_final]]
    hier_ids = [h.node.id for h in hmem.hits[:k_final]]
    overlap = len(set(flat_ids) & set(hier_ids))

    record = {
        "ts": time.time(),
        "query": query[:120],
        "top1_match": flat_top1 == hier_top1,
        "flat_top1": flat_top1,
        "hier_top1": hier_top1,
        "topk_overlap": overlap,
        "k": k_final,
        "hier_ms": round(hier_ms, 2),
        # Diagnostics that explain a divergence without re-running:
        "tree_nodes_visited": hmem.tree_nodes_visited,
        "orphan_pool_size": hmem.orphan_pool_size,
        "layers_descended": hmem.layers_descended,
        "broken_ptrs": hmem.broken_ptrs,
        # Honesty flag: with an unpopulated tree EVERY hit comes via the orphan
        # pool, so top1_match here measures flat-vs-flat, not the tree. The
        # divergence gate is only meaningful once tree_nodes_visited > orphans.
        "tree_populated": hmem.tree_nodes_visited > 0,
    }

    try:
        _SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _SHADOW_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:  # disk/permission — log the offender, never block the query
        import sys

        sys.stderr.write(f"[hmem_shadow] could not append to {_SHADOW_LOG}: {e}\n")

    return record


def summarize(log_path: Path = _SHADOW_LOG, since_days: float | None = None) -> dict:
    """Aggregate the shadow log into the promotion-gate metrics.

    Returns top-1 divergence rate, mean overlap, p50/p95 hierarchical latency,
    and whether the tree was actually populated for the sampled window — so the
    human deciding the flip sees real signal, not flat-vs-flat noise.

    >>> summarize()["top1_divergence_rate"]   # want < 0.02 to promote
    0.014
    """
    if not log_path.exists():
        return {"error": f"no shadow log at {log_path}", "n": 0}

    cutoff = (time.time() - since_days * 86400) if since_days else 0.0
    rows = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a torn line, count the rest
            if r.get("ts", 0) >= cutoff:
                rows.append(r)

    n = len(rows)
    if n == 0:
        return {"n": 0, "note": "no rows in window"}

    mismatches = sum(1 for r in rows if not r.get("top1_match", False))
    populated = sum(1 for r in rows if r.get("tree_populated", False))
    lats = sorted(r.get("hier_ms", 0.0) for r in rows)
    overlaps = [r.get("topk_overlap", 0) for r in rows]

    def _pct(p: float) -> float:
        return lats[min(int(p * n), n - 1)] if lats else 0.0

    div_rate = mismatches / n
    return {
        "n": n,
        "top1_divergence_rate": round(div_rate, 4),
        "top1_match_rate": round(1 - div_rate, 4),
        "mean_topk_overlap": round(sum(overlaps) / n, 2),
        "hier_ms_p50": round(_pct(0.50), 2),
        "hier_ms_p95": round(_pct(0.95), 2),
        "rows_with_populated_tree": populated,
        # The gate is only valid once the tree is doing the work:
        "gate_meaningful": populated >= n * 0.5,
        "promote_ok": div_rate < 0.02 and populated >= n * 0.5,
        "promotion_criterion": "top1_divergence < 0.02 AND tree populated AND canary 100% for 2 weeks",
    }
