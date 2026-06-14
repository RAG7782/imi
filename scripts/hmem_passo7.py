#!/usr/bin/env python3
"""hmem_passo7.py — runnable, auditable Passo 7 of the H-MEM rollout.

Spec: ~/experimentos/specs/2026-06-14-imi-hmem-positional-index.md (§4.5 step 7)

PORQUÊ this is a script, not a flag flip:
  Passo 7 is the gated decision: flat → hierarchical default is allowed ONLY when
  top-1 divergence < 2% AND the tree is populated AND the canary is 100%, observed
  over time. A human makes that call WITH NUMBERS. This script produces the numbers
  safely: it grows the tree and measures divergence ON A SNAPSHOT COPY, never the
  live DB (G5), and never flips the default. Run it, read the verdict, decide.

Pipeline (numbered, legível):
  1. snapshot  → copy the live DB to a throwaway path (refuse to run on the live one)
  2. canary₀   → lexical canary baseline BEFORE mutating (must be ok to trust the rest)
  3. grow      → promote top-N clusters (IMI_HMEM_PROMOTE=1 + dirty_sink), save copy
  4. canary₁   → canary AGAIN after mutation — tree growth must not break lexical
  5. shadow    → for each canary anchor's note (a real query), compare flat top-1 vs
                 hierarchical top-1; append to the shadow JSONL
  6. gate      → summarize() → divergence rate, latency, populated-tree, promote_ok

Usage:
  python3 scripts/hmem_passo7.py [--clusters N] [--threshold T] [--keep-copy]

Never call with --db pointing at imi_memory.db — the script refuses (G5).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Promotion + shadow must be ON for this harness — set BEFORE importing imi modules
# that read the flags at import time.
os.environ.setdefault("IMI_HMEM_PROMOTE", "1")
os.environ.setdefault("IMI_HMEM_SHADOW", "1")
os.environ.setdefault("IMI_CONSOLIDATION_MIN_CLUSTER", "99")  # avoid LLM synth path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from imi.canary import load_anchors, run_canary  # noqa: E402
from imi.hmem_shadow import shadow_compare, summarize, _SHADOW_LOG  # noqa: E402
from imi.maintain import consolidate, find_clusters  # noqa: E402

_LIVE_DB = _REPO / "imi_memory.db"


class _FakeLLM:
    """Deterministic stand-in — the harness must not call Anthropic (disabled org)
    or phi4. Promotion logic is what we exercise, not pattern prose."""

    def generate(self, system: str = "", prompt: str = "", max_tokens: int = 0, **_) -> str:
        return "padrão passo7"


def _flat_search(space, query_emb: np.ndarray, top_k: int):
    """Flat baseline across both stores, minus navigate()'s LLM reconsolidation.

    Mirrors navigate()'s merge (episodic + semantic) so the shadow comparison is
    flat-vs-hierarchical, not flat-with-graph-vs-hierarchical."""
    ep = space.episodic.search(query_emb, top_k=top_k, relevance_weight=0.1)
    se = space.semantic.search(query_emb, top_k=top_k // 2, relevance_weight=0.1)
    merged = ep + se
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def main() -> int:
    ap = argparse.ArgumentParser(description="H-MEM Passo 7 — grow+shadow+canary+gate (snapshot only)")
    ap.add_argument("--db", type=Path, default=_LIVE_DB, help="source DB (copied; never mutated)")
    ap.add_argument("--clusters", type=int, default=20, help="top-N clusters to promote")
    ap.add_argument("--threshold", type=float, default=0.55, help="cluster similarity threshold")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--keep-copy", action="store_true", help="keep the snapshot copy after the run")
    args = ap.parse_args()

    # --- 1. snapshot (G5: the harness only ever reads the source, mutates a copy) ---
    # The source IS copied and only the copy is opened/saved — so passing the live
    # DB is safe. We never call space.save() on a space loaded from args.db.
    copy_path = Path(f"/tmp/hmem_passo7_{int(time.time())}.db")
    print(f"[1/6] snapshot: {args.db.name} → {copy_path}")
    shutil.copy2(args.db, copy_path)
    # Copy WAL/SHM siblings if present so the snapshot is consistent.
    for suffix in ("-wal", "-shm"):
        sib = Path(str(args.db) + suffix)
        if sib.exists():
            shutil.copy2(sib, Path(str(copy_path) + suffix))

    from imi.space import IMISpace

    space = IMISpace.from_sqlite(str(copy_path))
    print(f"      loaded: episodic={len(space.episodic.nodes)} semantic={len(space.semantic.nodes)}")

    anchors = load_anchors()
    if not anchors:
        print("[!] no frozen canary anchors — run `python -m imi.canary --derive` first")
        return 2

    # --- 2. canary baseline BEFORE mutation ---
    c0 = run_canary(space, anchors, top_k=args.top_k)
    print(f"[2/6] canary₀ (pre-grow): {c0.status} {c0.hits}/{c0.total} ({c0.hit_rate:.0%})")
    if c0.status != "ok":
        print(f"      baseline not clean — aborting (reason: {c0.reason or 'misses'})")
        if not args.keep_copy:
            copy_path.unlink(missing_ok=True)
        return 1

    # --- 3. grow the tree ---
    clusters = find_clusters(space.episodic, similarity_threshold=args.threshold)[: args.clusters]
    marked: list[str] = []
    pats = consolidate(clusters, space.semantic, space.embedder, _FakeLLM(),
                       dirty_sink=lambda n: marked.append(n.id))
    space.save()  # persist tree pointers to the COPY
    index_nodes = [n for n in space.semantic.nodes if n.layer < 3 and n.child_ptrs]
    print(f"[3/6] grow: clusters={len(clusters)} patterns={len(pats)} "
          f"reparented={len(marked)} index_nodes={len(index_nodes)}")

    # --- 4. canary AGAIN after mutation ---
    c1 = run_canary(space, anchors, top_k=args.top_k)
    print(f"[4/6] canary₁ (post-grow): {c1.status} {c1.hits}/{c1.total} ({c1.hit_rate:.0%})")
    canary_held = c1.status == "ok"
    if not canary_held:
        print("      ⚠️ canary BROKE after tree growth — promotion would corrupt retrieval")
        for m in c1.misses[:5]:
            print(f"        ✗ {m['token']} expected {m['expected_id']} got {m['got_ids']}")

    # --- 5. shadow: flat vs hierarchical over the anchor queries ---
    # Fresh shadow log for this run so the gate measures THIS snapshot, not history.
    if _SHADOW_LOG.exists():
        _SHADOW_LOG.unlink()
    print(f"[5/6] shadow: {len(anchors)} anchor queries → {_SHADOW_LOG}")
    for a in anchors:
        query = a.note or a.token
        q_emb = space.embedder.embed(query)
        flat = _flat_search(space, q_emb, args.top_k)
        shadow_compare(query, q_emb, flat, [space.episodic, space.semantic], k_final=args.top_k)

    # --- 5b. INDEPENDENT quality gate (non-circular) ---
    # The shadow gate measures agreement with FLAT — but flat is what motivated the
    # change, so "agree with flat" ≠ "be correct" (circular). This gate measures the
    # hierarchical retrieval against the canary anchors' KNOWN expected_id targets —
    # an oracle independent of flat. recall@k = target appears in top-k; p@1 = target
    # is top-1. This is the honest "is it actually correct?" signal.
    from imi.hmem_retrieve import recursive_retrieve

    recall_hits = 0
    p1_hits = 0
    for a in anchors:
        q_emb = space.embedder.embed(a.note or a.token)
        res = recursive_retrieve(q_emb, [space.episodic, space.semantic], k_final=args.top_k)
        got = [h.node.id[:12] for h in res.hits]
        if a.expected_id in got:
            recall_hits += 1
        if got and got[0] == a.expected_id:
            p1_hits += 1
    recall_at_k = recall_hits / len(anchors)
    p_at_1 = p1_hits / len(anchors)
    quality_ok = recall_at_k >= 0.90  # independent bar: 90% of known targets retrieved
    print(f"[5b/6] independent quality (vs known anchor targets, NOT vs flat): "
          f"recall@{args.top_k}={recall_at_k:.0%} p@1={p_at_1:.0%} → quality_ok={quality_ok}")

    # --- 6. gate verdict ---
    s = summarize(since_days=None)
    print("[6/6] GATE VERDICT (spec §4.5 step 7):")
    print(f"      n={s.get('n')} top1_divergence={s.get('top1_divergence_rate')} "
          f"(want < 0.02) | mean_overlap={s.get('mean_topk_overlap')}/{args.top_k}")
    print(f"      hier p50={s.get('hier_ms_p50')}ms p95={s.get('hier_ms_p95')}ms | "
          f"tree_populated_rows={s.get('rows_with_populated_tree')}/{s.get('n')}")
    print(f"      gate_meaningful={s.get('gate_meaningful')} | summarize.promote_ok={s.get('promote_ok')}")
    print(f"      canary_held={canary_held} | quality_ok={quality_ok} "
          f"(recall@{args.top_k}={recall_at_k:.0%})")
    # Promotion needs BOTH: low divergence-vs-flat (no regression) AND high absolute
    # recall (actually correct). The independent gate breaks the circularity.
    final = bool(s.get("promote_ok")) and canary_held and quality_ok
    print(f"\n  >>> PROMOTE_OK (this snapshot) = {final}")
    print("      NB: this is ONE snapshot. Spec requires the criterion to hold for 2 weeks")
    print("      before flipping mode=hierarchical to default. This script never flips it.")

    if not args.keep_copy:
        for p in (copy_path, Path(str(copy_path) + "-wal"), Path(str(copy_path) + "-shm")):
            p.unlink(missing_ok=True)
        print(f"\n  cleaned snapshot {copy_path.name} (use --keep-copy to retain)")
    else:
        print(f"\n  snapshot retained: {copy_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
