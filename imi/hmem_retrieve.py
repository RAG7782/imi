"""hmem_retrieve.py — H-MEM recursive top-down TopK retrieval.

Spec: ~/experimentos/specs/2026-06-14-imi-hmem-positional-index.md (§3.3, §4)
Source: Sun & Zeng, H-MEM, arXiv:2507.22925 (§3.2).

PORQUÊ this module exists (not just "what"):
  Flat search is O(N·D) — it compares the query against every node. As the IMI
  store grows (it is the continuity thread across sessions), latency grows linear
  with N and precision degrades (irrelevants compete for attention). H-MEM routes
  the search down a parent→child tree: only the children of surviving nodes are
  expanded at each layer. Comparisons drop from O(N·D) toward O((a+k·fanout)·D).

  This is the LACUNA the spec fills — the recursive descent (§3.3). The tree it
  walks (layer / parent_id / child_ptrs on MemoryNode) is populated later, in the
  consolidation pass (§3.4). Until then the tree is empty and the ORPHAN FLAT POOL
  (ASSERT-6) carries 100% of the search — by design, not by accident.

Numbered pipeline (§3.3, fiel ao paper):
  input  → q, k_final
  step 0 → L0 (Domain): TopK_topo(sim(q, domain_nodes))         [k_topo ≥ 3·k_final]
  step l → for each survivor, expand only its child_ptrs, keep TopK by sim(q, child)
  repeat → until L3 (Episode = real content)
  output → top-k episodes, each with a confidence weight (ASSERT-5)

Hardened by the pre-mortem (every guard traces to a numbered cause):
  CHECK-2  cycle guard: a visited-set backstops the recursion even if the tree
           accidentally has a cycle (the writer also validates acyclicity, but the
           reader never trusts that — pre-mortem cause 5).
  CHECK-3  broken pointer → leaf: a child_ptr to a node that no longer exists is
           dropped and the parent is treated as a leaf for that branch, never a
           crash (pre-mortem cause 3: ptr → dead node post-reconsolidation).
  ASSERT-6 orphan flat pool: nodes with no layer/child_ptrs (legacy, pre-migration)
           are ALWAYS swept by a parallel flat search and merged in. An orphan can
           never fall out of the result set (pre-mortem cause 1, escore 25).
  ASSERT-7 k_topo ≥ 3·k_final: the Domain layer keeps a wider beam so the right
           branch is not pruned at the top (pre-mortem cause 6 / R1).

Rollback / activation:
    export IMI_HMEM_RETRIEVAL=1   # opt-in; default OFF (flat behaviour unchanged)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from imi.node import MemoryNode

# ---------------------------------------------------------------------------
# Feature flag — default OFF. Hierarchical search never runs as default until
# the spec §4.5 step-7 promotion criterion is met (shadow divergence <2% top-1
# AND canary 100% for 2 weeks). This flag only gates the explicit opt-in path.
# ---------------------------------------------------------------------------

HMEM_RETRIEVAL_ENABLED: bool = os.getenv("IMI_HMEM_RETRIEVAL", "0") == "1"

# k_topo multiplier (ASSERT-7). The Domain layer beam is this many times wider
# than the final k, so the correct branch is not pruned before the descent.
_K_TOPO_MULTIPLIER: int = int(os.getenv("IMI_HMEM_K_TOPO_MULT", "3"))

# Episode layer = real content (layer 3). Domain = most abstract (layer 0).
_EPISODE_LAYER = 3
_DOMAIN_LAYER = 0


@dataclass
class HMemHit:
    """One retrieved episode + its H-MEM confidence weight (ASSERT-5).

    `confidence` is the similarity-derived weight the paper injects into the LLM
    prompt context — NOT just an internal ranking key. The caller must surface it
    (see im_nav hierarchical branch) or the H-MEM quality gain does not materialise.
    """
    node: "MemoryNode"
    confidence: float
    via: str  # "tree" (reached by recursive descent) | "orphan" (flat-pool fallback)


@dataclass
class HMemResult:
    hits: list[HMemHit] = field(default_factory=list)
    tree_nodes_visited: int = 0
    orphan_pool_size: int = 0
    layers_descended: int = 0
    broken_ptrs: int = 0  # CHECK-3 counter — surfaces silent tree decay


def _cosine(q: np.ndarray, node: "MemoryNode") -> float:
    """sim(q, node) over the embedding ONLY (§3.2: parent_id/child_ptrs are
    addresses, not part of the similarity). Embeddings are L2-normalised by the
    embedder, so dot == cosine. Returns -1.0 for an embedding-less node so it
    sorts last without being silently dropped."""
    if node.embedding is None or q is None:
        return -1.0
    return float(np.dot(node.embedding, q))


def _topk_by_sim(q: np.ndarray, nodes: list["MemoryNode"], k: int) -> list["MemoryNode"]:
    """Return the k nodes with highest sim(q, node). Stable, no mutation."""
    if not nodes:
        return []
    scored = sorted(nodes, key=lambda n: _cosine(q, n), reverse=True)
    return scored[:k]


def _build_id_index(stores: list) -> dict[str, "MemoryNode"]:
    """Unified id→node map across every store (the H-MEM tree may span the
    episodic and semantic stores — a Domain index node lives in semantic, its
    Episode children in episodic). Built once per query, O(N)."""
    index: dict[str, "MemoryNode"] = {}
    for store in stores:
        for n in store.nodes:
            index[n.id] = n
    return index


def recursive_retrieve(
    query_embedding: np.ndarray,
    stores: list,
    k_final: int = 10,
) -> HMemResult:
    """H-MEM recursive top-down retrieval over the layer tree (§3.3).

    Parameters
    ----------
    query_embedding : L2-normalised query vector.
    stores          : list of VectorStore (episodic, semantic). The tree and the
                      orphan pool are both drawn from these.
    k_final         : episodes to return (default 10, == the paper).

    Returns
    -------
    HMemResult with hits (each carrying a confidence weight, ASSERT-5) plus
    diagnostics (orphan pool size, broken-ptr count) for shadow-mode logging.

    Example
    -------
    >>> res = recursive_retrieve(q_emb, [space.episodic, space.semantic], k_final=10)
    >>> res.hits[0].confidence, res.hits[0].via
    (0.83, 'tree')
    """
    index = _build_id_index(stores)
    result = HMemResult()
    k_topo = max(k_final * _K_TOPO_MULTIPLIER, k_final)

    # ----- ASSERT-6: orphan flat pool, ALWAYS swept in parallel to the tree -----
    # An orphan is any node NOT reachable as part of the tree: it has no children
    # and is not pointed at by any parent. Legacy nodes (layer==EPISODE default,
    # empty child_ptrs, never assigned a parent) live here. While the tree is
    # unpopulated, EVERY node is an orphan and this pool IS the whole search —
    # which is exactly why nothing breaks before consolidation runs.
    pointed_at: set[str] = set()
    for n in index.values():
        for c in n.child_ptrs:
            pointed_at.add(c)

    domain_nodes: list["MemoryNode"] = []
    orphan_nodes: list["MemoryNode"] = []
    for n in index.values():
        is_index_node = n.layer < _EPISODE_LAYER and n.child_ptrs
        if n.layer == _DOMAIN_LAYER and n.child_ptrs:
            domain_nodes.append(n)
        # Orphan = a leaf episode that no parent claims and that has no children.
        if not is_index_node and n.id not in pointed_at:
            orphan_nodes.append(n)
    result.orphan_pool_size = len(orphan_nodes)

    # ----- Tree descent (§3.3): L0 → ... → L3 -----
    visited: set[str] = set()  # CHECK-2 cycle guard — never trust tree acyclicity
    survivors: list["MemoryNode"] = _topk_by_sim(query_embedding, domain_nodes, k_topo)
    for s in survivors:
        visited.add(s.id)

    episodes: list["MemoryNode"] = [s for s in survivors if s.layer == _EPISODE_LAYER]

    # Descend until survivors are all episodes (or the tree runs out).
    while survivors and any(s.layer < _EPISODE_LAYER for s in survivors):
        result.layers_descended += 1
        next_children: list["MemoryNode"] = []
        for parent in survivors:
            if parent.layer == _EPISODE_LAYER:
                continue
            for cid in parent.child_ptrs:
                child = index.get(cid)
                if child is None:
                    # CHECK-3: broken pointer → drop branch, count it, never crash.
                    result.broken_ptrs += 1
                    continue
                if child.id in visited:
                    continue  # CHECK-2: cycle backstop
                visited.add(child.id)
                next_children.append(child)
        # Beam narrows toward k_final as we approach the Episode layer.
        beam = max(k_final, k_topo // (result.layers_descended + 1))
        survivors = _topk_by_sim(query_embedding, next_children, beam)
        episodes.extend(s for s in survivors if s.layer == _EPISODE_LAYER)

    result.tree_nodes_visited = len(visited)

    # ----- Merge tree episodes + orphan pool, dedup, rank by confidence -----
    seen: set[str] = set()
    merged: list[HMemHit] = []
    for ep in episodes:
        if ep.id in seen:
            continue
        seen.add(ep.id)
        merged.append(HMemHit(node=ep, confidence=max(0.0, _cosine(query_embedding, ep)), via="tree"))

    # Orphan pool: flat TopK, merged so no legacy node is ever lost (ASSERT-6).
    for orphan in _topk_by_sim(query_embedding, orphan_nodes, k_final):
        if orphan.id in seen:
            continue
        seen.add(orphan.id)
        merged.append(
            HMemHit(node=orphan, confidence=max(0.0, _cosine(query_embedding, orphan)), via="orphan")
        )

    merged.sort(key=lambda h: h.confidence, reverse=True)
    result.hits = merged[:k_final]
    return result
