"""Pattern Completion (A4) — reconstruct full memory from a partial fragment.

Inspired by CA3 Hopfield network in hippocampal architecture (CORTEX):
given a partial cue, reconstruct the most likely complete memory via
nearest-neighbour walk in the memory graph.

Algorithm
---------
1. Embed the fragment.
2. Retrieve top-K candidates by cosine similarity.
3. For each candidate, compute token overlap between fragment and node.seed.
4. If overlap >= threshold: reconsolidate (access-triggered update) and return.
5. Fallback: expand via graph neighbours of best candidate.
6. Return None if no match found above minimum confidence.

Threshold calibration
---------------------
Run the bundled calibration script before deploying:

    python scripts/calibrate_completion_threshold.py

The script measures precision@1 over 50 truncated seeds from the live IMI
corpus and reports the threshold with best F1. Export the result:

    export IMI_COMPLETION_THRESHOLD=0.35

Default (0.35) is conservative and covers most real-world partial cues.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imi.node import MemoryNode
    from imi.space import IMISpace

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD: float = float(os.getenv("IMI_COMPLETION_THRESHOLD", "0.35"))
_TOP_K: int = int(os.getenv("IMI_COMPLETION_TOP_K", "5"))
_GRAPH_EXPAND: int = int(os.getenv("IMI_COMPLETION_GRAPH_EXPAND", "3"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _token_overlap(fragment: str, text: str) -> float:
    """Jaccard overlap between lowercased token sets (length-normalised)."""
    frag_tokens = {w.lower() for w in fragment.split() if len(w) > 2}
    text_tokens = {w.lower() for w in text.split() if len(w) > 2}
    if not frag_tokens or not text_tokens:
        return 0.0
    union = frag_tokens | text_tokens
    return len(frag_tokens & text_tokens) / len(union)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def reconstruct_from_partial(
    space: "IMISpace",
    fragment: str,
    threshold: float | None = None,
) -> "MemoryNode | None":
    """Given a partial fragment, return the most likely full MemoryNode.

    Parameters
    ----------
    space:     IMISpace instance (loaded).
    fragment:  Partial cue — a few words or a truncated sentence.
    threshold: Token overlap minimum to accept a candidate. If None,
               reads IMI_COMPLETION_THRESHOLD (default 0.35).

    Returns
    -------
    MemoryNode if a confident match is found, else None.
    The returned node is touch()-ed (reconsolidation-eligible).
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    # Step 1: embed fragment
    frag_emb = space.embedder.embed(fragment)

    # Step 2: top-K candidates from both stores
    episodic_cands = space.episodic.search(frag_emb, top_k=_TOP_K, relevance_weight=0.0)
    semantic_cands = space.semantic.search(
        frag_emb, top_k=max(1, _TOP_K // 2), relevance_weight=0.0
    )
    candidates = episodic_cands + semantic_cands

    # Step 3+4: check token overlap
    best_node = None
    best_overlap = 0.0
    for node, _score in candidates:
        overlap = _token_overlap(fragment, node.seed or node.summary_medium or "")
        if overlap >= threshold and overlap > best_overlap:
            best_overlap = overlap
            best_node = node

    if best_node is not None:
        best_node.touch()
        return best_node

    # Step 5: graph expansion — explore neighbours of the top cosine candidate
    if candidates:
        top_node = candidates[0][0]
        neighbour_ids = (
            space.graph.neighbors(top_node.id) if hasattr(space.graph, "neighbors") else []
        )
        for nid in neighbour_ids[:_GRAPH_EXPAND]:
            # Try episodic first, then semantic
            neighbour = space.episodic.get(nid) or space.semantic.get(nid)
            if neighbour is None:
                continue
            overlap = _token_overlap(fragment, neighbour.seed or neighbour.summary_medium or "")
            if overlap >= threshold * 0.8:  # slightly relaxed for graph hop
                neighbour.touch()
                return neighbour

    return None
