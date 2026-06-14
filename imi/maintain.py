"""Maintenance — the living dynamics of IMI.

Consolidation (fast→slow), fade, and dreaming.
Implements CLS (Complementary Learning Systems):
  - Episodic store: fast, high-fidelity, decays
  - Semantic store: slow, generalized, persists
  - Consolidation: episodic patterns → semantic knowledge
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imi.core import get_llm
from imi.embedder import Embedder
from imi.events import (
    CONSOLIDATE,
    CONSOLIDATE_STRENGTHEN,
    FADE_CYCLE,
    PRUNE_CANDIDATE,
    MemoryEvent,
)
from imi.llm import LLMAdapter
from imi.node import MemoryNode
from imi.store import VectorStore


# --- H-MEM tree promotion (spec §3.4) ---------------------------------------
# A consolidated pattern node IS the H-MEM index node: it abstracts a cluster of
# episodes, exactly the Trace layer (L2) of the paper. Promotion wires the tree:
#   pattern_node.layer = TRACE; pattern_node.child_ptrs = [member ids]
#   each member.parent_id = pattern_node.id
# Acyclic BY CONSTRUCTION: the only edge direction is index(semantic) → member
# (episodic). Members never get child_ptrs here, so no cycle can form (CHECK-2 is
# the reader's backstop; this is the writer's guarantee).
# Batch-only: this runs inside consolidate() (the dream cycle), never in the
# im_enc hot path (R3). Default OFF until shadow data validates the tree (§4.5).
import os as _os_promote

_HMEM_TRACE_LAYER = 2  # Trace = abstraction over a cluster of episodes
_EPISODE_LAYER = 3


def _hmem_promote_enabled() -> bool:
    """Opt-in flag, read per-call so a long-running MCP picks it up on next dream.

    Default OFF: promotion writes tree pointers into the real store, and the
    pre-mortem forbids populating the tree before shadow-mode has data. Flip to
    "1" only when ready to start growing the tree under shadow observation.
    """
    return _os_promote.getenv("IMI_HMEM_PROMOTE", "0") == "1"


def _wire_tree(index_node: MemoryNode, members: list[MemoryNode]) -> None:
    """Make `index_node` the H-MEM parent of `members` (spec §3.4, CHECK-1).

    Sets index_node.layer/child_ptrs and each member.parent_id so the bidirectional
    invariant holds (CHECK-1: a child's parent_id == the id of the node whose
    child_ptrs contains it). Re-clustering a member into a new index node simply
    overwrites its parent_id (last consolidation wins) — the stale parent's
    child_ptr then dangles, which the reader tolerates as a broken ptr (CHECK-3).
    """
    index_node.layer = _HMEM_TRACE_LAYER
    index_node.child_ptrs = [m.id for m in members]
    for m in members:
        m.parent_id = index_node.id
        # Members stay at Episode layer — they are the concrete content (L3).
        if m.layer < _EPISODE_LAYER:
            m.layer = _EPISODE_LAYER


@dataclass
class PatternNode:
    """A consolidated pattern — emerged from recurring episodic memories.

    This is the 'neocortex' equivalent: slow, generalized, persistent.
    """

    id: str = ""
    summary: str = ""  # what the pattern IS
    source_count: int = 0  # how many episodes generated it
    source_ids: list[str] = field(default_factory=list)
    strength: float = 0.0  # 0-1, how confident the pattern is
    embedding: np.ndarray | None = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "summary": self.summary,
            "source_count": self.source_count,
            "source_ids": self.source_ids,
            "strength": self.strength,
            "created_at": self.created_at,
            "tags": self.tags,
        }
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PatternNode:
        emb = d.pop("embedding", None)
        node = cls(**d)
        if emb is not None:
            node.embedding = np.array(emb, dtype=np.float32)
        return node


@dataclass
class MaintenanceReport:
    """What happened during a maintenance cycle."""

    faded: int = 0  # nodes that lost relevance
    consolidated: int = 0  # patterns created/strengthened
    pruned: int = 0  # low-relevance nodes removed from fast store
    patterns_total: int = 0
    nodes_processed: int = 0  # total episodic nodes examined
    duration_ms: float = 0

    @property
    def clusters_formed(self) -> int:
        return self.consolidated

    @property
    def patterns_extracted(self) -> int:
        return self.consolidated

    def __str__(self) -> str:
        return (
            f"Maintenance: {self.faded} faded, {self.consolidated} consolidated, "
            f"{self.pruned} pruned, {self.patterns_total} patterns "
            f"({self.duration_ms:.0f}ms)"
        )


def count_fadeable(
    store: VectorStore,
    min_relevance: float = 0.1,
) -> int:
    """L4 fix: renamed from fade() — this only COUNTS, does not modify.

    Returns count of nodes that fell below min_relevance (>30 days old).
    Relevance decays naturally via the recency component.
    """
    faded = 0
    for node in store.nodes:
        days_since = (time.time() - node.last_accessed) / 86400
        if days_since > 30 and node.relevance < min_relevance:
            faded += 1
    return faded


# Keep backward-compatible alias
fade = count_fadeable


def find_clusters(
    store: VectorStore,
    similarity_threshold: float = 0.45,
) -> list[list[MemoryNode]]:
    """Find groups of highly similar memories (candidates for consolidation).

    Clusters are returned sorted by consolidation priority — surprise-bearing
    memories appear first so the maintenance budget processes them before
    routine episodic content (CLS + predictive coding alignment).
    """
    nodes = [n for n in store.nodes if n.embedding is not None]
    if len(nodes) < 2:
        return []

    matrix = np.vstack([n.embedding for n in nodes])
    # Cosine similarity matrix
    sim_matrix = matrix @ matrix.T

    visited = set()
    clusters = []

    for i in range(len(nodes)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i + 1, len(nodes)):
            if j in visited:
                continue
            if sim_matrix[i, j] >= similarity_threshold:
                cluster.append(j)
                visited.add(j)
        if len(cluster) >= 2:
            clusters.append([nodes[idx] for idx in cluster])

    # Sort clusters by max consolidation_priority of any node in the cluster.
    # High-surprise clusters consolidate first — this is the correct role
    # of surprise in CLS/predictive coding (not retrieval boost).
    clusters.sort(
        key=lambda c: max(n.consolidation_priority for n in c),
        reverse=True,
    )

    return clusters


def consolidate(
    clusters: list[list[MemoryNode]],
    semantic_store: VectorStore,
    embedder: Embedder,
    llm: LLMAdapter | None = None,
    dirty_sink: Any = None,
) -> list[PatternNode]:
    """Convert episodic clusters into semantic patterns.

    This is the sleep/dream consolidation: episodic → semantic.

    A3 — Cluster Consolidation with LLM cross-synthesis:
    - Clusters with >= IMI_CONSOLIDATION_MIN_CLUSTER members use LLM to
      synthesise a cross-episode pattern (not just summarise the top node).
    - Budget cap: IMI_CONSOLIDATION_LLM_BUDGET LLM calls per cycle (default 5)
      to prevent runaway phi4-mini usage.
    - Fallback: clusters below the threshold use the strongest node's summary
      (original behaviour).

    dirty_sink: optional callable(node) invoked for every EPISODIC member whose
    parent_id is mutated by H-MEM promotion. The caller (dream cycle) wires this
    to space.mark_node_dirty so the new parent_id survives a backend save —
    without it, _save_dirty_nodes() would silently drop the tree pointers (the
    exact silent-write-loss the canary guards against). No-op when promotion is off.
    """
    import os as _os

    _min_cluster = int(_os.getenv("IMI_CONSOLIDATION_MIN_CLUSTER", "3"))
    _llm_budget = int(_os.getenv("IMI_CONSOLIDATION_LLM_BUDGET", "5"))

    llm = llm or get_llm()
    new_patterns = []
    _llm_calls = 0

    for cluster in clusters:
        # Collect seeds from the cluster
        seeds = [n.seed for n in cluster]
        seeds_text = "\n---\n".join(seeds)

        # A3 — Cross-cluster synthesis when cluster is large enough and budget allows.
        if len(cluster) >= _min_cluster and _llm_calls < _llm_budget:
            all_summaries = "\n".join(n.summary_medium for n in cluster if n.summary_medium)
            try:
                pattern_summary = llm.generate(
                    system=(
                        "You are a pattern synthesis engine. Given related memory summaries, "
                        "extract the EMERGENT PATTERN — the recurring theme across ALL episodes, "
                        "not a summary of any single one. "
                        "Write in the same language as the input. Max 80 tokens."
                    ),
                    prompt=f"Memory summaries:\n{all_summaries}\n\nEmergent pattern:",
                    max_tokens=80,
                )
                _llm_calls += 1
            except Exception:
                # Fallback to strongest node summary on LLM failure
                strongest = max(cluster, key=lambda n: n.affect.salience if n.affect else 0)
                pattern_summary = strongest.summary_medium or seeds[0]
        else:
            # Original behaviour: use strongest node's medium summary
            strongest = max(cluster, key=lambda n: n.affect.salience if n.affect else 0)
            pattern_summary = strongest.summary_medium or llm.generate(
                system=(
                    "You are a pattern recognition engine. Given multiple related memories, "
                    "extract the GENERAL PATTERN — not a summary of the individual events, "
                    "but the recurring theme or rule they demonstrate. "
                    "Write in the same language as the input. Be concise (max 60 tokens)."
                ),
                prompt=(
                    f"Related memories:\n{seeds_text}\n\nWhat general pattern do these demonstrate?"
                ),
                max_tokens=120,
            )

        # Embed the pattern
        pattern_emb = embedder.embed(pattern_summary)

        # Collect tags from all sources
        all_tags = list({t for n in cluster for t in n.tags})

        pattern = PatternNode(
            id=f"pattern_{cluster[0].id[:6]}",
            summary=pattern_summary,
            source_count=len(cluster),
            source_ids=[n.id for n in cluster],
            strength=min(1.0, len(cluster) / 5.0),
            embedding=pattern_emb,
            tags=all_tags,
        )

        # Check if similar pattern already exists in semantic store
        existing = semantic_store.search(pattern_emb, top_k=1, relevance_weight=0.0)
        if existing and existing[0][1] > 0.9:
            # Strengthen existing pattern
            existing_node = existing[0][0]
            existing_node.access_count += len(cluster)
            existing_node.tags = list(set(existing_node.tags + all_tags))
            # H-MEM §3.4: extend the existing index node's child_ptrs with the new
            # cluster members (union, no dupes) and re-parent them. The existing
            # pattern absorbs this cluster, so it must own its children too.
            if _hmem_promote_enabled():
                merged_members = {c.id: c for c in cluster}
                for cid in existing_node.child_ptrs:
                    merged_members.setdefault(cid, None)  # keep prior ptrs as-is
                existing_node.layer = _HMEM_TRACE_LAYER
                existing_node.child_ptrs = list(merged_members.keys())
                if dirty_sink:
                    dirty_sink(existing_node)  # child_ptrs mutated on the index node
                for c in cluster:
                    c.parent_id = existing_node.id
                    if c.layer < _EPISODE_LAYER:
                        c.layer = _EPISODE_LAYER
                    if dirty_sink:
                        dirty_sink(c)  # parent_id mutated → mark dirty for save
            if semantic_store.backend:
                semantic_store.backend.log_event(
                    MemoryEvent(
                        event_type=CONSOLIDATE_STRENGTHEN,
                        node_id=existing_node.id,
                        store_name="semantic",
                        metadata={
                            "source_ids": [n.id for n in cluster],
                            "new_access_count": existing_node.access_count,
                        },
                    )
                )
        else:
            # H7 fix: patterns inherit aggregated affect/mass from source episodes
            from imi.affect import AffectiveTag

            avg_salience = sum(n.affect.salience for n in cluster if n.affect) / max(
                len(cluster), 1
            )
            avg_valence = sum(n.affect.valence for n in cluster if n.affect) / max(len(cluster), 1)
            avg_arousal = sum(n.affect.arousal for n in cluster if n.affect) / max(len(cluster), 1)
            total_mass = sum(n.mass for n in cluster)
            total_affordances = []
            for n in cluster:
                total_affordances.extend(n.affordances[:2])  # top 2 per source

            pattern_node = MemoryNode(
                id=pattern.id,
                summary_orbital=f"[PATTERN] {pattern_summary[:50]}",
                summary_medium=f"[PATTERN] {pattern_summary}",
                summary_detailed=(
                    f"[PATTERN] {pattern_summary} (from {pattern.source_count} experiences)"
                ),
                seed=pattern_summary,
                embedding=pattern_emb,
                tags=all_tags + ["_pattern"],
                source=f"consolidated from {pattern.source_count} episodes",
                affect=AffectiveTag(
                    salience=avg_salience,
                    valence=avg_valence,
                    arousal=avg_arousal,
                ),
                mass=min(total_mass / len(cluster) * 1.2, 10.0),  # slight boost for patterns
                affordances=total_affordances[:4],  # cap at 4
            )
            semantic_store.add(pattern_node)
            # H-MEM §3.4: this fresh pattern node IS the Trace-layer index for its
            # cluster. Wire the tree so recursive_retrieve() has something to descend.
            if _hmem_promote_enabled():
                _wire_tree(pattern_node, cluster)
                if dirty_sink:
                    # CRITICAL: the pattern node was already persisted by add() at
                    # layer=3 (default). _wire_tree just changed it to layer=2 +
                    # child_ptrs IN MEMORY — without re-marking it dirty, save()'s
                    # dirty-only path drops the promotion (silent-write-loss, found
                    # by the Passo 7 reload check 2026-06-14: index nodes 20→0).
                    dirty_sink(pattern_node)
                    for m in cluster:  # parent_id mutated → must be marked dirty
                        dirty_sink(m)
            new_patterns.append(pattern)

    return new_patterns


def run_maintenance(
    episodic: VectorStore,
    semantic: VectorStore,
    embedder: Embedder,
    llm: LLMAdapter | None = None,
    similarity_threshold: float = 0.45,
    budget: int = 100,
    dirty_sink: Any = None,
) -> MaintenanceReport:
    """Execute one maintenance cycle (dreaming).

    Fixed budget — no 'run until convergence'.
    """
    t0 = time.time()
    backend = episodic.backend  # may be None

    # 1. Fade
    faded = fade(episodic)
    if backend and faded > 0:
        backend.log_event(
            MemoryEvent(
                event_type=FADE_CYCLE,
                node_id="*",
                store_name="episodic",
                metadata={"faded_count": faded},
            )
        )

    # 2. Find clusters of similar episodic memories
    clusters = find_clusters(episodic, similarity_threshold=similarity_threshold)

    # 3. Consolidate clusters into semantic patterns
    if clusters:
        patterns = consolidate(clusters, semantic, embedder, llm, dirty_sink=dirty_sink)
        consolidated = len(patterns)
        if backend:
            for p in patterns:
                backend.log_event(
                    MemoryEvent(
                        event_type=CONSOLIDATE,
                        node_id=p.id,
                        store_name="semantic",
                        metadata={
                            "source_ids": p.source_ids,
                            "source_count": p.source_count,
                            "strength": p.strength,
                        },
                    )
                )
    else:
        consolidated = 0

    # 4. Prune: H6 fix — actually remove very old low-relevance episodic nodes
    pruned = 0
    for node in list(episodic.nodes):
        if node.relevance < 0.05 and "_keep" not in node.tags:
            pruned += 1
            if backend:
                backend.log_event(
                    MemoryEvent(
                        event_type=PRUNE_CANDIDATE,
                        node_id=node.id,
                        store_name="episodic",
                        metadata={"relevance": node.relevance},
                    )
                )
            # H6 fix: actually remove the node from the episodic store
            episodic.remove(node.id)

    duration_ms = (time.time() - t0) * 1000

    return MaintenanceReport(
        faded=faded,
        consolidated=consolidated,
        pruned=pruned,
        patterns_total=len(semantic.nodes),
        nodes_processed=len(episodic.nodes),
        duration_ms=duration_ms,
    )
