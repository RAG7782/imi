"""Causal Edge Detection — auto-detect relationships between memories.

Three strategies:
  1. Embedding-based (zero LLM calls): Detects cross-domain high-similarity
     pairs as potential causal links. Limited: causal pairs often have LOW
     cosine similarity (avg 0.31) because causality is logical, not semantic.
     Best for: finding obviously related cross-domain incidents.

  2. LLM-confirmed (1 call per candidate): Takes embedding candidates and
     asks the LLM to confirm/classify the relationship. Higher precision.
     Best for: confirming causal chains between semantically distant memories.

  3. Explicit hints (zero cost): Agent provides "caused_by" tag at encode
     time. Highest precision, requires agent cooperation.

Empirical finding (P2 experiment):
  - Causal pairs avg similarity = 0.308 (range 0.127-0.813)
  - Embedding-only at threshold 0.40: recall=30%, precision=10%
  - Embedding-only at threshold 0.65: recall=10%, precision=100%
  - Conclusion: LLM confirmation needed for most causal chains
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from imi.graph import EdgeType, MemoryGraph
from imi.llm import LLMAdapter
from imi.node import MemoryNode
from imi.store import VectorStore


@dataclass
class CausalCandidate:
    """A potential causal relationship between two memories."""

    source_id: str
    target_id: str
    similarity: float
    source_domain: str
    target_domain: str
    relationship: str = ""
    confirmed: bool = False


def detect_causal_candidates(
    new_node: MemoryNode,
    store: VectorStore,
    threshold: float = 0.55,
    max_candidates: int = 3,
    cross_domain_only: bool = True,
) -> list[CausalCandidate]:
    """Find potential causal relationships for a new memory.

    Strategy: high cosine similarity + different domain tags → likely related.
    Cross-domain relationships are the most valuable for multi-hop.

    Args:
        new_node: The newly encoded memory
        store: Existing memories to search against
        threshold: Minimum cosine similarity for candidate
        max_candidates: Max candidates to return
        cross_domain_only: Only consider cross-domain pairs (recommended)
    """
    if new_node.embedding is None:
        return []

    results = store.search(new_node.embedding, top_k=20, relevance_weight=0.0)

    new_domain = new_node.tags[0] if new_node.tags else ""
    candidates = []

    for node, score in results:
        if node.id == new_node.id:
            continue
        if score < threshold:
            continue

        node_domain = node.tags[0] if node.tags else ""

        # Cross-domain filter: only link between different domains
        if cross_domain_only and node_domain == new_domain:
            continue

        candidates.append(
            CausalCandidate(
                source_id=new_node.id,
                target_id=node.id,
                similarity=score,
                source_domain=new_domain,
                target_domain=node_domain,
            )
        )

        if len(candidates) >= max_candidates:
            break

    return candidates


CONFIRM_CAUSAL_SYSTEM = """\
You analyze whether two incidents are causally related.
Given two incident descriptions, determine if there is a causal, temporal,
or co-occurrence relationship between them.

Respond with ONLY valid JSON:
{
  "related": true/false,
  "relationship": "brief description of the relationship",
  "type": "causal" | "co_occurrence" | "similar",
  "confidence": 0.0-1.0
}

If the incidents are unrelated, set "related": false.
Write in the same language as the input."""


def confirm_causal_with_llm(
    node_a: MemoryNode,
    node_b: MemoryNode,
    llm: LLMAdapter,
) -> CausalCandidate | None:
    """Use LLM to confirm/classify a candidate relationship.

    Returns CausalCandidate with relationship description, or None if unrelated.
    """
    prompt = (
        f"Incident A:\n{node_a.seed}\n\n"
        f"Incident B:\n{node_b.seed}\n\n"
        "Are these incidents causally related?"
    )

    raw = llm.generate(
        system=CONFIRM_CAUSAL_SYSTEM,
        prompt=prompt,
        max_tokens=150,
        temperature=0.2,
    )

    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not data.get("related", False):
        return None

    return CausalCandidate(
        source_id=node_a.id,
        target_id=node_b.id,
        similarity=0.0,
        source_domain=node_a.tags[0] if node_a.tags else "",
        target_domain=node_b.tags[0] if node_b.tags else "",
        relationship=data.get("relationship", ""),
        confirmed=True,
    )


def auto_link_causal(
    new_node: MemoryNode,
    store: VectorStore,
    graph: MemoryGraph,
    threshold: float = 0.55,
    max_edges: int = 2,
    llm: LLMAdapter | None = None,
) -> int:
    """Auto-detect and link causal edges for a new memory.

    If llm is provided, candidates are confirmed via LLM (higher precision).
    Otherwise, embedding-only detection is used (zero cost).

    Returns number of edges added.
    """
    candidates = detect_causal_candidates(
        new_node,
        store,
        threshold=threshold,
        max_candidates=max_edges * 2,  # over-fetch for filtering
    )

    added = 0
    for candidate in candidates:
        if added >= max_edges:
            break

        if llm:
            # LLM confirmation
            target_node = store.get(candidate.target_id)
            if target_node:
                confirmed = confirm_causal_with_llm(new_node, target_node, llm)
                if confirmed:
                    graph.add_edge(
                        confirmed.source_id,
                        confirmed.target_id,
                        EdgeType.CAUSAL,
                        weight=0.8,
                        label=confirmed.relationship,
                    )
                    added += 1
        else:
            # Embedding-only: add as causal if cross-domain, similar if same-domain
            edge_type = (
                EdgeType.CAUSAL
                if candidate.source_domain != candidate.target_domain
                else EdgeType.SIMILAR
            )
            graph.add_edge(
                candidate.source_id,
                candidate.target_id,
                edge_type,
                weight=min(candidate.similarity, 0.8),
                label=f"auto-detected (cos={candidate.similarity:.2f})",
            )
            added += 1

    return added


def link_explicit(
    source_id: str,
    target_id: str,
    graph: MemoryGraph,
    label: str = "",
    edge_type: EdgeType = EdgeType.CAUSAL,
) -> None:
    """Strategy 3: Explicit causal link provided by the agent.

    Usage at encode time:
        node = space.encode("DNS failure caused auth timeout", caused_by="net_02")
    """
    graph.add_edge(source_id, target_id, edge_type, weight=0.9, label=label)
