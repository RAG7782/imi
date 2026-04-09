"""L0-L3 Tiered Memory — VIEW layer over CLS architecture.

L0-L3 is a presentation layer that controls how much information
is surfaced at different access levels, optimizing token economy
without changing the underlying episodic/semantic storage.

Inspired by MemPalace's spatial hierarchy, adapted for IMI's
graph-augmented CLS with affect and affordances.
"""
from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .node import MemoryNode


# ── Constants ──────────────────────────────────────────────

L0_MAX_TOKENS = 50
L1_MAX_TOKENS = 120
L0_L1_COMBINED_MAX = 200

PROMOTE_THRESHOLDS = {
    "salience_min": 0.7,
    "access_count_min": 3,
    "confidence_min": 0.8,
}

DEMOTE_THRESHOLDS = {
    "sessions_inactive": 5,
    "relevance_floor": 0.2,
}


# ── L0: Identity ──────────────────────────────────────────

@dataclass
class L0Identity:
    """Agent identity — always loaded, ~50 tokens."""
    agent_name: str = "IMI Agent"
    domain: str = ""
    session_id: str = ""
    user_context: str = ""
    custom_fields: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "L0Identity":
        """Load identity from ~/.imi/identity.json or create default."""
        p = path or Path.home() / ".imi" / "identity.json"
        if p.exists():
            data = json.loads(p.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()

    def save(self, path: Path | None = None) -> None:
        p = path or Path.home() / ".imi" / "identity.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict
        p.write_text(json.dumps(asdict(self), indent=2))

    def render(self) -> str:
        """Render L0 as compact text (~50 tokens)."""
        parts = [f"Agent: {self.agent_name}"]
        if self.domain:
            parts.append(f"Domain: {self.domain}")
        if self.session_id:
            parts.append(f"Session: {self.session_id}")
        if self.user_context:
            parts.append(f"Context: {self.user_context}")
        for k, v in self.custom_fields.items():
            parts.append(f"{k}: {v}")
        return " | ".join(parts)

    def token_estimate(self) -> int:
        return len(self.render()) // 4


# ── L1: Hot Facts ─────────────────────────────────────────

@dataclass
class L1HotFacts:
    """Top-N facts + Top-3 affordances — always loaded, ~120 tokens."""
    facts: list[dict[str, Any]] = field(default_factory=list)
    affordances: list[dict[str, Any]] = field(default_factory=list)
    generated_at: float = 0.0
    domain_filter: str | None = None

    def render(self) -> str:
        """Render L1 as compact text (~120 tokens)."""
        parts = []
        if self.facts:
            parts.append("Key facts:")
            for f in self.facts[:7]:  # Cap at 7 facts
                sal = f.get('salience', 0)
                parts.append(f"- {f['summary']} [sal:{sal:.1f}]")
        if self.affordances:
            parts.append("Actions available:")
            for a in self.affordances[:3]:
                conf = a.get('confidence', 0)
                parts.append(f"- {a['action']} (conf:{conf:.1f})")
        return "\n".join(parts)

    def token_estimate(self) -> int:
        return len(self.render()) // 4


def generate_l1(
    nodes: list[MemoryNode],
    *,
    domain_filter: str | None = None,
    max_facts: int = 12,  # Optimized via Modal L1 sweep (7→12 achieves tier_ratio=1.000)
    max_affordances: int = 3,
    channel_weights: dict[str, float] | None = None,
) -> L1HotFacts:
    """Generate L1 hot facts from memory nodes.

    Selection criteria (4 signals):
    1. TDA cluster coherence (approximated by tag frequency)
    2. Affect salience >= 0.7
    3. Access count >= 3 across sessions
    4. SYMBIONT channel weight >= 0.5 (optional signal)

    Args:
        nodes: All memory nodes (episodic + semantic)
        domain_filter: If set, prioritize nodes matching this domain
        max_facts: Maximum facts to include (default 7)
        max_affordances: Maximum affordances (default 3)
        channel_weights: Optional SYMBIONT Mycelium channel weights
    """
    # Score each node for L1 candidacy
    candidates = []
    for node in nodes:
        if not node.seed and not node.summary_orbital:
            continue

        score = node.relevance  # Base score from existing relevance property

        # Boost: high salience
        if node.affect and node.affect.salience >= PROMOTE_THRESHOLDS["salience_min"]:
            score *= 1.3

        # Boost: frequently accessed
        if node.access_count >= PROMOTE_THRESHOLDS["access_count_min"]:
            score *= 1.2

        # Boost: SYMBIONT channel weight (Sinergia 1)
        if channel_weights:
            for tag in node.tags:
                if tag in channel_weights and channel_weights[tag] >= 0.5:
                    score *= 1.25
                    break

        # Boost: domain match
        if domain_filter:
            if domain_filter.lower() in " ".join(node.tags).lower():
                score *= 1.5
            elif domain_filter.lower() in (node.seed or "").lower():
                score *= 1.2

        # Penalty: low confidence (suppress from L1)
        if node.affect and node.affect.salience < PROMOTE_THRESHOLDS["confidence_min"]:
            if "_pattern" in node.tags:  # Consolidated patterns need confidence
                score *= 0.5

        candidates.append((node, score))

    # Sort by score, take top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [n for n, _ in candidates[:max_facts]]

    # Extract facts
    facts = []
    for node in top_nodes:
        summary = node.summary_orbital or node.seed[:80] if node.seed else ""
        if summary:
            facts.append({
                "id": node.id,
                "summary": summary.strip(),
                "salience": node.affect.salience if node.affect else 0.5,
                "access_count": node.access_count,
                "tags": node.tags[:3],
            })

    # Extract top affordances across all high-relevance nodes
    all_affordances = []
    for node, score in candidates[:20]:  # Search in top 20 nodes
        for aff in node.affordances:
            all_affordances.append({
                "action": aff.action,
                "confidence": aff.confidence,
                "conditions": aff.conditions,
                "source_node": node.id,
            })

    # Sort affordances by confidence, take top N
    all_affordances.sort(key=lambda x: x["confidence"], reverse=True)

    return L1HotFacts(
        facts=facts,
        affordances=all_affordances[:max_affordances],
        generated_at=time.time(),
        domain_filter=domain_filter,
    )


# ── Tiering Policy ────────────────────────────────────────

def compute_tier(node: MemoryNode, *, channel_weights: dict[str, float] | None = None) -> int:
    """Compute recommended tier for a node based on relevance signals.

    Tiers as VIEW levels:
    - 3 (L3): Full access — graph expansion, deep search
    - 2 (L2): Filtered access — navigate with domain filter
    - 1 (L1): Hot facts — always available in compact form
    - 0 (L0): Identity only — not used for regular nodes

    Returns:
        int: Recommended tier (1, 2, or 3). L0 is identity-only.
    """
    rel = node.relevance

    # Patterns (consolidated) default to L1 (they're generalizations)
    if "_pattern" in node.tags:
        return 1 if rel >= 0.3 else 2

    # High relevance + frequently accessed -> L1
    if (rel >= 0.7
        and node.access_count >= PROMOTE_THRESHOLDS["access_count_min"]
        and node.affect and node.affect.salience >= PROMOTE_THRESHOLDS["salience_min"]):
        return 1

    # SYMBIONT channel boost (Sinergia 1)
    if channel_weights:
        for tag in node.tags:
            if tag in channel_weights and channel_weights[tag] >= 0.7:
                if rel >= 0.5:
                    return 1

    # Medium relevance -> L2
    if rel >= 0.3 or node.access_count >= 2:
        return 2

    # Low relevance -> L3
    return 3


def apply_tiering(
    nodes: list[MemoryNode],
    *,
    channel_weights: dict[str, float] | None = None,
) -> dict[str, int]:
    """Apply tiering policy to all nodes. Returns {node_id: new_tier}.

    Does NOT modify nodes directly — caller decides whether to apply.
    """
    changes = {}
    for node in nodes:
        new_tier = compute_tier(node, channel_weights=channel_weights)
        if new_tier != node.tier:
            changes[node.id] = new_tier
    return changes


def get_tier_stats(nodes: list[MemoryNode]) -> dict[str, Any]:
    """Get distribution of nodes across tiers."""
    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for node in nodes:
        dist[node.tier] = dist.get(node.tier, 0) + 1

    return {
        "l0": dist[0],
        "l1": dist[1],
        "l2": dist[2],
        "l3": dist[3],
        "total": sum(dist.values()),
    }
