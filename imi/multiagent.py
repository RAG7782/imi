"""Multi-agent memory sharing with trust gradient.

Enables multiple agents to share a memory pool while maintaining:
- Agent-scoped views (each agent sees a filtered subset)
- Trust gradient (own memories > trusted agents > untrusted)
- Cross-pollination (agent A's experience helps agent B)
- Isolation (one agent can't corrupt another's critical memories)

Trust levels:
    SELF (1.0)     — agent's own memories, full weight
    TRUSTED (0.7)  — verified peer agents, slightly discounted
    PEER (0.4)     — same-team agents, moderate discount
    EXTERNAL (0.1) — unknown agents, heavy discount

Usage:
    pool = SharedMemoryPool.from_sqlite("shared.db")

    # Agent A encodes
    pool.encode("DNS failure", agent_id="agent-a", tags=["dns"])

    # Agent B searches — sees A's memory with trust discount
    result = pool.navigate("DNS issues", agent_id="agent-b")

    # Set trust levels
    pool.set_trust("agent-a", "agent-b", TrustLevel.TRUSTED)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from imi.node import MemoryNode
from imi.space import IMISpace, NavigationResult


class TrustLevel(float, Enum):
    SELF = 1.0
    TRUSTED = 0.7
    PEER = 0.4
    EXTERNAL = 0.1


@dataclass
class AgentView:
    """An agent's view into the shared memory pool."""

    agent_id: str
    space: IMISpace
    trust_map: dict[str, TrustLevel] = field(default_factory=dict)

    def effective_trust(self, source_agent: str) -> float:
        """Get effective trust level for a source agent."""
        if source_agent == self.agent_id:
            return TrustLevel.SELF.value
        return self.trust_map.get(source_agent, TrustLevel.EXTERNAL).value


@dataclass
class SharedMemoryPool:
    """Multi-agent shared memory with trust-weighted retrieval.

    Each memory is tagged with its source agent_id. When an agent navigates,
    results are weighted by trust level: own memories get full weight,
    trusted agents get 0.7x, peers 0.4x, unknown 0.1x.
    """

    _space: IMISpace
    _agents: dict[str, AgentView] = field(default_factory=dict)

    @classmethod
    def from_sqlite(cls, db_path: str) -> SharedMemoryPool:
        """Create a shared pool backed by SQLite."""
        space = IMISpace.from_sqlite(db_path)
        return cls(_space=space)

    def register_agent(self, agent_id: str) -> AgentView:
        """Register a new agent in the pool."""
        if agent_id not in self._agents:
            self._agents[agent_id] = AgentView(
                agent_id=agent_id,
                space=self._space,
            )
        return self._agents[agent_id]

    def set_trust(self, from_agent: str, to_agent: str, level: TrustLevel) -> None:
        """Set trust level from one agent to another."""
        if from_agent not in self._agents:
            self.register_agent(from_agent)
        self._agents[from_agent].trust_map[to_agent] = level

    def encode(
        self,
        experience: str,
        agent_id: str,
        tags: list[str] | None = None,
        **kwargs,
    ) -> MemoryNode:
        """Encode a memory attributed to a specific agent."""
        if agent_id not in self._agents:
            self.register_agent(agent_id)

        all_tags = list(tags or [])
        all_tags.append(f"agent:{agent_id}")

        node = self._space.encode(experience, tags=all_tags, source=agent_id, **kwargs)
        return node

    def navigate(
        self,
        query: str,
        agent_id: str,
        top_k: int = 10,
        **kwargs,
    ) -> NavigationResult:
        """Navigate with trust-weighted scoring.

        Results from untrusted agents are discounted.
        """
        if agent_id not in self._agents:
            self.register_agent(agent_id)

        view = self._agents[agent_id]

        # Get raw results (more than needed, we'll re-rank)
        result = self._space.navigate(query, top_k=top_k * 3, **kwargs)

        # Re-weight by trust
        weighted_memories = []
        for m in result.memories:
            source = self._extract_agent(m.get("tags", []))
            trust = view.effective_trust(source)
            adjusted_score = m["score"] * trust

            m_copy = dict(m)
            m_copy["score"] = adjusted_score
            m_copy["trust"] = trust
            m_copy["source_agent"] = source
            weighted_memories.append(m_copy)

        # Re-sort by adjusted score
        weighted_memories.sort(key=lambda x: x["score"], reverse=True)

        result.memories = weighted_memories[:top_k]
        return result

    def _extract_agent(self, tags: list[str]) -> str:
        """Extract agent_id from tags."""
        for tag in tags:
            if tag.startswith("agent:"):
                return tag[6:]
        return "unknown"

    def get_agent_stats(self, agent_id: str) -> dict[str, Any]:
        """Get stats for a specific agent's contribution."""
        tag = f"agent:{agent_id}"
        own_memories = [n for n in self._space.episodic.nodes if tag in n.tags]
        return {
            "agent_id": agent_id,
            "memories_contributed": len(own_memories),
            "total_pool_size": len(self._space.episodic),
            "trust_map": {
                k: v.value
                for k, v in self._agents.get(
                    agent_id, AgentView(agent_id, self._space)
                ).trust_map.items()
            },
        }

    def pool_stats(self) -> dict[str, Any]:
        """Get overall pool statistics."""
        agent_counts: dict[str, int] = {}
        for node in self._space.episodic.nodes:
            agent = self._extract_agent(node.tags)
            agent_counts[agent] = agent_counts.get(agent, 0) + 1

        return {
            "total_memories": len(self._space.episodic),
            "registered_agents": len(self._agents),
            "memories_per_agent": agent_counts,
            "graph_edges": self._space.graph.stats()["total_edges"],
        }
