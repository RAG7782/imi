"""MemoryGraph — lightweight relation edges between memories.

Adds multi-hop capability to IMI without requiring a full knowledge graph.
Edges represent causal, temporal, or semantic relationships between nodes.

Three edge types:
  - CAUSAL: "A caused B" or "A was caused by B"
  - CO_OCCURRENCE: "A and B happened in the same session/context"
  - SIMILAR: "A and B are semantically similar" (auto-detected)

Graph-augmented retrieval:
  1. Standard cosine search → top-K seeds
  2. Expand seeds via edges (1-hop or 2-hop)
  3. Re-rank expanded set by combining cosine score + edge weight

Based on: spreading activation in semantic networks (Collins & Loftus, 1975).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from imi.node import MemoryNode
from imi.store import VectorStore


class EdgeType(str, Enum):
    CAUSAL = "causal"
    CO_OCCURRENCE = "co_occurrence"
    SIMILAR = "similar"


@dataclass
class Edge:
    """A directed relationship between two memory nodes."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Edge:
        return cls(
            source_id=d["source_id"],
            target_id=d["target_id"],
            edge_type=EdgeType(d["edge_type"]),
            weight=d.get("weight", 1.0),
            label=d.get("label", ""),
        )


@dataclass
class MemoryGraph:
    """Lightweight graph over memory nodes.

    Stores edges in adjacency lists. No external dependencies.
    Can be serialized alongside the VectorStore.
    """

    # Adjacency: node_id → [(target_id, edge)]
    _outgoing: dict[str, list[Edge]] = field(default_factory=lambda: defaultdict(list))
    _incoming: dict[str, list[Edge]] = field(default_factory=lambda: defaultdict(list))

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        label: str = "",
        bidirectional: bool = True,
    ) -> Edge:
        """Add a relationship between two memories."""
        edge = Edge(source_id, target_id, edge_type, weight, label)
        self._outgoing[source_id].append(edge)
        self._incoming[target_id].append(edge)

        if bidirectional:
            reverse = Edge(target_id, source_id, edge_type, weight, label)
            self._outgoing[target_id].append(reverse)
            self._incoming[source_id].append(reverse)

        return edge

    def remove_edges(self, node_id: str) -> int:
        """Remove all edges involving a node. Returns count removed."""
        count = 0
        # Remove outgoing
        if node_id in self._outgoing:
            for edge in self._outgoing[node_id]:
                self._incoming[edge.target_id] = [
                    e for e in self._incoming[edge.target_id] if e.source_id != node_id
                ]
                count += 1
            del self._outgoing[node_id]
        # Remove incoming
        if node_id in self._incoming:
            for edge in self._incoming[node_id]:
                self._outgoing[edge.source_id] = [
                    e for e in self._outgoing[edge.source_id] if e.target_id != node_id
                ]
                count += 1
            del self._incoming[node_id]
        return count

    def neighbors(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
        direction: str = "both",
    ) -> list[tuple[str, Edge]]:
        """Get neighboring nodes with their edges."""
        result = []
        if direction in ("out", "both"):
            for edge in self._outgoing.get(node_id, []):
                if edge_type is None or edge.edge_type == edge_type:
                    result.append((edge.target_id, edge))
        if direction in ("in", "both"):
            for edge in self._incoming.get(node_id, []):
                if edge_type is None or edge.edge_type == edge_type:
                    result.append((edge.source_id, edge))
        return result

    def expand(
        self,
        seed_ids: list[str],
        hops: int = 1,
        edge_type: EdgeType | None = None,
    ) -> dict[str, float]:
        """Expand from seed nodes via graph edges (spreading activation).

        Returns {node_id: activation_score} where activation decays with hops.
        """
        activation: dict[str, float] = {}
        frontier = {nid: 1.0 for nid in seed_ids}

        for hop in range(hops):
            next_frontier: dict[str, float] = {}
            decay = 1.0 / (hop + 2)  # 0.5 at hop 1, 0.33 at hop 2

            for nid, score in frontier.items():
                if nid not in activation:
                    activation[nid] = score
                else:
                    activation[nid] = max(activation[nid], score)

                for neighbor_id, edge in self.neighbors(nid, edge_type):
                    new_score = score * edge.weight * decay
                    if neighbor_id not in activation:
                        if neighbor_id not in next_frontier or next_frontier[neighbor_id] < new_score:
                            next_frontier[neighbor_id] = new_score

            frontier = next_frontier

        # Add remaining frontier
        for nid, score in frontier.items():
            if nid not in activation:
                activation[nid] = score

        return activation

    def auto_link_similar(
        self,
        store: VectorStore,
        threshold: float = 0.75,
        max_edges_per_node: int = 3,
    ) -> int:
        """Auto-detect and link semantically similar memories.

        Uses the existing embeddings — no LLM calls needed.
        """
        nodes = [n for n in store.nodes if n.embedding is not None]
        if len(nodes) < 2:
            return 0

        embeddings = np.vstack([n.embedding for n in nodes])
        # Cosine similarity matrix
        sim_matrix = embeddings @ embeddings.T

        edge_count = 0
        for i, node_a in enumerate(nodes):
            # Get top similar nodes (excluding self)
            sims = sim_matrix[i].copy()
            sims[i] = -1  # exclude self
            top_indices = np.argsort(sims)[::-1][:max_edges_per_node]

            for j in top_indices:
                if sims[j] >= threshold:
                    node_b = nodes[j]
                    # Avoid duplicate edges
                    existing = [e.target_id for e in self._outgoing.get(node_a.id, [])]
                    if node_b.id not in existing:
                        self.add_edge(
                            node_a.id,
                            node_b.id,
                            EdgeType.SIMILAR,
                            weight=float(sims[j]),
                            bidirectional=False,  # already handled by symmetric sim
                        )
                        edge_count += 1

        return edge_count

    def auto_link_co_occurring(
        self,
        store: VectorStore,
        tag_key: str | None = None,
    ) -> int:
        """Link memories that share tags (co-occurrence)."""
        tag_groups: dict[str, list[str]] = defaultdict(list)
        for node in store.nodes:
            for tag in node.tags:
                tag_groups[tag].append(node.id)

        edge_count = 0
        for tag, node_ids in tag_groups.items():
            if len(node_ids) < 2 or len(node_ids) > 50:
                continue  # skip singleton or overly broad tags
            for i in range(len(node_ids)):
                for j in range(i + 1, min(len(node_ids), i + 5)):
                    existing = [e.target_id for e in self._outgoing.get(node_ids[i], [])]
                    if node_ids[j] not in existing:
                        self.add_edge(
                            node_ids[i],
                            node_ids[j],
                            EdgeType.CO_OCCURRENCE,
                            weight=0.5,
                            label=tag,
                        )
                        edge_count += 1
        return edge_count

    # --- Graph-augmented retrieval ---

    def search_with_expansion(
        self,
        store: VectorStore,
        query_embedding: np.ndarray,
        top_k: int = 10,
        seed_k: int = 5,
        hops: int = 1,
        relevance_weight: float = 0.1,
        graph_weight: float = 0.2,
    ) -> list[tuple[MemoryNode, float]]:
        """Graph-augmented retrieval: cosine seeds + graph expansion + re-rank.

        Score = (1 - rw - gw) * cosine + rw * relevance + gw * graph_activation
        """
        # Step 1: Get cosine seeds
        all_results = store.search(query_embedding, top_k=top_k * 2, relevance_weight=0.0)
        seed_ids = [n.id for n, s in all_results[:seed_k]]

        # Step 2: Expand via graph
        activation = self.expand(seed_ids, hops=hops)

        # Step 3: Re-rank with graph signal
        cosine_weight = 1.0 - relevance_weight - graph_weight
        scored = []
        node_map = {n.id: n for n in store.nodes}

        for node, cosine_score in all_results:
            graph_score = activation.get(node.id, 0.0)
            relevance = node.relevance
            # Normalize relevance
            max_rel = max(n.relevance for n, _ in all_results) or 1.0
            norm_rel = relevance / max_rel

            combined = (
                cosine_weight * cosine_score
                + relevance_weight * norm_rel
                + graph_weight * graph_score
            )
            scored.append((node, combined))

        # Also add graph-discovered nodes not in original results
        result_ids = {n.id for n, _ in all_results}
        for node_id, graph_score in activation.items():
            if node_id not in result_ids and graph_score > 0.1:
                node = node_map.get(node_id)
                if node and node.embedding is not None:
                    cosine = float(np.dot(node.embedding, query_embedding))
                    max_rel = max(n.relevance for n, _ in all_results) or 1.0
                    norm_rel = node.relevance / max_rel
                    combined = (
                        cosine_weight * cosine
                        + relevance_weight * norm_rel
                        + graph_weight * graph_score
                    )
                    scored.append((node, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # --- Stats ---

    def stats(self) -> dict[str, Any]:
        all_edges = []
        for edges in self._outgoing.values():
            all_edges.extend(edges)

        type_counts = defaultdict(int)
        for e in all_edges:
            type_counts[e.edge_type.value] += 1

        nodes_with_edges = set(self._outgoing.keys()) | set(self._incoming.keys())

        return {
            "total_edges": len(all_edges),
            "nodes_with_edges": len(nodes_with_edges),
            "edge_types": dict(type_counts),
            "avg_degree": len(all_edges) / max(len(nodes_with_edges), 1),
        }

    # --- Serialization ---

    def to_dict(self) -> list[dict]:
        """Serialize all edges."""
        edges = []
        seen = set()
        for node_edges in self._outgoing.values():
            for e in node_edges:
                key = (e.source_id, e.target_id, e.edge_type.value)
                if key not in seen:
                    edges.append(e.to_dict())
                    seen.add(key)
        return edges

    @classmethod
    def from_dict(cls, edges: list[dict]) -> MemoryGraph:
        """H4 fix: reconstruct both directions for bidirectional edges.

        to_dict() deduplicates by (source, target, type), so from_dict()
        must reconstruct the reverse direction to restore bidirectionality.
        """
        graph = cls()
        for d in edges:
            edge = Edge.from_dict(d)
            graph._outgoing[edge.source_id].append(edge)
            graph._incoming[edge.target_id].append(edge)
            # H4: reconstruct reverse direction (to_dict deduped it)
            reverse = Edge(
                source_id=edge.target_id,
                target_id=edge.source_id,
                edge_type=edge.edge_type,
                weight=edge.weight,
                label=edge.label,
            )
            graph._outgoing[edge.target_id].append(reverse)
            graph._incoming[edge.source_id].append(reverse)
        return graph
