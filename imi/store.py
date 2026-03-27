"""VectorStore — numpy-based vector search with relevance weighting."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from imi.events import ENCODE, MemoryEvent
from imi.node import MemoryNode

if TYPE_CHECKING:
    from imi.storage import StorageBackend


@dataclass
class VectorStore:
    """Simple numpy-based vector store with cosine similarity.

    No external dependencies. Good enough for thousands of memories.
    For larger scales, swap with Qdrant/FAISS adapter.

    Optional `backend` delegates persistence to a StorageBackend.
    """

    nodes: list[MemoryNode] = field(default_factory=list)
    _matrix: np.ndarray | None = field(default=None, repr=False)
    _dirty: bool = field(default=True, repr=False)

    # Storage backend (optional — if None, uses legacy JSON persistence)
    backend: StorageBackend | None = field(default=None, repr=False)
    store_name: str = ""

    def add(self, node: MemoryNode) -> None:
        self.nodes.append(node)
        self._dirty = True
        if self.backend:
            self.backend.put_node(self.store_name, node)
            self.backend.log_event(MemoryEvent(
                event_type=ENCODE,
                node_id=node.id,
                store_name=self.store_name,
                metadata={
                    "tags": node.tags,
                    "surprise": node.surprise_magnitude,
                    "mass": node.mass,
                },
            ))

    def add_batch(self, nodes: list[MemoryNode]) -> None:
        self.nodes.extend(nodes)
        self._dirty = True
        if self.backend:
            self.backend.put_nodes(self.store_name, nodes)

    def remove(self, node_id: str) -> None:
        self.nodes = [n for n in self.nodes if n.id != node_id]
        self._dirty = True
        if self.backend:
            self.backend.remove_node(self.store_name, node_id)

    def get(self, node_id: str) -> MemoryNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def _build_matrix(self) -> None:
        if not self._dirty or not self.nodes:
            return
        embeddings = [n.embedding for n in self.nodes if n.embedding is not None]
        if embeddings:
            self._matrix = np.vstack(embeddings)
        else:
            self._matrix = None
        self._dirty = False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        relevance_weight: float = 0.3,
    ) -> list[tuple[MemoryNode, float]]:
        """Search by cosine similarity, weighted by relevance (recency × frequency).

        Returns list of (node, combined_score) sorted descending.
        """
        self._build_matrix()
        if self._matrix is None or len(self.nodes) == 0:
            return []

        # Nodes with embeddings
        valid = [(i, n) for i, n in enumerate(self.nodes) if n.embedding is not None]
        if not valid:
            return []

        indices, valid_nodes = zip(*valid)
        matrix = np.vstack([n.embedding for n in valid_nodes])

        # Cosine similarity (embeddings are normalized)
        similarities = matrix @ query_embedding

        # Combined score: (1 - w) * similarity + w * normalized_relevance
        relevances = np.array([n.relevance for n in valid_nodes])
        if relevances.max() > 0:
            norm_relevances = relevances / relevances.max()
        else:
            norm_relevances = relevances

        scores = (1 - relevance_weight) * similarities + relevance_weight * norm_relevances

        # Top K
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            node = valid_nodes[idx]
            results.append((node, float(scores[idx])))

        return results

    def __len__(self) -> int:
        return len(self.nodes)

    # --- Persistence ---

    def save(self, path: str | Path | None = None) -> None:
        if self.backend:
            self.backend.put_nodes(self.store_name, self.nodes)
        elif path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = [n.to_dict() for n in self.nodes]
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        nodes = [MemoryNode.from_dict(d) for d in data]
        store = cls(nodes=nodes)
        return store

    @classmethod
    def from_backend(
        cls, backend: StorageBackend, store_name: str
    ) -> VectorStore:
        """Load a VectorStore from a StorageBackend."""
        nodes = backend.get_all_nodes(store_name)
        return cls(nodes=nodes, backend=backend, store_name=store_name)
