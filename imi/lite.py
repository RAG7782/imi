"""IMI Lite-B: Zoom + Affordances as a thin wrapper over vector DB.

Tests the hypothesis from analysis P3: zoom + affordances capture 85% of
IMI's value in 20% of the code. If this ~100-line module matches IMI Full
on retrieval quality, the full system is over-engineered.

Usage:
    from imi.lite import ZoomRAG
    zr = ZoomRAG()
    zr.ingest("Production outage caused by DNS TTL of 300s...")
    results = zr.search("DNS issues", zoom="medium")
    actions = zr.search_actions("fix DNS timeout")
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import chromadb

from imi.embedder import Embedder, create_embedder_from_env


@dataclass
class ZoomRAG:
    """Minimal zoom + affordance layer over ChromaDB.

    ~100 lines that replicate IMI's core retrieval value proposition:
    - Multi-resolution zoom (orbital/medium/detailed/seed)
    - Affordance-based action search
    - Standard vector retrieval under the hood
    """

    embedder: Embedder = field(default_factory=create_embedder_from_env)
    collection_prefix: str = ""
    _client: Any = field(default=None, repr=False)
    _collection: Any = field(default=None, repr=False)
    _actions: Any = field(default=None, repr=False)

    def __post_init__(self):
        prefix = self.collection_prefix or uuid.uuid4().hex[:8]
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=f"{prefix}_memories",
            metadata={"hnsw:space": "cosine"},
        )
        self._actions = self._client.get_or_create_collection(
            name=f"{prefix}_actions",
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(
        self,
        text: str,
        *,
        summary_orbital: str = "",
        summary_medium: str = "",
        summary_detailed: str = "",
        seed: str = "",
        affordances: list[dict[str, str]] | None = None,
        tags: list[str] | None = None,
        node_id: str | None = None,
    ) -> str:
        """Ingest a memory with pre-computed zoom levels and affordances.

        If summaries are not provided, uses truncation as a fallback.
        In production, these would be LLM-generated (like IMI Full does).
        """
        node_id = node_id or uuid.uuid4().hex[:12]
        emb = self.embedder.embed(text).tolist()

        # Fallback zoom levels from text truncation
        if not summary_orbital:
            summary_orbital = text[:40]
        if not summary_medium:
            summary_medium = text[:160]
        if not summary_detailed:
            summary_detailed = text[:400]
        if not seed:
            seed = text[:320]

        self._collection.add(
            ids=[node_id],
            embeddings=[emb],
            documents=[text],
            metadatas=[
                {
                    "summary_orbital": summary_orbital,
                    "summary_medium": summary_medium,
                    "summary_detailed": summary_detailed,
                    "seed": seed,
                    "tags": json.dumps(tags or []),
                }
            ],
        )

        # Index affordances separately for action search
        if affordances:
            for i, aff in enumerate(affordances):
                action = aff.get("action", "")
                if action:
                    aff_emb = self.embedder.embed(action).tolist()
                    self._actions.add(
                        ids=[f"{node_id}_aff_{i}"],
                        embeddings=[aff_emb],
                        documents=[action],
                        metadatas=[
                            {
                                "node_id": node_id,
                                "action": action,
                                "confidence": str(aff.get("confidence", 0.5)),
                                "conditions": aff.get("conditions", ""),
                                "domain": aff.get("domain", ""),
                            }
                        ],
                    )

        return node_id

    def search(
        self,
        query: str,
        *,
        zoom: str = "medium",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search with zoom level selection.

        Returns the appropriate summary level for each result.
        """
        query_emb = self.embedder.embed(query).tolist()
        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self._collection.count() or 1),
            include=["metadatas", "distances", "documents"],
        )

        zoom_field = {
            "orbital": "summary_orbital",
            "medium": "summary_medium",
            "detailed": "summary_detailed",
            "seed": "seed",
            "full": "seed",  # full = seed in Lite-B (no LLM reconstruction)
        }.get(zoom, "summary_medium")

        memories = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = 1.0 - distance  # cosine distance → similarity

            memories.append(
                {
                    "id": doc_id,
                    "content": meta.get(zoom_field, meta.get("summary_medium", "")),
                    "score": score,
                    "tags": json.loads(meta.get("tags", "[]")),
                    "full_text": results["documents"][0][i],
                }
            )

        return memories

    def search_actions(
        self,
        query: str,
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search by action potential (affordance matching)."""
        if self._actions.count() == 0:
            return []

        query_emb = self.embedder.embed(query).tolist()
        results = self._actions.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self._actions.count()),
            include=["metadatas", "distances"],
        )

        actions = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1.0 - distance
            confidence = float(meta.get("confidence", 0.5))

            actions.append(
                {
                    "node_id": meta["node_id"],
                    "action": meta["action"],
                    "confidence": confidence,
                    "conditions": meta.get("conditions", ""),
                    "domain": meta.get("domain", ""),
                    "score": similarity * confidence,
                }
            )

        actions.sort(key=lambda x: x["score"], reverse=True)
        return actions[:top_k]

    @property
    def count(self) -> int:
        return self._collection.count()
