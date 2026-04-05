"""IMISpace v3 — the complete Infinite Memory Image.

100/100 tier: Predictive coding + CLS + Temporal + Affect + Affordances +
Reconsolidation + TDA + Annealing convergence.

One system. All the theory.
"""

from __future__ import annotations

import json as _json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from imi.adaptive import AdaptiveRW
from imi.affect import AffectiveTag, assess_affect
from imi.affordance import Affordance, extract_affordances
from imi.anchors import Anchor, ConfidenceReport, extract_anchors, compute_confidence
from imi.causal import auto_link_causal
from imi.core import compress_seed, remember, summarize, get_llm
from imi.positional import positional_reorder
from imi.embedder import Embedder, SentenceTransformerEmbedder
from imi.events import NAVIGATE_ACCESS, MemoryEvent
from imi.graph import MemoryGraph
from imi.llm import LLMAdapter
from imi.maintain import MaintenanceReport, run_maintenance
from imi.node import MemoryNode
from imi.reconsolidate import ReconsolidationEvent, reconsolidate
from imi.spatial import SpatialIndex, TopologyReport
from imi.storage import SQLiteBackend, StorageBackend
from imi.store import VectorStore
from imi.surprise import SurpriseResult, encode_with_surprise, reconstruct_from_surprise
from imi.tda import TDAReport, AnnealingState, compute_persistent_homology, compute_space_energy
from imi.temporal import TemporalContext, TemporalIndex


class Zoom(str, Enum):
    ORBITAL = "orbital"
    MEDIUM = "medium"
    DETAILED = "detailed"
    FULL = "full"


@dataclass
class NavigationResult:
    """What the agent 'sees' when navigating the memory space."""

    query: str
    zoom: Zoom
    memories: list[dict[str, Any]]
    total_tokens_approx: int = 0
    topology: TopologyReport | None = None
    tda: TDAReport | None = None

    def __str__(self) -> str:
        lines = [f"[Navigate: zoom={self.zoom.value}, hits={len(self.memories)}, ~{self.total_tokens_approx} tokens]"]
        for m in self.memories:
            store = m.get("store", "?")
            surprise = f" S={m['surprise']:.0%}" if "surprise" in m else ""
            affect_str = f" A={m.get('affect_str', '')}" if m.get("affect_str") else ""
            lines.append(f"  [{m['score']:.2f}] [{store}]{surprise}{affect_str} {m['content'][:100]}")
        if self.tda:
            lines.append(f"\n{self.tda}")
        elif self.topology:
            lines.append(f"\n{self.topology}")
        return "\n".join(lines)


@dataclass
class IMISpace:
    """The Infinite Memory Image v3 — 100/100 tier.

    Complete cognitive memory system:
      - Predictive coding (surprise-based encoding)
      - CLS dual-store (episodic + semantic)
      - Temporal context (when dimension)
      - Affective tagging (importance/emotion)
      - Affordances (action potentials)
      - Reconsolidation (memory changes on access)
      - TDA (persistent homology for topology)
      - Annealing (convergence-tracked dreaming)
    """

    # CLS: two stores
    episodic: VectorStore = field(default_factory=VectorStore)
    semantic: VectorStore = field(default_factory=VectorStore)

    # Adapters
    embedder: Embedder = field(default_factory=SentenceTransformerEmbedder)
    llm: LLMAdapter | None = None

    # Spatial + TDA
    spatial: SpatialIndex = field(default_factory=SpatialIndex)
    annealing: AnnealingState = field(default_factory=AnnealingState)

    # Temporal index
    temporal_index: TemporalIndex = field(default_factory=TemporalIndex)

    # Anchors
    _anchors: dict[str, list[Anchor]] = field(default_factory=dict)

    # Graph layer (multi-hop)
    graph: MemoryGraph = field(default_factory=MemoryGraph)

    # Adaptive relevance weight
    adaptive_rw: AdaptiveRW = field(default_factory=AdaptiveRW)

    # Reconsolidation log
    reconsolidation_log: list[ReconsolidationEvent] = field(default_factory=list)

    # Persistence
    persist_dir: Path | None = None
    backend: StorageBackend | None = None

    def __post_init__(self) -> None:
        if self.backend:
            self.episodic.backend = self.backend
            self.episodic.store_name = "episodic"
            self.semantic.backend = self.backend
            self.semantic.store_name = "semantic"

    def _llm(self) -> LLMAdapter:
        return self.llm or get_llm()

    # ========================= ENCODE =========================

    def encode(
        self,
        experience: str,
        *,
        tags: list[str] | None = None,
        source: str = "",
        context_hint: str = "",
        use_predictive_coding: bool = False,
        timestamp: float | None = None,
    ) -> MemoryNode:
        """Transform an experience into a memory node.

        v3 pipeline:
          1. Predict what to expect (predictive coding)
          2. Compute surprise (delta from prediction)
          3. Compress to seed
          4. Pre-compute zoom summaries
          5. Assess affect (salience, valence, arousal)
          6. Extract affordances (action potentials)
          7. Extract anchors (anti-confabulation)
          8. Register temporal context
          9. Embed and position
        """
        llm = self._llm()

        # 1-2. Predictive coding
        surprise = None
        if use_predictive_coding and context_hint:
            surprise = encode_with_surprise(experience, context_hint, llm)

        # 3. Compress to seed
        seed = compress_seed(experience, llm=llm)

        # 4. Zoom summaries
        summary_orbital = summarize(experience, max_tokens=10, llm=llm)
        summary_medium = summarize(experience, max_tokens=40, llm=llm)
        summary_detailed = summarize(experience, max_tokens=100, llm=llm)

        # 5. Affect
        affect = assess_affect(experience, llm)

        # 6. Affordances
        affordances = extract_affordances(experience, llm)

        # 7. Anchors
        anchors = extract_anchors(experience, llm)

        # 8. Embed
        embedding = self.embedder.embed(experience)

        # Build node
        node = MemoryNode(
            seed=seed,
            summary_orbital=summary_orbital,
            summary_medium=summary_medium,
            summary_detailed=summary_detailed,
            embedding=embedding,
            tags=tags or [],
            source=source,
            original=experience,
            affect=affect,
            mass=affect.initial_mass,
            affordances=affordances,
        )

        # Attach surprise data
        if surprise:
            node.surprise_summary = surprise.surprise
            node.surprise_magnitude = surprise.magnitude
            node.surprise_elements = surprise.surprise_elements
            node.prediction = surprise.prediction

        # 8. Temporal context
        temporal = self.temporal_index.register(
            node.id,
            timestamp=timestamp or time.time(),
        )
        node.temporal = temporal

        # Store anchors
        if anchors:
            self._anchors[node.id] = anchors

        # Store in EPISODIC
        self.episodic.add(node)

        # 10. Auto-link graph edges (similarity-based, zero LLM calls)
        if len(self.episodic) > 5:
            auto_link_causal(
                node, self.episodic, self.graph,
                threshold=0.65, max_edges=2, llm=None,
            )

        if self.persist_dir or self.backend:
            self.save()

        return node

    # ========================= NAVIGATE =========================

    def navigate(
        self,
        query: str,
        *,
        zoom: Zoom | str = Zoom.MEDIUM,
        top_k: int = 20,
        context: str = "",
        relevance_weight: float | None = None,
        include_semantic: bool = True,
        include_tda: bool = False,
        reconsolidate_on_access: bool = False,
        use_graph: bool = True,
        graph_weight: float = 0.2,
        positional_optimize: bool = True,
    ) -> NavigationResult:
        """Navigate the memory space with full v3 features.

        If relevance_weight is None, uses AdaptiveRW to select optimal rw
        based on query intent (temporal → 0.15, exploratory → 0.0, etc.).

        If positional_optimize is True (default), reorders results using the
        primacy-recency pattern from "Lost in the Middle" (Liu et al. 2023)
        so the highest-relevance memories sit at the START and END of the
        list — where LLM attention is strongest — and lower-relevance items
        occupy the center.
        """
        if isinstance(zoom, str):
            zoom = Zoom(zoom)

        # Adaptive relevance weight
        if relevance_weight is None:
            relevance_weight = self.adaptive_rw.classify(query)

        query_emb = self.embedder.embed(query)

        # Graph-augmented search if graph has edges
        if use_graph and self.graph.stats()["total_edges"] > 0:
            episodic_results = self.graph.search_with_expansion(
                self.episodic, query_emb, top_k=top_k,
                relevance_weight=relevance_weight,
                graph_weight=graph_weight,
            )
        else:
            episodic_results = self.episodic.search(
                query_emb, top_k=top_k, relevance_weight=relevance_weight
            )
        semantic_results = []
        if include_semantic and len(self.semantic) > 0:
            semantic_results = self.semantic.search(
                query_emb, top_k=top_k // 2, relevance_weight=0.1
            )

        all_results = episodic_results + semantic_results
        all_results.sort(key=lambda x: x[1], reverse=True)
        all_results = all_results[:top_k]

        memories = []
        total_tokens = 0

        for node, score in all_results:
            node.touch()

            # Reconsolidation (v3): access may modify the memory
            if reconsolidate_on_access and zoom == Zoom.FULL and context:
                event = reconsolidate(node, context, self._llm())
                if event:
                    node.reconsolidation_count += 1
                    node.last_reconsolidated = time.time()
                    self.reconsolidation_log.append(event)

            # Zoom level content
            if zoom == Zoom.ORBITAL:
                content = node.summary_orbital
                tok_est = 10
            elif zoom == Zoom.MEDIUM:
                content = node.summary_medium
                tok_est = 40
            elif zoom == Zoom.DETAILED:
                content = node.summary_detailed
                tok_est = 100
            elif zoom == Zoom.FULL:
                if len(memories) < 3:
                    # v3: Use predictive reconstruction if surprise data exists
                    if node.surprise_elements:
                        sr = SurpriseResult(
                            prediction=node.prediction,
                            actual="",
                            surprise=node.surprise_summary,
                            magnitude=node.surprise_magnitude,
                            surprise_elements=node.surprise_elements,
                        )
                        content = reconstruct_from_surprise(
                            sr, context or query, self._llm()
                        )
                    else:
                        content = remember(node.seed, context or query, llm=self._llm())
                    tok_est = 200
                else:
                    content = node.summary_detailed
                    tok_est = 100
            else:
                content = node.summary_medium
                tok_est = 40

            mem_entry: dict[str, Any] = {
                "id": node.id,
                "content": content,
                "score": score,
                "relevance": node.relevance,
                "created_at": node.created_at,
                "tags": node.tags,
                "store": "semantic" if "_pattern" in node.tags else "episodic",
                "surprise": node.surprise_magnitude,
                "affect_str": str(node.affect) if node.affect else "",
                "mass": node.mass,
                "affordances": [str(a) for a in node.affordances],
            }

            # Confidence for detailed+ zoom
            if zoom in (Zoom.DETAILED, Zoom.FULL) and node.id in self._anchors:
                conf = compute_confidence(content, self._anchors[node.id], self._llm())
                mem_entry["confidence"] = conf.confidence

            memories.append(mem_entry)
            total_tokens += tok_est

        # Positional optimization: primacy-recency reorder
        if positional_optimize and len(memories) > 2:
            memories = positional_reorder(memories)

        # TDA
        tda = None
        if include_tda:
            tda = self.compute_tda()

        # Emit batched navigate event
        if self.backend and memories:
            self.backend.log_event(MemoryEvent(
                event_type=NAVIGATE_ACCESS,
                node_id="*",
                store_name="episodic",
                metadata={
                    "query": query[:200],
                    "zoom": zoom.value,
                    "node_ids": [m["id"] for m in memories],
                    "top_score": memories[0]["score"] if memories else 0,
                },
            ))

        if self.persist_dir or self.backend:
            self.save()

        return NavigationResult(
            query=query,
            zoom=zoom,
            memories=memories,
            total_tokens_approx=total_tokens,
            tda=tda,
        )

    # ========================= TEMPORAL NAVIGATION =========================

    def navigate_temporal(
        self,
        target_time: float,
        window_hours: float = 24.0,
        zoom: Zoom | str = Zoom.MEDIUM,
    ) -> NavigationResult:
        """Navigate by TIME instead of semantic similarity."""
        if isinstance(zoom, str):
            zoom = Zoom(zoom)

        results = self.temporal_index.search_by_time(target_time, window_hours)

        memories = []
        total_tokens = 0
        for node_id, dist_hours in results:
            node = self.episodic.get(node_id) or self.semantic.get(node_id)
            if not node:
                continue
            node.touch()

            if zoom == Zoom.ORBITAL:
                content = node.summary_orbital
                tok_est = 10
            elif zoom == Zoom.MEDIUM:
                content = node.summary_medium
                tok_est = 40
            else:
                content = node.summary_detailed
                tok_est = 100

            memories.append({
                "id": node.id,
                "content": content,
                "score": 1.0 / (1.0 + dist_hours),
                "relevance": node.relevance,
                "created_at": node.created_at,
                "tags": node.tags,
                "store": "episodic",
                "temporal_distance_hours": dist_hours,
                "surprise": node.surprise_magnitude,
                "affect_str": str(node.affect),
                "mass": node.mass,
                "affordances": [str(a) for a in node.affordances],
            })
            total_tokens += tok_est

        return NavigationResult(
            query=f"time:{target_time}",
            zoom=zoom,
            memories=memories,
            total_tokens_approx=total_tokens,
        )

    # ========================= AFFORDANCE SEARCH =========================

    def search_affordances(self, action_query: str, top_k: int = 5) -> list[dict]:
        """Search memories by what actions they ENABLE, not just content."""
        results = []
        all_nodes = self.episodic.nodes + self.semantic.nodes

        query_emb = self.embedder.embed(action_query)

        for node in all_nodes:
            for aff in node.affordances:
                aff_emb = self.embedder.embed(aff.action)
                sim = float(np.dot(query_emb, aff_emb))
                results.append({
                    "node_id": node.id,
                    "affordance": str(aff),
                    "action": aff.action,
                    "confidence": aff.confidence,
                    "conditions": aff.conditions,
                    "similarity": sim,
                    "memory_summary": node.summary_medium,
                })

        results.sort(key=lambda x: x["similarity"] * x["confidence"], reverse=True)
        return results[:top_k]

    # ========================= SPATIAL / TDA =========================

    def compute_topology(self) -> TopologyReport:
        all_nodes = self.episodic.nodes + self.semantic.nodes
        valid = [n for n in all_nodes if n.embedding is not None]
        if len(valid) < 2:
            return TopologyReport()
        embeddings = np.vstack([n.embedding for n in valid])
        node_ids = [n.id for n in valid]
        self.spatial.project(embeddings, node_ids)
        self.spatial.cluster()
        return self.spatial.topology()

    def compute_tda(self) -> TDAReport:
        """Full persistent homology analysis."""
        all_nodes = self.episodic.nodes + self.semantic.nodes
        valid = [n for n in all_nodes if n.embedding is not None]
        if len(valid) < 3:
            return TDAReport(betti_0=len(valid))
        embeddings = np.vstack([n.embedding for n in valid])
        return compute_persistent_homology(embeddings)

    # ========================= DREAM (MAINTENANCE) =========================

    def dream(
        self,
        similarity_threshold: float = 0.45,
        track_convergence: bool = True,
    ) -> MaintenanceReport:
        """Run one dreaming cycle with convergence tracking."""
        report = run_maintenance(
            episodic=self.episodic,
            semantic=self.semantic,
            embedder=self.embedder,
            llm=self._llm(),
            similarity_threshold=similarity_threshold,
        )

        # Track energy for annealing convergence
        if track_convergence:
            all_nodes = self.episodic.nodes + self.semantic.nodes
            valid = [n for n in all_nodes if n.embedding is not None]
            if len(valid) >= 2:
                embeddings = np.vstack([n.embedding for n in valid])
                masses = np.array([n.mass for n in valid])
                energy = compute_space_energy(embeddings, masses)
                self.annealing.step(energy)

        if self.persist_dir or self.backend:
            self.save()

        return report

    # ========================= PERSISTENCE =========================

    def save(self, path: str | Path | None = None) -> None:
        anchors_data = {
            nid: [a.to_dict() for a in anchors]
            for nid, anchors in self._anchors.items()
        }

        if self.backend:
            self.backend.put_nodes("episodic", self.episodic.nodes)
            self.backend.put_nodes("semantic", self.semantic.nodes)
            self.backend.put_anchors(anchors_data)
            self.backend.put_temporal(self.temporal_index.contexts)
            # Save graph alongside the db file
            if hasattr(self.backend, 'db_path'):
                graph_path = Path(self.backend.db_path).with_suffix(".graph.json")
                graph_path.write_text(
                    _json.dumps(self.graph.to_dict(), ensure_ascii=False, indent=2)
                )
            return

        d = Path(path or self.persist_dir or "imi_data")
        d.mkdir(parents=True, exist_ok=True)
        self.episodic.save(d / "episodic.json")
        self.semantic.save(d / "semantic.json")

        (d / "anchors.json").write_text(
            _json.dumps(anchors_data, ensure_ascii=False, indent=2)
        )

        temporal_data = {
            nid: ctx.to_dict() for nid, ctx in self.temporal_index.contexts.items()
        }
        (d / "temporal.json").write_text(
            _json.dumps(temporal_data, ensure_ascii=False, indent=2)
        )

        # Save graph edges
        (d / "graph.json").write_text(
            _json.dumps(self.graph.to_dict(), ensure_ascii=False, indent=2)
        )

    @classmethod
    def from_backend(
        cls,
        backend: StorageBackend,
        embedder: Embedder | None = None,
        llm: LLMAdapter | None = None,
    ) -> IMISpace:
        """Load an IMISpace from a StorageBackend."""
        episodic = VectorStore.from_backend(backend, "episodic")
        semantic = VectorStore.from_backend(backend, "semantic")

        anchors_raw = backend.get_anchors()
        anchors = {}
        for nid, anchor_list in anchors_raw.items():
            anchors[nid] = [Anchor.from_dict(a) for a in anchor_list]

        temporal_index = TemporalIndex()
        temporal_index.contexts = backend.get_temporal()

        # Restore graph if saved alongside db
        graph = MemoryGraph()
        if hasattr(backend, 'db_path'):
            graph_path = Path(backend.db_path).with_suffix(".graph.json")
            if graph_path.exists():
                data = _json.loads(graph_path.read_text())
                graph = MemoryGraph.from_dict(data)

        return cls(
            episodic=episodic,
            semantic=semantic,
            embedder=embedder or SentenceTransformerEmbedder(),
            llm=llm,
            _anchors=anchors,
            temporal_index=temporal_index,
            graph=graph,
            backend=backend,
        )

    @classmethod
    def from_sqlite(
        cls,
        db_path: str | Path,
        embedder: Embedder | None = None,
        llm: LLMAdapter | None = None,
        enable_fts: bool = True,
    ) -> IMISpace:
        """Create or load an IMISpace backed by SQLite (recommended default).

        Zero-infra, O(1) inserts, WAL mode, FTS5 search.
        """
        backend = SQLiteBackend(db_path, enable_fts=enable_fts)
        backend.setup()
        return cls.from_backend(backend, embedder=embedder, llm=llm)

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedder: Embedder | None = None,
        llm: LLMAdapter | None = None,
    ) -> IMISpace:
        d = Path(path)
        episodic = VectorStore.load(d / "episodic.json") if (d / "episodic.json").exists() else VectorStore()
        semantic = VectorStore.load(d / "semantic.json") if (d / "semantic.json").exists() else VectorStore()

        anchors = {}
        if (d / "anchors.json").exists():
            data = _json.loads((d / "anchors.json").read_text())
            for nid, anchor_list in data.items():
                anchors[nid] = [Anchor.from_dict(a) for a in anchor_list]

        temporal_index = TemporalIndex()
        if (d / "temporal.json").exists():
            data = _json.loads((d / "temporal.json").read_text())
            for nid, ctx_d in data.items():
                temporal_index.contexts[nid] = TemporalContext.from_dict(ctx_d)

        graph = MemoryGraph()
        if (d / "graph.json").exists():
            data = _json.loads((d / "graph.json").read_text())
            graph = MemoryGraph.from_dict(data)

        return cls(
            episodic=episodic,
            semantic=semantic,
            embedder=embedder or SentenceTransformerEmbedder(),
            llm=llm,
            _anchors=anchors,
            temporal_index=temporal_index,
            graph=graph,
            persist_dir=d,
        )

    # ========================= INTROSPECTION =========================

    def stats(self) -> dict[str, Any]:
        ep_nodes = self.episodic.nodes
        sem_nodes = self.semantic.nodes

        result: dict[str, Any] = {
            "episodic_total": len(ep_nodes),
            "semantic_total": len(sem_nodes),
            "anchored_memories": len(self._anchors),
            "total_anchors": sum(len(a) for a in self._anchors.values()),
            "temporal_sessions": len(set(
                c.session_id for c in self.temporal_index.contexts.values()
            )),
            "total_affordances": sum(len(n.affordances) for n in ep_nodes + sem_nodes),
            "reconsolidations": len(self.reconsolidation_log),
            "annealing": str(self.annealing),
        }

        if ep_nodes:
            result["avg_surprise"] = np.mean([n.surprise_magnitude for n in ep_nodes])
            result["avg_mass"] = np.mean([n.mass for n in ep_nodes])
            result["avg_salience"] = np.mean([n.affect.salience for n in ep_nodes])

        return result
