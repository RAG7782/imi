"""IMI REST API — FastAPI wrapper over IMISpace.

Endpoints:
  POST /encode        — Store a new memory
  POST /navigate      — Search memories
  POST /dream         — Run consolidation cycle
  POST /search-actions — Search by affordances
  GET  /stats         — Memory space statistics
  POST /graph/link    — Add manual edge
  GET  /health        — Health check

Usage:
  uvicorn imi.api:app --port 8000
  IMI_DB=my_agent.db uvicorn imi.api:app --port 8000

Environment variables:
  IMI_DB: Path to SQLite database (default: imi_memory.db)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class EncodeRequest(BaseModel):
    experience: str = Field(..., description="The experience to memorize")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    source: str = Field("", description="Where this memory came from")
    context_hint: str = Field("", description="Additional context for encoding")


class EncodeResponse(BaseModel):
    id: str
    summary: str
    tags: list[str]
    affect: dict[str, float]
    affordances: list[str]
    mass: float
    total_memories: int


class NavigateRequest(BaseModel):
    query: str = Field(..., description="What to search for")
    top_k: int = Field(10, description="Number of results")
    zoom: str = Field("medium", description="Resolution: orbital, medium, detailed, full")
    context: str = Field("", description="Additional search context")
    relevance_weight: float | None = Field(None, description="Override adaptive weight (None=auto)")


class MemoryHit(BaseModel):
    score: float
    content: str
    id: str = ""
    tags: list[str] = Field(default_factory=list)
    affordances: list[str] = Field(default_factory=list)
    affect: str = ""


class NavigateResponse(BaseModel):
    query: str
    intent: str
    relevance_weight_used: float
    zoom: str
    hits: int
    memories: list[MemoryHit]


class DreamResponse(BaseModel):
    nodes_processed: int
    clusters_formed: int
    patterns_extracted: int
    convergence: dict[str, Any]
    total_episodic: int
    total_semantic: int


class SearchActionsRequest(BaseModel):
    action_query: str = Field(..., description="What action to search for")
    top_k: int = Field(5, description="Number of results")


class ActionHit(BaseModel):
    action: str
    confidence: float
    conditions: str
    similarity: float
    memory_summary: str
    node_id: str


class SearchActionsResponse(BaseModel):
    query: str
    results: list[ActionHit]


class GraphLinkRequest(BaseModel):
    source_id: str
    target_id: str
    edge_type: str = Field("causal", description="causal, co_occurrence, or similar")
    label: str = ""


class GraphLinkResponse(BaseModel):
    status: str
    edge: str
    label: str
    total_edges: int


class StatsResponse(BaseModel):
    episodic_count: int
    semantic_count: int
    total_memories: int
    graph: dict[str, Any]
    annealing: dict[str, Any]
    persist_dir: str | None


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_space = None


def _get_space():
    global _space
    if _space is None:
        from imi.space import IMISpace

        db_path = os.environ.get("IMI_DB", "imi_memory.db")
        persist_dir = Path(db_path).with_suffix("")
        if persist_dir.exists():
            _space = IMISpace.load(persist_dir)
        else:
            _space = IMISpace.from_sqlite(db_path)
    return _space


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_space()  # warm up on startup
    yield


app = FastAPI(
    title="IMI — Integrated Memory Intelligence",
    description="Cognitive memory API for AI agents. Temporal decay, affordances, graph-augmented retrieval, adaptive relevance weighting.",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "version": "0.2.0"}


@app.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest):
    """Store a new memory in the IMI space."""
    space = _get_space()
    node = space.encode(
        req.experience,
        tags=req.tags or None,
        source=req.source,
        context_hint=req.context_hint,
    )
    return EncodeResponse(
        id=node.id,
        summary=node.summary_medium,
        tags=node.tags,
        affect={
            "salience": node.affect.salience if node.affect else 0,
            "valence": node.affect.valence if node.affect else 0,
        },
        affordances=[str(a) for a in node.affordances[:3]],
        mass=round(node.mass, 3),
        total_memories=len(space.episodic),
    )


@app.post("/navigate", response_model=NavigateResponse)
def navigate(req: NavigateRequest):
    """Search memories by query with adaptive relevance weighting and graph expansion."""
    space = _get_space()
    nav = space.navigate(
        req.query,
        zoom=req.zoom,
        top_k=req.top_k,
        context=req.context,
        relevance_weight=req.relevance_weight,
    )
    rw_used, intent_obj = space.adaptive_rw.classify_with_info(req.query)

    memories = []
    for m in nav.memories[:req.top_k]:
        memories.append(MemoryHit(
            score=round(m["score"], 3),
            content=m["content"],
            id=m.get("id", ""),
            tags=m.get("tags", []),
            affordances=m.get("affordances", [])[:2],
            affect=m.get("affect_str", ""),
        ))

    return NavigateResponse(
        query=req.query,
        intent=intent_obj.name,
        relevance_weight_used=round(rw_used if req.relevance_weight is None else req.relevance_weight, 3),
        zoom=req.zoom,
        hits=len(memories),
        memories=memories,
    )


@app.post("/dream", response_model=DreamResponse)
def dream():
    """Run one consolidation (dream) cycle."""
    space = _get_space()
    report = space.dream()
    return DreamResponse(
        nodes_processed=report.nodes_processed,
        clusters_formed=report.clusters_formed,
        patterns_extracted=report.patterns_extracted,
        convergence={
            "energy": round(space.annealing.energy_history[-1], 4) if space.annealing.energy_history else None,
            "iteration": space.annealing.iteration,
            "converged": space.annealing.converged,
        },
        total_episodic=len(space.episodic),
        total_semantic=len(space.semantic),
    )


@app.post("/search-actions", response_model=SearchActionsResponse)
def search_actions(req: SearchActionsRequest):
    """Find memories by what actions they ENABLE."""
    space = _get_space()
    results = space.search_affordances(req.action_query, top_k=req.top_k)

    items = []
    for r in results:
        items.append(ActionHit(
            action=r["action"],
            confidence=round(r["confidence"], 2),
            conditions=r["conditions"],
            similarity=round(r["similarity"], 3),
            memory_summary=r["memory_summary"][:200],
            node_id=r["node_id"],
        ))

    return SearchActionsResponse(query=req.action_query, results=items)


@app.get("/stats", response_model=StatsResponse)
def stats():
    """Get memory space statistics."""
    space = _get_space()
    graph_stats = space.graph.stats()
    return StatsResponse(
        episodic_count=len(space.episodic),
        semantic_count=len(space.semantic),
        total_memories=len(space.episodic) + len(space.semantic),
        graph={
            "total_edges": graph_stats["total_edges"],
            "edge_types": graph_stats.get("by_type", {}),
            "nodes_with_edges": graph_stats.get("nodes_with_edges", 0),
        },
        annealing={
            "iteration": space.annealing.iteration,
            "converged": space.annealing.converged,
            "energy": round(space.annealing.energy_history[-1], 4) if space.annealing.energy_history else None,
        },
        persist_dir=str(space.persist_dir) if space.persist_dir else None,
    )


@app.post("/graph/link", response_model=GraphLinkResponse)
def graph_link(req: GraphLinkRequest):
    """Add a manual edge between two memories."""
    from imi.graph import EdgeType

    space = _get_space()
    type_map = {
        "causal": EdgeType.CAUSAL,
        "co_occurrence": EdgeType.CO_OCCURRENCE,
        "similar": EdgeType.SIMILAR,
    }
    et = type_map.get(req.edge_type, EdgeType.CAUSAL)
    space.graph.add_edge(req.source_id, req.target_id, et, label=req.label)

    if space.persist_dir:
        space.save()

    return GraphLinkResponse(
        status="ok",
        edge=f"{req.source_id} --[{req.edge_type}]--> {req.target_id}",
        label=req.label,
        total_edges=space.graph.stats()["total_edges"],
    )
