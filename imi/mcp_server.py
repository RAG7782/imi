"""IMI MCP Server — expose IMI as tools for any LLM client.

Tools:
  - imi_encode: Store a new memory (experience → memory node)
  - imi_navigate: Search memories by query (with adaptive rw + graph expansion)
  - imi_dream: Run consolidation cycle (cluster similar memories)
  - imi_search_actions: Find memories by what actions they enable
  - imi_stats: Get memory space statistics
  - imi_graph_link: Manually add a causal/co-occurrence edge between memories

Usage:
  python -m imi.mcp_server                           # stdio (for Claude Code)
  python -m imi.mcp_server --transport sse --port 8080  # SSE (for web clients)
  IMI_DB=path/to/memory.db python -m imi.mcp_server  # custom db path

Environment variables:
  IMI_DB: Path to SQLite database (default: imi_memory.db)
  IMI_TRANSPORT: "stdio" or "sse" (default: stdio)
  IMI_PORT: Port for SSE transport (default: 8080)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "IMI Memory",
    instructions="Cognitive memory for AI agents — encode, navigate, dream, search actions. v0.2.0",
    port=int(os.environ.get("IMI_PORT", "8080")),
)

# Lazy-load IMISpace to avoid heavy imports at startup
_space = None


def _get_space():
    """Get or create the global IMISpace instance."""
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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def imi_encode(
    experience: str,
    tags: str = "",
    source: str = "",
    context_hint: str = "",
) -> str:
    """Store a new memory in the IMI space.

    Args:
        experience: The experience/event to memorize (e.g., "DNS failure at 03:00 caused auth cascade")
        tags: Comma-separated tags for categorization (e.g., "dns,auth,incident")
        source: Where this memory came from (e.g., "slack", "terminal", "user")
        context_hint: Additional context for better encoding

    Returns:
        JSON with the memory node ID, summary, affect, and affordances.
    """
    space = _get_space()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    node = space.encode(
        experience,
        tags=tag_list,
        source=source,
        context_hint=context_hint,
    )

    result = {
        "id": node.id,
        "summary": node.summary_medium,
        "tags": node.tags,
        "affect": {
            "salience": node.affect.salience if node.affect else 0,
            "valence": node.affect.valence if node.affect else 0,
        },
        "affordances": [str(a) for a in node.affordances[:3]],
        "mass": round(node.mass, 3),
        "total_memories": len(space.episodic),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def imi_navigate(
    query: str,
    top_k: int = 10,
    zoom: str = "medium",
    context: str = "",
    relevance_weight: float = -1,
) -> str:
    """Search memories by query. Uses adaptive relevance weighting and graph expansion.

    Args:
        query: What to search for (e.g., "auth failures", "recent DNS issues")
        top_k: Number of results to return (default: 10)
        zoom: Resolution level — "orbital" (gist), "medium" (default), "detailed", "full"
        context: Additional context to refine search
        relevance_weight: Override adaptive weight (0.0=pure cosine, 0.15=recency bias, -1=auto)

    Returns:
        JSON with ranked memories, scores, and detected query intent.
    """
    space = _get_space()
    rw = None if relevance_weight < 0 else relevance_weight

    nav = space.navigate(
        query,
        zoom=zoom,
        top_k=top_k,
        context=context,
        relevance_weight=rw,
    )

    # Detect intent for transparency
    rw_used, intent_obj = space.adaptive_rw.classify_with_info(query)
    intent_name = intent_obj.name

    memories = []
    for m in nav.memories[:top_k]:
        mem = {
            "score": round(m["score"], 3),
            "content": m["content"],
            "id": m.get("id", ""),
            "tags": m.get("tags", []),
        }
        if m.get("affordances"):
            mem["affordances"] = m["affordances"][:2]
        if m.get("affect_str"):
            mem["affect"] = m["affect_str"]
        memories.append(mem)

    result = {
        "query": query,
        "intent": intent_name,
        "relevance_weight_used": round(rw_used if rw is None else rw, 3),
        "zoom": zoom,
        "hits": len(memories),
        "memories": memories,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def imi_dream() -> str:
    """Run one consolidation (dream) cycle — clusters similar memories into patterns.

    This is analogous to sleep consolidation in human memory. Call periodically
    (e.g., end of day, every N encodes) to organize the memory space.

    Returns:
        JSON with consolidation report (clusters formed, nodes processed, convergence).
    """
    space = _get_space()
    report = space.dream()

    result = {
        "nodes_processed": report.nodes_processed,
        "clusters_formed": report.clusters_formed,
        "patterns_extracted": report.patterns_extracted,
        "convergence": {
            "energy": round(space.annealing.energy_history[-1], 4) if space.annealing.energy_history else None,
            "steps": space.annealing.iteration,
            "converged": space.annealing.converged,
        },
        "total_episodic": len(space.episodic),
        "total_semantic": len(space.semantic),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def imi_search_actions(
    action_query: str,
    top_k: int = 5,
) -> str:
    """Find memories by what actions they ENABLE, not just content similarity.

    Searches affordances (action potentials) extracted from memories.
    E.g., "restart service" finds memories whose affordances mention restarting.

    Args:
        action_query: What action you want to take (e.g., "restart", "rollback", "escalate")
        top_k: Number of results (default: 5)

    Returns:
        JSON with matching affordances, their confidence, and source memories.
    """
    space = _get_space()
    results = space.search_affordances(action_query, top_k=top_k)

    items = []
    for r in results:
        items.append({
            "action": r["action"],
            "confidence": round(r["confidence"], 2),
            "conditions": r["conditions"],
            "similarity": round(r["similarity"], 3),
            "memory_summary": r["memory_summary"][:200],
            "node_id": r["node_id"],
        })

    return json.dumps({"query": action_query, "results": items}, ensure_ascii=False, indent=2)


@mcp.tool()
def imi_stats() -> str:
    """Get statistics about the current memory space.

    Returns:
        JSON with counts, graph stats, and convergence state.
    """
    space = _get_space()
    graph_stats = space.graph.stats()

    result = {
        "episodic_count": len(space.episodic),
        "semantic_count": len(space.semantic),
        "total_memories": len(space.episodic) + len(space.semantic),
        "graph": {
            "total_edges": graph_stats["total_edges"],
            "edge_types": graph_stats.get("by_type", {}),
            "nodes_with_edges": graph_stats.get("nodes_with_edges", 0),
        },
        "annealing": {
            "steps": space.annealing.iteration,
            "converged": space.annealing.converged,
            "energy": round(space.annealing.energy_history[-1], 4) if space.annealing.energy_history else None,
        },
        "persist_dir": str(space.persist_dir) if space.persist_dir else None,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def imi_graph_link(
    source_id: str,
    target_id: str,
    edge_type: str = "causal",
    label: str = "",
) -> str:
    """Add a manual edge between two memories in the graph.

    Args:
        source_id: ID of the source memory node
        target_id: ID of the target memory node
        edge_type: "causal", "co_occurrence", or "similar"
        label: Optional description of the relationship

    Returns:
        JSON confirming the edge was added.
    """
    from imi.graph import EdgeType

    space = _get_space()

    type_map = {
        "causal": EdgeType.CAUSAL,
        "co_occurrence": EdgeType.CO_OCCURRENCE,
        "similar": EdgeType.SIMILAR,
    }
    et = type_map.get(edge_type, EdgeType.CAUSAL)
    space.graph.add_edge(source_id, target_id, et, label=label)

    if space.persist_dir:
        space.save()

    return json.dumps({
        "status": "ok",
        "edge": f"{source_id} --[{edge_type}]--> {target_id}",
        "label": label,
        "total_edges": space.graph.stats()["total_edges"],
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    transport = os.environ.get("IMI_TRANSPORT", "stdio")
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        transport = sys.argv[idx + 1]

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
