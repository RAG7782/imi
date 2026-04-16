"""IMI MCP Server — Token-Optimized (CoD Pass 2)

Wrapper that imports original IMI tools but re-registers them with:
- Shortened tool names (im_enc, im_nav, im_drm, im_sact, im_sts, im_glnk)
- Compressed descriptions (CoD Pass 2 — max density, no arg repetition)
- No server instructions (documented in CLAUDE.md)

Original server: ~/experimentos/tools/imi/imi/mcp_server.py
This file does NOT modify the original — it wraps it.
"""

import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# --- Server setup (no instructions = 0 tokens) ---

mcp = FastMCP("imi", port=int(os.environ.get("IMI_PORT", "8080")))

# Lazy-load IMISpace
_space = None

def _get_space():
    global _space
    if _space is None:
        from imi.space import IMISpace
        db_path = os.environ.get("IMI_DB", "imi_memory.db")
        _space = IMISpace.from_sqlite(db_path)
    return _space

# --- Tools (CoD Pass 2 descriptions) ---

@mcp.tool()
def im_enc(
    experience: str,
    tags: str = "",
    source: str = "",
    context_hint: str = "",
    occurred_at: str = "",
) -> str:
    """Store new memory from experience"""
    import time as _time
    from datetime import datetime, timezone

    space = _get_space()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    event_timestamp = None
    occurred_float = None
    if occurred_at:
        try:
            dt = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
            event_timestamp = dt.timestamp()
            occurred_float = event_timestamp
        except ValueError:
            pass

    node = space.encode(
        experience, tags=tag_list, source=source,
        context_hint=context_hint, timestamp=event_timestamp,
    )
    node.occurred_at = occurred_float

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
def im_nav(
    query: str,
    top_k: int = 10,
    zoom: str = "medium",
    context: str = "",
    relevance_weight: float = -1,
    positional_optimize: bool = True,
) -> str:
    """Search memories with adaptive relevance and graph expansion"""
    space = _get_space()
    rw = None if relevance_weight < 0 else relevance_weight

    nav = space.navigate(
        query, zoom=zoom, top_k=top_k, context=context,
        relevance_weight=rw, positional_optimize=positional_optimize,
    )

    rw_used, intent_obj = space.adaptive_rw.classify_with_info(query)

    memories = []
    for m in nav.memories[:top_k]:
        mem = {"score": round(m["score"], 3), "content": m["content"],
               "id": m.get("id", ""), "tags": m.get("tags", [])}
        if m.get("affordances"): mem["affordances"] = m["affordances"][:2]
        if m.get("affect_str"): mem["affect"] = m["affect_str"]
        memories.append(mem)

    result = {
        "query": query, "intent": intent_obj.name,
        "relevance_weight_used": round(rw_used if rw is None else rw, 3),
        "zoom": zoom, "hits": len(memories), "memories": memories,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def im_drm() -> str:
    """Consolidation cycle — cluster memories into patterns"""
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
def im_sact(action_query: str, top_k: int = 5) -> str:
    """Find memories by enabled actions (affordance search)"""
    space = _get_space()
    results = space.search_affordances(action_query, top_k=top_k)
    items = [{
        "action": r["action"], "confidence": round(r["confidence"], 2),
        "conditions": r["conditions"], "similarity": round(r["similarity"], 3),
        "memory_summary": r["memory_summary"][:200], "node_id": r["node_id"],
    } for r in results]
    return json.dumps({"query": action_query, "results": items}, ensure_ascii=False, indent=2)


@mcp.tool()
def im_sts() -> str:
    """Memory space statistics"""
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
        "tiers": space.tier_stats(),
        "l0_l1_preview": space.get_l0_l1(),
        "l0_l1_tokens": len(space.get_l0_l1()) // 4,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def im_glnk(source_id: str, target_id: str, edge_type: str = "causal", label: str = "") -> str:
    """Add edge between memories in graph"""
    from imi.graph import EdgeType
    space = _get_space()
    type_map = {"causal": EdgeType.CAUSAL, "co_occurrence": EdgeType.CO_OCCURRENCE, "similar": EdgeType.SIMILAR}
    et = type_map.get(edge_type, EdgeType.CAUSAL)
    space.graph.add_edge(source_id, target_id, et, label=label)
    if space.persist_dir: space.save()
    return json.dumps({
        "status": "ok", "edge": f"{source_id} --[{edge_type}]--> {target_id}",
        "label": label, "total_edges": space.graph.stats()["total_edges"],
    }, ensure_ascii=False, indent=2)


# --- Entry point ---

def main():
    transport = os.environ.get("IMI_TRANSPORT", "stdio")
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        transport = sys.argv[idx + 1]
    mcp.run(transport=transport)

if __name__ == "__main__":
    main()
