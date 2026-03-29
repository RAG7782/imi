# MCP Server

IMI ships an MCP (Model Context Protocol) server that exposes its memory operations as tools for Claude Code and any other MCP-compatible client.

## Requirements

```bash
pip install "imi-memory[mcp,llm]"
```

## Start the server

```bash
# stdio transport (for Claude Code — default)
python -m imi.mcp_server

# SSE transport (for web clients)
python -m imi.mcp_server --transport sse --port 8080

# Custom database path
IMI_DB=path/to/my_agent.db python -m imi.mcp_server
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IMI_DB` | `imi_memory.db` | Path to the SQLite database |
| `IMI_TRANSPORT` | `stdio` | Transport: `stdio` or `sse` |
| `IMI_PORT` | `8080` | Port for SSE transport |

## .mcp.json configuration

Add this to your project's `.mcp.json` to make IMI available in Claude Code:

```json
{
  "mcpServers": {
    "imi": {
      "command": "python",
      "args": ["-m", "imi.mcp_server"],
      "env": {
        "IMI_DB": "/path/to/your/agent_memory.db"
      }
    }
  }
}
```

Or use a global config in `~/.claude/mcp.json` to share the memory across projects.

## Available tools

| Tool | Description |
|------|-------------|
| `imi_encode` | Store a new memory (experience → memory node with affect, affordances) |
| `imi_navigate` | Search memories by query with adaptive relevance weighting and graph expansion |
| `imi_dream` | Run one consolidation cycle — clusters similar memories into patterns |
| `imi_search_actions` | Find memories by what actions they enable (affordance search) |
| `imi_stats` | Get statistics about the memory space (counts, graph, convergence) |
| `imi_graph_link` | Manually add a causal/co-occurrence edge between two memory nodes |

## Tool signatures

### imi_encode

```
imi_encode(
    experience: str,        # The event/experience to memorize
    tags: str = "",         # Comma-separated tags: "dns,auth,incident"
    source: str = "",       # Origin: "slack", "terminal", "user"
    context_hint: str = "", # Additional context for richer encoding
) → JSON
```

Returns: `{id, summary, tags, affect, affordances, mass, total_memories}`

### imi_navigate

```
imi_navigate(
    query: str,
    top_k: int = 10,
    zoom: str = "medium",           # orbital | medium | detailed | full
    context: str = "",
    relevance_weight: float = -1,   # -1 = auto (adaptive)
) → JSON
```

Returns: `{query, intent, relevance_weight_used, zoom, hits, memories[]}`

### imi_dream

```
imi_dream() → JSON
```

Returns: `{nodes_processed, clusters_formed, patterns_extracted, convergence, total_episodic, total_semantic}`

### imi_search_actions

```
imi_search_actions(
    action_query: str,  # "rollback deployment", "restart service"
    top_k: int = 5,
) → JSON
```

Returns: `{query, results: [{action, confidence, conditions, similarity, memory_summary, node_id}]}`

### imi_stats

```
imi_stats() → JSON
```

Returns: `{episodic_count, semantic_count, total_memories, graph, annealing, persist_dir}`

### imi_graph_link

```
imi_graph_link(
    source_id: str,
    target_id: str,
    edge_type: str = "causal",   # causal | co_occurrence | similar
    label: str = "",
) → JSON
```

## Usage in Claude Code

Once configured in `.mcp.json`, IMI tools appear automatically in Claude Code. Example conversation:

```
User: Store this incident: "Redis OOM at 14:30, eviction policy maxmemory-policy
      was set to noeviction. All writes failed. Fixed by changing to allkeys-lru."

Claude: [calls imi_encode with the experience, tags="redis,oom,incident"]
        Memory stored: ID=f3a9c1, summary="Redis OOM: noeviction policy → write failures, fixed with allkeys-lru"

User: What should I do about a slow database?

Claude: [calls imi_search_actions with action_query="fix slow database"]
        Found: "add read replica" (90% confidence), "check index usage" (80%)...
```
