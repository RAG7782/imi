# REST API

IMI ships a FastAPI server for HTTP access to the memory space.

## Requirements

```bash
pip install "imi-memory[api,llm]"
```

## Start the server

```bash
uvicorn imi.api:app --port 8000

# Custom database
IMI_DB=my_agent.db uvicorn imi.api:app --port 8000

# With auto-reload (development)
uvicorn imi.api:app --port 8000 --reload
```

Interactive docs available at `http://localhost:8000/docs` (Swagger UI).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IMI_DB` | `imi_memory.db` | SQLite database path |

## Endpoints

### GET /health

Health check.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "version": "0.2.0"}
```

---

### POST /encode

Store a new memory.

```bash
curl -X POST http://localhost:8000/encode \
  -H "Content-Type: application/json" \
  -d '{
    "experience": "DNS resolver cert expired at 03:00 UTC, caused auth cascade for 12 minutes.",
    "tags": ["dns", "auth", "incident"],
    "source": "postmortem",
    "context_hint": "production incident"
  }'
```

Response:

```json
{
  "id": "abc123de",
  "summary": "DNS cert expired → auth cascade 12min outage (03:00 UTC)",
  "tags": ["dns", "auth", "incident"],
  "affect": {"salience": 0.85, "valence": -0.72},
  "affordances": [
    "[90%] monitor cert expiry proactively (when: certificate-backed services)",
    "[80%] set up cert renewal alerts (when: TLS cert rotation)"
  ],
  "mass": 1.38,
  "total_memories": 1
}
```

---

### POST /navigate

Search memories by query.

```bash
curl -X POST http://localhost:8000/navigate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "recent auth failures",
    "top_k": 5,
    "zoom": "medium",
    "context": "",
    "relevance_weight": null
  }'
```

Response:

```json
{
  "query": "recent auth failures",
  "intent": "TEMPORAL",
  "relevance_weight_used": 0.15,
  "zoom": "medium",
  "hits": 3,
  "memories": [
    {
      "score": 0.87,
      "content": "DNS cert expired → auth cascade 12min outage",
      "id": "abc123de",
      "tags": ["dns", "auth", "incident"],
      "affordances": ["[90%] monitor cert expiry proactively"],
      "affect": "sal=0.85 val=-0.72 aro=0.61"
    }
  ]
}
```

`zoom` options: `orbital`, `medium` (default), `detailed`, `full`.
`relevance_weight`: `null` for adaptive (recommended), or `0.0`–`1.0` to override.

---

### POST /dream

Run one consolidation cycle.

```bash
curl -X POST http://localhost:8000/dream
```

Response:

```json
{
  "nodes_processed": 12,
  "clusters_formed": 2,
  "patterns_extracted": 2,
  "convergence": {
    "energy": 2.3412,
    "iteration": 3,
    "converged": false
  },
  "total_episodic": 12,
  "total_semantic": 2
}
```

---

### POST /search-actions

Find memories by what actions they enable.

```bash
curl -X POST http://localhost:8000/search-actions \
  -H "Content-Type: application/json" \
  -d '{"action_query": "rollback deployment", "top_k": 3}'
```

Response:

```json
{
  "query": "rollback deployment",
  "results": [
    {
      "action": "rollback deployment using Helm release history",
      "confidence": 0.90,
      "conditions": "after a bad deploy causes service degradation",
      "similarity": 0.821,
      "memory_summary": "Prod deploy caused 3x latency spike. Rolled back via helm rollback...",
      "node_id": "xyz789ab"
    }
  ]
}
```

---

### GET /stats

Memory space statistics.

```bash
curl http://localhost:8000/stats
```

Response:

```json
{
  "episodic_count": 24,
  "semantic_count": 3,
  "total_memories": 27,
  "graph": {
    "total_edges": 18,
    "edge_types": {"causal": 8, "similar": 7, "co_occurrence": 3},
    "nodes_with_edges": 14
  },
  "annealing": {
    "iteration": 3,
    "converged": false,
    "energy": 2.3412
  },
  "persist_dir": null
}
```

---

### POST /graph/link

Add a manual edge between two memories.

```bash
curl -X POST http://localhost:8000/graph/link \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "abc123de",
    "target_id": "xyz789ab",
    "edge_type": "causal",
    "label": "cert expiry caused auth cascade"
  }'
```

Response:

```json
{
  "status": "ok",
  "edge": "abc123de --[causal]--> xyz789ab",
  "label": "cert expiry caused auth cascade",
  "total_edges": 19
}
```

`edge_type` options: `causal`, `co_occurrence`, `similar`.
