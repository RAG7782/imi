# API Reference

Quick reference for the public methods of `IMISpace`. All methods are on the `imi.space.IMISpace` class.

---

## Construction

### `IMISpace.from_sqlite(db_path, embedder=None, llm=None, enable_fts=True)`

Create or load an IMISpace backed by SQLite. Recommended default.

```python
space = IMISpace.from_sqlite("agent.db")
```

- `db_path` — path to the `.db` file; created if it does not exist
- `embedder` — custom `Embedder` instance; default is selected from env. Use `IMI_EMBEDDER_PROVIDER=ollama` with `IMI_EMBEDDER_MODEL=all-minilm` and `OLLAMA_BASE_URL=http://localhost:11434` for local Ollama embeddings.
- `llm` — custom `LLMAdapter`; defaults to Anthropic Claude via `ANTHROPIC_API_KEY`
- `enable_fts` — enable SQLite FTS5 full-text search index (default `True`)

### `IMISpace.load(path, embedder=None, llm=None)`

Load from a directory of JSON files (legacy format).

```python
space = IMISpace.load("./imi_data/")
```

### `IMISpace.save(path=None)`

Persist to the configured backend or directory.

```python
space.save()          # uses persist_dir or backend
space.save("./backup/")
```

---

## Core methods

### `encode(experience, *, tags=None, source="", context_hint="", use_predictive_coding=False, timestamp=None) → MemoryNode`

Transform an experience string into a memory node and store it in the episodic store.

```python
node = space.encode(
    "DNS cert expired, caused 12min auth outage",
    tags=["dns", "auth", "incident"],
    source="postmortem",
    context_hint="production incident",
)
```

Returns a `MemoryNode` with:
- `node.id` — UUID
- `node.summary_orbital` — ~10 token gist
- `node.summary_medium` — ~40 token summary
- `node.summary_detailed` — ~100 token summary
- `node.affect` — `AffectiveTag(salience, valence, arousal)`
- `node.affordances` — `list[Affordance]`
- `node.mass` — float (gravitational weight from affect)
- `node.temporal` — `TemporalContext`

**Note**: `use_predictive_coding=True` adds 2 LLM calls per encode for surprise computation. Off by default.

---

### `navigate(query, *, zoom="medium", top_k=20, context="", relevance_weight=None, include_semantic=True, include_tda=False, reconsolidate_on_access=False, use_graph=True, graph_weight=0.2) → NavigationResult`

Search the memory space.

```python
result = space.navigate("auth failures", zoom="medium", top_k=10)

for mem in result.memories:
    print(f"[{mem['score']:.2f}] {mem['content']}")
```

Key parameters:
- `zoom` — `"orbital"` | `"medium"` | `"detailed"` | `"full"` — content resolution
- `relevance_weight` — `None` for adaptive (recommended), `0.0`–`1.0` to override
- `use_graph` — enable graph-augmented expansion (default `True`)
- `graph_weight` — graph signal weight in re-ranking (default `0.2`)
- `reconsolidate_on_access` — run reconsolidation for zoom=FULL results (default `False`)

`NavigationResult`:
- `.memories` — list of dicts with `score`, `content`, `id`, `tags`, `affordances`, `affect_str`
- `.total_tokens_approx` — estimated token count for all results
- `.tda` — `TDAReport` if `include_tda=True`

---

### `dream(similarity_threshold=0.45, track_convergence=True) → MaintenanceReport`

Run one consolidation cycle (cluster episodic → semantic patterns).

```python
report = space.dream()
print(report)
# Maintenance: 2 faded, 3 consolidated, 0 pruned, 4 patterns (312ms)
```

`MaintenanceReport`:
- `.faded` — nodes that lost relevance
- `.consolidated` — new patterns created
- `.pruned` — low-relevance nodes flagged
- `.patterns_total` — total in semantic store
- `.duration_ms` — wall time

---

### `search_affordances(action_query, top_k=5) → list[dict]`

Find memories by what actions they enable, ranked by `similarity × confidence`.

```python
results = space.search_affordances("restart service after OOM", top_k=3)
for r in results:
    print(r["action"], r["confidence"], r["conditions"])
```

Each result dict:
- `node_id` — source memory ID
- `affordance` — full string representation
- `action` — verb phrase
- `confidence` — float 0–1
- `conditions` — when this applies
- `similarity` — cosine similarity to action_query
- `memory_summary` — summary_medium of the source node

---

## Utility methods

### `stats() → dict`

Summary statistics of the memory space.

```python
print(space.stats())
# {
#   'episodic_total': 24,
#   'semantic_total': 3,
#   'total_affordances': 72,
#   'temporal_sessions': 2,
#   'reconsolidations': 0,
#   'annealing': 'AnnealingState(iteration=3, converged=False)',
#   'avg_surprise': 0.0,
#   'avg_mass': 1.18,
#   'avg_salience': 0.71
# }
```

### `navigate_temporal(target_time, window_hours=24.0, zoom="medium") → NavigationResult`

Navigate by timestamp instead of semantic query.

```python
import time
result = space.navigate_temporal(
    target_time=time.time() - 3600,  # 1 hour ago
    window_hours=6.0,
)
```

### `compute_topology() → TopologyReport`

Run UMAP + HDBSCAN to compute the spatial topology of the memory space. Requires `[spatial]` extra.

### `compute_tda() → TDAReport`

Run persistent homology analysis (Betti numbers, fragmentation, rumination risk). No extra dependencies beyond numpy.
