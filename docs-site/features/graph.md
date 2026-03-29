# Graph Layer

The graph layer adds multi-hop retrieval to IMI without requiring a full knowledge graph or LLM calls at query time. It implements spreading activation over a lightweight edge set stored alongside the vector stores.

## Edge types

| Type | Meaning | How created |
|------|---------|-------------|
| `CAUSAL` | A caused B (or A was caused by B) | Auto-detected at encode time if cosine ≥ 0.65, or added manually |
| `CO_OCCURRENCE` | A and B happened in the same context/session | Auto-detected by shared tags |
| `SIMILAR` | A and B are semantically close | Auto-detected by cosine ≥ 0.75 |

All edges are bidirectional by default. Each edge carries a `weight` (float) and optional `label` string.

## Auto-linking at encode time

When a new memory is encoded, `auto_link_causal()` runs automatically if the episodic store has more than 5 nodes:

```python
auto_link_causal(
    node, episodic_store, graph,
    threshold=0.65,   # cosine similarity threshold
    max_edges=2,      # max edges added per new node
    llm=None,         # no LLM calls
)
```

This creates `SIMILAR` or `CAUSAL` edges based on embedding cosine similarity alone. Causality that requires reasoning (low-similarity pairs that are logically related) requires manual hints or LLM confirmation — see [Research: Insights](../research/insights.md).

You can also trigger bulk auto-linking:

```python
# Link by semantic similarity (cosine ≥ 0.75)
n = space.graph.auto_link_similar(space.episodic, threshold=0.75)

# Link by shared tags
n = space.graph.auto_link_co_occurring(space.episodic)
```

## Manual edge creation

```python
from imi.graph import EdgeType

space.graph.add_edge(
    source_id="abc123",
    target_id="def456",
    edge_type=EdgeType.CAUSAL,
    weight=1.0,
    label="DNS cert expiry caused auth cascade",
)
```

Or via the MCP tool `imi_graph_link` / REST endpoint `POST /graph/link`.

## Spreading activation

The `expand()` method implements Collins & Loftus (1975) spreading activation:

```python
# Expand from 3 seed nodes, 1 hop
activation = space.graph.expand(
    seed_ids=["abc123", "def456", "ghi789"],
    hops=1,
)
# Returns {node_id: activation_score}
# Score decays with distance: 0.5 at hop 1, 0.33 at hop 2
```

Activation decays by `1 / (hop + 2)` per hop. Seeds start at activation 1.0.

## Graph-augmented retrieval

When the graph has edges, `navigate()` automatically uses `search_with_expansion()`:

```
Score = (1 - rw - gw) × cosine
      + rw × normalized_relevance
      + gw × graph_activation
```

Default `graph_weight = 0.20`. Graph-discovered nodes that weren't in the original cosine results are added if their activation score > 0.1.

You can disable graph expansion per query:

```python
result = space.navigate("auth failures", use_graph=False)
result = space.navigate("auth failures", use_graph=True, graph_weight=0.3)
```

## Inspecting the graph

```python
stats = space.graph.stats()
# {
#   'total_edges': 14,
#   'nodes_with_edges': 8,
#   'edge_types': {'causal': 6, 'similar': 5, 'co_occurrence': 3},
#   'avg_degree': 1.75
# }

# Neighbors of a specific node
neighbors = space.graph.neighbors("abc123", edge_type=EdgeType.CAUSAL)
for target_id, edge in neighbors:
    print(f"→ {target_id} [{edge.weight:.2f}] {edge.label}")
```

## Multi-hop results from WS-G

Experiment WS-G validated graph-augmented retrieval on 20 multi-hop queries (5 causal chains, 4 nodes each):

| System | Multi-hop R@10 | Hits | Standard R@5 |
|--------|---------------|------|-------------|
| Cosine only | 0.750 | 15/20 | 0.341 |
| IMI + Graph (1-hop, gw=0.2) | **1.000** | **20/20** | **0.341** |

The 5 queries that cosine failed on were all resolved by 1-hop graph expansion. Critically, standard R@5 did not degrade — graph expansion adds signal without replacing cosine scoring.

Graph construction: 20 nodes, 10 causal edges (manually created for ground truth), threshold=0.65 for auto-linking similar nodes.
