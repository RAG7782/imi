# Architecture

IMI (Integrated Memory Intelligence) is a cognitive memory system for AI agents. It goes beyond vector retrieval by combining temporal decay, affective tagging, graph-augmented search, and action-oriented affordances.

## Query pipeline

```
                       ┌─────────────────────────────────────┐
                       │             IMI Space                │
                       │                                      │
  encode(experience)──▶│  ┌──────────────────────────────┐   │
                       │  │        Episodic Store         │   │
     ① compress seed   │  │  (fast, high-fidelity, decays)│   │
     ② zoom summaries  │  └───────────────┬──────────────┘   │
     ③ affect score    │                  │                   │
     ④ affordances     │          dream() │ consolidation     │
     ⑤ temporal ctx    │                  ▼                   │
     ⑥ embed + store   │  ┌──────────────────────────────┐   │◀── search_affordances()
     ⑦ graph auto-link │  │        Semantic Store        │   │
                       │  │  (slow, generalized patterns) │   │
  navigate(query) ────▶│  └───────────────┬──────────────┘   │
                       │                  │                   │
     ① embed query     │  ┌───────────────▼──────────────┐   │
     ② adaptive rw     │  │         Graph Layer           │   │
     ③ cosine search   │  │  (edges: causal/co-occ/sim)  │   │
     ④ graph expand    │  └──────────────────────────────┘   │
     ⑤ re-rank         │                                      │
     ⑥ zoom content    └─────────────────────────────────────┘
     ⑦ return result
```

## Module table

| Module | Responsibility |
|--------|---------------|
| `space.py` | `IMISpace` — top-level facade: encode, navigate, dream, search_affordances |
| `node.py` | `MemoryNode` — data structure with seed, embeddings, affect, affordances, mass |
| `store.py` | `VectorStore` — cosine search with configurable relevance_weight (default 0.10) |
| `graph.py` | `MemoryGraph` — edges (causal, co_occurrence, similar) + spreading activation |
| `adaptive.py` | `AdaptiveRW` — keyword-based query intent → optimal relevance_weight |
| `affect.py` | `AffectiveTag` — salience, valence, arousal → fade_resistance, mass |
| `affordance.py` | `Affordance` — action potentials extracted by LLM at encode time |
| `maintain.py` | Consolidation: cluster episodic → semantic patterns (dreaming) |
| `embedder.py` | `SentenceTransformerEmbedder` — all-MiniLM-L6-v2, 384d |
| `storage.py` | `SQLiteBackend` — WAL mode, FTS5 search, events log |
| `temporal.py` | `TemporalContext` — session tracking, time-based navigation |
| `tda.py` | `TDAReport`, `AnnealingState` — persistent homology, convergence tracking |
| `surprise.py` | Predictive coding — opt-in, 2 LLM calls per encode |
| `reconsolidate.py` | Memory reconsolidation on access (zoom=FULL) |
| `anchors.py` | Anti-confabulation anchors for confidence scoring |
| `llm.py` | `LLMAdapter` — wraps Anthropic Claude with temperature support |
| `mcp_server.py` | FastMCP server — exposes all IMI operations as MCP tools |
| `api.py` | FastAPI REST server — HTTP interface to IMISpace |
| `lite.py` | `IMI Lite-B` — minimal variant: cosine + zoom + affordances only |

## Data flow

### Encode path

```
experience (str)
    │
    ▼
compress_seed()          ← LLM: distill to ≤50 tokens
    │
    ├─ summarize(orbital)  ← ~10 tokens
    ├─ summarize(medium)   ← ~40 tokens
    └─ summarize(detailed) ← ~100 tokens
    │
assess_affect()          ← LLM: salience, valence, arousal
    │
extract_affordances()    ← LLM: list of Affordance objects (max 4)
    │
extract_anchors()        ← LLM: anti-confabulation facts
    │
embedder.embed()         ← SentenceTransformer: 384d vector
    │
temporal_index.register()  ← timestamp + session_id
    │
episodic.add(node)       ← VectorStore insert
    │
auto_link_causal()       ← Graph edges if cosine ≥ 0.65 (no LLM)
```

### Navigate path

```
query (str)
    │
adaptive_rw.classify()   ← keyword regex → rw value
    │
embedder.embed(query)    ← 384d query vector
    │
graph.search_with_expansion()  ← if graph has edges:
    ├─ cosine seeds (top-K)
    ├─ spreading activation (1-hop)
    └─ re-rank: cosine + relevance + graph_activation
    │
  OR episodic.search()   ← if no graph edges: pure cosine + rw
    │
semantic.search()        ← optional, rw=0.1
    │
merge + sort + zoom      ← apply zoom level to each result
    │
NavigationResult         ← memories[], total_tokens, tda
```

## Two-store memory (CLS)

IMI implements Complementary Learning Systems theory:

| Store | Hippocampal analog | Content | Behavior |
|-------|-------------------|---------|----------|
| `episodic` | Hippocampus | Raw experiences | Fast, high-fidelity, decays with time |
| `semantic` | Neocortex | Abstract patterns | Slow, generalized, persists |

Consolidation via `dream()` moves patterns from episodic → semantic, mimicking sleep consolidation.

## Relevance scoring

Each node has a `relevance` score combining:

```
relevance = recency × (1 + frequency) × mass × surprise_boost

recency       = 1 / (1 + days_since × (1 - 0.5 × fade_resistance))
frequency     = log(1 + access_count)
mass          = affect.encoding_strength
surprise_boost = 1 + 0.3 × surprise_magnitude

final_score   = (1 - rw) × cosine_similarity + rw × normalized_relevance
```

Optimal `rw = 0.10` (validated via ablation across 100 SRE incidents).
