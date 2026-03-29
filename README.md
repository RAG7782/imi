# IMI — Integrated Memory Intelligence

Cognitive memory system for AI agents. Goes beyond vector retrieval with temporal decay, affordances, graph-augmented multi-hop, and adaptive relevance weighting.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-53%20passing-brightgreen.svg)]()

## Why IMI?

Standard RAG treats memory as a flat vector store. IMI adds what RAG misses:

| Feature | RAG | IMI |
|---------|-----|-----|
| Temporal decay (recency bias) | No | Yes |
| Affordances (actionable suggestions) | No | Yes |
| Multi-hop graph retrieval | No | Yes |
| Adaptive query-aware weighting | No | Yes |
| Affective modulation | No | Yes |
| Consolidation (dream cycle) | No | Yes |
| Zero LLM calls at query time | N/A | Yes |

**Key finding**: Cognitive features trade -3.7% retrieval accuracy for +6.7% agent-relevant precision, +100% multi-hop recall, and -59% result recency. [Paper draft](docs/paper-draft.md)

## Install

```bash
pip install imi-memory
```

With LLM features (affordances, predictive coding):
```bash
pip install imi-memory[llm]
```

## Quickstart

```python
from imi import IMISpace

# Create a memory space (SQLite-backed, zero config)
space = IMISpace.from_sqlite("my_agent.db")

# Encode a memory
space.encode("DNS failure at 03:00 caused auth cascade across 3 services")

# Navigate (retrieve) — adaptive relevance weight, graph expansion
result = space.navigate("what caused the auth outage?")
for node, score in result.memories[:3]:
    print(f"  [{score:.2f}] {node.content[:80]}")
```

## Architecture

```
Query → AdaptiveRW(intent) → rw
     → Cosine search (rw) → seed results
     → Graph expansion (spreading activation) → expanded set
     → Re-rank (cosine x relevance x graph) → final results
     → Zoom level selection → response
```

**Zero LLM calls at query time.** Encoding optionally uses LLM for affordances and predictive coding.

### Core modules

| Module | Purpose |
|--------|---------|
| `space.py` | IMISpace — main API (encode, navigate, dream) |
| `store.py` | VectorStore over ChromaDB with relevance weighting |
| `storage.py` | SQLiteBackend for persistence |
| `graph.py` | MemoryGraph — multi-hop via spreading activation |
| `adaptive.py` | AdaptiveRW — query intent → optimal relevance weight |
| `causal.py` | Auto-detect causal edges between memories |
| `node.py` | MemoryNode with affect, surprise, relevance scoring |
| `maintain.py` | Consolidation (dream cycle) via clustering |
| `affordance.py` | Extract actionable suggestions from memories |
| `affect.py` | Affective tagging (urgency, severity, emotional weight) |
| `temporal.py` | Temporal context tracking |

## Features in depth

### Adaptive relevance weight

Queries have different intents. IMI auto-detects and adjusts:

| Intent | Example | rw |
|--------|---------|-----|
| Temporal | "recent auth failures" | 0.15 |
| Exploratory | "find all cert incidents" | 0.00 |
| Action | "how to fix DNS" | 0.05 |
| Default | "auth token errors" | 0.10 |

```python
# Automatic (default)
result = space.navigate("recent failures")  # uses rw=0.15

# Manual override
result = space.navigate("find all", relevance_weight=0.0)
```

### Graph-augmented retrieval

Lightweight edges between memories enable multi-hop reasoning:

```python
from imi.graph import MemoryGraph, EdgeType

# Edges are auto-created during encode (similarity-based)
# Or manually:
space.graph.add_edge("mem_01", "mem_02", EdgeType.CAUSAL, label="caused")

# Navigate uses graph expansion automatically
result = space.navigate("what caused the cascade?")
```

### Consolidation (dream cycle)

```python
# Cluster similar memories, extract patterns
report = space.dream()
print(f"Consolidated {report.nodes_processed} memories into {report.clusters_formed} clusters")
```

### Zoom levels

```python
from imi import Zoom

result = space.navigate("DNS issues", zoom=Zoom.MICRO)    # Full detail
result = space.navigate("DNS issues", zoom=Zoom.MACRO)    # High-level summary
```

## Experiments & benchmarks

All experiments are reproducible:

```bash
source .venv/bin/activate

# Ablation study — feature contribution analysis
PYTHONPATH=. python experiments/ws_a_ablation_study.py

# Temporal decay — recency/frequency impact
PYTHONPATH=. python experiments/ws_b_temporal_decay.py

# Graph-augmented retrieval — multi-hop improvement
PYTHONPATH=. python experiments/ws_g_graph_augmented_retrieval.py

# AMBench — Agent Memory Benchmark (300 incidents, 90 days)
PYTHONPATH=. python experiments/ws_d_agent_memory_benchmark.py

# Adaptive rw validation
PYTHONPATH=. python experiments/p1_adaptive_rw.py
```

## Key results

| Metric | Value | Source |
|--------|-------|--------|
| Recall@5 (standard) | 0.341 | WS3 |
| Multi-hop Recall@10 | 1.000 (20/20) | WS-G |
| Adaptive MRR | 0.651 (> best fixed 0.643) | P1 |
| Temporal coherence | avg age 16.8d vs RAG 41.2d | WS-D |
| Cluster purity | 0.736 | WS-D |

## Development

```bash
git clone https://github.com/renatoaparegomes/imi.git
cd imi
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"
python -m pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use IMI in research, please cite:

```bibtex
@software{gomes2026imi,
  title={IMI: Integrated Memory Intelligence for AI Agents},
  author={Gomes, Renato Aparecido},
  year={2026},
  url={https://github.com/renatoaparegomes/imi}
}
```
