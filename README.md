# IMI — Integrated Memory Intelligence

### *RAG finds what's similar. IMI finds what matters.*

A cognitive memory system for AI agents that remembers, learns, and acts — not just retrieves.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-84%20passing-brightgreen.svg)]()
[![MCP Server](https://img.shields.io/badge/MCP-6%20tools-purple.svg)]()

---

## The Problem

Your AI agent forgets everything between sessions. RAG gives it a vector store, but that's like giving someone a filing cabinet and calling it memory. Real memory has:

- **Recency** — last week's outage matters more than last year's
- **Actions** — "what can I DO?" not just "what do I know?"
- **Causality** — "A caused B caused C" across multiple memories
- **Emotion** — critical incidents should be harder to forget
- **Consolidation** — patterns emerge over time, like during sleep

IMI adds all of this. Zero LLM calls at query time. SQLite only. 84 tests.

## Why not just use RAG?

Honest comparison:

| Scenario | RAG | IMI | Winner |
|----------|-----|-----|--------|
| Static knowledge Q&A | R@5 = 1.000 | R@5 = 1.000 | Tie |
| Agent memory (temporal) | No recency signal | **+6.7% domain precision** | IMI |
| Multi-hop causal chains | 75% recall | **100% recall** | IMI |
| "What should I do?" | Returns documents | **Returns actions** | IMI |
| Result freshness | avg 41.2 days old | **avg 16.8 days old** | IMI |
| Setup complexity | 1 line | 1 line | Tie |
| LLM calls at query time | 0 | 0 | Tie |
| Pure retrieval accuracy | Baseline | -3.7% (trade-off) | RAG |

**Bottom line**: If your agent just answers questions, use RAG. If your agent needs to *learn from experience and act on it*, use IMI.

## Install

```bash
pip install imi-memory
```

Extras: `pip install imi-memory[llm]` (affordances) · `[mcp]` (MCP server) · `[api]` (REST API) · `[all]`

## Quickstart (3 lines)

```python
from imi import IMISpace

space = IMISpace.from_sqlite("my_agent.db")

# Your agent encodes an experience
space.encode("DNS failure at 03:00 caused auth cascade across 3 services")

# Later, it navigates — adaptive rw + graph expansion, zero LLM calls
result = space.navigate("what caused the auth outage?")
```

## What IMI returns that RAG doesn't

```python
node = space.encode("Redis sentinel failover during network partition, 30s of cache misses")
```

| Component | What it is | Example |
|-----------|-----------|---------|
| **Affect** | Emotional weight (salience, valence) | `salience=0.8, valence=-0.7` |
| **Affordances** | Actions this memory enables | `"implement circuit breaker"`, `"add sentinel monitoring"` |
| **Mass** | Gravitational pull (resists forgetting) | `0.76` — critical incident stays relevant longer |
| **Graph edges** | Causal links to other memories | `→ "auth cascade"`, `→ "DNS root cause"` |

## Integrations

### MCP Server (Claude Code, any LLM client)
```bash
python -m imi.mcp_server   # 6 tools: encode, navigate, dream, search_actions, stats, graph_link
```

### REST API (FastAPI)
```bash
uvicorn imi.api:app         # OpenAPI docs at /docs
```
```bash
curl -X POST localhost:8000/encode -H 'Content-Type: application/json' \
  -d '{"experience": "DNS failure caused auth cascade", "tags": ["dns", "auth"]}'
```

### LangChain
```python
from imi.integrations.langchain import IMIMemory

memory = IMIMemory.from_sqlite("agent.db")
memory.save_context({"input": "DNS failed"}, {"output": "Restarting..."})
relevant = memory.load_memory_variables({"input": "auth issues"})
```

### AMBench — Agent Memory Benchmark
```bash
python -m imi.benchmark                           # 300 incidents, 90 days
python -m imi.benchmark --rw 0.0 --name RAG       # Compare pure cosine
python -m imi.benchmark --json                     # Machine-readable output
```

## How it works

```
Query "recent auth failures"
  │
  ├─ AdaptiveRW: intent=TEMPORAL → rw=0.15
  │
  ├─ VectorStore: cosine search weighted by recency/frequency/affect
  │
  ├─ GraphExpansion: follow causal edges via spreading activation
  │
  ├─ Re-rank: (1-rw-gw) × cosine + rw × relevance + gw × graph_activation
  │
  └─ Zoom: orbital (gist) / medium / detailed / full
```

## Key Results (13 experiments, all reproducible)

| Finding | Evidence | Experiment |
|---------|----------|------------|
| Cognitive features trade -3.7% R@5 for +6.7% agent precision | Ablation across 6 feature variants | WS-A |
| Optimal relevance weight = 0.10-0.15 | Sweep 0.0→0.3, 9 scenarios | WS-A, WS-B |
| Graph expansion: 100% multi-hop recall (vs 75%) | 10 causal chains, 20 queries | WS-G |
| Temporal coherence: avg age 16.8d vs RAG 41.2d | 300 incidents, 90 days | WS-D (AMBench) |
| Causal pairs have low cosine (avg 0.308) | 10 ground truth chains | P2 |
| Adaptive rw beats best fixed (MRR 0.651 vs 0.643) | 16 mixed-intent queries | P1 |

## Development

```bash
git clone https://github.com/RAG7782/imi.git && cd imi
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm,mcp,api]"
python -m pytest tests/ -v   # 84 tests
```

## Docs

Full documentation: [docs site](docs-site/) · [Paper (arXiv-ready)](docs/arxiv/imi-paper.pdf) · [Interactive demo](examples/demo_notebook.ipynb)

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{gomes2026imi,
  title={IMI: Integrated Memory Intelligence for AI Agents},
  author={Gomes, Renato Aparecido},
  year={2026},
  url={https://github.com/RAG7782/imi}
}
```
