# IMI — Integrated Memory Intelligence

Cognitive memory system for AI agents. Goes beyond vector retrieval with temporal decay, affordances, graph-augmented multi-hop, and adaptive relevance weighting.

## Why IMI?

Standard RAG treats memory as a flat vector store. IMI adds what RAG misses:

| Feature | RAG | IMI |
|---------|-----|-----|
| Temporal decay | No | Yes |
| Affordances (actionable suggestions) | No | Yes |
| Multi-hop graph retrieval | No | Yes |
| Adaptive query-aware weighting | No | Yes |
| Consolidation (dream cycle) | No | Yes |
| Zero LLM calls at query time | N/A | Yes |

## Key finding

!!! info "The Retrieval vs Relevance Paradox"
    Cognitive features trade -3.7% retrieval accuracy for +6.7% agent-relevant precision, +100% multi-hop recall, and -59% result recency. **No RAG paper addresses this trade-off.**

## Quick install

```bash
pip install imi-memory
```

## 3-line quickstart

```python
from imi import IMISpace

space = IMISpace.from_sqlite("my_agent.db")
space.encode("DNS failure at 03:00 caused auth cascade across 3 services")
result = space.navigate("what caused the auth outage?")
```

## Integrations

- **MCP Server**: `python -m imi.mcp_server` — any LLM client can use IMI
- **REST API**: `uvicorn imi.api:app` — OpenAPI docs at `/docs`
- **LangChain**: `IMIMemory` class implementing `BaseMemory`
