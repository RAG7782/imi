# IMI v0.2.0 — Release Notes (DRAFT — not published yet)

> **RAG finds what's similar. IMI finds what matters.**

## Highlights

**MCP Server** — 6 tools, any LLM client can use IMI as cognitive memory
**REST API** — FastAPI with OpenAPI docs
**LangChain** — Drop-in memory backend
**AMBench** — First agent memory benchmark, now a standalone CLI
**Graph Layer** — 100% multi-hop recall via spreading activation
**84 Tests** — Full coverage of API, MCP, LangChain, graph, adaptive rw

## What's new

### For developers
- `pip install imi-memory` (or `[mcp]`, `[api]`, `[langchain]`, `[all]`)
- `python -m imi.mcp_server` — instant MCP server
- `uvicorn imi.api:app` — instant REST API
- `python -m imi.benchmark` — run AMBench in one command

### For researchers
- 13 reproducible experiment workstreams
- arXiv paper draft (7 pages)
- AMBench: 300 incidents, 90 days, 5 metrics, 3 baselines
- Key finding: the Retrieval vs Relevance Paradox

### Architecture changes
- Default rw: 0.30 → 0.10 (ablation-validated)
- Predictive coding: now opt-in (marginal benefit, high cost)
- TimescaleDB: removed (SQLite handles all workloads)
- Persistence: auto-save via backend on every encode

## Install

```bash
pip install imi-memory[all]
```

## Quick test

```python
from imi import IMISpace
space = IMISpace.from_sqlite("test.db")
space.encode("DNS failure caused auth cascade")
print(space.navigate("auth issues"))
```

## Full changelog

See [CHANGELOG.md](CHANGELOG.md)

---

*To publish: `gh release create v0.2.0 --title "v0.2.0" --notes-file docs/RELEASE-v0.2.0-draft.md dist/*`*
