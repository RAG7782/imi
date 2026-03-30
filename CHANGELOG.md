# Changelog

All notable changes to IMI are documented here.

## [0.2.0] - 2026-03-29

### Added
- **MCP Server** — 6 tools (encode, navigate, dream, search_actions, stats, graph_link) via stdio/SSE
- **REST API** — FastAPI with 7 endpoints, OpenAPI docs at `/docs`
- **LangChain integration** — `IMIMemory` implementing save_context/load_memory_variables
- **AMBench standalone** — `python -m imi.benchmark` CLI + importable API
- **Graph layer** — MemoryGraph with spreading activation for multi-hop retrieval (100% R@10)
- **Adaptive relevance weight** — Query intent classifier (temporal/exploratory/action/default)
- **Causal edge detection** — 3 strategies (embedding, LLM, explicit hints)
- **Demo notebook** — Interactive 7-section Jupyter demo
- **Docs site** — 14-page mkdocs-material documentation
- **CI/CD** — GitHub Actions (lint + test on Python 3.11-3.13, PyPI publish on release)
- **84 tests** — including API, MCP server, LangChain, graph, adaptive, causal

### Changed
- Default `relevance_weight` from 0.30 to 0.10 (evidence: WS-A ablation, WS-B temporal)
- Predictive coding now opt-in (`use_predictive_coding=False` default)
- Affordance extraction uses `temperature=0.3` for stability
- SQLite `check_same_thread=False` for async/multi-thread compatibility
- Encode auto-saves via backend (memories persist across restarts)

### Removed
- TimescaleDB backend (-425 lines, SQLite is sufficient for all workloads)
- `migrate.py` (was TSDB-only)
- `docker-compose.yml` (was TSDB-only)

### Research
- 13 experiment workstreams (WS1-WS4, WS-A to WS-I, P1-P4)
- Key finding: cognitive features trade -3.7% R@5 for +6.7% agent precision
- AMBench: first agent memory benchmark (300 incidents, 90 days, 5 metrics)
- arXiv paper draft (7 pages, 7 experiments)

## [0.1.0] - 2026-03-22

### Added
- Initial IMI v3 implementation
- SQLiteBackend, JSONBackend
- Dual-store (episodic + semantic) with ChromaDB
- Predictive coding, CLS, temporal, affect, affordances, reconsolidation, TDA
- 53 tests
