# Handoff: Próxima Sessão IMI

> Gerado em 2026-03-29 | Semana 1-2 completas + MCP server + arXiv LaTeX

---

## Estado Atual

### Concluído nesta sessão

| # | Tarefa | Detalhes |
|---|--------|----------|
| 1 | PyPI package | `imi-memory` v0.2.0, sdist+wheel OK |
| 2 | LICENSE + .gitignore | MIT, 20+ patterns |
| 3 | README.md | Quickstart, architecture, citation |
| 4 | CI/CD | lint+test (3.11-3.13) + PyPI publish on release |
| 5 | arXiv paper | LaTeX compilado, 7 páginas, `docs/arxiv/imi-paper.pdf` |
| 6 | **MCP Server** | 6 tools, stdio+SSE, testado com protocol handshake |
| 7 | **Claude Code config** | `.mcp.json` pronto para uso |
| 8 | NotebookLM | 67 sources (todos os arquivos do projeto) |

### Testes: 53 passando, zero regressões

### MCP Server: 6 tools
- `imi_encode` — store memory
- `imi_navigate` — search with adaptive rw + graph expansion
- `imi_dream` — consolidation cycle
- `imi_search_actions` — search by affordances
- `imi_stats` — memory space statistics
- `imi_graph_link` — manual edge creation

---

## Próximos passos

### 1. PyPI publish
```bash
twine upload dist/*  # precisa de token PyPI
# Ou: GitHub release → workflow publish.yml automático
```

### 2. arXiv submit
PDF pronto em `docs/arxiv/imi-paper.pdf`. Upload em arxiv.org, categoria cs.AI.

### 3. Real-world validation (Semana 3-4)
- Rodar Claude Code com IMI MCP server ativado (`.mcp.json` já existe)
- Usar por 2 semanas em tarefas reais
- Medir: task completion, recall qualitativo, memórias úteis vs noise

### 4. Semana 5-8
- API REST (FastAPI)
- Docs site (mkdocs)
- LangChain integration (BaseMemory adapter)
- FAISS backend para >50K memories

---

## Como rodar

```bash
source .venv/bin/activate
python -m pytest tests/ -v                    # 53 tests
python -m build                               # Package

# MCP Server (stdio)
python -m imi.mcp_server

# MCP Server (SSE)
IMI_TRANSPORT=sse IMI_PORT=8080 python -m imi.mcp_server

# Experiments
PYTHONPATH=. python experiments/ws_a_ablation_study.py
PYTHONPATH=. python experiments/ws_g_graph_augmented_retrieval.py
PYTHONPATH=. python experiments/p1_adaptive_rw.py
```
