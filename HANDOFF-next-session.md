# Handoff: Próxima Sessão IMI

> Gerado em 2026-03-29 | Semana 1 completa: Publish & Package

---

## Estado Atual

### Tarefas completadas nesta sessão

| # | Tarefa | Status | Detalhes |
|---|--------|--------|----------|
| 1 | PyPI package | Done | pyproject.toml completo, `imi-memory` v0.2.0, sdist+wheel building |
| 2 | LICENSE + .gitignore | Done | MIT license, .gitignore expandido |
| 3 | README.md | Done | Quickstart, badges, architecture, experiments, citation |
| 4 | CI/CD | Done | `.github/workflows/ci.yml` (lint+test 3.11-3.13) + `publish.yml` (PyPI on release) |
| 5 | arXiv paper | Done | English polished, acknowledgments, future work updated, submit-ready |

### Testes: 53 passando, zero regressões

### Package build: `imi_memory-0.2.0.tar.gz` + `.whl` building OK

---

## Próximos passos imediatos

### 1. Git init + primeiro push
```bash
cd ~/experimentos/imi
git init
git add .
git commit -m "Initial commit: IMI v0.2.0"
git remote add origin git@github.com:renatoaparegomes/imi.git
git push -u origin main
```

### 2. PyPI publish
```bash
pip install twine
twine upload dist/*
# Ou: criar GitHub release → workflow publish.yml faz automaticamente
```

### 3. arXiv submit
- Upload `docs/paper-draft.md` convertido para LaTeX/PDF
- Categoria: cs.AI ou cs.CL
- Incluir figuras de `docs/figures/`

### 4. Semana 3-4: Real-world Validation
- Integrar IMI no Claude Code como memory hook
- Rodar com agente SRE real por 2 semanas
- Medir task completion antes/depois

### 5. Semana 5-8: Product
- API REST (FastAPI): /encode, /navigate, /dream, /affordances
- IMI como MCP server
- LangChain integration

---

## Como rodar

```bash
source .venv/bin/activate
python -m pytest tests/ -v                                       # 53 tests
python -m build                                                  # Build package

# Experiments
PYTHONPATH=. python experiments/ws_a_ablation_study.py
PYTHONPATH=. python experiments/ws_b_temporal_decay.py
PYTHONPATH=. python experiments/ws_g_graph_augmented_retrieval.py
PYTHONPATH=. python experiments/ws_d_agent_memory_benchmark.py
PYTHONPATH=. python experiments/p1_adaptive_rw.py
PYTHONPATH=. python experiments/p2_causal_detection.py
PYTHONPATH=. python experiments/p3_paper_figures.py
```

## Docs de referência
- `docs/INSIGHTS.md` — Consolidação completa
- `docs/PERSPECTIVAS.md` — Análise multi-dimensional (8 perspectivas)
- `docs/paper-draft.md` — Paper arXiv-ready
- `docs/figures/` — 5 PNG + 1 TXT
