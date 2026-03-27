# Handoff: Próxima Sessão IMI

> Gerado em 2026-03-27 | Continuação da sessão WS1-WS4

---

## Estado Atual

### O que foi feito nesta sessão (5 commits)
1. **SQLiteBackend** — zero-infra, 87x mais rápido que JSON, FTS5, WAL mode (`imi/storage.py`)
2. **Fix threshold clustering** — 0.80/0.85 → 0.45 (consolidação estava morta) (`imi/maintain.py`)
3. **Surprise integrado no relevance** — era dead code, agora boost de 30% (`imi/node.py`)
4. **IMI Lite-B (ZoomRAG)** — zoom+affordances sobre ChromaDB, ~160 linhas (`imi/lite.py`)
5. **Validation framework WS3** — 100 postmortems, 5 domínios, métricas empíricas (`experiments/ws3_validation_framework.py`)

### Testes: 46 passando, zero regressões

### Backends disponíveis
- **SQLiteBackend** ← recomendado (default via `IMISpace.from_sqlite()`)
- JSONBackend (legado, O(n) put_node)
- TimescaleDBBackend (over-engineering, requer Docker)

---

## Resultados WS4 — Todas as 6 perguntas respondidas

| # | Pergunta | Resultado |
|---|----------|-----------|
| Q1 | Zoom = RAG wrapper? | **Não.** IMI core = ~200 linhas, zoom wrapper = ~50 mas perde affect/affordances/reconstrução |
| Q2 | Predictive coding funciona? | **Era dead code.** Integrado no relevance (boost 30%). 2 LLM calls (~1000 tok) por encoding |
| Q3 | Affordances estáveis? | **Textualmente 10-20%, semanticamente 75-85%.** Embedding absorve variação |
| Q4 | IMI vs RAG+Reranker? | **Empate** no retrieval puro. IMI ganha em features (zoom/affordances/affect) |
| Q5 | Bug no threshold? | **Sim, inverso**: 0.80/0.85 matava consolidação. 0.45 correto. Corrigido |
| Q6 | Competidores? | HippoRAG (NeurIPS'24), RAPTOR (ICLR'24), GraphRAG (MS). **IMI: zero benchmarks** |

## Resultados WS2 — IMI Full vs Lite-B

| Métrica | RAG Puro | Lite-B | IMI Full |
|---------|----------|--------|----------|
| Recall@5 | 0.583 | 0.583 | 0.583 |
| Features | Retrieval | +Zoom +Affordances | +Relevance +Affect +Surprise +Temporal +Consolidation |
| LOC | ~10 | ~160 | ~2000+ |

**Conclusão**: Retrieval idêntico. Full's value = features não-retrieval.

## Resultados WS3 — Validação Empírica (100 postmortems)

| Métrica | Valor |
|---------|-------|
| Recall@5 | 0.341 |
| Recall@10 | 0.525 |
| nDCG@5 | 0.447 |
| MRR | 0.643 |
| Cluster purity | **0.798** (14 clusters, 5 domains) |
| Affordance precision@3 | 0.67-1.00 |
| Zoom token savings | 10x |

---

## Workstreams para próxima sessão

### WS-A: Ablation Study (prioridade alta)
**Pergunta**: Quanto cada feature contribui individualmente?
- [ ] IMI Full vs no-surprise vs no-affect vs no-affordances vs pure-cosine
- [ ] Medir Recall@5/nDCG@5 para cada variante
- [ ] Determinar quais features justificam seu custo computacional
- Script base: `experiments/ws3_validation_framework.py`

### WS-B: Temporal Decay Test
**Pergunta**: Relevance weighting (recency/frequency) melhora retrieval ao longo do tempo?
- [ ] Simular 90 dias de uso com acesso desigual a memórias
- [ ] Comparar retrieval com/sem relevance weighting
- [ ] Testar se memórias acessadas recentemente sobem no ranking

### WS-C: HippoRAG Comparison
**Pergunta**: IMI vs HippoRAG no mesmo dataset
- [ ] Instalar HippoRAG
- [ ] Rodar nos 100 postmortems do WS3
- [ ] Comparar Recall@K e custo (LLM calls / tokens)
- Resultado esperado: IMI perde em multi-hop, ganha em action-oriented retrieval

### WS-D: Agent Memory Benchmark (contribuição acadêmica)
**Pergunta**: Criar o primeiro benchmark padrão para memória de agentes
- [ ] Definir task: agente SRE processando 300 incidentes em 90 dias simulados
- [ ] Métricas: retrieval accuracy, consolidation quality, action relevance, temporal coherence
- [ ] Baseline: RAG puro, IMI Lite-B, IMI Full
- Nenhum competidor publica benchmark para este caso de uso

### WS-E: Decisões de arquitetura pendentes
1. **Cortar TimescaleDBBackend?** — SQLite domina em tudo, TSDB é complexidade desnecessária
2. **Tornar predictive coding opt-in?** — 2 LLM calls extras, benefício marginal (30% boost)
3. **Adicionar temperature=0.3 às affordances?** — Melhoria de estabilidade grátis

---

## Como rodar

```bash
source .venv/bin/activate
python -m pytest tests/ -v                                    # 46 tests
PYTHONPATH=. python experiments/ws3_validation_framework.py   # WS3 validation
PYTHONPATH=. python experiments/ws2_full_vs_liteb_benchmark.py # WS2 comparison
PYTHONPATH=. python experiments/ws4_threshold_analysis.py      # Threshold analysis
PYTHONPATH=. python experiments/ws4_imi_vs_rag_reranker.py     # Q4 RAG comparison
PYTHONPATH=. python tests/test_benchmark.py                    # Storage benchmark
docker compose up -d  # TimescaleDB (optional)
```

## Arquivos-chave modificados nesta sessão
- `imi/storage.py` — SQLiteBackend (novo)
- `imi/node.py` — surprise no relevance scoring
- `imi/space.py` — from_sqlite() factory
- `imi/maintain.py` — threshold fix 0.80→0.45
- `imi/lite.py` — ZoomRAG Lite-B (novo)
- `experiments/ws3_validation_framework.py` — validation benchmark (novo)
- `experiments/ws2_full_vs_liteb_benchmark.py` — Full vs Lite-B (novo)
- `experiments/ws4_threshold_analysis.py` — threshold analysis (novo)
- `experiments/ws4_imi_vs_rag_reranker.py` — IMI vs RAG (novo)
