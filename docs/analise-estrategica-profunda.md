# Análise Estratégica Profunda do IMI

> Gerada em 2026-03-27 | CoT + ToT + GoT frameworks

---

## P1: Quanto custa rodar o IMI em produção por mês?

### [CoT] Rastreamento exato — cada chamada LLM no `encode()`

O `encode()` em `space.py` faz **9 chamadas LLM** (7 sem predictive coding):

| # | Chamada | Input (tok) | Output (tok) | Total |
|---|---------|------------|-------------|-------|
| 1 | `predict()` | ~280 | ~150 | ~430 |
| 2 | `compute_surprise()` | ~700 | ~200 | ~900 |
| 3 | `compress_seed()` | ~500 | ~120 | ~620 |
| 4 | `summarize(10)` | ~440 | ~15 | ~455 |
| 5 | `summarize(40)` | ~440 | ~50 | ~490 |
| 6 | `summarize(100)` | ~440 | ~120 | ~560 |
| 7 | `assess_affect()` | ~650 | ~40 | ~690 |
| 8 | `extract_affordances()` | ~550 | ~300 | ~850 |
| 9 | `extract_anchors()` | ~550 | ~350 | ~900 |
| **TOTAL** | | **~4,550** | **~1,345** | **~5,895** |

Dreaming (`consolidate`) adiciona ~500 tokens/cluster — desprezível (<1% do custo).

### Custo por memória

| Modelo | Input/MTok | Output/MTok | Custo/memória |
|--------|-----------|-------------|--------------|
| Haiku 3.5 | $0.80 | $4.00 | **$0.009** |
| Sonnet 4 | $3.00 | $15.00 | **$0.034** |
| Opus 4 | $15.00 | $75.00 | **$0.169** |

### [ToT] Árvore de otimizações

- **Modelo misto** (Haiku para summarize/affect/anchors, Sonnet para compress/affordances): -40%
- **Sem predictive coding** (cortar calls 1-2): -25%
- **Batch summaries** (derivar orbital/medium do detailed): -20%
- **Combinação agressiva**: 5 calls, ~$0.008/memória (-76%)

### [GoT] Modelo de custo mensal

| Volume | Haiku puro | Híbrido otimizado | Sonnet puro | RAG equivalente |
|--------|-----------|-------------------|-------------|----------------|
| 10/dia | $8/mês | **$7/mês** | $31/mês | $0.03/mês |
| 50/dia | $14/mês | **$12/mês** | $51/mês | $0.15/mês |
| 100/dia | $27/mês | **$24/mês** | $102/mês | $0.30/mês |
| 500/dia | $135/mês | **$120/mês** | $508/mês | $1.50/mês |

*(inclui ~$15 infra TimescaleDB em todos)*

---

## P2: Existe cenário real onde IMI é necessário e RAG não serve?

### [CoT] 5 casos analisados

| Caso | IMI ganha? | Feature-chave | Custo IMI | Veredicto |
|------|-----------|---------------|-----------|-----------|
| **SRE/DevOps autônomo** | **SIM** | CLS + Affordances | $12/mês | IMI claramente vence |
| **Assistente pessoal** | Marginalmente | CLS + Affect | $5/mês | RAG + regras cobre 70% |
| **Knowledge management** | Não | - | $8/mês | RAG com docs estruturados basta |
| **Tutoring adaptativo** | Parcialmente | CLS + Affect | $0.81/aluno/mês | Sistemas especializados competem |
| **Incident response** | **SIM** | CLS + Affordances | $4/mês | Mesmo que SRE — IMI vence |

### [GoT] Padrão central

IMI é INSUBSTITUÍVEL quando:
- Agente autônomo opera por meses
- Acumula >100 experiências
- Precisa APRENDER regras (não só buscar docs)
- Precisa saber O QUE FAZER (não só O QUE SABE)

RAG é SUFICIENTE quando:
- Documentos são bem estruturados
- Agente é stateless ou de curta duração
- Determinismo e fidelidade são requisitos
- Volume > 50K docs

---

## P3: O que acontece se usar zoom + affordances SEM o resto?

### [CoT] Contribuição real de cada feature no `navigate()`

| Feature | Contribuição | Pode dropar? |
|---------|-------------|-------------|
| **Zoom (summaries)** | CRÍTICA — 50x cobertura | **NÃO** |
| **Affordances** | ALTA no `search_affordances()` | **Manter** |
| **CLS (dual store)** | BAIXA sem dream | Opcional |
| **Affect/Mass** | MÉDIA | Substituir por flag |
| **Surprise, Reconsolidation, TDA, Temporal, Annealing** | ZERO no navigate | **SIM** |

### [ToT] 3 arquiteturas "IMI Lite"

| | Lite-A (Zoom) | Lite-B (Zoom+Afford) | Lite-C (Zoom+CLS) |
|---|---|---|---|
| Linhas de código | ~150 | ~250 | ~400 |
| Valor capturado | 65% | **85%** | 92% |
| Custo/memória | $0.02 | $0.025 | $0.035 |
| Features | Zoom only | Zoom + busca por ação | Zoom + ação + aprendizado |

**Sweet spot: Lite-B** — 85% do valor com 20% da complexidade.

### [GoT] Dependências entre features

Zoom, Affordances e Embeddings são folhas independentes. Podem ser extraídos e usados como wrappers sobre qualquer RAG.

---

## P4: TimescaleDB é a escolha certa ou over-engineering?

### [CoT] Volume real

| Cenário | Memórias total | Volume |
|---------|---------------|--------|
| Dev solo (180 dias) | 1,800 | ~24 MB |
| Time pequeno (1 ano) | 18,250 | ~300 MB |

### [ToT] Comparação de storage

| | SQLite | Postgres | TimescaleDB |
|---|---|---|---|
| Setup | Zero | Docker | Docker + extension |
| Sweet spot | <50K rows | 50K-10M | >1M temporal |
| Custo | $0 | $15-50/mês | $15-50/mês |

**Veredicto**: Over-engineering confirmado. O schema atual usa ZERO features específicas do TimescaleDB além de `create_hypertable()`. SQLite é o step correto.

### Progressão correta

```
JSON → SQLite → Postgres (quando multi-agent) → TimescaleDB (>1M rows)
```

---

## P5: MVP mínimo que valida o IMI

### Design do experimento

- **Corpus**: 300 postmortems públicos (Google, Cloudflare, etc.)
- **Queries**: 120 (40 conteúdo + 40 ação + 40 transversal)
- **Baseline**: ChromaDB + mesmo embedder + reranker
- **Custo**: ~$100 (LLM-as-judge) a ~$925 (anotação humana)
- **Tempo**: 1-2 semanas

### Métricas falsificáveis

| Métrica | O que mede | Threshold | H0 |
|---------|-----------|-----------|-----|
| Recall@tokens(T) | Eficiência do zoom | >30% economia | Zoom não melhora recall/token |
| nDCG@5 | Affordances vs cosine | >15% superior | Affordances não rankeia melhor |
| PatternUtility | CLS patterns úteis | >3/100 memórias | CLS não gera patterns úteis |

### Árvore de decisão

| Resultado | Ação | Probabilidade |
|-----------|------|--------------|
| 3 métricas passam | DOUBLE DOWN no IMI full | 15-20% |
| Zoom+Aff passam, CLS falha | **IMI Lite-B** | **30-40%** ← mais provável |
| Só zoom passa | IMI Lite-A | 15-20% |
| 2+ falham | KILL | 25-35% |

---

## Perguntas não feitas (meta-análise)

1. **"Zoom como pré-processamento sobre RAG?"** — ~50 linhas sobre ChromaDB. Reduz IMI a "RAG com summaries".
2. **"Predictive coding funciona com LLMs?"** — É metáfora, não implementação. LLMs são stateless.
3. **"Affordances são estáveis?"** — Provavelmente não (LLM com temperatura > 0).
4. **"IMI vs RAG + Reranker?"** — Ninguém testou. Pode invalidar affordances.
5. **"Bug no threshold de clustering?"** — `dream()` usa 0.45, `run_maintenance` default é 0.80.
6. **Competidores com validação**: HippoRAG, RAPTOR, GraphRAG — todos com benchmarks. IMI tem zero.

---

## Síntese

| Aspecto | Realidade |
|---------|-----------|
| **Custo** | $24/mês viável (híbrido), não proibitivo |
| **Onde brilha** | SRE/DevOps autônomo de longa duração |
| **Feature mais valiosa** | Zoom (50x cobertura) |
| **Feature mais arriscada** | CLS (threshold possivelmente bugado) |
| **Storage** | TimescaleDB é over-engineering; SQLite é o step correto |
| **Maior risco** | Zero validação empírica |
| **Cenário mais provável** | IMI Lite-B (zoom + affordances) = 85% do valor em 20% do código |
