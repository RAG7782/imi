# Handoff: Próxima Sessão IMI

> Gerado em 2026-03-27 | Continuação da sessão de integração TimescaleDB + análise estratégica

---

## Estado Atual

### O que foi feito nesta sessão
1. **StorageBackend ABC** implementado com 3 backends: JSON, TimescaleDB (`imi/storage.py`)
2. **Event system** para audit/replay de mutações (`imi/events.py`)
3. **Observability layer** com MetricsCollector + @timed (`imi/observe.py`)
4. **Migration tool** bidirecional JSON ↔ TimescaleDB (`imi/migrate.py`)
5. **23 testes passando** (11 JSON + 10 TSDB + 2 benchmarks)
6. **Benchmark comparativo** JSON vs TimescaleDB (harness completo)
7. **Análise estratégica profunda** com CoT/ToT/GoT das 5 perguntas decisórias
8. **Git repo inicializado** + pushed para GitHub privado

### Repo
- **GitHub**: https://github.com/RAG7782/imi (private)
- **Branch**: main
- **Commit**: `d20d74b` — IMI v3 complete + TimescaleDB persistence

### Infra rodando
- TimescaleDB: `docker compose up -d` → `localhost:5433`
- Conn: `postgresql://imi:imi_dev@localhost:5433/imi`

### Arquivos-chave criados/modificados nesta sessão
| Arquivo | Tipo |
|---------|------|
| `imi/storage.py` | NEW — ABC + JSONBackend + TimescaleDBBackend |
| `imi/events.py` | NEW — Event system |
| `imi/observe.py` | NEW — Observability |
| `imi/migrate.py` | NEW — Migration CLI |
| `imi/store.py` | MODIFIED — backend delegation |
| `imi/space.py` | MODIFIED — backend wiring + from_backend() |
| `imi/maintain.py` | MODIFIED — event hooks |
| `tests/test_storage.py` | NEW — 21 tests |
| `tests/test_benchmark.py` | NEW — benchmark harness |
| `docs/concurrency-model.md` | NEW — design doc |
| `docs/analise-estrategica-profunda.md` | NEW — análise P1-P5 |

---

## Agenda da Próxima Sessão (4 workstreams paralelos)

### WS1: SQLiteBackend
Implementar `SQLiteBackend` no `imi/storage.py` como step intermediário entre JSON e Postgres.

**Por que:** Análise P4 concluiu que TimescaleDB é over-engineering para o sweet spot do IMI (100-5000 memórias). SQLite resolve o bug de performance do JSONBackend (`put_node` reescreve arquivo inteiro = O(n)) com O(1) INSERTs, zero setup, zero infra.

**O que fazer:**
- Implementar `SQLiteBackend` com mesma interface do `StorageBackend` ABC
- Schema: `memory_nodes`, `memory_events`, `temporal_contexts`, `anchors` (mesma estrutura do TSDB mas em SQLite)
- Versioning append-only (mesmo padrão)
- FTS5 para busca textual nos seeds/summaries (bonus)
- Benchmark: JSON vs SQLite vs TimescaleDB no mesmo harness
- Testar WAL mode para reads concorrentes

**Arquivos:** `imi/storage.py` (adicionar classe), `tests/test_storage.py` (adicionar tests)

### WS2: Zoom como wrapper sobre RAG (IMI Lite-B)
Implementar e testar o conceito de zoom hierárquico como camada sobre RAG convencional.

**Por que:** Análise P3 mostrou que zoom + affordances capturam 85% do valor do IMI em 20% do código. Se zoom funciona como wrapper, o IMI completo é questionável para a maioria dos casos.

**O que fazer:**
- Criar `imi/lite.py` com `ZoomRAG` (~50 linhas) e `AffordanceIndex` (~80 linhas)
- Usar ChromaDB como vector DB base (pip install chromadb)
- Implementar: `ingest(text)` → gera 3 summaries + seed + embedding + affordances
- Implementar: `search(query, zoom)` → busca cosine + retorna no zoom level pedido
- Implementar: `search_by_action(query)` → busca por affordance
- Benchmark: IMI Full vs IMI Lite-B vs RAG puro
- Medir: tokens consumidos, recall, nDCG

**Arquivos:** `imi/lite.py` (NEW), `tests/test_lite.py` (NEW)

### WS3: Experimento de validação (300 postmortems)
Rodar o experimento MVP desenhado na análise P5.

**Por que:** IMI tem zero validação empírica. Sem dados, tudo é hipótese.

**O que fazer:**
- Coletar 300 postmortems públicos (fontes: Google SRE book, Cloudflare blog, PagerDuty postmortems, Incident.io public reports)
- Anotar 120 queries (40 conteúdo + 40 ação + 40 transversal)
- Encodar no IMI Full, IMI Lite-B, e RAG baseline (ChromaDB)
- Medir 3 métricas: Recall@tokens, nDCG@5, PatternUtility
- Usar LLM-as-judge para avaliação (custo ~$100)
- **Rodar na Modal** para paralelismo (encoding de 300 memórias × 3 sistemas)

**Arquivos:** `experiments/validation/` (NEW directory), scripts de coleta, encoding, avaliação

### WS4: Perguntas não feitas — investigação profunda
Investigar e responder com dados as perguntas que deveriam ter sido feitas.

**Perguntas a investigar:**

1. **"Zoom como pré-processamento sobre RAG reduz o IMI a ~50 linhas?"**
   - Testar se zoom wrapper é equivalente ao zoom do IMI full
   - Comparar qualidade dos summaries: pré-computados vs RAG top-K truncado

2. **"Predictive coding funciona com LLMs stateless?"**
   - Testar: encodar mesma experiência COM e SEM predictive coding
   - Medir: o surprise_magnitude correlaciona com utilidade da memória no retrieval?
   - Se não correlacionar → cortar as 2 chamadas LLM (economia de ~25%)

3. **"Affordances são estáveis entre runs?"**
   - Encodar mesma experiência 10 vezes
   - Medir: overlap das affordances extraídas
   - Se overlap < 50% → problema de confiabilidade

4. **"IMI vs RAG + Reranker?"**
   - Testar: ChromaDB + Cohere Rerank vs IMI navigate()
   - Se reranker iguala ou supera → affordances precisam repensar

5. **"Bug no threshold de clustering?"**
   - `dream()` usa 0.45, `run_maintenance()` default é 0.80
   - Testar ambos: qual gera patterns mais úteis?
   - Pode ser o bug mais impactante do sistema

6. **"Competidores com validação empírica"**
   - Ler e resumir: HippoRAG, RAPTOR, GraphRAG papers
   - Identificar: o que eles validaram que o IMI não validou?
   - Identificar: o que o IMI faz que eles não fazem?

---

## Decisões pendentes

| Decisão | Opções | Quem decide |
|---------|--------|-------------|
| Manter TimescaleDB ou migrar para SQLite? | Depende do benchmark WS1 | Dados do benchmark |
| IMI Full ou IMI Lite-B como produto? | Depende do experimento WS3 | Dados do experimento |
| Cortar predictive coding? | Depende de WS4 pergunta 2 | Dados de correlação |
| Cortar affect? | Depende de se mass tem impacto mensurável | Teste ablation no WS3 |

---

## Setup para a próxima sessão

```bash
# Navegar para o projeto
cd /Users/renatoaparegomes/experimentos/imi

# Ativar venv
source .venv/bin/activate

# Subir TimescaleDB (se não estiver rodando)
docker compose up -d

# Verificar que tudo funciona
python -m pytest tests/ -v

# Instalar deps extras para WS2 e WS3
pip install chromadb cohere  # para RAG baseline e reranker
pip install modal            # para paralelismo na Modal
```

---

## Referências

- Análise estratégica: `docs/analise-estrategica-profunda.md`
- Análise prática original: `docs/analise-pratica.md`
- Concurrency model: `docs/concurrency-model.md`
- Handoff anterior (TimescaleDB): `HANDOFF-timeseries-checkpointer.md`
- Sessão completa original: `docs/sessao-completa.md`

---

## Contexto de memória Claude Code

A memória persistente em `~/.claude/projects/-Users-renatoaparegomes-experimentos/memory/project_imi.md` está atualizada com:
- Estado do projeto (backends, tests, benchmark results)
- GitHub repo URL
- Decisão B→C hybrid approach
- Weaknesses honestas documentadas
