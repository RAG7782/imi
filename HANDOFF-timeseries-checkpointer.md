# Handoff: Integrar langgraph-checkpoint-timeseries no IMI

> Gerado em 2026-03-27 | Contexto de conversa anterior preservado aqui

---

## Objetivo

Avaliar e implementar a integração do `langgraph-checkpoint-timeseries` (v0.2.0) como camada de persistência do IMI, substituindo o atual sistema de JSON files por um backend time-series otimizado para workloads de agentes concorrentes.

---

## Estado Atual do IMI

### Arquitetura de Persistência (hoje)
- **Backend**: JSON puro em disco (`persist_dir/`)
- **4 arquivos**: `episodic.json`, `semantic.json`, `anchors.json`, `temporal.json`
- **Padrão**: save-on-mutation (após `encode()`, `navigate()`, `dream()`)
- **Sem transações**: writes sequenciais, não atômicos
- **Sem versionamento**: sem rollback, sem history de estado
- **Sem checkpointing**: não há recuperação mid-run

### LangGraph no IMI
- **NÃO usado atualmente** — zero dependências de LangGraph/LangChain
- IMI é pure Python com Anthropic SDK direto
- Orquestração feita pelo `IMISpace` (space.py), não por um graph framework

### Limitações conhecidas (auto-diagnosticadas)
- Scale ceiling: "thousands not millions" de memórias
- JSON persistence não é transacional
- Reconsolidação não-determinística (memórias mudam ao serem acessadas)
- ~500x mais caro que RAG por encoding (múltiplas chamadas LLM)

---

## O que é o langgraph-checkpoint-timeseries

Package: `langgraph-checkpoint-timeseries` v0.2.0 (já instalado)

Drop-in replacement do `PostgresSaver` do LangGraph para bancos time-series:

| Backend | Classe | Caso de uso |
|---------|--------|-------------|
| **TimescaleDB** | `TimescaleDBSaver` | Memória relacional + métricas. UNLOGGED tables + pipeline mode |
| **QuestDB** | `QuestDBSaver` | Write-heavy append-only. Sub-millisecond queries |
| **Kafka** | `KafkaSaver` | Event-sourced. Fire-and-forget. 110x mais rápido que Postgres |

### Benchmarks (vs PostgresSaver)

| Cenário | PostgresSaver | TimescaleDB | QuestDB | Kafka |
|---------|--------------|-------------|---------|-------|
| Sequential Writes (1K) | 345 ops/s | 411 ops/s | 341 ops/s | **37,959 ops/s** |
| Concurrent Writes (15T×200) | 333 ops/s | **1,192 ops/s** | 1,145 ops/s | 20,616 ops/s |
| History Query (list 100) | 6,788 ops/s | 1,259 ops/s | 577 ops/s | **93,186 ops/s** |

---

## Caminhos de Integração (avaliar)

### Caminho A: IMI como LangGraph workflow (refactor grande)
- Reescrever `IMISpace` como um `StateGraph` do LangGraph
- Cada operação (`encode`, `navigate`, `dream`) vira um node do graph
- Checkpointer time-series ganha uso nativo
- **Prós**: checkpointing real, rollback, state history, replay
- **Contras**: refactor massivo, IMI perde identidade arquitetural

### Caminho B: Adapter pattern — usar o saver como storage layer (refactor moderado)
- Manter `IMISpace` como está
- Substituir `VectorStore.save()/load()` por um adapter que grava em TimescaleDB/QuestDB
- Cada `MemoryNode` serializado vira um registro timestamped
- **Prós**: persistência robusta sem reescrever a lógica cognitiva
- **Contras**: não usa checkpointing do LangGraph (só o storage)

### Caminho C: Hybrid — IMI + LangGraph para orquestração multi-agente
- IMI continua como biblioteca de memória (o que ele é)
- Agentes que USAM IMI rodam como LangGraph workflows
- O checkpointer time-series persiste o estado dos agentes, não do IMI diretamente
- IMI persiste memórias; LangGraph persiste estado de execução
- **Prós**: separação limpa de responsabilidades, IMI inalterado
- **Contras**: requer infra LangGraph em volta

---

## Arquivos-chave do IMI para esta integração

| Arquivo | Papel | Impacto |
|---------|-------|---------|
| `imi/space.py` | Orquestrador + persistence logic (`save()`/`load()`) | **Alto** — ponto de entrada da mudança |
| `imi/store.py` | VectorStore com JSON save/load | **Alto** — candidato a substituição |
| `imi/node.py` | MemoryNode com `to_dict()`/`from_dict()` | **Médio** — serialização já existe |
| `imi/temporal.py` | TemporalIndex + TemporalContext | **Médio** — natural fit para time-series |
| `imi/maintain.py` | Dreaming (fade, cluster, consolidate) | **Baixo** — lógica não muda |
| `pyproject.toml` | Dependências | **Baixo** — adicionar langgraph-checkpoint-timeseries |

---

## Recomendação inicial

**Caminho C (Hybrid)** parece o mais alinhado:
- IMI é uma biblioteca de memória cognitiva — não precisa virar um graph
- O valor do time-series checkpointer está em persistir **estado de agentes** que usam IMI
- A camada temporal do IMI (`temporal.py`) já pensa em timestamps — um backend TSDB seria natural para queries temporais
- Kafka como write-ahead log para encodings de memória daria durabilidade sem bloquear

Mas vale avaliar se o **Caminho B** não resolve o problema imediato (JSON frágil) de forma mais simples.

---

## Perguntas para decidir na próxima sessão

1. O IMI vai ser usado por **múltiplos agentes simultâneos**? (Se sim, Caminho C é forte)
2. O problema principal é **fragilidade do JSON** ou **performance de escrita**?
3. Já existe infra TimescaleDB/QuestDB/Kafka rodando, ou precisa subir?
4. O IMI vai integrar com LangGraph em algum momento, ou permanece standalone?
5. O `TemporalIndex` do IMI se beneficiaria de queries SQL temporais nativas?

---

## Setup disponível (já instalado)

```bash
# Package
pip show langgraph-checkpoint-timeseries
# v0.2.0 — backends: TimescaleDB, QuestDB, Kafka

# Docker compose para infra local
# (do repo jersobh/langgraph-checkpoint-timeseries)
docker compose up -d
# TimescaleDB → localhost:5432
# QuestDB → localhost:9000 (UI) / 8812 (PG wire)
# Kafka → localhost:9092
# Kafka UI → localhost:8080
```

---

## Referências

- Repo: github.com/jersobh/langgraph-checkpoint-timeseries
- IMI source: `/Users/renatoaparegomes/experimentos/imi/`
- IMI docs: `/Users/renatoaparegomes/experimentos/imi/docs/`
- IMI análise prática: `/Users/renatoaparegomes/experimentos/imi/docs/analise-pratica.md`
- Demo completo: `/Users/renatoaparegomes/experimentos/imi/examples/demo_100_100.py`
