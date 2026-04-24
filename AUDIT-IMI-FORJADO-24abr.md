# Auditoria Completa IMI — Plano Forjado (24/abr/2026)

**Gerado por:** Claude Opus 4.6 | **IC:** 0.91 | **Rota:** A (completa)
**DB:** 3.4GB | 549K rows | 47 memórias corrompidas (já limpas)

---

## Contexto

Bug original: `secure_encode()` encriptava texto ANTES de passar para `space.encode()`. Summarizers LLM recebiam ciphertext e geravam "conteúdo criptografado". Fix aplicado em `crypto_layer.py` — plaintext vai para summarizers, ciphertext só no `node.original` at-rest. Também fixado: `node.py to_dict()` agora persiste `original`.

47 memórias corrompidas marcadas `is_deleted=1`. Backup: `imi_memory.db.bak-pre-cleanup-20260424`.

---

## 36 Achados — Consolidado por Severidade

### TIER 1 — CRITICAL (Perda de dados / Corrupção ativa)

| # | Arquivo | Linhas | Descrição | Impacto |
|---|---------|--------|-----------|---------|
| C1 | `space.py` + `storage.py` | 559-562, 516-541 | **`save()` chama `put_nodes()` para TODOS os nodes a cada encode E navigate** — cada operação insere uma nova version row para os 549K nodes | DB 3.4GB e crescendo sem limite. Esta é a causa raiz do bloat |
| C2 | `storage.py` | Schema geral | **Versioning append-only sem GC** — zero VACUUM, zero max-version, zero prune de versões antigas. Não existe nenhuma utility de cleanup | Disco enche inevitavelmente |
| C3 | `storage.py` | 452-468 | **FTS cresce sem dedup** — `_fts_index_node` faz INSERT sem DELETE prévio. Cada save cria N rows duplicados no FTS | Busca FTS retorna duplicatas, FTS table possivelmente maior que a node table |
| C4 | `node.py` | 153-156 | **`from_dict()` muta o dict do caller via `pop()`** — qualquer iteração dupla sobre os dicts (migração, batch import) perde embedding, affect, temporal, affordances | Corrupção silenciosa em loops de import/export |

### TIER 2 — HIGH (Bugs funcionais com impacto real)

| # | Arquivo | Linhas | Descrição |
|---|---------|--------|-----------|
| H1 | `space.py` | 378, 401 | **`positional_reorder()` chamado 2x em `navigate()`** — segunda aplicação desfaz a otimização primacy-recency |
| H2 | `mcp_server.py` | 700-717 | **`im_mw_update` sobrescreve `node.seed`** com JSON puro, destruindo a chave de reconstrução LLM. Qualquer memória que recebe MW update perde o seed |
| H3 | `mcp_server.py` | 357 | **`im_glnk` guarda `if space.persist_dir`** mas deploy SQLite não seta persist_dir → graph edges nunca persistem |
| H4 | `graph.py` | 332-351 | **Edges bidirecionais viram unidirecionais após save/load** — `to_dict()` deduplica mas `from_dict()` não reconstrói a direção reversa |
| H5 | `affect.py` | 31, 81-94 | **`_base_salience` não persistido** — baseline recalcula do valor atual a cada load, causando drift de salience para cima (infla salience de nodes muito acessados) |
| H6 | `maintain.py` | 281-291 | **Prune loga candidatos mas nunca remove** — `episodic.remove()` ausente. Episodic store nunca encolhe |
| H7 | `maintain.py` | 217-228 | **Patterns consolidados perdem affect/mass/affordances** — tiered para o fundo sempre, independente da importância dos episódios fonte |
| H8 | `anchors.py` | 67-69 | **`Anchor.from_dict()` muta dict in-place** (coerce string→enum) — pode crashar json.dumps downstream |
| H9 | `storage.py` | 589-597 | **`put_anchors` faz DELETE FROM anchors total** antes de re-insert — crash window perde todos os anchors |
| H10 | `storage.py` | 757-768 | **`get_node_history` retorna tombstones (is_deleted)** como nodes vazios |
| H11 | `space.py` | 594-628 | **`from_backend()` não chama `backend.setup()`** — crasharia em DB novo |
| H12 | `space.py` | 553-591 | **`save()` perde `reconsolidation_log` e `annealing` state** — histórico irrecuperável |
| H13 | `mcp_server.py` | 26-30, storage 396 | **Zero locking** em `_get_space()` e SQLite writes — race condition sob concorrência |

### TIER 3 — MEDIUM (Tech debt / Riscos latentes)

| # | Arquivo | Descrição |
|---|---------|-----------|
| M1 | `fcm_bridge.py` L139-141 | FCM emite `imi_seed` plaintext junto com `original` encriptado — leak parcial de conteúdo |
| M2 | `fcm_security.py` L147 | FCM emite ciphertext de `node.original` como conteúdo de evento |
| M3 | `crypto_layer.py` L120-129 | Ephemeral key sem persistência — restart perde acesso a todas memórias encriptadas |
| M4 | `fcm_bridge.py` L68 | `_consumed_ids` in-memory only — eventos reprocessados após restart |
| M5 | `fcm_security.py` L110 | `time.monotonic()` reseta no restart — dedup window quebra cross-restart |
| M6 | `immune_bridge.py` L380 | Hook queue truncada antes do processing completar — crash perde eventos |
| M7 | `sanitizer_wrapper.py` L69 | `_RE_PROC` compilado mas nunca usado — números de processo vazam sem redação |
| M8 | `mcp_server.py` L465-489 | `im_int_fulfill` não atualiza in-memory store — `im_int_list` vê status stale |
| M9 | `storage.py` L398 | Sem `PRAGMA busy_timeout` — write contention falha imediatamente |
| M10 | `maintain.py` L119-148 | Clustering greedy single-linkage encadeia memórias não-relacionadas |
| M11 | `space.py` L544-547 | `refresh_tiers()` após `save()` — tier updates não persistem no mesmo dream cycle |
| M12 | `langchain.py` L83-93 | `from_sqlite()` pode cair em JSON backend se diretório homônimo existir |
| M13 | `node.py` L158 | Keys desconhecidas silenciosamente descartadas — zero path de migração de schema |
| M14 | `crypto_layer.py` L313 | CLI test usa assignment local em vez de `global` — crypto path nunca testado |

### TIER 4 — LOW (Cosmético / Confuso mas inofensivo)

| # | Descrição |
|---|-----------|
| L1 | `temporal.py` from_dict usa `**d` sem field guard — TypeError em keys futuras |
| L2 | `affordance.py` from_dict idem |
| L3 | `space.py` import duplicado de `positional_reorder` |
| L4 | `maintain.py` `fade()` conta mas não modifica nada — nome enganoso |
| L5 | `mcp_server.py` `im_perf` lê `_space` direto sem `_get_space()` |

---

## Análise AGORA — Achados do Mapeamento

### Rotações Semânticas (6)
1. **Backward compatibility / migration path** — 549K nodes no schema antigo sem migration script
2. **Rollback strategy** — Backup existe mas não está no plano formal de compactação
3. **MCP server restart** — Mudanças no código fonte só efetivas após restart
4. **Test harness** — Deps faltando no venv, testes não rodam
5. **Ordem de persistência** — C1 deve vir antes de C2 (parar sangramento antes de limpar)
6. **Observabilidade pós-fix** — Sem métricas de DB size, FTS rows, versions per node

### Subtrações (3)
1. "DB 200-400MB" é otimista — floor realista ~800MB-1.2GB (embeddings + JSON + overhead)
2. "Fases independentes" é FALSO — C1↔H2, C1→C2→C3 têm acoplamento real
3. M3 como MEDIUM é subestimado se config mudar (bomba-relógio)

### Silêncios Críticos (3)
1. Ausência de testes — maior risco do plano
2. Ausência de rollback — DB de 3.4GB sem estratégia
3. Ausência de restart plan — código muda mas servidor não reinicia

### Grafo de Dependências
```
C1 (dirty tracking) ─────┐
                          ├── H2 (seed protection) depende de save() estável
C2 (compactação)  ◄───── C1 deve vir ANTES
C3 (FTS dedup)    ◄───── C2 (compactar nodes antes, depois limpar FTS)
C4 (from_dict)    ──────── independente
H1 (positional)   ──────── independente
H2 (seed/MW)      ◄───── C1 (save não pode re-persistir seed corrompido)
H3 (im_glnk)      ──────── independente
H4 (graph bidi)   ──────── independente, mas melhor com H3
H5 (base_salience)──────── independente
H6 (prune)        ◄───── C1 (prune precisa de save funcional)
H7 (pattern meta) ──────── independente
```

### Têmpera — 7 Ataques (todos legítimos, todos incorporados)
1. Gates entre sub-steps da Fase 1
2. Estimativa de DB size corrigida (800MB-1.2GB floor)
3. Compactação com MCP server parado
4. Smoke test pós-fase
5. `mw_data` com fallback no from_dict
6. Documentar compact como destrutivo para histórico
7. Schema MemoryNode v2 como passo atômico

---

## PLANO DE EXECUÇÃO FORJADO

### Fase 0 — Fundação (antes de qualquer fix)
1. Confirmar backup: `imi_memory.db.bak-pre-cleanup-20260424` (3.4GB) existe
2. Criar DB de teste: extrair ~1K nodes para `imi_test.db`
3. Instalar deps de teste no venv do IMI
4. Calcular floor realista do DB size: `SELECT count(DISTINCT node_id), avg(length(data)), avg(length(embedding)) FROM memory_nodes`
5. Documentar procedimento de restart MCP

### Fase 1 — Parar o sangramento
**Gate: Fase 0 completa**

**1a. C1 — Dirty tracking** em `VectorStore` + `save()` só persiste dirty nodes
- Adicionar `_dirty_nodes: set[str]` ao VectorStore
- `add()` marca node como dirty
- `save()` chama `put_node()` só para dirty nodes, depois limpa o set
- Remover `put_nodes()` batch do `save()` quando backend está ativo

**1b. Deploy + restart MCP** → smoke test:
- `im_enc` de texto PT → verificar summary legível
- `im_nav` busca → verificar resultados
- `SELECT count(*) FROM memory_nodes` não cresce desproporcionalmente

**Gate: C1 validado? Sim → próximo. Não → parar e diagnosticar.**

**1c. C2 — Compactação** (MCP server PARADO)
- `compact_versions(keep_versions=1)`:
  ```sql
  DELETE FROM memory_nodes WHERE rowid NOT IN (
    SELECT rowid FROM memory_nodes mn2
    WHERE mn2.node_id = memory_nodes.node_id
    AND mn2.store_name = memory_nodes.store_name
    ORDER BY version DESC LIMIT 1
  )
  ```
- `VACUUM`
- Restart MCP + smoke test

**1d. C3 — FTS dedup**
- Adicionar `DELETE FROM memory_fts WHERE node_id=? AND store_name=?` antes de INSERT em `_fts_index_node()`
- Rebuild FTS: `DELETE FROM memory_fts; re-index all current nodes`

**1e. Validação Fase 1:**
- `SELECT count(*) FROM memory_nodes` — deve ser ≈ count(DISTINCT node_id)
- `SELECT count(*) FROM memory_fts` — deve ser ≈ count de nodes
- DB size: `ls -lh imi_memory.db` — esperado 800MB-1.2GB

### Fase 2 — Corrigir corrupção ativa
**Gate: Fase 1 validada**

**2a. Schema MemoryNode v2** (passo atômico):
- `node.py`: adicionar `mw_data: dict | None = None`
- `node.py`: `to_dict()` inclui `mw_data` e `base_salience` (de `_base_salience`)
- `node.py`: `from_dict()` usa `d.copy()` antes dos `pop()` (C4)
- `affect.py`: `to_dict()` inclui `base_salience`; `from_dict()` restaura
- `temporal.py`, `affordance.py`, `anchors.py`: field guard nos `from_dict()` (L1, L2, H8)

**2b. Fixes isolados:**
- H1: remover segundo `positional_reorder()` em `space.py:401`
- H2: `im_mw_update` usa `node.mw_data` em vez de sobrescrever `node.seed`
- H3: `im_glnk` fix guard: `if space.persist_dir or space.backend: space.save()`
- H4: `graph.py from_dict()` reconstrói edges bidirecionais

**2c. Validação Fase 2:**
- Roundtrip test: `to_dict(from_dict(to_dict(node)))` para todos os tipos
- `im_enc` texto PT → `im_nav` → summary legível
- `im_glnk` → restart MCP → verificar que edge persiste

### Fase 3 — Sistema de manutenção
**Gate: Fase 2 validada**

- H5: persistir `_base_salience` (já em Schema v2)
- H6: implementar `episodic.remove()` no prune loop do `maintain.py`
- H7: patterns consolidados herdam affect/mass/affordances agregados
- H12: persistir `reconsolidation_log` em save()
- H9: `put_anchors` usa `INSERT OR REPLACE` em vez de DELETE-all
- H10: `get_node_history` filtra `is_deleted=1`
- H11: `from_backend` chama `setup()`
- Mover `refresh_tiers()` antes de `save()` em `dream()`

**Validação:** executar `im_drm` no DB de teste, verificar patterns com affect>0

### Fase 4 — Hardening
**Gate: Fase 3 validada**

- H13: threading.RLock em `_get_space()` + M9 busy_timeout
- M1+M2: FCM strip seed quando crypto ativo
- M3: check no boot — se `IMI_CRYPTO=1` sem key → refuse to start
- M4+M5: persistir consumed_ids; usar time.time()
- M6: atomic drain do hook queue
- M7: ativar `_RE_PROC` no sanitizer
- L3: remover import duplicado
- L4: renomear `fade()` para `count_fadeable()`

### Fase 5 — Monitoramento
- `im_perf` reporta: `versions_per_node`, `fts_row_count`, `db_size_mb`
- Alerta: `versions_per_node > 2` → dirty tracking falhou
- Documentar: `compact()` é destrutivo para histórico de versões

---

## Síntese da Forja

- **Input:** Plano de ação de 4 fases para 36 bugs encontrados na auditoria IMI
- **Rota:** A (completa) — complexo + operacional + fortalecer
- **Etapa 1 — Mapeamento:** 6 rotações, 3 subtrações, 6 silêncios críticos
- **Etapa 2 — Refinamento:** 3 passes, 12 operações, 6/6 achados AGORA integrados
- **Etapa 3 — Endurecimento:** 7 ataques, IC 0.91, 1 alucinação excisada
- **Ponto de maior impacto:** Reordenação de C1 antes de C2 + gates entre sub-steps
- **Tensões remanescentes:** Test harness insuficiente; tamanho do plano vs contexto de sessão
- **Confiança:** analítica (validação em campo pendente)

---

## Fixes Já Aplicados Nesta Sessão

1. **crypto_layer.py** — `secure_encode()` agora passa plaintext para `space.encode()`, encripta só `node.original` depois
2. **node.py** — `to_dict()` agora serializa `original`
3. **Cleanup DB** — 2.591 rows corrompidos marcados `is_deleted=1`
4. **Backup** — `imi_memory.db.bak-pre-cleanup-20260424` (3.4GB)
