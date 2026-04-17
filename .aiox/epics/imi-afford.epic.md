# Epic: IMI-AFFORD — Retrieval por Utilidade Prospectiva

> **ID:** IMI-E01
> **Status:** Ready
> **Prioridade:** P1
> **Depende de:** nenhuma
> **Fundamentação científica:** MemRL (arXiv:2601.03192) + Memory Worth Metric (arXiv:2604.12007)
> **Problema resolvido:** `navigate_access` = 98 eventos vs `encode` = 4.300 — memória sem navegação é inerte

---

## Objetivo

Transformar o retrieval do IMI de busca por **similaridade semântica** para busca por **utilidade prospectiva**. Uma memória é útil não porque é semanticamente próxima à query, mas porque, quando usada no passado, produziu bons outcomes.

Implementar Two-Phase Retrieval (MemRL) + Memory Worth scoring (MW = success_count / total_uses).

## Contexto técnico

**Estado atual:**
- `im_nav` faz cosine similarity sobre todos os embeddings (O(N) scan)
- Nenhum feedback de outcome é capturado
- Affordances existem nos nós mas nunca são critério de retrieval primário

**Estado alvo:**
- Phase 1: filtro semântico (score >= 0.62) → candidate pool ~50 nós
- Phase 2: reranking por MW × affordance_confidence
- `im_mw_update(session_id, outcome)` captura feedback pós-sessão
- Boot considera MW como fator de scoring (não só salience)

## Stories

### S01: Schema Migration — success_count + failure_count
Adicionar colunas `success_count INT DEFAULT 0` e `failure_count INT DEFAULT 0` em `memory_nodes` (coluna no campo `data` JSON, sem ALTER TABLE — IMI usa JSON blob).

- Verificar se IMISpace suporta campos extras no `data` JSON sem quebrar serialização
- Criar script de migração que inicializa contadores para nós existentes
- **Verify:** `python3 -c "from imi.space import IMISpace; s = IMISpace.from_sqlite('imi_memory.db'); print('OK')"` sem erros

### S02: Tool im_mw_update no MCP Server
Novo tool `im_mw_update(node_ids: list[str], outcome: str)` onde `outcome` é `"success"` ou `"failure"`.

- Incrementa `success_count` ou `failure_count` no campo `data` JSON de cada nó listado
- Persiste via `space.save()`
- Retorna JSON com `{updated: N, mw_scores: {node_id: float}}`
- **Verify:** chamar `im_mw_update(["test_node"], "success")` → `im_nav` retorna score MW atualizado

### S03: Two-Phase Retrieval em im_nav
Adicionar parâmetro `mode: str = "semantic"` em `im_nav`. Quando `mode="utility"`:

- Phase 1: filtrar candidatos com score semântico >= MIN_SCORE (0.62)
- Phase 2: rerankar por `MW × affordance_max_confidence`
- MW = `success_count / max(1, success_count + failure_count)`
- Retornar campo `retrieval_mode` e `mw_score` em cada hit

- **Verify:** criar 3 nós com MW artificialmente alto/baixo → `im_nav(query, mode="utility")` deve rankar por MW, não por similaridade semântica pura

### S04: Integração no imi_delta_nav.py
Quando delta nav detecta entidade conhecida (entities list), usar `mode="utility"` ao invés de `mode="semantic"`.

- Raciocínio: entidades conhecidas têm histórico de uso → MW é mais informativo
- Queries abertas sem entidade: manter `mode="semantic"` (sem histórico)
- **Verify:** mensagem com "OXÉ" → delta nav retorna affordances rankeadas por MW

### S05: im_mw_update automático no GRAVAR
Adicionar ao protocolo GRAVAR (Canal 1) step final:

- Identificar os node_ids que foram consultados (`navigate_access` events desta sessão)
- Se sessão concluiu tasks planejadas: `im_mw_update(nodes_acessados, "success")`
- Se sessão abandonou tasks: `im_mw_update(nodes_consultados_sem_resultado, "failure")`
- **Verify:** após 3 sessões de sucesso, `im_nav(mode="utility")` deve priorizar nós com MW > 0.7

## Métricas de sucesso

- `navigate_access` events por semana: baseline 98 → alvo > 500 (delta nav aumenta uso)
- MW distribution: após 30 sessões, nós especializados devem ter MW > 0.7
- Retrieval precision: usuário não precisa corrigir retrieval mais de 1x por sessão

## Referências

- arXiv:2601.03192 — MemRL: two-phase retrieval
- arXiv:2604.12007 — Memory Worth Metric: MW = p+(m) correlação 0.89 com utilidade real

## Status de implementação (2026-04-17)

| Story | Status | Arquivo |
|---|---|---|
| S01 Schema MW counters | ✅ Done | mcp_server.py — seed JSON (schema-free) |
| S02 im_mw_update tool | ✅ Done | mcp_server.py:im_mw_update |
| S03 Two-Phase Retrieval em im_nav | ✅ Done | mcp_server.py:im_nav(mode="utility") |
| S04 Integração imi_delta_nav.py | ✅ Done | .claude/imi_delta_nav.py — use_utility_rerank |
| S05 im_mw_update no GRAVAR | ✅ Done | .claude/commands/gravar.md — Canal 1.4 |

**Verify smoke test:** im_mw_update schema-free PASS | im_nav mode=utility PASS | delta use_utility_rerank PASS
