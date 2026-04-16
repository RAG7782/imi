# Epic: IMI-PATTERN — Promoção de Padrões Semânticos no Boot

> **ID:** IMI-E02
> **Status:** Ready
> **Prioridade:** P1
> **Depende de:** nenhuma (independente do IMI-E01)
> **Fundamentação científica:** GAM (arXiv:2604.12285) + Generative Agents reflection mechanism (arXiv:2304.03442)
> **Problema resolvido:** 57.253 padrões semânticos com avg_access_count=1.7 — conhecimento consolidado não é usado

---

## Objetivo

O `im_drm` consolida 40-85 episódios em padrões semânticos — sínteses de alto valor. Porém esses padrões têm `mass` frequentemente menor que episódios recentes individuais, então são preteridos no boot. O resultado: o destilado do seu conhecimento fica dormindo.

Implementar **Positional Reorder v3** com scoring diferenciado por store-type + slots explícitos para padrões semânticos no boot.

## Contexto técnico

**Estado atual (`imi_boot_semantic.py`):**
```python
# Score igual para episodic e semantic
score = salience × fade_resist × recency_weight
# Boot: top-7 por score, aplicar positional_reorder genérico
```

**Estado alvo:**
```python
# Score diferenciado por store
if store == 'semantic':
    episode_count = extraído de "consolidated from N episodes"
    boost = min(2.0, log2(episode_count))  # log2(85) ≈ 6.4 → cap em 2.0
    score = salience × fade_resist × recency_weight × boost
else:
    score = salience × fade_resist × recency_weight

# Positional Reorder v3: slots explícitos
# Slot 0-1: top-2 episódios recentes (hot context — o que acabou de acontecer)
# Slot 2-3: top-2 padrões semânticos (knowledge distilled — o que você aprendeu)
# Slot 4-5: top-2 por score misto (fronteiras = máx atenção LLM, Liu 2023)
# Slot 6:   intenções pendentes (IMI-E04, se implementado)
```

## Stories

### S01: Parser de episode_count de padrões semânticos
Extrair N de strings como `"consolidated from 40 episodes"` no campo `summary_orbital` ou `source`.

- Função `parse_episode_count(data: dict) -> int`
- Fallback para 1 se não encontrado
- **Verify:** `parse_episode_count({"source": "consolidated from 85 episodes"})` → `85`

### S02: Scoring diferenciado por store em fetch_top_memories()
Modificar `composite_score()` em `imi_boot_semantic.py` para aceitar `store_name` e `episode_count`.

- Boost formula: `min(2.0, math.log2(max(2, episode_count)))`
- Log do boost aplicado em `imi_boot.log`: `[boot] semantic boost: node_id score_before → score_after (N episodes)`
- Parâmetro `SEMANTIC_BOOST_ENABLED = True` (env var para disable em debug)
- **Verify:** nó semântico com 85 episódios deve ter score > nó episódico recente de salience similar

### S03: Positional Reorder v3 com slots explícitos
Refatorar `positional_reorder()` para aceitar `typed_memories: list[tuple[str, dict]]` onde o primeiro elemento é o store_name.

```python
def positional_reorder_v3(memories: list[dict]) -> list[dict]:
    episodic = [m for m in memories if m.get('store') != 'semantic'][:2]
    semantic = [m for m in memories if m.get('store') == 'semantic'][:2]
    mixed    = [m for m in memories if m not in episodic + semantic][:3]
    # Liu 2023: bordas = máx atenção. Episodic recente no topo, semantic no fim
    return episodic + mixed + semantic
```

- Atualizar campo `strategy` no bloco boot: `"semantic_v3"`
- Logar composição: `[boot] v3 composition: 2 episodic + 2 semantic + 3 mixed`
- **Verify:** bloco `<imi_boot>` deve conter pelo menos 1 memória com `[PATTERN]` nos primeiros 7 slots

### S04: Seção dedicada no bloco <imi_boot>
Separar visualmente episódios e padrões no output do boot:

```
## Memórias Recentes (episodic)
  1. [CP] ...
  2. [CP] ...

## Conhecimento Consolidado (semantic patterns)
  3. [LP] [PATTERN] consolidated from 85 episodes — ...
  4. [LP] [PATTERN] consolidated from 40 episodes — ...

## Contexto Misto
  5-7. ...
```

- **Verify:** output do boot inclui seção "Conhecimento Consolidado" com pelo menos 1 padrão quando `semantic_count > 0`

### S05: Métrica de semantic_ratio no log
Logar `semantic_ratio = semantic_slots / total_slots` após cada rebuild de cache.

- Alerta se `semantic_ratio == 0.0` por 3 rebuilds consecutivos (padrões estão sendo preteridos)
- **Verify:** `grep "semantic_ratio" ~/.claude/imi_boot.log` mostra valores > 0.0 após rebuild

## Métricas de sucesso

- `semantic_ratio` no boot: baseline ~0% → alvo ≥ 28% (2/7 slots)
- `access_count` médio de semantic nodes: baseline 1.7 → alvo > 5 após 30 sessões
- Redução em "não sabia que já havia aprendido X" — qualitativo, verificar em retrospectiva

### S06 (Research): Salience dinâmica para padrões semânticos
> **Registrado em sessão 2026-04-16 — pesquisar antes de implementar**

O `im_drm` grava padrões com salience estática (default 0.5). Isso causa o mesmo problema
dos slots de atenção: padrões valiosos não emergem naturalmente por score.

Hipótese: implementar *salience promotion* incremental:
- Cada vez que `im_nav` retorna um padrão → `access_count += 1`
- Cada vez que o padrão é linkado via `im_glnk` → `citation_count += 1`
- `salience = min(0.95, 0.5 + 0.05 * log2(access_count + citation_count + 1))`

Isso eliminaria a necessidade de threshold diferenciado e slots garantidos no boot —
padrões valiosos emergiriam por uso real (RL signal).

**Referência:** MemoryWorth (arXiv:2604.12007) — 2 contadores, correlação 0.89 com utilidade.

**Prerequisito:** S06 deve ser pesquisado antes de implementar — avaliar se promoção
incrementa ao longo de sessões (persistência no SQLite) ou só in-session.

## Métricas de sucesso

- `semantic_ratio` no boot: baseline ~0% → alvo ≥ 28% (2/7 slots) ✅ **ATINGIDO** (29%)
- `access_count` médio de semantic nodes: baseline 1.7 → alvo > 5 após 30 sessões
- Redução em "não sabia que já havia aprendido X" — qualitativo, verificar em retrospectiva

## Status de implementação

| Story | Status |
|---|---|
| S01 parse_episode_count | ✅ Done (2026-04-16) |
| S02 boosted_score | ✅ Done (2026-04-16) |
| S03 positional_reorder_v3 | ✅ Done (2026-04-16) |
| S04 seção semântica no output | ✅ Done (2026-04-16) |
| S05 semantic_ratio log | ✅ Done (2026-04-16) |
| S06 salience dinâmica | Research pendente |

## Referências

- arXiv:2604.12285 — GAM: dual graph com gating semântico
- arXiv:2304.03442 — Generative Agents: reflection mechanism (predecessor do im_drm)
- Liu et al. 2023 — Lost in the Middle: positional reorder para máxima atenção LLM
- arXiv:2604.12007 — MemoryWorth: salience dinâmica via contadores de acesso/citação
