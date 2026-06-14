# ADR 0002 — H-MEM rollout: runbook PROMOTE+SHADOW (2 semanas) + decisão de flip

> **Status:** Ativo (rollout iniciado 2026-06-14) · **Decisor:** Renato
> **Depende de:** ADR 0001 (index-as-hit + gate de paridade)
> **Commits:** daemon PROMOTE + MCP SHADOW + `scripts/hmem_rollout_daily.sh` + cron

## O que está ligado (e o que NÃO está)

Este NÃO é um canário de deploy web. O IMI é UM processo MCP + SQLite + cron — não há
load balancer, traffic split, nem P99 sob carga. O "canário de 2 semanas" mapeia para:

| Componente | Onde | O que faz | Reversível? |
|---|---|---|---|
| **SHADOW** | `~/.mcp.json` env do imi-memory (`IMI_HMEM_SHADOW=1`) | cada `im_nav` roda hierárquico ao lado do flat e loga divergência; **serve flat** | ✅ flag off + restart MCP |
| **PROMOTE** | daemon `imi_dream_daemon.sh` (`export IMI_HMEM_PROMOTE=1`) | consolidação grava árvore (layer/child_ptrs) no **DB real** | ⚠️ flag off; dados via snapshot |
| **Canário diário** | cron `30 22 * * *` → `hmem_rollout_daily.sh` | snapshot datado → gate de saúde → série temporal → auto-desliga PROMOTE se drift | ✅ observação |

**Default permanece `mode=semantic` (flat).** O hierárquico só é servido se o usuário pedir
`mode="hierarchical"` explicitamente. O flip do default é decisão HUMANA pós-2-semanas.

## Para ativar (já feito 2026-06-14)

1. `IMI_HMEM_SHADOW=1` no env do imi-memory em `~/.mcp.json` → **reiniciar o Claude Code**
   (o MCP só lê env no restart). Sem restart, SHADOW não roda.
2. `export IMI_HMEM_PROMOTE=1` no daemon → ativo no próximo dream (06/13/21h).
3. Cron diário `30 22` instalado → gate + snapshot + métricas.

## Decisão de flip (após 14 dias) — critério MENSURÁVEL

Inspecionar `~/.imi/hmem_rollout_metrics.jsonl` (uma linha/dia). Promover default
flat→hierárquico SOMENTE se, nos 14 dias:

- ✅ `canary_status == "ok"` em **todos** os dias (100% — zero drift silencioso)
- ✅ `parity_ok == true` em **todos** os dias (`recall_hier ≥ recall_flat − 0.02`)
- ✅ `index_nodes` crescendo (a árvore está sendo populada, não estagnada)
- ✅ divergência shadow (`~/.imi/hmem_shadow.jsonl` via `hmem_shadow.summarize`) estável < ~5%

Se TUDO acima: flip = adicionar `IMI_HMEM_RETRIEVAL=1` ao MCP + mudar o default de `im_nav`
para `hierarchical` (decisão sua, com a série em mãos). Senão: investigar, não promover.

## Rollback (instantâneo)

**Parar a mutação (PROMOTE):**
```bash
# comentar a linha no daemon (o gate diário faz isto sozinho em caso de drift):
sed -i.bak 's/^export IMI_HMEM_PROMOTE=1/# export IMI_HMEM_PROMOTE=1/' \
  ~/experimentos/tools/imi/scripts/imi_dream_daemon.sh
```

**Parar a observação (SHADOW):** remover `"IMI_HMEM_SHADOW": "1"` de `~/.mcp.json` + restart.

**Restaurar dados (se a árvore corromper algo):** com o MCP PARADO (G5 — nunca com processo vivo):
```bash
ls -t ~/experimentos/tools/imi/.hmem_rollout_snapshots/   # snapshot mais recente
cp ~/experimentos/tools/imi/.hmem_rollout_snapshots/imi_memory.<STAMP>.db \
   ~/experimentos/tools/imi/imi_memory.db
# snapshot pre-rollout absoluto: imi_memory.pre-hmem-20260614.db
```

## Por que é seguro mutar o DB real

- **Forward-compatible:** o schema só serializa layer/child_ptrs se promovido — nós legados
  (layer=3) ficam byte-idênticos. (ADR 0001, test_hmem_schema)
- **Acíclico por construção:** só arestas índice→membro; membros nunca ganham child_ptrs.
- **Dirty-sink provado:** test_promotion_survives_save_and_reload (round-trip SQLite real).
- **Snapshot diário** + snapshot absoluto pre-hmem. Janela de 14 snapshots retida.
- **Auto-halt:** o gate diário desliga PROMOTE se o canário acusar qualquer miss.
- **Default intocado:** o usuário continua recebendo retrieval flat até o flip humano.

## Monitoramento

- `tail -f ~/.claude/hmem_rollout.log` — log do gate diário
- `cat ~/.imi/hmem_rollout_metrics.jsonl` — série temporal (a evidência do flip)
- `python3 -c "from imi.hmem_shadow import summarize; print(summarize())"` — divergência shadow
