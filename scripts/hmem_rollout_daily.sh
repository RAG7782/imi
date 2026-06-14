#!/usr/bin/env bash
# hmem_rollout_daily.sh — daily health gate for the H-MEM PROMOTE+SHADOW rollout.
# =============================================================================
# Spec: ~/experimentos/specs/2026-06-14-imi-hmem-positional-index.md (§4.5 step 7)
# ADR:  ~/experimentos/tools/imi/docs/adr/0001-hmem-index-as-hit-and-parity-gate.md
#
# PORQUÊ este script (e por que NÃO é um canário de deploy web genérico):
#   O IMI não é um serviço com load balancer — é UM processo MCP + SQLite + cron.
#   Então "canário de 2 semanas" aqui significa, mecanicamente:
#     - PROMOTE (mutação): o dream daemon, com IMI_HMEM_PROMOTE=1, cresce a árvore
#       no DB real durante a consolidação. Esta é a ÚNICA parte irreversível.
#     - SHADOW (observação): o MCP, com IMI_HMEM_SHADOW=1, loga divergência por
#       chamada — mas SEMPRE serve flat. Zero risco ao que o usuário recebe.
#     - Canário diário (este script): snapshot → roda o gate de saúde → registra a
#       série temporal → SE o canário quebrar, DESLIGA PROMOTE sozinho (safety).
#
# Fluxo numerado:
#   1. lock (uma execução por vez)
#   2. snapshot datado do DB real ANTES de qualquer leitura de estado mutável
#   3. roda o gate diário (canário lexical + paridade + summarize) em modo read-only
#   4. anexa métricas à série temporal JSONL (a evidência da decisão de flip)
#   5. SE canário != ok → desliga PROMOTE (remove flag do daemon) e grita no log
#
# Rollback instantâneo (manual): ver runbook em docs/adr/0001 ou:
#   - parar mutação: remover IMI_HMEM_PROMOTE do env do daemon
#   - restaurar dados: cp do snapshot mais recente sobre imi_memory.db (MCP parado)
#
# Cron sugerido (1x/dia, após o último dream das 21h):
#   30 22 * * * bash ~/experimentos/tools/imi/scripts/hmem_rollout_daily.sh >> ~/.claude/hmem_rollout.log 2>&1

set -euo pipefail

IMI_DIR="$HOME/experimentos/tools/imi"
PY="$IMI_DIR/.venv/bin/python"
# .venv não tem pytest mas tem o pacote imi; para o gate usamos python3 do sistema
# que TEM as deps de runtime. Preferir o venv se ele importar imi; senão python3.
LIVE_DB="$IMI_DIR/imi_memory.db"
SNAP_DIR="$IMI_DIR/.hmem_rollout_snapshots"
METRICS="$HOME/.imi/hmem_rollout_metrics.jsonl"
LOG_PREFIX="[hmem-rollout]"
LOCK="/tmp/hmem_rollout.lock"
DAEMON="$IMI_DIR/scripts/imi_dream_daemon.sh"

log() { echo "$LOG_PREFIX $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

# ── 1. lock ──────────────────────────────────────────────────────────────────
if [ -e "$LOCK" ]; then
    log "lock presente ($LOCK) — outra execução em curso, abortando"
    exit 0
fi
trap 'rm -f "$LOCK"' EXIT
echo "$$" > "$LOCK"

# Escolher interpretador que importa `imi`
PYBIN="python3"
if "$PY" -c "import imi" 2>/dev/null; then PYBIN="$PY"; fi
log "interpretador: $PYBIN"

# ── 2. snapshot datado ANTES de tudo (G5: dado real só se toca com backup) ────
mkdir -p "$SNAP_DIR"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
SNAP="$SNAP_DIR/imi_memory.$STAMP.db"
if [ ! -f "$LIVE_DB" ]; then
    log "ERRO: DB real não encontrado em $LIVE_DB — abortando"
    exit 1
fi
cp "$LIVE_DB" "$SNAP"
log "snapshot: $SNAP ($(du -h "$SNAP" | cut -f1))"
# Reter só os últimos 14 snapshots (janela do rollout)
ls -1t "$SNAP_DIR"/imi_memory.*.db 2>/dev/null | tail -n +15 | xargs -r rm -f

# ── 3. gate diário (read-only sobre o DB real) ───────────────────────────────
# Roda canário lexical (sobrevive a Ollama cair) + paridade hier-vs-flat + summarize.
# NÃO muta — só mede. A mutação (crescer a árvore) é o daemon que faz, no horário dele.
cd "$IMI_DIR"
GATE_JSON="$("$PYBIN" - "$LIVE_DB" <<'PY'
import json, sys, time
from imi.space import IMISpace
from imi.canary import load_anchors, run_canary
from imi.hmem_retrieve import recursive_retrieve

db = sys.argv[1]
sp = IMISpace.from_sqlite(db)
anchors = load_anchors()

# Canário lexical (não toca embedder — sobrevive a Ollama cair)
c = run_canary(sp, anchors, top_k=10)

# Estado da árvore no DB real
idx_nodes = sum(1 for n in sp.semantic.nodes if n.layer < 3 and n.child_ptrs)
reparented = sum(1 for n in sp.episodic.nodes if n.parent_id)

# Paridade hier-vs-flat sobre os anchors (só se houver árvore)
def flat_ids(a):
    q = sp.embedder.embed(a.note or a.token)
    ep = sp.episodic.search(q, top_k=10, relevance_weight=0.1)
    se = sp.semantic.search(q, top_k=5, relevance_weight=0.1)
    m = sorted(ep + se, key=lambda x: x[1], reverse=True)[:10]
    return [n.id[:12] for n, _ in m]

def hier_ids(a):
    q = sp.embedder.embed(a.note or a.token)
    return [h.node.id[:12] for h in
            recursive_retrieve(q, [sp.episodic, sp.semantic], k_final=10).hits]

rec_h = rec_f = 0
try:
    for a in anchors:
        if a.expected_id in hier_ids(a): rec_h += 1
        if a.expected_id in flat_ids(a): rec_f += 1
    n = len(anchors)
    recall_hier, recall_flat = rec_h / n, rec_f / n
    parity_ok = recall_hier >= recall_flat - 0.02
except Exception as e:  # embedder/Ollama down — paridade indisponível, canário ainda vale
    recall_hier = recall_flat = None
    parity_ok = None
    sys.stderr.write(f"parity unavailable (embedder?): {type(e).__name__}: {e}\n")

print(json.dumps({
    "ts": time.time(),
    "canary_status": c.status,
    "canary_hits": c.hits,
    "canary_total": c.total,
    "index_nodes": idx_nodes,
    "reparented": reparented,
    "recall_hier": recall_hier,
    "recall_flat": recall_flat,
    "parity_ok": parity_ok,
}))
PY
)"
log "gate: $GATE_JSON"

# ── 4. anexar à série temporal ───────────────────────────────────────────────
mkdir -p "$(dirname "$METRICS")"
echo "$GATE_JSON" >> "$METRICS"

# ── 5. safety: se o canário quebrou, DESLIGAR PROMOTE ─────────────────────────
CANARY_STATUS="$(echo "$GATE_JSON" | "$PYBIN" -c 'import sys,json; print(json.load(sys.stdin)["canary_status"])')"
if [ "$CANARY_STATUS" != "ok" ]; then
    log "⚠️ CANÁRIO != ok ($CANARY_STATUS) — DESLIGANDO PROMOTE por segurança"
    # Remove a flag do env do daemon (best-effort): comenta a linha de export
    if grep -q "IMI_HMEM_PROMOTE" "$DAEMON" 2>/dev/null; then
        sed -i.bak 's/^export IMI_HMEM_PROMOTE=1/# AUTO-DISABLED (canary drift): export IMI_HMEM_PROMOTE=1/' "$DAEMON"
        log "PROMOTE desligado no daemon ($DAEMON). Restaurar snapshot: $SNAP"
    fi
    log "AÇÃO HUMANA: investigar drift; restore: cp $SNAP $LIVE_DB (com MCP parado)"
    exit 1
fi

SUMMARY="$(echo "$GATE_JSON" | "$PYBIN" -c 'import sys,json; d=json.load(sys.stdin); print("idx=%s reparented=%s parity_ok=%s" % (d["index_nodes"], d["reparented"], d["parity_ok"]))')"
log "OK — canário $CANARY_STATUS, $SUMMARY"
log "métrica anexada a $METRICS (série temporal de 14 dias para a decisão de flip)"
