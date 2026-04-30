#!/usr/bin/env bash
# imi_dream_daemon.sh — IMI Dream Consolidation Daemon
# =====================================================
# Roda im_drm (consolidação) via MCP direto ao Python.
# Verifica convergência — se não convergiu, re-executa (máx 3x).
# Log em ~/.claude/imi_dream.log
#
# Cron recomendado (3x/dia):
#   0 6,13,21 * * * bash ~/experimentos/tools/imi/scripts/imi_dream_daemon.sh
#
# Uso manual:
#   bash imi_dream_daemon.sh [--force]   # --force ignora lock

set -euo pipefail

LOG_FILE="$HOME/.claude/imi_dream.log"
LOCK_FILE="/tmp/imi_dream.lock"
IMI_DIR="$HOME/experimentos/tools/imi"
VENV_PYTHON="$IMI_DIR/.venv/bin/python"
MAX_ROUNDS=3
CONVERGE_TIMEOUT=300  # 5 min por round

# ── Funções de log ────────────────────────────────────────────────────────────

log() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[$ts] $*" | tee -a "$LOG_FILE"
}

log_json() {
    # Salva evento estruturado para métricas
    local event_file="$HOME/.claude/imi_dream_events.jsonl"
    echo "$1" >> "$event_file" 2>/dev/null || true
}

# ── Lock guard (evita múltiplas instâncias simultâneas) ───────────────────────

if [[ "${1:-}" != "--force" ]]; then
    if [[ -f "$LOCK_FILE" ]]; then
        # Verifica se o processo ainda está rodando
        pid="$(cat "$LOCK_FILE" 2>/dev/null || echo "")"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log "SKIP: dream já em andamento (PID=$pid)"
            exit 0
        fi
        # Lock stale — remove
        rm -f "$LOCK_FILE"
    fi
fi

echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# ── Seleciona Python ──────────────────────────────────────────────────────────

if [[ -f "$VENV_PYTHON" ]]; then
    PYTHON="$VENV_PYTHON"
elif command -v uv &>/dev/null; then
    PYTHON="uv run python"
else
    PYTHON="python3"
fi

# ── Script Python inline para executar dream + verificar convergência ─────────

DREAM_SCRIPT=$(cat <<'PYEOF'
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "experimentos/tools/imi"))

from imi.space import IMISpace

db_path = os.environ.get("IMI_DB", str(Path.home() / "experimentos/tools/imi/imi_memory.db"))

print(f"[dream] Opening DB: {db_path}", flush=True)
space = IMISpace.from_sqlite(db_path)

ep_before = len(space.episodic)
sem_before = len(space.semantic)
print(f"[dream] Before: episodic={ep_before} semantic={sem_before}", flush=True)

t0 = time.monotonic()
report = space.dream()
elapsed = time.monotonic() - t0

converged = space.annealing.converged
energy = round(space.annealing.energy_history[-1], 4) if space.annealing.energy_history else None
steps = space.annealing.iteration

result = {
    "nodes_processed": report.nodes_processed,
    "clusters_formed": report.clusters_formed,
    "patterns_extracted": report.patterns_extracted,
    "episodic_before": ep_before,
    "episodic_after": len(space.episodic),
    "semantic_before": sem_before,
    "semantic_after": len(space.semantic),
    "converged": converged,
    "energy": energy,
    "annealing_steps": steps,
    "elapsed_seconds": round(elapsed, 2),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

print(f"[dream] Result: {json.dumps(result)}", flush=True)

# Exit code encodes convergence: 0=converged, 2=not-yet
sys.exit(0 if converged else 2)
PYEOF
)

# ── Loop de consolidação com verificação de convergência ──────────────────────

log "IMI Dream Daemon iniciado"
log "DB: $IMI_DIR/imi_memory.db"

total_patterns=0
converged=false

for round in $(seq 1 $MAX_ROUNDS); do
    log "Round $round/$MAX_ROUNDS iniciando..."

    # Executa dream com timeout nativo bash (compatível macOS — sem dependência de coreutils)
    set +e
    $PYTHON -c "$DREAM_SCRIPT" > /tmp/imi_dream_out_$$.txt 2>&1 &
    dream_pid=$!
    ( sleep "$CONVERGE_TIMEOUT" && kill "$dream_pid" 2>/dev/null ) &
    watchdog_pid=$!
    wait "$dream_pid"
    exit_code=$?
    kill "$watchdog_pid" 2>/dev/null
    wait "$watchdog_pid" 2>/dev/null
    output=$(cat /tmp/imi_dream_out_$$.txt)
    rm -f /tmp/imi_dream_out_$$.txt
    set -e

    log "Round $round output: $output"

    # Extrai métricas do output JSON
    result_json=$(echo "$output" | grep "Result:" | sed 's/.*Result: //' || echo "{}")
    patterns=$(echo "$result_json" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('patterns_extracted',0))" 2>/dev/null || echo "0")
    energy=$(echo "$result_json" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('energy','null'))" 2>/dev/null || echo "null")
    ep_after=$(echo "$result_json" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('episodic_after',0))" 2>/dev/null || echo "0")
    sem_after=$(echo "$result_json" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('semantic_after',0))" 2>/dev/null || echo "0")

    total_patterns=$((total_patterns + patterns))

    # Log estruturado para métricas
    log_json "{\"round\": $round, \"patterns\": $patterns, \"energy\": $energy, \"episodic\": $ep_after, \"semantic\": $sem_after, \"ts\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"

    if [[ $exit_code -eq 0 ]]; then
        log "Round $round: CONVERGIDO ✅ (patterns=+$patterns, energy=$energy)"
        converged=true
        break
    elif [[ $exit_code -eq 2 ]]; then
        log "Round $round: não convergido ainda (patterns=+$patterns, energy=$energy)"
        if [[ $round -lt $MAX_ROUNDS ]]; then
            log "Aguardando 10s antes do próximo round..."
            sleep 10
        fi
    else
        log "Round $round: ERRO (exit=$exit_code)"
        break
    fi
done

# ── Invalida cache do boot para o próximo boot pegar padrões novos ────────────

BOOT_CACHE="$HOME/.imi_boot_cache"
if [[ -f "$BOOT_CACHE" ]]; then
    touch -t "$(date -v-5H +%Y%m%d%H%M 2>/dev/null || date -d '5 hours ago' +%Y%m%d%H%M 2>/dev/null)" "$BOOT_CACHE" 2>/dev/null || rm -f "$BOOT_CACHE"
    log "Boot cache invalidado (próximo boot vai reconstruir com novos padrões)"
fi

# ── Relatório final ───────────────────────────────────────────────────────────

if [[ "$converged" == "true" ]]; then
    log "Dream daemon concluído: CONVERGIDO | total_patterns=+$total_patterns"
else
    log "Dream daemon concluído: NÃO CONVERGIDO após $MAX_ROUNDS rounds | total_patterns=+$total_patterns"
    log "NOTA: espaço grande demais para convergir num ciclo — normal com 236K+ nós"
fi

log "────────────────────────────────────────"
