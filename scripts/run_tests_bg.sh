#!/usr/bin/env bash
# run_tests_bg.sh — Executa suite de testes IMI em background
# Saída gravada em logs/test_run_YYYYMMDD_HHMMSS.log
# Uso: bash scripts/run_tests_bg.sh [--full]
#
# Modos:
#   (default)  — testes rápidos: ignora chromadb, mcp, anthropic
#   --full     — suite completa (requer chromadb instalado)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/test_run_${TIMESTAMP}.log"
RESULT_FILE="$LOG_DIR/test_run_${TIMESTAMP}.result"

FULL_MODE=0
for arg in "$@"; do
    [[ "$arg" == "--full" ]] && FULL_MODE=1
done

cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

if [[ $FULL_MODE -eq 1 ]]; then
    IGNORE_FLAGS=""
    DESCRIPTION="suite completa"
else
    IGNORE_FLAGS="--ignore=tests/test_lite.py --ignore=tests/test_mcp_server.py"
    DESCRIPTION="suite core (sem chromadb/mcp/anthropic)"
fi

echo "[run_tests_bg] Iniciando $DESCRIPTION → $LOG_FILE"
echo "[run_tests_bg] Rodando em background — tail -f $LOG_FILE para acompanhar"

{
    echo "=== IMI Test Suite === $(date)"
    echo "Mode: $DESCRIPTION"
    echo "Log: $LOG_FILE"
    echo "========================================"

    PYTHONPATH="$ROOT" python3 -m pytest tests/ $IGNORE_FLAGS \
        -v \
        --tb=short \
        --no-header \
        -q \
        2>&1

    EXIT_CODE=$?
    echo "========================================"
    echo "Exit code: $EXIT_CODE"
    echo "Finished: $(date)"

    # Gravar resultado compacto
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "PASS" > "$RESULT_FILE"
        # Notificação macOS
        osascript -e "display notification \"IMI tests: PASS ✅\" with title \"Test Runner\" sound name \"Glass\"" 2>/dev/null || true
    else
        echo "FAIL" > "$RESULT_FILE"
        osascript -e "display notification \"IMI tests: FAIL ❌ — check logs\" with title \"Test Runner\" sound name \"Basso\"" 2>/dev/null || true
    fi

    exit $EXIT_CODE
} >> "$LOG_FILE" 2>&1 &

BG_PID=$!
echo "[run_tests_bg] PID=$BG_PID | Log: $LOG_FILE"
echo "$BG_PID" > "$LOG_DIR/test_run_${TIMESTAMP}.pid"
