#!/bin/bash
# IMI Activate — One command to enable all IMI features
#
# Usage:
#   source scripts/activate.sh         # Full activation
#   source scripts/activate.sh --check # Check status only
#
# What it does:
#   1. Registers IMI MCP server globally (if not already)
#   2. Installs /agora-imi skill into AGORA-OS (if present)
#   3. Installs IMI memory hook into AGORA-OS (if present)
#   4. Sets up IMI_DB and local embedding environment variables
#   5. Verifies everything works

IMI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGORA_ROOT="$HOME/experimentos/agora-os-plugin"
IMI_PYTHON="$IMI_ROOT/.venv/bin/python"
IMI_DB="${IMI_DB:-$HOME/.claude/plugins/data/imi/agora-memory.db}"
IMI_EMBEDDER_PROVIDER="${IMI_EMBEDDER_PROVIDER:-ollama}"
IMI_EMBEDDER_MODEL="${IMI_EMBEDDER_MODEL:-all-minilm}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
CLAUDE_JSON="$HOME/.claude.json"

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✔${NC} $1"; }
warn() { echo -e "  ${YELLOW}△${NC} $1"; }
fail() { echo -e "  ${RED}✘${NC} $1"; }
info() { echo -e "  ${BLUE}→${NC} $1"; }

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  IMI — Integrated Memory Intelligence${NC}"
echo -e "${BLUE}  Activation Script${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── Check mode ──
if [[ "$1" == "--check" ]]; then
    echo "Status check:"

    # MCP server
    if grep -q "imi-memory" "$CLAUDE_JSON" 2>/dev/null; then
        ok "MCP server registered globally"
    else
        fail "MCP server not registered"
    fi

    # AGORA skill
    if [[ -f "$AGORA_ROOT/skills/agora-imi/SKILL.md" ]]; then
        ok "/agora-imi skill installed in AGORA-OS"
    else
        warn "/agora-imi skill not installed (AGORA-OS integration)"
    fi

    # AGORA hook
    if [[ -f "$AGORA_ROOT/scripts/agora-imi-memory.sh" ]]; then
        ok "IMI memory hook installed in AGORA-OS"
    else
        warn "IMI memory hook not installed (AGORA-OS integration)"
    fi

    # Python + deps
    PYCHECK=$("$IMI_PYTHON" -c "from imi.space import IMISpace; print('OK')" 2>&1 | grep -c "OK")
    if [[ "$PYCHECK" -ge 1 ]]; then
        ok "IMI Python package OK"
    else
        fail "IMI Python package not importable"
    fi

    ok "Embedding provider: $IMI_EMBEDDER_PROVIDER/$IMI_EMBEDDER_MODEL via $OLLAMA_BASE_URL"

    # DB
    if [[ -f "$IMI_DB" ]]; then
        COUNT=$("$IMI_PYTHON" -c "
from imi.space import IMISpace
s = IMISpace.from_sqlite('$IMI_DB')
print(len(s.episodic))
" 2>/dev/null || echo "0")
        ok "Memory database: $COUNT memories in $IMI_DB"
    else
        info "No memory database yet (will be created on first encode)"
    fi

    echo ""
    return 0 2>/dev/null || exit 0
fi

# ── Step 1: Ensure IMI data directory ──
echo "Step 1: Data directory"
mkdir -p "$(dirname "$IMI_DB")"
ok "Created $(dirname "$IMI_DB")"

# ── Step 2: Register MCP server globally ──
echo "Step 2: MCP server"
if grep -q "imi-memory" "$CLAUDE_JSON" 2>/dev/null; then
    ok "Already registered in ~/.claude.json"
else
    # Use claude CLI to add
    if command -v claude &>/dev/null; then
        claude mcp add imi-memory -- "$IMI_PYTHON" -m imi.mcp_server 2>/dev/null
        ok "Registered via 'claude mcp add'"
    else
        warn "Claude CLI not found. Add manually: claude mcp add imi-memory -- $IMI_PYTHON -m imi.mcp_server"
    fi
fi

# ── Step 3: Set IMI_DB env ──
echo "Step 3: Environment"
export IMI_DB="$IMI_DB"
export IMI_PYTHON="$IMI_PYTHON"
export IMI_EMBEDDER_PROVIDER="$IMI_EMBEDDER_PROVIDER"
export IMI_EMBEDDER_MODEL="$IMI_EMBEDDER_MODEL"
export OLLAMA_BASE_URL="$OLLAMA_BASE_URL"
ok "IMI_DB=$IMI_DB"
ok "IMI_EMBEDDER_PROVIDER=$IMI_EMBEDDER_PROVIDER"
ok "IMI_EMBEDDER_MODEL=$IMI_EMBEDDER_MODEL"
ok "OLLAMA_BASE_URL=$OLLAMA_BASE_URL"

# ── Step 4: AGORA-OS integration (if present) ──
echo "Step 4: AGORA-OS integration"
if [[ -d "$AGORA_ROOT" ]]; then
    # Install skill
    mkdir -p "$AGORA_ROOT/skills/agora-imi"
    cp "$IMI_ROOT/integrations/agora-os/skills/agora-imi.md" "$AGORA_ROOT/skills/agora-imi/SKILL.md"
    ok "Installed /agora-imi skill"

    # Install hook
    cp "$IMI_ROOT/integrations/agora-os/hooks/agora-imi-memory.sh" "$AGORA_ROOT/scripts/agora-imi-memory.sh"
    chmod +x "$AGORA_ROOT/scripts/agora-imi-memory.sh"
    ok "Installed memory hook"

    # Check if hook is in hooks.json (don't modify automatically — just warn)
    if grep -q "agora-imi-memory" "$AGORA_ROOT/hooks/hooks.json" 2>/dev/null; then
        ok "Hook registered in hooks.json"
    else
        warn "Hook not in hooks.json yet. Add PostToolUse matcher for Skill:"
        info "  \"command\": \"\${CLAUDE_PLUGIN_ROOT}/scripts/agora-imi-memory.sh \\\"\$TOOL_INPUT\\\"\""
    fi
else
    info "AGORA-OS not found at $AGORA_ROOT (skipping integration)"
fi

# ── Step 5: Verify ──
echo "Step 5: Verification"
if "$IMI_PYTHON" -c "
from imi.mcp_server import mcp
from imi.space import IMISpace
s = IMISpace.from_sqlite('$IMI_DB')
print(f'OK: {len(s.episodic)} memories')
" 2>/dev/null; then
    ok "IMI fully operational"
else
    fail "Verification failed — check Python environment"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  IMI activated! Available in any Claude Code session.${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Usage:"
echo "    claude                    # Start Claude Code (IMI tools available)"
echo "    /imi stats                # Check memory space"
echo "    /imi save <experience>    # Encode a memory"
echo "    /imi recall <query>       # Search memories"
echo "    /agora-imi recall <ctx>   # AGORA-OS integration"
echo ""
echo "  Check status anytime:"
echo "    source scripts/activate.sh --check"
echo ""
