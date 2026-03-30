#!/bin/bash
# AGORA-OS Hook: Auto-encode workflow steps into IMI memory
#
# Triggered: PostToolUse on Skill execution
# Input: $1 = TOOL_INPUT (JSON with skill name and args)
#
# This hook captures the output of each AGORA skill execution
# and encodes it into IMI as an episodic memory with:
# - Tags: agora, skill name, workflow type
# - Source: agora-os
# - Context: skill output summary
#
# Install:
#   cp this file to ~/experimentos/agora-os-plugin/scripts/
#   Add to hooks.json PostToolUse for Skill tool

TOOL_INPUT="$1"
IMI_PYTHON="${IMI_PYTHON:-/Users/renatoaparegomes/experimentos/imi/.venv/bin/python}"
IMI_DB="${IMI_DB:-/Users/renatoaparegomes/.claude/plugins/data/imi/agora-memory.db}"

# Extract skill name from input
SKILL_NAME=$(echo "$TOOL_INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('skill', data.get('name', 'unknown')))
except:
    print('unknown')
" 2>/dev/null)

# Only encode AGORA skills (not other skills)
case "$SKILL_NAME" in
    agora-*|ax-*|ps-*)
        ;;
    *)
        exit 0
        ;;
esac

# Extract a summary from the input
SUMMARY=$(echo "$TOOL_INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    args = data.get('args', data.get('arguments', ''))
    skill = data.get('skill', data.get('name', 'unknown'))
    print(f'AGORA {skill} executed: {str(args)[:200]}')
except:
    print('AGORA skill executed')
" 2>/dev/null)

# Encode into IMI
"$IMI_PYTHON" -c "
import os
os.environ['IMI_DB'] = '$IMI_DB'
from imi.space import IMISpace
space = IMISpace.from_sqlite('$IMI_DB')
space.encode(
    '''$SUMMARY''',
    tags=['agora', '$SKILL_NAME', 'workflow-step'],
    source='agora-os',
)
" 2>/dev/null

exit 0
