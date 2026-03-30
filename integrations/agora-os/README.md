# IMI ↔ AGORA-OS Integration

Connects IMI (cognitive episodic memory) with AGORA-OS (semantic operating system).

## Architecture

```
AGORA-OS                          IMI
┌─────────────────┐              ┌──────────────────┐
│ agora-kb (KB)   │─── NAOs ───→│ Semantic memory   │
│ Facts, Relations│              │ (consolidated)    │
└────────┬────────┘              └──────────────────┘
         │                                ↑
┌────────┴────────┐              ┌──────────────────┐
│ agora-workflow   │──encode───→ │ Episodic memory   │
│ Steps, Decisions│              │ (temporal, affect) │
│ Quality Gates   │←─navigate── │ Graph, Affordances│
└─────────────────┘              └──────────────────┘
```

## Components

1. **Hook**: `agora-imi-memory.sh` — PostToolUse hook that auto-encodes after Skill execution
2. **Skill**: `/agora-imi` — Claude Code skill for explicit memory operations within AGORA workflows
3. **Bridge**: `bridge.py` — Python module that syncs NAOs ↔ MemoryNodes and relations ↔ graph edges

## Setup

1. Copy hook to AGORA-OS plugin:
   ```bash
   cp hooks/agora-imi-memory.sh ~/experimentos/agora-os-plugin/scripts/
   ```

2. Add hook to AGORA-OS hooks.json (PostToolUse on Skill)

3. Register IMI MCP server globally:
   ```bash
   claude mcp add imi-memory -- /path/to/imi/.venv/bin/python -m imi.mcp_server
   ```

4. Copy skill:
   ```bash
   cp skills/agora-imi.md ~/experimentos/agora-os-plugin/skills/agora-imi/SKILL.md
   ```
