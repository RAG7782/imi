---
name: imi
description: IMI Memory — encode decisions, retrieve context, dream consolidation. Use when the user wants to save/recall/search memories, or when working on tasks that benefit from past experience.
user_invocable: true
---

# /imi — Cognitive Memory for Claude Code

You have access to IMI memory tools via MCP. Use them to give this session persistent, cognitive memory.

## Commands

Parse the user's input and execute the appropriate action:

### `/imi save <text>` or `/imi encode <text>`
Encode a memory. Use `imi_encode` MCP tool.
- Auto-detect tags from content
- Report: ID, affect, affordances, mass

### `/imi recall <query>` or `/imi search <query>`
Search memories. Use `imi_navigate` MCP tool.
- Show top 5 results with scores
- Report detected intent and rw used

### `/imi actions <query>`
Find actionable memories. Use `imi_search_actions` MCP tool.
- Show affordances with confidence scores

### `/imi dream`
Run consolidation cycle. Use `imi_dream` MCP tool.
- Report clusters formed and patterns extracted

### `/imi stats`
Show memory space statistics. Use `imi_stats` MCP tool.

### `/imi link <id1> <id2> [type]`
Link two memories. Use `imi_graph_link` MCP tool.
- type: causal (default), co_occurrence, similar

### `/imi` (no args)
Show help with available commands.

## Auto-encode behavior

When the user completes a significant task (commit, deploy, fix a bug, make a decision), proactively suggest:
> "Want me to save this as a memory? `/imi save <summary>`"

## Context enrichment

When starting a new task, check if relevant memories exist:
1. Call `imi_navigate` with a summary of the current task
2. If hits > 0, briefly mention: "Found X relevant memories from past sessions"
3. Use the memories to inform your approach

## Response format

Keep responses concise. After encoding:
```
Saved: [ID] "summary" (salience=X, mass=X)
```

After recalling:
```
Found N memories (intent: TEMPORAL, rw=0.15):
1. [0.85] summary...
2. [0.72] summary...
```
