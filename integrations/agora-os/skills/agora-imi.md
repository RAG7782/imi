---
name: agora-imi
description: IMI Memory Bridge for AGORA-OS — episodic memory for workflows. Auto-recall past decisions, encode outcomes, learn from experience.
user_invocable: true
allowed-tools: [Read, Grep, Glob, Agent]
model: sonnet
effort: low
---

# /agora-imi — Episodic Memory for AGORA Workflows

You have access to IMI memory tools via MCP (`imi_encode`, `imi_navigate`, `imi_dream`, `imi_search_actions`, `imi_stats`, `imi_graph_link`).

This skill bridges AGORA-OS workflows with IMI cognitive memory.

## Commands

### `/agora-imi recall <context>`
**Before starting any AGORA workflow**, recall relevant past experience.

1. Call `imi_navigate` with a summary of the current task/workflow
2. If results found, present them as **MEMORY CONTEXT**:
   ```
   MEMORY CONTEXT (from past AGORA workflows):
   - [0.85] "Last deploy of auth-service failed due to missing env vars" (3 days ago)
   - [0.72] "Council recommended circuit breaker pattern for cascading services" (1 week ago)
   Affordances: "check env vars before deploy", "add circuit breaker"
   ```
3. This context should inform the current workflow execution

### `/agora-imi save <outcome>`
**After completing an AGORA workflow step**, save the outcome.

1. Call `imi_encode` with:
   - Experience: the outcome/decision/result
   - Tags: `agora`, skill name, workflow type, relevant domain
   - Source: `agora-os`
2. If this outcome RELATES to a previous memory (causal), call `imi_graph_link`

### `/agora-imi patterns`
**During planning (@pm) or review (@qa)**, find recurring patterns.

1. Call `imi_navigate` with "recurring patterns in AGORA workflows"
2. Call `imi_navigate` with "failed deployments" or "rejected stories" etc.
3. Synthesize into actionable intelligence:
   ```
   PATTERN ANALYSIS:
   - Deploys on Friday: 2x rollback rate (3 incidents vs 1.5 avg)
   - Auth stories: always miss Spec Constraints (rejected 3x by @po)
   - Database migrations: take 2x estimated time (4 stories measured)
   ```

### `/agora-imi dream`
**At end of sprint/milestone**, consolidate workflow memories.

1. Call `imi_dream` to cluster similar experiences
2. Report patterns extracted:
   ```
   CONSOLIDATION:
   - 3 clusters formed from 28 workflow memories
   - Pattern: "auth changes require extra QA" (5 memories)
   - Pattern: "deploy timing matters" (3 memories)
   ```

### `/agora-imi stats`
Show AGORA memory statistics.

### `/agora-imi` (no args)
Show this help.

## Auto-integration with AGORA Workflow

When used within an `agora-workflow` execution:

1. **Before each step**: Auto-recall relevant memories
   - Before @pm: recall past PRD decisions
   - Before @dev: recall past implementation issues
   - Before @qa: recall past quality concerns
   - Before @devops: recall past deploy incidents

2. **After each step**: Auto-encode the outcome
   - After @po validation: encode GO/NO-GO with reasons
   - After @qa review: encode PASS/FAIL with findings
   - After @devops deploy: encode success/rollback

3. **Cross-workflow learning**:
   - Link related memories via graph (e.g., "this deploy failure CAUSED this rollback")
   - Over time, the system learns: "auth deploys on Friday → high risk"

## Integration with AGORA KB

IMI stores **episodic** memory (experiences, decisions, temporal context).
AGORA KB stores **semantic** memory (facts, procedures, relations).

They complement each other:
- KB: "Circuit breaker pattern requires timeout configuration" (PROCEDURE, confidence=90)
- IMI: "Last time we added circuit breaker, timeout was too low, caused 5min outage" (EXPERIENCE, salience=0.8)

When recalling, merge both:
```
KB says: Circuit breaker needs timeout config
IMI says: Last time timeout was too low → 5min outage (3 days ago, salience=0.8)
→ Recommendation: Set timeout to 2x the p99 latency (learned from experience)
```
