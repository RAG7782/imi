# Affordances

Affordances represent what a memory enables an agent to **do** — not just what the memory is about. The concept is inspired by Gibson's (1979) ecological psychology: an affordance is an action potential offered by the environment.

In IMI, every memory node carries a list of affordances extracted at encode time.

## What an affordance is

An `Affordance` has four fields:

| Field | Type | Description |
|-------|------|-------------|
| `action` | str | What can be done (verb phrase) |
| `confidence` | float (0–1) | How reusable this affordance is across situations |
| `conditions` | str | When/where this applies |
| `domain` | str | Category tag (e.g., "debugging", "architecture", "process") |

Example affordance extracted from an incident postmortem:

```python
Affordance(
    action="set memory limits in Kubernetes Helm chart",
    confidence=0.90,
    conditions="deploying services that cache on startup",
    domain="kubernetes",
)
```

## How affordances are extracted

At encode time, after affect assessment, IMI calls the LLM with a structured prompt:

```
You extract ACTION POTENTIALS from experiences. An affordance is something
the agent CAN DO in the future because of what it learned from this experience.

For each affordance, specify:
- action: what can be done (verb phrase)
- confidence: 0.0-1.0 how reusable this is
- conditions: when/where this applies
- domain: category (e.g., "debugging", "architecture", "process", "tooling")

Return JSON array. Max 4 affordances per experience.
```

LLM temperature is `0.3` to keep outputs focused and consistent. Up to 4 affordances are extracted per experience.

The extraction is part of the standard `encode()` pipeline — no additional calls needed.

## Searching by affordances

`search_affordances()` ranks memories by what actions they enable, not by content similarity:

```python
results = space.search_affordances("rollback deployment", top_k=5)

for r in results:
    print(f"[{r['confidence']:.0%}] {r['action']}")
    print(f"  when: {r['conditions']}")
    print(f"  from: {r['memory_summary'][:100]}")
    print()
```

Output:

```
[90%] rollback deployment using Helm release history
  when: after a bad deploy causes service degradation
  from: Production deploy caused 3x latency spike. Rolled back via `helm rollback`

[80%] check diff between current and previous Helm values before rollback
  when: rollback to avoid introducing a regression
  from: Production deploy caused 3x latency spike. Rolled back via `helm rollback`

[75%] monitor error rate for 5 minutes after rollback
  when: after any production rollback
  from: Auth service OOM after deploy, rollback restored stability in 90s
```

### Scoring

Results are ranked by `similarity × confidence`:

```python
score = cosine_similarity(query_embedding, affordance_action_embedding) × affordance.confidence
```

This means high-confidence affordances are boosted, and low-confidence guesses are demoted even if they match the query keywords.

## Accessing affordances on a node

```python
node = space.encode("Fixed latency by adding read replica...")

for aff in node.affordances:
    print(aff)
# [90%] add read replica to PostgreSQL (when: read-heavy workloads)
# [75%] monitor replication lag after adding replica (when: any replica setup)
```

Affordances are also included in `navigate()` results:

```python
result = space.navigate("database performance", zoom="medium")
for mem in result.memories:
    print(mem["affordances"])  # list of str
```

## Why affordances matter for agents

Traditional RAG retrieves what is most similar. An SRE agent asking "what should I do about this OOM?" doesn't want the most similar memory — it wants memories that have proven useful for taking action in similar situations.

Affordances bridge this gap: they index memories by their future utility, not their past content.
