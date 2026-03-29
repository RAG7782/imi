# Quickstart

This guide walks through the core IMI workflow in under 5 minutes: create a space, encode memories, navigate, dream, and search by action.

## 1. Create a memory space

IMI uses SQLite by default — no external services required.

```python
from imi.space import IMISpace

space = IMISpace.from_sqlite("my_agent.db")
```

## 2. Encode memories

```python
n1 = space.encode(
    "DNS resolution failure at 03:00 UTC. All auth services went down for 12 minutes. "
    "Root cause: expired cert on the resolver pod.",
    tags=["dns", "auth", "incident"],
    source="postmortem",
)

n2 = space.encode(
    "Auth token validation service restarted after OOM kill. "
    "Cause: cache warming after deploy exceeded 4GB limit.",
    tags=["auth", "oom", "incident"],
    source="pagerduty",
)

n3 = space.encode(
    "Fixed the OOM issue by setting explicit memory limits in the Helm chart. "
    "Added pre-deploy cache warmup job to spread load.",
    tags=["auth", "fix", "kubernetes"],
    source="pr-description",
)

print(n1.id, n1.summary_medium)
```

Output:

```
abc123 DNS resolver cert expired → auth cascade, 12min outage (03:00 UTC)
```

Each `MemoryNode` carries:
- `summary_orbital` — 10-token gist
- `summary_medium` — 40-token summary (default zoom)
- `summary_detailed` — 100-token summary
- `affect` — salience, valence, arousal scores
- `affordances` — list of action potentials extracted by the LLM
- `mass` — gravitational weight (derived from affect)

## 3. Navigate (search)

```python
result = space.navigate("auth failures", zoom="medium", top_k=5)
print(result)
```

Output:

```
[Navigate: zoom=medium, hits=3, ~120 tokens]
  [0.87] [episodic] S=72% A=sal=0.85 Auth token OOM kill after deploy, cache exceeded 4GB
  [0.81] [episodic] S=68% A=sal=0.79 DNS cert expired → auth cascade 12min outage
  [0.74] [episodic] S=41% A=sal=0.55 OOM fix: Helm memory limits + pre-deploy warmup job
```

Zoom levels:
- `orbital` — gist only (~10 tokens each)
- `medium` — default, balanced (~40 tokens)
- `detailed` — full summary (~100 tokens)
- `full` — reconstruct from seed, uses LLM (~200 tokens, first 3 results only)

## 4. Dream (consolidation)

After encoding several related memories, run a consolidation cycle to extract patterns.

```python
report = space.dream()
print(report)
```

Output:

```
Maintenance: 0 faded, 1 consolidated, 0 pruned, 1 patterns (312ms)
```

The dream cycle:
1. **Fades** memories not accessed recently
2. **Clusters** episodic memories with cosine similarity >= 0.45
3. **Consolidates** clusters into semantic patterns (episodic → semantic store)
4. **Tracks convergence** via annealing energy

Check convergence:

```python
print(space.annealing)
# AnnealingState(iteration=1, converged=False, energy=2.341)
```

## 5. Search by action

Instead of searching by content, search by what memories enable you to **do**.

```python
results = space.search_affordances("restart service after OOM", top_k=3)
for r in results:
    print(f"[{r['confidence']:.0%}] {r['action']}")
    print(f"  when: {r['conditions']}")
    print(f"  from: {r['memory_summary'][:80]}")
    print()
```

Output:

```
[90%] set memory limits in Kubernetes Helm chart
  when: deploying services that cache on startup
  from: Fixed OOM by setting memory limits, added pre-deploy warmup job

[80%] trigger cache warmup before scaling up replicas
  when: after service restart following OOM or fresh deploy
  from: Fixed OOM by setting memory limits, added pre-deploy warmup job

[75%] check resolver pod certs before auth deploy
  when: any auth-related deploy or certificate renewal
  from: DNS cert expired → auth cascade 12min outage
```

## 6. Persist and reload

```python
# SQLite-backed spaces auto-persist on every encode/navigate/dream
# Load an existing space:
space2 = IMISpace.from_sqlite("my_agent.db")
print(space2.stats())
```

Output:

```python
{
  'episodic_total': 3,
  'semantic_total': 1,
  'total_affordances': 9,
  'temporal_sessions': 1,
  'annealing': 'AnnealingState(iteration=1, converged=False)',
  'avg_surprise': 0.0,
  'avg_mass': 1.24,
  'avg_salience': 0.73
}
```
