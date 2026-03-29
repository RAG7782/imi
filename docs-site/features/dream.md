# Dream (Consolidation)

`dream()` runs one consolidation cycle — the memory equivalent of sleep. It clusters similar episodic memories into abstract semantic patterns, mimicking hippocampal replay and neocortical consolidation (Complementary Learning Systems theory).

## What dream() does

One call to `space.dream()` executes four steps:

### 1. Fade

Marks episodic nodes that haven't been accessed in 30+ days and have low relevance (`< 0.1`). Does not delete them — marks them as faded candidates for future pruning.

### 2. Cluster

Finds groups of episodic memories with cosine similarity ≥ threshold (default `0.45`). Uses a greedy single-linkage approach over the full similarity matrix.

### 3. Consolidate

For each cluster with 2+ members, calls the LLM once to extract the general pattern:

```
Given multiple related memories, extract the GENERAL PATTERN — not a summary
of individual events, but the recurring theme or rule they demonstrate.
```

The resulting `PatternNode` is stored in the **semantic store** (not episodic). If a very similar pattern already exists (cosine > 0.90), it is strengthened instead of duplicated.

### 4. Track convergence

After consolidation, computes the space energy (weighted pairwise distances over embeddings and masses). This is fed into `AnnealingState` to track whether the memory space is converging.

```
energy = Σ mass_i × mass_j / distance_ij  (for all pairs)
```

The space is considered converged when energy stops decreasing across iterations.

## Calling dream()

```python
report = space.dream()
print(report)
# Maintenance: 2 faded, 3 consolidated, 1 pruned, 4 patterns (489ms)
```

### MaintenanceReport fields

| Field | Description |
|-------|-------------|
| `faded` | Nodes marked as low-relevance candidates |
| `consolidated` | New semantic patterns created |
| `pruned` | Nodes flagged for pruning (relevance < 0.05) |
| `patterns_total` | Total patterns in semantic store after this cycle |
| `duration_ms` | Wall time for the full cycle |

### Convergence state

```python
print(space.annealing)
# AnnealingState(iteration=5, converged=True, energy=1.234)

# Full history
print(space.annealing.energy_history)
# [4.21, 3.15, 2.41, 1.87, 1.23, 1.23]  ← plateau = converged
```

`converged=True` means the memory space is stable — further dreaming won't create new patterns.

## When to call dream()

| Scenario | Recommendation |
|----------|---------------|
| After every N encodes | Call every 20–50 encodes |
| End of session | Call once before saving/exiting |
| Periodic maintenance | Schedule daily or hourly via cron |
| Before long-horizon navigation | Consolidation improves semantic search quality |

A good heuristic: call `dream()` when `len(space.episodic) % 20 == 0`.

```python
node = space.encode(experience)
if len(space.episodic) % 20 == 0:
    space.dream()
```

## Threshold tuning

Lower `similarity_threshold` = more aggressive clustering (creates more patterns from less-similar memories):

```python
# More permissive — clusters memories with ≥ 45% cosine similarity (default)
space.dream(similarity_threshold=0.45)

# More conservative — only cluster very similar memories
space.dream(similarity_threshold=0.70)
```

## Annealing convergence

The annealing metaphor describes the long-term behavior of the memory space:

- **High energy**: many clusters of related memories still uncondensed, lots of redundancy
- **Low energy**: patterns have been extracted, episodic store is organized
- **Converged**: space is stable — further dreaming returns empty reports

The energy is not a quality metric — it measures organization, not accuracy. A converged space still retrieves correctly; it just won't produce new patterns on the next `dream()` call.
