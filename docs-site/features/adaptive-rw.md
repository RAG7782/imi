# Adaptive Relevance Weight

The relevance_weight (`rw`) parameter controls how much temporal and affective signals influence search results versus pure semantic similarity. Instead of a fixed value, IMI automatically adjusts `rw` based on detected query intent.

## The four intents

| Intent | Keywords | rw | When to use |
|--------|----------|----|-------------|
| `TEMPORAL` | recent, latest, last week, today, yesterday, just happened, this sprint | `0.15` | User wants recent or currently active memories |
| `EXPLORATORY` | find all, list every, all incidents, comprehensive, show me all | `0.00` | User wants maximum recall, pure cosine |
| `ACTION` | how to, fix, prevent, handle, resolve, mitigate, steps to, runbook | `0.05` | User wants actionable memories, slight recency |
| `DEFAULT` | anything else | `0.10` | Balanced agent retrieval |

## How it works

At navigate time, if `relevance_weight` is not explicitly passed, `AdaptiveRW.classify()` runs a zero-cost keyword regex against the query:

```python
from imi.adaptive import AdaptiveRW

arw = AdaptiveRW()

rw = arw.classify("recent auth failures")           # → 0.15 (TEMPORAL)
rw = arw.classify("find all cert expiry incidents")  # → 0.00 (EXPLORATORY)
rw = arw.classify("how to prevent DNS outages")      # → 0.05 (ACTION)
rw = arw.classify("auth token failures")             # → 0.10 (DEFAULT)
```

To get the intent alongside the weight:

```python
rw, intent = arw.classify_with_info("recent outages")
print(f"rw={rw}, intent={intent.value}")
# rw=0.15, intent=temporal
```

## Using with IMISpace

Adaptive weighting is on by default in `navigate()`:

```python
# Automatic — let AdaptiveRW pick the rw
result = space.navigate("recent auth failures")

# Manual override — bypass adaptive weighting
result = space.navigate("auth failures", relevance_weight=0.0)
```

The `NavigationResult` does not expose which `rw` was used directly, but the MCP and REST API responses include `relevance_weight_used` and `intent` fields for transparency.

## Overriding the rw map

You can supply a custom intent-to-rw mapping:

```python
from imi.adaptive import AdaptiveRW, QueryIntent

custom_arw = AdaptiveRW(intent_rw={
    QueryIntent.TEMPORAL: 0.20,    # more aggressive recency
    QueryIntent.EXPLORATORY: 0.00,
    QueryIntent.ACTION: 0.08,
    QueryIntent.DEFAULT: 0.12,
})
```

Pass it to `IMISpace` at construction:

```python
from imi.space import IMISpace

space = IMISpace.from_sqlite("agent.db")
space.adaptive_rw = custom_arw
```

## Evidence from experiments

### WS-A: Ablation study

Tested fixed rw values on 100 SRE postmortems with 15 queries (5 per scenario):

| rw | R@5 | Domain Precision@5 | MRR |
|----|-----|--------------------|-----|
| 0.00 (pure cosine) | **0.341** | 0.689 | **0.702** |
| 0.10 | 0.304 | 0.722 | 0.673 |
| 0.15 | 0.289 | 0.731 | 0.659 |
| 0.30 | 0.204 | 0.689 | 0.576 |

Finding: `rw=0.30` (previous default) degraded R@5 by -40%. Corrected to `rw=0.10`.

### WS-B: Temporal decay

90-day simulation with power-law access patterns (300 incidents):

| Scenario | rw=0.00 | rw=0.15 | Delta |
|----------|---------|---------|-------|
| All queries — domain precision@5 | 0.689 | 0.756 | **+6.7%** |
| Recent/recurring incidents | 0.800 | 0.900 | **+10.0%** |
| Old/historical incidents | 0.600 | 0.600 | 0.000 |

Finding: `rw=0.15` for temporal queries improves precision without degrading access to old memories.

### Adaptive vs fixed (WS3)

Keyword-based adaptive classification (16 mixed queries):

- Classification accuracy: 100% (16/16)
- Adaptive MRR: **0.651** vs best fixed (0.643)
- Cost: zero LLM calls (pure regex)
