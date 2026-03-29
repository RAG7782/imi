# Research Insights

Consolidated findings from 9 experimental workstreams (WS1 through WS-I). All metrics are reproducible by running the experiments in `/experiments/`.

## The core finding: Retrieval vs Relevance paradox

IMI uncovered a fundamental trade-off that RAG literature does not address:

> Features that model human memory (recency, affect, mass) **degrade pure retrieval** but **improve relevance for agents**.

An SRE agent querying "recent auth failures" doesn't want the most semantically similar memory — it wants the most useful one for the current situation. RAG optimizes the wrong metric.

### Quantified evidence

| Scenario | Metric | RAG (rw=0) | IMI (rw=0.1) | Delta |
|----------|--------|-----------|--------------|-------|
| Pure retrieval | R@5 | **0.341** | 0.304 | -0.037 |
| Agent temporal queries | DomPrec@5 | 0.689 | **0.756** | +0.067 |
| Recent incidents | DomPrec@5 | 0.800 | **0.900** | +0.100 |
| Multi-hop retrieval | R@10 | 0.750 | **1.000** | +0.250 |
| Temporal coherence | Avg age top-5 | 41.2d | **16.8d** | -24.4d |

---

## Insight 1: The sweet spot is rw=0.10

```
rw=0.00  → best pure retrieval, zero temporal awareness
rw=0.10  → -3.7% retrieval, +6.7% domain precision, -59% avg age of results
rw=0.15  → sweet spot for "recent" queries, no penalty for "old" queries
rw=0.30  → -40% retrieval (too aggressive — this was the previous default)
```

The previous default of `rw=0.30` was wrong. `rw=0.10` is the empirically validated optimum. Corrected in the code.

## Insight 2: Surprise is elegant but negligible for retrieval

- **Cost**: 2 LLM calls per encode (predict + compute surprise)
- **Benefit**: +0.003 R@5 — statistically negligible
- **Status**: Now opt-in (`use_predictive_coding=False` by default)

Surprise may have unmeasured value as an anomaly signal or a priority criterion for consolidation (high-surprise memories deserve consolidation first). Not yet validated.

## Insight 3: Graph edges solve multi-hop with zero LLM calls

| System | Multi-hop R@10 | LLM calls/query |
|--------|---------------|-----------------|
| Cosine only | 75% (15/20) | 0 |
| HippoRAG-Sim | 10% (1/10) | 1 NER/query |
| **IMI + Graph** | **100% (20/20)** | **0** |

Spreading activation (Collins & Loftus, 1975) over auto-detected similarity edges beats the HippoRAG simulation on multi-hop. The 5 queries that cosine failed on were all resolved by 1-hop expansion.

Note: The HippoRAG comparison used a regex-NER simulation, not real HippoRAG with LLM NER. Real HippoRAG would perform significantly better.

## Insight 4: Old memories do not "rot"

In a 365-day simulation with 600 incidents:
- Q1 incident searched from Q4 arrived at **rank 1** in both cosine and IMI
- R@5 declines over time, but this is a denominator effect (more similar incidents accumulate), not real degradation

`rw=0.10` does not push old memories down. `fade_resistance` from affective tagging protects emotionally salient memories from decay.

## Insight 5: Multi-agent shared memory has measurable value

3 specialized agents sharing memory vs isolated:
- Shared memory wins on 2/5 cross-domain queries
- Biggest gain: queries crossing 3+ domains
- Future work: agent-scoped views with trust gradients per agent

## Insight 6: AMBench is the first agent memory benchmark

No existing benchmark tests the full agent memory lifecycle: encode → retrieve → consolidate → act → learn.

AMBench tests 5 dimensions that RAG benchmarks ignore:
1. Retrieval accuracy (R@5, MRR)
2. Consolidation quality (cluster purity)
3. Action relevance (affordance precision@1)
4. Temporal coherence (avg age of top-5 results)
5. Learning curve (improvement over time)

Status: Functional with 300–600 incidents, 10 patterns, 90–365 simulated days.

## Insight 7: SQLite is 87x faster than TimescaleDB

For single-agent workloads, SQLite with WAL mode vastly outperforms TimescaleDB:
- 87x lower latency for O(1) inserts
- -425 lines of code removed
- -11 tests removed
- -1 docker-compose.yml removed

TimescaleDB would only make sense for multi-agent analytics at scale — a use case that does not yet exist for IMI.

---

## Evidence map

```
WS1 (Perception)    → embeddings work, zoom levels useful
WS2 (Surprise)      → implemented but negligible impact → made opt-in
WS3 (Validation)    → 46 tests, baseline metrics established
WS4 (Deep Dive)     → bug fixes, SQLite backend, surprise integrated
WS-A (Ablation)     → each feature's contribution quantified
                       ├─ surprise: +0.003 (negligible)
                       ├─ recency: biggest impact (-0.117 if removed)
                       └─ optimal rw = 0.10
WS-B (Temporal)     → rw helps agent scenarios (+6.7%)
                       └─ recent queries: +10% with rw=0.15
WS-C (HippoRAG)     → motivated graph layer (WS-G)
WS-D (AMBench)      → first agent memory benchmark
                       └─ temporal coherence: IMI 16.8d vs RAG 41.2d
WS-E (Architecture) → TimescaleDB removed, predictive opt-in
WS-F (rw default)   → 0.3 → 0.10 (evidence-based correction)
WS-G (Graph layer)  → 100% multi-hop recall, zero standard degradation
WS-H (Paper draft)  → full paper with all results
WS-I (Expanded)     → multi-agent, cross-domain, 365-day validated
```
