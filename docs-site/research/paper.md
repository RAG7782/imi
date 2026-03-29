# Paper

**IMI: Integrated Memory Intelligence for AI Agents — A Cognitive Memory Architecture with Benchmarks**

The paper draft is at `/Users/renatoaparegomes/experimentos/imi/docs/paper-draft.md` and is being prepared for arXiv submission.

## Abstract

We present IMI (Integrated Memory Intelligence), a memory system for AI agents that goes beyond vector-based retrieval by integrating cognitive features: affective modulation, temporal decay, surprise-based encoding, affordance extraction, multi-resolution zoom, and graph-augmented multi-hop retrieval. Through systematic ablation (100 memories, 15 queries), we show that while cognitive features trade off pure retrieval quality (R@5 drops from 0.341 to 0.204 at relevance_weight=0.3), they significantly improve agent-relevant metrics: domain precision increases +6.7% for temporal queries, and graph-augmented expansion achieves 100% multi-hop recall (vs 75% cosine-only). We also introduce AMBench, the first standardized benchmark for agent memory systems, evaluating the full encode-retrieve-consolidate-act lifecycle over 90 simulated days with 300 incidents. IMI is open-source, requires no external infrastructure (SQLite-only), and operates with zero LLM calls at query time.

## Key results

| Experiment | Metric | Baseline | IMI | Delta |
|-----------|--------|----------|-----|-------|
| Ablation (WS-A) | R@5 — rw=0 vs rw=0.10 | 0.341 | 0.304 | -0.037 |
| Temporal (WS-B) | Domain precision@5 | 0.689 | 0.756 | **+0.067** |
| Temporal (WS-B) — recent queries | Domain precision@5 | 0.800 | 0.900 | **+0.100** |
| Graph multi-hop (WS-G) | R@10 | 0.750 | **1.000** | **+0.250** |
| AMBench (WS-D) | Temporal coherence (avg age) | 41.2d | **16.8d** | **-24.4d** |
| Adaptive rw (P1) | MRR vs best fixed | 0.643 | **0.651** | +0.008 |

## Contributions

1. **IMI Architecture** — cognitive memory system with 7 composable features (affect, surprise, affordances, temporal decay, mass, zoom, graph edges)
2. **Ablation study** — systematic measurement of each feature's contribution to retrieval quality and agent relevance
3. **AMBench** — first standardized benchmark for AI agent memory (5 metrics, 3 baselines, 300 incidents, 90 simulated days)
4. **Graph-augmented retrieval** — lightweight edge layer achieving 100% multi-hop recall with zero LLM calls
5. **Empirical finding** — optimal relevance_weight for agent memory is 0.10–0.15, not the 0.3 commonly assumed

## Comparison to prior work

| System | Temporal | Affordances | Multi-hop | Agent Benchmark |
|--------|----------|-------------|-----------|-----------------|
| RAG (vanilla) | No | No | No | No |
| HippoRAG (2024) | No | No | Yes | No |
| RAPTOR (2024) | No | No | Partial | No |
| GraphRAG (2024) | No | No | Yes | No |
| MemGPT (2023) | Yes | No | No | No |
| **IMI** | **Yes** | **Yes** | **Yes** | **Yes (AMBench)** |

## Limitations

- Synthetic evaluation only — all experiments use generated incidents, not real-world agent traces
- Single embedding model (all-MiniLM-L6-v2, 384d) — larger models may shift the retrieval/relevance balance
- HippoRAG comparison used a regex-NER simulation, not the real system
- Single-agent only — AMBench simulates one agent; multi-agent results are preliminary (WS-I)

## Status

Draft complete. Target: arXiv preprint within 1–2 weeks of final edits. See `/docs/paper-draft.md` for the full text.
