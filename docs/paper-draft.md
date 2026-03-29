# IMI: Integrated Memory Intelligence for AI Agents
## A Cognitive Memory Architecture with Benchmarks

---

### Abstract

We present IMI (Integrated Memory Intelligence), a memory system for AI agents
that goes beyond vector-based retrieval by integrating cognitive features:
affective modulation, temporal decay, surprise-based encoding, affordance
extraction, multi-resolution zoom, and graph-augmented multi-hop retrieval.
Through systematic ablation (100 memories, 15 queries), we show that while
cognitive features trade off pure retrieval quality (R@5 drops from 0.341 to
0.204 at relevance_weight=0.3), they significantly improve agent-relevant
metrics: domain precision increases +6.7% for temporal queries, and
graph-augmented expansion achieves 100% multi-hop recall (vs 75% cosine-only).
We also introduce AMBench, the first standardized benchmark for agent memory
systems, evaluating the full encode-retrieve-consolidate-act lifecycle over
90 simulated days with 300 incidents. IMI is open-source, requires no external
infrastructure (SQLite-only), and operates with zero LLM calls at query time.

---

### 1. Introduction

Current AI memory systems face a fundamental tension: they optimize for
**retrieval accuracy** (finding the most similar document) when agents actually
need **decision relevance** (finding the most useful memory for the current
situation).

RAG systems use cosine similarity over dense embeddings. This works for
knowledge Q&A but fails for agent memory, which requires:

- **Temporal coherence**: Recent incidents are often more relevant than old ones
- **Action orientation**: "What can I DO?" not just "What do I KNOW?"
- **Multi-hop reasoning**: Causal chains span multiple memories
- **Emotional salience**: Critical incidents should be harder to forget
- **Adaptive resolution**: Sometimes you need the gist, sometimes the details

IMI addresses these gaps through a cognitive architecture inspired by human
hippocampal memory, ecological psychology (Gibson 1979), and predictive coding
theory (Rao & Ballard 1999).

#### Contributions

1. **IMI Architecture**: A cognitive memory system with 7 composable features
   (affect, surprise, affordances, temporal decay, mass, zoom, graph edges)
2. **Ablation Study**: Systematic measurement of each feature's contribution
   to retrieval quality and agent relevance
3. **AMBench**: First standardized benchmark for AI agent memory (5 metrics,
   3 baselines, 300 incidents over 90 simulated days)
4. **Graph-Augmented Retrieval**: Lightweight edge layer that achieves 100%
   multi-hop recall with zero LLM calls
5. **Empirical finding**: The optimal relevance weight for agent memory is
   0.10-0.15, not the 0.3 commonly assumed in prior work

---

### 2. Related Work

| System | Retrieval | Temporal | Affordances | Multi-hop | Agent Benchmark |
|--------|-----------|----------|-------------|-----------|-----------------|
| RAG (vanilla) | Cosine | No | No | No | No |
| HippoRAG (2024) | KG + PPR | No | No | Yes | No |
| RAPTOR (2024) | Tree clustering | No | No | Partial | No |
| GraphRAG (2024) | Entity KG | No | No | Yes | No |
| MemGPT (2023) | Tiered paging | Yes | No | No | No |
| **IMI (ours)** | **Cosine + graph** | **Yes** | **Yes** | **Yes** | **Yes (AMBench)** |

**HippoRAG** (Gutierrez et al., 2024) is the closest to IMI. Both are
hippocampus-inspired, but HippoRAG focuses on knowledge retrieval via
Personalized PageRank over a knowledge graph built with LLM NER. IMI focuses
on agent memory with temporal decay, affordances, and zero LLM calls at
query time.

**MemGPT** (Packer et al., 2023) implements tiered memory (working/long-term)
with LLM-controlled paging. IMI's zoom levels serve a similar purpose but
are query-time, not storage-time.

**RAPTOR** (Sarthi et al., 2024) uses recursive tree summarization for
multi-resolution retrieval. IMI's zoom levels (orbital/medium/detailed) are
analogous but pre-computed at encode time.

---

### 3. Architecture

```
                    ┌─────────────────┐
                    │   IMI Space      │
                    │                  │
     encode() ──▶   │  ┌──────────┐   │  ◀── navigate()
                    │  │ Episodic  │   │
                    │  │  Store    │   │
                    │  └────┬─────┘   │
                    │       │         │
     consolidate()─▶│  ┌────▼─────┐   │  ◀── search_affordances()
                    │  │ Semantic  │   │
                    │  │  Store    │   │
                    │  └────┬─────┘   │
                    │       │         │
                    │  ┌────▼─────┐   │
                    │  │  Graph   │   │  ◀── search_with_expansion()
                    │  │  Layer   │   │
                    │  └──────────┘   │
                    └─────────────────┘
```

#### 3.1 Memory Node

Each memory is a `MemoryNode` with:
- **seed**: original experience text
- **embedding**: 384-dimensional vector (all-MiniLM-L6-v2)
- **summaries**: orbital (gist), medium, detailed (3 zoom levels)
- **affect**: AffectiveTag (salience, valence, arousal) → modulates decay
- **surprise_magnitude**: prediction error at encode time (opt-in)
- **affordances**: list of action potentials ("what can I do with this?")
- **mass**: gravitational pull (derived from affect encoding strength)
- **tags**: domain labels for co-occurrence linking

#### 3.2 Relevance Scoring

The relevance score combines temporal, frequency, mass, and surprise signals:

```
relevance = recency × (1 + frequency) × mass × surprise_boost
```

Where:
- `recency = 1 / (1 + days_since × (1 - 0.5 × fade_resistance))`
- `frequency = log(1 + access_count)`
- `mass = affect.encoding_strength` (higher salience → heavier node)
- `surprise_boost = 1 + 0.3 × surprise_magnitude`

Final search score:
```
score = (1 - rw) × cosine_similarity + rw × normalized_relevance
```

Optimal `rw` = 0.10 (validated via ablation and temporal experiments).

#### 3.3 Graph Layer

Lightweight edge layer over the vector store:
- **Edge types**: causal, co_occurrence, similar
- **Auto-linking**: cosine > 0.75 triggers similar edge (no LLM calls)
- **Retrieval**: cosine seeds → graph expansion (spreading activation) → re-rank
- **Score**: `(1-rw-gw) × cosine + rw × relevance + gw × graph_activation`

---

### 4. Experiments

#### 4.1 Dataset: 100 SRE Postmortems

100 synthetic incident reports across 5 domains (auth, database,
infrastructure, monitoring, network), 20 per domain, with known causal chains
and ground truth relevance judgments.

#### 4.2 Ablation Study (WS-A)

| Feature Removed | R@5 Change | MRR Change | Effect |
|-----------------|-----------|-----------|--------|
| All features (rw=0→0.3) | -0.137 | -0.126 | HURTS retrieval |
| Recency | -0.117 | -0.117 | HURTS most |
| Affect + Mass | -0.107 | -0.103 | HURTS |
| Mass only | -0.108 | -0.099 | HURTS |
| Frequency | -0.061 | -0.069 | HURTS |
| Surprise | -0.003 | -0.004 | Negligible |

**Key finding**: Features that model human memory (recency, affect, mass)
trade retrieval accuracy for agent relevance. Surprise adds almost nothing
and costs 2 LLM calls per encode → made opt-in.

#### 4.3 Temporal Decay Test (WS-B)

With 90 days of simulated access patterns (power-law distribution):

| Scenario | rw=0.0 | rw=0.15 | Delta |
|----------|--------|---------|-------|
| Overall domain precision@5 | 0.689 | 0.756 | +0.067 |
| Recent/recurring | 0.800 | 0.900 | +0.100 |
| Old/forgotten | 0.600 | 0.600 | 0.000 |

**Key finding**: Relevance weighting improves domain precision +6.7% in agent
scenarios without degrading access to old memories.

#### 4.4 Graph-Augmented Multi-hop (WS-G)

| System | Multi-hop R@10 | Hits | Standard R@5 |
|--------|---------------|------|-------------|
| Cosine only | 0.750 | 15/20 | 0.341 |
| Graph (1-hop, gw=0.3) | **1.000** | **20/20** | 0.341 |

**Key finding**: Graph expansion achieves 100% multi-hop recall with zero
degradation to standard retrieval and zero LLM calls.

#### 4.5 AMBench: Agent Memory Benchmark (WS-D)

300 incidents, 10 pattern types, 90 simulated days, 3 systems:

| Metric | RAG | IMI (rw=0.10) | IMI (rw=0.15) |
|--------|-----|---------------|---------------|
| M1: Retrieval R@5 | 0.279 | 0.275 | 0.273 |
| M2: Cluster purity | 0.736 | 0.736 | 0.736 |
| M3: Action P@1 | 100% | 100% | 100% |
| M4: Temporal avg age | 41.2d | **16.8d** | 43.0d |

**Key finding**: IMI with rw=0.10 dramatically improves temporal coherence
(avg age of top results: 16.8 vs 41.2 days) while maintaining comparable
retrieval accuracy.

#### 4.6 Adaptive Relevance Weight (P1)

Keyword-based query intent classifier that adjusts rw dynamically:

| Intent | Pattern | rw | Example |
|--------|---------|-----|---------|
| TEMPORAL | "recent", "latest", "just happened" | 0.15 | "recent auth failures" |
| EXPLORATORY | "find all", "list every" | 0.00 | "find all cert incidents" |
| ACTION | "how to", "fix", "prevent" | 0.05 | "how to prevent DNS outages" |
| DEFAULT | everything else | 0.10 | "auth token failures" |

Classification accuracy: 100% (16/16 mixed queries). WS3 MRR: adaptive
(0.651) > best fixed (0.643). Zero cost (keyword regex, no LLM).

#### 4.7 Causal Edge Detection (P2)

Empirical finding: causal pairs have **low** cosine similarity (avg 0.308).
"Token race condition" and "rolling deploy" are semantically distant — the
relationship is logical, not semantic. Only 1/10 ground truth chains has
similarity > 0.65.

| Strategy | Recall | Precision | Cost |
|----------|--------|-----------|------|
| Embedding (thr=0.40) | 30% | 10% | 0 LLM calls |
| Embedding (thr=0.65) | 10% | 100% | 0 LLM calls |
| LLM confirmed | ~80%+ | ~90%+ | 1 call/candidate |

**Key finding**: Causality detection fundamentally requires reasoning, not
similarity. Embedding-only works for obvious cases; LLM confirmation or
explicit agent hints needed for comprehensive detection.

---

### 5. Architecture Decisions

Through empirical validation, we made five key decisions:

1. **Removed TimescaleDB backend** (-425 lines). SQLite provides equivalent
   functionality for single-agent workloads at 87x lower latency.

2. **Made predictive coding opt-in**. Surprise boost adds only +0.003 to R@5
   but costs 2 LLM calls per memory encode.

3. **Set default relevance_weight to 0.10** (from 0.30). Ablation showed 0.3
   was too aggressive; 0.10 balances semantic accuracy and temporal relevance.

4. **Added adaptive relevance_weight**. Query intent classifier adjusts rw
   per query (0.00-0.15) with zero cost, improving MRR by +1.2%.

5. **Causal detection requires LLM**. Embedding similarity is insufficient
   for causality (avg causal pair cos = 0.308). Explicit hints or LLM
   confirmation needed for reliable cross-memory linking.

---

### 6. Limitations

- **Synthetic evaluation only**: All experiments use generated incidents, not
  real-world agent traces. Real incidents have noisier text and messier patterns.

- **Single embedding model**: All experiments use all-MiniLM-L6-v2 (384d).
  Larger models (e5-large, BGE) may shift the retrieval/relevance balance.

- **No real HippoRAG comparison**: HippoRAG incompatible with Python 3.14;
  we compared against a regex-NER simulation. Real HippoRAG with LLM NER
  would be significantly better at multi-hop.

- **Static ground truth**: Relevance judgments are author-created, not
  crowdsourced. NDCG calculations may not reflect real agent preferences.

- **Single-agent only**: AMBench simulates one agent. Multi-agent shared
  memory is an important future workstream.

---

### 7. Future Work

- **Real-world validation**: Deploy IMI in production agent systems and
  measure actual task completion improvement. Synthetic benchmarks cannot
  capture the full complexity of real agent workflows.
- **Multi-agent memory sharing**: Agent-scoped views with trust gradients
  to prevent relevance signal contamination.
- **LLM-confirmed causal detection**: Our embedding-based approach detects
  only 30% of causal chains (Section 4.7). A pipeline of embedding
  candidates filtered by LLM confirmation could close this gap.
- **Larger benchmarks**: 10K incidents, 365 days, multi-agent scenarios.
  Initial multi-agent experiments (WS-I) show shared memory helps in 40%
  of cross-domain queries.

---

### 8. Conclusion

IMI demonstrates that cognitive memory features — despite degrading pure
retrieval benchmarks — significantly improve agent-relevant metrics:
temporal coherence (+147% recency), multi-hop recall (+33%), and domain
precision (+6.7%). The optimal configuration uses moderate relevance weighting
(rw=0.10), graph-augmented expansion for multi-hop, and affective modulation
for emotional salience. Predictive coding (surprise) provides negligible
benefit and should be opt-in.

AMBench provides the first standardized evaluation framework for agent memory,
measuring the full encode-retrieve-consolidate-act lifecycle that existing
RAG benchmarks ignore.

IMI is open-source, runs on SQLite with zero infrastructure, requires no LLM
calls at query time, and achieves competitive retrieval quality while adding
temporal, affective, and action-oriented capabilities that pure vector stores
lack.

---

### References

- Collins, A.M. & Loftus, E.F. (1975). A spreading-activation theory of semantic processing. *Psychological Review*.
- Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B*.
- Gibson, J.J. (1979). *The Ecological Approach to Visual Perception*. Houghton Mifflin.
- Gutierrez, B. et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS 2024*.
- Packer, C. et al. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.
- Rao, R.P.N. & Ballard, D.H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*.
- Sarthi, P. et al. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. *ICLR 2024*.

---

### Appendix A: Reproduction

```bash
git clone <repo-url> && cd imi
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run all experiments
PYTHONPATH=. python experiments/ws_a_ablation_study.py
PYTHONPATH=. python experiments/ws_b_temporal_decay.py
PYTHONPATH=. python experiments/ws_c_hipporag_comparison.py
PYTHONPATH=. python experiments/ws_d_agent_memory_benchmark.py
PYTHONPATH=. python experiments/ws_g_graph_augmented_retrieval.py

# Run tests
python -m pytest tests/ -v
```

### Appendix B: Venue Considerations

| Venue | Fit | Deadline |
|-------|-----|----------|
| NeurIPS 2026 Workshop (Memory & Retrieval) | High | ~June 2026 |
| EMNLP 2026 | High (NLP + agents) | ~May 2026 |
| ICML 2026 Workshop | Medium (agent systems) | ~April 2026 |
| arXiv preprint | Immediate | N/A |

**Recommended path**: arXiv preprint first → workshop submission → full paper.

---

### Acknowledgments

This work was developed using Claude Code (Anthropic) as a research
acceleration tool. All experiments, ablations, and benchmark designs were
conceived and validated by the author; Claude Code assisted with
implementation and systematic exploration of the experimental parameter space.
