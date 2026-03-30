# Your AI agent forgets everything between sessions. Here's what we built to fix it.

Not in a dramatic way. It just starts fresh every time. No memory of the incident it fixed last Tuesday. No sense that the same auth timeout has happened three times this week. No understanding that the last database upgrade caused a cascade failure.

This is the state of AI agent memory in 2025. And the obvious fix — just use RAG — turns out to be subtly wrong.

---

## The problem with RAG for agent memory

RAG is great for question answering. You embed a document corpus, do cosine similarity at query time, and retrieve the most semantically similar chunks. It works remarkably well for "what does the docs say about X?"

But agents don't ask questions. Agents act. And for agents, "most similar" is often not "most useful."

Consider an SRE agent responding to a Kubernetes pod crash. The most semantically similar past incident might be a pod crash from eight months ago — similar cause, same domain, high cosine score. But what the agent actually needs is the incident from three days ago that triggered an almost identical cascade, or the memory that links this symptom to a known deployment pattern.

RAG optimizes for retrieval accuracy. Agents need decision relevance. Those are different metrics.

We spent several months systematically measuring this gap. The findings turned into a system called IMI (Integrated Memory Intelligence) — and a benchmark called AMBench to make the comparison honest.

---

## The insight: the Retrieval vs. Relevance Paradox

Here is the central finding from our ablation study:

> Features that model human memory — recency, emotional salience, gravitational mass — **degrade** pure retrieval benchmarks but **improve** agent-relevant metrics.

We ran 15 queries over 100 synthetic SRE incident memories, varying a single `relevance_weight` parameter (`rw`) that controls how much temporal/affective signal gets mixed into the cosine score:

| Scenario | Metric | RAG (`rw=0`) | IMI (`rw=0.1`) | Delta |
|---|---|---|---|---|
| Pure retrieval | R@5 | **0.341** | 0.304 | -0.037 |
| Agent temporal queries | DomPrec@5 | 0.689 | **0.756** | +0.067 |
| Recent incident recall | DomPrec@5 | 0.800 | **0.900** | +0.100 |
| Multi-hop reasoning | R@10 | 0.750 | **1.000** | +0.250 |
| Temporal coherence | Avg age of top-5 results | 41.2 days | **16.8 days** | -24.4 days |

The tradeoff is real: IMI loses 3.7% on pure retrieval while gaining 6.7–10% on the metrics that actually matter for an acting agent. The temporal coherence improvement is stark — top results shift from 41 days old on average to 17 days. For recurring operational patterns, that's not a minor win.

Notably, the sweet spot is `rw=0.10`, not higher. At `rw=0.30` (our original default), retrieval drops 40% — too aggressive. The ablation found the right balance empirically.

---

## What IMI does differently

### 1. Memories have cognitive features, not just embeddings

Every `MemoryNode` in IMI carries features beyond the embedding vector:

```python
from imi import IMISpace

space = IMISpace()

# Encode an incident with full cognitive features
node = space.encode(
    "Auth service: token validation latency spike to 2400ms. "
    "Correlated with cert rotation job. Third occurrence this quarter.",
    domain="auth",
    tags=["latency", "cert-rotation", "recurring"]
)

# The node now has:
# - node.affect.salience     → high (recurring pattern)
# - node.affect.valence      → negative
# - node.mass                → weighted by encoding strength
# - node.summaries           → orbital / medium / detailed zoom levels
# - node.affordances         → ["check cert expiry", "review rotation schedule"]
```

These features flow into the relevance score:

```
relevance = recency × (1 + frequency) × mass × surprise_boost
score     = (1 - rw) × cosine_similarity + rw × normalized_relevance
```

Memories that are recent, frequently accessed, and emotionally salient score higher — not because of semantic similarity, but because they are more likely to be decision-relevant right now.

### 2. Graph-augmented retrieval for multi-hop chains

Causal chains in operational incidents span multiple memories. A token race condition leading to a service restart leading to a cascade failure — those three memories are semantically distant but causally linked.

IMI includes a lightweight graph layer with spreading activation (inspired by Collins & Loftus 1975). After the initial cosine retrieval, it expands into adjacent nodes via causal and co-occurrence edges:

```python
# Standard retrieval
results = space.navigate("cert rotation failure", top_k=5)

# Graph-augmented retrieval (follows causal chains)
results = space.search_with_expansion("cert rotation failure", top_k=5)
```

The result: 100% multi-hop recall (20/20 queries) versus 75% for cosine-only — with zero LLM calls at query time and zero degradation to standard retrieval. Auto-linking triggers on cosine > 0.75, so common patterns build edges naturally as the agent operates.

### 3. Affordances: "what can I do?" not just "what do I know?"

RAG returns text. IMI returns actions. Each encoded memory extracts action potentials — concrete steps an agent could take given that memory:

```python
# Retrieve with affordances
results = space.search_affordances("database connection pool exhausted")

for r in results:
    print(r.node.seed[:80])
    print("Actions:", r.node.affordances)
    # → ["increase pool size in config", "check for connection leaks",
    #    "add circuit breaker", "review slow query log"]
```

This shifts the agent's retrieval output from "here is related information" to "here is what this situation enables you to do" — an affordance-first framing borrowed from Gibson's ecological psychology.

### 4. Zero infrastructure, zero LLM calls at query time

IMI runs on SQLite with WAL mode. We initially built a TimescaleDB backend and removed it after benchmarking — SQLite was 87x faster for single-agent workloads and required no Docker, no migration tooling, nothing to operate.

```bash
pip install imi-memory

# That's it. No Postgres, no Redis, no vector database service.
```

At query time, zero LLM calls. The temporal/affective scoring is pure arithmetic. The graph expansion is a BFS over a local adjacency structure. Fast enough to run inline in an agent's tool call without noticeable latency.

---

## Results across the full benchmark

AMBench (described below) runs 300 incidents over 90 simulated days with 10 recurring patterns. Here is how IMI compares against vanilla RAG:

| Metric | RAG | IMI (rw=0.10) |
|---|---|---|
| M1: Retrieval R@5 | 0.279 | 0.275 (≈) |
| M2: Cluster purity | 0.736 | 0.736 (≈) |
| M3: Action P@1 | 100% | 100% |
| M4: Temporal coherence (avg age top-5) | 41.2 days | **16.8 days** |
| Multi-hop recall R@10 | 75% | **100%** |

The retrieval and cluster purity metrics are equivalent — IMI does not regress on the things RAG is already good at. The gains appear in the dimensions that matter for agents operating over time.

---

## How to try it

```bash
pip install imi-memory
```

Three lines to start:

```python
from imi import IMISpace

space = IMISpace()  # SQLite-backed, no config needed
space.encode("Deploy failed: image pull timeout on node-3. Fixed by pre-pulling.")
results = space.navigate("container startup failure", top_k=5)
```

To use graph-augmented retrieval:

```python
# Build edges as you go
space.encode("Auth timeout spike linked to cert rotation", tags=["auth", "cert"])
space.encode("Cert rotation triggered by scheduled job", tags=["cert", "cron"])

# Multi-hop expansion finds the cert→auth chain automatically
results = space.search_with_expansion("auth service degraded")
```

Full experiment reproduction:

```bash
git clone <repo-url> && cd imi
pip install -e .
PYTHONPATH=. python experiments/ws_d_agent_memory_benchmark.py
PYTHONPATH=. python experiments/ws_g_graph_augmented_retrieval.py
```

---

## AMBench: a benchmark for agent memory

One of the things that surprised us during this work: there is no standard benchmark for agent memory systems. RAG benchmarks (BEIR, MTEB) measure retrieval quality on static corpora. They have no notion of time, consolidation, action, or learning.

AMBench fills that gap. It evaluates the full encode-retrieve-consolidate-act lifecycle across five dimensions:

1. **Retrieval accuracy** — standard R@K over a growing corpus
2. **Consolidation quality** — cluster purity after the memory maintenance pass
3. **Action relevance** — affordance precision@1 (does the top action make sense?)
4. **Temporal coherence** — recency preservation in top results over time
5. **Learning curve** — does retrieval quality improve as the agent accumulates experience?

The benchmark ships with 300 incidents, 10 pattern types, 90-day simulation, and three baselines. It is designed to be extensible — submit your own agent memory system, your own incident corpus, or extend the 90-day horizon to 365.

We think AMBench is publishable as an independent contribution. If you work on agent memory and want to compare against it, the dataset and evaluation code are in `experiments/ws_d_agent_memory_benchmark.py`.

---

## What does not work yet (being honest)

- All experiments are on synthetic data. Real incidents have noisier text, ambiguous causality, and messy tagging. We have not deployed IMI in a production agent system yet.
- Causal edge detection via embeddings is weak. Causal pairs have low cosine similarity (avg 0.308 in our dataset) — meaning "token race condition" and "rolling deploy failure" score as unrelated. For comprehensive multi-hop, you either need explicit hints at encode time or an LLM pass to detect causality. We are working on this.
- All experiments use `all-MiniLM-L6-v2` (384-dimensional). Larger models may shift the retrieval/relevance balance.
- The benchmark uses author-created relevance judgments, not crowdsourced. NDCG numbers should be treated as directional.

---

## Call to action

If you build AI agents and memory is a problem you have hit, we would appreciate a GitHub star and a try. If you work on agent benchmarks and think AMBench could be more useful, open an issue — we are actively extending it.

If you want to compare your memory system against IMI on AMBench, the instructions are in `docs/ambench-contribute.md` (coming soon). The first entries will be RAG, IMI Lite-B (cosine + zoom + affordances, no temporal features), and full IMI.

The core claim we are making is not that IMI is the best memory system. It is that the field has been measuring the wrong thing — and that once you measure what agents actually need, the design space opens up considerably.

---

*IMI is open source. Paper draft, all experiment code, and the benchmark are in the repository. Feedback welcome.*
