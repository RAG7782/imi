# IMI Launch — Tweet Thread

---

**1/10**

Your AI agent forgets everything between sessions.

RAG doesn't fix it — it optimizes the wrong metric.

We built IMI: a cognitive memory system for agents with temporal decay, affordances, graph retrieval, and the first agent memory benchmark.

Thread 🧵

---

**2/10**

The problem: RAG finds the most *similar* past memory. Agents need the most *useful* one.

"Most similar incident" ≠ "most relevant for what I need to do right now."

Recency, emotional salience, and causal context don't appear in a cosine score.

---

**3/10**

We found a fundamental paradox:

Features that model human memory (recency, affect, mass) DEGRADE retrieval benchmarks but IMPROVE agent-relevant metrics.

RAG R@5: 0.341 → IMI: 0.304 (-3.7%)
Agent domain precision: 0.689 → 0.756 (+9.7%)
Temporal coherence: 41.2d → 16.8d avg age

---

**4/10**

IMI architecture in one line:

score = (1-rw) × cosine + rw × (recency × frequency × mass × surprise)

Optimal rw = 0.10. Not 0.0 (pure RAG). Not 0.3 (too aggressive). Validated across 90-day simulations with 300 incidents.

---

**5/10**

Graph-augmented multi-hop: 100% recall (20/20 queries), zero LLM calls.

Cosine alone: 75% (15/20).

Causal chains span semantically distant memories. "Token race condition" and "rolling deploy" score as unrelated by embedding — but they cause each other. A BFS over causal edges finds what cosine misses.

---

**6/10**

Affordances: agents ask "what can I DO?" not "what do I KNOW?"

```python
results = space.search_affordances("db pool exhausted")
# → ["increase pool size", "check for leaks",
#    "add circuit breaker", "review slow queries"]
```

Every encoded memory extracts action potentials. Retrieval returns actions, not just text.

---

**7/10**

IMI ships with an MCP server — plug it into any Claude agent in minutes.

Zero infrastructure: SQLite only (87x faster than TimescaleDB for single-agent workloads, -425 lines of infra code removed).

Zero LLM calls at query time. All scoring is arithmetic.

---

**8/10**

Introducing AMBench — the first standardized benchmark for agent memory.

5 metrics: retrieval, consolidation purity, action precision, temporal coherence, learning curve.

RAG benchmarks ignore time. AMBench simulates 90 days, 300 incidents, 10 recurring patterns.

Submit your system. Compare honestly.

---

**9/10**

Honest limitations:

- All experiments on synthetic data. Real incidents are messier.
- Causal edge detection needs LLM (avg causal pair cosine = 0.308 — semantically distant)
- Single embedding model tested (MiniLM 384d)
- Relevance judgments are author-created, not crowdsourced

We're measuring the right thing. The numbers are directional.

---

**10/10**

Try it:

```
pip install imi-memory
```

```python
space = IMISpace()
space.encode("Auth timeout — cert rotation again")
results = space.navigate("auth service degraded")
```

GitHub: [link]
Paper draft: docs/paper-draft.md
AMBench: experiments/ws_d_agent_memory_benchmark.py

Star if useful. Issues welcome. 🙏
