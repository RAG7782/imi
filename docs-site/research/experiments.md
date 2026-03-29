# Experiments

All experiments live in `/Users/renatoaparegomes/experimentos/imi/experiments/`. Each is a self-contained Python script.

## Running experiments

```bash
cd /Users/renatoaparegomes/experimentos/imi

# Install dependencies
pip install -e ".[all,dev]"

# Run any experiment
python experiments/ws_a_ablation_study.py
```

Most experiments generate a summary table printed to stdout. Some produce figures saved to `/docs/figures/`.

---

## Workstream experiments

| File | Description | Runtime |
|------|-------------|---------|
| `ws3_validation_framework.py` | Base validation: 100 synthetic postmortems, 15 queries, R@5/MRR baseline | ~2 min |
| `ws_a_ablation_study.py` | Feature contribution analysis: ablate each feature (recency, affect, mass, surprise, frequency) individually and measure R@5/MRR impact | ~5 min |
| `ws_b_temporal_decay.py` | 90-day temporal simulation with power-law access patterns — validates rw=0.15 for temporal queries | ~3 min |
| `ws_c_hipporag_comparison.py` | IMI vs HippoRAG-Sim (regex-NER): multi-hop recall comparison across 20 queries | ~2 min |
| `ws_d_agent_memory_benchmark.py` | AMBench — 300 incidents, 10 pattern types, 90 simulated days, 5 metrics (R@5, cluster purity, affordance precision, temporal coherence, learning curve) | ~8 min |
| `ws_g_graph_augmented_retrieval.py` | Graph expansion validation: 20 multi-hop queries over 5 causal chains, graph vs cosine-only | ~2 min |
| `ws_i_expanded_benchmark.py` | Full expanded benchmark: multi-agent shared memory, cross-domain queries, 365-day simulation with 600 incidents | ~15 min |
| `ws2_full_vs_liteb_benchmark.py` | IMI full vs IMI Lite-B: feature cost/benefit comparison | ~3 min |
| `ws4_imi_vs_rag_reranker.py` | IMI vs cross-encoder reranker: quality vs latency trade-off | ~4 min |
| `ws4_threshold_analysis.py` | Sensitivity analysis on similarity threshold for graph auto-linking | ~2 min |

## Priority experiments

| File | Description | Key finding |
|------|-------------|------------|
| `p1_adaptive_rw.py` | Validates the keyword-based intent classifier on 16 mixed queries — accuracy and MRR vs fixed rw | 100% classification accuracy, +1.2% MRR vs best fixed |
| `p2_causal_detection.py` | Tests embedding-based causal edge detection — cosine thresholds vs ground truth causal pairs | Causal pairs have avg cosine 0.308; embedding alone gets only 30% recall |
| `p3_paper_figures.py` | Generates all figures for the paper (R@5 vs rw, temporal coherence, graph expansion) | Saves to `/docs/figures/` |

## Running the full benchmark suite

```bash
# Run all workstream experiments in sequence
for exp in ws3 ws_a ws_b ws_c ws_d ws_g ws_i; do
    echo "=== Running $exp ==="
    python experiments/${exp}*.py
done
```

## Reproducing key results

### Ablation study (rw sweet spot)

```bash
python experiments/ws_a_ablation_study.py
```

Expected output:

```
rw=0.00: R@5=0.341, MRR=0.702
rw=0.10: R@5=0.304, MRR=0.673
rw=0.15: R@5=0.289, MRR=0.659
rw=0.30: R@5=0.204, MRR=0.576
```

### Graph multi-hop recall

```bash
python experiments/ws_g_graph_augmented_retrieval.py
```

Expected output:

```
Cosine only:    multi-hop R@10 = 0.750 (15/20)
IMI + Graph:    multi-hop R@10 = 1.000 (20/20)
Standard R@5:   unchanged at 0.341
```

### Temporal coherence (AMBench)

```bash
python experiments/ws_d_agent_memory_benchmark.py
```

Expected output:

```
M4 Temporal coherence:
  RAG (rw=0):    avg_age_top5 = 41.2 days
  IMI (rw=0.10): avg_age_top5 = 16.8 days
```
