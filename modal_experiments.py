"""IMI Experiments on Modal — Cloud compute for benchmark tuning.

Runs 3 experiment suites:
1. L1 Tuning Sweep: find optimal max_facts for TieredRecall L1 coverage ≥0.90
2. SD-Retrieval Domain-Diverse: test DS-d scoring with varied domain text
3. Full Benchmark Suite: official results on consistent hardware

Usage:
    modal run modal_experiments.py                    # Run all experiments
    modal run modal_experiments.py::run_l1_sweep      # L1 tuning only
    modal run modal_experiments.py::run_sd_diverse    # SD diverse only
    modal run modal_experiments.py::run_full_bench    # Full suite only
    modal run modal_experiments.py::run_all           # All three

Results are saved to /results/ volume and printed to stdout.
"""

from __future__ import annotations

import modal
import json
import time

# ── Modal App & Image ─────────────────────────────────

app = modal.App("imi-experiments")

imi_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # CPU-only torch (much smaller, no CUDA)
        "torch==2.6.0+cpu",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "numpy>=1.26.0",
        "sentence-transformers>=3.0.0",
        "anthropic>=0.80.0",
    )
    .run_commands(
        # Pre-cache the embedding model during build
        "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')\""
    )
    .add_local_dir(
        "/Users/renatoaparegomes/experimentos/tools/imi/imi",
        remote_path="/root/imi_pkg/imi",
    )
    .add_local_dir(
        "/Users/renatoaparegomes/experimentos/tools/imi/tests",
        remote_path="/root/imi_pkg/tests",
    )
)

results_volume = modal.Volume.from_name("imi-experiment-results", create_if_missing=True)


# ── Experiment 1: L1 Tuning Sweep ────────────────────

@app.function(
    image=imi_image,
    timeout=1200,
    memory=4096,
)
def l1_tuning_single(max_facts: int, n_incidents: int = 300, n_days: int = 90, seed: int = 42):
    """Run TieredRecall with a specific max_facts value."""
    import sys
    sys.path.insert(0, "/root/imi_pkg")

    from imi.benchmark.tiered_recall import TieredRecall
    from imi.tiering import generate_l1

    # Monkey-patch generate_l1 default max_facts
    import imi.benchmark.tiered_recall as tr_module

    # Store original
    original_generate_l1 = tr_module.generate_l1

    def patched_generate_l1(nodes, **kwargs):
        kwargs["max_facts"] = max_facts
        return original_generate_l1(nodes, **kwargs)

    tr_module.generate_l1 = patched_generate_l1

    bench = TieredRecall(n_incidents=n_incidents, n_days=n_days, seed=seed)
    results = bench.run(system_name=f"IMI-L1-{max_facts}")

    return {
        "max_facts": max_facts,
        "full_r5": results.full_r5,
        "l1_coverage": results.l1_coverage,
        "tier_ratio": results.tier_ratio,
        "n_queries": results.n_queries,
        "duration_s": results.duration_s,
    }


@app.function(
    image=imi_image,
    volumes={"/results": results_volume},
    timeout=900,
)
def run_l1_sweep():
    """Sweep max_facts from 3 to 20 to find optimal L1 coverage."""
    print("=" * 60)
    print("EXPERIMENT 1: L1 Tuning Sweep (max_facts 3→20)")
    print("=" * 60)

    # Launch all in parallel via .map()
    max_facts_values = list(range(3, 21))
    results = []

    for result in l1_tuning_single.map(max_facts_values):
        results.append(result)
        ratio = result["tier_ratio"]
        status = "✓" if ratio >= 0.90 else "✗"
        print(f"  max_facts={result['max_facts']:2d}  L1={result['l1_coverage']:.3f}  ratio={ratio:.3f} {status}")

    # Sort by tier_ratio
    results.sort(key=lambda r: r["tier_ratio"], reverse=True)

    # Find minimum max_facts that achieves ≥0.90
    optimal = None
    for r in sorted(results, key=lambda x: x["max_facts"]):
        if r["tier_ratio"] >= 0.90:
            optimal = r
            break

    print(f"\n--- RESULTS ---")
    if optimal:
        print(f"Optimal: max_facts={optimal['max_facts']} → tier_ratio={optimal['tier_ratio']:.3f}")
    else:
        best = results[0]
        print(f"Target 0.90 not reached. Best: max_facts={best['max_facts']} → tier_ratio={best['tier_ratio']:.3f}")

    # Save results
    output = {
        "experiment": "l1_tuning_sweep",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sweep_range": [3, 20],
        "n_incidents": 300,
        "optimal": optimal,
        "all_results": results,
    }

    with open("/results/l1_sweep_results.json", "w") as f:
        json.dump(output, f, indent=2)
    results_volume.commit()

    return output


@app.function(
    image=imi_image,
    volumes={"/results": results_volume},
    timeout=1800,
)
def run_l1_sweep_at_scale(n_incidents: int = 1000, max_range: int = 30):
    """Sweep max_facts at larger scale to find optimal L1 coverage.

    Usage:
        modal run modal_experiments.py::run_l1_sweep_at_scale
    """
    print("=" * 60)
    print(f"L1 SWEEP @ SCALE — {n_incidents} incidents, max_facts 3→{max_range}")
    print("=" * 60)

    max_facts_values = list(range(3, max_range + 1))
    n_list = [n_incidents] * len(max_facts_values)
    results = []

    for result in l1_tuning_single.map(max_facts_values, n_list):
        results.append(result)
        ratio = result["tier_ratio"]
        status = "✓" if ratio >= 0.90 else "✗"
        print(f"  max_facts={result['max_facts']:2d}  L1={result['l1_coverage']:.3f}  ratio={ratio:.3f} {status}")

    results.sort(key=lambda r: r["tier_ratio"], reverse=True)

    optimal = None
    for r in sorted(results, key=lambda x: x["max_facts"]):
        if r["tier_ratio"] >= 0.90:
            optimal = r
            break

    print(f"\n--- RESULTS ---")
    if optimal:
        print(f"Optimal: max_facts={optimal['max_facts']} → tier_ratio={optimal['tier_ratio']:.3f}")
    else:
        best = results[0]
        print(f"Target 0.90 not reached. Best: max_facts={best['max_facts']} → tier_ratio={best['tier_ratio']:.3f}")

    output = {
        "experiment": "l1_tuning_sweep_at_scale",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sweep_range": [3, max_range],
        "n_incidents": n_incidents,
        "optimal": optimal,
        "all_results": sorted(results, key=lambda x: x["max_facts"]),
    }

    with open(f"/results/l1_sweep_{n_incidents}_results.json", "w") as f:
        json.dump(output, f, indent=2)
    results_volume.commit()

    return output


# ── Experiment 2: SD-Retrieval Domain-Diverse ─────────

DOMAIN_INCIDENTS = {
    "legal": [
        "Contract breach: Party A failed to deliver goods within 30-day deadline per clause 4.2. Force majeure defense invoked citing supply chain disruption. Arbitration clause triggers ICC rules.",
        "LGPD violation: company processed biometric data without explicit consent. DPA investigation opened. Risk of 2% revenue fine under Art. 52.",
        "Tax audit: Federal Revenue challenged transfer pricing methodology. Arm's length principle dispute on intercompany royalties. CARF precedent favorable.",
        "Labor claim: overtime calculation error affecting 200 employees. Class action filed. Potential exposure R$2.4M including moral damages.",
        "Corporate restructuring: minority shareholder squeeze-out at fair value. Appraisal rights exercised. Judicial valuation pending.",
    ],
    "medical": [
        "Patient presented with acute chest pain, elevated troponin, ST-elevation in leads II, III, aVF. STEMI protocol activated. PCI within 45 minutes. Drug-eluting stent deployed to RCA.",
        "Type 2 diabetes management: HbA1c 9.2% despite metformin 2g/day. Added GLP-1 agonist (semaglutide 0.5mg weekly). Renal function eGFR 62, monitor for decline.",
        "Pediatric seizure: 8-year-old with generalized tonic-clonic lasting 4 minutes. EEG showed bilateral spike-wave at 3Hz. Childhood absence epilepsy diagnosed. Ethosuximide started.",
        "Post-surgical infection: hip replacement day 5, erythema and purulent drainage. Blood cultures: MRSA. Vancomycin IV initiated. Surgical washout scheduled.",
        "Prenatal screening: NIPT positive for Trisomy 21. Amniocentesis confirmed 47,XX,+21. Genetic counseling provided. Parents informed of developmental expectations.",
    ],
    "engineering": [
        "Kubernetes pod crash loop: OOMKilled at 512Mi limit. Memory profiling showed goroutine leak in gRPC connection pool. Fix: add MaxRecvMsgSize and idle connection timeout.",
        "Database migration failure: ALTER TABLE on 500M row table caused 3-hour lock. Rollback triggered failover. Solution: pt-online-schema-change with chunk size 10000.",
        "SSL certificate chain incomplete: intermediate CA missing from nginx config. Mobile clients failed validation. Desktop browsers cached intermediate. Fix: concatenate full chain.",
        "Load balancer health check: TCP check passed but HTTP 503. App started but not ready. Solution: implement /readyz endpoint checking database connection pool.",
        "Data pipeline backpressure: Kafka consumer lag 2M messages. Consumer group rebalance storm. Root cause: long GC pauses. Fix: tune G1GC with MaxGCPauseMillis=50.",
    ],
    "financial": [
        "Portfolio rebalancing: equity allocation drifted to 72% (target 60%). Sell S&P 500 ETF, buy aggregate bond ETF. Tax-loss harvesting opportunity on international small-cap position.",
        "Credit risk assessment: corporate borrower leverage ratio 4.5x EBITDA, interest coverage 2.1x. Moody's downgrade watch. Recommend covenant tightening on revolving facility.",
        "Derivatives pricing: Black-Scholes model inadequate for barrier options. Switch to Monte Carlo with 100K paths. Variance reduction via antithetic variates. Greeks computed via bumping.",
        "AML alert: unusual wire transfer pattern — 15 transactions just below reporting threshold within 72 hours. Structuring suspected. SAR filed. Account under enhanced monitoring.",
        "M&A valuation: DCF with WACC 9.2%, terminal growth 2.5%. Comparable transactions suggest 8-10x EBITDA. Synergy value estimated at 15% cost savings over 3 years.",
    ],
}


@app.function(
    image=imi_image,
    timeout=600,
    memory=2048,
)
def sd_retrieval_domain(domain: str, incidents: list[str], seed: int = 42):
    """Run SD-Retrieval on a single domain's incidents."""
    import sys
    sys.path.insert(0, "/root/imi_pkg")

    import numpy as np
    from imi.store import VectorStore
    from imi.node import MemoryNode, AffectiveTag
    from imi.embedder import SentenceTransformerEmbedder
    from imi.dialect import compute_ds_d

    embedder = SentenceTransformerEmbedder()
    store = VectorStore()

    ds_d_scores = []

    for i, text in enumerate(incidents):
        emb = embedder.embed(text)
        ds_d = compute_ds_d(text, embedder)
        ds_d_scores.append(ds_d)

        node = MemoryNode(
            seed=text,
            summary_medium=text[:120],
            embedding=emb,
            tags=[domain],
            affect=AffectiveTag(salience=0.7, valence=0.5, arousal=0.5),
        )
        node.id = f"{domain}_{i}"
        node.ds_d = ds_d
        store.add(node)

    # Query each incident against the store
    hits_baseline = 0
    hits_sd = 0
    n_queries = len(incidents)

    for i, text in enumerate(incidents):
        query_emb = embedder.embed(text)

        # Baseline: pure cosine
        results = store.search(query_emb, top_k=5, relevance_weight=0.0)
        baseline_tags = [n.tags[0] if n.tags else "" for n, _ in results]
        if domain in baseline_tags[:3]:
            hits_baseline += 1

        # SD-weighted: re-rank by DS-d
        results_with_sd = store.search(query_emb, top_k=10, relevance_weight=0.0)
        reranked = []
        for node, cosine_score in results_with_sd:
            sd_score = cosine_score * 0.7 + node.ds_d * 0.3
            reranked.append((node, sd_score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        sd_tags = [n.tags[0] if n.tags else "" for n, _ in reranked[:5]]
        if domain in sd_tags[:3]:
            hits_sd += 1

    return {
        "domain": domain,
        "n_incidents": len(incidents),
        "baseline_precision": hits_baseline / n_queries,
        "sd_precision": hits_sd / n_queries,
        "improvement": (hits_sd - hits_baseline) / max(n_queries, 1),
        "ds_d_mean": float(np.mean(ds_d_scores)),
        "ds_d_std": float(np.std(ds_d_scores)),
        "ds_d_range": [float(min(ds_d_scores)), float(max(ds_d_scores))],
    }


@app.function(
    image=imi_image,
    volumes={"/results": results_volume},
    timeout=900,
)
def run_sd_diverse():
    """Run SD-Retrieval across 4 diverse domains."""
    print("=" * 60)
    print("EXPERIMENT 2: SD-Retrieval Domain-Diverse (4 domains)")
    print("=" * 60)

    domains = list(DOMAIN_INCIDENTS.keys())
    incidents_list = [DOMAIN_INCIDENTS[d] for d in domains]

    results = []
    for result in sd_retrieval_domain.map(domains, incidents_list):
        results.append(result)
        imp = result["improvement"] * 100
        print(f"  {result['domain']:12s}  DS-d={result['ds_d_mean']:.3f}±{result['ds_d_std']:.3f}  improvement={imp:+.1f}%")

    # Cross-domain analysis
    all_means = [r["ds_d_mean"] for r in results]
    cross_domain_std = max(all_means) - min(all_means)

    output = {
        "experiment": "sd_retrieval_domain_diverse",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "domains": domains,
        "cross_domain_ds_d_range": cross_domain_std,
        "per_domain": results,
    }

    print(f"\nCross-domain DS-d range: {cross_domain_std:.3f}")
    print(f"(Higher range = DS-d discriminates between domains)")

    with open("/results/sd_diverse_results.json", "w") as f:
        json.dump(output, f, indent=2)
    results_volume.commit()

    return output


# ── Experiment 3: Full Benchmark Suite ────────────────

@app.function(
    image=imi_image,
    timeout=3600,
    memory=8192,
    secrets=[modal.Secret.from_name("anthropic-api")],
)
def run_full_bench(n_incidents: int = 300, n_days: int = 180):
    """Run all 6 benchmarks on Modal with consistent compute."""
    import sys
    sys.path.insert(0, "/root/imi_pkg")

    print("=" * 60)
    print(f"EXPERIMENT 3: Full Benchmark Suite ({n_incidents} incidents, {n_days} days)")
    print("=" * 60)

    results = {}

    # AMBench
    print("\n[1/6] AMBench...")
    from imi.benchmark.ambench import AMBench
    bench = AMBench(n_incidents=n_incidents, n_days=min(n_days, 90), seed=42)
    r = bench.run(system_name="IMI", relevance_weight=0.10)
    results["ambench"] = r.to_dict()
    print(f"  R@5={r.retrieval_r5:.3f}  Purity={r.cluster_purity_score:.3f}")

    # TieredRecall
    print("\n[2/6] TieredRecall...")
    from imi.benchmark.tiered_recall import TieredRecall
    bench = TieredRecall(n_incidents=n_incidents, n_days=min(n_days, 90), seed=42)
    r = bench.run(system_name="IMI")
    results["tiered_recall"] = r.to_dict()
    print(f"  Full={r.full_r5:.3f}  L1={r.l1_coverage:.3f}  Ratio={r.tier_ratio:.3f}")

    # TieredEfficiency
    print("\n[3/6] TieredEfficiency...")
    from imi.benchmark.tiered_efficiency import TieredEfficiency
    bench = TieredEfficiency(n_incidents=n_incidents, n_days=min(n_days, 90), seed=42, n_sessions=50)
    r = bench.run(system_name="IMI")
    results["tiered_efficiency"] = r.to_dict()
    print(f"  L0+L1={r.l0_l1_tokens}tok  Under200={r.pct_under_200:.0%}")

    # CrossSession
    print("\n[4/6] CrossSession...")
    from imi.benchmark.cross_session import CrossSession
    bench = CrossSession(n_incidents=min(n_incidents, 100), n_days=min(n_days, 30), n_sessions=30, seed=42)
    r = bench.run(system_name="IMI")
    results["cross_session"] = r.to_dict()
    print(f"  Retention={r.retention_rate:.3f}  Consolidated={r.patterns_consolidated}")

    # SDRetrieval
    print("\n[5/6] SDRetrieval...")
    from imi.benchmark.sd_retrieval import SDRetrieval
    bench = SDRetrieval(n_incidents=n_incidents, n_days=min(n_days, 90), seed=42)
    r = bench.run(system_name="IMI")
    results["sd_retrieval"] = r.to_dict()
    print(f"  Baseline={r.baseline_r5:.3f}  SD={r.sd_r5:.3f}  DS-d={r.ds_d_mean:.3f}")

    # LongMemEval
    print("\n[6/6] LongMemEval...")
    from imi.benchmark.longmem_eval import LongMemEval
    bench = LongMemEval(n_incidents=n_incidents, n_days=max(n_days, 180), seed=42)
    r = bench.run(system_name="IMI")
    results["longmem_eval"] = r.to_dict()
    print(f"  Recent={r.recent_r5:.3f}  Mid={r.mid_r5:.3f}  Old={r.old_r5:.3f}  Overall={r.overall_r5:.3f}")

    # FederatedRecall
    print("\n[7/7] FederatedRecall...")
    from imi.benchmark.federated_recall import FederatedRecall
    bench = FederatedRecall(n_incidents=n_incidents, n_days=min(n_days, 90), seed=42)
    r = bench.run(system_name="IMI")
    results["federated_recall"] = r.to_dict()
    print(f"  Isolated={r.isolated_r5:.3f}  Federated={r.federated_r5:.3f}  Boost={r.federation_boost:+.3f}")

    return results


# ── Orchestrator ──────────────────────────────────────

@app.function(
    image=imi_image,
    volumes={"/results": results_volume},
    timeout=3600,
)
def run_all():
    """Run all 3 experiments and save consolidated results."""
    print("╔" + "═" * 58 + "╗")
    print("║  IMI EXPERIMENTS — Modal Cloud Compute                    ║")
    print("╚" + "═" * 58 + "╝")
    print()

    all_results = {}

    # Experiment 1: L1 Sweep
    print("▸ Launching L1 Tuning Sweep...")
    l1_results = run_l1_sweep.remote()
    all_results["l1_sweep"] = l1_results

    # Experiment 2: SD Diverse
    print("\n▸ Launching SD-Retrieval Domain-Diverse...")
    sd_results = run_sd_diverse.remote()
    all_results["sd_diverse"] = sd_results

    # Experiment 3: Full Bench (default 300)
    print("\n▸ Launching Full Benchmark Suite (300 incidents)...")
    bench_results = run_full_bench.remote()
    all_results["full_benchmark"] = bench_results

    # Consolidated output
    consolidated = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "platform": "Modal",
            "imi_version": "0.2.0+L0L3+SDE",
        },
        **all_results,
    }

    with open("/results/consolidated_results.json", "w") as f:
        json.dump(consolidated, f, indent=2)
    results_volume.commit()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("Results saved to modal volume: imi-experiment-results")
    print("=" * 60)

    return consolidated


@app.function(
    image=imi_image,
    volumes={"/results": results_volume},
    timeout=7200,
)
def run_scale_comparison():
    """Run full benchmarks at 3 scales (300, 500, 1000) for paper-quality results.

    Usage:
        modal run modal_experiments.py::run_scale_comparison
    """
    print("╔" + "═" * 58 + "╗")
    print("║  IMI SCALE COMPARISON — 300 / 500 / 1000 incidents        ║")
    print("╚" + "═" * 58 + "╝")
    print()

    scales = [300, 500, 1000]

    # Launch all 3 scales in parallel
    handles = []
    for n in scales:
        print(f"▸ Launching scale {n} incidents...")
        handles.append(run_full_bench.spawn(n_incidents=n, n_days=180))

    # Collect results
    scale_results = {}
    for handle, n in zip(handles, scales):
        result = handle.get()
        scale_results[f"n{n}"] = result
        print(f"\n✓ Scale {n} complete")

        # Print summary per scale
        for bench_name, bench_data in result.items():
            key_metric = _extract_key_metric(bench_name, bench_data)
            print(f"  {bench_name}: {key_metric}")

    # Compute degradation analysis
    degradation = {}
    if "n300" in scale_results and "n1000" in scale_results:
        for bench_name in scale_results["n300"]:
            m300 = _extract_numeric_metric(bench_name, scale_results["n300"][bench_name])
            m1000 = _extract_numeric_metric(bench_name, scale_results["n1000"][bench_name])
            if m300 is not None and m1000 is not None and m300 > 0:
                degradation[bench_name] = {
                    "n300": m300,
                    "n1000": m1000,
                    "delta": m1000 - m300,
                    "pct_change": (m1000 - m300) / m300 * 100,
                }

    consolidated = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "platform": "Modal",
            "imi_version": "0.2.0+L0L3+SDE",
            "experiment": "scale_comparison",
            "scales": scales,
        },
        "results": scale_results,
        "degradation_300_vs_1000": degradation,
    }

    with open("/results/scale_comparison_results.json", "w") as f:
        json.dump(consolidated, f, indent=2)
    results_volume.commit()

    print("\n" + "=" * 60)
    print("SCALE COMPARISON COMPLETE")
    if degradation:
        print("\nDegradation 300→1000:")
        for bench, d in degradation.items():
            print(f"  {bench}: {d['pct_change']:+.1f}%  ({d['n300']:.3f} → {d['n1000']:.3f})")
    print("\nResults saved to modal volume: imi-experiment-results")
    print("=" * 60)

    return consolidated


def _extract_key_metric(bench_name: str, data: dict) -> str:
    """Extract the most important metric from a benchmark result for display."""
    key_map = {
        "ambench": ("retrieval_r5", "R@5"),
        "tiered_recall": ("tier_ratio", "Ratio"),
        "tiered_efficiency": ("pct_under_200", "Under200"),
        "cross_session": ("retention_rate", "Retention"),
        "sd_retrieval": ("sd_r5", "SD-R@5"),
        "longmem_eval": ("overall_r5", "Overall"),
        "federated_recall": ("federation_boost", "Boost"),
    }
    if bench_name in key_map:
        field, label = key_map[bench_name]
        val = data.get(field, "?")
        return f"{label}={val}"
    return str(data)[:60]


def _extract_numeric_metric(bench_name: str, data: dict) -> float | None:
    """Extract the primary numeric metric for degradation analysis."""
    key_map = {
        "ambench": "retrieval_r5",
        "tiered_recall": "tier_ratio",
        "cross_session": "retention_rate",
        "sd_retrieval": "sd_r5",
        "longmem_eval": "overall_r5",
        "federated_recall": "federated_r5",
    }
    field = key_map.get(bench_name)
    if field and field in data:
        try:
            return float(data[field])
        except (TypeError, ValueError):
            return None
    return None


# ── Local entrypoint ──────────────────────────────────

@app.local_entrypoint()
def main(scale: bool = False):
    """Run experiments from local CLI.

    Usage:
        modal run modal_experiments.py                  # Original 3 experiments (300 incidents)
        modal run modal_experiments.py --scale          # Scale comparison (300/500/1000)
    """
    if scale:
        results = run_scale_comparison.remote()
    else:
        results = run_all.remote()
    print(json.dumps(results, indent=2))
