"""CLI entry point: python -m imi.benchmark"""

import argparse
import json

from imi.benchmark import AMBench
from imi.benchmark.cross_session import CrossSession
from imi.benchmark.federated_recall import FederatedRecall
from imi.benchmark.longmem_eval import LongMemEval
from imi.benchmark.sd_retrieval import SDRetrieval
from imi.benchmark.tiered_efficiency import TieredEfficiency
from imi.benchmark.tiered_recall import TieredRecall


def main():
    parser = argparse.ArgumentParser(
        description="IMI Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m imi.benchmark                              # Default (AMBench)
  python -m imi.benchmark --suite tiered               # Tiered Recall + Efficiency
  python -m imi.benchmark --suite cross                # Cross-Session Recall
  python -m imi.benchmark --suite sd                   # SD Retrieval only
  python -m imi.benchmark --suite longmem              # LongMemEval only
  python -m imi.benchmark --suite federated            # Federated Recall only
  python -m imi.benchmark --suite full                 # ALL benchmarks
  python -m imi.benchmark --incidents 100 --days 30    # Quick run
  python -m imi.benchmark --rw 0.0 --name RAG          # Test pure cosine
  python -m imi.benchmark --json                       # JSON output
""",
    )
    parser.add_argument(
        "--suite",
        choices=["ambench", "tiered", "cross", "sd", "longmem", "federated", "full"],
        default="ambench",
        help="Benchmark suite to run",
    )
    parser.add_argument("--incidents", type=int, default=300, help="Number of incidents")
    parser.add_argument("--days", type=int, default=90, help="Simulated days")
    parser.add_argument("--rw", type=float, default=0.10, help="Relevance weight")
    parser.add_argument("--name", default="IMI", help="System name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--sessions", type=int, default=30, help="Sessions (for cross/tiered-efficiency)"
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    all_results = {}

    # --- AMBench ---
    if args.suite in ("ambench", "full"):
        if not args.json:
            print(f"Running AMBench: {args.incidents} incidents, {args.days} days, rw={args.rw}")
            print("=" * 60)

        bench = AMBench(n_incidents=args.incidents, n_days=args.days, seed=args.seed)
        results = bench.run(system_name=args.name, relevance_weight=args.rw)

        if args.json:
            all_results["ambench"] = results.to_dict()
        else:
            print(results)
            print()

    # --- Tiered Recall ---
    if args.suite in ("tiered", "full"):
        if not args.json:
            print(f"Running TieredRecall: {args.incidents} incidents, {args.days} days")
            print("=" * 60)

        bench = TieredRecall(n_incidents=args.incidents, n_days=args.days, seed=args.seed)
        results = bench.run(system_name=args.name, relevance_weight=args.rw)

        if args.json:
            all_results["tiered_recall"] = results.to_dict()
        else:
            print(results)
            print()

    # --- Tiered Efficiency ---
    if args.suite in ("tiered", "full"):
        if not args.json:
            print(f"Running TieredEfficiency: {args.incidents} incidents, {args.sessions} sessions")
            print("=" * 60)

        bench = TieredEfficiency(
            n_incidents=args.incidents,
            n_days=args.days,
            seed=args.seed,
            n_sessions=args.sessions,
        )
        results = bench.run(system_name=args.name)

        if args.json:
            all_results["tiered_efficiency"] = results.to_dict()
        else:
            print(results)
            print()

    # --- Cross-Session ---
    if args.suite in ("cross", "full"):
        if not args.json:
            print(f"Running CrossSession: {args.incidents} incidents, {args.sessions} sessions")
            print("=" * 60)

        bench = CrossSession(
            n_incidents=args.incidents,
            n_days=args.days,
            n_sessions=args.sessions,
            seed=args.seed,
        )
        results = bench.run(system_name=args.name, relevance_weight=args.rw)

        if args.json:
            all_results["cross_session"] = results.to_dict()
        else:
            print(results)
            print()

    # --- SD Retrieval ---
    if args.suite in ("sd", "full"):
        if not args.json:
            print(f"Running SDRetrieval: {args.incidents} incidents, {args.days} days")
            print("=" * 60)

        bench = SDRetrieval(n_incidents=args.incidents, n_days=args.days, seed=args.seed)
        results = bench.run(system_name=args.name, relevance_weight=args.rw)

        if args.json:
            all_results["sd_retrieval"] = results.to_dict()
        else:
            print(results)
            print()

    # --- LongMemEval ---
    if args.suite in ("longmem", "full"):
        if not args.json:
            print(f"Running LongMemEval: {args.incidents} incidents, {args.days} days")
            print("=" * 60)

        longmem_days = max(args.days, 180)  # LongMemEval needs ≥180 days for 3 time buckets
        bench = LongMemEval(n_incidents=args.incidents, n_days=longmem_days, seed=args.seed)
        results = bench.run(system_name=args.name, relevance_weight=args.rw)

        if args.json:
            all_results["longmem_eval"] = results.to_dict()
        else:
            print(results)
            print()

    # --- Federated Recall ---
    if args.suite in ("federated", "full"):
        if not args.json:
            print(f"Running FederatedRecall: {args.incidents} incidents, {args.days} days")
            print("=" * 60)

        bench = FederatedRecall(n_incidents=args.incidents, n_days=args.days, seed=args.seed)
        results = bench.run(system_name=args.name, relevance_weight=args.rw)

        if args.json:
            all_results["federated_recall"] = results.to_dict()
        else:
            print(results)
            print()

    # JSON output (combined)
    if args.json:
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
