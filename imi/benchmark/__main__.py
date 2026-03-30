"""CLI entry point: python -m imi.benchmark"""

import argparse
import json
import sys

from imi.benchmark import AMBench


def main():
    parser = argparse.ArgumentParser(
        description="AMBench — Agent Memory Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m imi.benchmark                          # Default (300 incidents, 90 days)
  python -m imi.benchmark --incidents 100 --days 30  # Quick run
  python -m imi.benchmark --rw 0.0 --name RAG      # Test pure cosine
  python -m imi.benchmark --json                    # JSON output
""",
    )
    parser.add_argument("--incidents", type=int, default=300, help="Number of incidents")
    parser.add_argument("--days", type=int, default=90, help="Simulated days")
    parser.add_argument("--rw", type=float, default=0.10, help="Relevance weight")
    parser.add_argument("--name", default="IMI", help="System name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    bench = AMBench(n_incidents=args.incidents, n_days=args.days, seed=args.seed)

    if not args.json:
        print(f"Running AMBench: {args.incidents} incidents, {args.days} days, rw={args.rw}")
        print("=" * 60)

    results = bench.run(system_name=args.name, relevance_weight=args.rw)

    if args.json:
        print(json.dumps(results.to_dict(), indent=2))
    else:
        print(results)


if __name__ == "__main__":
    main()
