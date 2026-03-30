"""AMBench — Agent Memory Benchmark.

First standardized benchmark for evaluating AI agent memory systems.
Tests the full encode-retrieve-consolidate-act lifecycle over 90 simulated days.

Usage:
    # As Python API
    from imi.benchmark import AMBench
    bench = AMBench()
    results = bench.run()
    print(results)

    # As CLI
    python -m imi.benchmark
    python -m imi.benchmark --days 90 --incidents 300 --patterns 10
"""

from imi.benchmark.ambench import AMBench, AMBenchResults

__all__ = ["AMBench", "AMBenchResults"]
