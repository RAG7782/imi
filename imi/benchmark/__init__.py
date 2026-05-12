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
from imi.benchmark.cross_session import CrossSession, CrossSessionResults
from imi.benchmark.federated_recall import FederatedRecall, FederatedRecallResults
from imi.benchmark.longmem_eval import LongMemEval, LongMemEvalResults
from imi.benchmark.sd_retrieval import SDRetrieval, SDRetrievalResults
from imi.benchmark.tiered_efficiency import TieredEfficiency, TieredEfficiencyResults
from imi.benchmark.tiered_recall import TieredRecall, TieredRecallResults

__all__ = [
    "AMBench",
    "AMBenchResults",
    "TieredRecall",
    "TieredRecallResults",
    "TieredEfficiency",
    "TieredEfficiencyResults",
    "CrossSession",
    "CrossSessionResults",
    "SDRetrieval",
    "SDRetrievalResults",
    "LongMemEval",
    "LongMemEvalResults",
    "FederatedRecall",
    "FederatedRecallResults",
]
