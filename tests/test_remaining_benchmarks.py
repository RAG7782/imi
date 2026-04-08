"""Tests for remaining benchmarks: SDRetrieval, LongMemEval, FederatedRecall."""
import pytest
from imi.benchmark.sd_retrieval import SDRetrieval
from imi.benchmark.longmem_eval import LongMemEval
from imi.benchmark.federated_recall import FederatedRecall


class TestSDRetrieval:
    def test_runs_without_error(self):
        bench = SDRetrieval(n_incidents=50, n_days=10)
        results = bench.run(eval_every=10)
        assert results.n_queries > 0
        assert 0 <= results.baseline_r5 <= 1
        assert 0 <= results.sd_r5 <= 1

    def test_ds_d_scores_computed(self):
        bench = SDRetrieval(n_incidents=30, n_days=10)
        results = bench.run(eval_every=10)
        assert results.ds_d_mean > 0
        assert results.ds_d_std >= 0
        assert results.n_incidents == 30

    def test_to_dict(self):
        bench = SDRetrieval(n_incidents=30, n_days=10)
        results = bench.run(eval_every=10)
        d = results.to_dict()
        assert "baseline_r5" in d
        assert "sd_r5" in d
        assert "improvement_pct" in d
        assert "ds_d_mean" in d
        assert "ds_d_std" in d
        assert d["system"] == "IMI"

    def test_str_output(self):
        bench = SDRetrieval(n_incidents=30, n_days=10)
        results = bench.run(eval_every=10)
        s = str(results)
        assert "SDRetrieval" in s
        assert "DS-d" in s


class TestLongMemEval:
    def test_runs_without_error(self):
        bench = LongMemEval(n_incidents=50, n_days=30)
        results = bench.run(n_queries_per_bucket=5)
        assert results.n_queries > 0
        assert 0 <= results.recent_r5 <= 1
        assert 0 <= results.mid_r5 <= 1
        assert 0 <= results.overall_r5 <= 1

    def test_time_buckets(self):
        bench = LongMemEval(n_incidents=50, n_days=60)
        results = bench.run(n_queries_per_bucket=5)
        # Overall should be a weighted combination
        expected = 0.40 * results.recent_r5 + 0.35 * results.mid_r5 + 0.25 * results.old_r5
        assert abs(results.overall_r5 - expected) < 0.001
        assert results.n_days == 60
        assert results.n_incidents == 50

    def test_to_dict(self):
        bench = LongMemEval(n_incidents=30, n_days=30)
        results = bench.run(n_queries_per_bucket=3)
        d = results.to_dict()
        assert "recent_r5" in d
        assert "mid_r5" in d
        assert "old_r5" in d
        assert "overall_r5" in d
        assert d["system"] == "IMI"

    def test_str_output(self):
        bench = LongMemEval(n_incidents=30, n_days=30)
        results = bench.run(n_queries_per_bucket=3)
        s = str(results)
        assert "LongMemEval" in s
        assert "Overall" in s


class TestFederatedRecall:
    def test_runs_without_error(self):
        bench = FederatedRecall(n_incidents=50, n_days=10)
        results = bench.run(n_queries=10)
        assert results.n_queries > 0
        assert 0 <= results.isolated_r5 <= 1
        assert 0 <= results.federated_r5 <= 1

    def test_federation_improves_recall(self):
        bench = FederatedRecall(n_incidents=50, n_days=10)
        results = bench.run(n_queries=10)
        # Federated should be >= isolated (more data = better or equal recall)
        assert results.federated_r5 >= results.isolated_r5
        assert results.federation_boost >= 0
        # Overlap should be around 20% (60%+60%-100%=20%)
        assert 0.1 <= results.overlap_pct <= 0.4

    def test_to_dict(self):
        bench = FederatedRecall(n_incidents=30, n_days=10)
        results = bench.run(n_queries=5)
        d = results.to_dict()
        assert "isolated_r5" in d
        assert "federated_r5" in d
        assert "federation_boost" in d
        assert "overlap_pct" in d
        assert d["system"] == "IMI"

    def test_str_output(self):
        bench = FederatedRecall(n_incidents=30, n_days=10)
        results = bench.run(n_queries=5)
        s = str(results)
        assert "FederatedRecall" in s
        assert "Boost" in s
