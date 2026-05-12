"""Tests for expanded benchmark suite."""

from imi.benchmark.cross_session import CrossSession
from imi.benchmark.tiered_efficiency import TieredEfficiency
from imi.benchmark.tiered_recall import TieredRecall


class TestTieredRecall:
    def test_runs_without_error(self):
        bench = TieredRecall(n_incidents=50, n_days=10)
        results = bench.run(eval_every=10)
        assert results.n_queries > 0
        assert 0 <= results.full_r5 <= 1
        assert 0 <= results.l1_coverage <= 1
        assert 0 <= results.tier_ratio

    def test_to_dict(self):
        bench = TieredRecall(n_incidents=50, n_days=10)
        results = bench.run(eval_every=10)
        d = results.to_dict()
        assert "full_r5" in d
        assert "tier_ratio" in d

    def test_str_output(self):
        bench = TieredRecall(n_incidents=50, n_days=10)
        results = bench.run(eval_every=10)
        s = str(results)
        assert "TieredRecall" in s


class TestTieredEfficiency:
    def test_runs_without_error(self):
        bench = TieredEfficiency(n_incidents=30, n_days=10, n_sessions=5)
        results = bench.run()
        assert results.l0_tokens > 0
        assert results.l0_l1_tokens > 0
        assert results.n_sessions == 5

    def test_token_budget(self):
        bench = TieredEfficiency(n_incidents=30, n_days=10, n_sessions=10)
        results = bench.run()
        # L0+L1 should be compact
        assert results.l0_l1_tokens < 500  # Generous upper bound for test
        # L3 should be much larger
        assert results.l3_avg_tokens > results.l0_l1_tokens


class TestCrossSession:
    def test_runs_without_error(self):
        bench = CrossSession(n_incidents=30, n_days=10, n_sessions=3)
        results = bench.run()
        assert results.initial_r5 >= 0
        assert results.final_r5 >= 0
        assert results.n_sessions == 3
        assert len(results.r5_per_session) == 4  # initial + 3 sessions

    def test_retention(self):
        bench = CrossSession(n_incidents=50, n_days=15, n_sessions=5)
        results = bench.run()
        # Recall shouldn't completely collapse
        assert results.retention_rate > 0.5
