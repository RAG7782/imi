"""Tests for positional optimization (primacy-recency reordering).

Based on "Lost in the Middle" (Liu et al. 2023):
LLMs lose information in the middle of the context window.
Reorder so highest-relevance items sit at start + end.
"""

import pytest

from imi.positional import positional_reorder


# ── Unit tests for the reorder algorithm ──────────────────────────────────


class TestPositionalReorder:
    """Pure algorithmic tests — no IMI dependency."""

    def test_empty_list(self):
        assert positional_reorder([]) == []

    def test_single_item(self):
        assert positional_reorder([1]) == [1]

    def test_two_items(self):
        assert positional_reorder([1, 2]) == [1, 2]

    def test_three_items(self):
        # [1,2,3] -> start=[1,3] end=[2] -> [1,3,2]
        assert positional_reorder([1, 2, 3]) == [1, 3, 2]

    def test_six_items(self):
        # [1,2,3,4,5,6] -> start=[1,3,5] end=[6,4,2] -> [1,3,5,6,4,2]
        result = positional_reorder([1, 2, 3, 4, 5, 6])
        assert result == [1, 3, 5, 6, 4, 2]

    def test_seven_items(self):
        # [1,2,3,4,5,6,7] -> start=[1,3,5,7] end=[6,4,2] -> [1,3,5,7,6,4,2]
        result = positional_reorder([1, 2, 3, 4, 5, 6, 7])
        assert result == [1, 3, 5, 7, 6, 4, 2]

    def test_preserves_elements(self):
        """All original elements must be present (no loss, no duplication)."""
        original = list(range(20))
        result = positional_reorder(original)
        assert sorted(result) == sorted(original)
        assert len(result) == len(original)

    def test_best_item_stays_first(self):
        """Rank-1 (index 0) must always be at position 0."""
        for n in range(1, 15):
            items = list(range(1, n + 1))
            result = positional_reorder(items)
            assert result[0] == 1, f"Failed for n={n}"

    def test_second_best_at_end(self):
        """Rank-2 (index 1) should land at the last position."""
        for n in range(3, 15):
            items = list(range(1, n + 1))
            result = positional_reorder(items)
            assert result[-1] == 2, f"Failed for n={n}: got {result}"

    def test_edges_more_relevant_than_center(self):
        """The average rank at the edges should be better (lower) than at the center."""
        items = list(range(1, 11))  # ranks 1-10
        result = positional_reorder(items)
        edge_avg = (result[0] + result[-1]) / 2
        mid = len(result) // 2
        center_avg = (result[mid - 1] + result[mid]) / 2
        assert edge_avg < center_avg

    def test_does_not_mutate_input(self):
        original = [1, 2, 3, 4, 5]
        copy = list(original)
        positional_reorder(original)
        assert original == copy

    def test_works_with_dicts(self):
        """Simulate memory dicts like IMI returns."""
        items = [
            {"score": 0.95, "content": "best"},
            {"score": 0.90, "content": "second"},
            {"score": 0.85, "content": "third"},
            {"score": 0.80, "content": "fourth"},
        ]
        result = positional_reorder(items)
        assert result[0]["content"] == "best"
        assert result[-1]["content"] == "second"
        assert len(result) == 4


# ── Integration tests with IMISpace.navigate ──────────────────────────────


class TestNavigatePositionalIntegration:
    """Test that navigate actually applies positional reordering."""

    @pytest.fixture(autouse=True)
    def setup_space(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IMI_DB", str(tmp_path / "test_pos.db"))

    def _make_space(self, tmp_path):
        from imi.space import IMISpace
        return IMISpace.from_sqlite(str(tmp_path / "test_pos.db"))

    def test_navigate_positional_default_on(self, tmp_path):
        """With positional_optimize=True (default), order differs from pure score sort."""
        space = self._make_space(tmp_path)

        # Encode enough memories to make reordering visible
        for i in range(6):
            space.encode(f"Memory about topic {i} with details {i * 10}")

        nav_on = space.navigate("topic", top_k=6, positional_optimize=True)
        nav_off = space.navigate("topic", top_k=6, positional_optimize=False)

        # Both should have same elements
        ids_on = [m["id"] for m in nav_on.memories]
        ids_off = [m["id"] for m in nav_off.memories]
        assert sorted(ids_on) == sorted(ids_off)

        # With enough items (>2), order should differ
        if len(ids_on) > 2:
            assert ids_on != ids_off, "Positional reorder should change the order for >2 items"

    def test_navigate_positional_off_preserves_score_order(self, tmp_path):
        """With positional_optimize=False, memories stay sorted by descending score."""
        space = self._make_space(tmp_path)

        for i in range(5):
            space.encode(f"Event number {i} happened during incident")

        nav = space.navigate("incident", top_k=5, positional_optimize=False)
        scores = [m["score"] for m in nav.memories]
        assert scores == sorted(scores, reverse=True)

    def test_navigate_small_result_unchanged(self, tmp_path):
        """With <=2 results, positional_optimize should not change order."""
        space = self._make_space(tmp_path)
        space.encode("Single memory about auth failure")

        nav_on = space.navigate("auth", top_k=5, positional_optimize=True)
        nav_off = space.navigate("auth", top_k=5, positional_optimize=False)

        ids_on = [m["id"] for m in nav_on.memories]
        ids_off = [m["id"] for m in nav_off.memories]
        assert ids_on == ids_off
