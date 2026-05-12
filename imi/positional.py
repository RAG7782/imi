"""Positional optimization for LLM context windows.

Based on "Lost in the Middle" (Liu et al. 2023): LLMs attend more strongly
to information at the START and END of the context, losing detail in the
MIDDLE. This module reorders retrieved memories so that the highest-relevance
items land at the edges (primacy-recency pattern) and lower-relevance items
occupy the center — where attention is weakest.

Example:
    Input (ranked by score):  [1, 2, 3, 4, 5, 6]
    Output (primacy-recency): [1, 3, 5, 6, 4, 2]

    Positions 0, 2, 4 come from even indices (highest relevance at edges).
    Positions filled from odd indices reversed sit in the interior/end.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def positional_reorder(items: list[T]) -> list[T]:
    """Reorder items so highest-relevance items are at start and end (primacy-recency).

    Assumes *items* arrive sorted by descending relevance (best first).
    The algorithm interleaves:
      - Even-indexed items (0, 2, 4, ...) keep their forward order  -> START
      - Odd-indexed items (1, 3, 5, ...) are reversed               -> END

    This places rank-1 at position 0, rank-3 at position 1, etc.,
    and rank-2 (reversed) at the very end — maximising attention on
    the most relevant items.

    Args:
        items: List sorted by descending relevance.

    Returns:
        New list with the same elements in primacy-recency order.
    """
    if len(items) <= 2:
        return list(items)

    start = items[::2]  # indices 0, 2, 4, ...
    end = items[1::2][::-1]  # indices 1, 3, 5, ... reversed
    return start + end
