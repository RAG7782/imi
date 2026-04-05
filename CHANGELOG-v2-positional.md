# CHANGELOG — v2 Positional Optimization

## v0.2.1-positional (2026-04-05)

### Added

- **Positional sensitivity in `navigate`** — Based on "Lost in the Middle"
  (Liu et al. 2023), LLMs lose information in the center of the context
  window. Navigate now reorders results using a primacy-recency pattern:
  highest-relevance memories are placed at the START and END of the list,
  while lower-relevance items occupy the CENTER (where attention is weakest).

- New parameter `positional_optimize` (default `True`) on:
  - `IMISpace.navigate()` in `imi/space.py`
  - `imi_navigate` MCP tool in `imi/mcp_server.py`
  - `POST /navigate` REST endpoint in `imi/api.py`

- New module `imi/positional.py` with the `positional_reorder()` function.

- Test suite `tests/test_positional.py` covering:
  - 11 unit tests for the reorder algorithm (edge cases, invariants,
    dict support, input immutability)
  - 3 integration tests with `IMISpace.navigate` (on/off behavior,
    score-order preservation, small-result passthrough)

### Algorithm

Given items sorted by descending relevance `[1, 2, 3, 4, 5, 6]`:

```
start = items[::2]       → [1, 3, 5]     (even indices, forward)
end   = items[1::2][::-1] → [6, 4, 2]    (odd indices, reversed)
result = start + end      → [1, 3, 5, 6, 4, 2]
```

Properties:
- Rank-1 always at position 0 (primacy)
- Rank-2 always at last position (recency)
- Center positions hold the least-relevant items
- `positional_optimize=False` gives identical behavior to the original code

### Backward Compatibility

- Default behavior changes: `positional_optimize=True` is the new default.
  To restore the original score-sorted order, pass `positional_optimize=False`.
- No changes to encode, dream, search_actions, or any other endpoint.
- No new dependencies.

### References

- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M.,
  Petroni, F., & Liang, P. (2023). "Lost in the Middle: How Language
  Models Use Long Contexts." *Transactions of the ACL*.
