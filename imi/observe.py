"""Observability layer — timing and metrics for IMI storage operations.

Instruments every storage backend method with duration tracking.
Metrics are collected in-memory and emitted as structured logs.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger("imi.observe")


@dataclass
class OperationMetrics:
    """A single recorded operation."""

    operation: str
    duration_ms: float
    store_name: str = ""
    node_count: int = 0
    success: bool = True
    error: str = ""
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects operation metrics. Thread-safe via append-only list."""

    def __init__(self) -> None:
        self._metrics: list[OperationMetrics] = []

    def record(self, m: OperationMetrics) -> None:
        self._metrics.append(m)
        level = logging.INFO if m.success else logging.WARNING
        logger.log(
            level,
            "imi.%s store=%s nodes=%d %.1fms%s",
            m.operation,
            m.store_name or "-",
            m.node_count,
            m.duration_ms,
            f" ERROR={m.error}" if m.error else "",
        )

    def summary(self) -> dict[str, Any]:
        """Return aggregate stats per operation type."""
        by_op: dict[str, list[float]] = defaultdict(list)
        errors: dict[str, int] = defaultdict(int)
        for m in self._metrics:
            by_op[m.operation].append(m.duration_ms)
            if not m.success:
                errors[m.operation] += 1

        result = {}
        for op, durations in by_op.items():
            result[op] = {
                "count": len(durations),
                "total_ms": sum(durations),
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "errors": errors.get(op, 0),
            }
        return result

    def reset(self) -> None:
        self._metrics.clear()

    @property
    def metrics(self) -> list[OperationMetrics]:
        return list(self._metrics)


# Global singleton, replaceable for testing
_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    return _collector


def set_collector(collector: MetricsCollector) -> None:
    global _collector
    _collector = collector


def timed(operation: str) -> Callable:
    """Decorator that records timing for storage operations."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                _collector.record(
                    OperationMetrics(
                        operation=operation,
                        duration_ms=(time.perf_counter() - t0) * 1000,
                        success=True,
                    )
                )
                return result
            except Exception as e:
                _collector.record(
                    OperationMetrics(
                        operation=operation,
                        duration_ms=(time.perf_counter() - t0) * 1000,
                        success=False,
                        error=str(e),
                    )
                )
                raise

        return wrapper

    return decorator
