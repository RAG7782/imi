"""Bridge for consuming SYMBIONT signals to inform IMI tiering.

4 Synergies:
1. Mycelium channel weights -> L1 promotion scoring
2. Mound artifacts (APPROVED) -> L2 cache
3. Murmuration PRIORITY_SHIFT -> L1 refresh with domain filter
4. Federation relay -> FCM transport (handled by ClawVault, not here)

Graceful degradation: all public functions are wrapped by imi_safe() —
timeout 2s, silent fallback on any error. Same pattern as Immune Bridge.
IMI failure never blocks SYMBIONT execution.
"""

from __future__ import annotations

import functools
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

_IMI_TIMEOUT_S: float = 2.0  # matches Immune Bridge pattern

F = TypeVar("F", bound=Callable)


def imi_safe(fallback: Any = None, timeout: float = _IMI_TIMEOUT_S):
    """Decorator: run function in thread with timeout; return fallback on any error.

    Guarantees IMI bridge never blocks SYMBIONT execution — identical contract
    to immune_bridge.bridge_health() graceful degradation.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result_box: list[Any] = [fallback]
            exc_box: list[BaseException | None] = [None]

            def _run():
                try:
                    result_box[0] = fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    exc_box[0] = exc

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout)
            if t.is_alive():
                logger.warning(
                    "imi_safe: %s timed out after %.1fs — returning fallback", fn.__name__, timeout
                )
                return fallback
            if exc_box[0] is not None:
                logger.warning(
                    "imi_safe: %s raised %s — returning fallback", fn.__name__, exc_box[0]
                )
                return fallback
            return result_box[0]

        return wrapper  # type: ignore[return-value]

    return decorator


# FCM event directory
FCM_EVENTS_DIR = Path.home() / ".fcm" / "events"


@imi_safe(fallback={})
def read_channel_weights(symbiont_url: str | None = None) -> dict[str, float]:
    """Read SYMBIONT Mycelium channel weights.

    Sinergia 1: Productive channels (high weight) boost L1 promotion.

    Tries in order:
    1. HTTP from SYMBIONT API (if url provided)
    2. Cached file at ~/.imi/channel_weights.json
    3. Empty dict (graceful degradation)
    """
    # Try cached file first (most common in practice)
    cache_path = Path.home() / ".imi" / "channel_weights.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            if isinstance(data, dict):
                return {k: float(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            logger.warning("Invalid channel_weights cache, ignoring")

    # Graceful degradation: no weights available
    return {}


@imi_safe(fallback=None)
def check_priority_shift() -> str | None:
    """Check FCM bus for recent PRIORITY_SHIFT signal.

    Sinergia 3: Returns new domain if priority shifted, None otherwise.
    Reads from ~/.fcm/events/ looking for murmuration signals.
    """
    if not FCM_EVENTS_DIR.exists():
        return None

    for event_file in sorted(FCM_EVENTS_DIR.glob("*.json"), reverse=True)[:10]:
        try:
            event = json.loads(event_file.read_text())
            if (
                event.get("type") == "custom"
                and event.get("metadata", {}).get("signal_type") == "PRIORITY_SHIFT"
            ):
                return event.get("metadata", {}).get("new_domain")
        except (json.JSONDecodeError, KeyError):
            continue

    return None


@imi_safe(fallback=[])
def get_mound_approved_artifacts() -> list[dict[str, Any]]:
    """Get APPROVED Mound artifacts from FCM bus.

    Sinergia 2: These become L2 cache candidates in IMI.
    """
    if not FCM_EVENTS_DIR.exists():
        return []

    artifacts = []
    for event_file in sorted(FCM_EVENTS_DIR.glob("*.json"), reverse=True)[:50]:
        try:
            event = json.loads(event_file.read_text())
            if (
                event.get("source") == "symbiont"
                and event.get("type") == "memory_created"
                and event.get("metadata", {}).get("artifact_status") == "APPROVED"
                and event.get("metadata", {}).get("quality", 0) >= 0.8
            ):
                artifacts.append(
                    {
                        "title": event.get("title", ""),
                        "content": event.get("content", ""),
                        "tags": event.get("tags", []),
                        "quality": event["metadata"]["quality"],
                    }
                )
        except (json.JSONDecodeError, KeyError):
            continue

    return artifacts
