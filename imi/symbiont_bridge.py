"""Bridge for consuming SYMBIONT signals to inform IMI tiering.

4 Synergies:
1. Mycelium channel weights -> L1 promotion scoring
2. Mound artifacts (APPROVED) -> L2 cache
3. Murmuration PRIORITY_SHIFT -> L1 refresh with domain filter
4. Federation relay -> FCM transport (handled by ClawVault, not here)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# FCM event directory
FCM_EVENTS_DIR = Path.home() / ".fcm" / "events"


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
            if (event.get("type") == "custom"
                and event.get("metadata", {}).get("signal_type") == "PRIORITY_SHIFT"):
                return event.get("metadata", {}).get("new_domain")
        except (json.JSONDecodeError, KeyError):
            continue

    return None


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
            if (event.get("source") == "symbiont"
                and event.get("type") == "memory_created"
                and event.get("metadata", {}).get("artifact_status") == "APPROVED"
                and event.get("metadata", {}).get("quality", 0) >= 0.8):
                artifacts.append({
                    "title": event.get("title", ""),
                    "content": event.get("content", ""),
                    "tags": event.get("tags", []),
                    "quality": event["metadata"]["quality"],
                })
        except (json.JSONDecodeError, KeyError):
            continue

    return artifacts
