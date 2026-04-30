"""FCM Bridge — IMI ↔ ClawVault Federation via FCM Bus.

Emits MemoryEvents to ~/.fcm/events/ after IMI encodes.
Reads ClawVault events from ~/.fcm/events/ for IMI ingestion.

Phase 0-D of Federated Cognitive Memory.

Safety:
    - Loop prevention: events tagged 'federated' are never re-emitted
    - Trust gradient: IMI source trust maps to salience floors
    - Dedup: consumed events tracked to prevent re-processing

Usage:
    from imi.integrations.fcm_bridge import FCMBridge

    bridge = FCMBridge()

    # After imi.encode():
    bridge.emit_encode(node)

    # Poll for ClawVault events (with loop guard):
    events = bridge.poll_clawvault_events()
    for event in events:
        space.encode(event["content"], tags=event["tags"], source="clawvault")
        bridge.mark_consumed(event["_filepath"])
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FCM_DIR = Path.home() / ".fcm"
FCM_EVENTS_DIR = FCM_DIR / "events"
FCM_PROCESSED_DIR = FCM_DIR / "processed"

# Content size limit for FCM events (configurable via env var)
_FCM_CONTENT_MAX_CHARS: int = int(os.environ.get("FCM_CONTENT_MAX_CHARS", "2000"))


def _minify(text: str, max_chars: int = _FCM_CONTENT_MAX_CHARS) -> str:
    """Truncate text to max_chars, appending a marker when cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…[truncated]"

# Trust gradient: IMI source → salience floor
# Memories from less-trusted sources get lower salience floors,
# making them more likely to be filtered by the membrane.
TRUST_GRADIENT = {
    "self": 0.9,      # This agent's own memories — highest trust
    "trusted": 0.7,   # Verified peer agents
    "peer": 0.5,      # Known but unverified agents
    "external": 0.3,  # Unknown external sources
}

# Tags that indicate an event was already federated (loop breakers)
FEDERATION_TAGS = frozenset({"federated", "from-imi", "from-clawvault", "from-external"})


class FCMBridge:
    """Bridge between IMI and FCM event bus.

    Safety invariants:
        1. Never emit events that carry federation tags (loop prevention)
        2. Never consume events that originated from this source (echo prevention)
        3. Trust gradient adjusts salience floor per source trust level
    """

    def __init__(self, source: str = "imi", trust_level: str = "self"):
        self.source = source
        self.trust_level = trust_level
        # M4 fix: persist consumed_ids to disk for cross-restart dedup
        self._consumed_ids_file = FCM_DIR / "consumed_ids.json"
        self._consumed_ids: set[str] = self._load_consumed_ids()
        self._ensure_dirs()

    def _load_consumed_ids(self) -> set[str]:
        """M4: Load consumed event IDs from disk."""
        try:
            if self._consumed_ids_file.exists():
                data = json.loads(self._consumed_ids_file.read_text("utf-8"))
                return set(data)
        except Exception:
            pass
        return set()

    def _save_consumed_ids(self) -> None:
        """M4: Persist consumed IDs to disk (keep last 1000)."""
        try:
            ids = list(self._consumed_ids)[-1000:]  # cap at 1000
            self._consumed_ids_file.write_text(json.dumps(ids), "utf-8")
        except Exception:
            pass

    def _ensure_dirs(self) -> None:
        FCM_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        FCM_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def emit_encode(
        self,
        node: Any,
        *,
        salience: float | None = None,
        extra_tags: list[str] | None = None,
    ) -> str | None:
        """Emit a memory_created event after IMI encode.

        Args:
            node: MemoryNode from IMI encode
            salience: Override salience (default: node.mass or 0.6)
            extra_tags: Additional tags beyond node.tags

        Returns:
            Event filepath if written, None if skipped/blocked

        Safety:
            - Skips nodes that originated from federation (source == 'clawvault')
            - Skips nodes with federation tags (loop prevention)
            - Applies trust gradient floor to salience
        """
        # LOOP GUARD: skip nodes that came from federation
        node_source = getattr(node, "source", "")
        if node_source in ("clawvault", "external"):
            return None

        # Extract from node
        content = getattr(node, "original", "") or getattr(node, "seed", "")
        if not content:
            return None

        tags = list(getattr(node, "tags", []))
        if extra_tags:
            tags.extend(extra_tags)

        # LOOP GUARD: skip if any tag indicates prior federation
        if FEDERATION_TAGS & set(t.lower() for t in tags):
            return None

        # Compute salience from node affect/mass
        if salience is None:
            mass = getattr(node, "mass", None)
            affect = getattr(node, "affect", None)
            if mass is not None:
                salience = min(mass / 10.0, 1.0)  # mass is 0-10 scale
            elif affect and hasattr(affect, "salience"):
                salience = affect.salience
            else:
                salience = 0.6

        # TRUST GRADIENT: apply floor based on trust level
        floor = TRUST_GRADIENT.get(self.trust_level, 0.3)
        salience = max(salience, floor)

        title = getattr(node, "summary_orbital", "") or content[:80]
        node_id = getattr(node, "id", "")

        # M1 fix: strip seed from FCM events when crypto is active
        # to prevent leaking plaintext seed alongside encrypted original
        imi_seed = getattr(node, "seed", "")
        try:
            from imi.integrations.crypto_layer import is_encrypted
            if is_encrypted(content):
                imi_seed = "[encrypted]"
        except ImportError:
            pass

        event = {
            "id": str(uuid.uuid4()),
            "timestamp": _iso_now(),
            "source": self.source,
            "type": "memory_created",
            "title": title,
            "content": _minify(content),
            "tags": tags,
            "salience": salience,
            "metadata": {
                "imi_node_id": node_id,
                "imi_seed": imi_seed,
                "trust_level": self.trust_level,
                "content_original_len": len(content),
            },
        }

        return self._write_event(event)

    def emit_session(
        self,
        event_type: str,
        summary: str,
        *,
        tags: list[str] | None = None,
        salience: float = 0.8,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Emit a session event (session_start, session_end, session_checkpoint)."""
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": _iso_now(),
            "source": self.source,
            "type": event_type,
            "title": f"{event_type}: {summary[:60]}",
            "content": _minify(summary),
            "tags": tags or ["session"],
            "salience": salience,
            "metadata": {**(metadata or {}), "content_original_len": len(summary)},
        }
        return self._write_event(event)

    def poll_clawvault_events(self, max_events: int = 50) -> list[dict[str, Any]]:
        """Read pending ClawVault events from FCM bus.

        Returns events where source == 'clawvault' that haven't been consumed yet.

        Safety:
            - Skips events from own source (echo prevention)
            - Skips already-consumed events (dedup)
            - Skips events with federation tags (loop prevention)
        """
        if not FCM_EVENTS_DIR.exists():
            return []

        events = []
        files = sorted(FCM_EVENTS_DIR.glob("*.json"))

        for f in files[:max_events]:
            if f.name.endswith(".tmp"):
                continue
            try:
                data = json.loads(f.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            # Echo prevention: skip events from our own source
            if data.get("source") == self.source:
                continue

            # Only accept ClawVault events
            if data.get("source") != "clawvault":
                continue

            # Dedup: skip already consumed
            event_id = data.get("id", "")
            if event_id in self._consumed_ids:
                continue

            # Loop prevention: skip events that were already federated
            event_tags = set(t.lower() for t in data.get("tags", []))
            if FEDERATION_TAGS & event_tags:
                continue

            data["_filepath"] = str(f)
            events.append(data)

        return events

    def mark_consumed(self, filepath: str) -> None:
        """Move a consumed event to processed dir and track its ID."""
        src = Path(filepath)
        if not src.exists():
            return

        # Track ID for dedup
        try:
            data = json.loads(src.read_text("utf-8"))
            self._consumed_ids.add(data.get("id", ""))
            # M4 fix: persist consumed IDs to disk
            self._save_consumed_ids()
        except (json.JSONDecodeError, OSError):
            pass

        dest = FCM_PROCESSED_DIR / src.name
        src.rename(dest)

    def _write_event(self, event: dict[str, Any]) -> str:
        """Atomic write to FCM events dir."""
        ts = event["timestamp"].replace(":", "-").replace(".", "-")
        eid = event["id"][:8]
        filename = f"{ts}_{eid}.json"
        filepath = FCM_EVENTS_DIR / filename
        tmp = filepath.with_suffix(".json.tmp")

        tmp.write_text(json.dumps(event, indent=2, ensure_ascii=False), "utf-8")
        tmp.rename(filepath)

        return str(filepath)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
