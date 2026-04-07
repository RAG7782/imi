"""FCM Bridge — IMI ↔ ClawVault Federation via FCM Bus.

Emits MemoryEvents to ~/.fcm/events/ after IMI encodes.
Reads ClawVault events from ~/.fcm/events/ for IMI ingestion.

Phase 0-C of Federated Cognitive Memory.

Usage:
    from imi.integrations.fcm_bridge import FCMBridge

    bridge = FCMBridge()

    # After imi.encode():
    bridge.emit_encode(node)

    # Poll for ClawVault events:
    events = bridge.poll_clawvault_events()
    for event in events:
        space.encode(event["content"], tags=event["tags"], source="clawvault")
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

FCM_DIR = Path.home() / ".fcm"
FCM_EVENTS_DIR = FCM_DIR / "events"
FCM_PROCESSED_DIR = FCM_DIR / "processed"


class FCMBridge:
    """Bridge between IMI and FCM event bus."""

    def __init__(self, source: str = "imi"):
        self.source = source
        self._ensure_dirs()

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
            Event filepath if written, None if skipped
        """
        # Extract from node
        content = getattr(node, "original", "") or getattr(node, "seed", "")
        if not content:
            return None

        tags = list(getattr(node, "tags", []))
        if extra_tags:
            tags.extend(extra_tags)

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

        title = getattr(node, "summary_orbital", "") or content[:80]
        node_id = getattr(node, "id", "")

        event = {
            "id": uuid.uuid4().hex[:16],
            "timestamp": _iso_now(),
            "source": self.source,
            "type": "memory_created",
            "title": title,
            "content": content,
            "tags": tags,
            "salience": salience,
            "metadata": {
                "imi_node_id": node_id,
                "imi_seed": getattr(node, "seed", ""),
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
            "id": uuid.uuid4().hex[:16],
            "timestamp": _iso_now(),
            "source": self.source,
            "type": event_type,
            "title": f"{event_type}: {summary[:60]}",
            "content": summary,
            "tags": tags or ["session"],
            "salience": salience,
            "metadata": metadata or {},
        }
        return self._write_event(event)

    def poll_clawvault_events(self, max_events: int = 50) -> list[dict[str, Any]]:
        """Read pending ClawVault events from FCM bus.

        Returns events where source == 'clawvault'.
        Does NOT move them to processed — the watcher.ts handles that.
        This is for direct polling when watcher isn't running.
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
                if data.get("source") == "clawvault":
                    data["_filepath"] = str(f)
                    events.append(data)
            except (json.JSONDecodeError, OSError):
                continue

        return events

    def mark_consumed(self, filepath: str) -> None:
        """Move a consumed event to processed dir."""
        src = Path(filepath)
        if src.exists():
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
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z"
