"""Memory event system — append-only log of all IMI mutations.

Every encode, fade, consolidate, prune, and reconsolidate is recorded
as a MemoryEvent. This enables replay, audit, debugging of cognitive
drift, and observability of the dreaming process.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# Event type constants
ENCODE = "encode"
TOUCH = "touch"
NAVIGATE_ACCESS = "navigate_access"
FADE_CYCLE = "fade_cycle"
CONSOLIDATE = "consolidate"
CONSOLIDATE_STRENGTHEN = "consolidate_strengthen"
PRUNE_CANDIDATE = "prune_candidate"
RECONSOLIDATE = "reconsolidate"
MIGRATE_IN = "migrate_in"
MIGRATE_OUT = "migrate_out"


@dataclass
class MemoryEvent:
    """A single mutation event in the IMI memory space."""

    event_type: str
    node_id: str
    store_name: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    node_version: int = 0
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "node_id": self.node_id,
            "store_name": self.store_name,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "node_version": self.node_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEvent:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
