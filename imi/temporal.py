"""Temporal Context Model — the WHEN dimension of memory.

Memories encoded close in time are linked, independent of semantic similarity.
Based on: Howard & Kahana (2002) Temporal Context Model.

Adds: temporal position, temporal search, temporal clustering.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime

# A1: externalize temporal window — override via IMI_TEMPORAL_WINDOW_HOURS
DEFAULT_TEMPORAL_WINDOW_HOURS: float = float(os.getenv("IMI_TEMPORAL_WINDOW_HOURS", "1.0"))


@dataclass
class TemporalContext:
    """Temporal metadata for a memory node."""

    # Absolute time
    timestamp: float = field(default_factory=time.time)

    # Session context: which "thinking session" was this encoded in?
    session_id: str = ""

    # Sequence position: nth memory in this session
    sequence_pos: int = 0

    # Temporal neighbors: memories encoded within temporal_window
    temporal_neighbors: list[str] = field(default_factory=list)  # node IDs

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)

    @property
    def age_days(self) -> float:
        return (time.time() - self.timestamp) / 86400

    def temporal_distance(self, other: TemporalContext) -> float:
        """Distance in days between two temporal contexts."""
        return abs(self.timestamp - other.timestamp) / 86400

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "sequence_pos": self.sequence_pos,
            "temporal_neighbors": self.temporal_neighbors,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TemporalContext:
        # L1 fix: field guard — only pass known fields to avoid TypeError on future keys
        known = {"timestamp", "session_id", "sequence_pos", "temporal_neighbors"}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TemporalIndex:
    """Index for temporal navigation — 'what was happening around time X?'"""

    contexts: dict[str, TemporalContext] = field(default_factory=dict)  # node_id → context
    _session_counter: int = 0
    _sequence_counter: int = 0
    _current_session: str = ""

    def new_session(self, session_id: str = "") -> str:
        """Start a new temporal session."""
        self._session_counter += 1
        self._sequence_counter = 0
        self._current_session = session_id or f"session_{self._session_counter}"
        return self._current_session

    def register(
        self,
        node_id: str,
        timestamp: float | None = None,
        temporal_window_hours: float | None = None,
    ) -> TemporalContext:
        if temporal_window_hours is None:
            temporal_window_hours = DEFAULT_TEMPORAL_WINDOW_HOURS
        """Register a new memory's temporal context."""
        if not self._current_session:
            self.new_session()

        self._sequence_counter += 1
        ts = timestamp or time.time()

        # Find temporal neighbors (within window)
        window_seconds = temporal_window_hours * 3600
        neighbors = []
        for nid, ctx in self.contexts.items():
            if abs(ctx.timestamp - ts) <= window_seconds:
                neighbors.append(nid)
                # Bidirectional: also add us as neighbor to them
                if node_id not in ctx.temporal_neighbors:
                    ctx.temporal_neighbors.append(node_id)

        context = TemporalContext(
            timestamp=ts,
            session_id=self._current_session,
            sequence_pos=self._sequence_counter,
            temporal_neighbors=neighbors,
        )
        self.contexts[node_id] = context
        return context

    def search_by_time(
        self,
        target_time: float,
        window_hours: float = 24.0,
        max_results: int = 10,
    ) -> list[tuple[str, float]]:
        """Find memories near a specific time. Returns (node_id, distance_hours)."""
        window_seconds = window_hours * 3600
        results = []
        for nid, ctx in self.contexts.items():
            dist = abs(ctx.timestamp - target_time)
            if dist <= window_seconds:
                results.append((nid, dist / 3600))
        results.sort(key=lambda x: x[1])
        return results[:max_results]

    def search_by_session(self, session_id: str) -> list[str]:
        """Find all memories from a specific session."""
        return [nid for nid, ctx in self.contexts.items() if ctx.session_id == session_id]

    def get_timeline(self, node_ids: list[str] | None = None) -> list[tuple[str, TemporalContext]]:
        """Get chronologically ordered timeline."""
        items = [
            (nid, ctx) for nid, ctx in self.contexts.items() if node_ids is None or nid in node_ids
        ]
        items.sort(key=lambda x: x[1].timestamp)
        return items
