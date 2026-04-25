"""MemoryNode — the fundamental unit of IMI (v3: full 100/100)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from math import log
from typing import Any

import numpy as np

from imi.affect import AffectiveTag
from imi.affordance import Affordance
from imi.temporal import TemporalContext


@dataclass
class MemoryNode:
    """A single memory in the infinite image.

    v3 fields:
      - Zoom levels (text at multiple resolutions)
      - Embedding (vector for search)
      - Surprise (predictive coding: what was unexpected)
      - Affect (salience, valence, arousal)
      - Temporal (when, session, sequence)
      - Affordances (what actions this memory enables)
      - Mass (gravitational weight, modulated by affect)
    """

    # Identity
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    occurred_at: float | None = None  # When the event actually happened (vs when it was recorded)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    # Zoom levels — pre-computed text at different resolutions
    summary_orbital: str = ""   # ~10 tokens: the theme
    summary_medium: str = ""    # ~40 tokens: theme + key facts
    summary_detailed: str = ""  # ~100 tokens: technical summary
    seed: str = ""              # ~80 tokens: full reconstruction key

    # Original experience (persisted; encrypted at-rest when IMI_CRYPTO=1)
    original: str | None = None

    # Embedding (for vector search)
    embedding: np.ndarray | None = field(default=None, repr=False)

    # --- v3: Predictive coding ---
    surprise_summary: str = ""         # what was unexpected
    surprise_magnitude: float = 0.0    # 0-1 how surprising
    surprise_elements: list[str] = field(default_factory=list)
    prediction: str = ""               # what was predicted (for reconstruction)

    # --- v3: Affect ---
    affect: AffectiveTag = field(default_factory=AffectiveTag)

    # --- v3: Temporal ---
    temporal: TemporalContext = field(default_factory=TemporalContext)

    # --- v3: Affordances ---
    affordances: list[Affordance] = field(default_factory=list)

    # --- v3: Gravitational mass ---
    mass: float = 1.0

    # Metadata
    tags: list[str] = field(default_factory=list)
    source: str = ""

    # Reconsolidation tracking
    reconsolidation_count: int = 0
    last_reconsolidated: float = 0.0

    # L0-L3 Tiering (VIEW layer)
    tier: int = 3  # Default: L3 (deep). 0=identity, 1=hot facts, 2=filtered, 3=deep
    tier_updated_at: float = 0.0

    # Schema v2: MW data (H2 fix — separate from seed)
    mw_data: dict | None = None

    # SDE-AAAK Dialect (metadata layer)
    sde_tag: str = ""       # Rendered SDE-AAAK tag string
    ds_d: float = 0.0       # Distributional semiotic density score (0-1)
    entities: list[str] = field(default_factory=list)  # 3-letter entity codes

    @property
    def effective_time(self) -> float:
        """When this event actually happened. Uses occurred_at if set, else created_at."""
        return self.occurred_at if self.occurred_at is not None else self.created_at

    def touch(self) -> None:
        """Record an access — updates recency, frequency, and dynamic salience (S06)."""
        self.last_accessed = time.time()
        self.access_count += 1
        if self.affect:
            self.affect.update_dynamic(self.access_count)

    @property
    def relevance(self) -> float:
        """Relevance score combining recency, frequency, affect, and surprise.

        v3: affect modulates relevance — high-affect memories stay relevant longer.
        v3.1: surprise_magnitude boosts novel memories (predictive coding integration).
        """
        days_since = (time.time() - self.last_accessed) / 86400
        fade_resist = self.affect.fade_resistance if self.affect else 0.3
        effective_decay = days_since * (1.0 - 0.5 * fade_resist)
        recency = 1.0 / (1.0 + effective_decay)
        frequency = log(1.0 + self.access_count)
        # Surprise boost: novel memories are more relevant (0→1.0x, 1.0→1.3x)
        surprise_boost = 1.0 + 0.3 * self.surprise_magnitude
        return recency * (1.0 + frequency) * self.mass * surprise_boost

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        d = {
            "id": self.id,
            "created_at": self.created_at,
            "occurred_at": self.occurred_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "summary_orbital": self.summary_orbital,
            "summary_medium": self.summary_medium,
            "summary_detailed": self.summary_detailed,
            "seed": self.seed,
            "surprise_summary": self.surprise_summary,
            "surprise_magnitude": self.surprise_magnitude,
            "surprise_elements": self.surprise_elements,
            "prediction": self.prediction,
            "affect": self.affect.to_dict() if self.affect else {},
            "temporal": self.temporal.to_dict() if self.temporal else {},
            "affordances": [a.to_dict() for a in self.affordances],
            "mass": self.mass,
            "tags": self.tags,
            "source": self.source,
            "reconsolidation_count": self.reconsolidation_count,
            "last_reconsolidated": self.last_reconsolidated,
            "tier": self.tier,
            "tier_updated_at": self.tier_updated_at,
            "sde_tag": self.sde_tag,
            "ds_d": self.ds_d,
            "entities": self.entities,
        }
        if self.mw_data is not None:
            d["mw_data"] = self.mw_data
        if self.original is not None:
            d["original"] = self.original
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryNode:
        """Deserialize from persistence.

        C4 fix: uses d.copy() to avoid mutating the caller's dict.
        Schema v2: tolerates unknown keys gracefully.
        """
        d = d.copy()  # C4: never mutate caller's dict
        emb = d.pop("embedding", None)
        affect_d = d.pop("affect", {})
        temporal_d = d.pop("temporal", {})
        affordances_d = d.pop("affordances", [])
        mw_data = d.pop("mw_data", None)

        # M13: filter to known fields only (forward-compatible)
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        node = cls(**known)
        if emb is not None:
            node.embedding = np.array(emb, dtype=np.float32)
        if affect_d:
            node.affect = AffectiveTag.from_dict(affect_d)
        if temporal_d:
            node.temporal = TemporalContext.from_dict(temporal_d)
        if affordances_d:
            node.affordances = [Affordance.from_dict(a) for a in affordances_d]
        if mw_data is not None:
            node.mw_data = mw_data
        return node
