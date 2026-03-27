"""Anchoring system — guardrails against confabulation.

Anchors are verifiable facts extracted from experiences.
They constrain reconstruction and provide confidence scoring.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AnchorType(str, Enum):
    FILE = "file"           # path to a file that existed
    COMMIT = "commit"       # git commit SHA
    FACT = "fact"           # verifiable factual statement
    DATE = "date"           # specific date/time
    ENTITY = "entity"       # named entity (person, tool, service)


@dataclass
class Anchor:
    """A verifiable fact tied to a memory."""

    type: AnchorType
    reference: str          # the fact/path/SHA itself
    snapshot: str = ""      # literal text from the original experience
    hash: str = ""          # hash of the referenced content at encoding time
    verified_at: float = 0.0

    def verify(self) -> bool | None:
        """Attempt to verify this anchor against current state.

        Returns True (verified), False (contradicted), None (unverifiable).
        """
        if self.type == AnchorType.FILE:
            return os.path.exists(self.reference)

        if self.type == AnchorType.COMMIT:
            try:
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%H", self.reference],
                    capture_output=True, text=True, timeout=5,
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return None

        # FACT, DATE, ENTITY: can't auto-verify without external knowledge
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "reference": self.reference,
            "snapshot": self.snapshot,
            "hash": self.hash,
            "verified_at": self.verified_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Anchor:
        d["type"] = AnchorType(d["type"])
        return cls(**d)


@dataclass
class ConfidenceReport:
    """Result of verifying a reconstruction against anchors."""

    hard_facts: list[str]       # verified, trustworthy
    soft_claims: list[str]      # plausible but unverified
    warnings: list[str]         # potential confabulation
    confidence: float           # 0.0-1.0 overall

    def __str__(self) -> str:
        lines = [f"Confidence: {self.confidence:.0%}"]
        if self.hard_facts:
            lines.append(f"  Hard facts ({len(self.hard_facts)}):")
            for f in self.hard_facts:
                lines.append(f"    [v] {f}")
        if self.soft_claims:
            lines.append(f"  Soft claims ({len(self.soft_claims)}):")
            for c in self.soft_claims:
                lines.append(f"    [~] {c}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    [!] {w}")
        return "\n".join(lines)


def extract_anchors(experience: str, llm) -> list[Anchor]:
    """Extract verifiable anchors from an experience using LLM."""
    import json as _json

    prompt = f"""Extract verifiable facts from this experience. Return JSON array.
Each item: {{"type": "fact|date|entity|file|commit", "reference": "the fact", "snapshot": "exact quote from text"}}

Only extract what is explicitly stated. Do not infer.

Experience:
{experience}

Return ONLY valid JSON array, no markdown fences."""

    raw = llm.generate(
        system="You extract structured facts from text. Return only valid JSON.",
        prompt=prompt,
        max_tokens=512,
    )

    # Parse JSON (handle potential markdown fences)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        items = _json.loads(raw)
    except _json.JSONDecodeError:
        return []

    anchors = []
    for item in items:
        try:
            anchor = Anchor(
                type=AnchorType(item.get("type", "fact")),
                reference=item.get("reference", ""),
                snapshot=item.get("snapshot", ""),
            )
            anchors.append(anchor)
        except (ValueError, KeyError):
            continue

    return anchors


def compute_confidence(
    reconstruction: str,
    anchors: list[Anchor],
    llm,
) -> ConfidenceReport:
    """Score a reconstruction against its anchors."""

    if not anchors:
        return ConfidenceReport(
            hard_facts=[], soft_claims=[reconstruction], warnings=[], confidence=0.5
        )

    # Verify anchors
    hard = []
    soft = []
    warnings = []

    for anchor in anchors:
        status = anchor.verify()
        if status is True:
            hard.append(f"{anchor.reference} (verified)")
        elif status is False:
            warnings.append(f"{anchor.reference} (no longer verifiable)")
        else:
            # Can't auto-verify — it's a soft claim
            soft.append(anchor.reference)

    # Confidence: proportion of anchors that are verified or at least not contradicted
    total = len(anchors)
    contradicted = len(warnings)
    verified = len(hard)

    if total == 0:
        confidence = 0.5
    else:
        confidence = (verified + 0.5 * len(soft)) / total

    return ConfidenceReport(
        hard_facts=hard,
        soft_claims=soft,
        warnings=warnings,
        confidence=min(1.0, confidence),
    )
