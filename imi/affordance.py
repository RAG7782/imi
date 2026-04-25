"""Affordances — memories as action potentials.

A memory is not just information — it's a potential for future action.
'I remember how I fixed the auth bug' = affordance for fixing similar bugs.

Based on: Gibson (1979) ecological psychology.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from imi.llm import LLMAdapter


@dataclass
class Affordance:
    """An action potential derived from a memory."""

    action: str              # what this memory enables doing
    confidence: float        # 0.0-1.0 how confident we are this is applicable
    conditions: str          # when/where this affordance applies
    domain: str = ""         # domain tag (e.g., "debugging", "architecture")

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "conditions": self.conditions,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Affordance:
        # L2 fix: field guard — only pass known fields to avoid TypeError on future keys
        known = {"action", "confidence", "conditions", "domain"}
        return cls(**{k: v for k, v in d.items() if k in known})

    def __str__(self) -> str:
        return f"[{self.confidence:.0%}] {self.action} (when: {self.conditions})"


EXTRACT_AFFORDANCES_SYSTEM = """\
You extract ACTION POTENTIALS from experiences. An affordance is something \
the agent CAN DO in the future because of what it learned from this experience.

For each affordance, specify:
- action: what can be done (verb phrase)
- confidence: 0.0-1.0 how reusable this is
- conditions: when/where this applies
- domain: category (e.g., "debugging", "architecture", "process", "tooling")

Return JSON array. Max 4 affordances per experience.
Return ONLY valid JSON array, no markdown fences.
Write in the same language as the input."""


def extract_affordances(experience: str, llm: LLMAdapter) -> list[Affordance]:
    """Extract action potentials from an experience."""
    raw = llm.generate(
        system=EXTRACT_AFFORDANCES_SYSTEM,
        prompt=f"Experience:\n{experience}\n\nWhat future actions does this enable?",
        max_tokens=400,
        temperature=0.3,
    )

    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []

    affordances = []
    for item in items:
        try:
            affordances.append(Affordance(
                action=item.get("action", ""),
                confidence=float(item.get("confidence", 0.5)),
                conditions=item.get("conditions", ""),
                domain=item.get("domain", ""),
            ))
        except (ValueError, KeyError):
            continue

    return affordances
