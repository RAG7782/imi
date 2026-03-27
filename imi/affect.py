"""Affective Tagging — the emotional dimension of memory.

Operationalizes 'emotion' for AI agents:
  - Salience: how important/significant was this? (error = high, routine = low)
  - Valence: positive or negative? (success = +, failure = -)
  - Arousal: how urgent/activating? (incident = high, refactor = low)

Affect modulates: encoding strength (initial mass), fade resistance,
retrieval priority. High-affect memories are remembered more vividly
and resist forgetting — exactly as in human memory.

Based on: Damasio's somatic marker hypothesis, LeDoux's emotional memory circuits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt

from imi.llm import LLMAdapter


@dataclass
class AffectiveTag:
    """Emotional/importance signature of a memory."""

    salience: float = 0.5    # 0.0 (routine) to 1.0 (critical)
    valence: float = 0.0     # -1.0 (very negative) to +1.0 (very positive)
    arousal: float = 0.5     # 0.0 (calm/background) to 1.0 (urgent/activating)

    @property
    def encoding_strength(self) -> float:
        """How strongly this memory should be encoded.

        High salience × high arousal = strong encoding (like adrenaline in humans).
        """
        return self.salience * (0.5 + 0.5 * self.arousal)

    @property
    def fade_resistance(self) -> float:
        """How resistant to forgetting this memory is.

        High-affect memories (positive or negative) resist fade.
        """
        emotional_intensity = sqrt(self.valence ** 2 + self.arousal ** 2) / sqrt(2)
        return self.salience * (0.3 + 0.7 * emotional_intensity)

    @property
    def initial_mass(self) -> float:
        """Initial gravitational mass in the memory space."""
        return max(0.1, self.encoding_strength)

    def to_dict(self) -> dict:
        return {
            "salience": self.salience,
            "valence": self.valence,
            "arousal": self.arousal,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AffectiveTag:
        return cls(**d)

    def __str__(self) -> str:
        valence_str = "+" if self.valence >= 0 else ""
        return (
            f"Affect(salience={self.salience:.1f}, "
            f"valence={valence_str}{self.valence:.1f}, "
            f"arousal={self.arousal:.1f} "
            f"→ strength={self.encoding_strength:.2f}, "
            f"fade_resist={self.fade_resistance:.2f})"
        )


ASSESS_AFFECT_SYSTEM = """\
You assess the emotional/importance signature of experiences for an AI agent.

Rate the experience on three dimensions:
- salience: 0.0 (completely routine) to 1.0 (critical/pivotal moment)
- valence: -1.0 (very negative: failure, error, loss) to +1.0 (very positive: success, breakthrough)
- arousal: 0.0 (calm, background task) to 1.0 (urgent, high-stakes, time-sensitive)

Examples:
- Routine refactor: salience=0.2, valence=0.1, arousal=0.1
- Bug fix that saved production: salience=0.8, valence=0.6, arousal=0.7
- P1 incident with data loss: salience=1.0, valence=-0.9, arousal=1.0
- New feature launched successfully: salience=0.6, valence=0.7, arousal=0.4

Respond with ONLY valid JSON: {"salience": 0.X, "valence": 0.X, "arousal": 0.X}"""


def assess_affect(experience: str, llm: LLMAdapter) -> AffectiveTag:
    """Use LLM to assess the affective signature of an experience."""
    raw = llm.generate(
        system=ASSESS_AFFECT_SYSTEM,
        prompt=f"Experience:\n{experience}",
        max_tokens=60,
    )

    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        data = json.loads(raw)
        return AffectiveTag(
            salience=max(0.0, min(1.0, float(data.get("salience", 0.5)))),
            valence=max(-1.0, min(1.0, float(data.get("valence", 0.0)))),
            arousal=max(0.0, min(1.0, float(data.get("arousal", 0.5)))),
        )
    except (json.JSONDecodeError, ValueError):
        return AffectiveTag()  # defaults
