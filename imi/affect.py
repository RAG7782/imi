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
from math import log2, sqrt

from imi.llm import LLMAdapter


@dataclass
class AffectiveTag:
    """Emotional/importance signature of a memory."""

    salience: float = 0.5  # 0.0 (routine) to 1.0 (critical)
    valence: float = 0.0  # -1.0 (very negative) to +1.0 (very positive)
    arousal: float = 0.5  # 0.0 (calm/background) to 1.0 (urgent/activating)
    _base_salience: float = -1.0  # initial salience before dynamic updates (-1 = not set)

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
        emotional_intensity = sqrt(self.valence**2 + self.arousal**2) / sqrt(2)
        return self.salience * (0.3 + 0.7 * emotional_intensity)

    @property
    def initial_mass(self) -> float:
        """Initial gravitational mass in the memory space."""
        return max(0.1, self.encoding_strength)

    def update_dynamic(self, access_count: int) -> None:
        """S06 — MemoryWorth salience dinâmica.

        Aumenta salience proporcionalmente ao uso observado.
        Fórmula: salience = min(0.95, base_salience + 0.05 * log2(access_count + 1))

        base_salience é o valor inicial (capturado na primeira chamada) — garante
        que a curva de crescimento é relativa ao ponto de partida, não cumulativa.

        Exemplos (base=0.5):
          access=0  → 0.50 (sem mudança)
          access=1  → 0.55 (log2(2)=1 → +0.05)
          access=3  → 0.60 (log2(4)=2 → +0.10)
          access=7  → 0.65 (log2(8)=3 → +0.15)
          access=15 → 0.70 (log2(16)=4 → +0.20)

        Cap em 0.95 para nunca tornar um nó "absoluto".
        """
        if access_count <= 0:
            return
        # Captura base na primeira chamada dinâmica
        if self._base_salience < 0:
            self._base_salience = self.salience
        boost = 0.05 * log2(access_count + 1)
        self.salience = min(0.95, self._base_salience + boost)

    def to_dict(self) -> dict:
        d = {
            "salience": self.salience,
            "valence": self.valence,
            "arousal": self.arousal,
        }
        # H5: persist _base_salience to prevent upward drift on reload
        if self._base_salience >= 0:
            d["_base_salience"] = self._base_salience
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AffectiveTag:
        # H5: restore _base_salience if persisted; tolerates unknown keys (L2-style guard)
        return cls(
            salience=d.get("salience", 0.5),
            valence=d.get("valence", 0.0),
            arousal=d.get("arousal", 0.5),
            _base_salience=d.get("_base_salience", -1.0),
        )

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
