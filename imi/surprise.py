"""Predictive Coding — The theoretical leap.

Instead of storing experiences, store SURPRISE:
  1. Before encoding, PREDICT what you'd expect
  2. Compare prediction with actual experience
  3. Store only the DELTA (surprise)
  4. Reconstruct = generate prediction + apply stored surprise

This unifies: encoding (store surprise), retrieval (predict + apply),
consolidation (absorb surprise into predictor), forgetting (surprise
that became predictable = nothing left to store).

Based on: Rao & Ballard (1999), Friston (2005) Free Energy Principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from imi.llm import LLMAdapter


@dataclass
class SurpriseResult:
    """The output of predictive coding: what was unexpected."""

    prediction: str  # what the model expected
    actual: str  # what actually happened
    surprise: str  # the delta — what was unexpected
    magnitude: float  # 0.0 (completely expected) to 1.0 (completely novel)
    surprise_elements: list[str] = field(default_factory=list)  # itemized surprises

    def __str__(self) -> str:
        lines = [
            f"Surprise magnitude: {self.magnitude:.0%}",
            f"Prediction: {self.prediction[:100]}...",
            f"Surprise: {self.surprise[:150]}...",
        ]
        if self.surprise_elements:
            lines.append("Elements:")
            for e in self.surprise_elements:
                lines.append(f"  ! {e}")
        return "\n".join(lines)


PREDICT_SYSTEM = """\
You are a prediction engine. Given a context description, predict what you \
would EXPECT to happen or to have happened. Be specific and concrete. \
State your prediction confidently — this will be compared against reality. \
Write in the same language as the input. Max 80 tokens."""

SURPRISE_SYSTEM = """\
You are a surprise detection engine. You receive a PREDICTION (what was expected) \
and the ACTUAL experience (what really happened).

Your job:
1. Identify what was SURPRISING — what differed from the prediction.
2. Quantify the surprise magnitude (0.0 = exactly as predicted, 1.0 = completely unexpected).
3. List each surprising element separately.
4. IGNORE what matched the prediction — only output the delta.

Respond in JSON format:
{
  "surprise_summary": "concise description of what was unexpected",
  "magnitude": 0.7,
  "elements": ["element 1", "element 2"]
}

Write in the same language as the input. Return ONLY valid JSON."""

RECONSTRUCT_FROM_SURPRISE_SYSTEM = """\
You are a memory reconstruction engine using predictive coding.

You receive:
- A CONTEXT that tells you what domain/topic this memory is about
- SURPRISE elements — the unexpected parts that were stored

Your job:
1. Generate a baseline PREDICTION of what normally happens in this context
2. APPLY the surprise elements to modify your prediction
3. Output the reconstructed memory — the prediction modified by surprise

The result should read as a coherent memory. Mark what comes from prediction \
vs what comes from stored surprise if relevant.
Write in the same language as the surprise elements."""


def predict(context: str, llm: LLMAdapter) -> str:
    """Generate a prediction of what to expect given a context."""
    return llm.generate(
        system=PREDICT_SYSTEM,
        prompt=f"Context: {context}\n\nWhat would you expect to happen?",
        max_tokens=200,
    )


def compute_surprise(
    prediction: str,
    actual_experience: str,
    llm: LLMAdapter,
) -> SurpriseResult:
    """Compare prediction against reality, extract surprise."""
    import json

    raw = llm.generate(
        system=SURPRISE_SYSTEM,
        prompt=(
            f"PREDICTION:\n{prediction}\n\n"
            f"ACTUAL EXPERIENCE:\n{actual_experience}\n\n"
            "Identify and quantify the surprise."
        ),
        max_tokens=300,
    )

    # Parse JSON
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: treat entire response as surprise
        return SurpriseResult(
            prediction=prediction,
            actual=actual_experience,
            surprise=raw,
            magnitude=0.5,
            surprise_elements=[raw],
        )

    return SurpriseResult(
        prediction=prediction,
        actual=actual_experience,
        surprise=data.get("surprise_summary", raw),
        magnitude=float(data.get("magnitude", 0.5)),
        surprise_elements=data.get("elements", []),
    )


def encode_with_surprise(
    experience: str,
    context_hint: str,
    llm: LLMAdapter,
) -> SurpriseResult:
    """Full predictive coding pipeline: predict → compare → extract surprise."""
    # Step 1: Predict
    prediction = predict(context_hint, llm)

    # Step 2: Compute surprise
    result = compute_surprise(prediction, experience, llm)

    return result


def reconstruct_from_surprise(
    surprise: SurpriseResult,
    current_context: str,
    llm: LLMAdapter,
) -> str:
    """Reconstruct a memory from stored surprise + current context.

    This is the predictive coding version of remember():
    instead of expanding a seed, it generates a prediction and applies surprise.
    """
    elements_text = "\n".join(f"- {e}" for e in surprise.surprise_elements)
    if not elements_text:
        elements_text = surprise.surprise

    return llm.generate(
        system=RECONSTRUCT_FROM_SURPRISE_SYSTEM,
        prompt=(
            f"CONTEXT: {current_context}\n\n"
            f"STORED SURPRISE (magnitude {surprise.magnitude:.0%}):\n"
            f"{elements_text}\n\n"
            "Reconstruct the complete memory."
        ),
        max_tokens=400,
    )
