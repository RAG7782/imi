"""
IMI Core — The irreducible atom.

The 1/99 insight: memory is not data, memory is function.
  remember(seed, context) → reconstruction

Everything else in IMI is elaboration on this single idea.
"""

from __future__ import annotations

from imi.llm import LLMAdapter, ClaudeLLM

# ---------------------------------------------------------------------------
# Default LLM (lazy singleton)
# ---------------------------------------------------------------------------
_default_llm: LLMAdapter | None = None


def get_llm() -> LLMAdapter:
    global _default_llm
    if _default_llm is None:
        _default_llm = ClaudeLLM()
    return _default_llm


def set_llm(llm: LLMAdapter) -> None:
    global _default_llm
    _default_llm = llm


# ---------------------------------------------------------------------------
# The Atom: compress + remember
# ---------------------------------------------------------------------------

COMPRESS_SYSTEM = """\
You are a memory compression engine. Your job is to extract the essential seed \
from an experience. A seed is NOT a summary — it is the minimal set of facts \
needed to reconstruct the experience later.

Extract: theme, trigger/cause, actors involved, key actions, outcome, date/time \
if available. Maximum {max_tokens} tokens. Be telegraphic. Preserve proper nouns, \
numbers, and specific technical terms exactly."""

REMEMBER_SYSTEM = """\
You are a memory reconstruction engine. You receive a compressed seed from a \
past experience and the current context. Your job is to reconstruct the memory \
— not invent new facts, but flesh out the seed into a coherent recollection.

Rules:
- Stay faithful to the seed. Do not add facts not implied by it.
- Use the current context to frame the reconstruction relevantly.
- If the seed mentions specific names, dates, or technical terms, preserve them exactly.
- Clearly distinguish what the seed states from what you are inferring.
- Write in the same language as the seed."""

SUMMARIZE_SYSTEM = """\
Summarize the following experience in at most {max_tokens} tokens. \
Be telegraphic. Preserve key entities and outcomes. \
Write in the same language as the input."""


def compress_seed(
    experience: str,
    *,
    max_tokens: int = 80,
    llm: LLMAdapter | None = None,
    domain: str = "",
) -> str:
    """Compress an experience into a seed — the minimal reconstruction key.

    When domain is provided, applies SDE Densify operation: the compression
    prompt instructs the LLM to prefer high-density domain terms over generic
    ones, increasing distributional semiotic density (DS-d) of the seed.
    """
    from .dialect import densify_prompt

    llm = llm or get_llm()
    system = COMPRESS_SYSTEM.format(max_tokens=max_tokens)
    if domain:
        system += densify_prompt(domain)
    return llm.generate(
        system=system,
        prompt=experience,
        max_tokens=max_tokens * 2,  # allow some headroom
    )


def remember(
    seed: str,
    context: str = "",
    *,
    llm: LLMAdapter | None = None,
) -> str:
    """
    Reconstruct a memory from its seed, informed by current context.

    This is the irreducible atom of IMI:
      memory is not data retrieved — it is experience reconstructed.
    """
    llm = llm or get_llm()
    prompt_parts = [f"SEED:\n{seed}"]
    if context:
        prompt_parts.append(f"\nCURRENT CONTEXT:\n{context}")
    prompt_parts.append(
        "\nReconstruct this memory. Be faithful to the seed, "
        "use the context to frame relevance."
    )
    return llm.generate(
        system=REMEMBER_SYSTEM,
        prompt="\n".join(prompt_parts),
        max_tokens=512,
    )


def summarize(
    text: str,
    *,
    max_tokens: int = 30,
    llm: LLMAdapter | None = None,
) -> str:
    """Create a summary at a specific token budget."""
    llm = llm or get_llm()
    return llm.generate(
        system=SUMMARIZE_SYSTEM.format(max_tokens=max_tokens),
        prompt=text,
        max_tokens=max_tokens * 2,
    )
