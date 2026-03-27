"""Memory Reconsolidation — memories change on access.

Every time a memory is recalled, it enters a LABILE state and is
reconsolidated — potentially modified by current context.

This is not a bug. It is the mechanism by which memories stay relevant.

Based on: Nader, Schafe & LeDoux (2000).

The cycle:
  stable → access → labile → reconsolidate → stable (potentially modified)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from imi.core import summarize
from imi.llm import LLMAdapter
from imi.node import MemoryNode


class MemoryState(str, Enum):
    STABLE = "stable"
    LABILE = "labile"
    RECONSOLIDATING = "reconsolidating"


@dataclass
class ReconsolidationEvent:
    """Record of a reconsolidation that occurred."""

    node_id: str
    timestamp: float
    context: str              # what context triggered the reconsolidation
    changes: list[str]        # what changed
    previous_orbital: str     # snapshot before reconsolidation
    new_orbital: str          # snapshot after

    def __str__(self) -> str:
        return (
            f"Reconsolidation [{self.node_id}]:\n"
            f"  Context: {self.context[:80]}\n"
            f"  Before: {self.previous_orbital}\n"
            f"  After:  {self.new_orbital}\n"
            f"  Changes: {', '.join(self.changes)}"
        )


def reconsolidate(
    node: MemoryNode,
    access_context: str,
    llm: LLMAdapter,
    force: bool = False,
) -> ReconsolidationEvent | None:
    """Reconsolidate a memory after access.

    Only reconsolidates if:
    - It hasn't been reconsolidated recently (cooldown)
    - The access context is significantly different from current summaries
    - force=True overrides cooldown

    Returns the reconsolidation event, or None if no change was needed.
    """
    # Cooldown: don't reconsolidate the same memory too frequently
    cooldown_hours = 1.0
    hours_since_access = (time.time() - node.last_accessed) / 3600
    if not force and hours_since_access < cooldown_hours:
        return None

    # Check if reconsolidation is warranted
    # (is the access context bringing new perspective?)
    relevance_check = llm.generate(
        system=(
            "You decide if a memory should be updated based on new context. "
            "Answer with JSON: {\"should_update\": true/false, \"reason\": \"...\", "
            "\"new_framing\": \"...\"}\n"
            "Only update if the new context genuinely adds insight or reframes the memory. "
            "Do NOT update if the context is unrelated."
        ),
        prompt=(
            f"CURRENT MEMORY SUMMARY:\n{node.summary_medium}\n\n"
            f"NEW ACCESS CONTEXT:\n{access_context}\n\n"
            "Should this memory's summary be updated to reflect the new context?"
        ),
        max_tokens=200,
    )

    import json
    try:
        data = json.loads(relevance_check.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        return None

    if not data.get("should_update", False):
        return None

    # --- RECONSOLIDATION ---
    previous_orbital = node.summary_orbital
    new_framing = data.get("new_framing", "")
    reason = data.get("reason", "")

    # Update summaries with new context integration
    combined = f"{node.seed}\n\nNew perspective: {new_framing}"
    node.summary_orbital = summarize(combined, max_tokens=10, llm=llm)
    node.summary_medium = summarize(combined, max_tokens=40, llm=llm)

    return ReconsolidationEvent(
        node_id=node.id,
        timestamp=time.time(),
        context=access_context,
        changes=[reason],
        previous_orbital=previous_orbital,
        new_orbital=node.summary_orbital,
    )
