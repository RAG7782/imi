"""Adaptive relevance_weight — adjusts rw based on query intent.

Query intents and optimal rw (from WS-A/WS-B empirical data):
  - TEMPORAL ("recent", "latest", "last week") → rw=0.15 (boost recency)
  - EXPLORATORY ("find all", "list", "every") → rw=0.00 (pure cosine)
  - ACTION ("how to", "fix", "prevent", "handle") → rw=0.05 (slight relevance)
  - DEFAULT (anything else) → rw=0.10

Two strategies:
  1. Keyword-based (zero cost, high precision for explicit signals)
  2. Embedding-based (uses the embedder, higher recall but more compute)

Keyword-based is the default — it catches explicit temporal/exploratory cues
with zero overhead. Embedding-based is opt-in for finer-grained classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np


class QueryIntent(str, Enum):
    TEMPORAL = "temporal"       # User wants recent/frequent memories
    EXPLORATORY = "exploratory" # User wants comprehensive search
    ACTION = "action"           # User wants to know what to DO
    DEFAULT = "default"         # General query


# Optimal rw per intent (from WS-A ablation + WS-B temporal experiments)
INTENT_RW = {
    QueryIntent.TEMPORAL: 0.15,
    QueryIntent.EXPLORATORY: 0.00,
    QueryIntent.ACTION: 0.05,
    QueryIntent.DEFAULT: 0.10,
}

# Keyword patterns per intent
TEMPORAL_PATTERNS = re.compile(
    r'\b(recent|latest|last\s+(?:week|month|day|hour|time)|'
    r'today|yesterday|just\s+(?:happened|now|saw)|'
    r'this\s+(?:week|month|sprint)|current|ongoing|active|'
    r'recente|último|última|hoje|ontem|esta\s+semana)\b',
    re.IGNORECASE,
)

EXPLORATORY_PATTERNS = re.compile(
    r'\b(find\s+all|list\s+(?:all|every)|every|all\s+(?:incidents|cases|times)|'
    r'comprehensive|exhaustive|complete\s+list|'
    r'search\s+for|show\s+me\s+all|everything\s+about|'
    r'todos|listar|buscar\s+todos|completo)\b',
    re.IGNORECASE,
)

ACTION_PATTERNS = re.compile(
    r'\b(how\s+to|fix|prevent|handle|resolve|mitigate|avoid|'
    r'what\s+(?:should|can|do)\s+(?:I|we)|steps?\s+to|'
    r'procedure|runbook|action|remediat|'
    r'como|corrigir|prevenir|resolver|evitar)\b',
    re.IGNORECASE,
)


@dataclass
class AdaptiveRW:
    """Adaptive relevance_weight classifier.

    Usage:
        arw = AdaptiveRW()
        rw = arw.classify("recent auth failures")  # → 0.15
        rw = arw.classify("find all cert expiry incidents")  # → 0.00
        rw = arw.classify("how to prevent DNS outages")  # → 0.05
    """

    # Allow override of default mappings
    intent_rw: dict[QueryIntent, float] | None = None

    def _get_rw(self, intent: QueryIntent) -> float:
        if self.intent_rw:
            return self.intent_rw.get(intent, INTENT_RW[intent])
        return INTENT_RW[intent]

    def classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent using keyword patterns."""
        if TEMPORAL_PATTERNS.search(query):
            return QueryIntent.TEMPORAL
        if EXPLORATORY_PATTERNS.search(query):
            return QueryIntent.EXPLORATORY
        if ACTION_PATTERNS.search(query):
            return QueryIntent.ACTION
        return QueryIntent.DEFAULT

    def classify(self, query: str) -> float:
        """Return optimal relevance_weight for this query."""
        intent = self.classify_intent(query)
        return self._get_rw(intent)

    def classify_with_info(self, query: str) -> tuple[float, QueryIntent]:
        """Return (rw, intent) for debugging/logging."""
        intent = self.classify_intent(query)
        return self._get_rw(intent), intent
