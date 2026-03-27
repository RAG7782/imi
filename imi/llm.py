"""LLM adapter — abstracts the language model used for reconstruction."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


class LLMAdapter(Protocol):
    """Protocol for any LLM backend."""

    def generate(self, system: str, prompt: str, max_tokens: int = 1024) -> str: ...


@dataclass
class ClaudeLLM:
    """Anthropic Claude adapter."""

    model: str = "claude-sonnet-4-20250514"
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        import anthropic

        self._client = anthropic.Anthropic()

    def generate(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
