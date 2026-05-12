"""LLM adapter — abstracts the language model used for reconstruction."""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


class LLMAdapter(Protocol):
    """Protocol for any LLM backend."""

    def generate(
        self, system: str, prompt: str, max_tokens: int = 1024, temperature: float | None = None
    ) -> str: ...


@dataclass
class ClaudeLLM:
    """Anthropic Claude adapter via API key (ANTHROPIC_API_KEY)."""

    model: str = "claude-sonnet-4-20250514"
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        import anthropic

        self._client = anthropic.Anthropic()

    def generate(
        self, system: str, prompt: str, max_tokens: int = 1024, temperature: float | None = None
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self._client.messages.create(**kwargs)
        return response.content[0].text


@dataclass
class ClaudeCodeLLM:
    """Claude Code CLI adapter — authenticates via OAuth (Claude Max account).

    Uses `claude -p` (print mode) to invoke the model without API key.
    Set IMI_LLM_BACKEND=claude-code in env to activate automatically.
    """

    model: str = "claude-sonnet-4-5"

    def generate(
        self, system: str, prompt: str, max_tokens: int = 1024, temperature: float | None = None
    ) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        # --effort low keeps responses concise; claude CLI has no --max-tokens flag
        effort = "low" if max_tokens <= 512 else "medium"
        cmd = [
            "claude",
            "-p",
            full_prompt,
            "--model",
            self.model,
            "--effort",
            effort,
            "--output-format",
            "text",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ClaudeCodeLLM error: {result.stderr.strip()}")
        return result.stdout.strip()


@dataclass
class OllamaLLM:
    """Ollama local adapter — zero cost, no API key needed.

    Default model: phi4-mini (2.5GB, ~4s latency, clean JSON output).
    Set IMI_LLM_BACKEND=ollama and optionally IMI_LLM_MODEL=<model> in env.

    Benchmark results (2026-04-30):
      phi4-mini   → 4.2s, clean JSON after strip, recommended default
      qwen3.5:4b  → 5.9s, best semantic quality but verbose thinking
      deepseek-coder:6.7b → 9.5s, descriptive not analytical
    """

    model: str = "phi4-mini:latest"
    base_url: str = "http://localhost:11434"

    def generate(
        self, system: str, prompt: str, max_tokens: int = 1024, temperature: float | None = None
    ) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature if temperature is not None else 0.1,
                    "num_predict": min(max_tokens, 512),
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        raw = data.get("response", "").strip()
        return self._strip_markdown(raw)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove ```json ... ``` wrapper that phi4-mini adds."""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()


def create_llm_from_env() -> "ClaudeLLM | ClaudeCodeLLM | OllamaLLM":
    """Factory — selects backend based on IMI_LLM_BACKEND env var.

    IMI_LLM_BACKEND=ollama       → OllamaLLM (local, free, phi4-mini default)
    IMI_LLM_BACKEND=claude-code  → ClaudeCodeLLM (OAuth, Claude Max)
    IMI_LLM_BACKEND=api          → ClaudeLLM (ANTHROPIC_API_KEY)
    (unset)                      → ollama if no API key, else api
    """
    backend = os.environ.get("IMI_LLM_BACKEND", "").lower()
    if backend == "ollama":
        model = os.environ.get("IMI_LLM_MODEL", "phi4-mini:latest")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaLLM(model=model, base_url=base_url)
    if backend == "claude-code":
        return ClaudeCodeLLM()
    if backend == "api":
        return ClaudeLLM()
    # Auto-detect: prefer ollama when no API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        model = os.environ.get("IMI_LLM_MODEL", "phi4-mini:latest")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaLLM(model=model, base_url=base_url)
    return ClaudeLLM()
