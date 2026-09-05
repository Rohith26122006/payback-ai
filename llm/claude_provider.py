"""
PayBack AI - Optional real Claude provider.

Only used if BOTH the `anthropic` package is installed AND ANTHROPIC_API_KEY
is set. Any failure (missing package, missing key, API error, bad response)
raises LLMProviderError so message_generator falls back to MockLLMProvider.
This module is never imported at package load time - only lazily, so its
absence/failure can never break the application.
"""
from __future__ import annotations

from llm.provider import LLMProvider, LLMProviderError

DEFAULT_MODEL = "claude-sonnet-4-6"


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise LLMProviderError("ClaudeProvider requires a non-empty API key.")
        try:
            import anthropic  # noqa: F401  (import validated here, used below)
        except ImportError as exc:
            raise LLMProviderError(
                "The 'anthropic' package is not installed; cannot use ClaudeProvider."
            ) from exc

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, timeout_seconds: float = 10.0) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure -> safe fallback
            raise LLMProviderError(f"Claude API call failed: {exc}") from exc

        text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        if not text_blocks:
            raise LLMProviderError("Claude API returned no text content.")
        return "\n".join(text_blocks)
