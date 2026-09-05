"""
PayBack AI - LLM provider abstraction.

Any provider (mock, Claude, or otherwise) implements this same interface.
Providers return RAW TEXT (expected to be JSON) - parsing/validation and
all safety enforcement happens in llm/message_generator.py, never here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProviderError(Exception):
    """Raised by a provider on failure, timeout, or any unusable response."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, timeout_seconds: float = 10.0) -> str:
        """
        Return raw text output (expected to be a JSON object as a string).
        MUST raise LLMProviderError on any failure - never return None or
        silently swallow an error.
        """
        raise NotImplementedError
