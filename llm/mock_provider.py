"""
PayBack AI - Mock LLM provider.

Deterministic, requires no API key, and is the DEFAULT provider whenever a
real provider is unavailable or unconfigured. Produces plausible, safe
customer-message JSON so the rest of the pipeline (parsing, Pydantic
validation, content filtering) is exercised the same way it would be with
a real provider.
"""
from __future__ import annotations

import json

from llm.provider import LLMProvider, LLMProviderError

# Deterministic, action-aware templates. Kept generic and safe by
# construction - no invented discounts, no payment-status claims.
_ACTION_TEMPLATES: dict[str, str] = {
    "delayed_retry": (
        "We noticed an issue with your recent payment and will automatically "
        "retry it shortly. No action is needed from you right now."
    ),
    "alternate_payment_link": (
        "We were unable to process your recent payment. Please use the secure "
        "payment link we've shared to complete it using another method."
    ),
    "reminder": (
        "This is a reminder that your recent payment did not go through. "
        "Please retry at your convenience."
    ),
    "human_escalation": (
        "We're reviewing an issue with your recent payment, and our team will "
        "follow up with you shortly."
    ),
    "stop_recovery": (
        "We were unable to process your recent payment. Please contact support "
        "if you'd like to complete it manually."
    ),
    "no_action": "",
}


class MockLLMProvider(LLMProvider):
    """
    Deterministic provider. Optional test hooks (`always_fail`,
    `return_invalid_json`) let tests exercise message_generator's fallback
    path without needing a real, flaky external service.
    """

    def __init__(self, always_fail: bool = False, return_invalid_json: bool = False):
        self.always_fail = always_fail
        self.return_invalid_json = return_invalid_json

    def generate(self, system_prompt: str, user_prompt: str, timeout_seconds: float = 10.0) -> str:
        if self.always_fail:
            raise LLMProviderError("MockLLMProvider configured to always fail (test mode).")
        if self.return_invalid_json:
            return "this is not valid json {{{"

        action = _extract_action_from_prompt(user_prompt)
        message = _ACTION_TEMPLATES.get(action, "We're reviewing your recent payment and will follow up shortly.")
        return json.dumps({"customer_message": message, "llm_confidence": 0.75})


def _extract_action_from_prompt(user_prompt: str) -> str:
    """Best-effort extraction of the action name embedded in the prompt by message_generator."""
    for action in _ACTION_TEMPLATES:
        if f"action: {action}" in user_prompt:
            return action
    return ""
