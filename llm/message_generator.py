"""
PayBack AI - LLM message generator.

Orchestrates: build a constrained prompt -> call the provider with a hard
timeout -> parse as JSON -> validate with Pydantic -> run a content safety
filter -> on ANY failure at any step, fall back to a deterministic, safe
template message instead.

The LLM is used ONLY to draft `customer_message`. It never decides action,
confidence, or requires_human_review - those come from the decision engine
(Stage 8) and pass through unchanged, matching the required output schema:
  {
    "action": "...",
    "reason": "...",
    "customer_message": "...",
    "confidence": 0.0,
    "requires_human_review": true
  }
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sqlite3
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from core.audit_logger import log_event
from core.decision_engine import DecisionOutput, make_decision
from llm.mock_provider import _ACTION_TEMPLATES, MockLLMProvider
from llm.provider import LLMProvider, LLMProviderError

SYSTEM_PROMPT = (
    "You are drafting a short customer-facing message about a failed payment for "
    "PayBack AI, a synthetic revenue-recovery prototype. Rules you MUST follow: "
    "1) Do not claim the payment succeeded or failed with certainty beyond what is stated. "
    "2) Never invent discounts, refunds, fees, or policies. "
    "3) Never reveal internal risk scores, model confidence, or account/customer identifiers. "
    "4) Keep the message under 300 characters, polite, and actionable. "
    "5) Respond with ONLY a JSON object: "
    '{"customer_message": "...", "llm_confidence": 0.0} - no other text.'
)

# Content the LLM must never produce - checked in code, not just instructed.
FORBIDDEN_SUBSTRINGS = [
    "discount",
    "refund",
    "% off",
    "cashback",
    "guarantee",
    "guarantee",
    "promise",
    "free ",
    "payment was successful",
    "payment has succeeded",
    "confirmed successful",
    "your payment is confirmed",
]


class RawLLMMessageResponse(BaseModel):
    customer_message: str = Field(..., min_length=1, max_length=500)
    llm_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class DecisionMessageOutput(BaseModel):
    action: str
    reason: str
    customer_message: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    requires_human_review: bool

    def to_dict(self) -> dict:
        return self.model_dump()


@dataclass(frozen=True)
class MessageGenerationResult:
    output: DecisionMessageOutput
    used_fallback: bool
    provider_name: str


def deterministic_fallback_message(action: str) -> str:
    return _ACTION_TEMPLATES.get(
        action, "We're reviewing your recent payment and will follow up shortly."
    )


def _contains_forbidden_content(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in FORBIDDEN_SUBSTRINGS)


def _build_user_prompt(decision: DecisionOutput) -> str:
    return (
        f"action: {decision.action}\n"
        f"internal_reason: {decision.reason}\n"
        "Draft a short customer-facing message appropriate for this action. "
        "Do not mention 'internal_reason' verbatim - rephrase for a customer audience."
    )


def _call_with_timeout(
    provider: LLMProvider, system_prompt: str, user_prompt: str, timeout_seconds: float
) -> str:
    """Enforce a real wall-clock timeout regardless of the provider implementation."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(provider.generate, system_prompt, user_prompt, timeout_seconds)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            raise LLMProviderError(
                f"Provider timed out after {timeout_seconds} seconds."
            ) from exc


def select_default_provider() -> tuple[LLMProvider, str]:
    """
    Real Claude provider if an API key AND the anthropic package are both
    available; MockLLMProvider otherwise. Never raises - always returns a
    usable provider, since the app must run without any API key.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from llm.claude_provider import ClaudeProvider

            return ClaudeProvider(api_key=api_key), "claude"
        except LLMProviderError:
            pass
    return MockLLMProvider(), "mock"


def generate_message(
    decision: DecisionOutput,
    provider: LLMProvider | None = None,
    timeout_seconds: float = 10.0,
) -> MessageGenerationResult:
    """
    Generate the customer-facing message for a decision. Falls back to a
    deterministic template on ANY failure: timeout, invalid JSON, Pydantic
    validation error, or forbidden content.
    """
    provider_name = "mock"
    if provider is None:
        provider, provider_name = select_default_provider()
    else:
        provider_name = type(provider).__name__

    used_fallback = False

    if decision.action == "no_action":
        customer_message = ""
    else:
        try:
            user_prompt = _build_user_prompt(decision)
            raw_text = _call_with_timeout(provider, SYSTEM_PROMPT, user_prompt, timeout_seconds)
            parsed = json.loads(raw_text)
            validated = RawLLMMessageResponse(**parsed)
            if _contains_forbidden_content(validated.customer_message):
                raise ValueError("LLM output contained forbidden content; discarding.")
            customer_message = validated.customer_message
        except (LLMProviderError, json.JSONDecodeError, ValidationError, ValueError, TypeError):
            used_fallback = True
            customer_message = deterministic_fallback_message(decision.action)

    output = DecisionMessageOutput(
        action=decision.action,
        reason=decision.reason,
        customer_message=customer_message,
        confidence=decision.confidence,
        requires_human_review=decision.requires_human_review,
    )
    return MessageGenerationResult(output=output, used_fallback=used_fallback, provider_name=provider_name)


def generate_message_for_transaction(
    conn: sqlite3.Connection,
    transaction_id: str,
    provider: LLMProvider | None = None,
    timeout_seconds: float = 10.0,
) -> MessageGenerationResult:
    """Look up (or compute) the decision for a transaction and generate its message, with audit logging."""
    decision = make_decision(conn, transaction_id)
    result = generate_message(decision, provider=provider, timeout_seconds=timeout_seconds)

    log_event(
        conn,
        transaction_id,
        actor="message_generator",
        event_type="customer_message_generated",
        detail=(
            f"provider={result.provider_name} used_fallback={result.used_fallback} "
            f"message={result.output.customer_message!r}"
        ),
    )
    return result
