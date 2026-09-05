"""
PayBack AI - Shared API response schemas.
"""
from __future__ import annotations

from pydantic import BaseModel


class PaymentSummary(BaseModel):
    transaction_id: str
    merchant_id: str
    amount_inr: float
    payment_method: str
    failure_category: str
    attempt_number: int
    event_timestamp: str


class SafetyCheckSchema(BaseModel):
    rule: str
    triggered: bool
    detail: str


class DecisionResponse(BaseModel):
    transaction_id: str
    action: str
    recovery_probability: float | None
    confidence: float
    reason: str
    safety_checks: list[SafetyCheckSchema]
    requires_human_review: bool


class PaymentDetail(PaymentSummary):
    customer_id_hash: str
    failure_code: str
    subscription_flag: bool
    device_change_flag: bool
    merchant_category: str
    classified_category: str | None = None
    classification_confidence: float | None = None
    recovery_probability: float | None = None
    decision: DecisionResponse | None = None
    baseline_action_status: str | None = None
    payback_ai_action_status: str | None = None
