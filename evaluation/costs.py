"""
PayBack AI - Cost ASSUMPTIONS for evaluation.

These are ASSUMPTIONS used to illustrate cost-per-recovery, NOT Razorpay's
actual internal costs. Configurable here in one place so the assumption is
easy to find, question, and change.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostAssumptions:
    retry_cost_inr: float = 2.0  # ASSUMPTION: cost of one automated retry attempt
    message_cost_inr: float = 0.5  # ASSUMPTION: cost of one customer message (SMS/email/push)
    human_review_cost_inr: float = 50.0  # ASSUMPTION: cost of one human review/escalation
