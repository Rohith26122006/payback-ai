"""
PayBack AI - Policy and safety engine.

THIS ENGINE IS AUTHORITATIVE. Neither the ML recovery model nor the LLM
message generator may override its decisions. It only ever RESTRICTS or
REPLACES a proposed action - it never invents a riskier one.

Safeguards implemented (defaults, all configurable via PolicyConfig):
  - Maximum automatic retries: 2
  - Minimum retry cooldown: 30 minutes
  - Maximum customer messages: 1 per 24 hours
  - No automatic retry for suspected-risk transactions
  - Escalate uncertain (low classifier confidence) or high-value transactions
  - Stop all pending actions after payment success
  - Prevent duplicate action execution
  - Escalate unknown failure types
  - Deterministic fallback if the model or LLM output is invalid/unavailable

Every rule's outcome - fired or not - is recorded in `safety_checks` for
the audit trail, even when it didn't change anything.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

VALID_ACTIONS = {
    "delayed_retry",
    "alternate_payment_link",
    "reminder",
    "human_escalation",
    "stop_recovery",
    "no_action",
}


@dataclass(frozen=True)
class PolicyConfig:
    max_auto_retries: int = 2
    retry_cooldown_minutes: int = 30
    max_messages_per_24h: int = 1
    high_value_threshold_inr: float = 50000.0
    low_confidence_threshold: float = 0.5


@dataclass(frozen=True)
class PolicyInput:
    transaction_id: str
    proposed_action: str
    failure_category: str
    amount_inr: float
    attempt_number: int
    classification_confidence: float
    recovery_probability: float | None
    retry_count_so_far: int
    minutes_since_last_action: float | None  # None = no prior recorded action
    messages_sent_last_24h: int
    already_recovered: bool
    duplicate_action_pending: bool
    model_or_llm_failed: bool = False


@dataclass(frozen=True)
class SafetyCheck:
    rule: str
    triggered: bool
    detail: str

    def to_dict(self) -> dict:
        return {"rule": self.rule, "triggered": self.triggered, "detail": self.detail}


@dataclass(frozen=True)
class PolicyResult:
    final_action: str
    requires_human_review: bool
    safety_checks: list[SafetyCheck] = field(default_factory=list)
    was_overridden: bool = False


def deterministic_fallback_action(
    failure_category: str, attempt_number: int, config: PolicyConfig
) -> str:
    """
    Safe, rule-only action used when the ML model or LLM output is invalid
    or unavailable. Ignores everything except category and attempt count.
    """
    if failure_category in ("suspected_risk", "unknown"):
        return "human_escalation"
    if attempt_number <= config.max_auto_retries:
        return "delayed_retry"
    return "stop_recovery"


def evaluate_policy(inp: PolicyInput, config: PolicyConfig | None = None) -> PolicyResult:
    config = config or PolicyConfig()
    checks: list[SafetyCheck] = []
    action = inp.proposed_action
    overridden = False
    requires_review = False

    # 0. Deterministic fallback if model/LLM output failed - replaces the
    #    proposed action entirely, before any other rule is even considered.
    if inp.model_or_llm_failed:
        fallback = deterministic_fallback_action(inp.failure_category, inp.attempt_number, config)
        checks.append(
            SafetyCheck(
                "deterministic_fallback",
                True,
                f"Model/LLM output was invalid or unavailable; using deterministic "
                f"fallback action '{fallback}' instead of proposed '{action}'.",
            )
        )
        action = fallback
        overridden = True
    else:
        checks.append(
            SafetyCheck("deterministic_fallback", False, "Model and LLM output were valid.")
        )

    # 1. Stop after success - highest priority. Nothing below can override this.
    if inp.already_recovered:
        if action != "no_action":
            checks.append(
                SafetyCheck(
                    "stop_after_success",
                    True,
                    f"Transaction already recovered; overriding '{action}' to 'no_action'.",
                )
            )
            action = "no_action"
            overridden = True
        else:
            checks.append(
                SafetyCheck("stop_after_success", False, "Already recovered; no action needed.")
            )
        return PolicyResult(
            final_action=action,
            requires_human_review=False,
            safety_checks=checks,
            was_overridden=overridden,
        )
    checks.append(SafetyCheck("stop_after_success", False, "Transaction not yet recovered."))

    # 2. Prevent duplicate execution.
    if inp.duplicate_action_pending:
        if action != "no_action":
            checks.append(
                SafetyCheck(
                    "prevent_duplicate_execution",
                    True,
                    f"An equivalent action was just executed for this transaction; "
                    f"overriding '{action}' to 'no_action'.",
                )
            )
            action = "no_action"
            overridden = True
        else:
            checks.append(
                SafetyCheck("prevent_duplicate_execution", False, "No duplicate pending.")
            )
    else:
        checks.append(SafetyCheck("prevent_duplicate_execution", False, "No duplicate pending."))

    # 3. No auto-retry for suspected risk.
    if inp.failure_category == "suspected_risk" and action == "delayed_retry":
        checks.append(
            SafetyCheck(
                "no_auto_retry_suspected_risk",
                True,
                "Suspected-risk transactions are never auto-retried; "
                "overriding to 'human_escalation'.",
            )
        )
        action = "human_escalation"
        overridden = True
    else:
        checks.append(
            SafetyCheck("no_auto_retry_suspected_risk", False, "Not applicable.")
        )

    # 4. Escalate unknown failure category.
    if inp.failure_category == "unknown" and action not in ("human_escalation", "no_action"):
        checks.append(
            SafetyCheck(
                "escalate_unknown_failure",
                True,
                f"Unknown failure category; overriding '{action}' to 'human_escalation'.",
            )
        )
        action = "human_escalation"
        overridden = True
    else:
        checks.append(SafetyCheck("escalate_unknown_failure", False, "Not applicable."))

    # 5. Maximum automatic retries.
    if action == "delayed_retry" and inp.retry_count_so_far >= config.max_auto_retries:
        checks.append(
            SafetyCheck(
                "max_auto_retries",
                True,
                f"Retry count {inp.retry_count_so_far} reached the configured maximum "
                f"({config.max_auto_retries}); overriding to 'stop_recovery'.",
            )
        )
        action = "stop_recovery"
        overridden = True
    else:
        checks.append(
            SafetyCheck(
                "max_auto_retries",
                False,
                f"Retry count {inp.retry_count_so_far} within limit ({config.max_auto_retries}).",
            )
        )

    # 6. Retry cooldown.
    if (
        action == "delayed_retry"
        and inp.minutes_since_last_action is not None
        and inp.minutes_since_last_action < config.retry_cooldown_minutes
    ):
        checks.append(
            SafetyCheck(
                "retry_cooldown",
                True,
                f"Only {inp.minutes_since_last_action:.0f} min since last action "
                f"(< {config.retry_cooldown_minutes} min cooldown); overriding to 'no_action'.",
            )
        )
        action = "no_action"
        overridden = True
    else:
        checks.append(SafetyCheck("retry_cooldown", False, "Cooldown satisfied or not applicable."))

    # 7. Message limit (24h).
    if action == "reminder" and inp.messages_sent_last_24h >= config.max_messages_per_24h:
        checks.append(
            SafetyCheck(
                "max_messages_per_24h",
                True,
                f"Already sent {inp.messages_sent_last_24h} message(s) in the last 24h "
                f"(limit {config.max_messages_per_24h}); overriding to 'stop_recovery'.",
            )
        )
        action = "stop_recovery"
        overridden = True
    else:
        checks.append(
            SafetyCheck("max_messages_per_24h", False, "Message limit not exceeded or not applicable.")
        )

    # 8. Escalate uncertain or high-value transactions.
    is_high_value = inp.amount_inr > config.high_value_threshold_inr
    is_uncertain = inp.classification_confidence < config.low_confidence_threshold
    if is_high_value or is_uncertain:
        requires_review = True
        reasons = []
        if is_high_value:
            reasons.append(f"amount {inp.amount_inr} exceeds high-value threshold {config.high_value_threshold_inr}")
        if is_uncertain:
            reasons.append(
                f"classification confidence {inp.classification_confidence} below "
                f"threshold {config.low_confidence_threshold}"
            )
        if action not in ("human_escalation", "no_action", "stop_recovery"):
            checks.append(
                SafetyCheck(
                    "escalate_uncertain_or_high_value",
                    True,
                    f"{'; '.join(reasons)}; overriding '{action}' to 'human_escalation'.",
                )
            )
            action = "human_escalation"
            overridden = True
        else:
            checks.append(
                SafetyCheck(
                    "escalate_uncertain_or_high_value",
                    True,
                    f"{'; '.join(reasons)}; action already safe/escalated, flagged for human review.",
                )
            )
    else:
        checks.append(
            SafetyCheck("escalate_uncertain_or_high_value", False, "Not high-value and confidence acceptable.")
        )

    if action == "human_escalation":
        requires_review = True

    return PolicyResult(
        final_action=action,
        requires_human_review=requires_review,
        safety_checks=checks,
        was_overridden=overridden,
    )


def _parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def evaluate_policy_for_transaction(
    conn: sqlite3.Connection,
    transaction_id: str,
    proposed_action: str,
    config: PolicyConfig | None = None,
    model_or_llm_failed: bool = False,
) -> PolicyResult:
    """
    Gather context from the database for one transaction and run evaluate_policy().
    Looks only at action_events with source='payback_ai' (the baseline's history
    is irrelevant to PayBack AI's own safety state).
    """
    payment = conn.execute(
        "SELECT * FROM payments WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    if payment is None:
        raise ValueError(f"No payment found for transaction_id={transaction_id!r}")

    prediction = conn.execute(
        "SELECT * FROM model_predictions WHERE transaction_id = ? ORDER BY id DESC LIMIT 1",
        (transaction_id,),
    ).fetchone()
    # Missing prediction is treated as maximally uncertain (safe default: 0.0 confidence).
    classification_confidence = prediction["classification_confidence"] if prediction else 0.0
    recovery_probability = prediction["recovery_probability"] if prediction else None

    history = conn.execute(
        "SELECT action_type, status, executed_at FROM action_events "
        "WHERE transaction_id = ? AND source = 'payback_ai' ORDER BY executed_at",
        (transaction_id,),
    ).fetchall()

    retry_count_so_far = sum(1 for h in history if h["action_type"] == "delayed_retry")
    already_recovered = any(h["status"] == "recovered" for h in history)

    now = datetime.now(timezone.utc)
    action_times = [_parse_iso(h["executed_at"]) for h in history]
    minutes_since_last_action = (
        (now - max(action_times)).total_seconds() / 60 if action_times else None
    )

    cutoff_24h = now - timedelta(hours=24)
    messages_sent_last_24h = sum(
        1
        for h in history
        if h["action_type"] == "reminder" and _parse_iso(h["executed_at"]) >= cutoff_24h
    )

    duplicate_cutoff = now - timedelta(minutes=5)
    duplicate_action_pending = any(
        h["action_type"] == proposed_action and _parse_iso(h["executed_at"]) >= duplicate_cutoff
        for h in history
    )

    policy_input = PolicyInput(
        transaction_id=transaction_id,
        proposed_action=proposed_action,
        failure_category=payment["failure_category"],
        amount_inr=payment["amount_inr"],
        attempt_number=payment["attempt_number"],
        classification_confidence=classification_confidence,
        recovery_probability=recovery_probability,
        retry_count_so_far=retry_count_so_far,
        minutes_since_last_action=minutes_since_last_action,
        messages_sent_last_24h=messages_sent_last_24h,
        already_recovered=already_recovered,
        duplicate_action_pending=duplicate_action_pending,
        model_or_llm_failed=model_or_llm_failed,
    )
    return evaluate_policy(policy_input, config)
