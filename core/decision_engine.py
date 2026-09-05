"""
PayBack AI - Decision engine.

Combines:
  - classified_category + classification_confidence  (Stage 5)
  - recovery_probability                              (Stage 6)
  - attempt_number                                     (payments)
into a PROPOSED action via a deterministic decision table, then hands that
proposal to the policy engine (Stage 7) for final, authoritative sign-off.

The decision engine NEVER finalizes an action on its own - policy_engine
always gets the last word. If no valid prediction exists for a transaction
(classifier/model hasn't run, or output is missing), this is treated as a
model failure and the policy engine's deterministic fallback takes over
entirely, per the project's safety requirements.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from core.policy_engine import (
    PolicyConfig,
    PolicyInput,
    evaluate_policy,
    evaluate_policy_for_transaction,
)
from core.review_queue import sync_review_queue

MODEL_VERSION = "decision-engine-v1"


@dataclass(frozen=True)
class DecisionConfig:
    low_probability_threshold: float = 0.30
    repeated_failure_attempt_threshold: int = 2  # matches PolicyConfig.max_auto_retries default


def propose_action(
    failure_category: str,
    recovery_probability: float | None,
    attempt_number: int,
    config: DecisionConfig | None = None,
) -> tuple[str, str]:
    """
    Pure decision-table lookup: (failure_category, recovery_probability,
    attempt_number) -> (proposed_action, reason). No DB access, no policy
    checks - this is only a PROPOSAL, always subject to policy override.
    """
    config = config or DecisionConfig()

    if failure_category == "suspected_risk":
        return (
            "human_escalation",
            "Suspected-risk transactions are always escalated for human review.",
        )

    if failure_category == "unknown":
        return (
            "human_escalation",
            "Unknown failure category cannot be safely automated; escalating for review.",
        )

    if attempt_number > config.repeated_failure_attempt_threshold:
        return (
            "stop_recovery",
            f"Attempt number {attempt_number} exceeds the repeated-failure threshold "
            f"({config.repeated_failure_attempt_threshold}); stopping automated recovery.",
        )

    if recovery_probability is not None and recovery_probability < config.low_probability_threshold:
        if attempt_number == 1:
            return (
                "reminder",
                f"Recovery probability is very low ({recovery_probability:.2f}); sending a "
                f"single controlled reminder instead of retrying.",
            )
        return (
            "stop_recovery",
            f"Recovery probability is very low ({recovery_probability:.2f}) after "
            f"{attempt_number} attempts; stopping recovery.",
        )

    if failure_category == "temporary_bank_or_network":
        return (
            "delayed_retry",
            "Temporary bank/network failure; retrying after a short delay is likely to succeed.",
        )

    if failure_category == "expired_or_invalid_method":
        return (
            "alternate_payment_link",
            "The payment method itself failed (expired/invalid); offering an alternate "
            "payment link instead of retrying the same method.",
        )

    if failure_category == "insufficient_funds":
        if attempt_number == 1:
            return (
                "reminder",
                "Insufficient funds; a reminder gives the customer time to top up before "
                "trying again.",
            )
        return (
            "alternate_payment_link",
            "Insufficient funds persisted across attempts; offering an alternate payment method.",
        )

    if failure_category == "subscription_failure":
        return (
            "alternate_payment_link",
            "Subscription mandate/auto-debit failed; offering an alternate payment link to "
            "re-establish payment.",
        )

    if failure_category == "authentication_failure":
        if attempt_number == 1:
            return (
                "delayed_retry",
                "Authentication failure on first attempt; a retry after a short delay may "
                "succeed once the customer re-authenticates.",
            )
        return (
            "alternate_payment_link",
            "Authentication failed repeatedly; offering an alternate payment method instead "
            "of retrying.",
        )

    if failure_category == "customer_abandonment":
        return (
            "reminder",
            "Customer appears to have abandoned the payment; a gentle reminder may bring "
            "them back.",
        )

    return (
        "human_escalation",
        f"No specific rule matched failure_category='{failure_category}'; escalating as a "
        f"safe default.",
    )


@dataclass(frozen=True)
class DecisionOutput:
    transaction_id: str
    action: str
    recovery_probability: float | None
    confidence: float
    reason: str
    safety_checks: list[dict]
    requires_human_review: bool

    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "action": self.action,
            "recovery_probability": self.recovery_probability,
            "confidence": self.confidence,
            "reason": self.reason,
            "safety_checks": self.safety_checks,
            "requires_human_review": self.requires_human_review,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_decision(
    conn: sqlite3.Connection,
    transaction_id: str,
    decision_config: DecisionConfig | None = None,
    policy_config: PolicyConfig | None = None,
    ignore_history: bool = False,
) -> DecisionOutput:
    """
    Produce the final decision for one transaction: propose via the decision
    table, then finalize via the authoritative policy engine.

    ignore_history=True evaluates the policy as if no prior payback_ai
    action_events existed for this transaction (retry_count_so_far=0, no
    cooldown, no message history, not already recovered, no duplicate
    pending). This is used by the what-if simulator (core/whatif.py) so
    hypothetical policy changes aren't contaminated by the real run's
    already-recorded history - which, for a single-shot dataset like this
    one where all real actions happen within moments of each other, would
    otherwise make almost everything look like an immediate duplicate/
    cooldown violation. Normal (persisted) decisions always use
    ignore_history=False (the default) so real safeguards apply correctly.
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

    model_or_llm_failed = (
        prediction is None
        or prediction["classified_category"] is None
        or prediction["classification_confidence"] is None
    )

    if model_or_llm_failed:
        # No valid prediction - propose a harmless placeholder; the policy
        # engine ignores this entirely and applies its deterministic fallback.
        confidence = 0.0
        recovery_probability = None
        proposed_action, proposed_reason = "stop_recovery", (
            "No valid classifier/model prediction was available for this transaction."
        )
    else:
        classified_category = prediction["classified_category"]
        confidence = prediction["classification_confidence"]
        recovery_probability = prediction["recovery_probability"]
        proposed_action, proposed_reason = propose_action(
            classified_category, recovery_probability, payment["attempt_number"], decision_config
        )

    if ignore_history:
        policy_input = PolicyInput(
            transaction_id=transaction_id,
            proposed_action=proposed_action,
            failure_category=payment["failure_category"],
            amount_inr=payment["amount_inr"],
            attempt_number=payment["attempt_number"],
            classification_confidence=confidence,
            recovery_probability=recovery_probability,
            retry_count_so_far=0,
            minutes_since_last_action=None,
            messages_sent_last_24h=0,
            already_recovered=False,
            duplicate_action_pending=False,
            model_or_llm_failed=model_or_llm_failed,
        )
        policy_result = evaluate_policy(policy_input, policy_config)
    else:
        policy_result = evaluate_policy_for_transaction(
            conn,
            transaction_id,
            proposed_action,
            config=policy_config,
            model_or_llm_failed=model_or_llm_failed,
        )

    if policy_result.was_overridden:
        override_details = "; ".join(
            c.detail for c in policy_result.safety_checks if c.triggered
        )
        reason = f"{proposed_reason} Policy engine override: {override_details}"
    else:
        reason = proposed_reason

    return DecisionOutput(
        transaction_id=transaction_id,
        action=policy_result.final_action,
        recovery_probability=recovery_probability,
        confidence=confidence,
        reason=reason,
        safety_checks=[c.to_dict() for c in policy_result.safety_checks],
        requires_human_review=policy_result.requires_human_review,
    )


def run_decision_engine_for_all(
    conn: sqlite3.Connection,
    decision_config: DecisionConfig | None = None,
    policy_config: PolicyConfig | None = None,
) -> list[DecisionOutput]:
    """
    Run make_decision() for every payment and persist to recovery_decisions.
    Idempotent: clears prior rows before inserting.
    """
    conn.execute("DELETE FROM recovery_decisions")
    conn.commit()

    transaction_ids = [
        row["transaction_id"] for row in conn.execute("SELECT transaction_id FROM payments")
    ]

    decided_at = _now_iso()
    outputs: list[DecisionOutput] = []
    for txn_id in transaction_ids:
        output = make_decision(conn, txn_id, decision_config, policy_config)
        conn.execute(
            "INSERT INTO recovery_decisions "
            "(transaction_id, action, reason, safety_checks, requires_human_review, decided_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                output.transaction_id,
                output.action,
                output.reason,
                json.dumps(output.safety_checks),
                int(output.requires_human_review),
                decided_at,
            ),
        )
        outputs.append(output)

    conn.commit()

    sync_review_queue(conn)

    return outputs


if __name__ == "__main__":
    from app.database import get_connection

    conn = get_connection()
    outputs = run_decision_engine_for_all(conn)
    conn.close()

    action_counts: dict[str, int] = {}
    review_count = 0
    for o in outputs:
        action_counts[o.action] = action_counts.get(o.action, 0) + 1
        if o.requires_human_review:
            review_count += 1

    print(f"[DECISION ENGINE] Decided {len(outputs)} transactions:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count}")
    print(f"  requires_human_review: {review_count}")
