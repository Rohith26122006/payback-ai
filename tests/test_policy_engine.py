from datetime import datetime, timedelta, timezone

import pytest

from app.database import get_connection
from core.classifier import classify_all_payments
from core.policy_engine import (
    PolicyConfig,
    PolicyInput,
    deterministic_fallback_action,
    evaluate_policy,
    evaluate_policy_for_transaction,
)
from data.generate_synthetic_data import GeneratorConfig, generate_dataset
from data.seed_database import seed_database

ALL_RULE_NAMES = {
    "deterministic_fallback",
    "stop_after_success",
    "prevent_duplicate_execution",
    "no_auto_retry_suspected_risk",
    "escalate_unknown_failure",
    "max_auto_retries",
    "retry_cooldown",
    "max_messages_per_24h",
    "escalate_uncertain_or_high_value",
}


def base_input(**overrides) -> PolicyInput:
    defaults = dict(
        transaction_id="TXN000001",
        proposed_action="delayed_retry",
        failure_category="temporary_bank_or_network",
        amount_inr=1000.0,
        attempt_number=1,
        classification_confidence=0.95,
        recovery_probability=0.6,
        retry_count_so_far=0,
        minutes_since_last_action=None,
        messages_sent_last_24h=0,
        already_recovered=False,
        duplicate_action_pending=False,
        model_or_llm_failed=False,
    )
    defaults.update(overrides)
    return PolicyInput(**defaults)


def test_clean_case_passes_through_unchanged() -> None:
    result = evaluate_policy(base_input())
    assert result.final_action == "delayed_retry"
    assert result.was_overridden is False
    rule_names = {c.rule for c in result.safety_checks}
    assert ALL_RULE_NAMES.issubset(rule_names)


def test_stop_after_success_forces_no_action() -> None:
    result = evaluate_policy(base_input(already_recovered=True, proposed_action="delayed_retry"))
    assert result.final_action == "no_action"
    assert result.was_overridden is True


def test_deterministic_fallback_overrides_proposed_action() -> None:
    result = evaluate_policy(
        base_input(
            model_or_llm_failed=True,
            proposed_action="stop_recovery",
            failure_category="temporary_bank_or_network",
            attempt_number=1,
        )
    )
    assert result.final_action == "delayed_retry"  # deterministic fallback logic
    assert result.was_overridden is True


def test_suspected_risk_never_auto_retried() -> None:
    result = evaluate_policy(
        base_input(failure_category="suspected_risk", proposed_action="delayed_retry")
    )
    assert result.final_action == "human_escalation"
    assert result.requires_human_review is True


def test_unknown_failure_escalated() -> None:
    result = evaluate_policy(
        base_input(failure_category="unknown", proposed_action="delayed_retry")
    )
    assert result.final_action == "human_escalation"


def test_max_auto_retries_enforced() -> None:
    result = evaluate_policy(
        base_input(proposed_action="delayed_retry", retry_count_so_far=2)
    )
    assert result.final_action == "stop_recovery"


def test_retry_cooldown_enforced() -> None:
    result = evaluate_policy(
        base_input(proposed_action="delayed_retry", minutes_since_last_action=5.0)
    )
    assert result.final_action == "no_action"


def test_retry_allowed_after_cooldown() -> None:
    result = evaluate_policy(
        base_input(proposed_action="delayed_retry", minutes_since_last_action=45.0)
    )
    assert result.final_action == "delayed_retry"


def test_message_limit_enforced() -> None:
    result = evaluate_policy(
        base_input(proposed_action="reminder", messages_sent_last_24h=1)
    )
    assert result.final_action == "stop_recovery"


def test_high_value_transaction_escalated() -> None:
    result = evaluate_policy(
        base_input(proposed_action="delayed_retry", amount_inr=100000.0)
    )
    assert result.final_action == "human_escalation"
    assert result.requires_human_review is True


def test_low_confidence_escalated() -> None:
    result = evaluate_policy(
        base_input(proposed_action="alternate_payment_link", classification_confidence=0.2)
    )
    assert result.final_action == "human_escalation"
    assert result.requires_human_review is True


def test_duplicate_action_blocked() -> None:
    result = evaluate_policy(
        base_input(proposed_action="delayed_retry", duplicate_action_pending=True)
    )
    assert result.final_action == "no_action"


def test_deterministic_fallback_for_risk_category_escalates() -> None:
    action = deterministic_fallback_action(
        "suspected_risk", attempt_number=1, config=PolicyConfig()
    )
    assert action == "human_escalation"


def test_deterministic_fallback_stops_after_max_attempts() -> None:
    action = deterministic_fallback_action(
        "temporary_bank_or_network", attempt_number=5, config=PolicyConfig()
    )
    assert action == "stop_recovery"


# --- DB-integration tests -------------------------------------------------


@pytest.fixture()
def seeded_db(tmp_path):
    df = generate_dataset(GeneratorConfig(n_records=300, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")
    seed_database(str(csv_path), db_path, reset=True)
    conn = get_connection(db_path)
    classify_all_payments(conn)
    conn.close()
    return db_path


def test_evaluate_policy_for_transaction_uses_real_row(seeded_db: str) -> None:
    conn = get_connection(seeded_db)
    try:
        row = conn.execute(
            "SELECT transaction_id FROM payments WHERE failure_category = 'suspected_risk' LIMIT 1"
        ).fetchone()
        assert row is not None, "seeded dataset should contain a suspected_risk case"
        result = evaluate_policy_for_transaction(conn, row["transaction_id"], "delayed_retry")
    finally:
        conn.close()
    assert result.final_action == "human_escalation"


def test_evaluate_policy_for_transaction_stops_after_recorded_success(seeded_db: str) -> None:
    conn = get_connection(seeded_db)
    try:
        row = conn.execute("SELECT transaction_id FROM payments LIMIT 1").fetchone()
        txn_id = row["transaction_id"]
        conn.execute(
            "INSERT INTO action_events (transaction_id, action_type, source, executed_at, status) "
            "VALUES (?, 'delayed_retry', 'payback_ai', ?, 'recovered')",
            (txn_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        result = evaluate_policy_for_transaction(conn, txn_id, "reminder")
    finally:
        conn.close()
    assert result.final_action == "no_action"


def test_evaluate_policy_for_transaction_missing_prediction_is_treated_as_uncertain(
    seeded_db: str,
) -> None:
    conn = get_connection(seeded_db)
    try:
        conn.execute("DELETE FROM model_predictions")
        conn.commit()
        row = conn.execute("SELECT transaction_id FROM payments LIMIT 1").fetchone()
        result = evaluate_policy_for_transaction(conn, row["transaction_id"], "delayed_retry")
    finally:
        conn.close()
    # No prediction -> confidence defaults to 0.0 -> below threshold -> escalated.
    assert result.final_action == "human_escalation"
    assert result.requires_human_review is True
