import json

import pytest

from app.database import get_connection
from core.classifier import classify_all_payments
from core.decision_engine import (
    DecisionConfig,
    make_decision,
    propose_action,
    run_decision_engine_for_all,
)
from core.policy_engine import VALID_ACTIONS
from core.recovery_model import train_recovery_model, update_recovery_probabilities
from data.generate_synthetic_data import GeneratorConfig, generate_dataset
from data.seed_database import seed_database


# --- propose_action (pure decision table) ---------------------------------


def test_suspected_risk_always_escalated() -> None:
    action, _ = propose_action("suspected_risk", 0.9, 1)
    assert action == "human_escalation"


def test_unknown_always_escalated() -> None:
    action, _ = propose_action("unknown", 0.9, 1)
    assert action == "human_escalation"


def test_repeated_failures_stop_regardless_of_category() -> None:
    action, _ = propose_action("temporary_bank_or_network", 0.9, attempt_number=3)
    assert action == "stop_recovery"


def test_very_low_probability_first_attempt_sends_reminder() -> None:
    action, _ = propose_action("temporary_bank_or_network", 0.1, attempt_number=1)
    assert action == "reminder"


def test_very_low_probability_repeat_attempt_stops() -> None:
    action, _ = propose_action("temporary_bank_or_network", 0.1, attempt_number=2)
    assert action == "stop_recovery"


def test_temporary_failure_first_attempt_is_retry() -> None:
    action, _ = propose_action("temporary_bank_or_network", 0.7, attempt_number=1)
    assert action == "delayed_retry"


def test_expired_method_gets_alternate_link() -> None:
    action, _ = propose_action("expired_or_invalid_method", 0.6, attempt_number=1)
    assert action == "alternate_payment_link"


def test_insufficient_funds_first_attempt_reminder_then_link() -> None:
    action1, _ = propose_action("insufficient_funds", 0.6, attempt_number=1)
    action2, _ = propose_action("insufficient_funds", 0.6, attempt_number=2)
    assert action1 == "reminder"
    assert action2 == "alternate_payment_link"


def test_subscription_failure_gets_alternate_link() -> None:
    action, _ = propose_action("subscription_failure", 0.6, attempt_number=1)
    assert action == "alternate_payment_link"


def test_authentication_failure_first_then_repeat() -> None:
    action1, _ = propose_action("authentication_failure", 0.6, attempt_number=1)
    action2, _ = propose_action("authentication_failure", 0.6, attempt_number=2)
    assert action1 == "delayed_retry"
    assert action2 == "alternate_payment_link"


def test_customer_abandonment_gets_reminder() -> None:
    action, _ = propose_action("customer_abandonment", 0.6, attempt_number=1)
    assert action == "reminder"


def test_all_proposed_actions_are_valid() -> None:
    categories = [
        "temporary_bank_or_network",
        "insufficient_funds",
        "expired_or_invalid_method",
        "authentication_failure",
        "suspected_risk",
        "customer_abandonment",
        "subscription_failure",
        "unknown",
    ]
    for category in categories:
        for attempt in (1, 2, 3):
            for prob in (0.1, 0.5, 0.9):
                action, reason = propose_action(category, prob, attempt)
                assert action in VALID_ACTIONS
                assert isinstance(reason, str) and len(reason) > 0


# --- full pipeline integration --------------------------------------------


@pytest.fixture()
def fully_prepared_db(tmp_path):
    df = generate_dataset(GeneratorConfig(n_records=400, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")
    seed_database(str(csv_path), db_path, reset=True)

    conn = get_connection(db_path)
    classify_all_payments(conn)
    result = train_recovery_model(df)
    update_recovery_probabilities(conn, result.pipeline)
    conn.close()
    return db_path


def test_make_decision_returns_expected_json_shape(fully_prepared_db: str) -> None:
    conn = get_connection(fully_prepared_db)
    try:
        row = conn.execute("SELECT transaction_id FROM payments LIMIT 1").fetchone()
        output = make_decision(conn, row["transaction_id"])
    finally:
        conn.close()

    d = output.to_dict()
    expected_keys = {
        "transaction_id",
        "action",
        "recovery_probability",
        "confidence",
        "reason",
        "safety_checks",
        "requires_human_review",
    }
    assert expected_keys == set(d.keys())
    assert d["action"] in VALID_ACTIONS
    assert isinstance(d["safety_checks"], list)
    # Must be JSON-serializable (as it will be stored in the DB).
    json.dumps(d["safety_checks"])


def test_make_decision_suspected_risk_is_escalated(fully_prepared_db: str) -> None:
    conn = get_connection(fully_prepared_db)
    try:
        row = conn.execute(
            "SELECT transaction_id FROM payments WHERE failure_category = 'suspected_risk' LIMIT 1"
        ).fetchone()
        assert row is not None
        output = make_decision(conn, row["transaction_id"])
    finally:
        conn.close()
    assert output.action == "human_escalation"
    assert output.requires_human_review is True


def test_make_decision_falls_back_when_prediction_missing(fully_prepared_db: str) -> None:
    conn = get_connection(fully_prepared_db)
    try:
        row = conn.execute("SELECT transaction_id FROM payments LIMIT 1").fetchone()
        txn_id = row["transaction_id"]
        conn.execute("DELETE FROM model_predictions WHERE transaction_id = ?", (txn_id,))
        conn.commit()
        output = make_decision(conn, txn_id)
    finally:
        conn.close()
    # No prediction -> model_or_llm_failed -> deterministic fallback via policy engine.
    assert output.action in VALID_ACTIONS
    assert output.confidence == 0.0
    assert output.recovery_probability is None


def test_run_decision_engine_for_all_covers_every_payment(fully_prepared_db: str) -> None:
    conn = get_connection(fully_prepared_db)
    try:
        outputs = run_decision_engine_for_all(conn)
        total_payments = conn.execute("SELECT COUNT(*) AS c FROM payments").fetchone()["c"]
        decision_rows = conn.execute("SELECT COUNT(*) AS c FROM recovery_decisions").fetchone()["c"]
    finally:
        conn.close()
    assert len(outputs) == total_payments
    assert decision_rows == total_payments


def test_run_decision_engine_for_all_is_idempotent(fully_prepared_db: str) -> None:
    conn = get_connection(fully_prepared_db)
    try:
        run_decision_engine_for_all(conn)
        run_decision_engine_for_all(conn)
        total_payments = conn.execute("SELECT COUNT(*) AS c FROM payments").fetchone()["c"]
        decision_rows = conn.execute("SELECT COUNT(*) AS c FROM recovery_decisions").fetchone()["c"]
    finally:
        conn.close()
    assert decision_rows == total_payments


def test_stored_safety_checks_are_valid_json(fully_prepared_db: str) -> None:
    conn = get_connection(fully_prepared_db)
    try:
        run_decision_engine_for_all(conn)
        rows = conn.execute("SELECT safety_checks FROM recovery_decisions LIMIT 5").fetchall()
    finally:
        conn.close()
    for row in rows:
        parsed = json.loads(row["safety_checks"])
        assert isinstance(parsed, list)
        assert all("rule" in item and "triggered" in item for item in parsed)


def test_high_value_transaction_requires_human_review(fully_prepared_db: str) -> None:
    conn = get_connection(fully_prepared_db)
    try:
        row = conn.execute(
            "SELECT transaction_id FROM payments ORDER BY amount_inr DESC LIMIT 1"
        ).fetchone()
        output = make_decision(conn, row["transaction_id"])
    finally:
        conn.close()
    assert output.requires_human_review is True
