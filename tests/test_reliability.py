"""
PayBack AI - Reliability test suite.

Each test below is explicitly mapped to one item from the project spec's
required reliability scenarios:
  1. Duplicate events
  2. Idempotent action execution
  3. Payment success cancelling a scheduled retry
  4. Maximum retry count
  5. Cooldown period
  6. Message limit
  7. Suspected-risk transaction
  8. Unknown failure
  9. High-value transaction
  10. Invalid model output
  11. LLM unavailable
  12. Database validation
  13. API errors

Some scenarios are already covered in depth by their owning module's test
file (e.g. tests/test_policy_engine.py) - this file exists as a single
place a reviewer can check every required scenario is accounted for, not
to duplicate all of that depth.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.database import get_connection, init_db
from app.main import app
from core.classifier import classify_all_payments
from core.decision_engine import make_decision, run_decision_engine_for_all
from core.policy_engine import VALID_ACTIONS, PolicyConfig, PolicyInput, evaluate_policy
from core.recovery_model import train_recovery_model, update_recovery_probabilities
from core.simulator import run_simulator_for_all
from data.generate_synthetic_data import GeneratorConfig, generate_dataset
from data.seed_database import seed_database
from llm.message_generator import DecisionOutput, generate_message
from llm.mock_provider import MockLLMProvider


def base_policy_input(**overrides) -> PolicyInput:
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


@pytest.fixture()
def fully_run_db(tmp_path):
    df = generate_dataset(GeneratorConfig(n_records=300, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")
    seed_database(str(csv_path), db_path, reset=True)

    conn = get_connection(db_path)
    classify_all_payments(conn)
    result = train_recovery_model(df)
    update_recovery_probabilities(conn, result.pipeline)
    run_decision_engine_for_all(conn)
    run_simulator_for_all(conn)
    conn.close()
    return db_path


# 1. Duplicate events -------------------------------------------------------


def test_1_duplicate_events_do_not_duplicate_rows(tmp_path) -> None:
    df = generate_dataset(GeneratorConfig(n_records=200, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")

    seed_database(str(csv_path), db_path, reset=True)
    seed_database(str(csv_path), db_path, reset=False)  # re-seed the SAME events again

    conn = get_connection(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) AS c FROM payments").fetchone()["c"]
    finally:
        conn.close()
    assert count == 200  # not 400 - duplicate events were ignored, not duplicated


# 2. Idempotent action execution --------------------------------------------


def test_2_idempotent_action_execution(fully_run_db: str) -> None:
    conn = get_connection(fully_run_db)
    try:
        run_simulator_for_all(conn)  # run again
        events = conn.execute(
            "SELECT COUNT(*) AS c FROM action_events WHERE source = 'payback_ai'"
        ).fetchone()["c"]
        payments = conn.execute("SELECT COUNT(*) AS c FROM payments").fetchone()["c"]
    finally:
        conn.close()
    assert events == payments  # exactly one event per payment, not accumulating


# 3. Payment success cancelling a scheduled retry ---------------------------


def test_3_payment_success_cancels_scheduled_retry(fully_run_db: str) -> None:
    conn = get_connection(fully_run_db)
    try:
        txn_id = conn.execute("SELECT transaction_id FROM payments LIMIT 1").fetchone()[
            "transaction_id"
        ]
        conn.execute(
            "DELETE FROM action_events WHERE transaction_id = ? AND source = 'payback_ai'",
            (txn_id,),
        )
        conn.execute(
            "INSERT INTO action_events (transaction_id, action_type, source, executed_at, status) "
            "VALUES (?, 'reminder', 'payback_ai', '2026-01-01T00:00:00+00:00', 'not_recovered')",
            (txn_id,),
        )
        conn.execute(
            "INSERT INTO action_events (transaction_id, action_type, source, executed_at, status) "
            "VALUES (?, 'delayed_retry', 'payback_ai', '2026-01-01T00:05:00+00:00', 'recovered')",
            (txn_id,),
        )
        conn.commit()
        from core.simulator import cancel_pending_actions_after_success

        cancel_pending_actions_after_success(conn, txn_id)
        statuses = {
            r["status"]
            for r in conn.execute(
                "SELECT status FROM action_events WHERE transaction_id = ? AND source = 'payback_ai'",
                (txn_id,),
            ).fetchall()
        }
    finally:
        conn.close()
    assert "cancelled" in statuses


# 4. Maximum retry count -----------------------------------------------------


def test_4_maximum_retry_count_enforced() -> None:
    result = evaluate_policy(
        base_policy_input(proposed_action="delayed_retry", retry_count_so_far=2),
        PolicyConfig(max_auto_retries=2),
    )
    assert result.final_action == "stop_recovery"


# 5. Cooldown period ----------------------------------------------------------


def test_5_cooldown_period_enforced() -> None:
    result = evaluate_policy(
        base_policy_input(proposed_action="delayed_retry", minutes_since_last_action=5.0),
        PolicyConfig(retry_cooldown_minutes=30),
    )
    assert result.final_action == "no_action"


# 6. Message limit -------------------------------------------------------------


def test_6_message_limit_enforced() -> None:
    result = evaluate_policy(
        base_policy_input(proposed_action="reminder", messages_sent_last_24h=1),
        PolicyConfig(max_messages_per_24h=1),
    )
    assert result.final_action == "stop_recovery"


# 7. Suspected-risk transaction ------------------------------------------------


def test_7_suspected_risk_transaction_never_auto_retried() -> None:
    result = evaluate_policy(
        base_policy_input(failure_category="suspected_risk", proposed_action="delayed_retry")
    )
    assert result.final_action == "human_escalation"
    assert result.requires_human_review is True


# 8. Unknown failure --------------------------------------------------------


def test_8_unknown_failure_escalated() -> None:
    result = evaluate_policy(
        base_policy_input(failure_category="unknown", proposed_action="delayed_retry")
    )
    assert result.final_action == "human_escalation"


# 9. High-value transaction --------------------------------------------------


def test_9_high_value_transaction_escalated() -> None:
    result = evaluate_policy(
        base_policy_input(proposed_action="delayed_retry", amount_inr=200000.0),
        PolicyConfig(high_value_threshold_inr=50000.0),
    )
    assert result.final_action == "human_escalation"
    assert result.requires_human_review is True


# 10. Invalid model output ---------------------------------------------------


def test_10_missing_prediction_triggers_safe_fallback(fully_run_db: str) -> None:
    conn = get_connection(fully_run_db)
    try:
        txn_id = conn.execute("SELECT transaction_id FROM payments LIMIT 1").fetchone()[
            "transaction_id"
        ]
        conn.execute("DELETE FROM model_predictions WHERE transaction_id = ?", (txn_id,))
        conn.commit()
        output = make_decision(conn, txn_id)  # must not raise
    finally:
        conn.close()
    assert output.action in VALID_ACTIONS
    assert output.confidence == 0.0
    assert output.recovery_probability is None


def test_10_corrupted_out_of_range_probability_does_not_crash(fully_run_db: str) -> None:
    """A hand-corrupted (out-of-[0,1]-range) recovery_probability must not crash the decision engine."""
    conn = get_connection(fully_run_db)
    try:
        txn_id = conn.execute("SELECT transaction_id FROM payments LIMIT 1").fetchone()[
            "transaction_id"
        ]
        conn.execute(
            "UPDATE model_predictions SET recovery_probability = 1.5 WHERE transaction_id = ?",
            (txn_id,),
        )
        conn.commit()
        output = make_decision(conn, txn_id)  # must not raise
    finally:
        conn.close()
    assert output.action in VALID_ACTIONS


# 11. LLM unavailable ---------------------------------------------------------


def test_11_llm_unavailable_falls_back_to_deterministic_message() -> None:
    decision = DecisionOutput(
        transaction_id="TXN000001",
        action="reminder",
        recovery_probability=0.4,
        confidence=0.8,
        reason="test",
        safety_checks=[],
        requires_human_review=False,
    )
    result = generate_message(decision, provider=MockLLMProvider(always_fail=True))
    assert result.used_fallback is True
    assert result.output.customer_message != ""


# 12. Database validation -----------------------------------------------------


def test_12_invalid_action_events_source_rejected() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/empty.db"
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            conn.execute(
                "INSERT INTO merchants (merchant_id, merchant_category) VALUES ('M001', 'e_commerce')"
            )
            conn.execute(
                "INSERT INTO payments (transaction_id, merchant_id, customer_id_hash, amount_inr, "
                "payment_method, failure_code, failure_category, attempt_number, "
                "customer_success_rate, previous_failed_attempts, time_since_failure_minutes, "
                "subscription_flag, merchant_category, device_change_flag, recovery_outcome, "
                "recovered_amount, event_timestamp) VALUES "
                "('TXN000001','M001','hash',100.0,'upi','BANK_TIMEOUT','temporary_bank_or_network',"
                "1,0.5,0,60,0,'e_commerce',0,'not_recovered',0.0,'2026-01-01T00:00:00')"
            )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO action_events (transaction_id, action_type, source, executed_at, status) "
                    "VALUES ('TXN000001', 'delayed_retry', 'not_a_real_source', '2026-01-01', 'recovered')"
                )
        finally:
            conn.close()


def test_12_invalid_boolean_flag_rejected() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/empty.db"
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            conn.execute(
                "INSERT INTO merchants (merchant_id, merchant_category) VALUES ('M001', 'e_commerce')"
            )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO payments (transaction_id, merchant_id, customer_id_hash, amount_inr, "
                    "payment_method, failure_code, failure_category, attempt_number, "
                    "customer_success_rate, previous_failed_attempts, time_since_failure_minutes, "
                    "subscription_flag, merchant_category, device_change_flag, recovery_outcome, "
                    "recovered_amount, event_timestamp) VALUES "
                    "('TXN000001','M001','hash',100.0,'upi','BANK_TIMEOUT','temporary_bank_or_network',"
                    "1,0.5,0,60,2,'e_commerce',0,'not_recovered',0.0,'2026-01-01T00:00:00')"
                )  # subscription_flag=2 violates CHECK (0 or 1)
        finally:
            conn.close()


def test_12_missing_required_field_rejected() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/empty.db"
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                # merchant_category omitted (NOT NULL) - must be rejected
                conn.execute("INSERT INTO merchants (merchant_id) VALUES ('M001')")
        finally:
            conn.close()


# 13. API errors ---------------------------------------------------------------


@pytest.fixture()
def api_client(fully_run_db: str, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{fully_run_db}")
    return TestClient(app)


def test_13_api_404_for_missing_transaction(api_client: TestClient) -> None:
    assert api_client.get("/transactions/DOES_NOT_EXIST").status_code == 404
    assert api_client.get("/decisions/DOES_NOT_EXIST").status_code == 404
    assert api_client.post("/decisions/DOES_NOT_EXIST/run").status_code == 404


def test_13_api_422_for_invalid_query_params(api_client: TestClient) -> None:
    assert api_client.get("/transactions", params={"limit": 0}).status_code == 422
    assert api_client.get("/transactions", params={"limit": 100000}).status_code == 422
    assert api_client.get("/transactions", params={"offset": -1}).status_code == 422
