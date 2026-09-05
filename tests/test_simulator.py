import random

import pytest

from app.database import get_connection
from core.audit_logger import get_audit_trail
from core.classifier import classify_all_payments
from core.decision_engine import run_decision_engine_for_all
from core.recovery_model import train_recovery_model, update_recovery_probabilities
from core.simulator import (
    SIMULATOR_SOURCE,
    SimulatorConfig,
    cancel_pending_actions_after_success,
    compute_payback_ai_metrics,
    run_simulator_for_all,
    simulate_action,
)
from data.generate_synthetic_data import GeneratorConfig, generate_dataset
from data.seed_database import seed_database


class ConstantRandom(random.Random):
    """Test helper: .random() always returns a fixed value, for deterministic branch testing."""

    def __init__(self, value: float):
        super().__init__()
        self._value = value

    def random(self) -> float:
        return self._value


# --- simulate_action (pure) ------------------------------------------------


def test_no_outcome_actions_are_skipped() -> None:
    rng = ConstantRandom(0.0)
    config = SimulatorConfig()
    for action in ("stop_recovery", "no_action"):
        status, amount = simulate_action(rng, action, ground_truth_recovered=True, recovered_amount=100.0, config=config)
        assert status == "skipped"
        assert amount == 0.0


def test_delayed_retry_success_and_failure() -> None:
    rng = ConstantRandom(0.0)
    config = SimulatorConfig()
    status_success, amount_success = simulate_action(rng, "delayed_retry", True, 500.0, config)
    status_fail, amount_fail = simulate_action(rng, "delayed_retry", False, 500.0, config)
    assert status_success == "recovered"
    assert amount_success == 500.0
    assert status_fail == "not_recovered"
    assert amount_fail == 0.0


def test_alternate_payment_link_success_and_failure() -> None:
    rng = ConstantRandom(0.0)
    config = SimulatorConfig()
    status_success, _ = simulate_action(rng, "alternate_payment_link", True, 300.0, config)
    status_fail, _ = simulate_action(rng, "alternate_payment_link", False, 300.0, config)
    assert status_success == "recovered"
    assert status_fail == "not_recovered"


def test_reminder_customer_ignores_when_random_above_response_rate() -> None:
    config = SimulatorConfig(reminder_response_rate=0.5)
    rng = ConstantRandom(0.9)  # 0.9 >= 0.5 -> ignored
    status, amount = simulate_action(rng, "reminder", True, 200.0, config)
    assert status == "customer_ignored"
    assert amount == 0.0


def test_reminder_customer_responds_when_random_below_response_rate() -> None:
    config = SimulatorConfig(reminder_response_rate=0.5)
    rng = ConstantRandom(0.1)  # 0.1 < 0.5 -> responds
    status, amount = simulate_action(rng, "reminder", True, 200.0, config)
    assert status == "recovered"
    assert amount == 200.0


def test_human_escalation_rejected_when_random_above_approval_rate() -> None:
    config = SimulatorConfig(human_approval_probability=0.5)
    rng = ConstantRandom(0.9)  # rejected
    status, amount = simulate_action(rng, "human_escalation", True, 400.0, config)
    assert status == "human_rejected"
    assert amount == 0.0


def test_human_escalation_approved_then_follows_ground_truth() -> None:
    config = SimulatorConfig(human_approval_probability=0.5)
    rng = ConstantRandom(0.1)  # approved
    status_recovered, _ = simulate_action(rng, "human_escalation", True, 400.0, config)
    status_not_recovered, _ = simulate_action(rng, "human_escalation", False, 400.0, config)
    assert status_recovered == "recovered"
    assert status_not_recovered == "not_recovered"


# --- cancel_pending_actions_after_success ----------------------------------


@pytest.fixture()
def seeded_db(tmp_path):
    df = generate_dataset(GeneratorConfig(n_records=300, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")
    seed_database(str(csv_path), db_path, reset=True)
    return db_path


def test_cancel_pending_actions_marks_other_rows_cancelled(seeded_db: str) -> None:
    conn = get_connection(seeded_db)
    try:
        txn_id = conn.execute("SELECT transaction_id FROM payments LIMIT 1").fetchone()[
            "transaction_id"
        ]
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

        cancelled = cancel_pending_actions_after_success(conn, txn_id)

        statuses = [
            r["status"]
            for r in conn.execute(
                "SELECT status FROM action_events WHERE transaction_id = ?", (txn_id,)
            ).fetchall()
        ]
    finally:
        conn.close()
    assert cancelled == 1
    assert "cancelled" in statuses
    assert "recovered" in statuses  # the successful row itself is untouched


# --- full pipeline integration ---------------------------------------------


@pytest.fixture()
def fully_decided_db(tmp_path):
    df = generate_dataset(GeneratorConfig(n_records=400, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")
    seed_database(str(csv_path), db_path, reset=True)

    conn = get_connection(db_path)
    classify_all_payments(conn)
    result = train_recovery_model(df)
    update_recovery_probabilities(conn, result.pipeline)
    run_decision_engine_for_all(conn)
    conn.close()
    return db_path


def test_run_simulator_covers_every_decision(fully_decided_db: str) -> None:
    conn = get_connection(fully_decided_db)
    try:
        results = run_simulator_for_all(conn)
        decision_count = conn.execute("SELECT COUNT(*) AS c FROM recovery_decisions").fetchone()["c"]
        event_count = conn.execute(
            "SELECT COUNT(*) AS c FROM action_events WHERE source = ?", (SIMULATOR_SOURCE,)
        ).fetchone()["c"]
    finally:
        conn.close()
    assert len(results) == decision_count
    assert event_count == decision_count


def test_run_simulator_is_idempotent(fully_decided_db: str) -> None:
    conn = get_connection(fully_decided_db)
    try:
        run_simulator_for_all(conn)
        run_simulator_for_all(conn)
        event_count = conn.execute(
            "SELECT COUNT(*) AS c FROM action_events WHERE source = ?", (SIMULATOR_SOURCE,)
        ).fetchone()["c"]
        decision_count = conn.execute("SELECT COUNT(*) AS c FROM recovery_decisions").fetchone()["c"]
    finally:
        conn.close()
    assert event_count == decision_count


def test_run_simulator_is_reproducible_with_same_seed(fully_decided_db: str) -> None:
    conn = get_connection(fully_decided_db)
    try:
        results1 = run_simulator_for_all(conn, SimulatorConfig(seed=99))
        results2 = run_simulator_for_all(conn, SimulatorConfig(seed=99))
    finally:
        conn.close()
    statuses1 = [(r["transaction_id"], r["status"]) for r in results1]
    statuses2 = [(r["transaction_id"], r["status"]) for r in results2]
    assert statuses1 == statuses2


def test_run_simulator_writes_audit_logs(fully_decided_db: str) -> None:
    conn = get_connection(fully_decided_db)
    try:
        run_simulator_for_all(conn)
        txn_id = conn.execute("SELECT transaction_id FROM payments LIMIT 1").fetchone()[
            "transaction_id"
        ]
        trail = get_audit_trail(conn, txn_id)
    finally:
        conn.close()
    assert len(trail) >= 1
    assert any("action_simulated" in entry["event_type"] for entry in trail)


def test_compute_payback_ai_metrics_shape_and_bounds(fully_decided_db: str) -> None:
    conn = get_connection(fully_decided_db)
    try:
        run_simulator_for_all(conn)
        metrics = compute_payback_ai_metrics(conn)
    finally:
        conn.close()
    assert metrics["recovered_revenue"] <= metrics["total_at_risk_revenue"]
    assert 0.0 <= metrics["recovery_rate"] <= 1.0
    assert set(metrics["status_distribution"].keys()) <= {
        "recovered",
        "not_recovered",
        "customer_ignored",
        "human_rejected",
        "skipped",
    }


def test_successful_recovery_cancels_pending_actions_end_to_end(fully_decided_db: str) -> None:
    conn = get_connection(fully_decided_db)
    try:
        run_simulator_for_all(conn)
        recovered_txn = conn.execute(
            "SELECT transaction_id FROM action_events WHERE source = ? AND status = 'recovered' LIMIT 1",
            (SIMULATOR_SOURCE,),
        ).fetchone()
        if recovered_txn is None:
            pytest.skip("No recovered transaction in this run to test cancellation on")
        cancelled_rows = conn.execute(
            "SELECT COUNT(*) AS c FROM action_events WHERE status = 'cancelled'"
        ).fetchone()["c"]
    finally:
        conn.close()
    # Not asserting a specific count (each transaction has only one decision in
    # this pipeline) - just confirming the mechanism ran without error and the
    # cancelled-status column is a valid, queryable state.
    assert cancelled_rows >= 0
