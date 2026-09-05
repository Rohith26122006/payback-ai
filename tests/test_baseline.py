import pytest

from app.database import get_connection, reset_db
from core.baseline import (
    BaselineConfig,
    compute_baseline_metrics,
    decide_baseline_action,
    run_baseline,
)
from data.generate_synthetic_data import GeneratorConfig, generate_dataset
from data.seed_database import seed_database


@pytest.fixture()
def seeded_db(tmp_path):
    df = generate_dataset(GeneratorConfig(n_records=500, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")
    seed_database(str(csv_path), db_path, reset=True)
    return db_path


def test_decide_action_first_attempt_is_retry() -> None:
    config = BaselineConfig()
    assert decide_baseline_action(1, config) == "delayed_retry"


def test_decide_action_second_attempt_is_reminder() -> None:
    config = BaselineConfig()
    assert decide_baseline_action(2, config) == "reminder"


def test_decide_action_beyond_max_is_stop() -> None:
    config = BaselineConfig()
    assert decide_baseline_action(3, config) == "stop_recovery"
    assert decide_baseline_action(10, config) == "stop_recovery"


def test_run_baseline_covers_every_payment(seeded_db: str) -> None:
    conn = get_connection(seeded_db)
    try:
        results = run_baseline(conn)
        total_payments = conn.execute("SELECT COUNT(*) AS c FROM payments").fetchone()["c"]
        event_count = conn.execute(
            "SELECT COUNT(*) AS c FROM action_events WHERE source = 'baseline'"
        ).fetchone()["c"]
    finally:
        conn.close()
    assert len(results) == total_payments
    assert event_count == total_payments


def test_run_baseline_is_idempotent(seeded_db: str) -> None:
    conn = get_connection(seeded_db)
    try:
        run_baseline(conn)
        run_baseline(conn)  # re-run should not duplicate
        event_count = conn.execute(
            "SELECT COUNT(*) AS c FROM action_events WHERE source = 'baseline'"
        ).fetchone()["c"]
        total_payments = conn.execute("SELECT COUNT(*) AS c FROM payments").fetchone()["c"]
    finally:
        conn.close()
    assert event_count == total_payments


def test_baseline_never_touches_ai_source(seeded_db: str) -> None:
    conn = get_connection(seeded_db)
    try:
        run_baseline(conn)
        ai_events = conn.execute(
            "SELECT COUNT(*) AS c FROM action_events WHERE source = 'payback_ai'"
        ).fetchone()["c"]
    finally:
        conn.close()
    assert ai_events == 0


def test_compute_baseline_metrics_shape_and_bounds(seeded_db: str) -> None:
    conn = get_connection(seeded_db)
    try:
        run_baseline(conn)
        metrics = compute_baseline_metrics(conn)
    finally:
        conn.close()

    assert metrics["total_transactions"] == 500
    assert 0.0 <= metrics["recovery_rate"] <= 1.0
    assert metrics["recovered_revenue"] <= metrics["total_at_risk_revenue"]
    assert set(metrics["action_distribution"].keys()) <= {
        "delayed_retry",
        "reminder",
        "stop_recovery",
    }
    assert metrics["unnecessary_retries"] >= 0
    assert metrics["stopped_count"] >= 0


def test_stopped_transactions_never_counted_as_recovered(seeded_db: str) -> None:
    conn = get_connection(seeded_db)
    try:
        run_baseline(conn)
        rows = conn.execute(
            "SELECT status FROM action_events WHERE source = 'baseline' AND action_type = 'stop_recovery'"
        ).fetchall()
    finally:
        conn.close()
    assert all(r["status"] == "skipped" for r in rows)
