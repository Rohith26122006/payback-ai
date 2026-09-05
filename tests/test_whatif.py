import pytest

from app.database import get_connection
from core.classifier import classify_all_payments
from core.policy_engine import PolicyConfig
from core.recovery_model import train_recovery_model, update_recovery_probabilities
from core.simulator import SimulatorConfig
from core.whatif import run_what_if, run_what_if_from_db
from data.generate_synthetic_data import GeneratorConfig, generate_dataset
from data.seed_database import seed_database


@pytest.fixture()
def prepared_db(tmp_path):
    df = generate_dataset(GeneratorConfig(n_records=300, seed=42))
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


def test_run_what_if_returns_expected_shape(prepared_db: str) -> None:
    conn = get_connection(prepared_db)
    try:
        result = run_what_if(conn, PolicyConfig())
    finally:
        conn.close()
    expected_keys = {
        "policy_config",
        "total_transactions",
        "total_at_risk_revenue",
        "recovered_revenue",
        "recovery_rate",
        "action_distribution",
        "status_distribution",
        "human_review_count",
        "note",
    }
    assert expected_keys.issubset(result.keys())
    assert 0.0 <= result["recovery_rate"] <= 1.0
    assert result["recovered_revenue"] <= result["total_at_risk_revenue"]


def test_run_what_if_does_not_write_to_db(prepared_db: str) -> None:
    conn = get_connection(prepared_db)
    try:
        decisions_before = conn.execute("SELECT COUNT(*) AS c FROM recovery_decisions").fetchone()["c"]
        events_before = conn.execute("SELECT COUNT(*) AS c FROM action_events").fetchone()["c"]
        run_what_if(conn, PolicyConfig())
        decisions_after = conn.execute("SELECT COUNT(*) AS c FROM recovery_decisions").fetchone()["c"]
        events_after = conn.execute("SELECT COUNT(*) AS c FROM action_events").fetchone()["c"]
    finally:
        conn.close()
    assert decisions_before == decisions_after == 0
    assert events_before == events_after == 0


def test_stricter_policy_never_increases_delayed_retry_count(prepared_db: str) -> None:
    """A stricter max_auto_retries should never lead to MORE retries being approved."""
    conn = get_connection(prepared_db)
    try:
        lenient = run_what_if(conn, PolicyConfig(max_auto_retries=5))
        strict = run_what_if(conn, PolicyConfig(max_auto_retries=0))
    finally:
        conn.close()
    lenient_retries = lenient["action_distribution"].get("delayed_retry", 0)
    strict_retries = strict["action_distribution"].get("delayed_retry", 0)
    assert strict_retries <= lenient_retries


def test_lower_high_value_threshold_increases_escalations(prepared_db: str) -> None:
    conn = get_connection(prepared_db)
    try:
        normal = run_what_if(conn, PolicyConfig(high_value_threshold_inr=50000.0))
        aggressive = run_what_if(conn, PolicyConfig(high_value_threshold_inr=0.0))
    finally:
        conn.close()
    assert aggressive["human_review_count"] >= normal["human_review_count"]


def test_run_what_if_is_reproducible_with_same_seed(prepared_db: str) -> None:
    conn = get_connection(prepared_db)
    try:
        result1 = run_what_if(conn, PolicyConfig(), simulator_config=SimulatorConfig(seed=7))
        result2 = run_what_if(conn, PolicyConfig(), simulator_config=SimulatorConfig(seed=7))
    finally:
        conn.close()
    assert result1["recovery_rate"] == result2["recovery_rate"]
    assert result1["action_distribution"] == result2["action_distribution"]


def test_run_what_if_from_db_matches_direct_call(prepared_db: str) -> None:
    conn = get_connection(prepared_db)
    try:
        direct = run_what_if(conn, PolicyConfig(), simulator_config=SimulatorConfig(seed=42))
    finally:
        conn.close()
    via_wrapper = run_what_if_from_db(prepared_db, PolicyConfig(), simulator_config=SimulatorConfig(seed=42))
    assert direct["recovery_rate"] == via_wrapper["recovery_rate"]
    assert direct["action_distribution"] == via_wrapper["action_distribution"]
