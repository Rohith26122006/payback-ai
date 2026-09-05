import json

import pytest

from app.database import get_connection
from core.baseline import run_baseline
from core.classifier import classify_all_payments
from core.decision_engine import run_decision_engine_for_all
from core.recovery_model import train_recovery_model, update_recovery_probabilities
from core.simulator import run_simulator_for_all
from data.generate_synthetic_data import GeneratorConfig, generate_dataset
from data.seed_database import seed_database
from evaluation.costs import CostAssumptions
from evaluation.evaluate import run_evaluation, write_json_report, write_markdown_report
from evaluation.metrics import (
    compute_full_baseline_metrics,
    compute_full_payback_ai_metrics,
    measure_decision_latency,
)

EXTENDED_KEYS = {
    "average_recovery_time_minutes",
    "unnecessary_retry_rate",
    "messages_per_recovered_payment",
    "cost_per_recovery_inr",
    "human_escalation_rate",
    "false_positive_decision_rate",
    "policy_violation_count",
    "cost_assumptions",
}


@pytest.fixture()
def fully_run_db(tmp_path):
    df = generate_dataset(GeneratorConfig(n_records=400, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")
    seed_database(str(csv_path), db_path, reset=True)

    conn = get_connection(db_path)
    run_baseline(conn)
    classify_all_payments(conn)
    result = train_recovery_model(df)
    update_recovery_probabilities(conn, result.pipeline)
    run_decision_engine_for_all(conn)
    run_simulator_for_all(conn)
    conn.close()
    return db_path


def test_full_baseline_metrics_has_all_extended_keys(fully_run_db: str) -> None:
    conn = get_connection(fully_run_db)
    try:
        metrics = compute_full_baseline_metrics(conn)
    finally:
        conn.close()
    assert EXTENDED_KEYS.issubset(metrics.keys())


def test_full_payback_ai_metrics_has_all_extended_keys(fully_run_db: str) -> None:
    conn = get_connection(fully_run_db)
    try:
        metrics = compute_full_payback_ai_metrics(conn)
    finally:
        conn.close()
    assert EXTENDED_KEYS.issubset(metrics.keys())


def test_rates_are_within_valid_range(fully_run_db: str) -> None:
    conn = get_connection(fully_run_db)
    try:
        baseline = compute_full_baseline_metrics(conn)
        payback_ai = compute_full_payback_ai_metrics(conn)
    finally:
        conn.close()
    for metrics in (baseline, payback_ai):
        assert 0.0 <= metrics["unnecessary_retry_rate"] <= 1.0
        assert 0.0 <= metrics["human_escalation_rate"] <= 1.0
        assert 0.0 <= metrics["false_positive_decision_rate"] <= 1.0
        assert metrics["cost_per_recovery_inr"] >= 0.0
        assert metrics["policy_violation_count"] >= 0


def test_baseline_never_has_policy_violations(fully_run_db: str) -> None:
    conn = get_connection(fully_run_db)
    try:
        baseline = compute_full_baseline_metrics(conn)
    finally:
        conn.close()
    assert baseline["policy_violation_count"] == 0


def test_baseline_never_has_human_escalations(fully_run_db: str) -> None:
    """The baseline action set has no human_escalation - confirm the rate is exactly 0."""
    conn = get_connection(fully_run_db)
    try:
        baseline = compute_full_baseline_metrics(conn)
    finally:
        conn.close()
    assert baseline["human_escalation_rate"] == 0.0


def test_cost_assumptions_change_cost_per_recovery(fully_run_db: str) -> None:
    conn = get_connection(fully_run_db)
    try:
        cheap = compute_full_payback_ai_metrics(conn, CostAssumptions(retry_cost_inr=0.01, message_cost_inr=0.01, human_review_cost_inr=0.01))
        expensive = compute_full_payback_ai_metrics(conn, CostAssumptions(retry_cost_inr=100.0, message_cost_inr=100.0, human_review_cost_inr=100.0))
    finally:
        conn.close()
    assert expensive["cost_per_recovery_inr"] > cheap["cost_per_recovery_inr"]


def test_measure_decision_latency_both_sources(fully_run_db: str) -> None:
    conn = get_connection(fully_run_db)
    try:
        baseline_latency = measure_decision_latency(conn, "baseline", sample_size=50)
        ai_latency = measure_decision_latency(conn, "payback_ai", sample_size=50)
    finally:
        conn.close()
    assert baseline_latency["sample_size"] == 50
    assert ai_latency["sample_size"] == 50
    assert baseline_latency["avg_decision_latency_ms"] >= 0.0
    assert ai_latency["avg_decision_latency_ms"] >= 0.0


def test_measure_decision_latency_invalid_source_raises(fully_run_db: str) -> None:
    conn = get_connection(fully_run_db)
    try:
        with pytest.raises(ValueError):
            measure_decision_latency(conn, "not_a_real_source")
    finally:
        conn.close()


def test_run_evaluation_returns_combined_structure(fully_run_db: str) -> None:
    results = run_evaluation(db_path=fully_run_db, latency_sample_size=20)
    assert set(results.keys()) == {"baseline", "payback_ai", "cost_assumptions", "data_note"}
    assert "decision_latency" in results["baseline"]
    assert "decision_latency" in results["payback_ai"]


def test_run_evaluation_does_not_mutate_db(fully_run_db: str) -> None:
    conn = get_connection(fully_run_db)
    try:
        decisions_before = conn.execute("SELECT COUNT(*) AS c FROM recovery_decisions").fetchone()["c"]
        events_before = conn.execute("SELECT COUNT(*) AS c FROM action_events").fetchone()["c"]
    finally:
        conn.close()

    run_evaluation(db_path=fully_run_db, latency_sample_size=20)

    conn = get_connection(fully_run_db)
    try:
        decisions_after = conn.execute("SELECT COUNT(*) AS c FROM recovery_decisions").fetchone()["c"]
        events_after = conn.execute("SELECT COUNT(*) AS c FROM action_events").fetchone()["c"]
    finally:
        conn.close()
    assert decisions_before == decisions_after
    assert events_before == events_after


def test_write_json_report_produces_valid_json(tmp_path, fully_run_db: str) -> None:
    results = run_evaluation(db_path=fully_run_db, latency_sample_size=20)
    json_path = tmp_path / "results.json"
    write_json_report(results, path=json_path)
    parsed = json.loads(json_path.read_text())
    assert parsed["baseline"]["recovery_rate"] == results["baseline"]["recovery_rate"]


def test_write_markdown_report_produces_readable_table(tmp_path, fully_run_db: str) -> None:
    results = run_evaluation(db_path=fully_run_db, latency_sample_size=20)
    md_path = tmp_path / "results.md"
    write_markdown_report(results, path=md_path)
    content = md_path.read_text()
    assert "Recovery Rate" in content
    assert "Baseline (fixed-retry)" in content
    assert "PayBack AI" in content
    assert "Cost Assumptions" in content


def test_evaluation_is_reproducible(fully_run_db: str) -> None:
    results1 = run_evaluation(db_path=fully_run_db, latency_sample_size=20)
    results2 = run_evaluation(db_path=fully_run_db, latency_sample_size=20)
    assert results1["baseline"]["recovery_rate"] == results2["baseline"]["recovery_rate"]
    assert results1["payback_ai"]["recovery_rate"] == results2["payback_ai"]["recovery_rate"]
    assert results1["baseline"]["policy_violation_count"] == results2["baseline"]["policy_violation_count"]
