"""
PayBack AI - Full evaluation metrics.

Extends the base metrics from core/baseline.py and core/simulator.py with
every metric required for the baseline-vs-PayBack-AI comparison:
recovery rate, recovered revenue, average recovery time, unnecessary retry
rate, messages per recovered payment, cost per recovery, human-escalation
rate, false-positive decision rate, policy-violation count, and decision
latency.

DEFINITIONS (stated explicitly since some of these terms are ambiguous):
  - false_positive_decision_rate: share of transactions where the system
    chose 'stop_recovery' but the transaction was actually recoverable
    (ground truth) - i.e. gave up on something that could have been
    recovered. Measured against ground truth, which only exists because
    this is synthetic data with a known answer.
  - policy_violation_count: number of decisions where the policy engine
    had to override the decision engine's initial proposal (parsed from
    the "Policy engine override:" marker make_decision writes into its
    reason string). For the baseline this is always 0, since the baseline
    never goes through the policy engine at all - a real distinction
    worth surfacing, not a bug.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict

from core.baseline import BaselineConfig, compute_baseline_metrics, decide_baseline_action
from core.decision_engine import make_decision
from core.simulator import compute_payback_ai_metrics
from evaluation.costs import CostAssumptions


def _extended_metrics(
    conn: sqlite3.Connection, source: str, base_metrics: dict, cost_assumptions: CostAssumptions
) -> dict:
    total = base_metrics["total_transactions"]

    recovered_count = conn.execute(
        "SELECT COUNT(*) AS c FROM action_events WHERE source = ? AND status = 'recovered'", (source,)
    ).fetchone()["c"]

    avg_recovery_time = conn.execute(
        "SELECT AVG(p.time_since_failure_minutes) AS a FROM action_events ae "
        "JOIN payments p ON p.transaction_id = ae.transaction_id "
        "WHERE ae.source = ? AND ae.status = 'recovered'",
        (source,),
    ).fetchone()["a"]

    messages_sent = conn.execute(
        "SELECT COUNT(*) AS c FROM action_events WHERE source = ? AND action_type = 'reminder'", (source,)
    ).fetchone()["c"]

    retries_count = conn.execute(
        "SELECT COUNT(*) AS c FROM action_events WHERE source = ? AND action_type = 'delayed_retry'",
        (source,),
    ).fetchone()["c"]

    reviews_count = conn.execute(
        "SELECT COUNT(*) AS c FROM action_events WHERE source = ? AND action_type = 'human_escalation'",
        (source,),
    ).fetchone()["c"]

    false_positive_count = conn.execute(
        "SELECT COUNT(*) AS c FROM action_events ae "
        "JOIN payments p ON p.transaction_id = ae.transaction_id "
        "WHERE ae.source = ? AND ae.action_type = 'stop_recovery' AND p.recovery_outcome = 'recovered'",
        (source,),
    ).fetchone()["c"]

    if source == "payback_ai":
        policy_violation_count = conn.execute(
            "SELECT COUNT(*) AS c FROM recovery_decisions WHERE reason LIKE '%Policy engine override:%'"
        ).fetchone()["c"]
    else:
        # The baseline never goes through the policy engine at all, by design.
        policy_violation_count = 0

    total_cost = (
        retries_count * cost_assumptions.retry_cost_inr
        + messages_sent * cost_assumptions.message_cost_inr
        + reviews_count * cost_assumptions.human_review_cost_inr
    )
    cost_per_recovery = round(total_cost / recovered_count, 4) if recovered_count else 0.0

    return {
        **base_metrics,
        "average_recovery_time_minutes": round(avg_recovery_time, 2) if avg_recovery_time is not None else None,
        "unnecessary_retry_rate": round(base_metrics["unnecessary_retries"] / total, 4) if total else 0.0,
        "messages_per_recovered_payment": round(messages_sent / recovered_count, 4) if recovered_count else 0.0,
        "cost_per_recovery_inr": cost_per_recovery,
        "human_escalation_rate": round(reviews_count / total, 4) if total else 0.0,
        "false_positive_decision_rate": round(false_positive_count / total, 4) if total else 0.0,
        "policy_violation_count": policy_violation_count,
        "cost_assumptions": asdict(cost_assumptions),
    }


def compute_full_baseline_metrics(
    conn: sqlite3.Connection, cost_assumptions: CostAssumptions | None = None
) -> dict:
    cost_assumptions = cost_assumptions or CostAssumptions()
    base = compute_baseline_metrics(conn)
    return _extended_metrics(conn, "baseline", base, cost_assumptions)


def compute_full_payback_ai_metrics(
    conn: sqlite3.Connection, cost_assumptions: CostAssumptions | None = None
) -> dict:
    cost_assumptions = cost_assumptions or CostAssumptions()
    base = compute_payback_ai_metrics(conn)
    return _extended_metrics(conn, "payback_ai", base, cost_assumptions)


def measure_decision_latency(conn: sqlite3.Connection, source: str, sample_size: int = 100) -> dict:
    """
    Measure real wall-clock decision latency by actually timing the
    decision function. NOTE: baseline's decision (decide_baseline_action)
    is a pure O(1) lookup with no DB access, while PayBack AI's
    (make_decision) performs several real SQL reads - these numbers are
    intentionally NOT apples-to-apples; that asymmetry is itself part of
    what the comparison is meant to show.
    """
    if source == "baseline":
        rows = conn.execute(
            "SELECT attempt_number FROM payments ORDER BY transaction_id LIMIT ?", (sample_size,)
        ).fetchall()
        config = BaselineConfig()
        durations = []
        for row in rows:
            t0 = time.perf_counter()
            decide_baseline_action(row["attempt_number"], config)
            durations.append(time.perf_counter() - t0)
    elif source == "payback_ai":
        txn_ids = [
            r["transaction_id"]
            for r in conn.execute(
                "SELECT transaction_id FROM payments ORDER BY transaction_id LIMIT ?", (sample_size,)
            ).fetchall()
        ]
        durations = []
        for txn_id in txn_ids:
            t0 = time.perf_counter()
            make_decision(conn, txn_id)  # read-only, safe to time repeatedly
            durations.append(time.perf_counter() - t0)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    avg_ms = (sum(durations) / len(durations)) * 1000 if durations else 0.0
    return {"sample_size": len(durations), "avg_decision_latency_ms": round(avg_ms, 4)}
