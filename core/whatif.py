"""
PayBack AI - What-if simulator.

Recomputes decisions and metrics under a HYPOTHETICAL PolicyConfig, entirely
in memory. Nothing is written to recovery_decisions, action_events, or any
other table - this is purely exploratory.

LIMITATION (documented, not hidden): each transaction is evaluated as a
single fresh decision under the new policy - as if this were the first
time a decision was made for it (retry_count_so_far=0, no cooldown, no
message history, not already recovered). It intentionally does NOT read
the real run's action_events history (see make_decision(..., ignore_history=True)
in core/decision_engine.py) - reusing that history would make the what-if
results depend on exactly when the real pipeline happened to run, which
is not a meaningful counterfactual. This is not a full multi-round
re-simulation of repeated attempts over time either - for this single-shot
synthetic dataset (one event per transaction), that would require modeling
several rounds explicitly, which is out of scope here. Read the what-if
numbers as "how would this policy score on a fresh first pass over this
batch", not as a perfect replay of history under a different policy.
"""
from __future__ import annotations

import random
import sqlite3
from dataclasses import asdict

from app.database import get_connection
from core.decision_engine import DecisionConfig, make_decision
from core.policy_engine import PolicyConfig
from core.simulator import SimulatorConfig, simulate_action

WHAT_IF_NOTE = "WHAT-IF SIMULATION: hypothetical only, nothing persisted to the database."


def run_what_if(
    conn: sqlite3.Connection,
    policy_config: PolicyConfig,
    decision_config: DecisionConfig | None = None,
    simulator_config: SimulatorConfig | None = None,
) -> dict:
    simulator_config = simulator_config or SimulatorConfig()
    rng = random.Random(simulator_config.seed)

    payment_rows = conn.execute(
        "SELECT transaction_id, amount_inr, recovery_outcome, recovered_amount FROM payments "
        "ORDER BY transaction_id"
    ).fetchall()

    total_at_risk = 0.0
    recovered_revenue = 0.0
    action_distribution: dict[str, int] = {}
    status_distribution: dict[str, int] = {}
    human_review_count = 0

    for row in payment_rows:
        txn_id = row["transaction_id"]
        decision = make_decision(conn, txn_id, decision_config, policy_config, ignore_history=True)

        total_at_risk += row["amount_inr"]
        ground_truth_recovered = row["recovery_outcome"] == "recovered"
        status, amount = simulate_action(
            rng, decision.action, ground_truth_recovered, row["recovered_amount"], simulator_config
        )
        if status == "recovered":
            recovered_revenue += amount

        action_distribution[decision.action] = action_distribution.get(decision.action, 0) + 1
        status_distribution[status] = status_distribution.get(status, 0) + 1
        if decision.requires_human_review:
            human_review_count += 1

    total = len(payment_rows)
    recovered_count = status_distribution.get("recovered", 0)

    return {
        "policy_config": asdict(policy_config),
        "total_transactions": total,
        "total_at_risk_revenue": round(total_at_risk, 2),
        "recovered_revenue": round(recovered_revenue, 2),
        "recovery_rate": round(recovered_count / total, 4) if total else 0.0,
        "action_distribution": action_distribution,
        "status_distribution": status_distribution,
        "human_review_count": human_review_count,
        "note": WHAT_IF_NOTE,
    }


def run_what_if_from_db(
    db_path: str,
    policy_config: PolicyConfig,
    decision_config: DecisionConfig | None = None,
    simulator_config: SimulatorConfig | None = None,
) -> dict:
    """Convenience wrapper that opens/closes its own connection - useful for cache-friendly callers (e.g. Streamlit)."""
    conn = get_connection(db_path)
    try:
        return run_what_if(conn, policy_config, decision_config, simulator_config)
    finally:
        conn.close()
