"""
PayBack AI - Payment simulator.

Executes the action chosen by the decision engine (core/decision_engine.py)
for every transaction and records a SIMULATED outcome. Nothing here talks
to any real payment gateway - every status written is clearly a simulation.

Simulated scenarios (per spec):
  - Retry success / retry failure           (delayed_retry)
  - Payment-link success / failure           (alternate_payment_link)
  - Customer ignoring a message              (reminder, no response)
  - Human approval / rejection               (human_escalation)
  - Payment success cancelling future actions (cancel_pending_actions_after_success)

ASSUMPTIONS (documented, not measured from any real system):
  - reminder_response_rate: probability the customer responds to a reminder
    at all, before ground-truth recoverability is applied.
  - human_approval_probability: probability a human reviewer approves
    recovery action on an escalated transaction.
  Both are configurable in SimulatorConfig and clearly labeled as assumptions.

Reuses the same leakage-safe assumption as the baseline (Stage 4): the
synthetic `recovery_outcome` field represents whether a transaction was
recoverable AT ALL, independent of strategy - realized only if an action
is actually attempted.
"""
from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from core.audit_logger import log_event

SIMULATOR_SOURCE = "payback_ai"

ATTEMPT_ACTIONS = {"delayed_retry", "alternate_payment_link"}
NO_OUTCOME_ACTIONS = {"stop_recovery", "no_action"}


@dataclass(frozen=True)
class SimulatorConfig:
    reminder_response_rate: float = 0.55  # ASSUMPTION
    human_approval_probability: float = 0.65  # ASSUMPTION
    seed: int = 42


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def simulate_action(
    rng: random.Random,
    action: str,
    ground_truth_recovered: bool,
    recovered_amount: float,
    config: SimulatorConfig,
) -> tuple[str, float]:
    """
    Pure simulation step (no DB). Returns (status, simulated_recovered_amount).
    status in {'recovered', 'not_recovered', 'customer_ignored', 'human_rejected', 'skipped'}.
    """
    if action in NO_OUTCOME_ACTIONS:
        return "skipped", 0.0

    if action == "reminder":
        if rng.random() >= config.reminder_response_rate:
            return "customer_ignored", 0.0
        return ("recovered", recovered_amount) if ground_truth_recovered else ("not_recovered", 0.0)

    if action == "human_escalation":
        if rng.random() >= config.human_approval_probability:
            return "human_rejected", 0.0
        return ("recovered", recovered_amount) if ground_truth_recovered else ("not_recovered", 0.0)

    if action in ATTEMPT_ACTIONS:
        return ("recovered", recovered_amount) if ground_truth_recovered else ("not_recovered", 0.0)

    # Unrecognized action - safe default, no outcome simulated.
    return "skipped", 0.0


def cancel_pending_actions_after_success(conn: sqlite3.Connection, transaction_id: str) -> int:
    """
    Mark any other still-pending action_events for this transaction as
    'cancelled', since the payment has already been recovered. Returns the
    number of rows cancelled.
    """
    cur = conn.execute(
        "UPDATE action_events SET status = 'cancelled' "
        "WHERE transaction_id = ? AND source = ? AND status NOT IN ('recovered', 'cancelled')",
        (transaction_id, SIMULATOR_SOURCE),
    )
    return cur.rowcount


def run_simulator_for_all(
    conn: sqlite3.Connection, config: SimulatorConfig | None = None
) -> list[dict]:
    """
    Execute the action from `recovery_decisions` for every transaction,
    record the simulated outcome to `action_events` (source='payback_ai'),
    and log every step to `audit_logs`. Idempotent: clears prior payback_ai
    action_events before running. Deterministic given the same seed and
    the same recovery_decisions content (transactions processed in sorted
    transaction_id order).
    """
    config = config or SimulatorConfig()

    conn.execute("DELETE FROM action_events WHERE source = ?", (SIMULATOR_SOURCE,))
    conn.commit()

    rows = conn.execute(
        """
        SELECT rd.transaction_id, rd.action, p.recovery_outcome, p.recovered_amount
        FROM recovery_decisions rd
        JOIN payments p ON p.transaction_id = rd.transaction_id
        ORDER BY rd.transaction_id
        """
    ).fetchall()

    rng = random.Random(config.seed)
    executed_at = _now_iso()
    results: list[dict] = []

    for row in rows:
        ground_truth_recovered = row["recovery_outcome"] == "recovered"
        status, amount = simulate_action(
            rng, row["action"], ground_truth_recovered, row["recovered_amount"], config
        )

        conn.execute(
            "INSERT INTO action_events "
            "(transaction_id, action_type, source, executed_at, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (row["transaction_id"], row["action"], SIMULATOR_SOURCE, executed_at, status),
        )
        log_event(
            conn,
            row["transaction_id"],
            actor="simulator",
            event_type=f"action_simulated:{row['action']}",
            detail=f"Simulated action '{row['action']}' -> status '{status}' "
            f"(amount={amount}). All outcomes SIMULATED, not real.",
            commit=False,
        )

        if status == "recovered":
            cancelled = cancel_pending_actions_after_success(conn, row["transaction_id"])
            if cancelled:
                log_event(
                    conn,
                    row["transaction_id"],
                    actor="simulator",
                    event_type="pending_actions_cancelled",
                    detail=f"Payment recovered; cancelled {cancelled} pending action(s).",
                    commit=False,
                )

        results.append(
            {
                "transaction_id": row["transaction_id"],
                "action": row["action"],
                "status": status,
                "recovered_amount": amount,
            }
        )

    conn.commit()
    return results


def compute_payback_ai_metrics(conn: sqlite3.Connection) -> dict:
    """Metrics for the payback_ai source, mirroring compute_baseline_metrics for comparability."""
    rows = conn.execute(
        """
        SELECT ae.action_type, ae.status, p.amount_inr
        FROM action_events ae
        JOIN payments p ON p.transaction_id = ae.transaction_id
        WHERE ae.source = ?
        """,
        (SIMULATOR_SOURCE,),
    ).fetchall()

    total_transactions = len(rows)
    total_at_risk_revenue = sum(r["amount_inr"] for r in rows)
    recovered_rows = [r for r in rows if r["status"] == "recovered"]
    recovered_revenue = sum(r["amount_inr"] for r in recovered_rows)

    status_distribution: dict[str, int] = {}
    action_distribution: dict[str, int] = {}
    for r in rows:
        status_distribution[r["status"]] = status_distribution.get(r["status"], 0) + 1
        action_distribution[r["action_type"]] = action_distribution.get(r["action_type"], 0) + 1

    unnecessary_retries = sum(
        1
        for r in rows
        if r["action_type"] in ("delayed_retry", "alternate_payment_link")
        and r["status"] == "not_recovered"
    )

    return {
        "total_transactions": total_transactions,
        "total_at_risk_revenue": round(total_at_risk_revenue, 2),
        "recovered_revenue": round(recovered_revenue, 2),
        "recovery_rate": round(len(recovered_rows) / total_transactions, 4)
        if total_transactions
        else 0.0,
        "action_distribution": action_distribution,
        "status_distribution": status_distribution,
        "unnecessary_retries": unnecessary_retries,
    }


if __name__ == "__main__":
    from app.database import get_connection

    conn = get_connection()
    run_simulator_for_all(conn)
    metrics = compute_payback_ai_metrics(conn)
    conn.close()

    print("[SIMULATOR] PayBack AI metrics (SIMULATED outcomes, synthetic data):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
