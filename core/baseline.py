"""
PayBack AI - Fixed-retry baseline recovery engine.

Deliberately simple/"dumb" comparison point for PayBack AI's decision engine:
  - Retry once, after a fixed delay.
  - If that's not applicable, send one generic reminder.
  - Stop once the configured maximum attempts is exceeded.

This baseline does NOT look at failure_category, risk signals, or anything
else - only attempt_number. That is intentional: it must stay genuinely dumb
so the Stage 14 comparison against PayBack AI is meaningful.

ASSUMPTION (documented, not hidden): the synthetic `recovery_outcome` /
`recovered_amount` fields in `payments` represent whether a transaction was
recoverable AT ALL, independent of strategy. The baseline only realizes that
recovery if it actually takes an action (delayed_retry or reminder); if it
stops early, that potential revenue counts as missed, not recovered.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from app.database import get_connection

BASELINE_SOURCE = "baseline"


@dataclass(frozen=True)
class BaselineConfig:
    max_retry_attempts: int = 1  # retry once
    retry_delay_minutes: int = 30  # fixed delay before the retry
    max_attempts: int = 2  # stop once attempt_number exceeds this


def decide_baseline_action(attempt_number: int, config: BaselineConfig) -> str:
    """Pure decision function: attempt_number -> baseline action. No DB access."""
    if attempt_number > config.max_attempts:
        return "stop_recovery"
    if attempt_number <= config.max_retry_attempts:
        return "delayed_retry"
    return "reminder"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_baseline(
    conn: sqlite3.Connection, config: BaselineConfig | None = None
) -> list[dict]:
    """
    Run the baseline policy over every row in `payments`, record each
    decision + simulated outcome to `action_events` (source='baseline'),
    and return the list of per-transaction results.

    Idempotent: clears prior baseline action_events before re-running.
    """
    config = config or BaselineConfig()

    conn.execute("DELETE FROM action_events WHERE source = ?", (BASELINE_SOURCE,))
    conn.commit()

    rows = conn.execute(
        "SELECT transaction_id, attempt_number, recovery_outcome, recovered_amount "
        "FROM payments"
    ).fetchall()

    results: list[dict] = []
    executed_at = _now_iso()

    for row in rows:
        action = decide_baseline_action(row["attempt_number"], config)

        attempted = action in ("delayed_retry", "reminder")
        recovered = attempted and row["recovery_outcome"] == "recovered"
        status = "recovered" if recovered else ("not_recovered" if attempted else "skipped")

        conn.execute(
            "INSERT INTO action_events "
            "(transaction_id, action_type, source, executed_at, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (row["transaction_id"], action, BASELINE_SOURCE, executed_at, status),
        )

        results.append(
            {
                "transaction_id": row["transaction_id"],
                "action": action,
                "status": status,
                "recovered_amount": row["recovered_amount"] if recovered else 0.0,
            }
        )

    conn.commit()
    return results


def compute_baseline_metrics(conn: sqlite3.Connection) -> dict:
    """
    Compute baseline metrics from `action_events` (source='baseline') joined
    against `payments` for amount_inr. Assumes run_baseline() has been called.
    """
    rows = conn.execute(
        """
        SELECT ae.action_type, ae.status, p.amount_inr
        FROM action_events ae
        JOIN payments p ON p.transaction_id = ae.transaction_id
        WHERE ae.source = ?
        """,
        (BASELINE_SOURCE,),
    ).fetchall()

    total_transactions = len(rows)
    total_at_risk_revenue = sum(r["amount_inr"] for r in rows)
    recovered_rows = [r for r in rows if r["status"] == "recovered"]
    recovered_revenue = sum(r["amount_inr"] for r in recovered_rows)

    action_distribution: dict[str, int] = {}
    for r in rows:
        action_distribution[r["action_type"]] = action_distribution.get(r["action_type"], 0) + 1

    unnecessary_retries = sum(
        1 for r in rows if r["action_type"] in ("delayed_retry", "reminder") and r["status"] == "not_recovered"
    )
    stopped_count = action_distribution.get("stop_recovery", 0)

    return {
        "total_transactions": total_transactions,
        "total_at_risk_revenue": round(total_at_risk_revenue, 2),
        "recovered_revenue": round(recovered_revenue, 2),
        "recovery_rate": round(len(recovered_rows) / total_transactions, 4)
        if total_transactions
        else 0.0,
        "action_distribution": action_distribution,
        "unnecessary_retries": unnecessary_retries,
        "stopped_count": stopped_count,
    }


if __name__ == "__main__":
    conn = get_connection()
    run_baseline(conn)
    metrics = compute_baseline_metrics(conn)
    conn.close()
    print("[BASELINE] Metrics (SIMULATED outcomes, synthetic data):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
