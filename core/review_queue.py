"""
PayBack AI - Human review queue.

review_queue is populated from recovery_decisions (any row with
requires_human_review=1) and resolved manually via the dashboard.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from core.audit_logger import log_event


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sync_review_queue(conn: sqlite3.Connection) -> int:
    """
    Ensure every transaction currently requiring human review has an 'open'
    review_queue entry. Idempotent - never creates a duplicate open entry
    for the same transaction. Returns the number of entries newly added.
    """
    rows = conn.execute(
        "SELECT transaction_id, reason FROM recovery_decisions WHERE requires_human_review = 1"
    ).fetchall()

    added = 0
    now = _now_iso()
    for row in rows:
        existing = conn.execute(
            "SELECT 1 FROM review_queue WHERE transaction_id = ? AND status = 'open'",
            (row["transaction_id"],),
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO review_queue (transaction_id, escalation_reason, status, added_at) "
                "VALUES (?, ?, 'open', ?)",
                (row["transaction_id"], row["reason"], now),
            )
            added += 1
    conn.commit()
    return added


def list_open_reviews(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT rq.id, rq.transaction_id, rq.escalation_reason, rq.added_at,
               p.amount_inr, p.failure_category, rd.action
        FROM review_queue rq
        JOIN payments p ON p.transaction_id = rq.transaction_id
        LEFT JOIN recovery_decisions rd ON rd.transaction_id = rq.transaction_id
        WHERE rq.status = 'open'
        ORDER BY p.amount_inr DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def resolve_review_item(
    conn: sqlite3.Connection, review_id: int, resolved_by: str = "merchant", note: str = ""
) -> None:
    """Mark one review_queue entry as resolved and log it to the audit trail."""
    row = conn.execute("SELECT * FROM review_queue WHERE id = ?", (review_id,)).fetchone()
    if row is None:
        raise ValueError(f"No review_queue entry with id={review_id}")

    conn.execute("UPDATE review_queue SET status = 'resolved' WHERE id = ?", (review_id,))
    conn.commit()

    log_event(
        conn,
        row["transaction_id"],
        actor=resolved_by,
        event_type="review_resolved",
        detail=note or "Marked resolved via dashboard.",
    )
