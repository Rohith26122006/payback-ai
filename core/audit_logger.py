"""
PayBack AI - Audit logger.

Append-only: this module intentionally provides no update or delete
functions. Every state transition anywhere in the pipeline should be
logged here for the audit-log dashboard view (Stage 12/13).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(
    conn: sqlite3.Connection,
    transaction_id: str,
    actor: str,
    event_type: str,
    detail: str,
    commit: bool = True,
) -> None:
    """
    Append one audit entry. `commit` defaults to True for standalone calls;
    callers doing many inserts in a loop can pass commit=False and commit
    once at the end for performance.
    """
    conn.execute(
        "INSERT INTO audit_logs (transaction_id, actor, event_type, detail, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (transaction_id, actor, event_type, detail, _now_iso()),
    )
    if commit:
        conn.commit()


def get_audit_trail(conn: sqlite3.Connection, transaction_id: str) -> list[dict]:
    """Return the full audit history for one transaction, oldest first."""
    rows = conn.execute(
        "SELECT actor, event_type, detail, created_at FROM audit_logs "
        "WHERE transaction_id = ? ORDER BY created_at, id",
        (transaction_id,),
    ).fetchall()
    return [dict(row) for row in rows]
