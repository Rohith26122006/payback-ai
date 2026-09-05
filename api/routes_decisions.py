"""
PayBack AI - /decisions routes.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.database import get_db
from app.schemas import DecisionResponse
from core.audit_logger import log_event
from core.decision_engine import make_decision

router = APIRouter(prefix="/decisions", tags=["decisions"])


def _payment_exists(conn: sqlite3.Connection, transaction_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM payments WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    return row is not None


@router.get("/{transaction_id}", response_model=DecisionResponse)
def get_decision(transaction_id: str, conn: sqlite3.Connection = Depends(get_db)) -> DecisionResponse:
    """Compute the decision live (read-only, does not persist)."""
    if not _payment_exists(conn, transaction_id):
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id!r} not found")
    output = make_decision(conn, transaction_id)
    return DecisionResponse(**output.to_dict())


@router.post("/{transaction_id}/run", response_model=DecisionResponse)
def run_decision(transaction_id: str, conn: sqlite3.Connection = Depends(get_db)) -> DecisionResponse:
    """Compute the decision fresh and persist it to recovery_decisions (replaces any prior row)."""
    if not _payment_exists(conn, transaction_id):
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id!r} not found")

    output = make_decision(conn, transaction_id)

    conn.execute("DELETE FROM recovery_decisions WHERE transaction_id = ?", (transaction_id,))
    conn.execute(
        "INSERT INTO recovery_decisions "
        "(transaction_id, action, reason, safety_checks, requires_human_review, decided_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            output.transaction_id,
            output.action,
            output.reason,
            json.dumps(output.safety_checks),
            int(output.requires_human_review),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()

    log_event(
        conn,
        transaction_id,
        actor="api",
        event_type="decision_recomputed",
        detail=f"action={output.action} requires_human_review={output.requires_human_review}",
    )

    return DecisionResponse(**output.to_dict())
