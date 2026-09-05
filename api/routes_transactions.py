"""
PayBack AI - /transactions routes.
"""
from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query

from app.database import get_db
from app.schemas import DecisionResponse, PaymentDetail, PaymentSummary

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.get("", response_model=list[PaymentSummary])
def list_transactions(
    failure_category: str | None = Query(default=None),
    merchant_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[dict]:
    query = (
        "SELECT transaction_id, merchant_id, amount_inr, payment_method, "
        "failure_category, attempt_number, event_timestamp FROM payments WHERE 1=1"
    )
    params: list = []
    if failure_category:
        query += " AND failure_category = ?"
        params.append(failure_category)
    if merchant_id:
        query += " AND merchant_id = ?"
        params.append(merchant_id)
    query += " ORDER BY transaction_id LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


@router.get("/{transaction_id}", response_model=PaymentDetail)
def get_transaction(transaction_id: str, conn: sqlite3.Connection = Depends(get_db)) -> PaymentDetail:
    payment = conn.execute(
        "SELECT * FROM payments WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    if payment is None:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id!r} not found")

    prediction = conn.execute(
        "SELECT * FROM model_predictions WHERE transaction_id = ? ORDER BY id DESC LIMIT 1",
        (transaction_id,),
    ).fetchone()

    decision_row = conn.execute(
        "SELECT * FROM recovery_decisions WHERE transaction_id = ? ORDER BY id DESC LIMIT 1",
        (transaction_id,),
    ).fetchone()

    decision = None
    if decision_row is not None:
        decision = DecisionResponse(
            transaction_id=transaction_id,
            action=decision_row["action"],
            recovery_probability=prediction["recovery_probability"] if prediction else None,
            confidence=prediction["classification_confidence"] if prediction else 0.0,
            reason=decision_row["reason"],
            safety_checks=json.loads(decision_row["safety_checks"]),
            requires_human_review=bool(decision_row["requires_human_review"]),
        )

    baseline_event = conn.execute(
        "SELECT status FROM action_events WHERE transaction_id = ? AND source = 'baseline' "
        "ORDER BY id DESC LIMIT 1",
        (transaction_id,),
    ).fetchone()
    ai_event = conn.execute(
        "SELECT status FROM action_events WHERE transaction_id = ? AND source = 'payback_ai' "
        "ORDER BY id DESC LIMIT 1",
        (transaction_id,),
    ).fetchone()

    return PaymentDetail(
        transaction_id=payment["transaction_id"],
        merchant_id=payment["merchant_id"],
        amount_inr=payment["amount_inr"],
        payment_method=payment["payment_method"],
        failure_category=payment["failure_category"],
        attempt_number=payment["attempt_number"],
        event_timestamp=payment["event_timestamp"],
        customer_id_hash=payment["customer_id_hash"],
        failure_code=payment["failure_code"],
        subscription_flag=bool(payment["subscription_flag"]),
        device_change_flag=bool(payment["device_change_flag"]),
        merchant_category=payment["merchant_category"],
        classified_category=prediction["classified_category"] if prediction else None,
        classification_confidence=prediction["classification_confidence"] if prediction else None,
        recovery_probability=prediction["recovery_probability"] if prediction else None,
        decision=decision,
        baseline_action_status=baseline_event["status"] if baseline_event else None,
        payback_ai_action_status=ai_event["status"] if ai_event else None,
    )
