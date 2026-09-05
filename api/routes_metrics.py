"""
PayBack AI - /metrics routes.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends

from app.database import get_db
from core.baseline import compute_baseline_metrics
from core.simulator import compute_payback_ai_metrics

router = APIRouter(prefix="/metrics", tags=["metrics"])

RECOVERY_MODEL_METRICS_PATH = Path("models/recovery_model_metrics.json")


@router.get("/baseline")
def baseline_metrics(conn: sqlite3.Connection = Depends(get_db)) -> dict:
    return compute_baseline_metrics(conn)


@router.get("/payback-ai")
def payback_ai_metrics(conn: sqlite3.Connection = Depends(get_db)) -> dict:
    return compute_payback_ai_metrics(conn)


@router.get("/comparison")
def comparison_metrics(conn: sqlite3.Connection = Depends(get_db)) -> dict:
    return {
        "baseline": compute_baseline_metrics(conn),
        "payback_ai": compute_payback_ai_metrics(conn),
    }


@router.get("/recovery-model")
def recovery_model_metrics() -> dict:
    if not RECOVERY_MODEL_METRICS_PATH.exists():
        return {
            "detail": (
                "Recovery model metrics not found. Run `python -m core.recovery_model` first."
            )
        }
    return json.loads(RECOVERY_MODEL_METRICS_PATH.read_text())
