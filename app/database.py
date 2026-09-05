"""
PayBack AI - SQLite schema and connection helpers.

Stage 3 scope only: table definitions + connection/init utilities.
No ML, policy, decision, or LLM logic lives here.

Tables:
  merchants          - merchant_id -> merchant_category
  payments           - the raw synthetic failed-payment event
  model_predictions  - classifier + recovery-probability output (Stage 5/6)
  recovery_decisions - decision engine output (Stage 8)
  action_events      - baseline vs payback_ai executed actions (Stage 4/9)
  audit_logs         - append-only audit trail (Stage 9)
  review_queue       - human-review escalations (Stage 8/13)
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = "payback.db"

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS merchants (
    merchant_id TEXT PRIMARY KEY,
    merchant_category TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payments (
    transaction_id TEXT PRIMARY KEY,
    merchant_id TEXT NOT NULL,
    customer_id_hash TEXT NOT NULL,
    amount_inr REAL NOT NULL,
    payment_method TEXT NOT NULL,
    failure_code TEXT NOT NULL,
    failure_category TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    customer_success_rate REAL NOT NULL,
    previous_failed_attempts INTEGER NOT NULL,
    time_since_failure_minutes INTEGER NOT NULL,
    subscription_flag INTEGER NOT NULL CHECK (subscription_flag IN (0, 1)),
    merchant_category TEXT NOT NULL,
    device_change_flag INTEGER NOT NULL CHECK (device_change_flag IN (0, 1)),
    recovery_outcome TEXT NOT NULL,
    recovered_amount REAL NOT NULL,
    event_timestamp TEXT NOT NULL,
    FOREIGN KEY (merchant_id) REFERENCES merchants (merchant_id)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    classified_category TEXT NOT NULL,
    classification_confidence REAL NOT NULL,
    classification_reason TEXT NOT NULL,
    recovery_probability REAL,
    model_version TEXT NOT NULL,
    predicted_at TEXT NOT NULL,
    FOREIGN KEY (transaction_id) REFERENCES payments (transaction_id)
);

CREATE TABLE IF NOT EXISTS recovery_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    safety_checks TEXT NOT NULL,
    requires_human_review INTEGER NOT NULL CHECK (requires_human_review IN (0, 1)),
    decided_at TEXT NOT NULL,
    FOREIGN KEY (transaction_id) REFERENCES payments (transaction_id)
);

CREATE TABLE IF NOT EXISTS action_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('baseline', 'payback_ai')),
    executed_at TEXT NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY (transaction_id) REFERENCES payments (transaction_id)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    escalation_reason TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'resolved')) DEFAULT 'open',
    added_at TEXT NOT NULL,
    FOREIGN KEY (transaction_id) REFERENCES payments (transaction_id)
);

CREATE INDEX IF NOT EXISTS idx_payments_merchant ON payments (merchant_id);
CREATE INDEX IF NOT EXISTS idx_predictions_txn ON model_predictions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_decisions_txn ON recovery_decisions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_actions_txn ON action_events (transaction_id);
CREATE INDEX IF NOT EXISTS idx_audit_txn ON audit_logs (transaction_id);
CREATE INDEX IF NOT EXISTS idx_review_txn ON review_queue (transaction_id);
"""


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enforced and dict-like rows."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create all tables (and indexes) if they do not already exist."""
    conn = get_connection(db_path)
    try:
        conn.executescript(CREATE_TABLES_SQL)
        conn.commit()
    finally:
        conn.close()


def reset_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Drop and recreate all tables. Used by the seed script and tests."""
    conn = get_connection(db_path)
    try:
        tables = [
            "review_queue",
            "audit_logs",
            "action_events",
            "recovery_decisions",
            "model_predictions",
            "payments",
            "merchants",
        ]
        for table in tables:
            conn.execute(f"DROP TABLE IF EXISTS {table};")
        conn.commit()
    finally:
        conn.close()
    init_db(db_path)


def resolve_db_path() -> str:
    """
    Resolve the SQLite path from the DATABASE_URL env var (e.g.
    'sqlite:///./payback.db'), falling back to DEFAULT_DB_PATH. Used by
    both the FastAPI dependency below and anything else that wants to
    respect .env without duplicating this parsing logic.
    """
    url = os.environ.get("DATABASE_URL", DEFAULT_DB_PATH)
    prefix = "sqlite:///"
    if url.startswith(prefix):
        return url[len(prefix) :]
    return url


def get_db():
    """FastAPI dependency: yields a connection, always closes it after the request."""
    conn = get_connection(resolve_db_path())
    try:
        yield conn
    finally:
        conn.close()
