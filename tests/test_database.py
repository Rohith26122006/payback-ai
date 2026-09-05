import sqlite3

import pandas as pd
import pytest

from app.database import get_connection, init_db, reset_db
from data.generate_synthetic_data import GeneratorConfig, generate_dataset
from data.seed_database import seed_database

EXPECTED_TABLES = {
    "merchants",
    "payments",
    "model_predictions",
    "recovery_decisions",
    "action_events",
    "audit_logs",
    "review_queue",
}


@pytest.fixture()
def tmp_db_path(tmp_path):
    return str(tmp_path / "test_payback.db")


def test_init_db_creates_all_tables(tmp_db_path: str) -> None:
    init_db(tmp_db_path)
    conn = get_connection(tmp_db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row["name"] for row in rows}
    finally:
        conn.close()
    assert EXPECTED_TABLES.issubset(table_names)


def test_foreign_key_enforced_for_payments(tmp_db_path: str) -> None:
    init_db(tmp_db_path)
    conn = get_connection(tmp_db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO payments (
                    transaction_id, merchant_id, customer_id_hash, amount_inr,
                    payment_method, failure_code, failure_category, attempt_number,
                    customer_success_rate, previous_failed_attempts,
                    time_since_failure_minutes, subscription_flag, merchant_category,
                    device_change_flag, recovery_outcome, recovered_amount, event_timestamp
                ) VALUES (
                    'TXN000001', 'NONEXISTENT_MERCHANT', 'abc123', 100.0,
                    'upi', 'BANK_TIMEOUT', 'temporary_bank_or_network', 1,
                    0.5, 0, 60, 0, 'e_commerce', 0, 'not_recovered', 0.0,
                    '2026-08-01T00:00:00'
                )
                """
            )
            conn.commit()
    finally:
        conn.close()


def test_reset_db_clears_existing_data(tmp_db_path: str) -> None:
    init_db(tmp_db_path)
    conn = get_connection(tmp_db_path)
    try:
        conn.execute(
            "INSERT INTO merchants (merchant_id, merchant_category) VALUES ('M001', 'e_commerce')"
        )
        conn.commit()
    finally:
        conn.close()

    reset_db(tmp_db_path)

    conn = get_connection(tmp_db_path)
    try:
        count = conn.execute("SELECT COUNT(*) AS c FROM merchants").fetchone()["c"]
    finally:
        conn.close()
    assert count == 0


def test_seed_database_loads_expected_row_counts(tmp_path, tmp_db_path: str) -> None:
    df = generate_dataset(GeneratorConfig(n_records=500, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)

    n_merchants, n_payments = seed_database(str(csv_path), tmp_db_path, reset=True)

    assert n_payments == 500
    assert n_merchants == df["merchant_id"].nunique()

    conn = get_connection(tmp_db_path)
    try:
        payments_count = conn.execute("SELECT COUNT(*) AS c FROM payments").fetchone()["c"]
        merchants_count = conn.execute("SELECT COUNT(*) AS c FROM merchants").fetchone()["c"]
    finally:
        conn.close()
    assert payments_count == 500
    assert merchants_count == df["merchant_id"].nunique()


def test_seed_database_is_idempotent_on_rerun(tmp_path, tmp_db_path: str) -> None:
    df = generate_dataset(GeneratorConfig(n_records=200, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)

    seed_database(str(csv_path), tmp_db_path, reset=True)
    # Re-run without reset - INSERT OR IGNORE should not duplicate or error.
    seed_database(str(csv_path), tmp_db_path, reset=False)

    conn = get_connection(tmp_db_path)
    try:
        payments_count = conn.execute("SELECT COUNT(*) AS c FROM payments").fetchone()["c"]
    finally:
        conn.close()
    assert payments_count == 200
