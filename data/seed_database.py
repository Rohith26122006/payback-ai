"""
PayBack AI - Seed the database from data/synthetic_payments.csv.

Loads distinct merchants into `merchants` and every row into `payments`.
Does NOT touch model_predictions/recovery_decisions/action_events/audit_logs/
review_queue - those are populated by later stages.

Usage:
    python -m data.seed_database --csv data/synthetic_payments.csv --db payback.db --reset
"""
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from app.database import get_connection, init_db, reset_db

PAYMENTS_COLUMNS = [
    "transaction_id",
    "merchant_id",
    "customer_id_hash",
    "amount_inr",
    "payment_method",
    "failure_code",
    "failure_category",
    "attempt_number",
    "customer_success_rate",
    "previous_failed_attempts",
    "time_since_failure_minutes",
    "subscription_flag",
    "merchant_category",
    "device_change_flag",
    "recovery_outcome",
    "recovered_amount",
    "event_timestamp",
]


def seed_merchants(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    merchants = (
        df[["merchant_id", "merchant_category"]]
        .drop_duplicates(subset="merchant_id")
        .itertuples(index=False, name=None)
    )
    rows = list(merchants)
    conn.executemany(
        "INSERT OR IGNORE INTO merchants (merchant_id, merchant_category) VALUES (?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def seed_payments(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    df = df.copy()
    # SQLite has no native bool - store as 0/1 integers.
    df["subscription_flag"] = df["subscription_flag"].astype(bool).astype(int)
    df["device_change_flag"] = df["device_change_flag"].astype(bool).astype(int)

    rows = list(df[PAYMENTS_COLUMNS].itertuples(index=False, name=None))
    placeholders = ", ".join(["?"] * len(PAYMENTS_COLUMNS))
    columns_sql = ", ".join(PAYMENTS_COLUMNS)
    conn.executemany(
        f"INSERT OR IGNORE INTO payments ({columns_sql}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    return len(rows)


def seed_database(csv_path: str, db_path: str, reset: bool) -> tuple[int, int]:
    if reset:
        reset_db(db_path)
    else:
        init_db(db_path)

    df = pd.read_csv(csv_path)
    conn = get_connection(db_path)
    try:
        n_merchants = seed_merchants(conn, df)
        n_payments = seed_payments(conn, df)
    finally:
        conn.close()
    return n_merchants, n_payments


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed PayBack AI database from synthetic CSV.")
    parser.add_argument("--csv", type=str, default="data/synthetic_payments.csv")
    parser.add_argument("--db", type=str, default="payback.db")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables before seeding.",
    )
    args = parser.parse_args()

    n_merchants, n_payments = seed_database(args.csv, args.db, args.reset)
    print(
        f"[SEED] Loaded {n_merchants} merchants and {n_payments} payments "
        f"into {args.db} (source: {args.csv}, all data SYNTHETIC)"
    )


if __name__ == "__main__":
    main()
