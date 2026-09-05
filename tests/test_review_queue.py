import pytest

from app.database import get_connection
from core.classifier import classify_all_payments
from core.decision_engine import run_decision_engine_for_all
from core.recovery_model import train_recovery_model, update_recovery_probabilities
from core.review_queue import list_open_reviews, resolve_review_item, sync_review_queue
from data.generate_synthetic_data import GeneratorConfig, generate_dataset
from data.seed_database import seed_database


@pytest.fixture()
def decided_db(tmp_path):
    df = generate_dataset(GeneratorConfig(n_records=400, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")
    seed_database(str(csv_path), db_path, reset=True)

    conn = get_connection(db_path)
    classify_all_payments(conn)
    result = train_recovery_model(df)
    update_recovery_probabilities(conn, result.pipeline)
    run_decision_engine_for_all(conn)  # already calls sync_review_queue internally
    conn.close()
    return db_path


def test_run_decision_engine_populates_review_queue(decided_db: str) -> None:
    conn = get_connection(decided_db)
    try:
        review_count = conn.execute("SELECT COUNT(*) AS c FROM review_queue").fetchone()["c"]
        decision_review_count = conn.execute(
            "SELECT COUNT(*) AS c FROM recovery_decisions WHERE requires_human_review = 1"
        ).fetchone()["c"]
    finally:
        conn.close()
    assert review_count == decision_review_count
    assert review_count > 0


def test_sync_review_queue_is_idempotent(decided_db: str) -> None:
    conn = get_connection(decided_db)
    try:
        added_first = sync_review_queue(conn)  # already synced by fixture - should add 0 more
        count_after = conn.execute("SELECT COUNT(*) AS c FROM review_queue").fetchone()["c"]
        sync_review_queue(conn)
        count_after_again = conn.execute("SELECT COUNT(*) AS c FROM review_queue").fetchone()["c"]
    finally:
        conn.close()
    assert added_first == 0
    assert count_after == count_after_again


def test_list_open_reviews_shape(decided_db: str) -> None:
    conn = get_connection(decided_db)
    try:
        reviews = list_open_reviews(conn)
    finally:
        conn.close()
    assert len(reviews) > 0
    for item in reviews:
        assert set(item.keys()) >= {
            "id",
            "transaction_id",
            "escalation_reason",
            "amount_inr",
            "failure_category",
            "action",
        }


def test_list_open_reviews_sorted_by_amount_descending(decided_db: str) -> None:
    conn = get_connection(decided_db)
    try:
        reviews = list_open_reviews(conn)
    finally:
        conn.close()
    amounts = [r["amount_inr"] for r in reviews]
    assert amounts == sorted(amounts, reverse=True)


def test_resolve_review_item_updates_status_and_logs_audit(decided_db: str) -> None:
    conn = get_connection(decided_db)
    try:
        review = list_open_reviews(conn)[0]
        resolve_review_item(conn, review["id"], resolved_by="test_merchant", note="Approved manually.")

        status = conn.execute(
            "SELECT status FROM review_queue WHERE id = ?", (review["id"],)
        ).fetchone()["status"]
        audit_rows = conn.execute(
            "SELECT * FROM audit_logs WHERE transaction_id = ? AND event_type = 'review_resolved'",
            (review["transaction_id"],),
        ).fetchall()
    finally:
        conn.close()
    assert status == "resolved"
    assert len(audit_rows) == 1
    assert audit_rows[0]["actor"] == "test_merchant"


def test_resolved_item_no_longer_in_open_list(decided_db: str) -> None:
    conn = get_connection(decided_db)
    try:
        review = list_open_reviews(conn)[0]
        resolve_review_item(conn, review["id"])
        remaining_ids = [r["id"] for r in list_open_reviews(conn)]
    finally:
        conn.close()
    assert review["id"] not in remaining_ids


def test_resolve_nonexistent_review_raises() -> None:
    from app.database import init_db

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/empty.db"
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            with pytest.raises(ValueError):
                resolve_review_item(conn, 9999)
        finally:
            conn.close()
