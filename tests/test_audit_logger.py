import pytest

from app.database import get_connection, init_db
from core.audit_logger import get_audit_trail, log_event


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "test_payback.db")
    init_db(path)
    return path


def test_log_event_inserts_row(db_path: str) -> None:
    conn = get_connection(db_path)
    try:
        log_event(conn, "TXN000001", "simulator", "action_simulated:reminder", "test detail")
        rows = conn.execute("SELECT * FROM audit_logs").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["transaction_id"] == "TXN000001"
    assert rows[0]["actor"] == "simulator"


def test_get_audit_trail_returns_ordered_events(db_path: str) -> None:
    conn = get_connection(db_path)
    try:
        log_event(conn, "TXN000001", "decision_engine", "decision_made", "first")
        log_event(conn, "TXN000001", "simulator", "action_simulated:reminder", "second")
        log_event(conn, "TXN000002", "simulator", "action_simulated:reminder", "other transaction")
        trail = get_audit_trail(conn, "TXN000001")
    finally:
        conn.close()
    assert len(trail) == 2
    assert trail[0]["detail"] == "first"
    assert trail[1]["detail"] == "second"


def test_get_audit_trail_empty_for_unknown_transaction(db_path: str) -> None:
    conn = get_connection(db_path)
    try:
        trail = get_audit_trail(conn, "NO_SUCH_TXN")
    finally:
        conn.close()
    assert trail == []


def test_log_event_commit_false_requires_manual_commit(db_path: str) -> None:
    conn = get_connection(db_path)
    conn2 = get_connection(db_path)
    try:
        log_event(conn, "TXN000001", "simulator", "test", "detail", commit=False)
        # Not yet committed - a second connection should not see it.
        rows_before = conn2.execute("SELECT * FROM audit_logs").fetchall()
        conn.commit()
        rows_after = conn2.execute("SELECT * FROM audit_logs").fetchall()
    finally:
        conn.close()
        conn2.close()
    assert len(rows_before) == 0
    assert len(rows_after) == 1
