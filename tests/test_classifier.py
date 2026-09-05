import pytest

from app.database import get_connection
from core.classifier import (
    CONFIDENCE_KNOWN_BUT_UNKNOWN_CATEGORY,
    CONFIDENCE_KNOWN_CATEGORY,
    CONFIDENCE_UNRECOGNIZED_CODE,
    MODEL_VERSION,
    classify_all_payments,
    classify_failure,
)
from data.generate_synthetic_data import FAILURE_CODE_TO_CATEGORY, GeneratorConfig, generate_dataset
from data.seed_database import seed_database


@pytest.fixture()
def seeded_db(tmp_path):
    df = generate_dataset(GeneratorConfig(n_records=400, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")
    seed_database(str(csv_path), db_path, reset=True)
    return db_path


def test_all_known_codes_map_to_correct_category() -> None:
    for code, expected_category in FAILURE_CODE_TO_CATEGORY.items():
        result = classify_failure(code)
        assert result.category == expected_category


def test_known_definite_category_has_high_confidence() -> None:
    result = classify_failure("BANK_TIMEOUT")
    assert result.category == "temporary_bank_or_network"
    assert result.confidence == CONFIDENCE_KNOWN_CATEGORY


def test_known_unknown_category_has_reduced_confidence() -> None:
    result = classify_failure("UNKNOWN_ERROR")
    assert result.category == "unknown"
    assert result.confidence == CONFIDENCE_KNOWN_BUT_UNKNOWN_CATEGORY


def test_unrecognized_code_falls_back_safely() -> None:
    result = classify_failure("SOME_BRAND_NEW_CODE_NOBODY_HAS_SEEN")
    assert result.category == "unknown"
    assert result.confidence == CONFIDENCE_UNRECOGNIZED_CODE
    assert "not recognized" in result.reason


def test_classify_never_raises_on_empty_or_none() -> None:
    assert classify_failure("").category == "unknown"
    assert classify_failure(None).category == "unknown"  # type: ignore[arg-type]


def test_confidence_always_in_valid_range() -> None:
    codes = list(FAILURE_CODE_TO_CATEGORY.keys()) + ["UNSEEN_CODE"]
    for code in codes:
        result = classify_failure(code)
        assert 0.0 <= result.confidence <= 1.0


def test_reason_is_nonempty_string() -> None:
    for code in list(FAILURE_CODE_TO_CATEGORY.keys()) + ["UNSEEN_CODE"]:
        result = classify_failure(code)
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0


def test_classify_all_payments_covers_every_row(seeded_db: str) -> None:
    conn = get_connection(seeded_db)
    try:
        n = classify_all_payments(conn)
        total_payments = conn.execute("SELECT COUNT(*) AS c FROM payments").fetchone()["c"]
        prediction_count = conn.execute(
            "SELECT COUNT(*) AS c FROM model_predictions WHERE model_version = ?",
            (MODEL_VERSION,),
        ).fetchone()["c"]
    finally:
        conn.close()
    assert n == total_payments
    assert prediction_count == total_payments


def test_classify_all_payments_matches_ground_truth_category(seeded_db: str) -> None:
    conn = get_connection(seeded_db)
    try:
        classify_all_payments(conn)
        rows = conn.execute(
            """
            SELECT p.failure_category AS ground_truth, mp.classified_category AS predicted
            FROM payments p
            JOIN model_predictions mp ON mp.transaction_id = p.transaction_id
            WHERE mp.model_version = ?
            """,
            (MODEL_VERSION,),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) > 0
    assert all(r["ground_truth"] == r["predicted"] for r in rows)


def test_classify_all_payments_is_idempotent(seeded_db: str) -> None:
    conn = get_connection(seeded_db)
    try:
        classify_all_payments(conn)
        classify_all_payments(conn)  # re-run
        prediction_count = conn.execute(
            "SELECT COUNT(*) AS c FROM model_predictions WHERE model_version = ?",
            (MODEL_VERSION,),
        ).fetchone()["c"]
        total_payments = conn.execute("SELECT COUNT(*) AS c FROM payments").fetchone()["c"]
    finally:
        conn.close()
    assert prediction_count == total_payments
