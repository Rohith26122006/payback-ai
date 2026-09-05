import numpy as np
import pytest

from app.database import get_connection
from core.classifier import classify_all_payments
from core.recovery_model import (
    FEATURE_COLUMNS,
    LEAKAGE_EXCLUDED_COLUMNS,
    build_feature_frame,
    load_model,
    predict_recovery_probability,
    save_model,
    train_recovery_model,
    update_recovery_probabilities,
)
from data.generate_synthetic_data import GeneratorConfig, generate_dataset
from data.seed_database import seed_database


@pytest.fixture()
def payments_df():
    return generate_dataset(GeneratorConfig(n_records=1200, seed=42))


@pytest.fixture()
def seeded_db(tmp_path, payments_df):
    csv_path = tmp_path / "synthetic_payments.csv"
    payments_df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")
    seed_database(str(csv_path), db_path, reset=True)
    return db_path


def test_no_leakage_columns_in_feature_set() -> None:
    for col in LEAKAGE_EXCLUDED_COLUMNS:
        assert col not in FEATURE_COLUMNS


def test_build_feature_frame_shapes(payments_df) -> None:
    X, y = build_feature_frame(payments_df)
    assert len(X) == len(payments_df)
    assert len(y) == len(payments_df)
    assert set(y.unique()) <= {0, 1}
    assert list(X.columns) == FEATURE_COLUMNS


def test_train_recovery_model_returns_expected_metric_keys(payments_df) -> None:
    result = train_recovery_model(payments_df)
    expected_keys = {
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
        "train_size",
        "test_size",
        "random_seed",
        "model_version",
        "data_note",
    }
    assert expected_keys.issubset(result.metrics.keys())


def test_metrics_are_within_valid_ranges(payments_df) -> None:
    result = train_recovery_model(payments_df)
    m = result.metrics
    for key in ("precision", "recall", "f1_score", "roc_auc"):
        assert 0.0 <= m[key] <= 1.0
    assert m["train_size"] + m["test_size"] == len(payments_df)


def test_train_test_split_is_reproducible(payments_df) -> None:
    result1 = train_recovery_model(payments_df)
    result2 = train_recovery_model(payments_df)
    assert result1.metrics["precision"] == result2.metrics["precision"]
    assert result1.metrics["roc_auc"] == result2.metrics["roc_auc"]
    assert result1.metrics["confusion_matrix"] == result2.metrics["confusion_matrix"]


def test_model_beats_random_guessing(payments_df) -> None:
    """Sanity check: the model should do meaningfully better than a coin flip."""
    result = train_recovery_model(payments_df)
    assert result.metrics["roc_auc"] > 0.6


def test_save_and_reload_model_gives_identical_predictions(tmp_path, payments_df) -> None:
    result = train_recovery_model(payments_df)
    model_path = str(tmp_path / "model.joblib")
    save_model(result.pipeline, model_path)
    reloaded = load_model(model_path)

    sample = payments_df.head(20)
    original_preds = predict_recovery_probability(result.pipeline, sample)
    reloaded_preds = predict_recovery_probability(reloaded, sample)
    assert np.allclose(original_preds.values, reloaded_preds.values)


def test_predict_recovery_probability_in_valid_range(payments_df) -> None:
    result = train_recovery_model(payments_df)
    probs = predict_recovery_probability(result.pipeline, payments_df.head(50))
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_update_recovery_probabilities_populates_db(seeded_db: str, payments_df) -> None:
    conn = get_connection(seeded_db)
    try:
        classify_all_payments(conn)  # Stage 5 must run first
        result = train_recovery_model(payments_df)
        n_updated = update_recovery_probabilities(conn, result.pipeline)

        rows = conn.execute(
            "SELECT recovery_probability FROM model_predictions"
        ).fetchall()
    finally:
        conn.close()

    assert n_updated == len(payments_df)
    assert all(r["recovery_probability"] is not None for r in rows)
    assert all(0.0 <= r["recovery_probability"] <= 1.0 for r in rows)
