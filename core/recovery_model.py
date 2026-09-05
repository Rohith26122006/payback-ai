"""
PayBack AI - Recovery probability model.

Predicts P(recovery) for a failed payment using ONLY features knowable
before the outcome. Logistic regression is used as the interpretable
first model, per project instructions (gradient boosting only if this
proves insufficient later - it hasn't been needed yet).

LEAKAGE BOUNDARY (explicit, code-enforced):
  Features used:   failure_category, payment_method, merchant_category,
                    amount_inr, attempt_number, customer_success_rate,
                    previous_failed_attempts, time_since_failure_minutes,
                    subscription_flag, device_change_flag
  Target:           recovery_outcome (binarized: recovered=1, else 0)
  NEVER features:   recovery_outcome, recovered_amount (the label itself),
                    transaction_id, customer_id_hash, merchant_id (identifiers),
                    failure_code (redundant with failure_category),
                    event_timestamp (not used to avoid encoding leakage
                    from time-of-generation artifacts)

IMPORTANT: this model is trained and evaluated entirely on SYNTHETIC data.
Reported metrics describe how well the model fits the synthetic generator's
assumptions, NOT real-world payment recovery behavior. Do not treat these
numbers as validated real-world performance.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ["failure_category", "payment_method", "merchant_category"]
NUMERIC_FEATURES = [
    "amount_inr",
    "attempt_number",
    "customer_success_rate",
    "previous_failed_attempts",
    "time_since_failure_minutes",
]
BOOLEAN_FEATURES = ["subscription_flag", "device_change_flag"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOLEAN_FEATURES

# Columns that must NEVER be used as features - kept explicit for tests/audits.
LEAKAGE_EXCLUDED_COLUMNS = [
    "recovery_outcome",
    "recovered_amount",
    "transaction_id",
    "customer_id_hash",
    "merchant_id",
    "failure_code",
    "event_timestamp",
]

TARGET_COLUMN = "recovery_outcome"
RANDOM_SEED = 42
MODEL_VERSION = "recovery-logreg-v1"

DEFAULT_MODEL_PATH = "models/recovery_model.joblib"
DEFAULT_METRICS_PATH = "models/recovery_model_metrics.json"


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract leakage-free features (X) and binary target (y) from raw payments."""
    X = df[FEATURE_COLUMNS].copy()
    X["subscription_flag"] = X["subscription_flag"].astype(int)
    X["device_change_flag"] = X["device_change_flag"].astype(int)
    y = (df[TARGET_COLUMN] == "recovered").astype(int)
    return X, y


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES + BOOLEAN_FEATURES),
        ]
    )
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    metrics: dict


def train_recovery_model(df: pd.DataFrame) -> TrainingResult:
    """Train + evaluate on a train/test split with a fixed seed."""
    X, y = build_feature_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "random_seed": RANDOM_SEED,
        "model_version": MODEL_VERSION,
        "data_note": (
            "Trained and evaluated on SYNTHETIC data only. Metrics reflect fit to "
            "the synthetic generator's assumptions, not validated real-world "
            "payment-recovery performance."
        ),
    }
    return TrainingResult(pipeline=pipeline, metrics=metrics)


def save_model(pipeline: Pipeline, model_path: str = DEFAULT_MODEL_PATH) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    return joblib.load(model_path)


def save_metrics(metrics: dict, metrics_path: str = DEFAULT_METRICS_PATH) -> None:
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def predict_recovery_probability(pipeline: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Predict P(recovery) for arbitrary payment rows (must contain FEATURE_COLUMNS)."""
    X, _ = build_feature_frame_for_inference(df)
    return pd.Series(pipeline.predict_proba(X)[:, 1], index=df.index)


def build_feature_frame_for_inference(df: pd.DataFrame) -> tuple[pd.DataFrame, None]:
    """Like build_feature_frame but tolerant of a missing target column (real inference use)."""
    X = df[FEATURE_COLUMNS].copy()
    X["subscription_flag"] = X["subscription_flag"].astype(int)
    X["device_change_flag"] = X["device_change_flag"].astype(int)
    return X, None


def update_recovery_probabilities(conn: sqlite3.Connection, pipeline: Pipeline) -> int:
    """
    Score every payment and UPDATE the recovery_probability column on its
    existing model_predictions row (written by classify_all_payments in
    Stage 5). Rows with no existing prediction (classifier not yet run)
    are skipped and reported.
    """
    payments = pd.read_sql_query("SELECT * FROM payments", conn)
    probabilities = predict_recovery_probability(pipeline, payments)

    updated = 0
    for idx, row in payments.iterrows():
        cur = conn.execute(
            "UPDATE model_predictions SET recovery_probability = ? WHERE transaction_id = ?",
            (float(probabilities[idx]), row["transaction_id"]),
        )
        updated += cur.rowcount
    conn.commit()
    return updated


if __name__ == "__main__":
    from app.database import get_connection

    conn = get_connection()
    payments_df = pd.read_sql_query("SELECT * FROM payments", conn)

    result = train_recovery_model(payments_df)
    save_model(result.pipeline)
    save_metrics(result.metrics)
    n_updated = update_recovery_probabilities(conn, result.pipeline)
    conn.close()

    print(f"[RECOVERY MODEL] Trained on SYNTHETIC data ({MODEL_VERSION}):")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    print(f"[RECOVERY MODEL] Updated recovery_probability for {n_updated} payments")
    print(f"[RECOVERY MODEL] Saved model to {DEFAULT_MODEL_PATH}")
