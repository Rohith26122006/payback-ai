"""
PayBack AI - Failure classifier.

Deterministic, rule-based classification of failure_code -> failure_category.

WHY NO ML HERE: failure_code -> failure_category is a fixed, exhaustive
mapping (defined once in data/generate_synthetic_data.py as the ground
truth used to generate the synthetic dataset). An ML model trained to
reproduce a dictionary lookup would add complexity without improving
accuracy or explainability - so this stays rule-based, per the project's
"add ML only where it improves the prototype" instruction. A real ML
classifier is used downstream in Stage 6, where the target (recovery
probability) is genuinely not deterministic from the input features.

This module reuses FAILURE_CODE_TO_CATEGORY from the data generator so the
mapping is defined in exactly one place.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from data.generate_synthetic_data import FAILURE_CODE_TO_CATEGORY

MODEL_VERSION = "rule-based-classifier-v1"

# Confidence levels are ASSUMPTIONS about certainty, not learned values:
#   - a code with a clearly-known category: high confidence.
#   - a code that itself means "unknown reason" (UNKNOWN_ERROR/OTHER):
#     the category assignment is deterministic, but the underlying cause
#     is genuinely unclear, so confidence is deliberately lower.
#   - a code the classifier has never seen at all: safe fallback to
#     "unknown" with the lowest confidence, flagged for review.
CONFIDENCE_KNOWN_CATEGORY = 0.95
CONFIDENCE_KNOWN_BUT_UNKNOWN_CATEGORY = 0.40
CONFIDENCE_UNRECOGNIZED_CODE = 0.15


@dataclass(frozen=True)
class ClassificationResult:
    category: str
    confidence: float
    reason: str


def classify_failure(failure_code: str) -> ClassificationResult:
    """Classify a single failure_code. Never raises - always returns a result."""
    normalized = (failure_code or "").strip().upper()

    if normalized in FAILURE_CODE_TO_CATEGORY:
        category = FAILURE_CODE_TO_CATEGORY[normalized]
        if category == "unknown":
            return ClassificationResult(
                category=category,
                confidence=CONFIDENCE_KNOWN_BUT_UNKNOWN_CATEGORY,
                reason=(
                    f"failure_code '{normalized}' is a recognized code that itself "
                    f"indicates no specific reason was reported by the gateway; "
                    f"classified as 'unknown' with reduced confidence."
                ),
            )
        return ClassificationResult(
            category=category,
            confidence=CONFIDENCE_KNOWN_CATEGORY,
            reason=(
                f"failure_code '{normalized}' matches the known mapping to "
                f"category '{category}'."
            ),
        )

    return ClassificationResult(
        category="unknown",
        confidence=CONFIDENCE_UNRECOGNIZED_CODE,
        reason=(
            f"failure_code '{failure_code}' was not recognized by the classifier "
            f"(not present in the known mapping); defaulting to 'unknown' as a "
            f"safe fallback pending manual review."
        ),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_all_payments(conn: sqlite3.Connection) -> int:
    """
    Classify every row in `payments` and write results to `model_predictions`
    (recovery_probability left NULL - populated later by Stage 6).

    Idempotent: clears prior rows for this model_version before inserting.
    Returns the number of rows written.
    """
    conn.execute(
        "DELETE FROM model_predictions WHERE model_version = ?", (MODEL_VERSION,)
    )
    conn.commit()

    rows = conn.execute("SELECT transaction_id, failure_code FROM payments").fetchall()
    predicted_at = _now_iso()

    for row in rows:
        result = classify_failure(row["failure_code"])
        conn.execute(
            "INSERT INTO model_predictions "
            "(transaction_id, classified_category, classification_confidence, "
            "classification_reason, recovery_probability, model_version, predicted_at) "
            "VALUES (?, ?, ?, ?, NULL, ?, ?)",
            (
                row["transaction_id"],
                result.category,
                result.confidence,
                result.reason,
                MODEL_VERSION,
                predicted_at,
            ),
        )

    conn.commit()
    return len(rows)


if __name__ == "__main__":
    from app.database import get_connection

    conn = get_connection()
    n = classify_all_payments(conn)
    conn.close()
    print(f"[CLASSIFIER] Classified {n} payments using {MODEL_VERSION}")
