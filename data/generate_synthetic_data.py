"""
PayBack AI - Synthetic failed-payment data generator.

Generates a reproducible, clearly-labeled SYNTHETIC dataset of failed payment
events for the AI Revenue Recovery pipeline. No real customer, card, bank,
or payment data is used anywhere in this file.

IMPORTANT (leakage note):
  `recovery_outcome` and `recovered_amount` are GROUND-TRUTH LABEL fields,
  simulated here to make the dataset useful for training/evaluation. They
  must NEVER be used as input features when training the recovery
  probability model (Stage 6) - only as the prediction target / evaluation
  reference. This will be enforced again explicitly in Stage 6.

Usage:
    python -m data.generate_synthetic_data --n 1200 --seed 42 \
        --output data/synthetic_payments.csv
"""
from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

# ---------------------------------------------------------------------------
# Fixed reference data (deterministic, documented, not arbitrary)
# ---------------------------------------------------------------------------

FAILURE_CODE_TO_CATEGORY: dict[str, str] = {
    "BANK_TIMEOUT": "temporary_bank_or_network",
    "NETWORK_ERROR": "temporary_bank_or_network",
    "GATEWAY_TIMEOUT": "temporary_bank_or_network",
    "INSUFFICIENT_BALANCE": "insufficient_funds",
    "CARD_EXPIRED": "expired_or_invalid_method",
    "INVALID_UPI_ID": "expired_or_invalid_method",
    "INVALID_CARD": "expired_or_invalid_method",
    "OTP_FAILED": "authentication_failure",
    "3DS_FAILED": "authentication_failure",
    "RISK_BLOCKED": "suspected_risk",
    "FRAUD_SUSPECTED": "suspected_risk",
    "USER_CANCELLED": "customer_abandonment",
    "TIMEOUT_ABANDONED": "customer_abandonment",
    "MANDATE_FAILED": "subscription_failure",
    "AUTO_DEBIT_FAILED": "subscription_failure",
    "UNKNOWN_ERROR": "unknown",
    "OTHER": "unknown",
}

# Relative frequency weights for each failure code (must stay realistic:
# temporary/insufficient-funds dominate, risk/unknown are rarer).
FAILURE_CODE_WEIGHTS: dict[str, float] = {
    "BANK_TIMEOUT": 14.0,
    "NETWORK_ERROR": 10.0,
    "GATEWAY_TIMEOUT": 8.0,
    "INSUFFICIENT_BALANCE": 20.0,
    "CARD_EXPIRED": 8.0,
    "INVALID_UPI_ID": 5.0,
    "INVALID_CARD": 5.0,
    "OTP_FAILED": 9.0,
    "3DS_FAILED": 4.0,
    "RISK_BLOCKED": 3.0,
    "FRAUD_SUSPECTED": 1.5,
    "USER_CANCELLED": 6.0,
    "TIMEOUT_ABANDONED": 3.0,
    "MANDATE_FAILED": 2.0,
    "AUTO_DEBIT_FAILED": 1.5,
    "UNKNOWN_ERROR": 2.0,
    "OTHER": 1.0,
}

PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet", "emi"]
PAYMENT_METHOD_WEIGHTS = [30, 40, 15, 10, 5]

MERCHANT_CATEGORIES = [
    "e_commerce",
    "subscription",
    "travel",
    "food_delivery",
    "education",
    "utilities",
]

# Base recovery probability per failure category (before attempt/history
# adjustments). These are ASSUMPTIONS for synthetic-data realism, not
# real Razorpay recovery statistics.
BASE_RECOVERY_PROBABILITY: dict[str, float] = {
    "temporary_bank_or_network": 0.70,
    "insufficient_funds": 0.45,
    "expired_or_invalid_method": 0.35,
    "authentication_failure": 0.50,
    "suspected_risk": 0.03,
    "customer_abandonment": 0.20,
    "subscription_failure": 0.40,
    "unknown": 0.15,
}

NUM_MERCHANTS = 30
NUM_CUSTOMERS = 600


@dataclass(frozen=True)
class GeneratorConfig:
    n_records: int = 1200
    seed: int = 42
    unknown_failure_rate: float = 0.02  # forced minimum share of "unknown"


def _make_customer_hash(customer_index: int) -> str:
    """Deterministic, non-reversible synthetic customer identifier."""
    raw = f"synthetic-customer-{customer_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _pick_failure_code(rng: random.Random) -> str:
    codes = list(FAILURE_CODE_WEIGHTS.keys())
    weights = list(FAILURE_CODE_WEIGHTS.values())
    return rng.choices(codes, weights=weights, k=1)[0]


def _simulate_recovery(
    rng: random.Random,
    failure_category: str,
    attempt_number: int,
    customer_success_rate: float,
    previous_failed_attempts: int,
    amount_inr: float,
) -> tuple[str, float]:
    """
    Simulate the ground-truth recovery outcome for this synthetic event.
    Returns (recovery_outcome, recovered_amount).
    recovery_outcome in {"recovered", "not_recovered"}.
    """
    prob = BASE_RECOVERY_PROBABILITY[failure_category]

    # Higher customer success rate -> more likely to recover.
    prob += (customer_success_rate - 0.5) * 0.3

    # Repeated attempts / prior failures reduce recovery odds (fatigue).
    prob -= 0.08 * max(0, attempt_number - 1)
    prob -= 0.03 * min(previous_failed_attempts, 5)

    # Very high value transactions recover slightly less often (more friction,
    # more manual review) - deliberate edge-case correlation.
    if amount_inr > 50000:
        prob -= 0.10

    prob = min(max(prob, 0.0), 0.97)
    recovered = rng.random() < prob

    if recovered:
        # Recovered amount equals the original failed amount (full recovery).
        return "recovered", round(amount_inr, 2)
    return "not_recovered", 0.0


def generate_dataset(config: GeneratorConfig) -> pd.DataFrame:
    rng = random.Random(config.seed)

    merchant_ids = [f"M{str(i).zfill(3)}" for i in range(1, NUM_MERCHANTS + 1)]
    merchant_category_map = {
        m: rng.choice(MERCHANT_CATEGORIES) for m in merchant_ids
    }

    now = datetime(2026, 8, 1)  # fixed reference date for reproducibility
    rows: list[dict] = []

    n_unknown_forced = int(config.n_records * config.unknown_failure_rate)

    for i in range(config.n_records):
        merchant_id = rng.choice(merchant_ids)
        merchant_category = merchant_category_map[merchant_id]

        customer_index = rng.randint(0, NUM_CUSTOMERS - 1)
        customer_id_hash = _make_customer_hash(customer_index)

        # Force a minimum share of unknown failures deliberately (edge case),
        # otherwise sample from the weighted distribution.
        if i < n_unknown_forced:
            failure_code = rng.choice(["UNKNOWN_ERROR", "OTHER"])
        else:
            failure_code = _pick_failure_code(rng)
        failure_category = FAILURE_CODE_TO_CATEGORY[failure_code]

        subscription_flag = (
            merchant_category == "subscription" and rng.random() < 0.85
        ) or (failure_category == "subscription_failure" and rng.random() < 0.9)

        attempt_number = rng.choices([1, 2, 3, 4], weights=[55, 25, 13, 7], k=1)[0]
        previous_failed_attempts = max(0, attempt_number - 1 + rng.choice([0, 0, 1]))

        # customer_success_rate: beta-like skew via two random draws (no numpy dependency).
        customer_success_rate = round((rng.random() + rng.random()) / 2, 3)

        time_since_failure_minutes = rng.randint(1, 1440)

        # device_change_flag correlates with suspected_risk (deliberate edge case).
        if failure_category == "suspected_risk":
            device_change_flag = rng.random() < 0.75
        else:
            device_change_flag = rng.random() < 0.08

        # amount distribution varies by merchant category, with occasional outliers.
        base_amount = {
            "e_commerce": 1800,
            "subscription": 600,
            "travel": 9000,
            "food_delivery": 450,
            "education": 15000,
            "utilities": 1200,
        }[merchant_category]
        amount_inr = round(max(50, rng.gauss(base_amount, base_amount * 0.5)), 2)
        if rng.random() < 0.02:  # deliberate high-value outlier edge case
            amount_inr = round(amount_inr * rng.uniform(5, 12), 2)

        event_timestamp = now - timedelta(
            days=rng.randint(0, 89), minutes=rng.randint(0, 1439)
        )

        recovery_outcome, recovered_amount = _simulate_recovery(
            rng,
            failure_category,
            attempt_number,
            customer_success_rate,
            previous_failed_attempts,
            amount_inr,
        )

        rows.append(
            {
                "transaction_id": f"TXN{str(i + 1).zfill(6)}",
                "merchant_id": merchant_id,
                "customer_id_hash": customer_id_hash,
                "amount_inr": amount_inr,
                "payment_method": rng.choices(
                    PAYMENT_METHODS, weights=PAYMENT_METHOD_WEIGHTS, k=1
                )[0],
                "failure_code": failure_code,
                "failure_category": failure_category,
                "attempt_number": attempt_number,
                "customer_success_rate": customer_success_rate,
                "previous_failed_attempts": previous_failed_attempts,
                "time_since_failure_minutes": time_since_failure_minutes,
                "subscription_flag": subscription_flag,
                "merchant_category": merchant_category,
                "device_change_flag": device_change_flag,
                "recovery_outcome": recovery_outcome,
                "recovered_amount": recovered_amount,
                "event_timestamp": event_timestamp.isoformat(),
            }
        )

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic PayBack AI dataset.")
    parser.add_argument("--n", type=int, default=1200, help="Number of records (>=1000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_payments.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    config = GeneratorConfig(n_records=args.n, seed=args.seed)
    df = generate_dataset(config)
    df.to_csv(args.output, index=False)
    print(f"[SYNTHETIC DATA] Wrote {len(df)} records to {args.output} (seed={args.seed})")


if __name__ == "__main__":
    main()
