from data.generate_synthetic_data import GeneratorConfig, generate_dataset

REQUIRED_COLUMNS = {
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
}


def test_generates_minimum_required_records() -> None:
    df = generate_dataset(GeneratorConfig(n_records=1200, seed=42))
    assert len(df) >= 1000


def test_has_all_required_columns() -> None:
    df = generate_dataset(GeneratorConfig(n_records=200, seed=42))
    assert REQUIRED_COLUMNS.issubset(set(df.columns))


def test_transaction_ids_are_unique() -> None:
    df = generate_dataset(GeneratorConfig(n_records=1200, seed=42))
    assert df["transaction_id"].is_unique


def test_reproducible_with_same_seed() -> None:
    df1 = generate_dataset(GeneratorConfig(n_records=300, seed=7))
    df2 = generate_dataset(GeneratorConfig(n_records=300, seed=7))
    assert df1.equals(df2)


def test_different_seed_gives_different_data() -> None:
    df1 = generate_dataset(GeneratorConfig(n_records=300, seed=1))
    df2 = generate_dataset(GeneratorConfig(n_records=300, seed=2))
    assert not df1.equals(df2)


def test_failure_code_maps_to_correct_category() -> None:
    from data.generate_synthetic_data import FAILURE_CODE_TO_CATEGORY

    df = generate_dataset(GeneratorConfig(n_records=800, seed=42))
    for _, row in df.iterrows():
        assert FAILURE_CODE_TO_CATEGORY[row["failure_code"]] == row["failure_category"]


def test_recovered_amount_only_positive_when_recovered() -> None:
    df = generate_dataset(GeneratorConfig(n_records=800, seed=42))
    recovered = df[df["recovery_outcome"] == "recovered"]
    not_recovered = df[df["recovery_outcome"] == "not_recovered"]
    assert (recovered["recovered_amount"] > 0).all()
    assert (not_recovered["recovered_amount"] == 0).all()


def test_amounts_are_positive() -> None:
    df = generate_dataset(GeneratorConfig(n_records=800, seed=42))
    assert (df["amount_inr"] > 0).all()


def test_contains_unknown_failure_edge_case() -> None:
    df = generate_dataset(GeneratorConfig(n_records=800, seed=42))
    assert (df["failure_category"] == "unknown").any()


def test_contains_suspected_risk_edge_case() -> None:
    df = generate_dataset(GeneratorConfig(n_records=800, seed=42))
    assert (df["failure_category"] == "suspected_risk").any()
