import json

import pytest

from app.database import get_connection
from core.classifier import classify_all_payments
from core.decision_engine import DecisionOutput, make_decision, run_decision_engine_for_all
from core.recovery_model import train_recovery_model, update_recovery_probabilities
from data.generate_synthetic_data import GeneratorConfig, generate_dataset
from data.seed_database import seed_database
from llm.message_generator import (
    DecisionMessageOutput,
    deterministic_fallback_message,
    generate_message,
    generate_message_for_transaction,
    select_default_provider,
)
from llm.mock_provider import MockLLMProvider
from llm.provider import LLMProvider, LLMProviderError


def sample_decision(**overrides) -> DecisionOutput:
    defaults = dict(
        transaction_id="TXN000001",
        action="delayed_retry",
        recovery_probability=0.7,
        confidence=0.95,
        reason="Temporary failure; retrying.",
        safety_checks=[],
        requires_human_review=False,
    )
    defaults.update(overrides)
    return DecisionOutput(**defaults)


# --- mock provider works with zero API key ---------------------------------


def test_mock_provider_returns_valid_json() -> None:
    provider = MockLLMProvider()
    raw = provider.generate("system", "action: delayed_retry\nsomething")
    parsed = json.loads(raw)
    assert "customer_message" in parsed
    assert isinstance(parsed["customer_message"], str)


def test_select_default_provider_with_no_api_key_returns_mock(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider, name = select_default_provider()
    assert name == "mock"
    assert isinstance(provider, MockLLMProvider)


# --- happy path --------------------------------------------------------


def test_generate_message_happy_path_uses_llm_output() -> None:
    decision = sample_decision(action="delayed_retry")
    result = generate_message(decision, provider=MockLLMProvider())
    assert result.used_fallback is False
    assert result.output.customer_message != ""
    assert result.output.action == "delayed_retry"
    assert result.output.confidence == decision.confidence
    assert result.output.requires_human_review == decision.requires_human_review


def test_no_action_produces_empty_message_without_calling_provider() -> None:
    decision = sample_decision(action="no_action")
    # always_fail provider would raise if called - proves no_action skips the LLM entirely.
    result = generate_message(decision, provider=MockLLMProvider(always_fail=True))
    assert result.output.customer_message == ""
    assert result.used_fallback is False


def test_output_matches_required_json_shape() -> None:
    decision = sample_decision()
    result = generate_message(decision, provider=MockLLMProvider())
    d = result.output.to_dict()
    assert set(d.keys()) == {"action", "reason", "customer_message", "confidence", "requires_human_review"}
    json.dumps(d)  # must be JSON-serializable


# --- fallback triggers ------------------------------------------------


def test_provider_failure_triggers_fallback() -> None:
    decision = sample_decision(action="reminder")
    result = generate_message(decision, provider=MockLLMProvider(always_fail=True))
    assert result.used_fallback is True
    assert result.output.customer_message == deterministic_fallback_message("reminder")


def test_invalid_json_triggers_fallback() -> None:
    decision = sample_decision(action="alternate_payment_link")
    result = generate_message(decision, provider=MockLLMProvider(return_invalid_json=True))
    assert result.used_fallback is True
    assert result.output.customer_message == deterministic_fallback_message("alternate_payment_link")


class ForbiddenContentProvider(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str, timeout_seconds: float = 10.0) -> str:
        return json.dumps({"customer_message": "Enjoy a 20% discount as an apology!", "llm_confidence": 0.9})


def test_forbidden_content_triggers_fallback() -> None:
    decision = sample_decision(action="reminder")
    result = generate_message(decision, provider=ForbiddenContentProvider())
    assert result.used_fallback is True
    assert "discount" not in result.output.customer_message.lower()


class FalseSuccessClaimProvider(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str, timeout_seconds: float = 10.0) -> str:
        return json.dumps({"customer_message": "Your payment was successful!", "llm_confidence": 0.9})


def test_false_success_claim_triggers_fallback() -> None:
    decision = sample_decision(action="delayed_retry")
    result = generate_message(decision, provider=FalseSuccessClaimProvider())
    assert result.used_fallback is True


class MissingFieldProvider(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str, timeout_seconds: float = 10.0) -> str:
        return json.dumps({"not_the_right_field": "oops"})


def test_missing_required_field_triggers_fallback() -> None:
    decision = sample_decision(action="human_escalation")
    result = generate_message(decision, provider=MissingFieldProvider())
    assert result.used_fallback is True


class HangingProvider(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str, timeout_seconds: float = 10.0) -> str:
        import time

        time.sleep(5)
        return json.dumps({"customer_message": "too slow", "llm_confidence": 0.9})


def test_timeout_triggers_fallback() -> None:
    decision = sample_decision(action="reminder")
    result = generate_message(decision, provider=HangingProvider(), timeout_seconds=0.2)
    assert result.used_fallback is True
    assert result.output.customer_message == deterministic_fallback_message("reminder")


def test_llm_never_changes_action_or_confidence_even_on_success() -> None:
    decision = sample_decision(action="alternate_payment_link", confidence=0.42, requires_human_review=True)
    result = generate_message(decision, provider=MockLLMProvider())
    assert result.output.action == "alternate_payment_link"
    assert result.output.confidence == 0.42
    assert result.output.requires_human_review is True


# --- DB integration ------------------------------------------------------


@pytest.fixture()
def fully_decided_db(tmp_path):
    df = generate_dataset(GeneratorConfig(n_records=200, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")
    seed_database(str(csv_path), db_path, reset=True)

    conn = get_connection(db_path)
    classify_all_payments(conn)
    result = train_recovery_model(df)
    update_recovery_probabilities(conn, result.pipeline)
    run_decision_engine_for_all(conn)
    conn.close()
    return db_path


def test_generate_message_for_transaction_logs_audit_entry(fully_decided_db: str) -> None:
    conn = get_connection(fully_decided_db)
    try:
        row = conn.execute("SELECT transaction_id FROM payments LIMIT 1").fetchone()
        generate_message_for_transaction(conn, row["transaction_id"], provider=MockLLMProvider())
        audit_rows = conn.execute(
            "SELECT * FROM audit_logs WHERE transaction_id = ? AND actor = 'message_generator'",
            (row["transaction_id"],),
        ).fetchall()
    finally:
        conn.close()
    assert len(audit_rows) == 1


def test_generate_message_for_transaction_works_without_api_key(fully_decided_db: str, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    conn = get_connection(fully_decided_db)
    try:
        row = conn.execute("SELECT transaction_id FROM payments LIMIT 1").fetchone()
        result = generate_message_for_transaction(conn, row["transaction_id"])
    finally:
        conn.close()
    assert result.provider_name == "mock"
    assert isinstance(result.output, DecisionMessageOutput)
