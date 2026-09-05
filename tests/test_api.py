import pytest
from fastapi.testclient import TestClient

from app.database import get_connection
from app.main import app
from core.baseline import run_baseline
from core.classifier import classify_all_payments
from core.decision_engine import run_decision_engine_for_all
from core.policy_engine import VALID_ACTIONS
from core.recovery_model import train_recovery_model, update_recovery_probabilities
from core.simulator import run_simulator_for_all
from data.generate_synthetic_data import GeneratorConfig, generate_dataset
from data.seed_database import seed_database


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    df = generate_dataset(GeneratorConfig(n_records=300, seed=42))
    csv_path = tmp_path / "synthetic_payments.csv"
    df.to_csv(csv_path, index=False)
    db_path = str(tmp_path / "test_payback.db")
    seed_database(str(csv_path), db_path, reset=True)

    conn = get_connection(db_path)
    run_baseline(conn)
    classify_all_payments(conn)
    result = train_recovery_model(df)
    update_recovery_probabilities(conn, result.pipeline)
    run_decision_engine_for_all(conn)
    run_simulator_for_all(conn)
    conn.close()

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    return TestClient(app)


def test_health_still_works(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "payback-ai"}


# --- /transactions ----------------------------------------------------


def test_list_transactions_returns_items(api_client: TestClient) -> None:
    response = api_client.get("/transactions", params={"limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5
    assert set(data[0].keys()) == {
        "transaction_id",
        "merchant_id",
        "amount_inr",
        "payment_method",
        "failure_category",
        "attempt_number",
        "event_timestamp",
    }


def test_list_transactions_filters_by_category(api_client: TestClient) -> None:
    response = api_client.get(
        "/transactions", params={"failure_category": "suspected_risk", "limit": 100}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert all(item["failure_category"] == "suspected_risk" for item in data)


def test_get_transaction_detail_success(api_client: TestClient) -> None:
    listing = api_client.get("/transactions", params={"limit": 1}).json()
    txn_id = listing[0]["transaction_id"]

    response = api_client.get(f"/transactions/{txn_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == txn_id
    assert data["decision"] is not None
    assert data["decision"]["action"] in VALID_ACTIONS
    assert data["payback_ai_action_status"] is not None


def test_get_transaction_detail_404_for_missing(api_client: TestClient) -> None:
    response = api_client.get("/transactions/TXN_DOES_NOT_EXIST")
    assert response.status_code == 404


# --- /decisions ---------------------------------------------------------


def test_get_decision_live_compute(api_client: TestClient) -> None:
    listing = api_client.get("/transactions", params={"limit": 1}).json()
    txn_id = listing[0]["transaction_id"]

    response = api_client.get(f"/decisions/{txn_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == txn_id
    assert data["action"] in VALID_ACTIONS
    assert isinstance(data["safety_checks"], list)


def test_get_decision_404_for_missing(api_client: TestClient) -> None:
    response = api_client.get("/decisions/TXN_DOES_NOT_EXIST")
    assert response.status_code == 404


def test_run_decision_persists_to_db(api_client: TestClient) -> None:
    listing = api_client.get("/transactions", params={"limit": 1}).json()
    txn_id = listing[0]["transaction_id"]

    response = api_client.post(f"/decisions/{txn_id}/run")
    assert response.status_code == 200
    data = response.json()
    assert data["action"] in VALID_ACTIONS

    # Confirm persisted via the transaction detail endpoint.
    detail = api_client.get(f"/transactions/{txn_id}").json()
    assert detail["decision"]["action"] == data["action"]


def test_run_decision_404_for_missing(api_client: TestClient) -> None:
    response = api_client.post("/decisions/TXN_DOES_NOT_EXIST/run")
    assert response.status_code == 404


# --- /metrics -------------------------------------------------------------


def test_baseline_metrics_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/metrics/baseline")
    assert response.status_code == 200
    data = response.json()
    assert "recovery_rate" in data
    assert 0.0 <= data["recovery_rate"] <= 1.0


def test_payback_ai_metrics_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/metrics/payback-ai")
    assert response.status_code == 200
    data = response.json()
    assert "recovery_rate" in data
    assert "status_distribution" in data


def test_comparison_metrics_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/metrics/comparison")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"baseline", "payback_ai"}
    assert "recovery_rate" in data["baseline"]
    assert "recovery_rate" in data["payback_ai"]


def test_recovery_model_metrics_endpoint_does_not_crash(api_client: TestClient) -> None:
    response = api_client.get("/metrics/recovery-model")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
