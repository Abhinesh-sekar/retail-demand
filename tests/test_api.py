"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "seasonal_naive" in data
    assert "lightgbm" in data
    assert "relative_wape_reduction_pct" in data
    assert data["lightgbm"]["wape"] < data["seasonal_naive"]["wape"]
    assert data["relative_wape_reduction_pct"] > 0


def test_predict_endpoint(client):
    payload = {
        "item_id": "FOODS_1_001",
        "store_id": "CA_1",
        "horizon": 7,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["item_id"] == "FOODS_1_001"
    assert data["store_id"] == "CA_1"
    assert len(data["forecasts"]) == 7
    for item in data["forecasts"]:
        assert "date" in item
        assert item["predicted_demand"] >= 0.0


def test_predict_horizon_validation(client):
    # Horizon greater than 28 days should fail Pydantic validation
    payload = {
        "item_id": "FOODS_1_001",
        "store_id": "CA_1",
        "horizon": 50,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_detect_anomalies_endpoint(client):
    records = [
        {"date": f"2024-01-{i:02d}", "item_id": "SKU1", "store_id": "S1", "actual": 10.0, "forecast": 10.0}
        for i in range(1, 15)
    ]
    # Add a surge anomaly
    records.append({"date": "2024-01-15", "item_id": "SKU1", "store_id": "S1", "actual": 85.0, "forecast": 10.0})

    payload = {
        "records": records,
        "z_threshold": 3.0,
    }
    response = client.post("/api/v1/detect-anomalies", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_evaluated"] == 15
    assert data["anomalies_flagged"] >= 1
    flagged = data["anomalies"][0]
    assert flagged["anomaly_type"] == "DEMAND_SPIKE"
    assert "replenishment" in flagged["recommended_action"].lower()
