import json
import pytest
from api.main import app


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code in [200, 503]
    data = response.get_json()
    assert "status" in data
    assert "model_loaded" in data
    assert "scaler_loaded" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"# HELP" in response.data  # Prometheus format


def test_predict_invalid_payload(client):
    # Missing required fields (should fail validation)
    payload = {"instances": [{}]}
    response = client.post(
        "/predict", data=json.dumps(payload), content_type="application/json"
    )
    assert response.status_code in [422, 503]


def test_predict_valid_payload(client):
    payload = {
        "instances": [
            {
                "MedInc": 5.0,
                "HouseAge": 20.0,
                "AveRooms": 5.0,
                "AveBedrms": 1.0,
                "Population": 100.0,
                "AveOccup": 3.0,
                "Latitude": 34.0,
                "Longitude": -118.0
            }
        ]
    }
    response = client.post(
        "/predict", data=json.dumps(payload), content_type="application/json"
    )

    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.get_json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
