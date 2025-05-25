from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "alive"}

def test_valid_prediction():
    data = {"features": [5.1, 3.5, 1.4, 0.2]}
    res = client.post("/predict", json=data)
    assert res.status_code == 200
    assert "prediction" in res.json()

def test_invalid_short_input():
    res = client.post("/predict", json={"features": [1.0, 2.0]})
    assert res.status_code == 422

def test_non_numeric_input():
    res = client.post("/predict", json={"features": ["a", 2.0, 3.0, 4.0]})
    assert res.status_code == 422

def test_too_many_values():
    res = client.post("/predict", json={"features": [1, 2, 3, 4, 5]})
    assert res.status_code == 422
