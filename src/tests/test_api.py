from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_predict():
    resp = client.post("/predict", json={"features":[0.0]*10})
    assert resp.status_code == 200
    assert "score" in resp.json()