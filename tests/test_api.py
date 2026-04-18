"""
tests/test_api.py — Integration tests for the anomaly detection API.

Run with:
    pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# ─── Sample transactions ───────────────────────────────────────────────────────

NORMAL_TX = {
    "Time": 406.0,
    "V1": 1.191, "V2": 0.266, "V3": 0.166, "V4": 0.448,
    "V5": 0.060, "V6": -0.082, "V7": -0.078, "V8": 0.085,
    "V9": -0.255, "V10": -0.167, "V11": 1.613, "V12": 1.065,
    "V13": 0.489, "V14": -0.144, "V15": 0.636, "V16": 0.464,
    "V17": -0.115, "V18": -0.183, "V19": -0.146, "V20": -0.069,
    "V21": -0.225, "V22": -0.638, "V23": 0.101, "V24": -0.340,
    "V25": 0.167, "V26": 0.126, "V27": -0.009, "V28": 0.015,
    "Amount": 149.62,
}

# Known suspicious-pattern transaction (extreme V values)
FRAUD_TX = {
    "Time": 406.0,
    "V1": -4.771, "V2": 3.201, "V3": -4.101, "V4": 5.407,
    "V5": -2.200, "V6": -3.210, "V7": -4.050, "V8": 2.711,
    "V9": -3.500, "V10": -5.100, "V11": 4.300, "V12": -5.801,
    "V13": -1.020, "V14": -7.500, "V15": 0.700, "V16": -2.510,
    "V17": -5.200, "V18": -0.510, "V19": 0.820, "V20": 0.310,
    "V21": 0.910, "V22": -0.076, "V23": -0.821, "V24": 0.610,
    "V25": 0.095, "V26": -0.390, "V27": 0.820, "V28": 0.700,
    "Amount": 1.0,
}


# ─── Health & info ─────────────────────────────────────────────────────────────

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "FraudGuard" in r.json()["message"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "model_version" in data


def test_model_info_when_loaded():
    r = client.get("/health")
    if not r.json()["model_loaded"]:
        pytest.skip("Model not trained yet — skipping model info test")

    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert data["model_type"] == "IsolationForest"
    assert data["feature_count"] == 30
    assert data["n_estimators"] == 200


# ─── Prediction ────────────────────────────────────────────────────────────────

def _skip_if_no_model():
    r = client.get("/health")
    if not r.json()["model_loaded"]:
        pytest.skip("Model not trained yet")


def test_predict_returns_valid_schema():
    _skip_if_no_model()
    r = client.post("/predict", json=NORMAL_TX)
    assert r.status_code == 200
    data = r.json()

    assert "is_fraud" in data
    assert "label" in data
    assert "anomaly_score" in data
    assert "confidence" in data
    assert "explanation" in data
    assert "model_version" in data

    assert data["label"] in ("FRAUD", "NORMAL")
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["explanation"]["top_features"]) == 5


def test_predict_normal_transaction():
    _skip_if_no_model()
    r = client.post("/predict", json=NORMAL_TX)
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == "NORMAL"
    assert data["is_fraud"] is False

def test_predict_suspicious_transaction():
    _skip_if_no_model()
    normal_r = client.post("/predict", json=NORMAL_TX)
    fraud_r = client.post("/predict", json=FRAUD_TX)
    assert fraud_r.status_code == 200
    # Suspicious transaction should score lower (more anomalous) than normal
    assert fraud_r.json()["anomaly_score"] < normal_r.json()["anomaly_score"]
    assert fraud_r.json()["confidence"] > normal_r.json()["confidence"]


def test_predict_missing_field():
    _skip_if_no_model()
    bad_tx = {k: v for k, v in NORMAL_TX.items() if k != "V1"}
    r = client.post("/predict", json=bad_tx)
    assert r.status_code == 422  # Unprocessable Entity


def test_predict_wrong_type():
    _skip_if_no_model()
    bad_tx = {**NORMAL_TX, "Amount": "not-a-number"}
    r = client.post("/predict", json=bad_tx)
    assert r.status_code == 422


# ─── Batch prediction ──────────────────────────────────────────────────────────

def test_batch_predict():
    _skip_if_no_model()
    payload = {"transactions": [NORMAL_TX, FRAUD_TX, NORMAL_TX]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    data = r.json()

    assert data["total"] == 3
    assert data["fraud_count"] + data["normal_count"] == 3
    assert len(data["results"]) == 3


def test_batch_empty():
    _skip_if_no_model()
    r = client.post("/predict/batch", json={"transactions": []})
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_batch_too_large():
    _skip_if_no_model()
    big = {"transactions": [NORMAL_TX] * 501}
    r = client.post("/predict/batch", json=big)
    assert r.status_code == 422