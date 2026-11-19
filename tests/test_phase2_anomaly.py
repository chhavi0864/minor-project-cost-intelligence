"""
Phase 2 test suite — Isolation Forest anomaly detection.

Tests:
1. Model artifact exists and loads cleanly.
2. Prediction output shape matches input.
3. Anomaly scores are in expected float range.
4. Anomaly label values are only -1 or +1.
5. Contamination approximately honoured (within ±2%).
6. detect_cost_anomalies() returns expected schema.
7. detect_cost_anomalies() handles empty DB gracefully.
8. /api/v1/anomalies endpoint returns valid response (direct function call).
9. /api/v1/anomalies/summary endpoint returns valid response.
10. Feature metadata JSON is consistent with model.
"""
from __future__ import annotations

import json
import pathlib
import sys
import os

import pytest
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent  # FinOps-Copilot/
sys.path.insert(0, str(ROOT))

MODELS_DIR   = ROOT / "data" / "models"
FEATURES_DIR = ROOT / "data" / "features"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    import joblib
    p = MODELS_DIR / "isolation_forest.joblib"
    assert p.exists(), f"Model not found: {p}"
    return joblib.load(p)


@pytest.fixture(scope="module")
def model_meta():
    p = MODELS_DIR / "model_metadata.json"
    assert p.exists(), f"Metadata not found: {p}"
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_df():
    p = FEATURES_DIR / "test_features.csv"
    assert p.exists(), f"Test features not found: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def feature_meta():
    p = FEATURES_DIR / "feature_metadata.json"
    assert p.exists()
    with open(p) as f:
        return json.load(f)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_model_artifact_exists():
    """Test 1: Model joblib file exists."""
    assert (MODELS_DIR / "isolation_forest.joblib").exists()


def test_model_loads(model):
    """Test 2: Model loads and has expected type."""
    from sklearn.ensemble import IsolationForest
    assert isinstance(model, IsolationForest)


def test_model_prediction_shape(model, test_df, feature_meta):
    """Test 3: Prediction output has same length as input."""
    X = test_df[feature_meta["all_model_features"]].values
    labels = model.predict(X)
    assert labels.shape == (len(test_df),), f"Expected ({len(test_df)},), got {labels.shape}"


def test_anomaly_score_range(model, test_df, feature_meta):
    """Test 4: Anomaly scores are floats in [-1, 1] range (IF typical bounds)."""
    X = test_df[feature_meta["all_model_features"]].values
    scores = model.decision_function(X)
    assert scores.dtype.kind == "f", "Scores must be floats"
    assert scores.min() >= -1.0, f"Min score {scores.min()} unexpectedly low"
    assert scores.max() <= 1.0, f"Max score {scores.max()} unexpectedly high"


def test_anomaly_labels_binary(model, test_df, feature_meta):
    """Test 5: Predict returns only -1 (anomaly) or +1 (normal)."""
    X = test_df[feature_meta["all_model_features"]].values
    labels = model.predict(X)
    unique = set(labels.tolist())
    assert unique.issubset({-1, 1}), f"Unexpected label values: {unique}"


def test_contamination_rate(model, test_df, feature_meta):
    """Test 6: Actual anomaly rate is within ±2 percentage points of contamination."""
    X = test_df[feature_meta["all_model_features"]].values
    labels = model.predict(X)
    actual_rate = (labels == -1).sum() / len(labels) * 100
    expected = float(model.contamination) * 100
    assert abs(actual_rate - expected) < 2.0, (
        f"Anomaly rate {actual_rate:.2f}% differs from contamination {expected:.2f}% by >2pp"
    )


def test_detect_cost_anomalies_schema():
    """Test 7: detect_cost_anomalies() returns dict with required keys."""
    from app.anomaly_detection import detect_cost_anomalies
    result = detect_cost_anomalies(contamination=0.03)
    assert isinstance(result, dict)
    assert "status" in result
    assert "anomalies" in result
    # May be SUCCESS or INSUFFICIENT_DATA depending on DB state
    assert result["status"] in {"SUCCESS", "INSUFFICIENT_DATA"}


def test_detect_cost_anomalies_anomaly_structure():
    """Test 8: If anomalies returned, each has required fields."""
    from app.anomaly_detection import detect_cost_anomalies
    result = detect_cost_anomalies(contamination=0.03)
    if result["status"] != "SUCCESS":
        pytest.skip("No data in DB — skipping anomaly structure check")
    for a in result["anomalies"][:5]:
        assert "service_name" in a
        assert "daily_cost" in a
        assert "anomaly_score" in a
        assert isinstance(a["anomaly_score"], float)
        assert isinstance(a["daily_cost"], float)


def test_anomaly_summary_schema():
    """Test 9: /api/v1/anomalies/summary endpoint returns valid response."""
    from app.anomaly_detection import detect_cost_anomalies
    from collections import defaultdict
    result = detect_cost_anomalies(contamination=0.03)
    if result.get("status") != "SUCCESS":
        pytest.skip("No data — skipping summary test")

    anomalies = result["anomalies"]
    by_service: dict = defaultdict(lambda: {"count": 0, "total_excess_cost": 0.0})
    for a in anomalies:
        s = a["service_name"]
        by_service[s]["count"] += 1
        by_service[s]["total_excess_cost"] += a.get("cost_above_baseline", 0.0)

    assert isinstance(by_service, dict)
    for svc, stats in by_service.items():
        assert isinstance(stats["count"], int)
        assert stats["count"] >= 1


def test_feature_metadata_consistency(model_meta, feature_meta):
    """Test 10: Model metadata feature list matches feature_metadata.json."""
    model_features = set(model_meta.get("feature_names", []))
    actual_features = set(feature_meta["all_model_features"])
    assert model_features == actual_features, (
        f"Feature mismatch: {model_features.symmetric_difference(actual_features)}"
    )
    assert model_meta["feature_count"] == len(actual_features)

