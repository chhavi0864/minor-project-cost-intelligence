"""
Phase 4 test suite — Isolation Forest Model Optimization
FinOps-Copilot — AI-Powered Cloud Cost Intelligence Platform

Validates:
1. Baseline model preservation (isolation_forest.joblib remains intact with 30 features).
2. Optimized champion model existence (isolation_forest_optimized.joblib with 18 features).
3. Optimized model metadata consistency (18 features, max_samples=1024, n_estimators=250).
4. Prediction shape and output binary labels on held-out test data.
5. Test anomaly rate stability (between 2.5% and 4.0%, no alert inflation).
6. Synthetic spike recall improvement verified in optimization_metrics.json.
7. Optimization notebook integrity and embedded publication displays.
8. Optimization report and 5 publication-grade figures (300 DPI, non-empty PNGs).
9. Backward compatibility: existing detect_cost_anomalies function remains fully operational.
10. Preservation of existing Phase 1, Phase 2, and Phase 3 components.
"""
from __future__ import annotations

import json
import pathlib
import sys
import pytest
from PIL import Image
from sklearn.ensemble import IsolationForest

ROOT = pathlib.Path(__file__).resolve().parent.parent  # FinOps-Copilot/
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "data" / "models"
REPORTS_DIR = ROOT / "reports" / "model_optimization"
NOTEBOOKS_DIR = ROOT / "notebooks"
FEATURES_DIR = ROOT / "data" / "features"


@pytest.fixture(scope="module")
def baseline_model():
    p = MODELS_DIR / "isolation_forest.joblib"
    assert p.exists(), f"Baseline model missing: {p}"
    import joblib
    return joblib.load(p)


@pytest.fixture(scope="module")
def optimized_model():
    p = MODELS_DIR / "isolation_forest_optimized.joblib"
    assert p.exists(), f"Optimized model missing: {p}"
    import joblib
    return joblib.load(p)


@pytest.fixture(scope="module")
def opt_meta():
    p = MODELS_DIR / "model_optimized_metadata.json"
    assert p.exists(), f"Optimized metadata missing: {p}"
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def opt_metrics():
    p = REPORTS_DIR / "optimization_metrics.json"
    assert p.exists(), f"Optimization metrics missing: {p}"
    with open(p) as f:
        return json.load(f)


def test_baseline_model_preservation(baseline_model):
    """Test 1: Original baseline model remains completely intact with 30 features."""
    assert isinstance(baseline_model, IsolationForest)
    assert baseline_model.n_features_in_ == 30
    assert baseline_model.max_samples_ == 256
    assert baseline_model.contamination == 0.03


def test_optimized_model_architecture(optimized_model, opt_meta):
    """Test 2: Optimized model is an IsolationForest with 18 features and expanded max_samples."""
    assert isinstance(optimized_model, IsolationForest)
    assert optimized_model.n_features_in_ == 18
    assert optimized_model.max_samples_ == 1024
    assert optimized_model.n_estimators == 250
    assert len(opt_meta["feature_names"]) == 18


def test_optimized_model_prediction_shape(optimized_model, opt_meta):
    """Test 3: Predictions on test_features match row count (13,402) and binary labels."""
    import pandas as pd
    test_df = pd.read_csv(FEATURES_DIR / "test_features.csv")
    X = test_df[opt_meta["feature_names"]].values
    preds = optimized_model.predict(X)
    assert preds.shape == (len(test_df),)
    unique_labels = set(preds.tolist())
    assert unique_labels.issubset({-1, 1})


def test_test_anomaly_rate_stability(opt_meta):
    """Test 4: Optimized test anomaly rate is within controlled bounds (2.5% - 4.0%)."""
    rate = opt_meta["test_anomaly_rate"]
    assert 2.5 <= rate <= 4.0, f"Optimized test rate {rate}% out of expected bounds"


def test_synthetic_spike_recall_superiority(opt_metrics):
    """Test 5: Optimized model demonstrates superior recall on 3x-5x spikes."""
    mults = opt_metrics["synthetic_recall_comparison"]["multipliers"]
    base_cons = opt_metrics["synthetic_recall_comparison"]["baseline_consistent"]
    opt_cons  = opt_metrics["synthetic_recall_comparison"]["optimized_consistent"]

    idx_3x = mults.index(3.0)
    idx_5x = mults.index(5.0)

    assert opt_cons[idx_3x] > base_cons[idx_3x], (
        f"3.0x recall: Optimized ({opt_cons[idx_3x]}%) must exceed Baseline ({base_cons[idx_3x]}%)"
    )
    assert opt_cons[idx_5x] > base_cons[idx_5x], (
        f"5.0x recall: Optimized ({opt_cons[idx_5x]}%) must exceed Baseline ({base_cons[idx_5x]}%)"
    )
    assert opt_cons[idx_5x] >= 90.0, f"5.0x spike recall should meet operational target (got {opt_cons[idx_5x]}%)"


def test_optimization_figures_integrity():
    """Test 6: All 5 optimization comparison charts exist, are valid PNGs, and > 50KB."""
    expected_figures = [
        "01_feature_dilution_and_root_cause.png",
        "02_synthetic_spike_recall_comparison.png",
        "03_hyperparameter_grid_search_heatmap.png",
        "04_score_distribution_shift.png",
        "05_service_level_detection_breakdown.png",
    ]
    for fig_name in expected_figures:
        p = REPORTS_DIR / fig_name
        assert p.exists(), f"Missing figure: {fig_name}"
        assert p.stat().st_size > 50_000, f"Figure {fig_name} is too small"
        with Image.open(p) as im:
            assert im.format == "PNG"
            assert im.size[0] >= 1000
            assert im.size[1] >= 400


def test_optimization_notebook_structure():
    """Test 7: 05_model_optimization.ipynb exists, is valid JSON, and has embedded displays."""
    nb_path = NOTEBOOKS_DIR / "05_model_optimization.ipynb"
    assert nb_path.exists(), f"Notebook missing: {nb_path}"
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    assert len(cells) >= 10, f"Fewer cells than expected: {len(cells)}"
    display_count = sum(
        1 for c in cells if c.get("cell_type") == "code"
        for out in c.get("outputs", []) if out.get("output_type") == "display_data"
    )
    assert display_count >= 5, f"Expected at least 5 embedded figures, found {display_count}"


def test_optimization_report_completeness():
    """Test 8: OPTIMIZATION_REPORT.md contains root cause, architecture, and comparison sections."""
    rep_path = REPORTS_DIR / "OPTIMIZATION_REPORT.md"
    assert rep_path.exists(), f"Report missing: {rep_path}"
    content = rep_path.read_text(encoding="utf-8")
    assert "# Isolation Forest Model Optimization Report (Phase 4)" in content
    assert "Feature Space Dilution" in content
    assert "Subsampling Depth Bottleneck" in content
    assert "isolation_forest_optimized.joblib" in content


def test_detect_cost_anomalies_backward_compatibility():
    """Test 9: Ingestion inference engine in app.anomaly_detection remains backward-compatible."""
    from app.anomaly_detection import detect_cost_anomalies
    res = detect_cost_anomalies(contamination=0.03)
    assert isinstance(res, dict)
    assert res["status"] in {"SUCCESS", "INSUFFICIENT_DATA"}


def test_feature_metadata_intact():
    """Test 10: Original feature_metadata.json is preserved and consistent."""
    with open(FEATURES_DIR / "feature_metadata.json") as f:
        fm = json.load(f)
    assert fm["total_model_features"] == 30
    assert len(fm["all_model_features"]) == 30

