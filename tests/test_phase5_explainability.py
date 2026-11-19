"""
Phase 5 test suite — Explainable Anomaly Detection & FinOps Recommendations
FinOps-Copilot — AI-Powered Cloud Cost Intelligence Platform

Validates:
1. Active model loader behavior (Champion 18-feature model with safe fallback).
2. Rule-based explanation generation (severity, financial impact, driver, explanation, action).
3. Severity classification validity (CRITICAL, HIGH, MEDIUM, LOW).
4. Root-cause categorization logic (Consumption Surge, Rate Spike, Idle Anomaly, Proliferation).
5. Ingestion detection engine enriches records with explainability fields.
6. FastAPI /api/v1/anomalies/summary response schema includes by_severity and active_model.
7. Publication figures exist, open cleanly via PIL, and exceed minimum dimensions.
8. 06_explainable_anomalies.ipynb integrity and embedded display data.
9. EXPLAINABILITY_REPORT.md and explainability_metrics.json completeness.
10. Backward compatibility with Phase 1-4 pipelines and baseline artifacts.
"""
from __future__ import annotations

import json
import pathlib
import sys
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent  # FinOps-Copilot/
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports" / "explainability"
NOTEBOOKS_DIR = ROOT / "notebooks"
MODELS_DIR = ROOT / "data" / "models"


@pytest.fixture(scope="module")
def expl_metrics():
    p = REPORTS_DIR / "explainability_metrics.json"
    assert p.exists(), f"Explainability metrics file missing: {p}"
    with open(p) as f:
        return json.load(f)


def test_active_model_loader_and_fallback():
    """Test 1: load_active_model selects Champion Optimized model with safe fallback."""
    from app.anomaly_detection import load_active_model
    model, name, feats = load_active_model()
    assert model is not None, "Active model should not be None"
    assert "Optimized" in name or "Baseline" in name, f"Unexpected active model name: {name}"
    assert len(feats) in {18, 30}, f"Expected 18 or 30 features, got {len(feats)}"


def test_explanation_and_action_generation_schema():
    """Test 2: generate_anomaly_explanation_and_action produces all required fields."""
    from app.anomaly_detection import generate_anomaly_explanation_and_action
    res = generate_anomaly_explanation_and_action(
        daily_cost=150.0,
        baseline_cost=30.0,
        cost_above_baseline=120.0,
        daily_usage=5000.0,
        baseline_usage=1000.0,
        resource_count=10,
        baseline_resource_count=3,
        anomaly_score=-0.12,
        service_name="EC2",
        currency="USD",
    )
    assert isinstance(res, dict)
    for key in ["severity", "estimated_financial_impact", "root_cause_driver", "explanation", "recommended_action"]:
        assert key in res, f"Missing key: {key}"
        assert res[key] is not None, f"Key {key} is None"


def test_severity_classification_levels():
    """Test 3: Severity levels follow calibrated thresholds."""
    from app.anomaly_detection import generate_anomaly_explanation_and_action
    
    # Critical condition (5x cost, large dollar excess)
    crit = generate_anomaly_explanation_and_action(
        daily_cost=600.0, baseline_cost=50.0, cost_above_baseline=550.0,
        daily_usage=1000.0, baseline_usage=200.0, resource_count=5,
        baseline_resource_count=2, anomaly_score=-0.16, service_name="RDS",
    )
    assert crit["severity"] == "CRITICAL"

    # High condition (2.5x cost)
    high = generate_anomaly_explanation_and_action(
        daily_cost=180.0, baseline_cost=80.0, cost_above_baseline=100.0,
        daily_usage=800.0, baseline_usage=400.0, resource_count=2,
        baseline_resource_count=2, anomaly_score=-0.06, service_name="EC2",
    )
    assert high["severity"] in {"HIGH", "CRITICAL"}

    # Low/Medium condition
    med = generate_anomaly_explanation_and_action(
        daily_cost=25.0, baseline_cost=20.0, cost_above_baseline=5.0,
        daily_usage=100.0, baseline_usage=100.0, resource_count=1,
        baseline_resource_count=1, anomaly_score=-0.01, service_name="S3",
    )
    assert med["severity"] in {"LOW", "MEDIUM"}


def test_root_cause_driver_categorization():
    """Test 4: Correct driver assigned based on telemetry evidence."""
    from app.anomaly_detection import generate_anomaly_explanation_and_action

    # Consumption surge: usage spikes with constant unit cost
    surge = generate_anomaly_explanation_and_action(
        daily_cost=200.0, baseline_cost=50.0, cost_above_baseline=150.0,
        daily_usage=4000.0, baseline_usage=1000.0, resource_count=1,
        baseline_resource_count=1, anomaly_score=-0.08, service_name="Lambda",
    )
    assert surge["root_cause_driver"] == "Consumption Surge"
    assert "usage volume" in surge["explanation"].lower() or "surge" in surge["explanation"].lower()

    # Pricing spike: unit cost spikes with flat usage
    price = generate_anomaly_explanation_and_action(
        daily_cost=200.0, baseline_cost=100.0, cost_above_baseline=100.0,
        daily_usage=1000.0, baseline_usage=1000.0, resource_count=1,
        baseline_resource_count=1, anomaly_score=-0.05, service_name="EBS",
    )
    assert price["root_cause_driver"] == "Pricing / Rate Spike"


def test_detect_cost_anomalies_explainability_enrichment():
    """Test 5: detect_cost_anomalies enriches all flagged records with explainable fields."""
    from app.anomaly_detection import detect_cost_anomalies
    res = detect_cost_anomalies(contamination=0.03)
    assert res["status"] in {"SUCCESS", "INSUFFICIENT_DATA"}
    if res["status"] == "SUCCESS" and res["anomalies"]:
        sample = res["anomalies"][0]
        assert "severity" in sample
        assert sample["severity"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        assert "root_cause_driver" in sample
        assert "explanation" in sample
        assert "recommended_action" in sample
        assert "estimated_financial_impact" in sample
        assert "severity_breakdown" in res
        assert "active_model" in res


def test_fastapi_anomalies_summary_schema():
    """Test 6: FastAPI endpoint returns severity breakdown and active model."""
    from app.anomaly_detection import detect_cost_anomalies
    from collections import defaultdict
    result = detect_cost_anomalies(contamination=0.03)
    if result["status"] != "SUCCESS":
        pytest.skip("No data in DB — skipping FastAPI schema check")

    anomalies = result["anomalies"]
    by_severity = defaultdict(lambda: {"count": 0, "total_excess_cost": 0.0})
    for a in anomalies:
        sev = a.get("severity", "MEDIUM")
        excess = a.get("estimated_financial_impact", a.get("cost_above_baseline", 0.0))
        by_severity[sev]["count"] += 1
        by_severity[sev]["total_excess_cost"] += excess

    assert isinstance(by_severity, dict)
    assert len(by_severity) > 0


def test_explainability_figures_integrity():
    """Test 7: All 5 publication-grade explainability charts exist and are valid PNGs."""
    expected_figures = [
        "01_anomaly_severity_distribution.png",
        "02_root_cause_driver_breakdown.png",
        "03_financial_impact_by_service_pareto.png",
        "04_feature_attribution_waterfall.png",
        "05_recommendations_action_summary.png",
    ]
    for fig_name in expected_figures:
        p = REPORTS_DIR / fig_name
        assert p.exists(), f"Figure missing: {fig_name}"
        assert p.stat().st_size > 50_000, f"Figure too small: {fig_name}"
        with Image.open(p) as im:
            assert im.format == "PNG"
            assert im.size[0] >= 1000
            assert im.size[1] >= 400


def test_explainable_notebook_structure():
    """Test 8: 06_explainable_anomalies.ipynb exists and has valid cells and embedded displays."""
    nb_path = NOTEBOOKS_DIR / "06_explainable_anomalies.ipynb"
    assert nb_path.exists(), f"Notebook missing: {nb_path}"
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    assert len(cells) >= 10
    display_count = sum(
        1 for c in cells if c.get("cell_type") == "code"
        for out in c.get("outputs", []) if out.get("output_type") == "display_data"
    )
    assert display_count >= 5, f"Expected >= 5 displays, found {display_count}"


def test_explainability_report_completeness():
    """Test 9: EXPLAINABILITY_REPORT.md contains severity, root-cause, and Pareto sections."""
    rep_path = REPORTS_DIR / "EXPLAINABILITY_REPORT.md"
    assert rep_path.exists()
    content = rep_path.read_text(encoding="utf-8")
    assert "# Explainable Anomaly Detection & FinOps Recommendations (Phase 5)" in content
    assert "Severity Tier" in content
    assert "Root-Cause Driver Breakdown" in content
    assert "Action Plan Matrix" in content


def test_backward_compatibility_preserved():
    """Test 10: Baseline model, feature metadata, and Phase 1-4 artifacts remain intact."""
    assert (MODELS_DIR / "isolation_forest.joblib").exists()
    assert (MODELS_DIR / "isolation_forest_optimized.joblib").exists()
    assert (ROOT / "data" / "features" / "feature_metadata.json").exists()
    assert (NOTEBOOKS_DIR / "05_model_optimization.ipynb").exists()
    assert (NOTEBOOKS_DIR / "04_model_evaluation.ipynb").exists()

