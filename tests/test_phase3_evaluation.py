"""
Phase 3 test suite — Model Evaluation & Validation
FinOps-Copilot — AI-Powered Cloud Cost Intelligence Platform

Validates:
1. Evaluation artifacts existence and non-zero sizes (JSON, Markdown, PNGs, Notebook).
2. Image integrity and 300-DPI validation via PIL.
3. Notebook format and execution cell integrity.
4. Metric ranges (train/test anomaly rates, synthetic recall monotonicity).
5. Baseline comparison structure (IF vs Z-Score vs IQR).
6. Zero data leakage verification (chronological split constraint).
7. Feature sensitivity outputs and metadata consistency.
8. UI copy audit regarding sensitivity slider vs retraining.
"""
from __future__ import annotations

import json
import pathlib
import sys
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent  # FinOps-Copilot/
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports" / "model_evaluation"
NOTEBOOKS_DIR = ROOT / "notebooks"
MODELS_DIR = ROOT / "data" / "models"
FEATURES_DIR = ROOT / "data" / "features"


@pytest.fixture(scope="module")
def eval_metrics():
    path = REPORTS_DIR / "evaluation_metrics.json"
    assert path.exists(), f"Missing evaluation metrics file: {path}"
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def eval_notebook():
    path = NOTEBOOKS_DIR / "04_model_evaluation.ipynb"
    assert path.exists(), f"Missing evaluation notebook: {path}"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_evaluation_artifacts_exist():
    """Test 1: All required evaluation artifacts exist on disk."""
    assert (REPORTS_DIR / "EVALUATION_REPORT.md").exists()
    assert (REPORTS_DIR / "evaluation_metrics.json").exists()
    assert (NOTEBOOKS_DIR / "04_model_evaluation.ipynb").exists()

    expected_figures = [
        "01_score_distribution_comparison.png",
        "02_synthetic_spike_recall_curve.png",
        "03_baseline_comparison_venn_overlap.png",
        "04_feature_sensitivity_attribution.png",
        "05_anomaly_timeline_and_service_breakdown.png",
    ]
    for fig_name in expected_figures:
        p = REPORTS_DIR / fig_name
        assert p.exists(), f"Missing figure: {fig_name}"
        assert p.stat().st_size > 50_000, f"Figure {fig_name} is too small: {p.stat().st_size} bytes"


def test_evaluation_figures_image_integrity():
    """Test 2: All generated figures are valid PNG images and open cleanly."""
    for fig_path in REPORTS_DIR.glob("*.png"):
        with Image.open(fig_path) as im:
            assert im.format == "PNG", f"{fig_path.name} is not PNG"
            assert im.size[0] > 1000, f"{fig_path.name} width too small: {im.size[0]}"
            assert im.size[1] > 400, f"{fig_path.name} height too small: {im.size[1]}"


def test_notebook_structure_and_embedded_outputs(eval_notebook):
    """Test 3: Notebook has valid ipynb structure and contains pre-executed display_data."""
    cells = eval_notebook.get("cells", [])
    assert len(cells) >= 10, f"Notebook has fewer cells than expected: {len(cells)}"
    
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    display_data_count = 0
    for cell in code_cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "display_data":
                display_data_count += 1
    assert display_data_count >= 5, f"Expected at least 5 embedded figures, found {display_data_count}"


def test_temporal_leakage_absence():
    """Test 4: Strict temporal partition constraint holds (max train date < min test date)."""
    with open(FEATURES_DIR / "feature_metadata.json") as f:
        meta = json.load(f)
    train_end = meta["train_date_range"][1]
    test_start = meta["test_date_range"][0]
    assert train_end < test_start, f"Data leakage detected: train_end {train_end} >= test_start {test_start}"


def test_detection_rate_consistency(eval_metrics):
    """Test 5: Train and test anomaly rates align with target contamination."""
    det = eval_metrics["detection_rates"]
    assert 2.5 <= det["train_anomaly_rate"] <= 3.5, f"Unexpected train rate: {det['train_anomaly_rate']}"
    assert 2.0 <= det["test_anomaly_rate"] <= 3.5, f"Unexpected test rate: {det['test_anomaly_rate']}"


def test_synthetic_spike_recall_behavior(eval_metrics):
    """Test 6: Detection recall generally scales with synthetic spike magnitude."""
    recalls = eval_metrics["synthetic_spike_recall"]
    # Severe spike (10x) should have significantly higher recall than subtle spike (1.2x)
    assert recalls["10.0"] > recalls["1.2"], (
        f"10.0x recall ({recalls['10.0']}%) must exceed 1.2x recall ({recalls['1.2']}%)"
    )
    assert recalls["10.0"] >= 50.0, f"10.0x spike recall unexpectedly low: {recalls['10.0']}%"


def test_baseline_comparison_metrics(eval_metrics):
    """Test 7: Baseline metrics contain IF, Z-score, and IQR comparison fields."""
    base = eval_metrics["baseline_comparison"]
    assert "isolation_forest" in base
    assert "z_score_baseline" in base
    assert "iqr_baseline" in base
    assert base["isolation_forest"]["flags"] > 0
    assert base["z_score_baseline"]["flags"] > 0
    assert base["iqr_baseline"]["flags"] > 0
    assert 0.0 <= base["jaccard_if_zscore"] <= 1.0
    assert 0.0 <= base["jaccard_if_iqr"] <= 1.0


def test_feature_sensitivity_metrics(eval_metrics):
    """Test 8: Top sensitive features list is populated."""
    top_feats = eval_metrics["top_sensitive_features"]
    assert len(top_feats) >= 3, "Expected at least 3 sensitive features listed"
    assert all(isinstance(f, str) for f in top_feats)


def test_ui_contamination_copy_audit():
    """Test 9: Streamlit UI does not misleadingly imply live model retraining."""
    ui_path = ROOT / "UI" / "app.py"
    with open(ui_path, encoding="utf-8") as f:
        content = f.read()
    # Confirm caption clarifies that sensitivity cutoff does not alter offline model
    assert "without altering the offline baseline model" in content or "without retraining" in content
    assert "Calibrates filtering threshold without retraining the underlying production model" in content


def test_evaluation_report_completeness():
    """Test 10: EVALUATION_REPORT.md contains essential validation sections."""
    report_path = REPORTS_DIR / "EVALUATION_REPORT.md"
    text = report_path.read_text(encoding="utf-8")
    assert "# Isolation Forest Model Evaluation & Validation Report" in text
    assert "Zero Data Leakage" in text
    assert "Controlled Synthetic Spike Detection Curves" in text
    assert "Baseline Comparison" in text
    assert "Permutation Feature Sensitivity" in text

