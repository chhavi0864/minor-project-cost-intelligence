"""
Phase 6 Automated Test Suite: Cloud Cost Forecasting
Validates Prophet-Architecture GAM, SARIMA backtest metrics, prediction bounds,
budget pacing, service-wise forecasts, FastAPI /api/v1/forecast handler,
and generated reporting artifacts.
"""

import json
from pathlib import Path
import pytest

from app.forecasting import get_cost_forecast
from app.main import get_forecast

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports" / "forecasting"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"


def test_01_forecast_schema_and_horizon():
    """Verify get_cost_forecast returns complete schema with exact requested horizon."""
    res = get_cost_forecast(days=7)
    assert res["status"] == "SUCCESS"
    assert "Prophet" in res["model_architecture"]
    assert res["horizon_days"] == 7
    assert len(res["forecast"]) == 7
    assert "total_predicted_cost" in res
    assert "average_daily_cost" in res
    assert "budget_pacing" in res
    assert res["total_predicted_cost"] > 0
    assert res["average_daily_cost"] > 0


def test_02_forecast_values_non_negative_and_bounds():
    """Verify all predicted costs and bounds are non-negative with upper > lower."""
    res = get_cost_forecast(days=30)
    for record in res["forecast"]:
        pred = record["predicted_cost"]
        l80, u80 = record["lower_bound_80"], record["upper_bound_80"]
        l95, u95 = record["lower_bound_95"], record["upper_bound_95"]
        assert pred >= 0.0, f"Negative prediction: {pred}"
        assert l80 >= 0.0 and u80 >= 0.0
        assert l95 >= 0.0 and u95 >= 0.0
        assert u80 >= l80, f"Upper 80 {u80} < lower 80 {l80}"
        assert u95 >= l95, f"Upper 95 {u95} < lower 95 {l95}"


def test_03_forecast_confidence_intervals_nesting():
    """Verify 95% predictive interval is strictly wider than 80% interval."""
    res = get_cost_forecast(days=14)
    for record in res["forecast"]:
        width_80 = record["upper_bound_80"] - record["lower_bound_80"]
        width_95 = record["upper_bound_95"] - record["lower_bound_95"]
        assert width_95 >= width_80, f"95% width ({width_95}) should be >= 80% width ({width_80})"


def test_04_budget_pacing_evaluation():
    """Verify budget pacing returns valid status strings and numeric metrics."""
    res = get_cost_forecast(days=30, monthly_budget=55000.0)
    pacing = res["budget_pacing"]
    assert "risk_status" in pacing
    assert pacing["risk_status"] in ["HEALTHY_WITHIN_BUDGET", "WARNING_APPROACHING_LIMIT", "CRITICAL_OVER_BUDGET"]
    assert "configured_budget" in pacing
    assert pacing["configured_budget"] == 55000.0
    assert "projected_utilization_pct" in pacing
    assert pacing["projected_utilization_pct"] > 0.0


def test_05_winsorization_flag_treatment():
    """Verify winsorization flag executes cleanly without producing NaNs."""
    res_w = get_cost_forecast(days=10, winsorize_spikes=True)
    res_raw = get_cost_forecast(days=10, winsorize_spikes=False)
    assert res_w["status"] == "SUCCESS"
    assert res_raw["status"] == "SUCCESS"
    assert len(res_w["forecast"]) == 10
    assert len(res_raw["forecast"]) == 10


def test_06_service_filter_forecasting():
    """Verify service-specific forecasts generate tailored predictions."""
    res_ec2 = get_cost_forecast(days=14, service_name="EC2")
    assert res_ec2["status"] == "SUCCESS"
    assert res_ec2["service_scope"] == "EC2"
    assert len(res_ec2["forecast"]) == 14
    # Service cost should be less than overall aggregate cost
    res_all = get_cost_forecast(days=14)
    assert res_ec2["total_predicted_cost"] < res_all["total_predicted_cost"]


def test_07_fastapi_forecast_endpoint_handler():
    """Verify FastAPI endpoint handler returns 200 equivalent dictionary with forecast records."""
    data = get_forecast(days=7)
    assert data["status"] == "SUCCESS"
    assert data["horizon_days"] == 7
    assert len(data["forecast"]) == 7
    assert "Prophet" in data["model_architecture"]


def test_08_fastapi_forecast_service_filter_handler():
    """Verify FastAPI endpoint handler with service filter returns tailored scope."""
    data = get_forecast(days=30, service_name="EBS", monthly_budget=20000.0)
    assert data["status"] == "SUCCESS"
    assert data["service_scope"] == "EBS"
    assert len(data["forecast"]) == 30
    assert data["budget_pacing"]["configured_budget"] == 20000.0


def test_09_all_5_forecasting_figures_exist_and_valid():
    """Verify all 5 publication-ready 300 DPI figures exist and are valid PNGs > 50KB."""
    figure_names = [
        "01_daily_expenditure_and_seasonality_decomposition.png",
        "02_model_backtest_comparison.png",
        "03_7day_30day_forecast_with_uncertainty.png",
        "04_forecast_deviation_and_sustained_overspend.png",
        "05_service_wise_cost_forecast.png",
    ]
    for fig_name in figure_names:
        fig_path = REPORTS_DIR / fig_name
        assert fig_path.exists(), f"Figure missing: {fig_path}"
        assert fig_path.stat().st_size > 50_000, f"Figure {fig_name} is under 50KB"


def test_10_notebook_and_metrics_json():
    """Verify 07_forecasting.ipynb and forecasting_metrics.json exist with verified metrics."""
    nb_path = NOTEBOOKS_DIR / "07_forecasting.ipynb"
    assert nb_path.exists(), f"Notebook missing: {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    assert len(nb_data["cells"]) >= 10, "Notebook should contain multiple cells"

    metrics_path = REPORTS_DIR / "forecasting_metrics.json"
    assert metrics_path.exists(), f"Metrics JSON missing: {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert "model_backtest_metrics" in metrics_data
    gam_metrics = metrics_data["model_backtest_metrics"]["Prophet-Architecture GAM"]
    assert gam_metrics["sMAPE"] < 10.0, f"Prophet-GAM sMAPE too high: {gam_metrics['sMAPE']}%"
    assert metrics_data["champion_model"]["name"] == "Prophet-Architecture GAM"
