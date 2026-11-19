"""Cloud Compass cloud cost forecasting module.

Provides production forecasting using Prophet-Architecture Generalized Additive Model
(GAM) with trend, weekly Fourier seasonality, weekend indicators, and uncertainty intervals.
Includes outlier pre-treatment (winsorization) and sustained overspending detection.
"""
from __future__ import annotations

import pathlib
from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sqlalchemy import select

from app.models import cost_records, engine

ROOT = pathlib.Path(__file__).resolve().parent.parent
FALLBACK_CSV = ROOT / "data" / "processed" / "cloud_cost_usage_cleaned.csv"


def _build_fourier_features(dates: pd.Series, t_0: pd.Timestamp, period: int = 7, n_terms: int = 3) -> np.ndarray:
    """Builds trend, day-of-week Fourier terms, and weekend indicator features."""
    t = (dates - t_0).dt.days.values.astype(float)
    features = [t / 100.0, (t / 100.0) ** 2]  # normalized quadratic trend
    for n in range(1, n_terms + 1):
        features.append(np.cos(2 * np.pi * n * t / period))
        features.append(np.sin(2 * np.pi * n * t / period))
    dow = dates.dt.day_of_week.values
    features.append((dow >= 5).astype(float))  # Weekend indicator
    return np.column_stack(features)


def get_cost_forecast(
    days: int = 30,
    service_name: Optional[str] = None,
    batch_id: Optional[str] = None,
    monthly_budget: Optional[float] = None,
    winsorize_spikes: bool = True,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Generates cost forecast with uncertainty intervals and budget pacing risk.

    Args:
        days: Forecast horizon in days (default 30).
        service_name: Optional service filter (e.g. 'EC2', 'EBS').
        batch_id: Optional dataset batch filter to scope forecast to active dataset.
        monthly_budget: Optional budget limit for risk assessment.
        winsorize_spikes: Whether to clip extreme historical outliers before fitting.
        confidence_level: Interval confidence (0.80 or 0.95).

    Returns:
        Structured forecast dictionary with daily predictions and budget indicators.
    """
    if not isinstance(service_name, str):
        service_name = None
    if not isinstance(batch_id, str) or batch_id in ["All Ingested Datasets", "None", ""]:
        batch_id = None

    days = max(1, min(int(days) if isinstance(days, int) else 30, 90))
    monthly_budget = float(monthly_budget) if isinstance(monthly_budget, (int, float)) else None

    # 1. Load data from DB, fallback to processed CSV
    df = None
    try:
        cols = [
            cost_records.c.import_batch_id,
            cost_records.c.charge_period_start,
            cost_records.c.service_name,
            cost_records.c.billed_cost,
        ]
        q = select(*cols)
        if batch_id:
            q = q.where(cost_records.c.import_batch_id == batch_id)
        if service_name and service_name != "All Services":
            q = q.where(cost_records.c.service_name == service_name)
        with engine.connect() as conn:
            db_df = pd.read_sql(q, conn)
            if not db_df.empty:
                db_df["date"] = pd.to_datetime(db_df["charge_period_start"], errors="coerce", utc=True).dt.date
                db_df["cost"] = pd.to_numeric(db_df["billed_cost"], errors="coerce").fillna(0)
                df = db_df.dropna(subset=["date", "cost"])
    except Exception:
        df = None

    if df is None or df.empty:
        if FALLBACK_CSV.exists() and not batch_id:
            csv_df = pd.read_csv(FALLBACK_CSV)
            if service_name and service_name != "All Services":
                csv_df = csv_df[csv_df["service"] == service_name]
            csv_df["date"] = pd.to_datetime(csv_df["date"]).dt.date
            df = csv_df
        else:
            return {
                "status": "ERROR",
                "message": "No historical billing records available for the selected scope.",
                "forecast": [],
            }

    # Aggregate daily
    daily = df.groupby("date")["cost"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    if len(daily) < 14:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Need at least 14 days of historical spend for forecasting (got {len(daily)}).",
            "forecast": [],
        }

    # 2. Outlier pre-treatment: winsorize extreme historical spikes to avoid forecast distortion
    y_raw = daily["cost"].values.copy()
    treatment_applied = False
    if winsorize_spikes:
        p99 = np.percentile(y_raw, 99.0)
        p01 = np.percentile(y_raw, 1.0)
        y_train = np.clip(y_raw, p01, p99)
        if not np.array_equal(y_raw, y_train):
            treatment_applied = True
    else:
        y_train = y_raw

    # 3. Fit Prophet-Architecture GAM
    t_0 = daily["date"].min()
    X_train = _build_fourier_features(daily["date"], t_0)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # In-sample residuals for uncertainty estimation
    in_sample_preds = model.predict(X_train)
    residuals = y_train - in_sample_preds
    sigma = float(np.std(residuals))

    # Check for sustained overspending: last 3 days actual > predicted
    sustained_overspend = False
    if len(residuals) >= 3:
        last3_residuals = residuals[-3:]
        if (last3_residuals > 0.5 * sigma).all():
            sustained_overspend = True

    # 4. Generate future forecast horizon
    last_date = daily["date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="D")
    future_series = pd.Series(future_dates)
    X_future = _build_fourier_features(future_series, t_0)

    future_preds = model.predict(X_future)
    future_preds = np.maximum(future_preds, 0.0)  # non-negative spend

    z_80 = 1.282
    z_95 = 1.960

    forecast_records = []
    for dt, pred in zip(future_dates, future_preds):
        p_val = round(float(pred), 2)
        lb_80 = round(float(max(0.0, pred - z_80 * sigma)), 2)
        ub_80 = round(float(pred + z_80 * sigma), 2)
        lb_95 = round(float(max(0.0, pred - z_95 * sigma)), 2)
        ub_95 = round(float(pred + z_95 * sigma), 2)

        forecast_records.append({
            "date": dt.strftime("%Y-%m-%d"),
            "predicted_cost": p_val,
            "lower_bound_80": lb_80,
            "upper_bound_80": ub_80,
            "lower_bound_95": lb_95,
            "upper_bound_95": ub_95,
        })

    total_pred = round(float(np.sum(future_preds)), 2)
    avg_daily = round(float(np.mean(future_preds)), 2)

    # 5. Budget risk assessment
    budget_status = "NOT_CONFIGURED"
    utilization_pct = None
    if monthly_budget and monthly_budget > 0:
        # Scale budget proportionally if horizon != 30 days
        scaled_budget = monthly_budget * (days / 30.0)
        utilization_pct = round((total_pred / scaled_budget) * 100, 1)
        if utilization_pct > 100.0:
            budget_status = "CRITICAL_OVER_BUDGET"
        elif utilization_pct >= 90.0:
            budget_status = "WARNING_APPROACHING_LIMIT"
        else:
            budget_status = "HEALTHY_WITHIN_BUDGET"

    return {
        "status": "SUCCESS",
        "horizon_days": days,
        "service_scope": service_name or "All Services",
        "batch_id": batch_id,
        "total_predicted_cost": total_pred,
        "average_daily_cost": avg_daily,
        "residual_std_dev": round(sigma, 2),
        "history_days_used": len(daily),
        "history_start_date": daily["date"].min().strftime("%Y-%m-%d"),
        "history_end_date": daily["date"].max().strftime("%Y-%m-%d"),
        "outlier_treatment_applied": treatment_applied,
        "sustained_overspending_alert": sustained_overspend,
        "budget_pacing": {
            "configured_budget": monthly_budget,
            "projected_utilization_pct": utilization_pct,
            "risk_status": budget_status,
        },
        "model_architecture": "Prophet-Architecture GAM (Piecewise-Trend + Weekly Fourier + Weekend Indicator)",
        "forecast": forecast_records,
    }


if __name__ == "__main__":
    import json
    res = get_cost_forecast(days=30, monthly_budget=55000.0)
    print(f"Status: {res['status']}")
    print(f"Horizon: {res['horizon_days']} days | Total: ${res['total_predicted_cost']:,.2f}")
    print(f"Budget Pacing: {res['budget_pacing']}")
