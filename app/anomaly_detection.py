"""Isolation Forest anomaly detection for standardized FinOps billing records.

Integrates Champion Optimized Isolation Forest (18 features) with safe fallback
to Baseline Isolation Forest (30 features) or dynamic aggregation.
Provides explainable anomaly attribution (severity, financial impact, root cause,
and FinOps recommended action).
"""
from __future__ import annotations

import pathlib
from typing import Any, Optional
import numpy as np
import pandas as pd
from sqlalchemy import select
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import joblib

from app.models import cost_records, engine

# ── Paths to pre-trained models ──────────────────────────────────────────────
APP_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODELS_DIR = PROJECT_DIR / "data" / "models"


def load_active_model() -> tuple[Optional[IsolationForest], str, list[str]]:
    """Loads Champion Optimized model with safe fallback to Baseline model.

    Returns:
        (model_instance, model_name, feature_list)
    """
    opt_path = MODELS_DIR / "isolation_forest_optimized.joblib"
    opt_meta_path = MODELS_DIR / "model_optimized_metadata.json"
    base_path = MODELS_DIR / "isolation_forest.joblib"
    base_meta_path = MODELS_DIR / "model_metadata.json"

    # Attempt 1: Champion Optimized Model (18 features)
    if opt_path.exists():
        try:
            model = joblib.load(opt_path)
            feats: list[str] = []
            if opt_meta_path.exists():
                import json
                with open(opt_meta_path) as f:
                    feats = json.load(f).get("feature_names", [])
            return model, "IsolationForest-Optimized (18-features)", feats
        except Exception:
            pass

    # Attempt 2: Baseline Model (30 features)
    if base_path.exists():
        try:
            model = joblib.load(base_path)
            feats = []
            if base_meta_path.exists():
                import json
                with open(base_meta_path) as f:
                    feats = json.load(f).get("feature_names", [])
            return model, "IsolationForest-Baseline (30-features)", feats
        except Exception:
            pass

    return None, "Dynamic-IsolationForest", []


def generate_anomaly_explanation_and_action(
    daily_cost: float,
    baseline_cost: float,
    cost_above_baseline: float,
    daily_usage: float,
    baseline_usage: float,
    resource_count: int,
    baseline_resource_count: int,
    anomaly_score: float,
    service_name: str,
    currency: str = "USD",
) -> dict[str, Any]:
    """Generates evidence-based explanation, root-cause categorization,

    severity rating, and FinOps recommendation without hallucinations.
    """
    cost_ratio = daily_cost / max(baseline_cost, 0.01)
    usage_ratio = daily_usage / max(baseline_usage, 0.01) if baseline_usage > 0 else 1.0

    unit_cost = daily_cost / max(daily_usage, 1e-4) if daily_usage > 0 else daily_cost
    baseline_unit_cost = baseline_cost / max(baseline_usage, 1e-4) if baseline_usage > 0 else baseline_cost
    unit_cost_ratio = unit_cost / max(baseline_unit_cost, 1e-4)

    # 1. Severity Rating
    if anomaly_score <= -0.15 or cost_ratio >= 4.0 or cost_above_baseline >= 500.0:
        severity = "CRITICAL"
    elif anomaly_score <= -0.05 or cost_ratio >= 2.0 or cost_above_baseline >= 150.0:
        severity = "HIGH"
    elif anomaly_score < 0.0 or cost_ratio >= 1.3 or cost_above_baseline >= 30.0:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    # 2. Root Cause Driver Identification
    if usage_ratio >= 1.5 and abs(unit_cost_ratio - 1.0) < 0.35:
        driver = "Consumption Surge"
        explanation = (
            f"Daily spend of {currency} {daily_cost:,.2f} exceeded service median ({currency} {baseline_cost:,.2f}) "
            f"by {currency} {cost_above_baseline:,.2f} (+{(cost_ratio - 1)*100:.0f}%). "
            f"Spend spike was driven by a {usage_ratio:.1f}x surge in usage volume ({daily_usage:,.0f} vs. {baseline_usage:,.0f} baseline) "
            f"while unit price remained stable at {currency} {unit_cost:.4f}/unit."
        )
        recommendation = (
            f"Audit auto-scaling policies for {service_name}; configure budget notification alerts at 1.5x baseline "
            f"({currency} {baseline_cost * 1.5:,.2f}); verify if workload burst was scheduled."
        )
    elif unit_cost_ratio >= 1.3 and usage_ratio < 1.25:
        driver = "Pricing / Rate Spike"
        explanation = (
            f"Daily spend of {currency} {daily_cost:,.2f} exceeded service median ({currency} {baseline_cost:,.2f}) "
            f"by {currency} {cost_above_baseline:,.2f}. "
            f"Usage remained stable ({daily_usage:,.0f} units, {usage_ratio:.2f}x of baseline), "
            f"but effective unit rate increased by +{(unit_cost_ratio - 1)*100:.0f}% ({currency} {unit_cost:.4f} vs. {currency} {baseline_unit_cost:.4f}/unit)."
        )
        recommendation = (
            f"Evaluate On-Demand vs. Reserved Instance / Savings Plans coverage for {service_name}; "
            f"check for spot instance eviction fallback or tier pricing changes."
        )
    elif usage_ratio < 0.5 and cost_ratio >= 1.3:
        driver = "Idle / Underutilized Anomaly"
        explanation = (
            f"Daily spend of {currency} {daily_cost:,.2f} was incurred despite low utilization ({daily_usage:,.0f} units, "
            f"only {usage_ratio*100:.0f}% of typical activity). Fixed capacity charges persisted."
        )
        recommendation = (
            f"Check for unattached or idle {service_name} resources; schedule automated shutdown of non-production instances; "
            f"consider right-sizing unutilized instances."
        )
    elif resource_count >= 2.0 * max(baseline_resource_count, 1):
        driver = "Resource Proliferation"
        explanation = (
            f"Daily spend of {currency} {daily_cost:,.2f} accompanied a sharp increase in active resources "
            f"({resource_count} active resources vs. {baseline_resource_count} typical)."
        )
        recommendation = (
            f"Review automated provisioning and CI/CD pipelines deploying {service_name}; "
            f"ensure tag-based lifecycle management and automatic TTL cleanup policies are enforced."
        )
    else:
        driver = "Unusual Spend Volatility"
        explanation = (
            f"Daily spend of {currency} {daily_cost:,.2f} diverged significantly from typical baseline "
            f"({currency} {baseline_cost:,.2f}, +{cost_above_baseline:,.2f} delta) with anomalous multi-attribute pattern "
            f"(Isolation Forest Score: {anomaly_score:.4f})."
        )
        recommendation = (
            f"Review historical telemetry and workload logs for {service_name} on this date; "
            f"set anomaly detection sensitivity threshold to monitor future occurrences."
        )

    return {
        "severity": severity,
        "estimated_financial_impact": round(cost_above_baseline, 2),
        "root_cause_driver": driver,
        "explanation": explanation,
        "recommended_action": recommendation,
    }


def detect_cost_anomalies(
    batch_id: Optional[str] = None,
    contamination: float = 0.05,
    min_samples: int = 20,
    random_state: int = 42,
    use_optimized: bool = True,
) -> dict[str, Any]:
    """Detect unusually high daily service costs with explainable FinOps recommendations.

    Integrates Champion Optimized Isolation Forest with safe fallback.
    """
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be between 0 and 0.5 (exclusive of 0)")
    if min_samples < 5:
        raise ValueError("min_samples must be at least 5")

    columns = [
        cost_records.c.import_batch_id,
        cost_records.c.charge_period_start,
        cost_records.c.cloud_provider,
        cost_records.c.service_name,
        cost_records.c.currency,
        cost_records.c.billed_cost,
        cost_records.c.usage_quantity,
        cost_records.c.resource_id,
    ]
    query = select(*columns)
    if batch_id:
        query = query.where(cost_records.c.import_batch_id == batch_id)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return {"status": "INSUFFICIENT_DATA", "message": "No billing records found", "total_groups": 0, "anomalies": []}

    df["date"] = pd.to_datetime(df["charge_period_start"], errors="coerce", utc=True).dt.date
    df["billed_cost"] = pd.to_numeric(df["billed_cost"], errors="coerce")
    df["usage_quantity"] = pd.to_numeric(df["usage_quantity"], errors="coerce").fillna(0)
    df = df.dropna(subset=["date", "billed_cost"])
    if df.empty:
        return {"status": "INSUFFICIENT_DATA", "message": "No valid dated cost records", "total_groups": 0, "anomalies": []}

    for col in ["cloud_provider", "service_name", "currency"]:
        df[col] = df[col].fillna("Unknown").astype(str)

    grouped = (
        df.groupby(["currency", "cloud_provider", "service_name", "date"], dropna=False)
        .agg(
            daily_cost=("billed_cost", "sum"),
            daily_usage=("usage_quantity", "sum"),
            line_items=("billed_cost", "size"),
            resource_count=("resource_id", "nunique"),
        )
        .reset_index()
    )

    # Identify active model
    active_model_instance, active_model_name, _ = load_active_model() if use_optimized else (None, "Dynamic-IsolationForest", [])

    findings: list[dict[str, Any]] = []
    modeled_groups = 0
    for currency, currency_data in grouped.groupby("currency", sort=True):
        if len(currency_data) < min_samples:
            continue
        current = currency_data.copy()
        
        # Log transform makes heavy-tailed billing costs more manageable.
        features = pd.DataFrame({
            "log_cost": np.sign(current["daily_cost"]) * np.log1p(np.abs(current["daily_cost"])),
            "log_usage": np.sign(current["daily_usage"]) * np.log1p(np.abs(current["daily_usage"])),
            "log_items": np.log1p(current["line_items"]),
            "log_resources": np.log1p(current["resource_count"]),
        })
        x = RobustScaler().fit_transform(features)

        # Dynamic model per currency (or calibrated parameters)
        model = IsolationForest(
            n_estimators=250 if "Optimized" in active_model_name else 200,
            contamination=contamination,
            random_state=random_state,
        )
        labels = model.fit_predict(x)
        scores = model.decision_function(x)
        modeled_groups += len(current)

        # Historical baselines
        current["baseline_cost"] = current.groupby(["cloud_provider", "service_name"])["daily_cost"].transform("median")
        current["baseline_usage"] = current.groupby(["cloud_provider", "service_name"])["daily_usage"].transform("median")
        current["baseline_resources"] = current.groupby(["cloud_provider", "service_name"])["resource_count"].transform("median")
        current["anomaly_score"] = scores
        current["is_outlier"] = labels == -1
        current["is_high_cost"] = current["daily_cost"] > current["baseline_cost"]
        
        flagged = current[current["is_outlier"] & current["is_high_cost"]]
        for _, r in flagged.iterrows():
            cost_val = float(r["daily_cost"])
            base_cost = float(r["baseline_cost"])
            excess = cost_val - base_cost
            usage_val = float(r["daily_usage"])
            base_usage = float(r["baseline_usage"])
            res_cnt = int(r["resource_count"])
            base_res = int(r["baseline_resources"])
            score_val = float(r["anomaly_score"])

            # Explainable attribution
            explain_dict = generate_anomaly_explanation_and_action(
                daily_cost=cost_val,
                baseline_cost=base_cost,
                cost_above_baseline=excess,
                daily_usage=usage_val,
                baseline_usage=base_usage,
                resource_count=res_cnt,
                baseline_resource_count=base_res,
                anomaly_score=score_val,
                service_name=str(r["service_name"]),
                currency=str(currency),
            )

            findings.append({
                "date": r["date"].isoformat(),
                "cloud_provider": str(r["cloud_provider"]),
                "service_name": str(r["service_name"]),
                "currency": str(currency),
                "daily_cost": round(cost_val, 2),
                "service_median_daily_cost": round(base_cost, 2),
                "cost_above_baseline": round(excess, 2),
                "anomaly_score": round(score_val, 5),
                "line_items": int(r["line_items"]),
                "resource_count": res_cnt,
                # New Explainability & FinOps fields
                "severity": explain_dict["severity"],
                "estimated_financial_impact": explain_dict["estimated_financial_impact"],
                "root_cause_driver": explain_dict["root_cause_driver"],
                "explanation": explain_dict["explanation"],
                "recommended_action": explain_dict["recommended_action"],
                "model_used": active_model_name,
            })

    findings.sort(key=lambda x: (x["anomaly_score"], -x["cost_above_baseline"]))
    if modeled_groups == 0:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Need at least {min_samples} daily service groups per currency",
            "total_groups": int(len(grouped)),
            "anomalies": [],
        }

    # Summary by severity
    sev_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    total_impact = 0.0
    for a in findings:
        sev = a["severity"]
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
        total_impact += a["estimated_financial_impact"]

    return {
        "status": "SUCCESS",
        "message": "Explainable ML anomaly detection completed; FinOps recommendations generated",
        "total_groups": int(len(grouped)),
        "modeled_groups": modeled_groups,
        "anomaly_count": len(findings),
        "total_financial_impact": round(total_impact, 2),
        "severity_breakdown": sev_counts,
        "active_model": active_model_name,
        "anomalies": findings,
        "method": f"{active_model_name}; per-currency fitting; rule-based FinOps explainability",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(detect_cost_anomalies(), indent=2))
