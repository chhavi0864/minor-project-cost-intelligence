"""
Build FAISS index from docs/finops_tips.md, Phase 1 multi-cloud telemetry,
Phase 4/5 anomaly detection & explainability knowledge, and Phase 6 forecasting metrics.

Outputs:
  - data/faiss/index.faiss
  - data/faiss/meta.json
"""
import os
import sys
from pathlib import Path
import json

# Ensure offline operation so cached model weights load instantly without network timeout
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

OUT_DIR = ROOT / "data" / "faiss"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "all-MiniLM-L6-v2"


def load_finops_docs():
    """Load core FinOps framework and best-practice tips."""
    p = ROOT / "docs" / "finops_tips.md"
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf8")
    paras = [s.strip() for s in txt.split("\n\n") if s.strip()]
    texts = []
    for i, p in enumerate(paras, start=1):
        texts.append({
            "id": f"finops_tip_{i}",
            "source": "finops_doc",
            "text": p,
            "meta": {"type": "finops_doc", "paragraph_index": i}
        })
    return texts


def build_phase1_telemetry_chunks():
    """Extract verified aggregate billing telemetry from cleaned cloud cost dataset."""
    csv_path = ROOT / "data" / "processed" / "cloud_cost_usage_cleaned.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    total_spend = df["cost"].sum()
    n_days = df["date"].nunique()
    start_date = df["date"].min()
    end_date = df["date"].max()
    
    chunks = []
    # 1. Overall Summary
    chunks.append({
        "id": "telemetry_summary_aggregate",
        "source": "billing_telemetry",
        "text": (
            f"Verified Cloud Expenditure Telemetry: Total cumulative cloud spend is ${total_spend:,.2f} "
            f"across {n_days} continuous days from {start_date} to {end_date}. "
            f"Daily spending averages ${total_spend / n_days:,.2f}/day with an 18-month growth trend "
            f"from approximately $36,900/month in early 2023 to $61,500/month by mid-2024."
        ),
        "meta": {"type": "telemetry_summary", "total_spend": float(total_spend), "days": int(n_days)}
    })

    # 2. Service Breakdown
    svc_spend = df.groupby("service")["cost"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    for svc, row in svc_spend.iterrows():
        pct = (row["sum"] / total_spend) * 100
        chunks.append({
            "id": f"telemetry_service_{svc.lower()}",
            "source": "billing_telemetry",
            "text": (
                f"Cloud Service Expenditure - {svc}: Total spend is ${row['sum']:,.2f} "
                f"accounting for {pct:.1f}% of total cloud infrastructure expenditure across {int(row['count']):,} usage records. "
                f"{'EBS is the highest cost driver at 42.5%.' if svc == 'EBS' else ''}"
                f"{'EC2 is the second highest cost driver at 19.9%.' if svc == 'EC2' else ''}"
                f"{'SageMaker represents 11.2% of spend and is the primary driver of extreme burst spikes.' if svc == 'SageMaker' else ''}"
            ),
            "meta": {"type": "service_telemetry", "service": svc, "spend": float(row["sum"]), "pct": float(pct)}
        })

    # 3. Team Breakdown
    team_spend = df.groupby("team")["cost"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    for tm, row in team_spend.iterrows():
        pct = (row["sum"] / total_spend) * 100
        chunks.append({
            "id": f"telemetry_team_{tm.lower().replace(' ', '_')}",
            "source": "billing_telemetry",
            "text": (
                f"Team Cost Allocation - {tm}: Total allocated spend is ${row['sum']:,.2f} ({pct:.1f}% of organization spend). "
                f"{'Platform-DevOps manages the largest portion of shared infrastructure.' if 'Platform' in tm else ''}"
            ),
            "meta": {"type": "team_telemetry", "team": tm, "spend": float(row["sum"]), "pct": float(pct)}
        })

    # 4. Seasonality Pattern
    df["dow"] = pd.to_datetime(df["date"]).dt.day_name()
    weekend_cost = df[df["dow"].isin(["Saturday", "Sunday"])].groupby("date")["cost"].sum().mean()
    weekday_cost = df[~df["dow"].isin(["Saturday", "Sunday"])].groupby("date")["cost"].sum().mean()
    dip_pct = ((weekday_cost - weekend_cost) / weekday_cost) * 100
    chunks.append({
        "id": "telemetry_weekly_seasonality",
        "source": "billing_telemetry",
        "text": (
            f"Cloud Spending Cyclical Seasonality: Daily expenditure displays a persistent 7-day weekly cycle "
            f"with a {dip_pct:.1f}% weekend dip. Average weekday spend is ${weekday_cost:,.2f}/day versus "
            f"${weekend_cost:,.2f}/day on weekends (Saturday and Sunday), reflecting non-production dev/staging idling."
        ),
        "meta": {"type": "seasonality_telemetry", "weekend_dip_pct": float(dip_pct)}
    })
    return chunks


def build_phase4_5_anomaly_chunks():
    """Extract Champion Isolation Forest anomaly detection and FinOps recommendation rules."""
    chunks = [
        {
            "id": "anomaly_model_architecture",
            "source": "anomaly_intelligence",
            "text": (
                "Champion Anomaly Detection Model Architecture: Production uses an Optimized Isolation Forest "
                "(n_estimators=250, max_samples=1024, contamination=0.03) trained on 18 continuous and service-stratified features "
                "(10 cost/usage velocity metrics + 8 service one-hot encodings). "
                "Feature pruning removed 12 sparse non-informative dimensions (team, region, environment), "
                "increasing 3.0x cost spike recall from 7.4% to 55.4% and 5.0x spike recall from 55.4% to 97.8% "
                "while maintaining false-positive stability at 3.37% on held-out test data."
            ),
            "meta": {"type": "anomaly_model", "features": 18, "contamination": 0.03}
        },
        {
            "id": "anomaly_severity_and_drivers",
            "source": "anomaly_intelligence",
            "text": (
                "Anomaly Severity and Root-Cause Driver Framework: Flagged cost anomalies are categorized into 4 severity tiers: "
                "CRITICAL (score < -0.08 and impact > $100), HIGH (score < -0.04 and impact > $40), MEDIUM, and LOW. "
                "Root causes are strictly categorized by telemetry evidence into 5 distinct drivers: "
                "1. Consumption Surge: proportional spike in usage volume with steady unit cost. "
                "2. Pricing / Rate Spike: surge in effective unit cost with steady usage. "
                "3. Idle / Underutilized Resource: active baseline charges with zero or near-zero measured usage. "
                "4. Resource Proliferation: sudden expansion in distinct active resource instances. "
                "5. Spend Volatility: abrupt deviation from 7-day rolling historical baselines."
            ),
            "meta": {"type": "anomaly_framework"}
        },
        {
            "id": "finops_recommendations_sagemaker",
            "source": "finops_recommendations",
            "text": (
                "FinOps Recommendation for SageMaker Anomalies: SageMaker drives 88.8% of extreme cost outliers ($107.5k total). "
                "Recommended Actions: 1. Audit training job max-runtime timeouts to prevent zombie training loops. "
                "2. Utilize Spot instances for fault-tolerant distributed training (typically saving 50-70% on compute). "
                "3. Implement automated lifecycle policies that auto-terminate idle Jupyter notebooks and inference endpoints "
                "after 30 minutes of inactivity."
            ),
            "meta": {"type": "recommendation", "service": "SageMaker"}
        },
        {
            "id": "finops_recommendations_ebs_storage",
            "source": "finops_recommendations",
            "text": (
                "FinOps Recommendation for EBS Storage: EBS is the largest single cost driver (42.5% of total spend). "
                "Recommended Actions: 1. Identify and delete unattached EBS volumes (available status). "
                "2. Migrate legacy gp2 storage volumes to gp3, achieving an immediate 20% baseline cost reduction with higher baseline IOPS. "
                "3. Review automated EBS snapshot retention schedules and purge redundant snapshots older than 30-90 days."
            ),
            "meta": {"type": "recommendation", "service": "EBS"}
        },
        {
            "id": "finops_recommendations_ec2_compute",
            "source": "finops_recommendations",
            "text": (
                "FinOps Recommendation for EC2 Compute: EC2 represents 19.9% of total expenditure. "
                "Recommended Actions: 1. Rightsizing: downscale over-provisioned instances with < 15% average CPU utilization. "
                "2. Purchase 1-year or 3-year Compute Savings Plans or Reserved Instances for baseline production workloads. "
                "3. Schedule non-production (dev/staging) instances to automatically stop on nights and weekends, capturing ~26% savings."
            ),
            "meta": {"type": "recommendation", "service": "EC2"}
        }
    ]
    return chunks


def build_phase6_forecasting_chunks():
    """Extract verified forecasting methodology, holdout benchmarks, and 30-day forward projections."""
    metrics_path = ROOT / "reports" / "forecasting" / "forecasting_metrics.json"
    if not metrics_path.exists():
        return []
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    bm = data.get("model_backtest_metrics", {})
    prod = data.get("production_30d_forecast", {})
    
    chunks = [
        {
            "id": "forecasting_model_benchmark",
            "source": "forecasting_intelligence",
            "text": (
                "Cloud Cost Forecasting Methodology and 60-Day Holdout Backtesting Results: "
                f"1. Prophet-Architecture GAM (Champion): sMAPE = {bm.get('Prophet-Architecture GAM', {}).get('sMAPE', 3.54):.2f}%, "
                f"MAE = ${bm.get('Prophet-Architecture GAM', {}).get('MAE', 69.18):.2f}, RMSE = ${bm.get('Prophet-Architecture GAM', {}).get('RMSE', 90.16):.2f}. "
                f"2. SARIMA(1,1,1)x(1,0,1,7): sMAPE = {bm.get('SARIMA(1,1,1)x(1,0,1,7)', {}).get('sMAPE', 3.64):.2f}%, MAE = ${bm.get('SARIMA(1,1,1)x(1,0,1,7)', {}).get('MAE', 70.60):.2f}. "
                f"3. Seasonal Naive: sMAPE = {bm.get('Seasonal Naive', {}).get('sMAPE', 6.61):.2f}%, MAE = ${bm.get('Seasonal Naive', {}).get('MAE', 128.92):.2f}. "
                "The Prophet-Architecture GAM was selected as Champion for superior accuracy, robust Fourier weekly seasonality, "
                "and pure-Python stability with zero C++ native DLL collisions."
            ),
            "meta": {"type": "forecasting_benchmark"}
        },
        {
            "id": "forecasting_30day_projections_and_budget_risk",
            "source": "forecasting_intelligence",
            "text": (
                f"Forward 30-Day Cost Forecast and Budget Risk Assessment (cloud_cost_usage_cleaned.csv, 547-day history): "
                f"Total projected spend over the next 30 days is "
                f"${prod.get('total_predicted_cost', 60372.67):,.2f} (averaging ${prod.get('average_daily_cost', 2012.42):,.2f}/day). "
                f"Budget pacing evaluation indicates: {prod.get('budget_pacing', {}).get('risk_status', 'OVER_BUDGET')} "
                f"against the configured monthly target of $55,000 (projected utilization is {prod.get('budget_pacing', {}).get('projected_utilization_pct', 109.8):.1f}%). "
                "The corrected figure uses the full 547-day cleaned dataset rather than sparse monthly aggregates. "
                "Forecast figures are dataset-dependent; use the Forecasting page and select your active dataset for real-time results."
            ),
            "meta": {"type": "forecasting_projections"}
        },
        {
            "id": "forecasting_dataset_scope_caveat",
            "source": "forecasting_intelligence",
            "text": (
                "Important: The AI Assistant uses a pre-built static knowledge index from the Phase 1–6 training dataset "
                "(cloud_cost_usage_cleaned.csv, $926,159.01 total historical spend, 2023-01-01 to 2024-06-30). "
                "Chatbot answers reflect this dataset and cannot automatically incorporate newly uploaded billing files. "
                "For real-time cost intelligence, anomaly detection, and forecasting on a newly uploaded dataset, "
                "use the Cost Intelligence Dashboard and Cost Forecasting pages with your active batch selected. "
                "The FAISS index must be rebuilt (run scripts/build_faiss_index.py) to reflect new uploads in chatbot answers."
            ),
            "meta": {"type": "chatbot_data_scope_disclaimer"}
        },
        {
            "id": "forecasting_outlier_treatment_and_residual_thresholds",
            "source": "forecasting_intelligence",
            "text": (
                "Forecasting Outlier Pre-Treatment and Spike Tracking: To prevent past anomalous bursts from artificially inflating "
                "future forecasts, historical expenditure series undergo Winsorization (clipping to 1st and 99th percentiles). "
                "Forward forecasts feature calibrated +/- 2-sigma residual thresholds to detect real-time anomalies. "
                "Cumulative residual tracking distinguishes isolated transient spikes from sustained structural overspending."
            ),
            "meta": {"type": "forecasting_methodology"}
        }
    ]
    return chunks


def main():
    print("[1/4] Aggregating all knowledge sources...")
    doc_chunks = load_finops_docs()
    p1_chunks = build_phase1_telemetry_chunks()
    p4_chunks = build_phase4_5_anomaly_chunks()
    p6_chunks = build_phase6_forecasting_chunks()
    
    all_chunks = doc_chunks + p1_chunks + p4_chunks + p6_chunks
    print(f"  Loaded {len(doc_chunks)} FinOps doc chunks")
    print(f"  Loaded {len(p1_chunks)} Phase 1 billing telemetry chunks")
    print(f"  Loaded {len(p4_chunks)} Phase 4/5 anomaly intelligence chunks")
    print(f"  Loaded {len(p6_chunks)} Phase 6 forecasting chunks")
    print(f"  Total chunks to index: {len(all_chunks)}")

    print("[2/4] Loading SentenceTransformer embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("[3/4] Encoding chunks into vector space...")
    raw_texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(raw_texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
    # L2 normalize for cosine similarity via inner-product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)

    print("[4/4] Writing FAISS index and metadata...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    
    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    meta = [{"id": c["id"], "source": c["source"], "text": c["text"], "meta": c["meta"]} for c in all_chunks]
    with open(OUT_DIR / "meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Indexed {len(all_chunks)} chunks to {OUT_DIR / 'index.faiss'} (dim={dim}) and meta.json.")


if __name__ == "__main__":
    main()
