from fastapi import FastAPI, Query, HTTPException, Request, UploadFile, File
import sys, os
from typing import Optional, List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from app.analytics import (
    unit_cost_changes, six_month_trend, top_n_cost_drivers,
    monthly_cost_by_owner, monthly_cost_by_env, owner_coverage,
    load_month_results, cache_month_results
)
from app.models import engine, resources
from pydantic import BaseModel
from app.rag_qa import answer as rag_answer
from app.recommendations import get_all_recommendations
from app.validators import validate_request
from app.universal_etl import ingest_universal_csv, list_ingestion_batches
from app.cost_intelligence import (
    get_cost_summary, get_service_cost_breakdown, get_category_distribution,
    get_expenditure_trends, get_top_resources, get_provider_distribution,
    get_filter_options
)

app = FastAPI(
    title="Cloud Compass API",
    description="Cloud cost intelligence and optimization API with FinOps and RAG-powered insights",
    version="1.0.0"
)

# No monitoring setup or middleware needed

@app.get("/debug-rag")
def debug_rag():
    """Diagnostic: returns the computed meta.json path and whether it exists."""
    import glob
    this_file = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(this_file))
    meta_path = os.path.join(repo_root, "data", "faiss", "meta.json")
    # Also scan /var/task for the file
    found_in = glob.glob("/var/task/**/meta.json", recursive=True) if os.path.exists("/var/task") else []
    cwd_found = glob.glob(os.path.join(os.getcwd(), "**", "meta.json"), recursive=True)
    return {
        "__file__": this_file,
        "repo_root": repo_root,
        "meta_path": meta_path,
        "meta_exists": os.path.exists(meta_path),
        "cwd": os.getcwd(),
        "found_at": found_in + cwd_found,
    }

@app.get("/kpi")
def get_kpis(
    month: str = Query(..., description="Month in YYYY-MM format"),
    refresh: bool = Query(False, description="Force refresh and rebuild cache")
):
    """
    Returns cost KPIs for a given month.
    Uses cache unless refresh=true is passed.
    """
    if not refresh:
        cached = load_month_results(month)
    else:
        cached = None

    if cached:
        owner_df, env_df, coverage = cached
    else:
        owner_df = monthly_cost_by_owner(month)
        env_df = monthly_cost_by_env(month)
        coverage = owner_coverage(month)
        cache_month_results(month)

    return {
        "month": month,
        "cost_by_owner": owner_df.to_dict(orient="records"),
        "cost_by_env": env_df.to_dict(orient="records"),
        "owner_coverage": coverage
    }
@app.get("/kpi/months")
def get_available_months():
    """
    Returns all distinct months present in the billing table
    (for use in frontend dropdowns)
    """
    try:
        df = pd.read_sql_query("SELECT DISTINCT invoice_month FROM billing", engine)
        if df.empty:
            return {"months": [], "message": "No billing data found in database"}
        months = sorted(df["invoice_month"].astype(str).tolist())
        return {"months": months}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching months: {str(e)}"
        )

@app.get("/kpi/top")
def get_top_cost_drivers(
    month: str = Query(..., description="Month in YYYY-MM format"),
    n: int = Query(10, description="Number of top cost drivers to return")
):
    """
    Returns Top N cost-driving resources for the given month.
    """
    df = top_n_cost_drivers(month, n)
    return {
        "month": month,
        "top_n": n,
        "results": df.to_dict(orient="records")
    }
    
@app.get("/kpi/unit-changes")
def get_unit_cost_changes(
    threshold: float = Query(0.2, description="Percentage threshold (e.g. 0.2 = 20%)")
):
    """
    Returns all resources whose unit cost changed more than the threshold
    between consecutive months.
    """
    df = unit_cost_changes(threshold)
    return {
        "threshold": threshold,
        "results": df.to_dict(orient="records")
    }
    
@app.get("/kpi/trend")
def get_six_month_trend(
    group_by: str = Query("owner", description="Group by 'owner' or 'env'")
):
    """
    Returns 6-month cost trend grouped by owner or env.
    """
    df = six_month_trend(group_by)
    # pivot table: months as index, group values as columns
    return {
        "group_by": group_by,
        "trend": df.reset_index().to_dict(orient="records")
    }
    
@app.get("/kpi/quality-checks")
def run_quality_checks():
    """
    Performs basic data quality checks.
    """
    import pandas as pd
    from app.models import engine

    billing = pd.read_sql_query("SELECT * FROM billing", engine)
    resources = pd.read_sql_query("SELECT * FROM resources", engine)

    issues = []

    # 1) Null checks
    nulls = billing[billing[["invoice_month","resource_id","cost"]].isnull().any(axis=1)]
    if not nulls.empty:
        issues.append({"type": "null_values", "count": len(nulls)})

    # 2) Negative cost
    neg = billing[billing["cost"] < 0]
    if not neg.empty:
        issues.append({"type": "negative_costs", "count": len(neg)})

    # 3) Duplicate resource_ids in resources
    dup = resources["resource_id"].duplicated().sum()
    if dup > 0:
        issues.append({"type": "duplicate_resource_ids", "count": int(dup)})

    return {"issues": issues or "No issues found"}

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    model: str | None = None   # optional Groq model override
    batch_id: str | None = None  # optional active dataset batch scoping

@app.get("/recommendations")
def get_recommendations(
    usage_threshold: float = Query(0.1, description="Usage threshold for idle detection (0-1)"),
    cost_threshold: float = Query(100, description="Minimum monthly cost for recommendations ($)"),
    spike_threshold: float = Query(0.3, description="Unit cost increase threshold (e.g. 0.3 = 30%)"),
    batch_id: Optional[str] = Query(None, description="Optional dataset batch ID filter"),
    all_batches: bool = Query(False, description="Use normalized cost records across all ingested datasets"),
):
    """
    Get cost optimization recommendations including:
    - Idle/underutilized resources
    - Resources with sudden unit cost increases
    - Resources missing tags causing unknown costs
    
    Returns estimated savings and specific actions for each recommendation.
    """
    try:
        return get_all_recommendations(
            batch_id=batch_id,
            all_batches=all_batches,
            usage_threshold=usage_threshold,
            cost_threshold=cost_threshold,
            spike_threshold=spike_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(req: AskRequest):
    try:
        # Validate and sanitize input
        try:
            validated_data = validate_request(req.dict())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Check if knowledge-base meta.json exists (resolved relative to repo root, not CWD)
        faiss_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "faiss")
        meta_path = os.path.join(faiss_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise HTTPException(
                status_code=500,
                detail=f"Knowledge base meta.json not found at: {meta_path}"
            )
            
        res = await rag_answer(
            validated_data["question"],
            top_k=validated_data["top_k"],
            groq_model=validated_data.get("model"),
            batch_id=validated_data.get("batch_id")
        )
        
        if res.get("error"):
            raise HTTPException(status_code=500, detail=res["error"])
            
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Universal Cost Intelligence API Routes
# ==========================================

@app.post("/api/v1/upload-billing")
async def upload_billing_csv(
    file: UploadFile = File(...),
    force_reimport: bool = Query(False, description="Force re-import even if file was already ingested")
):
    """
    Universal Cloud Cost CSV upload endpoint.
    Supports FOCUS 1.0, Legacy FinOps, and custom cloud billing exports.
    Validates, normalizes, prevents duplicate ingestion, and loads into SQLite.
    """
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Only CSV or text billing files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = ingest_universal_csv(
        file_source=content,
        filename=file.filename,
        force_reimport=force_reimport
    )

    if result["status"] == "ERROR":
        raise HTTPException(status_code=500, detail=result["message"])
    elif result["status"] == "VALIDATION_FAILED":
        raise HTTPException(status_code=422, detail=result)

    return result

@app.get("/api/v1/batches")
def get_ingested_batches():
    """Retrieve all ingested billing batches with file hashes and metadata."""
    return {"batches": list_ingestion_batches()}

@app.get("/api/v1/cost-intelligence/summary")
def get_summary(
    batch_id: Optional[str] = Query(None, description="Optional batch ID filter"),
    billing_period: Optional[str] = Query(None, description="Optional billing period (YYYY-MM)"),
    cloud_provider: Optional[str] = Query(None, description="Optional cloud provider filter")
):
    """Primary financial KPIs: Total Spend, Record Count, Services, Resources, Top Service."""
    b_id = batch_id if isinstance(batch_id, str) and batch_id != "None" else None
    b_period = billing_period if isinstance(billing_period, str) and billing_period != "None" else None
    c_provider = cloud_provider if isinstance(cloud_provider, str) and cloud_provider != "None" else None
    return get_cost_summary(batch_id=b_id, billing_period=b_period, cloud_provider=c_provider)

@app.get("/api/v1/cost-intelligence/by-service")
def get_by_service(
    batch_id: Optional[str] = Query(None),
    billing_period: Optional[str] = Query(None),
    cloud_provider: Optional[str] = Query(None),
    limit: int = Query(15, ge=1, le=100)
):
    """Service-wise expenditure breakdown."""
    return {"services": get_service_cost_breakdown(batch_id, billing_period, cloud_provider, limit)}

@app.get("/api/v1/cost-intelligence/by-category")
def get_by_category(
    batch_id: Optional[str] = Query(None),
    billing_period: Optional[str] = Query(None),
    cloud_provider: Optional[str] = Query(None)
):
    """Cost distribution across Compute, Database, Storage, Networking, and Other."""
    return {"categories": get_category_distribution(batch_id, billing_period, cloud_provider)}

@app.get("/api/v1/cost-intelligence/trends")
def get_trends(
    batch_id: Optional[str] = Query(None),
    cloud_provider: Optional[str] = Query(None)
):
    """Daily or monthly expenditure trends over time."""
    return {"trends": get_expenditure_trends(batch_id, cloud_provider)}

@app.get("/api/v1/cost-intelligence/top-resources")
def get_top_cost_resources(
    batch_id: Optional[str] = Query(None),
    billing_period: Optional[str] = Query(None),
    cloud_provider: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50)
):
    """Top highest-cost resources."""
    return {"resources": get_top_resources(batch_id, billing_period, cloud_provider, limit)}

@app.get("/api/v1/cost-intelligence/by-provider")
def get_by_provider(
    batch_id: Optional[str] = Query(None),
    billing_period: Optional[str] = Query(None)
):
    """Cloud provider cost distribution."""
    return {"providers": get_provider_distribution(batch_id, billing_period)}

@app.get("/api/v1/cost-intelligence/filters")
def get_filters(batch_id: Optional[str] = Query(None)):
    """Available filter options for dropdowns."""
    return get_filter_options(batch_id)


# ---------------------------------------------------------------------------
# Phase 2: Anomaly Detection Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/anomalies")
def get_anomalies(
    batch_id: Optional[str] = Query(None, description="Optional batch ID filter"),
    contamination: float = Query(0.03, ge=0.01, le=0.5, description="Expected anomaly rate (0.01–0.5)"),
    limit: int = Query(50, ge=1, le=500, description="Max anomalies to return"),
):
    """Run Isolation Forest and return flagged high-cost anomalies."""
    from app.anomaly_detection import detect_cost_anomalies
    result = detect_cost_anomalies(batch_id=batch_id, contamination=contamination)
    if result.get("status") == "SUCCESS":
        result["anomalies"] = result["anomalies"][:limit]
    return result


@app.get("/api/v1/anomalies/summary")
def get_anomaly_summary(
    batch_id: Optional[str] = Query(None),
    contamination: float = Query(0.03, ge=0.01, le=0.5),
):
    """Return anomaly counts broken down by service, provider, and severity tier."""
    from app.anomaly_detection import detect_cost_anomalies
    from collections import defaultdict
    result = detect_cost_anomalies(batch_id=batch_id, contamination=contamination)
    if result.get("status") != "SUCCESS":
        return result
    anomalies = result["anomalies"]
    by_service: dict = defaultdict(lambda: {"count": 0, "total_excess_cost": 0.0})
    by_provider: dict = defaultdict(lambda: {"count": 0, "total_excess_cost": 0.0})
    by_severity: dict = defaultdict(lambda: {"count": 0, "total_excess_cost": 0.0})
    total_impact = 0.0

    for a in anomalies:
        s = a["service_name"]
        p = a["cloud_provider"]
        sev = a.get("severity", "MEDIUM")
        excess = a.get("estimated_financial_impact", a.get("cost_above_baseline", 0.0))
        total_impact += excess

        by_service[s]["count"] += 1
        by_service[s]["total_excess_cost"] = round(
            by_service[s]["total_excess_cost"] + excess, 2
        )
        by_provider[p]["count"] += 1
        by_provider[p]["total_excess_cost"] = round(
            by_provider[p]["total_excess_cost"] + excess, 2
        )
        by_severity[sev]["count"] += 1
        by_severity[sev]["total_excess_cost"] = round(
            by_severity[sev]["total_excess_cost"] + excess, 2
        )

    return {
        "status": "SUCCESS",
        "total_anomalies": result["anomaly_count"],
        "total_groups": result["total_groups"],
        "total_financial_impact": round(total_impact, 2),
        "active_model": result.get("active_model", "IsolationForest-Optimized"),
        "by_service": dict(by_service),
        "by_provider": dict(by_provider),
        "by_severity": dict(by_severity),
    }


# ---------------------------------------------------------------------------
# Phase 6: Cloud Cost Forecasting Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/forecast")
def get_forecast(
    days: int = Query(30, ge=1, le=90, description="Forecast horizon in days (1–90)"),
    service_name: Optional[str] = Query(None, description="Optional service filter"),
    batch_id: Optional[str] = Query(None, description="Optional dataset batch ID filter"),
    monthly_budget: Optional[float] = Query(None, description="Optional monthly budget target"),
):
    """Generate cloud cost forecast with uncertainty intervals and budget risk indicators."""
    from app.forecasting import get_cost_forecast
    svc = service_name if isinstance(service_name, str) else None
    b_id = batch_id if isinstance(batch_id, str) else None
    budget = float(monthly_budget) if isinstance(monthly_budget, (int, float)) else None
    d = int(days) if isinstance(days, int) else 30
    return get_cost_forecast(
        days=d,
        service_name=svc,
        batch_id=b_id,
        monthly_budget=budget,
    )
