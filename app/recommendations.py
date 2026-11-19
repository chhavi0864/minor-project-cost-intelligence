"""
Cost optimization recommendations module.
Provides functions to detect:
- Idle/underutilized resources
- Sudden unit cost increases 
- Tagging gaps causing unknown costs

Each function returns recommendations with:
- Impacted resources
- Estimated monthly savings
- Action items
"""
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy import select
from app.models import engine, cost_records
from app.analytics import enrich_billing, load_tables

class Recommendation:
    def __init__(self, type: str, resources: List[Dict], estimated_savings: float, actions: List[str]):
        self.type = type
        self.resources = resources
        self.estimated_savings = estimated_savings
        self.actions = actions
        
    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "resources": self.resources,
            "estimated_monthly_savings": round(self.estimated_savings, 2),
            "recommended_actions": self.actions
        }


def _load_recommendation_data(
    batch_id: Optional[str] = None,
    all_batches: bool = False,
) -> pd.DataFrame:
    """Load legacy data or normalized records for the selected dataset scope."""
    if not batch_id and not all_batches:
        return enrich_billing()

    stmt = select(
        cost_records.c.billing_period.label("invoice_month"),
        cost_records.c.resource_id,
        cost_records.c.usage_quantity.label("usage_qty"),
        cost_records.c.billed_cost.label("cost"),
        cost_records.c.tags_json,
    )
    if batch_id:
        stmt = stmt.where(cost_records.c.import_batch_id == batch_id)
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)

    if df.empty:
        return pd.DataFrame(columns=[
            "invoice_month", "resource_id", "usage_qty", "unit_cost",
            "cost", "owner", "env",
        ])

    df["usage_qty"] = pd.to_numeric(df["usage_qty"], errors="coerce").fillna(0.0)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)
    df["unit_cost"] = (
        df["cost"] / df["usage_qty"].where(df["usage_qty"] != 0)
    ).fillna(0.0)

    def parse_tags(raw_tags: Any) -> Dict[str, Any]:
        try:
            return {
                str(key).lower(): value
                for key, value in json.loads(raw_tags or "{}").items()
            }
        except (AttributeError, TypeError, ValueError):
            return {}

    tags = df["tags_json"].map(parse_tags)
    df["owner"] = tags.map(lambda values: str(values.get("owner") or "unassigned"))
    df["env"] = tags.map(
        lambda values: str(values.get("environment") or values.get("env") or "unassigned")
    )
    return df


def find_idle_resources(
    usage_threshold: float = 0.1,
    cost_threshold: float = 100,
    batch_id: Optional[str] = None,
    all_batches: bool = False,
) -> List[Recommendation]:
    """
    Detects potentially idle resources based on:
    - Low usage (< usage_threshold) over last month
    - Monthly cost above cost_threshold
    
    Returns recommendations with estimated savings (70% of current cost)
    """
    df = _load_recommendation_data(batch_id, all_batches)

    if df.empty:
        return []
    
    # Get the most recent month
    latest_month = df["invoice_month"].max()
    
    # Focus on latest month's data
    recent_df = df[df["invoice_month"] == latest_month].copy()
    
    # Calculate utilization where possible
    recent_df["utilization"] = recent_df["usage_qty"] / recent_df["usage_qty"].max() if "usage_qty" in recent_df.columns else 1
    
    # Find potentially idle resources
    idle = recent_df[
        (recent_df["utilization"] < usage_threshold) & 
        (recent_df["cost"] > cost_threshold)
    ]
    
    if idle.empty:
        return []
        
    # Group by resource for summary
    idle_summary = idle.groupby(["resource_id", "owner", "env"]).agg({
        "cost": "sum",
        "utilization": "mean"
    }).reset_index()
    
    # Estimate savings (assume 70% cost reduction from optimization/termination)
    total_savings = idle_summary["cost"].sum() * 0.7
    
    # Build resource details
    resources = []
    for _, row in idle_summary.iterrows():
        resources.append({
            "resource_id": row["resource_id"],
            "owner": row["owner"],
            "environment": row["env"],
            "current_monthly_cost": round(row["cost"], 2),
            "utilization": round(row["utilization"] * 100, 1),
            "potential_savings": round(row["cost"] * 0.7, 2)
        })
    
    actions = [
        "Review and terminate resources with 0% utilization",
        "Right-size resources with low utilization",
        "Implement auto-scaling where applicable",
        "Enable automated start/stop schedules for non-production resources"
    ]
    
    return [Recommendation(
        type="idle_resources",
        resources=resources,
        estimated_savings=total_savings,
        actions=actions
    )]

def find_cost_spikes(
    threshold_pct: float = 0.3,
    batch_id: Optional[str] = None,
    all_batches: bool = False,
) -> List[Recommendation]:
    """
    Detects resources with sudden unit cost increases:
    - Unit cost increase > threshold_pct between consecutive months
    - Focuses on significant cost impact
    
    Returns recommendations for cost investigation and potential actions
    """
    df = _load_recommendation_data(batch_id, all_batches)

    if df.empty:
        return []
    
    if "unit_cost" not in df.columns:
        return []
    
    # Calculate month-over-month unit cost changes
    agg = df.groupby(["resource_id", "invoice_month", "owner", "env"])["unit_cost"].mean().reset_index()
    agg = agg.sort_values(["resource_id", "invoice_month"])
    
    # Calculate percentage change
    agg["prev_unit_cost"] = agg.groupby("resource_id")["unit_cost"].shift(1)
    agg = agg.dropna(subset=["prev_unit_cost"])
    
    # Avoid divide by zero
    agg = agg[agg["prev_unit_cost"] != 0]
    agg["pct_change"] = (agg["unit_cost"] - agg["prev_unit_cost"]) / agg["prev_unit_cost"]
    
    # Filter significant increases
    spikes = agg[agg["pct_change"] >= threshold_pct].copy()
    
    if spikes.empty:
        return []
    
    # Calculate cost impact
    latest_costs = df[df["invoice_month"] == df["invoice_month"].max()]
    spikes = spikes.merge(latest_costs[["resource_id", "cost"]], on="resource_id", how="left")
    
    # Estimate potential savings (assume 50% recovery through optimization)
    total_savings = (spikes["cost"] * spikes["pct_change"]).sum() * 0.5
    
    resources = []
    for _, row in spikes.iterrows():
        resources.append({
            "resource_id": row["resource_id"],
            "owner": row["owner"],
            "environment": row["env"],
            "unit_cost_increase": f"{round(row['pct_change'] * 100, 1)}%",
            "current_monthly_cost": round(row["cost"], 2),
            "potential_savings": round(row["cost"] * row["pct_change"] * 0.5, 2)
        })
    
    actions = [
        "Investigate recent configuration changes",
        "Review resource pricing tier changes",
        "Check for unexpected usage patterns",
        "Consider moving to reserved instances or savings plans",
        "Evaluate alternative service options"
    ]
    
    return [Recommendation(
        type="cost_spikes",
        resources=resources,
        estimated_savings=total_savings,
        actions=actions
    )]

def find_tagging_gaps(
    batch_id: Optional[str] = None,
    all_batches: bool = False,
) -> List[Recommendation]:
    """
    Detects resources with missing tags causing unknown costs:
    - Resources without owner/environment tags
    - Significant costs in 'unassigned' category
    
    Returns recommendations for improving tag coverage
    """
    df = _load_recommendation_data(batch_id, all_batches)

    if df.empty:
        return []
    latest_month = df["invoice_month"].max()
    recent_df = df[df["invoice_month"] == latest_month].copy()
    
    # Find resources without proper tags
    untagged = recent_df[
        (recent_df["owner"] == "unassigned") | 
        (recent_df["env"] == "unassigned")
    ]
    
    if untagged.empty:
        return []
    
    # Group by resource
    untagged_summary = untagged.groupby("resource_id").agg({
        "cost": "sum",
        "owner": lambda x: "missing" if all(x == "unassigned") else "partial",
        "env": lambda x: "missing" if all(x == "unassigned") else "partial"
    }).reset_index()
    
    # Estimate savings (assume 20% through better allocation and optimization)
    total_savings = untagged_summary["cost"].sum() * 0.2
    
    resources = []
    for _, row in untagged_summary.iterrows():
        resources.append({
            "resource_id": row["resource_id"],
            "owner_tag": row["owner"],
            "environment_tag": row["env"],
            "monthly_unattributed_cost": round(row["cost"], 2),
            "potential_savings": round(row["cost"] * 0.2, 2)
        })
    
    actions = [
        "Implement mandatory tagging policy",
        "Add missing owner tags",
        "Add missing environment tags",
        "Set up automated tag compliance checks",
        "Create tag inheritance rules where applicable"
    ]
    
    return [Recommendation(
        type="tagging_gaps",
        resources=resources,
        estimated_savings=total_savings,
        actions=actions
    )]

def get_all_recommendations(
    batch_id: Optional[str] = None,
    all_batches: bool = False,
    usage_threshold: float = 0.1,
    cost_threshold: float = 100,
    spike_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Get all available cost optimization recommendations.
    Returns a dictionary with:
    - Total estimated savings
    - List of recommendations by type
    """
    all_recs = []
    all_recs.extend(find_idle_resources(
        usage_threshold=usage_threshold,
        cost_threshold=cost_threshold,
        batch_id=batch_id,
        all_batches=all_batches,
    ))
    all_recs.extend(find_cost_spikes(
        threshold_pct=spike_threshold,
        batch_id=batch_id,
        all_batches=all_batches,
    ))
    all_recs.extend(find_tagging_gaps(batch_id=batch_id, all_batches=all_batches))
    
    total_savings = sum(r.estimated_savings for r in all_recs)
    
    return {
        "total_estimated_monthly_savings": round(total_savings, 2),
        "recommendations": [r.to_dict() for r in all_recs]
    }