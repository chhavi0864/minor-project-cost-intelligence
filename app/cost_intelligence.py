"""
Cost Intelligence Analytics Module (Phase 1)
Calculates multi-cloud cost KPIs, category distributions, expenditure trends,
and top resource drivers directly from standardized cost_records table.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from sqlalchemy import select, func, desc
from app.models import engine, cost_records, ingestion_batches


def get_available_batches() -> List[Dict[str, Any]]:
    """Fetch all ingested batches with high-level summary info."""
    with engine.connect() as conn:
        stmt = select(ingestion_batches).order_by(ingestion_batches.c.created_at.desc())
        return [dict(r) for r in conn.execute(stmt).mappings().all()]


def get_filter_options(batch_id: Optional[str] = None) -> Dict[str, List[str]]:
    """Return available billing periods, cloud providers, and categories."""
    with engine.connect() as conn:
        # Base filter condition
        cond = True if not batch_id else (cost_records.c.import_batch_id == batch_id)

        # Billing periods
        p_stmt = select(cost_records.c.billing_period).where(cond).distinct().order_by(cost_records.c.billing_period)
        periods = [r[0] for r in conn.execute(p_stmt).fetchall() if r[0]]

        # Cloud providers
        c_stmt = select(cost_records.c.cloud_provider).where(cond).distinct().order_by(cost_records.c.cloud_provider)
        providers = [r[0] for r in conn.execute(c_stmt).fetchall() if r[0]]

        # Categories
        cat_stmt = select(cost_records.c.service_category).where(cond).distinct().order_by(cost_records.c.service_category)
        categories = [r[0] for r in conn.execute(cat_stmt).fetchall() if r[0]]

        return {
            "billing_periods": periods,
            "cloud_providers": providers,
            "service_categories": categories,
        }


def get_cost_summary(
    batch_id: Optional[str] = None,
    billing_period: Optional[str] = None,
    cloud_provider: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate primary financial and operational KPIs for the selected scope.
    """
    with engine.connect() as conn:
        conditions = []
        if batch_id:
            conditions.append(cost_records.c.import_batch_id == batch_id)
        if billing_period and billing_period != "All Periods":
            conditions.append(cost_records.c.billing_period == billing_period)
        if cloud_provider and cloud_provider != "All Providers":
            conditions.append(cost_records.c.cloud_provider == cloud_provider)

        where_clause = True if not conditions else (conditions[0] if len(conditions) == 1 else conditions[0] & conditions[1])
        if len(conditions) > 2:
            where_clause = conditions[0] & conditions[1] & conditions[2]

        # 1. Total Spend, Records, Services, Resources
        base_stmt = select(
            func.coalesce(func.sum(cost_records.c.billed_cost), 0.0).label("total_cost"),
            func.count(cost_records.c.id).label("total_records"),
            func.count(func.distinct(cost_records.c.service_name)).label("service_count"),
            func.count(func.distinct(cost_records.c.resource_id)).label("resource_count"),
        ).where(where_clause)

        row = conn.execute(base_stmt).mappings().first()
        total_cost = round(float(row["total_cost"]), 2)
        total_records = int(row["total_records"])
        service_count = int(row["service_count"])
        resource_count = int(row["resource_count"])

        # 2. Currency
        curr_stmt = select(cost_records.c.currency).where(where_clause).limit(1)
        curr_row = conn.execute(curr_stmt).fetchone()
        currency = curr_row[0] if curr_row else "USD"

        # 3. Highest-Cost Service
        top_svc_stmt = select(
            cost_records.c.service_name,
            func.sum(cost_records.c.billed_cost).label("svc_cost")
        ).where(where_clause).group_by(cost_records.c.service_name).order_by(desc("svc_cost")).limit(1)

        top_svc_row = conn.execute(top_svc_stmt).mappings().first()
        if top_svc_row and total_cost > 0:
            top_svc_cost = round(float(top_svc_row["svc_cost"]), 2)
            top_svc = {
                "name": top_svc_row["service_name"],
                "cost": top_svc_cost,
                "share_pct": round((top_svc_cost / total_cost) * 100, 1),
            }
        else:
            top_svc = {"name": "None", "cost": 0.0, "share_pct": 0.0}

        return {
            "total_expenditure": total_cost,
            "total_records": total_records,
            "service_count": service_count,
            "resource_count": resource_count,
            "highest_cost_service": top_svc,
            "currency": currency,
            "selected_period": billing_period or "All Periods",
            "selected_provider": cloud_provider or "All Providers",
        }


def get_service_cost_breakdown(
    batch_id: Optional[str] = None,
    billing_period: Optional[str] = None,
    cloud_provider: Optional[str] = None,
    limit: int = 12
) -> List[Dict[str, Any]]:
    """
    Service-wise cost breakdown sorted by expenditure descending.
    """
    with engine.connect() as conn:
        conditions = []
        if batch_id:
            conditions.append(cost_records.c.import_batch_id == batch_id)
        if billing_period and billing_period != "All Periods":
            conditions.append(cost_records.c.billing_period == billing_period)
        if cloud_provider and cloud_provider != "All Providers":
            conditions.append(cost_records.c.cloud_provider == cloud_provider)

        where_clause = True if not conditions else (conditions[0] if len(conditions) == 1 else conditions[0] & conditions[1])
        if len(conditions) > 2:
            where_clause = conditions[0] & conditions[1] & conditions[2]

        stmt = select(
            cost_records.c.service_name,
            cost_records.c.service_category,
            func.sum(cost_records.c.billed_cost).label("cost"),
            func.count(cost_records.c.id).label("record_count")
        ).where(where_clause).group_by(
            cost_records.c.service_name, cost_records.c.service_category
        ).order_by(desc("cost")).limit(limit)

        rows = conn.execute(stmt).mappings().all()
        # Compute total cost for share calculation
        total_stmt = select(func.coalesce(func.sum(cost_records.c.billed_cost), 0.0)).where(where_clause)
        total_cost = float(conn.execute(total_stmt).scalar() or 0.0)

        results = []
        for r in rows:
            c = round(float(r["cost"]), 2)
            results.append({
                "service_name": r["service_name"],
                "service_category": r["service_category"],
                "cost": c,
                "record_count": int(r["record_count"]),
                "share_pct": round((c / total_cost * 100), 1) if total_cost > 0 else 0.0
            })
        return results


def get_category_distribution(
    batch_id: Optional[str] = None,
    billing_period: Optional[str] = None,
    cloud_provider: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Cost distribution by standardized category (Compute, Database, Storage, Networking, Other).
    """
    with engine.connect() as conn:
        conditions = []
        if batch_id:
            conditions.append(cost_records.c.import_batch_id == batch_id)
        if billing_period and billing_period != "All Periods":
            conditions.append(cost_records.c.billing_period == billing_period)
        if cloud_provider and cloud_provider != "All Providers":
            conditions.append(cost_records.c.cloud_provider == cloud_provider)

        where_clause = True if not conditions else (conditions[0] if len(conditions) == 1 else conditions[0] & conditions[1])
        if len(conditions) > 2:
            where_clause = conditions[0] & conditions[1] & conditions[2]

        stmt = select(
            cost_records.c.service_category,
            func.sum(cost_records.c.billed_cost).label("cost"),
            func.count(cost_records.c.id).label("record_count")
        ).where(where_clause).group_by(cost_records.c.service_category).order_by(desc("cost"))

        rows = conn.execute(stmt).mappings().all()
        total_stmt = select(func.coalesce(func.sum(cost_records.c.billed_cost), 0.0)).where(where_clause)
        total_cost = float(conn.execute(total_stmt).scalar() or 0.0)

        return [
            {
                "category": r["service_category"],
                "cost": round(float(r["cost"]), 2),
                "record_count": int(r["record_count"]),
                "share_pct": round((float(r["cost"]) / total_cost * 100), 1) if total_cost > 0 else 0.0,
            }
            for r in rows
        ]


def get_expenditure_trends(
    batch_id: Optional[str] = None,
    cloud_provider: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Expenditure trend over time.
    If daily timestamps are available, aggregates daily; otherwise aggregates by billing_period.
    """
    with engine.connect() as conn:
        conditions = []
        if batch_id:
            conditions.append(cost_records.c.import_batch_id == batch_id)
        if cloud_provider and cloud_provider != "All Providers":
            conditions.append(cost_records.c.cloud_provider == cloud_provider)

        where_clause = True if not conditions else (conditions[0] if len(conditions) == 1 else conditions[0] & conditions[1])

        # Test if we have distinct dates within charge_period_start
        sample_stmt = select(cost_records.c.charge_period_start).where(where_clause).limit(10)
        sample_dates = [r[0] for r in conn.execute(sample_stmt).fetchall() if r[0]]

        has_daily = any(len(str(d)) >= 10 for d in sample_dates)

        if has_daily:
            # Group by substr of charge_period_start YYYY-MM-DD
            date_expr = func.substr(cost_records.c.charge_period_start, 1, 10).label("date_label")
        else:
            date_expr = cost_records.c.billing_period.label("date_label")

        stmt = select(
            date_expr,
            cost_records.c.service_category,
            func.sum(cost_records.c.billed_cost).label("cost")
        ).where(where_clause).group_by(
            date_expr, cost_records.c.service_category
        ).order_by("date_label")

        rows = conn.execute(stmt).mappings().all()
        return [
            {
                "date": r["date_label"],
                "category": r["service_category"],
                "cost": round(float(r["cost"]), 2),
            }
            for r in rows
        ]


def get_top_resources(
    batch_id: Optional[str] = None,
    billing_period: Optional[str] = None,
    cloud_provider: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Top highest-cost resources with service, category, region, and cost.
    """
    with engine.connect() as conn:
        conditions = [cost_records.c.resource_id.isnot(None), cost_records.c.resource_id != ""]
        if batch_id:
            conditions.append(cost_records.c.import_batch_id == batch_id)
        if billing_period and billing_period != "All Periods":
            conditions.append(cost_records.c.billing_period == billing_period)
        if cloud_provider and cloud_provider != "All Providers":
            conditions.append(cost_records.c.cloud_provider == cloud_provider)

        where_clause = conditions[0]
        for c in conditions[1:]:
            where_clause = where_clause & c

        stmt = select(
            cost_records.c.resource_id,
            cost_records.c.service_name,
            cost_records.c.service_category,
            cost_records.c.region,
            func.sum(cost_records.c.billed_cost).label("cost"),
            func.sum(cost_records.c.usage_quantity).label("usage_qty"),
            func.max(cost_records.c.usage_unit).label("usage_unit"),
        ).where(where_clause).group_by(
            cost_records.c.resource_id,
            cost_records.c.service_name,
            cost_records.c.service_category,
            cost_records.c.region,
        ).order_by(desc("cost")).limit(limit)

        rows = conn.execute(stmt).mappings().all()
        return [
            {
                "resource_id": r["resource_id"],
                "service_name": r["service_name"],
                "service_category": r["service_category"],
                "region": r["region"],
                "cost": round(float(r["cost"]), 2),
                "usage_quantity": round(float(r["usage_qty"] or 0.0), 2),
                "usage_unit": r["usage_unit"] or "",
            }
            for r in rows
        ]


def get_provider_distribution(
    batch_id: Optional[str] = None,
    billing_period: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Cost distribution across cloud providers (AWS, Microsoft, Google, Oracle, etc.).
    """
    with engine.connect() as conn:
        conditions = []
        if batch_id:
            conditions.append(cost_records.c.import_batch_id == batch_id)
        if billing_period and billing_period != "All Periods":
            conditions.append(cost_records.c.billing_period == billing_period)

        where_clause = True if not conditions else (conditions[0] if len(conditions) == 1 else conditions[0] & conditions[1])

        stmt = select(
            cost_records.c.cloud_provider,
            func.sum(cost_records.c.billed_cost).label("cost"),
            func.count(cost_records.c.id).label("record_count")
        ).where(where_clause).group_by(cost_records.c.cloud_provider).order_by(desc("cost"))

        rows = conn.execute(stmt).mappings().all()
        total_stmt = select(func.coalesce(func.sum(cost_records.c.billed_cost), 0.0)).where(where_clause)
        total_cost = float(conn.execute(total_stmt).scalar() or 0.0)

        return [
            {
                "provider": r["cloud_provider"],
                "cost": round(float(r["cost"]), 2),
                "record_count": int(r["record_count"]),
                "share_pct": round((float(r["cost"]) / total_cost * 100), 1) if total_cost > 0 else 0.0,
            }
            for r in rows
        ]

