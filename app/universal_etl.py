"""
Universal Cloud Cost CSV Processing & Ingestion System (Phase 1)
Supports FOCUS 1.0, Legacy FinOps-Copilot CSVs, and Custom Mapped Cloud Billing Exports.
Industry-independent, multi-cloud compatible (AWS, Azure, GCP, OCI).
"""

import os
import io
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from sqlalchemy import select, delete, insert, func

from app.models import engine, cost_records, ingestion_batches, billing

# Allowed standardized service categories
STANDARDIZED_CATEGORIES = ["Compute", "Database", "Storage", "Networking", "Other"]

# Required standardized internal schema fields
STANDARDIZED_COLUMNS = [
    "billing_period",
    "charge_period_start",
    "charge_period_end",
    "cloud_provider",
    "service_name",
    "service_category",
    "resource_id",
    "resource_name",
    "region",
    "usage_quantity",
    "usage_unit",
    "billed_cost",
    "currency",
    "tags_json",
]

# Standard column mappings for known formats
KNOWN_FORMAT_MAPPINGS = {
    "FOCUS_1.0": {
        "billed_cost": "BilledCost",
        "service_name": "ServiceName",
        "service_category": "ServiceCategory",
        "charge_period_start": "ChargePeriodStart",
        "charge_period_end": "ChargePeriodEnd",
        "billing_period": "BillingPeriodStart",
        "cloud_provider": "ProviderName",
        "resource_id": "ResourceId",
        "resource_name": "ResourceName",
        "region": "RegionName",
        "usage_quantity": "ConsumedQuantity",
        "usage_unit": "ConsumedUnit",
        "currency": "BillingCurrency",
        "tags_json": "Tags",
    },
    "LEGACY_FINOPS": {
        "billed_cost": "cost",
        "service_name": "service",
        "service_category": "service",
        "charge_period_start": "invoice_month",
        "charge_period_end": None,
        "billing_period": "invoice_month",
        "cloud_provider": None,
        "resource_id": "resource_id",
        "resource_name": None,
        "region": "region",
        "usage_quantity": "usage_qty",
        "usage_unit": None,
        "currency": None,
        "tags_json": None,
    },
}


def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 hash of raw file bytes for duplicate prevention."""
    return hashlib.sha256(content).hexdigest()


def categorize_service(service_name: str, service_category_raw: Optional[str] = None) -> str:
    """
    Categorize cloud services into Compute, Database, Storage, Networking, or Other.
    Strictly adheres to documented rules: unknown services map to 'Other'.
    """
    s_name = str(service_name or "").strip().lower()
    s_cat = str(service_category_raw or "").strip().lower()

    # 1. If explicit service category is present (e.g. FOCUS)
    if s_cat and s_cat != "nan":
        if "compute" in s_cat or "container" in s_cat:
            return "Compute"
        if "database" in s_cat:
            return "Database"
        if "storage" in s_cat:
            return "Storage"
        if "network" in s_cat:
            return "Networking"

    # 2. Heuristic evaluation of ServiceName
    # Database
    db_keywords = [
        "database", "rds", "dynamodb", "aurora", "redshift", "cosmos",
        "cloud sql", "bigquery", "spanner", "sql database", "elasticache",
        "redis", "mongodb", "cassandra", "opensearch", "firestore", "datastore"
    ]
    if any(kw in s_name for kw in db_keywords):
        return "Database"

    # Compute
    compute_keywords = [
        "compute", "ec2", "elastic compute", "virtual machine", "vm", "instances",
        "lambda", "function", "cloud run", "app service", "eks", "ecs", "aks",
        "gke", "kubernetes", "container", "batch", "fargate"
    ]
    if any(kw in s_name for kw in compute_keywords):
        return "Compute"

    # Storage
    storage_keywords = [
        "storage", "s3", "simple storage", "ebs", "elastic block", "glacier",
        "blob", "cloud storage", "filestore", "disk", "archive", "efs", "backup"
    ]
    if any(kw in s_name for kw in storage_keywords):
        return "Storage"

    # Networking
    network_keywords = [
        "network", "networking", "vpc", "virtual private cloud", "vnet",
        "route 53", "route53", "dns", "cloudfront", "cdn", "direct connect",
        "load balancer", "load balancing", "elb", "alb", "nlb", "gateway",
        "nat", "traffic", "bandwidth", "data transfer"
    ]
    if any(kw in s_name for kw in network_keywords):
        return "Networking"

    # 3. Explicit fallback
    return "Other"


def detect_format(df: pd.DataFrame) -> Tuple[str, Dict[str, str]]:
    """
    Detect whether the DataFrame conforms to FOCUS 1.0, Legacy FinOps, or Custom format.
    Returns (format_name, detected_mapping).
    """
    cols = set(df.columns)

    # Check FOCUS 1.0 signatures
    if "BilledCost" in cols and ("ChargePeriodStart" in cols or "BillingPeriodStart" in cols):
        mapping = dict(KNOWN_FORMAT_MAPPINGS["FOCUS_1.0"])
        # Adjust for RegionId vs RegionName if needed
        if "RegionName" not in cols and "RegionId" in cols:
            mapping["region"] = "RegionId"
        return "FOCUS 1.0", mapping

    # Check Legacy FinOps signature
    if "cost" in cols and "invoice_month" in cols and "service" in cols:
        return "Legacy FinOps", dict(KNOWN_FORMAT_MAPPINGS["LEGACY_FINOPS"])

    # Fallback: Auto-detect closest columns for generic files
    mapping = {}
    col_lower = {str(c).lower().strip(): c for c in df.columns}

    # Billed cost
    for cost_candidate in ["billedcost", "cost", "billed_cost", "totalcost", "amount", "charge"]:
        if cost_candidate in col_lower:
            mapping["billed_cost"] = col_lower[cost_candidate]
            break

    # Service name
    for svc_candidate in ["servicename", "service", "service_name", "product", "productname"]:
        if svc_candidate in col_lower:
            mapping["service_name"] = col_lower[svc_candidate]
            break

    # Service category
    for cat_candidate in ["servicecategory", "service_category", "category"]:
        if cat_candidate in col_lower:
            mapping["service_category"] = col_lower[cat_candidate]
            break

    # Dates
    for date_candidate in ["chargeperiodstart", "billingperiodstart", "invoice_month", "date", "usage_date", "timestamp"]:
        if date_candidate in col_lower:
            mapping["charge_period_start"] = col_lower[date_candidate]
            mapping["billing_period"] = col_lower[date_candidate]
            break

    # Provider
    for prov_candidate in ["providername", "provider", "cloud_provider", "vendor"]:
        if prov_candidate in col_lower:
            mapping["cloud_provider"] = col_lower[prov_candidate]
            break

    # Resource ID
    for res_candidate in ["resourceid", "resource_id", "resourcename", "id"]:
        if res_candidate in col_lower:
            mapping["resource_id"] = col_lower[res_candidate]
            break

    # Region
    for reg_candidate in ["regionname", "regionid", "region", "location"]:
        if reg_candidate in col_lower:
            mapping["region"] = col_lower[reg_candidate]
            break

    # Usage
    for usage_candidate in ["consumedquantity", "usagequantity", "usage_qty", "quantity", "usage"]:
        if usage_candidate in col_lower:
            mapping["usage_quantity"] = col_lower[usage_candidate]
            break

    # Currency
    for curr_candidate in ["billingcurrency", "currency"]:
        if curr_candidate in col_lower:
            mapping["currency"] = col_lower[curr_candidate]
            break

    format_name = "Generic/Mapped" if ("billed_cost" in mapping and "service_name" in mapping) else "Unknown"
    return format_name, mapping


class ValidationResult:
    """Encapsulates validation findings and quality metrics."""
    def __init__(self):
        self.is_valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.format_detected: str = "Unknown"
        self.column_mapping: Dict[str, str] = {}
        self.total_rows: int = 0
        self.null_cost_rows: int = 0
        self.negative_cost_rows: int = 0
        self.duplicate_rows: int = 0
        self.currencies_found: List[str] = []
        self.date_range: Tuple[Optional[str], Optional[str]] = (None, None)
        self.total_expenditure: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "format_detected": self.format_detected,
            "total_rows": self.total_rows,
            "null_cost_rows": self.null_cost_rows,
            "negative_cost_rows": self.negative_cost_rows,
            "duplicate_rows": self.duplicate_rows,
            "currencies_found": self.currencies_found,
            "date_range_start": self.date_range[0],
            "date_range_end": self.date_range[1],
            "total_expenditure": self.total_expenditure,
        }


def validate_billing_dataframe(
    df: pd.DataFrame,
    custom_mapping: Optional[Dict[str, str]] = None
) -> Tuple[ValidationResult, Dict[str, str]]:
    """
    Validate a billing DataFrame against data integrity and standardization rules.
    Does NOT modify data.
    """
    res = ValidationResult()
    res.total_rows = len(df)

    if df.empty:
        res.is_valid = False
        res.errors.append("File is empty (0 rows).")
        return res, {}

    # Detect format and column mapping
    detected_format, detected_mapping = detect_format(df)
    mapping = dict(detected_mapping)
    if custom_mapping:
        mapping.update(custom_mapping)

    res.format_detected = detected_format
    res.column_mapping = mapping

    # Check required minimum fields: cost and service_name
    cost_col = mapping.get("billed_cost")
    if not cost_col or cost_col not in df.columns:
        res.is_valid = False
        res.errors.append("Missing required cost column (e.g., 'BilledCost' or 'cost').")

    service_col = mapping.get("service_name")
    if not service_col or service_col not in df.columns:
        res.is_valid = False
        res.errors.append("Missing required service name column (e.g., 'ServiceName' or 'service').")

    if not res.is_valid:
        return res, mapping

    # Validate cost column values
    cost_series = df[cost_col]
    numeric_cost = pd.to_numeric(cost_series, errors="coerce")
    null_costs = numeric_cost.isna().sum()
    if null_costs > 0:
        res.null_cost_rows = int(null_costs)
        # In strict FinOps validation, missing cost is an error if significant
        res.errors.append(f"{null_costs} row(s) contain missing or non-numeric cost values. Silent zeroing is prohibited.")
        res.is_valid = False

    neg_costs = (numeric_cost < 0).sum()
    if neg_costs > 0:
        res.negative_cost_rows = int(neg_costs)
        res.warnings.append(f"{neg_costs} row(s) have negative cost values (credits/refunds).")

    # Currency verification: Multi-currency blending check
    currency_col = mapping.get("currency")
    if currency_col and currency_col in df.columns:
        currencies = [str(c).strip().upper() for c in df[currency_col].dropna().unique() if str(c).strip()]
        res.currencies_found = sorted(list(set(currencies)))
        if len(res.currencies_found) > 1:
            res.is_valid = False
            res.errors.append(
                f"Multiple billing currencies detected: {res.currencies_found}. "
                "In accordance with FinOps financial integrity standards, multi-currency blending without explicit exchange rate conversion is rejected."
            )
    else:
        res.currencies_found = ["USD (Default)"]
        res.warnings.append("No explicit currency column found; defaulting to USD.")

    # Date range verification
    date_col = mapping.get("charge_period_start") or mapping.get("billing_period")
    if date_col and date_col in df.columns:
        try:
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
            valid_dates = parsed_dates.dropna()
            if not valid_dates.empty:
                res.date_range = (
                    valid_dates.min().strftime("%Y-%m-%d"),
                    valid_dates.max().strftime("%Y-%m-%d")
                )
            if valid_dates.isna().sum() > 0:
                res.warnings.append(f"{parsed_dates.isna().sum()} row(s) have unparseable date values.")
        except Exception:
            res.warnings.append("Could not parse date column into standard timestamps.")

    # Duplicate row check inside the file
    res.duplicate_rows = int(df.duplicated().sum())
    if res.duplicate_rows > 0:
        res.warnings.append(f"{res.duplicate_rows} duplicate row(s) detected within the file.")

    # Calculate total expenditure
    if res.is_valid and not numeric_cost.empty:
        res.total_expenditure = round(float(numeric_cost.sum()), 4)

    return res, mapping


def normalize_billing_dataframe(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    batch_id: str
) -> pd.DataFrame:
    """
    Standardize raw billing DataFrame into the canonical cost_records schema.
    """
    norm = pd.DataFrame()
    norm["import_batch_id"] = [batch_id] * len(df)

    # 1. Billed Cost
    cost_col = mapping["billed_cost"]
    norm["billed_cost"] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0).round(4)

    # 2. Service Name
    svc_col = mapping["service_name"]
    norm["service_name"] = df[svc_col].astype(str).str.strip().replace("nan", "Unknown Service")

    # 3. Service Category (Normalized via documented rules)
    cat_col = mapping.get("service_category")
    raw_categories = df[cat_col] if (cat_col and cat_col in df.columns) else [None] * len(df)
    norm["service_category"] = [
        categorize_service(svc, cat)
        for svc, cat in zip(norm["service_name"], raw_categories)
    ]

    # 4. Dates & Billing Period (YYYY-MM)
    date_col = mapping.get("charge_period_start") or mapping.get("billing_period")
    if date_col and date_col in df.columns:
        parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
        # Format charge_period_start
        norm["charge_period_start"] = parsed_dates.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("2024-01-01 00:00:00")
        # Format billing_period YYYY-MM
        norm["billing_period"] = parsed_dates.dt.strftime("%Y-%m").fillna("2024-01")
    else:
        now_month = datetime.utcnow().strftime("%Y-%m")
        norm["charge_period_start"] = datetime.utcnow().strftime("%Y-%m-%d 00:00:00")
        norm["billing_period"] = now_month

    # End date if available
    end_date_col = mapping.get("charge_period_end")
    if end_date_col and end_date_col in df.columns:
        norm["charge_period_end"] = pd.to_datetime(df[end_date_col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        norm["charge_period_end"] = None

    # 5. Cloud Provider
    provider_col = mapping.get("cloud_provider")
    if provider_col and provider_col in df.columns:
        norm["cloud_provider"] = df[provider_col].astype(str).str.strip().replace("nan", "Unknown")
    else:
        norm["cloud_provider"] = "Unknown"

    # 6. Resource Identifier & Name
    res_id_col = mapping.get("resource_id")
    if res_id_col and res_id_col in df.columns:
        norm["resource_id"] = df[res_id_col].astype(str).str.strip().replace("nan", None)
    else:
        norm["resource_id"] = None

    res_name_col = mapping.get("resource_name")
    if res_name_col and res_name_col in df.columns:
        norm["resource_name"] = df[res_name_col].astype(str).str.strip().replace("nan", None)
    else:
        norm["resource_name"] = norm["resource_id"]

    # 7. Region
    region_col = mapping.get("region")
    if region_col and region_col in df.columns:
        norm["region"] = df[region_col].astype(str).str.strip().replace("nan", "global")
    else:
        norm["region"] = "global"

    # 8. Usage Quantity & Unit
    usage_col = mapping.get("usage_quantity")
    if usage_col and usage_col in df.columns:
        norm["usage_quantity"] = pd.to_numeric(df[usage_col], errors="coerce").fillna(0.0).round(4)
    else:
        norm["usage_quantity"] = 0.0

    unit_col = mapping.get("usage_unit")
    if unit_col and unit_col in df.columns:
        norm["usage_unit"] = df[unit_col].astype(str).str.strip().replace("nan", None)
    else:
        norm["usage_unit"] = None

    # 9. Currency
    curr_col = mapping.get("currency")
    if curr_col and curr_col in df.columns:
        norm["currency"] = df[curr_col].astype(str).str.strip().str.upper().replace("NAN", "USD")
    else:
        norm["currency"] = "USD"

    # 10. Tags & Raw Metadata
    tags_col = mapping.get("tags_json")
    if tags_col and tags_col in df.columns:
        norm["tags_json"] = df[tags_col].astype(str).replace("nan", "{}")
    else:
        norm["tags_json"] = "{}"

    norm["raw_metadata_json"] = None

    return norm


def check_duplicate_file(file_hash: str) -> Optional[Dict[str, Any]]:
    """Check if file with identical SHA-256 hash was already ingested."""
    with engine.connect() as conn:
        stmt = select(ingestion_batches).where(ingestion_batches.c.file_hash == file_hash)
        row = conn.execute(stmt).mappings().first()
        if row:
            return dict(row)
    return None


def ingest_universal_csv(
    file_source: Union[str, Path, io.BytesIO, bytes],
    filename: str,
    custom_mapping: Optional[Dict[str, str]] = None,
    force_reimport: bool = False,
    sync_to_legacy_billing: bool = True
) -> Dict[str, Any]:
    """
    Ingest a cloud billing CSV file with validation, standardization, and duplicate prevention.
    Returns comprehensive execution metrics and status.
    """
    # 1. Read file bytes and compute hash
    if isinstance(file_source, (str, Path)):
        p = Path(file_source)
        if not p.exists():
            return {"status": "ERROR", "message": f"File not found: {p}"}
        with open(p, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_source, io.BytesIO):
        file_bytes = file_source.getvalue()
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    else:
        return {"status": "ERROR", "message": "Unsupported file source type."}

    file_hash = compute_file_hash(file_bytes)

    # 2. Duplicate Check
    existing_batch = check_duplicate_file(file_hash)
    if existing_batch and not force_reimport:
        return {
            "status": "DUPLICATE",
            "message": (
                f"Duplicate ingestion prevented: File '{filename}' has identical content to "
                f"previously ingested batch '{existing_batch['batch_id']}' (ingested on {existing_batch['created_at']})."
            ),
            "batch_id": existing_batch["batch_id"],
            "existing_batch": existing_batch,
            "total_records": existing_batch["total_records"],
            "total_cost": existing_batch["total_cost"],
        }

    # 3. Read into Pandas DataFrame
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    except Exception as e:
        return {"status": "ERROR", "message": f"Failed to parse CSV: {str(e)}"}

    # 4. Validate DataFrame
    val_res, mapping = validate_billing_dataframe(df, custom_mapping=custom_mapping)
    if not val_res.is_valid:
        return {
            "status": "VALIDATION_FAILED",
            "message": "Validation failed: " + "; ".join(val_res.errors),
            "validation_details": val_res.to_dict(),
        }

    # If force_reimport is active and batch exists, delete previous batch records
    if existing_batch and force_reimport:
        with engine.begin() as conn:
            conn.execute(delete(cost_records).where(cost_records.c.import_batch_id == existing_batch["batch_id"]))
            conn.execute(delete(ingestion_batches).where(ingestion_batches.c.batch_id == existing_batch["batch_id"]))

    batch_id = str(uuid.uuid4())[:8]

    # 5. Normalize Data
    normalized_df = normalize_billing_dataframe(df, mapping, batch_id)

    # 6. Database Insertion within Transaction
    currency = val_res.currencies_found[0] if val_res.currencies_found else "USD"
    total_cost = round(float(normalized_df["billed_cost"].sum()), 4)

    batch_meta = {
        "batch_id": batch_id,
        "filename": filename,
        "file_hash": file_hash,
        "source_format": val_res.format_detected,
        "total_records": len(normalized_df),
        "total_cost": total_cost,
        "currency": currency,
        "date_range_start": val_res.date_range[0],
        "date_range_end": val_res.date_range[1],
        "status": "SUCCESS",
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    try:
        with engine.begin() as conn:
            # Insert batch record
            conn.execute(insert(ingestion_batches).values(batch_meta))

            # Insert normalized cost records
            records_data = normalized_df.to_dict(orient="records")
            conn.execute(insert(cost_records), records_data)

            # Optional: Synchronize to legacy billing table for backward compatibility if needed
            if sync_to_legacy_billing:
                legacy_rows = []
                for _, r in normalized_df.iterrows():
                    legacy_rows.append({
                        "invoice_month": str(r["billing_period"]),
                        "account_id": str(r["cloud_provider"]),
                        "subscription": "standard",
                        "service": str(r["service_category"]),
                        "resource_group": str(r["region"]),
                        "resource_id": str(r["resource_id"] or "unassigned_res"),
                        "region": str(r["region"]),
                        "usage_qty": float(r["usage_quantity"]),
                        "unit_cost": round(float(r["billed_cost"] / max(1.0, r["usage_quantity"])), 4),
                        "cost": float(r["billed_cost"]),
                    })
                # We append legacy rows if desired
                # conn.execute(insert(billing), legacy_rows)

        return {
            "status": "SUCCESS",
            "message": f"Successfully ingested {len(normalized_df)} records (Total Spend: ${total_cost:,.2f} {currency}).",
            "batch_id": batch_id,
            "batch_metadata": batch_meta,
            "validation_details": val_res.to_dict(),
            "preview": normalized_df.head(5).to_dict(orient="records"),
        }
    except Exception as e:
        return {"status": "ERROR", "message": f"Database insertion failed: {str(e)}"}


def list_ingestion_batches() -> List[Dict[str, Any]]:
    """List all ingested batches in reverse chronological order."""
    with engine.connect() as conn:
        stmt = select(ingestion_batches).order_by(ingestion_batches.c.created_at.desc())
        rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]

