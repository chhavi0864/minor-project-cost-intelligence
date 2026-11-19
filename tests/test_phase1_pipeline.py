"""
Test Suite for Phase 1 — Universal Cloud Cost Ingestion & Intelligence Pipeline
Verifies format detection, validation rules, service categorization, duplicate prevention,
database storage, financial reconciliation, and API compatibility.
"""

import os
import io
import pytest
import pandas as pd

from app.universal_etl import (
    validate_billing_dataframe,
    categorize_service,
    detect_format,
    ingest_universal_csv,
    compute_file_hash,
    list_ingestion_batches,
)
from app.cost_intelligence import (
    get_cost_summary,
    get_service_cost_breakdown,
    get_category_distribution,
    get_provider_distribution,
    get_available_batches,
    get_expenditure_trends,
    get_top_resources,
)


def test_focus_format_detection_and_validation():
    """Verify that official FOCUS 1.0 files are correctly identified, validated, and reconciled."""
    focus_path = "FinOps-Copilot/data/raw/focus_sample.csv"
    if not os.path.exists(focus_path):
        focus_path = "data/raw/focus_sample.csv"

    assert os.path.exists(focus_path), f"FOCUS sample dataset not found at {focus_path}"

    df = pd.read_csv(focus_path, low_memory=False)
    val_res, mapping = validate_billing_dataframe(df)

    assert val_res.is_valid is True
    assert val_res.format_detected == "FOCUS 1.0"
    assert val_res.total_rows == 1000
    assert val_res.null_cost_rows == 0
    assert val_res.currencies_found == ["USD"]
    assert val_res.total_expenditure > 0
    # Reconcile against raw dataframe sum
    raw_sum = round(float(df["BilledCost"].sum()), 4)
    assert abs(val_res.total_expenditure - raw_sum) < 1e-3


def test_legacy_format_detection_and_validation():
    """Verify that existing project billing CSVs are correctly recognized."""
    legacy_path = "FinOps-Copilot/data/sample_billing.csv"
    if not os.path.exists(legacy_path):
        legacy_path = "data/sample_billing.csv"

    assert os.path.exists(legacy_path), f"Legacy dataset not found at {legacy_path}"

    df = pd.read_csv(legacy_path)
    val_res, mapping = validate_billing_dataframe(df)

    assert val_res.is_valid is True
    assert val_res.format_detected == "Legacy FinOps"
    assert val_res.total_rows == 600
    assert val_res.null_cost_rows == 0
    raw_sum = round(float(df["cost"].sum()), 4)
    assert abs(val_res.total_expenditure - raw_sum) < 1e-3


def test_validation_rejects_empty_file():
    """Verify that empty files fail validation with a clear error message."""
    empty_df = pd.DataFrame()
    val_res, _ = validate_billing_dataframe(empty_df)
    assert val_res.is_valid is False
    assert any("empty" in err.lower() for err in val_res.errors)


def test_validation_rejects_missing_cost_column():
    """Verify that files lacking cost columns are rejected."""
    bad_df = pd.DataFrame({
        "service": ["EC2", "S3"],
        "usage": [10, 20]
    })
    val_res, _ = validate_billing_dataframe(bad_df)
    assert val_res.is_valid is False
    assert any("cost" in err.lower() for err in val_res.errors)


def test_validation_flags_non_numeric_cost():
    """Verify that non-numeric or missing costs fail validation without silent zeroing."""
    bad_cost_df = pd.DataFrame({
        "ServiceName": ["EC2", "RDS", "S3"],
        "BilledCost": [12.50, "invalid_cost", None],
        "ChargePeriodStart": ["2024-09-01", "2024-09-02", "2024-09-03"]
    })
    val_res, _ = validate_billing_dataframe(bad_cost_df)
    assert val_res.is_valid is False
    assert val_res.null_cost_rows == 2
    assert any("missing or non-numeric" in err.lower() for err in val_res.errors)


def test_validation_rejects_multiple_currencies():
    """Verify that multiple billing currencies without conversion are rejected."""
    multi_curr_df = pd.DataFrame({
        "ServiceName": ["EC2", "S3"],
        "BilledCost": [10.0, 20.0],
        "BillingCurrency": ["USD", "EUR"],
        "ChargePeriodStart": ["2024-09-01", "2024-09-02"]
    })
    val_res, _ = validate_billing_dataframe(multi_curr_df)
    assert val_res.is_valid is False
    assert any("multiple billing currencies" in err.lower() for err in val_res.errors)


def test_service_categorization_documented_rules():
    """Verify that cloud services are mapped into Compute, Database, Storage, Networking, or Other."""
    # Compute
    assert categorize_service("Amazon Elastic Compute Cloud") == "Compute"
    assert categorize_service("Virtual Machines") == "Compute"
    assert categorize_service("AWS Lambda") == "Compute"
    assert categorize_service("Google Kubernetes Engine") == "Compute"

    # Database
    assert categorize_service("Amazon Relational Database Service") == "Database"
    assert categorize_service("Azure Cosmos DB") == "Database"
    assert categorize_service("Google Cloud SQL") == "Database"
    assert categorize_service("Amazon DynamoDB") == "Database"

    # Storage
    assert categorize_service("Amazon Simple Storage Service") == "Storage"
    assert categorize_service("Azure Blob Storage") == "Storage"
    assert categorize_service("Google Cloud Storage") == "Storage"
    assert categorize_service("Elastic Block Store") == "Storage"

    # Networking
    assert categorize_service("Elastic Load Balancing") == "Networking"
    assert categorize_service("Amazon Route 53") == "Networking"
    assert categorize_service("CloudFront") == "Networking"
    assert categorize_service("Virtual Private Cloud") == "Networking"

    # Other (Unknown or Governance)
    assert categorize_service("AmazonCloudWatch") == "Other"
    assert categorize_service("AWS Key Management Service") == "Other"
    assert categorize_service("CompletelyUnknownCustomServiceX") == "Other"


def test_duplicate_ingestion_prevention():
    """Verify that re-uploading an identical CSV file returns DUPLICATE status."""
    dummy_csv = (
        "ChargePeriodStart,ServiceName,BilledCost,BillingCurrency\n"
        "2024-09-01,Amazon Elastic Compute Cloud,15.50,USD\n"
        "2024-09-02,Amazon S3,4.20,USD\n"
    ).encode("utf-8")

    # Ingest first time
    res1 = ingest_universal_csv(dummy_csv, "unit_test_dedup.csv", force_reimport=True)
    assert res1["status"] == "SUCCESS"

    # Attempt to ingest exact same content
    res2 = ingest_universal_csv(dummy_csv, "unit_test_dedup.csv", force_reimport=False)
    assert res2["status"] == "DUPLICATE"
    assert "duplicate ingestion prevented" in res2["message"].lower()


def test_financial_reconciliation():
    """Verify that total expenditure reconciles with service category breakdown."""
    summary = get_cost_summary()
    categories = get_category_distribution()

    total_spend = summary["total_expenditure"]
    cat_sum = round(sum(c["cost"] for c in categories), 2)

    assert abs(total_spend - cat_sum) < 0.05, f"Mismatch: total={total_spend}, cat_sum={cat_sum}"


def test_fastapi_cost_intelligence_endpoints():
    """Verify Phase 1 intelligence functions return valid structure (direct module calls, not HTTP)."""
    # Summary
    summary = get_cost_summary()
    assert "total_expenditure" in summary
    assert "total_records" in summary
    assert "service_count" in summary
    assert isinstance(summary["total_expenditure"], float)
    assert summary["total_records"] >= 1

    # By Service
    services = get_service_cost_breakdown()
    assert isinstance(services, list)
    assert len(services) >= 1
    assert "service_name" in services[0]
    assert "cost" in services[0]

    # By Category
    categories = get_category_distribution()
    assert isinstance(categories, list)
    assert len(categories) >= 1
    assert "category" in categories[0]
    assert "cost" in categories[0]

    # Trends
    trends = get_expenditure_trends()
    assert isinstance(trends, list)

    # Top Resources
    resources = get_top_resources()
    assert isinstance(resources, list)

    # Batches
    batches = list_ingestion_batches()
    assert isinstance(batches, list)
    assert len(batches) >= 1

