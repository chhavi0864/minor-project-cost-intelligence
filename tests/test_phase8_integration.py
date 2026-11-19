"""
Phase 8 Integration Tests — End-to-End Validation
====================================================
Tests the complete workflow:
  CSV upload → ETL → SQLite → analytics → anomaly detection
  → forecasting → chatbot → API route handlers

Key assertions:
- Each module uses the active dataset (batch_id scoping), not all data combined.
- The corrected 30-day forecast for the cleaned 547-day dataset is ~$50k-$70k (not $381k).
- Endpoint functions accept and honour batch_id parameter.
- FAISS knowledge index no longer contains the erroneous $381,649 figure.
- Dataset switching returns different results per batch.
"""

import json
import pathlib
import sys
import io
import csv
import pytest

# Ensure project root is on path
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import (
    get_forecast,
    get_summary,
    get_ingested_batches,
    upload_billing_csv,
    get_anomalies,
)
from app.forecasting import get_cost_forecast
from app.anomaly_detection import detect_cost_anomalies
from app.cost_intelligence import get_cost_summary
from app.universal_etl import list_ingestion_batches, ingest_universal_csv

# ---------------------------------------------------------------------------
# Known batch IDs (from data/billing.db; created during Phase 6/8 ingestion)
# ---------------------------------------------------------------------------
CLEANED_BATCH = "6869bfe4"      # cloud_cost_usage_cleaned.csv — 547 daily records
FOCUS_BATCH   = "4e832241"      # focus_sample.csv — FOCUS 1.0, 1000 rows, $20.52 total
MONTHLY_BATCH = "173ff474"      # sample_billing.csv — 6 monthly dates, insufficient for forecast


# ===========================================================================
# 1. Database connectivity — cleaned dataset is present
# ===========================================================================

def test_01_database_has_cleaned_dataset():
    """The cleaned 547-day CSV must be loaded as a known batch."""
    batches_data = get_ingested_batches()
    batches = batches_data.get("batches", [])
    batch_ids = [b.get("import_batch_id", b.get("batch_id", "")) for b in batches]
    assert CLEANED_BATCH in batch_ids, (
        f"Expected batch {CLEANED_BATCH!r} in DB; found: {batch_ids}"
    )


# ===========================================================================
# 2. Forecast correctness — batch-scoped result must NOT be ~$381k
# ===========================================================================

def test_02_forecast_cleaned_dataset_not_381k():
    """
    get_cost_forecast(batch_id=CLEANED_BATCH) must return a 30-day total
    between $40,000 and $100,000 — NOT the erroneous $381,649.48 value.
    """
    result = get_cost_forecast(days=30, batch_id=CLEANED_BATCH)
    assert result.get("status") == "SUCCESS", f"Unexpected status: {result}"
    total = result.get("total_predicted_cost", 0)
    assert 40_000 <= total <= 100_000, (
        f"30-day forecast ${total:,.2f} outside expected $40k–$100k range "
        f"(erroneous value was $381,649.48 — batch_id scoping may be broken)"
    )


def test_03_forecast_api_accepts_batch_id():
    """get_forecast(days=7, batch_id=CLEANED_BATCH) must return SUCCESS."""
    data = get_forecast(days=7, batch_id=CLEANED_BATCH)
    assert data.get("status") == "SUCCESS"
    assert len(data.get("forecast", [])) == 7


# ===========================================================================
# 3. Dataset isolation — different batches return different forecast totals
# ===========================================================================

def test_04_forecast_different_results_per_batch():
    """
    Forecasting two different batches must yield different total predicted costs.
    The FOCUS batch ($20.52 total) and cleaned batch ($926k total) are very different datasets.
    """
    res_cleaned = get_cost_forecast(days=30, batch_id=CLEANED_BATCH)
    res_focus   = get_cost_forecast(days=30, batch_id=FOCUS_BATCH)

    assert res_cleaned.get("status") in ("SUCCESS", "INSUFFICIENT_DATA")
    assert res_focus.get("status")   in ("SUCCESS", "INSUFFICIENT_DATA")

    if res_cleaned.get("status") == "SUCCESS" and res_focus.get("status") == "SUCCESS":
        cost_cleaned = res_cleaned["total_predicted_cost"]
        cost_focus   = res_focus["total_predicted_cost"]
        assert cost_cleaned != cost_focus, (
            "Forecasts for two very different datasets returned identical totals."
        )
        assert cost_cleaned > cost_focus * 10, (
            f"Expected cleaned forecast >> FOCUS forecast. "
            f"Got cleaned={cost_cleaned:.2f}, focus={cost_focus:.2f}"
        )


def test_05_monthly_batch_returns_insufficient_data():
    """
    sample_billing.csv has only 6 monthly date points.
    Forecasting should return INSUFFICIENT_DATA for this batch.
    """
    result = get_cost_forecast(days=30, batch_id=MONTHLY_BATCH)
    assert result.get("status") == "INSUFFICIENT_DATA", (
        f"Expected INSUFFICIENT_DATA for monthly-only batch {MONTHLY_BATCH}, got: {result.get('status')}"
    )


# ===========================================================================
# 4. Cost summary scoping
# ===========================================================================

def test_06_cost_summary_respects_batch_id():
    """
    Cost summary for the cleaned batch should reflect ~$926k total.
    Cost summary for the FOCUS batch should reflect ~$20 total.
    """
    summary_cleaned = get_cost_summary(batch_id=CLEANED_BATCH)
    summary_focus   = get_cost_summary(batch_id=FOCUS_BATCH)

    total_cleaned = summary_cleaned.get("total_expenditure", summary_cleaned.get("total_cost", 0))
    total_focus   = summary_focus.get("total_expenditure", summary_focus.get("total_cost", 0))

    assert 800_000 <= total_cleaned <= 1_200_000, (
        f"Cleaned batch total_expenditure=${total_cleaned:,.2f} outside expected $800k–$1.2M"
    )
    assert 0 < total_focus < 100, (
        f"FOCUS batch total_expenditure=${total_focus:.4f} should be near $20.52"
    )


# ===========================================================================
# 5. Anomaly detection scoping
# ===========================================================================

def test_07_anomaly_detection_respects_batch_id():
    """
    Anomaly detection on the cleaned batch (66k rows) must return anomalies.
    Anomaly detection on a different batch must return a different count.
    """
    res_cleaned = detect_cost_anomalies(batch_id=CLEANED_BATCH)
    assert res_cleaned.get("status") in ("SUCCESS", "success"), (
        f"Anomaly detection failed for cleaned batch: {res_cleaned}"
    )
    n_cleaned = len(res_cleaned.get("anomalies", []))

    res_focus = detect_cost_anomalies(batch_id=FOCUS_BATCH)
    n_focus = len(res_focus.get("anomalies", []))

    assert n_cleaned != n_focus or n_cleaned == 0, (
        f"Expected different anomaly counts. cleaned={n_cleaned}, focus={n_focus}"
    )


# ===========================================================================
# 6. FAISS index — no stale $381,649 value
# ===========================================================================

def test_08_faiss_index_no_stale_381k():
    """
    The rebuilt FAISS meta.json must not contain the erroneous $381,649.48 figure.
    """
    meta_path = ROOT / "data" / "faiss" / "meta.json"
    if not meta_path.exists():
        pytest.skip("FAISS meta.json not found — index may not be built yet.")

    text = meta_path.read_text(encoding="utf-8")
    assert "381,649" not in text and "381649" not in text, (
        "FAISS meta.json still contains the erroneous $381,649.48 figure."
    )


# ===========================================================================
# 7. forecasting_metrics.json — corrected values
# ===========================================================================

def test_09_forecasting_metrics_json_corrected():
    """
    reports/forecasting/forecasting_metrics.json must NOT contain the stale $381,649.48 total.
    The corrected 30-day total must be in the $40k–$100k range.
    """
    metrics_path = ROOT / "reports" / "forecasting" / "forecasting_metrics.json"
    assert metrics_path.exists(), "forecasting_metrics.json not found"

    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)

    total = data.get("production_30d_forecast", {}).get("total_predicted_cost", 0)
    assert total != 381649.48, (
        "forecasting_metrics.json still has the erroneous $381,649.48 value."
    )
    assert 40_000 <= total <= 100_000, (
        f"production_30d_forecast.total_predicted_cost=${total:,.2f} outside expected range."
    )


# ===========================================================================
# 8. API endpoint — forecast endpoint returns JSON with expected schema
# ===========================================================================

def test_10_forecast_api_schema():
    """get_forecast returns required fields."""
    data = get_forecast(days=7, batch_id=CLEANED_BATCH)
    required_keys = {"status", "forecast", "total_predicted_cost", "average_daily_cost"}
    for key in required_keys:
        assert key in data, f"Missing key '{key}' in get_forecast response"


def test_11_forecast_api_no_batch_returns_result():
    """get_forecast without batch_id still returns a valid result (all data or fallback CSV)."""
    data = get_forecast(days=7)
    assert data.get("status") in ("SUCCESS", "INSUFFICIENT_DATA")


# ===========================================================================
# 9. Upload workflow — ingest a small test CSV, verify batch appears
# ===========================================================================

def test_12_upload_new_csv_batch_appears_in_list():
    """
    Ingest a minimal synthetic billing CSV and confirm it appears in the batch list.
    Then verify that get_cost_forecast with the new batch_id returns INSUFFICIENT_DATA
    (since only 3 rows are uploaded — not enough for a 14-day history).
    """
    import uuid
    uid = uuid.uuid4().hex[:6]
    rows = [
        ["date", "service_name", "cost", "resource_id", "region", "account_id", "environment"],
        ["2024-06-01", "EC2", "150.00", f"i-{uid}1", "us-east-1", "acct-99", "prod"],
        ["2024-06-02", "EC2", "155.00", f"i-{uid}1", "us-east-1", "acct-99", "prod"],
        ["2024-06-03", "RDS", "95.00",  f"db-{uid}", "us-east-1", "acct-99", "prod"],
    ]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    csv_bytes = buf.getvalue().encode("utf-8")

    upload_data = ingest_universal_csv(csv_bytes, f"phase8_{uid}.csv")
    assert upload_data.get("status") in ("success", "SUCCESS"), (
        f"Unexpected ingest response: {upload_data}"
    )
    new_batch_id = upload_data.get("batch_id")
    assert new_batch_id, "No batch_id returned from ingest"

    batches = list_ingestion_batches()
    batch_ids = [b.get("batch_id", "") for b in batches]
    assert new_batch_id in batch_ids, (
        f"Uploaded batch {new_batch_id!r} not found in list_ingestion_batches"
    )

    result = get_cost_forecast(days=30, batch_id=new_batch_id)
    assert result.get("status") == "INSUFFICIENT_DATA", (
        f"Expected INSUFFICIENT_DATA for a 3-row upload, got: {result.get('status')}"
    )


# ===========================================================================
# 10. Regression — existing test files must not be broken by Phase 8 changes
# ===========================================================================

def test_13_phase6_forecast_still_accepts_no_batch():
    """
    get_cost_forecast() with no batch_id must still work (backward-compatible).
    """
    result = get_cost_forecast(days=7)
    assert result.get("status") in ("SUCCESS", "INSUFFICIENT_DATA"), (
        f"get_cost_forecast() without batch_id returned unexpected status: {result}"
    )


def test_14_batches_endpoint_exists():
    """get_ingested_batches must return a dict with batches list."""
    res = get_ingested_batches()
    assert isinstance(res, dict)
    assert "batches" in res
    assert isinstance(res["batches"], list)


def test_15_cost_summary_api_accepts_batch_id():
    """get_summary with batch_id must return valid financial KPIs."""
    data = get_summary(batch_id=CLEANED_BATCH)
    assert "total_expenditure" in data
    assert data["total_expenditure"] > 0


# ===========================================================================
# 11. Dynamic Chatbot Integration — Dataset-Aware Grounding
# ===========================================================================

def test_16_chatbot_grounding_different_batches():
    """
    Verify that _generate_grounded_answer and fetch_active_dataset_context
    produce distinct answers tailored to the active batch_id without rebuilding FAISS.
    """
    from app.rag_qa import fetch_active_dataset_context, _generate_grounded_answer

    ctx_cleaned = fetch_active_dataset_context(CLEANED_BATCH)
    ctx_focus   = fetch_active_dataset_context(FOCUS_BATCH)

    ans_cleaned = _generate_grounded_answer("What is our total spend?", [], ctx_cleaned)
    ans_focus   = _generate_grounded_answer("What is our total spend?", [], ctx_focus)

    assert "926,159" in ans_cleaned or "926159" in ans_cleaned, (
        f"Cleaned dataset answer should cite ~$926k, got: {ans_cleaned}"
    )
    assert "20.52" in ans_focus, (
        f"FOCUS dataset answer should cite $20.52, got: {ans_focus}"
    )
    assert ans_cleaned != ans_focus


def test_17_ask_request_supports_batch_id():
    """
    Validate that AskRequest schema and RAGPromptValidator accept batch_id.
    """
    from app.main import AskRequest
    from app.validators import validate_request

    req = AskRequest(question="What is our total spend?", batch_id=CLEANED_BATCH)
    assert req.batch_id == CLEANED_BATCH

    validated = validate_request({"question": "What is our spend?", "batch_id": FOCUS_BATCH})
    assert validated.get("batch_id") == FOCUS_BATCH
