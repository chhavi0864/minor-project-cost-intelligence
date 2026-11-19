import asyncio
from pathlib import Path

from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import app.main as api_module
import app.rag_qa as rag_qa


ROOT = Path(__file__).resolve().parent.parent


def test_rag_type_annotations_resolve():
    from typing import get_type_hints

    assert "return" in get_type_hints(rag_qa.fetch_active_dataset_context)


def test_dashboard_kpis_and_anomaly_button(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "status": "SUCCESS",
                "anomaly_count": 1,
                "total_groups": 20,
                "anomalies": [{
                    "date": "2024-09-10",
                    "severity": "HIGH",
                    "cloud_provider": "AWS",
                    "service_name": "EC2",
                    "daily_cost": 125.0,
                    "service_median_daily_cost": 50.0,
                    "cost_above_baseline": 75.0,
                    "root_cause_driver": "Consumption Surge",
                    "explanation": "Spend exceeded its baseline.",
                    "recommended_action": "Review recent usage.",
                    "anomaly_score": -0.2,
                }],
            }

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("requests.get", fake_get)
    app = AppTest.from_file(str(ROOT / "UI" / "app.py")).run(timeout=60)

    assert not app.exception
    assert len(app.metric) == 6
    assert all(metric.value for metric in app.metric)

    dataset_options = app.selectbox(key="dashboard_scope_selector").options
    cleaned_label = next(
        option for option in dataset_options if "cloud_cost_usage_cleaned.csv" in option
    )
    focus_label = next(option for option in dataset_options if "focus_sample.csv" in option)

    app.selectbox(key="dashboard_scope_selector").select(cleaned_label).run(timeout=60)
    assert app.metric[0].value == "USD 926,159.01"

    app.selectbox(key="dashboard_scope_selector").select(focus_label).run(timeout=60)
    assert app.metric[0].value == "USD 20.52"
    assert all(
        app.session_state[key] == focus_label
        for key in (
            "sb_global_scope_selector",
            "dashboard_scope_selector",
            "forecast_scope_selector",
            "chat_batch_selector",
        )
    )

    app.button(key="btn_anomaly").click().run(timeout=60)

    assert not app.exception
    assert calls
    assert calls[-1][0].endswith("/api/v1/anomalies")
    assert calls[-1][1]["params"].get("batch_id") == "4e832241"
    assert any("1** anomalous daily service groups" in item.value for item in app.success)


def test_data_ingestion_page_renders_native_upload_controls():
    app = AppTest.from_file(str(ROOT / "UI" / "app.py")).run(timeout=60)
    app.radio[0].set_value("📤  Data Ingestion").run(timeout=60)

    assert not app.exception
    assert len(app.file_uploader) == 1
    assert any("Financial Data Integrity" in item.value for item in app.info)
    assert not any("<div" in item.value for item in app.code)


def test_cost_optimization_page_renders_native_recommendation_metrics(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "total_estimated_monthly_savings": 75.0,
                "recommendations": [{
                    "type": "cost_spikes",
                    "resources": [{"resource_id": "resource-1"}],
                    "estimated_monthly_savings": 75.0,
                    "recommended_actions": ["Review recent pricing changes."],
                }],
            }

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    app = AppTest.from_file(str(ROOT / "UI" / "app.py")).run(timeout=60)
    app.radio[0].set_value("💡  Cost Optimization").run(timeout=60)

    assert not app.exception
    assert any(metric.label == "🟡 Cost Spikes" for metric in app.metric)
    assert any(
        metric.label == "Aggregate Preliminary Estimate"
        and metric.value == "$75.00/month"
        for metric in app.metric
    )
    assert any("Preliminary savings estimate" in item.value for item in app.warning)
    assert not any("<div" in item.value for item in app.code)


def test_recommendations_endpoint_forwards_dataset_scope(monkeypatch):
    expected_batch_id = "batch-regression"
    monkeypatch.setattr(
        api_module,
        "get_all_recommendations",
        lambda **kwargs: {**kwargs, "recommendations": []},
    )

    response = TestClient(api_module.app).get(
        "/recommendations",
        params={"batch_id": expected_batch_id},
    )

    assert response.status_code == 200
    assert response.json()["batch_id"] == expected_batch_id
    assert response.json()["all_batches"] is False


def test_suggested_chat_question_reaches_backend_with_active_batch(monkeypatch):
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "answer": "The selected dataset spend is $20.52.",
                "mode": "grounded_telemetry",
                "sources": [],
            }

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    app = AppTest.from_file(str(ROOT / "UI" / "app.py")).run(timeout=60)
    app.radio[0].set_value("🤖  AI Assistant").run(timeout=60)

    focus_label = next(
        option for option in app.selectbox(key="chat_batch_selector").options
        if "focus_sample.csv" in option
    )
    app.selectbox(key="chat_batch_selector").select(focus_label).run(timeout=60)
    app.button(key="sg_0").click().run(timeout=60)

    assert not app.exception
    assert calls[-1][0].endswith("/ask")
    assert calls[-1][1]["json"]["batch_id"] == "4e832241"
    assert len(app.session_state["messages"]) == 2
    assert app.session_state["messages"][-1]["role"] == "assistant"


def test_chatbot_surfaces_partial_active_context_errors(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(rag_qa, "retrieve", lambda question, top_k: [])
    monkeypatch.setattr(
        rag_qa,
        "fetch_active_dataset_context",
        lambda batch_id: {"batch_id": batch_id, "warnings": ["Forecast projections unavailable."]},
    )

    response = asyncio.run(rag_qa.answer("What is our spend?", batch_id="batch-regression"))

    assert response["mode"] == "grounded_telemetry"
    assert "Forecast projections unavailable." in response["warning"]
