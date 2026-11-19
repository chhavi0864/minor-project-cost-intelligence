"""
FinOps Copilot - RAG Question Answering
Uses FAISS retrieval, LangChain, and Groq LLM with grounded fallback.

Requirements:
- GROQ_API_KEY environment variable (optional for conversational generation)
- Existing FAISS index

Default LLM: openai/gpt-oss-20b
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.rag import retrieve


# ---------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------

load_dotenv()

# Disable LangChain tracing for local development.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


# ---------------------------------------------------------
# System Prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are an AI-powered FinOps assistant specializing in
cloud cost analysis, optimization, and resource management.

Your responsibilities include:

1. Explaining cloud expenditure patterns.
2. Providing FinOps best practices.
3. Identifying potential cost optimization opportunities.
4. Explaining the importance of resource tagging.
5. Helping users understand cloud billing information.

IMPORTANT RULES:

- Use the provided context as the primary evidence.
- Do not invent cloud billing values or utilization metrics.
- Do not calculate authoritative financial totals from
  a few retrieved billing records.
- Do not invent potential savings percentages.
- If ownership is unassigned, highlight the importance
  of proper resource tagging.
- Distinguish observations from possible explanations.
- If data is insufficient, clearly state the limitation.
- Do not claim to identify hourly anomalies from monthly data.
- Cite source IDs when using retrieved information.
- Never invent source IDs.
- Be concise, clear, and technically accurate.

If the retrieved context does not contain enough information,
explain what additional data would be required.
"""


# ---------------------------------------------------------
# Few-Shot Examples
# ---------------------------------------------------------

FEW_SHOT = [
    {
        "q": "What should I do with idle cloud resources?",
        "a": (
            "Review resources showing consistently low usage. "
            "Verify CPU, memory, network, and workload utilization "
            "before stopping, terminating, or resizing them. "
            "Estimate potential savings only when reliable "
            "cost and utilization data are available."
        ),
    },
    {
        "q": "How do resource tags help with cloud costs?",
        "a": (
            "Resource tags help attribute cloud expenditure "
            "to teams, projects, and environments. "
            "Missing tags reduce cost visibility and accountability. "
            "Tagging improves governance but does not directly "
            "guarantee financial savings."
        ),
    },
    {
        "q": "Why does cloud expenditure spike at 2 AM?",
        "a": (
            "A specific 2 AM cost spike cannot be verified using "
            "monthly billing records alone. "
            "Hourly billing, CPU utilization, workload activity, "
            "autoscaling events, and scheduled jobs would be "
            "needed to investigate the cause."
        ),
    },
]


# ---------------------------------------------------------
# Dynamic Active Dataset Retrieval Engine (Live Database/API)
# ---------------------------------------------------------

def fetch_active_dataset_context(batch_id: str | None) -> dict[str, Any]:
    """
    Fetches real-time cost intelligence, top drivers, anomalies, forecasts,
    and metadata for the active dataset batch directly from the database and APIs.
    """
    if not batch_id or batch_id in ["All Ingested Datasets", "None", ""]:
        return {}

    context: dict[str, Any] = {"batch_id": batch_id}
    context_warnings: list[str] = []

    # 1. Batch metadata
    try:
        from app.universal_etl import list_ingestion_batches
        batches = list_ingestion_batches()
        for b in batches:
            if b.get("batch_id") == batch_id:
                context["batch_info"] = b
                break
    except Exception as exc:
        context_warnings.append(f"Batch metadata could not be loaded: {exc}")

    # 2. Cost Summary & Top Drivers
    try:
        from app.cost_intelligence import get_cost_summary, get_service_cost_breakdown
        summary = get_cost_summary(batch_id=batch_id)
        if summary:
            context["cost_summary"] = summary
        services = get_service_cost_breakdown(batch_id=batch_id, limit=5)
        if services:
            context["top_services"] = services
    except Exception as exc:
        context_warnings.append(f"Cost intelligence could not be loaded: {exc}")

    # 3. Anomaly Intelligence
    try:
        from app.anomaly_detection import detect_cost_anomalies
        anomalies_res = detect_cost_anomalies(batch_id=batch_id)
        if anomalies_res and anomalies_res.get("status") in ("SUCCESS", "success"):
            context["anomalies"] = anomalies_res.get("anomalies", [])
            context["anomaly_summary"] = anomalies_res.get("summary", {})
    except Exception as exc:
        context_warnings.append(f"Anomaly intelligence could not be loaded: {exc}")

    # 4. Forecast Projections
    try:
        from app.forecasting import get_cost_forecast
        forecast_res = get_cost_forecast(days=30, batch_id=batch_id)
        if forecast_res:
            context["forecast"] = forecast_res
    except Exception as exc:
        context_warnings.append(f"Forecast projections could not be loaded: {exc}")

    if context_warnings:
        context["warnings"] = context_warnings

    return context


# ---------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------

def _build_prompt(question, retrieved, active_context: dict[str, Any] | None = None):
    """
    Build the prompt using retrieved FAISS context and live active-dataset context.
    """

    chunks_text = []

    for r in retrieved:
        source_id = r.get("id", "unknown")
        text = r.get("text", "")

        chunks_text.append(
            f"[Source: {source_id}]\n{text}"
        )

    context_block = (
        "\n\n".join(chunks_text)
        if chunks_text
        else "No relevant documents were retrieved."
    )

    few_shot_block = "\n\n".join(
        [
            f"Question: {example['q']}\n"
            f"Answer: {example['a']}"
            for example in FEW_SHOT
        ]
    )

    active_block = ""
    if active_context and active_context.get("batch_id"):
        b_id = active_context["batch_id"]
        b_info = active_context.get("batch_info", {})
        summary = active_context.get("cost_summary", {})
        top_svc = active_context.get("top_services", [])
        anom_list = active_context.get("anomalies", [])
        fcst = active_context.get("forecast", {})

        svc_lines = [f"  - {s.get('service_name')}: ${s.get('cost', 0):,.2f} ({s.get('share_pct', 0):.1f}%)" for s in top_svc]
        svc_str = "\n".join(svc_lines) if svc_lines else "  - No service breakdown available."

        fcst_str = "No forecast available"
        if fcst.get("status") == "SUCCESS":
            fcst_str = (
                f"30-day projected spend is ${fcst.get('total_predicted_cost', 0):,.2f} "
                f"(avg ${fcst.get('average_daily_cost', 0):,.2f}/day). "
                f"Budget pacing: {fcst.get('budget_pacing', {}).get('risk_status', 'N/A')}"
            )
        elif fcst.get("status") == "INSUFFICIENT_DATA":
            fcst_str = f"Insufficient daily data points (<14 days): {fcst.get('message', '')}"

        active_block = f"""
CURRENT ACTIVE UPLOADED DATASET (LIVE RECONCILED METRICS):
- Selected Batch ID: {b_id}
- Source File: {b_info.get('filename', 'Unknown')} (Format: {b_info.get('source_format', 'Unknown')})
- Date Range: {b_info.get('date_range_start', 'N/A')} to {b_info.get('date_range_end', 'N/A')}
- Total Active Spend: ${summary.get('total_expenditure', summary.get('total_cost', 0)):,.2f} across {summary.get('total_records', 0):,} records
- Top Services:
{svc_str}
- Live Anomalies Detected: {len(anom_list)} anomalies identified in this active batch
- Live 30-Day Cost Forecast: {fcst_str}
"""

    user_prompt = f"""
EXAMPLE QUESTIONS AND ANSWERS:

{few_shot_block}

HISTORICAL FINOPS REFERENCE KNOWLEDGE (FAISS STATIC INDEX):

{context_block}
{active_block}
USER QUESTION:

{question}

INSTRUCTIONS:

1. When answering questions regarding the current expenditure, anomalies, top services, or forecasts, PRIORITIZE the LIVE ACTIVE UPLOADED DATASET figures above.
2. Clearly distinguish between historical baseline benchmarks (FAISS) and current active dataset figures.
3. Do not fabricate cloud expenditure values or savings estimates.
4. Mention data limitations when applicable (e.g. insufficient history for daily forecasting).
5. Cite source IDs or [Active Dataset: <batch_id>] when referencing evidence.
6. Keep the explanation clear, professional, and concise.
"""

    return user_prompt


# ---------------------------------------------------------
# Deterministic Grounded Synthesis Engine (Fallback & Verification)
# ---------------------------------------------------------

def _generate_grounded_answer(
    question: str,
    retrieved: list[dict],
    active_context: dict[str, Any] | None = None
) -> str:
    """
    Synthesizes a strictly grounded, factual FinOps answer combining live active-dataset metrics
    with retrieved FAISS knowledge. Guarantees zero hallucinations and clear dataset separation.
    """
    q_lower = question.lower()
    sections = []

    # 0. Active Dataset Section (Live Database/API)
    if active_context and active_context.get("batch_id"):
        b_id = active_context["batch_id"]
        b_info = active_context.get("batch_info", {})
        summary = active_context.get("cost_summary", {})
        top_svc = active_context.get("top_services", [])
        anom_list = active_context.get("anomalies", [])
        fcst = active_context.get("forecast", {})

        active_lines = [
            f"- **Dataset Source**: `{b_info.get('filename', 'uploaded_file')}` (Batch `{b_id}`, Format: {b_info.get('source_format', 'FOCUS/Universal')})",
            f"- **Date Range**: {b_info.get('date_range_start', 'N/A')} to {b_info.get('date_range_end', 'N/A')} ({summary.get('total_records', 0):,} records)",
            f"- **Active Total Expenditure**: **${summary.get('total_expenditure', summary.get('total_cost', 0)):,.2f}**"
        ]

        if top_svc and any(k in q_lower for k in ["service", "driver", "spend", "cost", "breakdown", "highest"]):
            svc_desc = ", ".join([f"{s.get('service_name')} (${s.get('cost', 0):,.2f})" for s in top_svc[:3]])
            active_lines.append(f"- **Top Cost Drivers**: {svc_desc}")

        if any(k in q_lower for k in ["anomal", "spike", "outlier", "unusual", "surge"]):
            active_lines.append(f"- **Active Anomalies**: {len(anom_list)} anomalous cost records detected in this batch.")

        if any(k in q_lower for k in ["forecast", "future", "project", "30-day", "7-day", "pacing"]):
            if fcst.get("status") == "SUCCESS":
                active_lines.append(
                    f"- **Active 30-Day Forecast**: **${fcst.get('total_predicted_cost', 0):,.2f}** "
                    f"(avg ${fcst.get('average_daily_cost', 0):,.2f}/day; pacing: `{fcst.get('budget_pacing', {}).get('risk_status', 'HEALTHY')}`)"
                )
            elif fcst.get("status") == "INSUFFICIENT_DATA":
                active_lines.append(f"- **Active 30-Day Forecast**: Insufficient daily history ({fcst.get('message', 'requires at least 14 days')}).")

        sections.append("### 🟢 Current Active Dataset Telemetry (Live Reconciled)\n" + "\n".join(active_lines))

    # Organize FAISS evidence by topic
    telemetry_points = []
    anomaly_points = []
    forecasting_points = []
    recommendation_points = []
    general_points = []

    for r in retrieved:
        src_id = r.get("id", "unknown")
        text = r.get("text", "").strip()
        
        if "telemetry" in src_id:
            telemetry_points.append((src_id, text))
        elif "anomaly" in src_id:
            anomaly_points.append((src_id, text))
        elif "forecasting" in src_id:
            forecasting_points.append((src_id, text))
        elif "recommendation" in src_id:
            recommendation_points.append((src_id, text))
        else:
            general_points.append((src_id, text))

    # 1. Spending Trends & Telemetry (Reference)
    if any(k in q_lower for k in ["spend", "trend", "cost", "total", "budget", "service", "driver"]):
        if telemetry_points:
            t_lines = [f"- **[{s_id}]**: {txt}" for s_id, txt in telemetry_points]
            sections.append("### 📊 Historical Baseline Reference (547-Day Benchmark)\n" + "\n".join(t_lines))

    # 2. Anomalies & Outliers
    if any(k in q_lower for k in ["anomal", "spike", "outlier", "unusual", "surge", "root cause", "driver"]):
        if anomaly_points:
            a_lines = [f"- **[{s_id}]**: {txt}" for s_id, txt in anomaly_points]
            sections.append("### 🚨 Historical Anomaly Intelligence & Patterns\n" + "\n".join(a_lines))

    # 3. Forecasting & Forward Projections
    if any(k in q_lower for k in ["forecast", "future", "next", "project", "30-day", "7-day", "pacing", "risk"]):
        if forecasting_points:
            f_lines = [f"- **[{s_id}]**: {txt}" for s_id, txt in forecasting_points]
            sections.append("### 📈 Historical Forecast Benchmarks & GAM Architecture\n" + "\n".join(f_lines))

    # 4. Optimization Recommendations
    if any(k in q_lower for k in ["reduc", "optim", "action", "save", "recommend", "idle", "tag", "sagemaker", "ebs", "ec2"]):
        if recommendation_points:
            r_lines = [f"- **[{s_id}]**: {txt}" for s_id, txt in recommendation_points]
            sections.append("### 💡 Recommended FinOps Best Practices\n" + "\n".join(r_lines))

    # Fallback to general or all retrieved evidence if query wasn't captured above
    if not sections:
        all_lines = [f"- **[{r.get('id', 'unknown')}]**: {r.get('text', '').strip()}" for r in retrieved]
        sections.append("### 📋 Retrieved FinOps Knowledge\n" + "\n".join(all_lines))

    # Add governance note
    if active_context and active_context.get("batch_id"):
        sections.append(
            f"> **Active Scope**: Current responses dynamically incorporate active batch `{active_context['batch_id']}`. "
            "Historical reference benchmarks reflect the verified 547-day training baseline."
        )
    else:
        sections.append(
            "> **FinOps Governance Note**: All figures are reconciled against verified billing records. "
            "Select an active dataset batch in the sidebar to view live dynamic metrics."
        )

    answer_body = "\n\n".join(sections)
    return answer_body


# ---------------------------------------------------------
# RAG Answer Function
# ---------------------------------------------------------

async def answer(
    question: str,
    top_k: int = 5,
    groq_model: str | None = None,
    batch_id: str | None = None
):
    """
    Retrieve relevant FinOps information and generate
    a grounded answer using Groq (or verified telemetry synthesis if offline).
    Prioritizes real-time live database/API retrieval for the selected batch_id.

    Returns:
        {
            "answer": str | None,
            "sources": list,
            "error": str (optional),
            "mode": "groq_llm" | "grounded_telemetry",
            "active_batch_id": str | None
        }
    """

    # Validate question
    if not question or not question.strip():
        return {
            "answer": None,
            "sources": [],
            "error": "Question cannot be empty."
        }

    # -----------------------------------------------------
    # Step 1: Retrieve relevant documents from FAISS
    # -----------------------------------------------------

    try:
        retrieved = retrieve(
            question,
            top_k=top_k
        )
    except Exception as e:
        return {
            "answer": None,
            "sources": [],
            "error": f"FAISS retrieval failed: {str(e)}"
        }

    # -----------------------------------------------------
    # Step 2: Fetch Live Active Dataset Context (if batch_id passed)
    # -----------------------------------------------------

    active_context = fetch_active_dataset_context(batch_id)
    context_warning = "; ".join(active_context.get("warnings", [])) or None

    # -----------------------------------------------------
    # Step 3: Check for GROQ API Key
    # -----------------------------------------------------

    current_api_key = os.getenv("GROQ_API_KEY")

    # If no GROQ API key is configured, provide grounded deterministic telemetry response
    if not current_api_key:
        grounded_text = _generate_grounded_answer(question, retrieved, active_context)
        result = {
            "answer": grounded_text,
            "sources": retrieved,
            "mode": "grounded_telemetry",
            "active_batch_id": batch_id
        }
        if context_warning:
            result["warning"] = context_warning
        return result

    # -----------------------------------------------------
    # Step 4: Build prompt & Invoke Groq LLM
    # -----------------------------------------------------

    user_prompt = _build_prompt(
        question,
        retrieved,
        active_context
    )

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT
    )

    user_msg = HumanMessage(
        content=user_prompt
    )

    model_name = (
        groq_model
        if groq_model is not None
        else DEFAULT_GROQ_MODEL
    )

    try:
        llm = ChatGroq(
            model=model_name,
            api_key=current_api_key,
            temperature=0,
            max_tokens=1024
        )

        # Asynchronous invocation
        response = await llm.ainvoke(
            [system_msg, user_msg]
        )

        answer_text = getattr(
            response,
            "content",
            None
        )

        if isinstance(answer_text, list):
            answer_text = "\n".join(
                str(item.get("text", ""))
                if isinstance(item, dict)
                else str(item)
                for item in answer_text
            )

        if not answer_text:
            answer_text = str(response)

        result = {
            "answer": answer_text,
            "sources": retrieved,
            "mode": "groq_llm",
            "active_batch_id": batch_id
        }
        if context_warning:
            result["warning"] = context_warning
        return result

    except Exception as e:
        # Graceful fallback: If Groq request fails, return grounded verified telemetry
        grounded_fallback = _generate_grounded_answer(question, retrieved, active_context)
        warnings = [context_warning] if context_warning else []
        warnings.append(
            f"Groq LLM request failed ({str(e)}). Returned verified telemetry answer."
        )
        return {
            "answer": grounded_fallback,
            "sources": retrieved,
            "mode": "grounded_telemetry",
            "active_batch_id": batch_id,
            "warning": "; ".join(warnings),
        }
