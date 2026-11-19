"""

FinOps Intelligence — AI-Powered Cloud Cost Analytics & Optimization Platform

Modern, Industry-Independent Cloud Cost Intelligence Dashboard (Phase 1).

Supports Universal CSV Uploads, FOCUS 1.0, Multi-Cloud Analytics, and RAG Assistant.

"""

import os

import io

import json

from pathlib import Path
import sys

# Resolve the backend `app` package before Streamlit imports UI/app.py as `app`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or '.').resolve() != UI_DIR]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# If a previously loaded UI/app.py shadows the backend package, clear it.
loaded_app = sys.modules.get('app')
if loaded_app is not None and not hasattr(loaded_app, '__path__'):
    sys.modules.pop('app', None)

import streamlit as st

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from datetime import datetime

import requests

from typing import Dict, List, Any, Optional

# Import backend modules directly for robust zero-latency data access

from app.universal_etl import (

    ingest_universal_csv,

    validate_billing_dataframe,

    list_ingestion_batches,

    STANDARDIZED_CATEGORIES,

)

from app.cost_intelligence import (

    get_available_batches,

    get_filter_options,

    get_cost_summary,

    get_service_cost_breakdown,

    get_category_distribution,

    get_expenditure_trends,

    get_top_resources,

    get_provider_distribution,

)

# ──────────────────────────────────────────────

# Constants & Theming

# ──────────────────────────────────────────────

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    os.getenv("API_URL", "http://localhost:8000"),
).rstrip("/")

COLORS = {

    "bg_primary": "#0B1020",

    "bg_secondary": "#111827",

    "bg_card": "#1E293B",

    "accent_blue": "#3B82F6",

    "accent_purple": "#8B5CF6",

    "success": "#10B981",

    "warning": "#F59E0B",

    "critical": "#EF4444",

    "text_primary": "#F8FAFC",

    "text_secondary": "#94A3B8",

    "border": "#334155",

}

CATEGORY_COLORS = {

    "Compute": "#3B82F6",

    "Database": "#8B5CF6",

    "Storage": "#10B981",

    "Networking": "#F59E0B",

    "Other": "#64748B",

}

CHART_PALETTE = ["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B", "#EF4444", "#06B6D4", "#EC4899", "#14B8A6"]

# ──────────────────────────────────────────────

# Streamlit Page Config

# ──────────────────────────────────────────────

st.set_page_config(

    page_title="FinOps Intelligence",

    page_icon="⚡",

    layout="wide",

    initial_sidebar_state="expanded",

)

# ──────────────────────────────────────────────

# Modern Dark SaaS Styling

# ──────────────────────────────────────────────

st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {

    font-family: 'Inter', sans-serif;

}

.stApp {

    background-color: #0B1020;

}

/* Sidebar */

section[data-testid="stSidebar"] {

    background-color: #111827;

    border-right: 1px solid #1E293B;

}

section[data-testid="stSidebar"] .stRadio > div > label {

    background-color: transparent;

    border-radius: 8px;

    padding: 0.55rem 0.85rem;

    margin-bottom: 2px;

    color: #CBD5E1;

}

section[data-testid="stSidebar"] .stRadio > div > label:hover {

    background-color: #1E293B;

}

/* Custom Card Boxes */

.card-box {

    background-color: #1E293B;

    border: 1px solid #334155;

    border-radius: 12px;

    padding: 1.15rem 1.25rem;

    height: 100%;

    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);

}

.card-title {

    color: #94A3B8;

    font-size: 0.75rem;

    font-weight: 600;

    text-transform: uppercase;

    letter-spacing: 0.05em;

    margin-bottom: 0.3rem;

}

.card-value {

    color: #F8FAFC;

    font-size: 1.65rem;

    font-weight: 700;

    margin-bottom: 0.25rem;

}

.card-subtitle {

    color: #64748B;

    font-size: 0.78rem;

}

.kpi-icon {

    font-size: 1.5rem;

    margin-bottom: 0.35rem;

}

/* Badges */

.badge {

    display: inline-block;

    padding: 0.25rem 0.6rem;

    border-radius: 6px;

    font-size: 0.75rem;

    font-weight: 600;

    text-transform: uppercase;

    letter-spacing: 0.04em;

}

.badge-blue   { background: rgba(59,130,246,0.15); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }

.badge-green  { background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }

.badge-amber  { background: rgba(245,158,11,0.15); color: #FBBF24; border: 1px solid rgba(245,158,11,0.3); }

.badge-red    { background: rgba(239,68,68,0.15);  color: #F87171; border: 1px solid rgba(239,68,68,0.3); }

.badge-purple { background: rgba(139,92,246,0.15); color: #A78BFA; border: 1px solid rgba(139,92,246,0.3); }

/* Header titles */

.page-header h1 {

    color: #F8FAFC;

    font-size: 1.75rem;

    font-weight: 700;

    margin-bottom: 0.1rem;

}

.page-subtitle {

    color: #94A3B8;

    font-size: 0.92rem;

    margin-bottom: 1.5rem;

    line-height: 1.4;

}

.section-label {

    color: #94A3B8;

    font-size: 0.78rem;

    font-weight: 600;

    text-transform: uppercase;

    letter-spacing: 0.06em;

    margin-bottom: 0.6rem;

    margin-top: 1.25rem;

}

/* Disclaimer & info box */

.disclaimer-box {

    background: rgba(245,158,11,0.06);

    border: 1px solid rgba(245,158,11,0.25);

    border-radius: 10px;

    padding: 0.85rem 1rem;

    margin-bottom: 1rem;

}

.disclaimer-box p {

    color: #FBBF24;

    font-size: 0.82rem;

    margin: 0;

    line-height: 1.5;

}

.src-card {

    background: #111827;

    border: 1px solid #334155;

    border-radius: 8px;

    padding: 0.75rem 1rem;

    margin-bottom: 0.5rem;

}

.src-card .src-id {

    color: #60A5FA;

    font-size: 0.75rem;

    font-weight: 600;

    margin-bottom: 0.25rem;

}

.src-card .src-text {

    color: #CBD5E1;

    font-size: 0.82rem;

    line-height: 1.5;

}

/* Hide Streamlit default chrome */

#MainMenu { visibility: hidden; }

footer { visibility: hidden; }

</style>

""", unsafe_allow_html=True)

def _plotly_layout(**overrides) -> dict:

    """Return standard dark theme layout for Plotly figures."""

    base = dict(

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(family="Inter, sans-serif", color="#CBD5E1", size=12),

        margin=dict(l=30, r=20, t=35, b=35),

        xaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"),

        yaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"),

        hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_color="#F8FAFC"),

        legend=dict(font=dict(color="#94A3B8")),

    )

    base.update(overrides)

    return base


def _section_label(text: str) -> None:
    st.markdown(f"**{text}**")


DATASET_SCOPE_KEYS = (
    "sb_global_scope_selector",
    "dashboard_scope_selector",
    "forecast_scope_selector",
    "chat_batch_selector",
)


def _sync_dataset_scope(widget_key: str, batch_map: Dict[str, Optional[str]]) -> None:
    selected_label = st.session_state[widget_key]
    st.session_state.active_batch_id = batch_map[selected_label]
    st.session_state.global_batch_label = selected_label
    for key in DATASET_SCOPE_KEYS:
        if key != widget_key:
            st.session_state[key] = selected_label


# ──────────────────────────────────────────────

# Sidebar Navigation & Branding

# ──────────────────────────────────────────────

with st.sidebar:

    st.title("⚡ FinOps Intelligence")
    st.caption("AI-Powered Cloud Cost Analytics & Optimization")

    _section_label("Navigation")

    page = st.radio(

        "nav",

        [
            "📊  Cost Intelligence",
            "📤  Data Ingestion",
            "📈  Cost Forecasting",
            "💡  Cost Optimization",
            "🤖  AI Assistant"
        ],

        label_visibility="collapsed",

    )

    st.markdown("---")

    # Global Active Dataset Scope Selector
    _section_label("Active Dataset Scope")
    all_global_batches = get_available_batches()
    global_batch_map = {"All Ingested Datasets": None}
    for b in all_global_batches:
        b_label = f"{b['filename']} ({b.get('source_format', 'CSV')})"
        global_batch_map[b_label] = b["batch_id"]

    global_batch_keys = list(global_batch_map.keys())
    if "global_batch_label" not in st.session_state or st.session_state.global_batch_label not in global_batch_keys:
        st.session_state.global_batch_label = global_batch_keys[0]
    for scope_key in DATASET_SCOPE_KEYS:
        if st.session_state.get(scope_key) not in global_batch_keys:
            st.session_state[scope_key] = st.session_state.global_batch_label

    selected_global_label = st.selectbox(
        "Active Scope",
        options=global_batch_keys,
        index=global_batch_keys.index(st.session_state.global_batch_label),
        key="sb_global_scope_selector",
        on_change=_sync_dataset_scope,
        args=("sb_global_scope_selector", global_batch_map),
        label_visibility="collapsed",
        help="Global scope applies across Overview, Anomalies, Forecasting, Recommendations, and AI Assistant."
    )
    st.session_state.global_batch_label = selected_global_label
    global_active_batch_id = global_batch_map[selected_global_label]
    st.session_state.active_batch_id = global_active_batch_id

    st.caption("Platform mode: Multi-Cloud FinOps Engine")
    st.caption("FOCUS 1.0 & Custom Schemas")

# ══════════════════════════════════════════════

#  PAGE 1 — COST INTELLIGENCE DASHBOARD

# ══════════════════════════════════════════════

if page == "📊  Cost Intelligence":

    st.markdown(

        '<div class="page-header"><h1>📊 Cloud Cost Overview</h1></div>'

        '<p class="page-subtitle">'

        'Comprehensive multi-cloud expenditure tracking, categorization, and resource-level cost attribution.'

        '</p>',

        unsafe_allow_html=True,

    )

    # 1. Scope Selector Bar (Dataset / Batch selection synchronized with Global Scope)

    col_batch, col_period, col_prov = st.columns([2, 1, 1])

    with col_batch:

        selected_batch_label = st.selectbox(

            "Active Dataset Scope",

            options=global_batch_keys,
            key="dashboard_scope_selector",
            on_change=_sync_dataset_scope,
            args=("dashboard_scope_selector", global_batch_map),

            help="Select an ingested billing batch or view consolidated expenditure across all datasets."

        )

        active_batch_id = global_batch_map[selected_batch_label]
        st.session_state.active_batch_id = active_batch_id

    filter_opts = get_filter_options(batch_id=active_batch_id)

    with col_period:

        periods = ["All Periods"] + filter_opts.get("billing_periods", [])

        selected_period = st.selectbox("Billing Period", options=periods, index=0)

    with col_prov:

        providers = ["All Providers"] + filter_opts.get("cloud_providers", [])

        selected_provider = st.selectbox("Cloud Provider", options=providers, index=0)

    # 2. Primary Financial KPIs

    summary = get_cost_summary(

        batch_id=active_batch_id,

        billing_period=selected_period,

        cloud_provider=selected_provider

    )

    _section_label("Primary Financial KPIs")
    top_svc = summary["highest_cost_service"]
    kpi_rows = [st.columns(3), st.columns(3)]
    kpi_values = [
        ("💰 Total Spend", f"{summary['currency']} {summary['total_expenditure']:,.2f}", None),
        ("📋 Billing Records", f"{summary['total_records']:,}", "Validated line items"),
        ("☁️ Cloud Services", str(summary["service_count"]), "Unique service types"),
        ("🖥️ Unique Resources", f"{summary['resource_count']:,}", "Identified assets"),
        (
            "📈 Highest-Cost Service",
            str(top_svc["name"]),
            f"${top_svc['cost']:,.2f} · {top_svc['share_pct']}%",
        ),
        ("📅 Billing Period", selected_period, selected_provider),
    ]
    for index, (label, value, delta) in enumerate(kpi_values):
        kpi_rows[index // 3][index % 3].metric(label, value, delta)

    # 3. Visualizations Row 1: Service Breakdown & Category Distribution

    _section_label("Cost Distribution & Breakdown")

    c_left, c_right = st.columns([3, 2])

    with c_left:

        services_data = get_service_cost_breakdown(

            batch_id=active_batch_id,

            billing_period=selected_period,

            cloud_provider=selected_provider,

            limit=10

        )

        if services_data:

            df_svc = pd.DataFrame(services_data).sort_values("cost", ascending=True)

            fig_svc = go.Figure(go.Bar(

                x=df_svc["cost"],

                y=df_svc["service_name"],

                orientation="h",

                marker=dict(

                    color=df_svc["service_category"].map(lambda c: CATEGORY_COLORS.get(c, "#64748B")),

                    line=dict(color="#334155", width=1)

                ),

                text=df_svc["cost"].apply(lambda v: f"${v:,.2f}"),

                textposition="auto",

                hovertemplate="<b>%{y}</b><br>Cost: $%{x:,.2f}<br>Category: %{customdata[0]}<br>Share: %{customdata[1]}%<extra></extra>",

                customdata=df_svc[["service_category", "share_pct"]].values,

            ))

            fig_svc.update_layout(

                **_plotly_layout(

                    title="Top 10 Services by Expenditure",

                    height=380,

                    xaxis_title="Expenditure ($)",

                    yaxis_title="",

                )

            )

            st.plotly_chart(fig_svc, width="stretch")

        else:

            st.info("No service cost data found for the selected scope.")

    with c_right:

        cat_data = get_category_distribution(

            batch_id=active_batch_id,

            billing_period=selected_period,

            cloud_provider=selected_provider

        )

        if cat_data:

            df_cat = pd.DataFrame(cat_data)

            fig_cat = go.Figure(go.Pie(

                labels=df_cat["category"],

                values=df_cat["cost"],

                hole=0.55,

                marker=dict(colors=[CATEGORY_COLORS.get(c, "#64748B") for c in df_cat["category"]]),

                textinfo="label+percent",

                textfont=dict(color="#F8FAFC", size=12),

                hovertemplate="<b>%{label}</b>: $%{value:,.2f} (%{percent})<extra></extra>",

            ))

            fig_cat.update_layout(

                **_plotly_layout(

                    title="Cost Distribution by Service Category",

                    height=380,

                    showlegend=True,

                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)

                )

            )

            st.plotly_chart(fig_cat, width="stretch")

        else:

            st.info("No category data available.")

    # 4. Visualizations Row 2: Expenditure Trends & Multi-Cloud Provider Share

    _section_label("Expenditure Trends & Multi-Cloud Share")

    t_left, t_right = st.columns([3, 2])

    with t_left:

        trend_data = get_expenditure_trends(

            batch_id=active_batch_id,

            cloud_provider=selected_provider

        )

        if trend_data:

            df_trend = pd.DataFrame(trend_data)

            fig_trend = px.area(

                df_trend,

                x="date",

                y="cost",

                color="category",

                color_discrete_map=CATEGORY_COLORS,

                title="Expenditure Trend Over Time",

            )

            fig_trend.update_layout(

                **_plotly_layout(

                    height=350,

                    xaxis_title="",

                    yaxis_title="Cost ($)",

                    hovermode="x unified",

                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)

                )

            )

            st.plotly_chart(fig_trend, width="stretch")

        else:

            st.info("Trend data unavailable for the selected scope.")

    with t_right:

        prov_data = get_provider_distribution(

            batch_id=active_batch_id,

            billing_period=selected_period

        )

        if prov_data:

            df_prov = pd.DataFrame(prov_data)

            fig_prov = go.Figure(go.Pie(

                labels=df_prov["provider"],

                values=df_prov["cost"],

                hole=0.45,

                marker=dict(colors=CHART_PALETTE),

                textinfo="label+percent",

                textfont=dict(color="#F8FAFC", size=12),

                hovertemplate="<b>%{label}</b>: $%{value:,.2f} (%{percent})<extra></extra>",

            ))

            fig_prov.update_layout(

                **_plotly_layout(

                    title="Cloud Provider Share",

                    height=350,

                    showlegend=True,

                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)

                )

            )

            st.plotly_chart(fig_prov, width="stretch")

        else:

            st.info("Cloud provider data unavailable.")

    # 5. Top Cost-Driving Resources

    _section_label("Top 10 Highest-Cost Resources")

    top_resources = get_top_resources(

        batch_id=active_batch_id,

        billing_period=selected_period,

        cloud_provider=selected_provider,

        limit=10

    )

    if top_resources:

        df_res = pd.DataFrame(top_resources)

        df_res["cost"] = df_res["cost"].apply(lambda v: f"${v:,.2f}")

        df_res.columns = [c.replace("_", " ").title() for c in df_res.columns]

        st.dataframe(df_res, width="stretch", hide_index=True)

    else:

        st.info("No resource-level identifiers detected in the selected billing scope.")

    # ── Anomaly Detection ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🚨 Cost Anomaly Detection")
    st.caption(
        "Isolation Forest ML detection — flags statistically unusual daily service spend. "
        "Anomalies are daily provider/service cost groups that exceed their historical median "
        "and fall in the lowest outlier score quantile (set by the sensitivity threshold below). "
        "Adjusting sensitivity calibrates the reporting cutoff without altering the offline baseline model."
    )

    col_cont, col_lim = st.columns([1, 1])
    with col_cont:
        anom_contamination = st.slider(
            "Anomaly Sensitivity / Contamination Cutoff", 0.01, 0.10, 0.03, 0.01,
            key="anom_contamination",
            help="Defines the outlier score percentile threshold (0.03 = top 3% most anomalous). Calibrates filtering threshold without retraining the underlying production model."
        )
    with col_lim:
        anom_limit = st.number_input(
            "Max anomalies to display", min_value=10, max_value=200, value=50, step=10,
            key="anom_limit"
        )

    if st.button("🔍 Run Anomaly Detection", type="primary", key="btn_anomaly"):
        with st.spinner("Running Isolation Forest…"):
            try:
                anom_params = {
                    "contamination": anom_contamination,
                    "limit": int(anom_limit)
                }
                if active_batch_id:
                    anom_params["batch_id"] = active_batch_id

                resp = requests.get(
                    f"{API_BASE_URL}/api/v1/anomalies",
                    params=anom_params,
                    timeout=90,
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("status") == "SUCCESS":
                    anomalies = data.get("anomalies", [])
                    total_anom = data.get("anomaly_count", 0)
                    total_grp  = data.get("total_groups", 0)

                    st.success(
                        f"**{total_anom}** anomalous daily service groups detected "
                        f"out of **{total_grp}** total groups "
                        f"({100*total_anom/max(total_grp,1):.1f}%)"
                    )

                    if anomalies:
                        df_anom = pd.DataFrame(anomalies)
                        df_anom["date"] = pd.to_datetime(df_anom["date"])
                        df_anom = df_anom.sort_values("anomaly_score")

                        # ── Summary metrics ───────────────────────────────
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Anomalies", f"{total_anom}")
                        c2.metric("Total Excess Cost",
                                  f"${df_anom['cost_above_baseline'].sum():,.2f}")
                        c3.metric("Worst IF Score",
                                  f"{df_anom['anomaly_score'].min():.4f}")
                        c4.metric("Services Affected",
                                  str(df_anom["service_name"].nunique()))

                        # Severity breakdown pills
                        if "severity" in df_anom.columns:
                            sev_counts = df_anom["severity"].value_counts().to_dict()
                            st.caption(
                                f"🔴 **Critical:** {sev_counts.get('CRITICAL', 0)}  |  "
                                f"🟠 **High:** {sev_counts.get('HIGH', 0)}  |  "
                                f"🟡 **Medium:** {sev_counts.get('MEDIUM', 0)}  |  "
                                f"🟢 **Low:** {sev_counts.get('LOW', 0)}  |  "
                                f"🤖 **Engine:** {data.get('active_model', 'IsolationForest-Optimized')}"
                            )

                        # ── Anomalies table ───────────────────────────────
                        st.markdown("#### Flagged Anomalies & FinOps Root Causes")
                        display_cols = {
                            "date": "Date",
                            "severity": "Severity",
                            "cloud_provider": "Provider",
                            "service_name": "Service",
                            "daily_cost": "Daily Cost ($)",
                            "service_median_daily_cost": "Median ($)",
                            "cost_above_baseline": "Excess ($)",
                            "root_cause_driver": "Driver",
                            "explanation": "Evidence Explanation",
                            "recommended_action": "FinOps Recommended Action",
                            "anomaly_score": "IF Score",
                        }
                        st.dataframe(
                            df_anom[[c for c in display_cols if c in df_anom.columns]]
                                .rename(columns=display_cols),
                            use_container_width=True,
                            hide_index=True,
                        )

                        # ── Bar chart: excess cost by service ─────────────
                        st.markdown("#### Anomalous Excess Cost by Service")
                        excess_by_svc = (
                            df_anom.groupby("service_name")["cost_above_baseline"]
                            .sum()
                            .sort_values(ascending=False)
                            .reset_index()
                        )
                        fig_svc = px.bar(
                            excess_by_svc,
                            x="service_name",
                            y="cost_above_baseline",
                            labels={
                                "service_name": "Cloud Service",
                                "cost_above_baseline": "Total Excess Cost ($)",
                            },
                            title="Anomalous Excess Spend by Service",
                            color="cost_above_baseline",
                            color_continuous_scale="Reds",
                        )
                        fig_svc.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            coloraxis_showscale=False,
                        )
                        st.plotly_chart(fig_svc, use_container_width=True)

                        # ── Scatter: anomaly score vs excess cost ─────────
                        if "cloud_provider" in df_anom.columns:
                            st.markdown("#### Anomaly Score vs Excess Cost")
                            fig_scatter = px.scatter(
                                df_anom,
                                x="cost_above_baseline",
                                y="anomaly_score",
                                color="service_name",
                                hover_data=["date", "cloud_provider", "daily_cost"],
                                labels={
                                    "cost_above_baseline": "Excess Cost ($)",
                                    "anomaly_score": "Isolation Forest Score (lower = more anomalous)",
                                    "service_name": "Service",
                                },
                                title="IF Score vs Excess Cost (each point = one anomalous day)",
                            )
                            fig_scatter.update_layout(
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)

                    else:
                        st.info("No anomalies found at the selected contamination level.")

                elif data.get("status") == "INSUFFICIENT_DATA":
                    st.warning(
                        f"Insufficient data: {data.get('message')}  \n"
                        "Please upload billing CSV data first (Data Ingestion page)."
                    )
                else:
                    st.error(f"Anomaly detection returned an unexpected status: {data}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "⚠️ Cannot reach the FastAPI backend. "
                    f"Make sure the server is running at {API_BASE_URL}."
                )
            except Exception as exc:
                st.error(f"Anomaly detection error: {exc}")

# ══════════════════════════════════════════════

#  PAGE 2 — DATA INGESTION (UNIVERSAL CSV)

# ══════════════════════════════════════════════

elif page == "📤  Data Ingestion":

    st.markdown(

        '<div class="page-header"><h1>📤 Universal Cloud Billing CSV Ingestion</h1></div>'

        '<p class="page-subtitle">'

        'Upload, validate, and standardize cloud billing data from AWS, Azure, GCP, OCI, or FOCUS 1.0 datasets.'

        '</p>',

        unsafe_allow_html=True,

    )

    tab_upload, tab_demo, tab_history = st.tabs([

        "📁 Upload Billing CSV",

        "⚡ Quick-Load Official Datasets",

        "📜 Ingestion History & Batches"

    ])

    with tab_upload:

        st.info(
            "🛡️ **Financial Data Integrity & Duplicate Protection:** Uploaded CSVs "
            "are validated for numeric costs, single-currency consistency, and valid "
            "dates. SHA-256 fingerprinting prevents unintentional duplicate imports."
        )

        uploaded_file = st.file_uploader(

            "Select Cloud Cost & Usage CSV File",

            type=["csv", "txt"],

            help="Upload FOCUS 1.0, AWS Cost & Usage Reports, Azure Exports, or Legacy FinOps CSVs."

        )

        if uploaded_file is not None:

            file_bytes = uploaded_file.getvalue()

            filename = uploaded_file.name

            try:

                raw_df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

                val_res, mapping = validate_billing_dataframe(raw_df)

                # Validation & Preview Card

                _section_label("File Inspection & Validation Summary")

                v1, v2, v3, v4 = st.columns(4)

                with v1:
                    st.metric("File & Records", f"{len(raw_df):,} rows")
                    st.caption(f"{filename} ({len(file_bytes) / 1024:.1f} KB)")

                with v2:
                    st.metric("Detected Format", val_res.format_detected)
                    st.caption(f"{len(raw_df.columns)} columns detected")

                with v3:
                    curr_str = ", ".join(val_res.currencies_found)
                    st.metric("Billing Currency", curr_str or "Not detected")
                    st.caption("Currency validation")

                with v4:
                    date_str = f"{val_res.date_range[0]} to {val_res.date_range[1]}" if val_res.date_range[0] else "N/A"
                    st.metric("Date Span", date_str)
                    st.caption("Charge period timestamps")

                # Display Validation Findings

                if val_res.is_valid:

                    st.success(f"✅ Data validation passed! Total detected expenditure: ${val_res.total_expenditure:,.2f}")

                else:

                    st.error("❌ Validation Failed:")

                    for err in val_res.errors:

                        st.write(f"• {err}")

                if val_res.warnings:

                    with st.expander("⚠️ Data Quality Warnings", expanded=False):

                        for warn in val_res.warnings:

                            st.write(f"• {warn}")

                # Preview Raw Data

                with st.expander("🔍 Preview First 5 Rows of Source File", expanded=True):

                    st.dataframe(raw_df.head(5), width="stretch")

                # Ingestion Action

                st.markdown("---")

                col_btn, col_force = st.columns([1, 2])

                with col_force:

                    force_reimport = st.checkbox(

                        "Force re-import (Overwrite previous import if file was already ingested)",

                        value=False

                    )

                with col_btn:

                    if st.button("🚀 Ingest & Standardize Billing Data", type="primary", disabled=not val_res.is_valid):

                        with st.spinner("Processing, normalizing, and storing billing records..."):

                            ingest_res = ingest_universal_csv(

                                file_source=file_bytes,

                                filename=filename,

                                force_reimport=force_reimport

                            )

                        if ingest_res["status"] == "SUCCESS":

                            st.success(f"🎉 {ingest_res['message']}")

                            st.info(f"Registered Batch ID: `{ingest_res['batch_id']}`. Navigate to the Cost Intelligence Dashboard to analyze.")

                        elif ingest_res["status"] == "DUPLICATE":

                            st.warning(f"⚠️ {ingest_res['message']}")

                            st.info("Check 'Force re-import' above if you deliberately wish to reload this batch.")

                        else:

                            st.error(f"Ingestion failed: {ingest_res.get('message')}")

            except Exception as e:

                st.error(f"Error reading uploaded file: {str(e)}")

    with tab_demo:

        st.markdown("### Official FinOps Datasets")

        st.markdown(

            "You can immediately load verified sample datasets into the SQLite database for exploration and viva demonstration:"

        )

        d1, d2 = st.columns(2)

        with d1:
            with st.container(border=True):
                st.caption("FOCUS 1.0 Specification · 1,000 rows")
                st.subheader("FinOps Foundation FOCUS Multi-Cloud")
                st.write(
                    "Real anonymized multi-cloud billing records from AWS, Azure, "
                    "and Oracle Cloud, conforming to the FOCUS 1.0 schema."
                )

            if st.button("Load FinOps Foundation FOCUS Sample", key="btn_load_focus"):

                focus_path = Path("FinOps-Copilot/data/raw/focus_sample.csv")

                if not focus_path.exists():

                    focus_path = Path("data/raw/focus_sample.csv")

                if focus_path.exists():

                    with st.spinner("Ingesting official FOCUS 1.0 sample dataset..."):

                        res = ingest_universal_csv(focus_path, "focus_sample.csv", force_reimport=True)

                    if res["status"] in ["SUCCESS", "DUPLICATE"]:

                        st.success(res["message"])

                        st.rerun()

                    else:

                        st.error(res["message"])

                else:

                    st.error("focus_sample.csv not found in data/raw/. Please ensure it has been downloaded.")

        with d2:
            with st.container(border=True):
                st.caption("Legacy FinOps Schema · 600 rows")
                st.subheader("Standardized Synthetic Cloud Data")
                st.write(
                    "Six-month synthetic cloud expenditure across 100 resources "
                    "covering Compute, Storage, Database, and Networking."
                )

            if st.button("Load Legacy Synthetic Sample", key="btn_load_legacy"):

                legacy_path = Path("FinOps-Copilot/data/sample_billing.csv")

                if not legacy_path.exists():

                    legacy_path = Path("data/sample_billing.csv")

                if legacy_path.exists():

                    with st.spinner("Ingesting legacy synthetic billing dataset..."):

                        res = ingest_universal_csv(legacy_path, "sample_billing.csv", force_reimport=True)

                    if res["status"] in ["SUCCESS", "DUPLICATE"]:

                        st.success(res["message"])

                        st.rerun()

                    else:

                        st.error(res["message"])

                else:

                    st.error("sample_billing.csv not found in data/.")

    with tab_history:

        st.markdown("### Registered Ingestion Batches")

        batches_history = list_ingestion_batches()

        if batches_history:

            df_hist = pd.DataFrame(batches_history)

            df_hist["total_cost"] = df_hist["total_cost"].apply(lambda v: f"${v:,.2f}" if pd.notna(v) else "$0.00")

            df_hist = df_hist[[

                "batch_id", "filename", "source_format", "total_records",

                "total_cost", "currency", "date_range_start", "date_range_end", "created_at"

            ]]

            df_hist.columns = [c.replace("_", " ").title() for c in df_hist.columns]

            st.dataframe(df_hist, width="stretch", hide_index=True)

        else:

            st.info("No batches ingested yet.")

# ══════════════════════════════════════════════

#  PAGE 3 — COST OPTIMIZATION (PRESERVED)

# ══════════════════════════════════════════════

elif page == "💡  Cost Optimization":

    st.markdown(

        '<div class="page-header"><h1>💡 Cost Optimization Insights</h1></div>'

        '<p class="page-subtitle">'

        'Explore potential cloud cost inefficiencies and resource governance gaps.'

        '</p>',

        unsafe_allow_html=True,

    )

    def fetch_recommendations():
        try:
            params = {}
            if st.session_state.get("active_batch_id"):
                params["batch_id"] = st.session_state.active_batch_id
            else:
                params["all_batches"] = True
            r = requests.get(
                f"{API_BASE_URL}/recommendations",
                params=params,
                timeout=15,
            )

            r.raise_for_status()

            return r.json()

        except requests.exceptions.ConnectionError as exc:
            st.warning(f"Backend unavailable; using local recommendation analysis: {exc}")
            from app.recommendations import get_all_recommendations

            return get_all_recommendations(
                batch_id=st.session_state.get("active_batch_id"),
                all_batches=not st.session_state.get("active_batch_id"),
            )
        except requests.RequestException as exc:
            st.error(f"Could not load recommendations: {exc}")
            return None

    with st.spinner("Analyzing cost optimization opportunities…"):
        recs = fetch_recommendations()

    if recs is None:
        st.stop()

    if recs:
        all_recs = recs.get("recommendations", [])
        total_savings = recs.get("total_estimated_monthly_savings", 0)
        if not all_recs:
            st.info("No cost optimization recommendations were found for the selected dataset.")

        st.warning(
            "⚠️ **Preliminary savings estimate:** These prototype-level estimates "
            "require validation against actual resource utilization and billing data."
        )

        rec_type_map = {

            "idle_resources": ("🔴", "Idle Resources"),

            "cost_spikes":    ("🟡", "Cost Spikes"),

            "tagging_gaps":   ("🟣", "Tagging Gaps"),

        }

        summary_cols = st.columns(len(all_recs) if all_recs else 1)

        for idx, rec in enumerate(all_recs):

            rtype = rec.get("type", "unknown")

            icon, label = rec_type_map.get(rtype, ("📌", rtype.replace("_", " ").title()))

            res_count = len(rec.get("resources", []))

            est_save = rec.get("estimated_monthly_savings", 0)

            with summary_cols[idx % len(summary_cols)]:
                st.metric(
                    f"{icon} {label}",
                    f"{res_count} resource{'s' if res_count != 1 else ''}",
                    delta=f"${est_save:,.2f}/mo estimated savings",
                    delta_color="off",
                )

        _section_label("Detailed Recommendations")

        for rec in all_recs:

            rtype = rec.get("type", "unknown")

            icon, label = rec_type_map.get(rtype, ("📌", rtype.replace("_", " ").title()))

            resources_list = rec.get("resources", [])

            actions = rec.get("recommended_actions", [])

            est_save = rec.get("estimated_monthly_savings", 0)

            with st.expander(f"{icon}  {label} — {len(resources_list)} resource{'s' if len(resources_list) != 1 else ''}  ·  Est. ${est_save:,.2f}/mo"):

                if actions:

                    st.markdown("**Recommended Actions**")

                    for a in actions:

                        st.markdown(f"- {a}")

                if resources_list:

                    st.markdown("**Affected Resources**")

                    df_res = pd.DataFrame(resources_list)

                    for col in df_res.columns:

                        if "cost" in col.lower() or "savings" in col.lower():

                            try:

                                df_res[col] = pd.to_numeric(df_res[col], errors="coerce").apply(

                                    lambda v: f"${v:,.2f}" if pd.notna(v) else "—"

                                )

                            except Exception:

                                pass

                    df_res.columns = [c.replace("_", " ").title() for c in df_res.columns]

                    st.dataframe(df_res, width="stretch", hide_index=True)

        st.metric(
            "Aggregate Preliminary Estimate",
            f"${total_savings:,.2f}/month",
            help=(
                f"Combined across {len(all_recs)} recommendation categories. "
                "Validate against actual resource utilization and billing data."
            ),
        )

# ══════════════════════════════════════════════

#  PAGE 4 — AI ASSISTANT (PRESERVED)

# ══════════════════════════════════════════════

elif page == "🤖  AI Assistant":

    st.markdown(

        '<div class="page-header"><h1>🤖 FinOps AI Assistant</h1></div>'

        '<p class="page-subtitle">'

        'Ask questions about cloud expenditure, optimization, and FinOps practices.'

        '</p>',

        unsafe_allow_html=True,

    )

    # Active dataset selection for dynamic query grounding (synchronized with global scope)
    selected_chat_batch_label = st.selectbox(
        "Active Dataset Scope for AI Context",
        options=global_batch_keys,
        key="chat_batch_selector",
        on_change=_sync_dataset_scope,
        args=("chat_batch_selector", global_batch_map),
        help="Select which uploaded billing batch the AI Assistant will prioritize for current spend, anomaly, and forecast calculations.",
    )
    active_chat_batch_id = global_batch_map[selected_chat_batch_label]
    st.session_state.active_batch_id = active_chat_batch_id

    def ask_backend(question: str, batch_id: Optional[str] = None) -> Optional[Dict[str, Any]]:

        try:

            payload = {"question": question}
            if batch_id:
                payload["batch_id"] = batch_id

            r = requests.post(

                f"{API_BASE_URL}/ask",

                json=payload,

                headers={"Content-Type": "application/json"},

                timeout=30,

            )

            if r.status_code == 500:

                detail = r.text

                # Accept GROQ_API_KEY unavailable as a soft condition; response still returned
                if "GROQ_API_KEY not set" in detail:

                    st.info(

                        "To enable enhanced AI-generated answers, set the `GROQ_API_KEY` environment variable. "

                        "Grounded telemetry-based answers are still available."

                    )

                    # Try to parse a valid response body anyway if present
                    try:

                        return r.json()

                    except Exception:

                        return None

                st.error(f"Server error: {detail}")

                return None

            r.raise_for_status()

            return r.json()

        except requests.RequestException as e:

            st.error(f"⚠️ Could not reach AI backend: {e}")

            st.info("Make sure the FastAPI backend is running with `uvicorn app.main:app --reload`.")

            return None

    if "messages" not in st.session_state:

        st.session_state.messages = []

    if not st.session_state.messages:

        _section_label("Suggested Questions")

        suggestions = [

            "What is our total cloud spend and spending trend?",

            "Which services are driving the highest costs?",

            "What is our 30-day cost forecast and budget risk?",

            "What anomalies have been detected and what caused them?",

            "How can we reduce SageMaker and EBS costs?",

            "What are FinOps best practices for resource tagging?",

        ]

        sg_cols = st.columns(min(len(suggestions), 3))

        for i, q in enumerate(suggestions):

            with sg_cols[i % len(sg_cols)]:

                if st.button(q, key=f"sg_{i}", width="stretch"):
                    st.session_state.pending_chat_question = q
                    st.rerun()

    for msg in st.session_state.messages:

        avatar = "⚡" if msg["role"] == "assistant" else None

        with st.chat_message(msg["role"], avatar=avatar):

            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about cloud costs, optimization, or FinOps practices…")
    if prompt is None:
        prompt = st.session_state.pop("pending_chat_question", None)
    else:
        st.session_state.pop("pending_chat_question", None)

    if prompt:

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):

            st.markdown(prompt)

        with st.chat_message("assistant", avatar="⚡"):

            with st.spinner("Analyzing your question…"):

                response = ask_backend(prompt, batch_id=active_chat_batch_id)

            if response:

                answer = response.get("answer", "I'm unable to answer that right now.")

                mode = response.get("mode", "")

                if mode == "grounded_telemetry":

                    st.caption("🔍 _Answer sourced from verified billing telemetry & FinOps intelligence index._")

                elif mode == "groq_llm":

                    st.caption("⚡ _Answer generated by Groq LLM grounded on retrieved evidence._")

                st.markdown(answer)

                warning = response.get("warning")

                if warning:

                    st.warning(f"ℹ️ {warning}")

                sources = response.get("sources", [])

                if sources:

                    with st.expander("📎  View Retrieved Sources", expanded=False):

                        for idx, src in enumerate(sources, 1):

                            src_id = src.get("id", f"source-{idx}")

                            src_text = src.get("text", "")

                            src_score = src.get("score")

                            score_str = f" · relevance {src_score:.3f}" if src_score is not None else ""

                            st.markdown(

                                f"""<div class="src-card">

                                    <div class="src-id">Source {idx} — {src_id}{score_str}</div>

                                    <div class="src-text">{src_text}</div>

                                </div>""",

                                unsafe_allow_html=True,

                            )

                st.session_state.messages.append({"role": "assistant", "content": answer})

# ══════════════════════════════════════════════
#  PAGE 5 — CLOUD COST FORECASTING & BUDGET PACING
# ══════════════════════════════════════════════
elif page == "📈  Cost Forecasting":
    st.markdown(
        '<div class="page-header"><h1>📈 Cloud Cost Forecasting & Budget Pacing</h1>'
        '<p style="color:#94A3B8; margin-top:-0.5rem; font-size:0.95rem;">'
        'Predict forward cloud expenditure with weekly seasonality, uncertainty intervals, and budget risk analysis.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    # Scope Selector Bar for Forecasting (synchronized with global scope)
    col_batch_f, col_h, col_s, col_b = st.columns([1.5, 1, 1, 1])
    with col_batch_f:
        selected_fcst_batch_label = st.selectbox(
            "Dataset Scope",
            options=global_batch_keys,
            key="forecast_scope_selector",
            on_change=_sync_dataset_scope,
            args=("forecast_scope_selector", global_batch_map),
            help="Select an ingested billing batch to isolate forecast or model across all datasets."
        )
        active_fcst_batch_id = global_batch_map[selected_fcst_batch_label]
        st.session_state.active_batch_id = active_fcst_batch_id

    with col_h:
        forecast_days = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30, step=7)
    with col_s:
        forecast_service = st.selectbox(
            "Service Filter",
            ["All Services", "EC2", "EBS", "RDS", "S3", "SageMaker", "Redshift", "Lambda", "CloudFront"],
        )
    with col_b:
        monthly_budget = st.number_input(
            "Target Monthly Budget ($ USD)",
            min_value=1000.0,
            max_value=1000000.0,
            value=55000.0,
            step=5000.0,
        )

    with st.spinner("Generating time-series cost forecast..."):
        try:
            params = {
                "days": int(forecast_days),
                "monthly_budget": float(monthly_budget),
            }
            if active_fcst_batch_id:
                params["batch_id"] = active_fcst_batch_id
            if forecast_service != "All Services":
                params["service_name"] = forecast_service

            resp = requests.get(f"{API_BASE_URL}/api/v1/forecast", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "SUCCESS":
                f_records = data.get("forecast", [])
                pacing = data.get("budget_pacing", {})
                risk_status = pacing.get("risk_status", "NOT_CONFIGURED")

                # Sustained Overspending Alert Banner
                if data.get("sustained_overspending_alert"):
                    st.warning(
                        "⚠️ **Sustained Overspending Alert:** Actual spending over the last 3 days consistently "
                        "exceeded projected baselines. This indicates sustained expenditure drift rather than an isolated spike."
                    )

                # 4 KPI Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Projected Total Spend", f"${data['total_predicted_cost']:,.2f}")
                c2.metric("Average Daily Cost", f"${data['average_daily_cost']:,.2f}/day")
                
                util_pct = pacing.get("projected_utilization_pct")
                c3.metric(
                    "Budget Utilization",
                    f"{util_pct:.1f}%" if util_pct is not None else "N/A",
                    delta=f"{util_pct - 100:.1f}%" if util_pct is not None else None,
                    delta_color="inverse",
                )

                risk_badge = {
                    "HEALTHY_WITHIN_BUDGET": "🟢 Healthy (<90%)",
                    "WARNING_APPROACHING_LIMIT": "🟡 Warning (90-100%)",
                    "CRITICAL_OVER_BUDGET": "🔴 Critical Over Budget (>100%)",
                }.get(risk_status, risk_status)
                c4.metric("Budget Risk Status", risk_badge)

                if f_records:
                    df_fcst = pd.DataFrame(f_records)
                    df_fcst["date"] = pd.to_datetime(df_fcst["date"])

                    # Interactive Plotly Chart
                    fig_fcst = go.Figure()

                    # 95% Uncertainty Interval Band
                    fig_fcst.add_trace(go.Scatter(
                        x=df_fcst["date"],
                        y=df_fcst["upper_bound_95"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                    fig_fcst.add_trace(go.Scatter(
                        x=df_fcst["date"],
                        y=df_fcst["lower_bound_95"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(59, 130, 246, 0.15)",
                        name="95% Confidence Interval",
                    ))

                    # 80% Uncertainty Interval Band
                    fig_fcst.add_trace(go.Scatter(
                        x=df_fcst["date"],
                        y=df_fcst["upper_bound_80"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                    fig_fcst.add_trace(go.Scatter(
                        x=df_fcst["date"],
                        y=df_fcst["lower_bound_80"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(59, 130, 246, 0.25)",
                        name="80% Confidence Interval",
                    ))

                    # Predicted Cost Line
                    fig_fcst.add_trace(go.Scatter(
                        x=df_fcst["date"],
                        y=df_fcst["predicted_cost"],
                        mode="lines+markers",
                        line=dict(color="#2563EB", width=2.5),
                        marker=dict(size=4),
                        name="Predicted Daily Cost",
                    ))

                    # Daily Budget Benchmark Line
                    daily_budget = monthly_budget / 30.0
                    fig_fcst.add_hline(
                        y=daily_budget,
                        line_dash="dash",
                        line_color="#DC2626",
                        annotation_text=f"Daily Budget Limit (${daily_budget:,.0f})",
                        annotation_position="bottom right",
                    )

                    fig_fcst.update_layout(
                        title=f"{forecast_days}-Day Cloud Cost Forecast ({forecast_service})",
                        xaxis_title="Date",
                        yaxis_title="Cost ($ USD)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_fcst, use_container_width=True)

                    # Detailed forecast table
                    with st.expander("📋 View Daily Forecast Data Table", expanded=False):
                        display_df = df_fcst.copy()
                        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
                        st.dataframe(
                            display_df.rename(columns={
                                "date": "Date",
                                "predicted_cost": "Forecast ($)",
                                "lower_bound_80": "80% Lower ($)",
                                "upper_bound_80": "80% Upper ($)",
                                "lower_bound_95": "95% Lower ($)",
                                "upper_bound_95": "95% Upper ($)",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )

            elif data.get("status") == "INSUFFICIENT_DATA":
                st.warning(f"⚠️ {data.get('message')}")
            else:
                st.error(f"Forecasting error: {data.get('message', data)}")

        except Exception as exc:
            st.error(f"Could not connect to forecasting service: {exc}")
