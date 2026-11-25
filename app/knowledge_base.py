"""
Cloud Compass – embedded knowledge base (27 chunks).
Generated from data/faiss/meta.json so the cloud deployment
has zero dependency on the filesystem for RAG retrieval.
"""

KNOWLEDGE_CHUNKS = [
    {
        "id": "finops_tip_1",
        "source": "finops_doc",
        "text": "# FinOps Tips",
        "meta": {"type": "finops_doc", "paragraph_index": 1},
    },
    {
        "id": "finops_tip_2",
        "source": "finops_doc",
        "text": (
            "1. Right-size compute: pick instance families with lower unit cost and scale vertically only when necessary.\n"
            "2. Use reserved instances or savings plans for steady workloads.\n"
            "3. Delete idle resources older than 30 days; reclaim snapshots and unattached volumes.\n"
            "4. Ensure every resource has required tags:\n"
            "   - owner: Team/individual responsible for costs (e.g., \"team-a\", \"data-science\")\n"
            "   - project: Associated business project\n"
            "   - env: Environment type (e.g., \"prod\", \"dev\", \"staging\")\n"
            "   Missing tags create unknown/unattributed spend and prevent cost accountability.\n"
            "5. Monitor sudden unit cost changes and verify billing currency/units.\n"
            "6. Group small resources into aggregated budgets to avoid per-resource billing noise."
        ),
        "meta": {"type": "finops_doc", "paragraph_index": 2},
    },
    {
        "id": "finops_tip_3",
        "source": "finops_doc",
        "text": 'Note: Resources showing "unassigned" ownership need immediate tagging.',
        "meta": {"type": "finops_doc", "paragraph_index": 3},
    },
    {
        "id": "telemetry_summary_aggregate",
        "source": "billing_telemetry",
        "text": (
            "Verified Cloud Expenditure Telemetry: Total cumulative cloud spend is $926,159.01 across 547 continuous "
            "days from 2023-01-01 to 2024-06-30. Daily spending averages $1,693.16/day with an 18-month growth trend "
            "from approximately $36,900/month in early 2023 to $61,500/month by mid-2024."
        ),
        "meta": {"type": "telemetry_summary", "total_spend": 926159.0133, "days": 547},
    },
    {
        "id": "telemetry_service_ebs",
        "source": "billing_telemetry",
        "text": (
            "Cloud Service Expenditure - EBS: Total spend is $393,810.13 accounting for 42.5% of total cloud "
            "infrastructure expenditure across 18,126 usage records. EBS is the highest cost driver at 42.5%."
        ),
        "meta": {"type": "service_telemetry", "service": "EBS", "spend": 393810.1263, "pct": 42.52},
    },
    {
        "id": "telemetry_service_ec2",
        "source": "billing_telemetry",
        "text": (
            "Cloud Service Expenditure - EC2: Total spend is $184,565.00 accounting for 19.9% of total cloud "
            "infrastructure expenditure across 13,225 usage records. EC2 is the second highest cost driver at 19.9%."
        ),
        "meta": {"type": "service_telemetry", "service": "EC2", "spend": 184564.9991, "pct": 19.93},
    },
    {
        "id": "telemetry_service_sagemaker",
        "source": "billing_telemetry",
        "text": (
            "Cloud Service Expenditure - SageMaker: Total spend is $110,512.81 accounting for 11.9% of total cloud "
            "infrastructure expenditure across 1,641 usage records. SageMaker represents 11.2% of spend and is the "
            "primary driver of extreme burst spikes."
        ),
        "meta": {"type": "service_telemetry", "service": "SageMaker", "spend": 110512.8054, "pct": 11.93},
    },
    {
        "id": "telemetry_service_s3",
        "source": "billing_telemetry",
        "text": (
            "Cloud Service Expenditure - S3: Total spend is $88,571.64 accounting for 9.6% of total cloud "
            "infrastructure expenditure across 15,316 usage records."
        ),
        "meta": {"type": "service_telemetry", "service": "S3", "spend": 88571.6375, "pct": 9.56},
    },
    {
        "id": "telemetry_service_cloudfront",
        "source": "billing_telemetry",
        "text": (
            "Cloud Service Expenditure - CloudFront: Total spend is $60,431.22 accounting for 6.5% of total cloud "
            "infrastructure expenditure across 5,484 usage records."
        ),
        "meta": {"type": "service_telemetry", "service": "CloudFront", "spend": 60431.2155, "pct": 6.52},
    },
    {
        "id": "telemetry_service_rds",
        "source": "billing_telemetry",
        "text": (
            "Cloud Service Expenditure - RDS: Total spend is $45,540.02 accounting for 4.9% of total cloud "
            "infrastructure expenditure across 3,282 usage records."
        ),
        "meta": {"type": "service_telemetry", "service": "RDS", "spend": 45540.0185, "pct": 4.92},
    },
    {
        "id": "telemetry_service_redshift",
        "source": "billing_telemetry",
        "text": (
            "Cloud Service Expenditure - Redshift: Total spend is $40,024.03 accounting for 4.3% of total cloud "
            "infrastructure expenditure across 1,695 usage records."
        ),
        "meta": {"type": "service_telemetry", "service": "Redshift", "spend": 40024.0336, "pct": 4.32},
    },
    {
        "id": "telemetry_service_lambda",
        "source": "billing_telemetry",
        "text": (
            "Cloud Service Expenditure - Lambda: Total spend is $2,704.18 accounting for 0.3% of total cloud "
            "infrastructure expenditure across 7,658 usage records."
        ),
        "meta": {"type": "service_telemetry", "service": "Lambda", "spend": 2704.1774, "pct": 0.29},
    },
    {
        "id": "telemetry_team_platform-devops",
        "source": "billing_telemetry",
        "text": (
            "Team Cost Allocation - Platform-DevOps: Total allocated spend is $317,838.36 (34.3% of organization "
            "spend). Platform-DevOps manages the largest portion of shared infrastructure."
        ),
        "meta": {"type": "team_telemetry", "team": "Platform-DevOps", "spend": 317838.358, "pct": 34.32},
    },
    {
        "id": "telemetry_team_engineering",
        "source": "billing_telemetry",
        "text": (
            "Team Cost Allocation - Engineering: Total allocated spend is $257,654.41 (27.8% of organization spend)."
        ),
        "meta": {"type": "team_telemetry", "team": "Engineering", "spend": 257654.4131, "pct": 27.82},
    },
    {
        "id": "telemetry_team_ml-research",
        "source": "billing_telemetry",
        "text": (
            "Team Cost Allocation - ML-Research: Total allocated spend is $201,001.02 (21.7% of organization spend)."
        ),
        "meta": {"type": "team_telemetry", "team": "ML-Research", "spend": 201001.0239, "pct": 21.70},
    },
    {
        "id": "telemetry_team_data-analytics",
        "source": "billing_telemetry",
        "text": (
            "Team Cost Allocation - Data-Analytics: Total allocated spend is $95,324.01 (10.3% of organization spend)."
        ),
        "meta": {"type": "team_telemetry", "team": "Data-Analytics", "spend": 95324.0097, "pct": 10.29},
    },
    {
        "id": "telemetry_team_marketing-web",
        "source": "billing_telemetry",
        "text": (
            "Team Cost Allocation - Marketing-Web: Total allocated spend is $54,341.21 (5.9% of organization spend)."
        ),
        "meta": {"type": "team_telemetry", "team": "Marketing-Web", "spend": 54341.2086, "pct": 5.87},
    },
    {
        "id": "telemetry_weekly_seasonality",
        "source": "billing_telemetry",
        "text": (
            "Cloud Spending Cyclical Seasonality: Daily expenditure displays a persistent 7-day weekly cycle with a "
            "25.7% weekend dip. Average weekday spend is $1,828.07/day versus $1,358.03/day on weekends (Saturday and "
            "Sunday), reflecting non-production dev/staging idling."
        ),
        "meta": {"type": "seasonality_telemetry", "weekend_dip_pct": 25.71},
    },
    {
        "id": "anomaly_model_architecture",
        "source": "anomaly_intelligence",
        "text": (
            "Champion Anomaly Detection Model Architecture: Production uses an Optimized Isolation Forest "
            "(n_estimators=250, max_samples=1024, contamination=0.03) trained on 18 continuous and service-stratified "
            "features (10 cost/usage velocity metrics + 8 service one-hot encodings). Feature pruning removed 12 sparse "
            "non-informative dimensions (team, region, environment), increasing 3.0x cost spike recall from 7.4% to "
            "55.4% and 5.0x spike recall from 55.4% to 97.8% while maintaining false-positive stability at 3.37% on "
            "held-out test data."
        ),
        "meta": {"type": "anomaly_model", "features": 18, "contamination": 0.03},
    },
    {
        "id": "anomaly_severity_and_drivers",
        "source": "anomaly_intelligence",
        "text": (
            "Anomaly Severity and Root-Cause Driver Framework: Flagged cost anomalies are categorized into 4 severity "
            "tiers: CRITICAL (score < -0.08 and impact > $100), HIGH (score < -0.04 and impact > $40), MEDIUM, and "
            "LOW. Root causes are strictly categorized by telemetry evidence into 5 distinct drivers: 1. Consumption "
            "Surge: proportional spike in usage volume with steady unit cost. 2. Pricing / Rate Spike: surge in "
            "effective unit cost with steady usage. 3. Idle / Underutilized Resource: active baseline charges with zero "
            "or near-zero measured usage. 4. Resource Proliferation: sudden expansion in distinct active resource "
            "instances. 5. Spend Volatility: abrupt deviation from 7-day rolling historical baselines."
        ),
        "meta": {"type": "anomaly_framework"},
    },
    {
        "id": "finops_recommendations_sagemaker",
        "source": "finops_recommendations",
        "text": (
            "FinOps Recommendation for SageMaker Anomalies: SageMaker drives 88.8% of extreme cost outliers ($107.5k "
            "total). Recommended Actions: 1. Audit training job max-runtime timeouts to prevent zombie training loops. "
            "2. Utilize Spot instances for fault-tolerant distributed training (typically saving 50-70% on compute). "
            "3. Implement automated lifecycle policies that auto-terminate idle Jupyter notebooks and inference endpoints "
            "after 30 minutes of inactivity."
        ),
        "meta": {"type": "recommendation", "service": "SageMaker"},
    },
    {
        "id": "finops_recommendations_ebs_storage",
        "source": "finops_recommendations",
        "text": (
            "FinOps Recommendation for EBS Storage: EBS is the largest single cost driver (42.5% of total spend). "
            "Recommended Actions: 1. Identify and delete unattached EBS volumes (available status). 2. Migrate legacy "
            "gp2 storage volumes to gp3, achieving an immediate 20% baseline cost reduction with higher baseline IOPS. "
            "3. Review automated EBS snapshot retention schedules and purge redundant snapshots older than 30-90 days."
        ),
        "meta": {"type": "recommendation", "service": "EBS"},
    },
    {
        "id": "finops_recommendations_ec2_compute",
        "source": "finops_recommendations",
        "text": (
            "FinOps Recommendation for EC2 Compute: EC2 represents 19.9% of total expenditure. Recommended Actions: "
            "1. Rightsizing: downscale over-provisioned instances with < 15% average CPU utilization. 2. Purchase 1-year "
            "or 3-year Compute Savings Plans or Reserved Instances for baseline production workloads. 3. Schedule "
            "non-production (dev/staging) instances to automatically stop on nights and weekends, capturing ~26% savings."
        ),
        "meta": {"type": "recommendation", "service": "EC2"},
    },
    {
        "id": "forecasting_model_benchmark",
        "source": "forecasting_intelligence",
        "text": (
            "Cloud Cost Forecasting Methodology and 60-Day Holdout Backtesting Results: 1. Prophet-Architecture GAM "
            "(Champion): sMAPE = 3.54%, MAE = $69.18, RMSE = $90.16. 2. SARIMA(1,1,1)x(1,0,1,7): sMAPE = 3.64%, "
            "MAE = $70.60. 3. Seasonal Naive: sMAPE = 6.61%, MAE = $128.92. The Prophet-Architecture GAM was selected "
            "as Champion for superior accuracy, robust Fourier weekly seasonality, and pure-Python stability with zero "
            "C++ native DLL collisions."
        ),
        "meta": {"type": "forecasting_benchmark"},
    },
    {
        "id": "forecasting_30day_projections_and_budget_risk",
        "source": "forecasting_intelligence",
        "text": (
            "Forward 30-Day Cost Forecast and Budget Risk Assessment (cloud_cost_usage_cleaned.csv, 547-day history): "
            "Total projected spend over the next 30 days is $60,372.67 (averaging $2,012.42/day). Budget pacing "
            "evaluation indicates: OVER_BUDGET against the configured monthly target of $55,000 (projected utilization "
            "is 109.8%). The corrected figure uses the full 547-day cleaned dataset rather than sparse monthly "
            "aggregates. Forecast figures are dataset-dependent; use the Forecasting page and select your active "
            "dataset for real-time results."
        ),
        "meta": {"type": "forecasting_projections"},
    },
    {
        "id": "forecasting_dataset_scope_caveat",
        "source": "forecasting_intelligence",
        "text": (
            "Important: The AI Assistant uses a pre-built static knowledge index from the Phase 1-6 training dataset "
            "(cloud_cost_usage_cleaned.csv, $926,159.01 total historical spend, 2023-01-01 to 2024-06-30). Chatbot "
            "answers reflect this dataset and cannot automatically incorporate newly uploaded billing files. For "
            "real-time cost intelligence, anomaly detection, and forecasting on a newly uploaded dataset, use the Cost "
            "Intelligence Dashboard and Cost Forecasting pages with your active batch selected."
        ),
        "meta": {"type": "chatbot_data_scope_disclaimer"},
    },
    {
        "id": "forecasting_outlier_treatment_and_residual_thresholds",
        "source": "forecasting_intelligence",
        "text": (
            "Forecasting Outlier Pre-Treatment and Spike Tracking: To prevent past anomalous bursts from artificially "
            "inflating future forecasts, historical expenditure series undergo Winsorization (clipping to 1st and 99th "
            "percentiles). Forward forecasts feature calibrated +/- 2-sigma residual thresholds to detect real-time "
            "anomalies. Cumulative residual tracking distinguishes isolated transient spikes from sustained structural "
            "overspending."
        ),
        "meta": {"type": "forecasting_methodology"},
    },
]
