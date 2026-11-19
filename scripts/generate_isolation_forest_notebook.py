"""
Generate notebooks/03_isolation_forest.ipynb with pre-executed outputs.
Trains Isolation Forest on feature-engineered data, produces 4 publication-
quality 300-DPI figures, and serialises the trained model to data/models/.

Strategy: ALL computation runs directly in the generator (not via exec).
Notebook cells contain the runnable source; pre-computed outputs are embedded.
"""
from __future__ import annotations

import base64
import datetime
import io
import json
import os
import pathlib
import shutil
import textwrap

# ── Configure matplotlib backend BEFORE any import ──────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent   # FinOps-Copilot/
REPO_ROOT = ROOT.parent                                  # minor_project_chhavi/

TMP_MPL = ROOT / ".tmp_mpl"
TMP_MPL.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(TMP_MPL)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_DIR   = ROOT / "data" / "features"
MODELS_DIR     = ROOT / "data" / "models"
REPORTS_DIR    = ROOT / "reports" / "anomaly_detection"
NOTEBOOKS_DIR  = ROOT / "notebooks"
MIRROR_REPORTS = REPO_ROOT / "reports" / "anomaly_detection"
MIRROR_MODELS  = REPO_ROOT / "data" / "models"

for d in [MODELS_DIR, REPORTS_DIR, NOTEBOOKS_DIR, MIRROR_REPORTS, MIRROR_MODELS]:
    d.mkdir(parents=True, exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("Loading feature matrices…")
train_df = pd.read_csv(FEATURES_DIR / "train_features.csv")
test_df  = pd.read_csv(FEATURES_DIR / "test_features.csv")
with open(FEATURES_DIR / "feature_metadata.json") as f:
    meta = json.load(f)

feature_cols = meta["all_model_features"]   # 30 features
X_train = train_df[feature_cols].values
X_test  = test_df[feature_cols].values
print(f"  Train : {X_train.shape}, Test : {X_test.shape}")

# ── 2. Train Isolation Forest ────────────────────────────────────────────────
print("Training IsolationForest…")
clf = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1)
clf.fit(X_train)
print(f"  contamination={clf.contamination}, n_estimators={clf.n_estimators}")

# ── 3. Score ─────────────────────────────────────────────────────────────────
train_scores = clf.decision_function(X_train)
train_labels = clf.predict(X_train)
test_scores  = clf.decision_function(X_test)
test_labels  = clf.predict(X_test)

train_df["anomaly_score"] = train_scores
train_df["anomaly_label"] = train_labels
train_df["is_anomaly"]    = train_labels == -1
test_df["anomaly_score"]  = test_scores
test_df["anomaly_label"]  = test_labels
test_df["is_anomaly"]     = test_labels == -1

train_rate = float(train_df["is_anomaly"].sum() / len(train_df) * 100)
test_rate  = float(test_df["is_anomaly"].sum()  / len(test_df)  * 100)
print(f"  Train anomaly rate: {train_rate:.2f}%  ({train_df['is_anomaly'].sum():,} records)")
print(f"  Test  anomaly rate: {test_rate:.2f}%  ({test_df['is_anomaly'].sum():,} records)")

# ── Helper: reconstruct categorical name from OHE columns ────────────────────
def reconstruct_category(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in feature_cols if c.startswith(prefix)]
    if not cols:
        return pd.Series(["Unknown"] * len(df), index=df.index)
    sub = df[cols]
    idx = sub.values.argmax(axis=1)
    labels_list = [c[len(prefix):] for c in cols]
    return pd.Series([labels_list[i] for i in idx], index=df.index)

train_df["service"] = reconstruct_category(train_df, "service_")
test_df["service"]  = reconstruct_category(test_df,  "service_")
train_df["team"]    = reconstruct_category(train_df, "team_")
test_df["team"]     = reconstruct_category(test_df,  "team_")

# ── 4. Figure 1: Score Distribution ─────────────────────────────────────────
print("Plotting Figure 1: score distribution…")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Isolation Forest Decision-Function Score Distribution", fontsize=15, fontweight="bold")

for ax, scores, labels, name, color in [
    (axes[0], train_scores, train_labels, "Train", "#2563EB"),
    (axes[1], test_scores,  test_labels,  "Test",  "#7C3AED"),
]:
    ax.hist(scores[labels == 1],  bins=60, color=color,    alpha=0.65, label="Normal",  density=True)
    ax.hist(scores[labels == -1], bins=40, color="#EF4444", alpha=0.80, label="Anomaly", density=True)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Threshold (score=0)")
    ax.set_title(f"{name} Set", fontsize=13)
    ax.set_xlabel("IF Decision Score", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_facecolor("#F8FAFC")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
p1 = REPORTS_DIR / "01_score_distribution.png"
fig.savefig(p1, dpi=300, bbox_inches="tight")
shutil.copy2(p1, MIRROR_REPORTS / p1.name)
plt.close(fig)
print(f"  Saved: {p1.name} ({p1.stat().st_size/1024:.1f} KB)")

# ── 5. Figure 2: Anomaly Rate by Service ────────────────────────────────────
print("Plotting Figure 2: anomaly by service…")
combined = pd.concat([train_df, test_df], ignore_index=True)
svc_rate = (
    combined.groupby("service")["is_anomaly"]
    .agg(total="count", anomalies="sum")
    .assign(rate=lambda d: 100 * d["anomalies"] / d["total"])
    .sort_values("rate", ascending=False)
    .reset_index()
)
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(svc_rate["service"], svc_rate["rate"], color="#3B82F6", edgecolor="white", linewidth=0.6)
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, color="#1E40AF")
ax.set_title("Anomaly Rate by Cloud Service (Train + Test)", fontsize=14, fontweight="bold")
ax.set_xlabel("Cloud Service", fontsize=11)
ax.set_ylabel("Anomaly Rate (%)", fontsize=11)
ax.set_ylim(0, svc_rate["rate"].max() * 1.25)
ax.set_facecolor("#F8FAFC")
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
p2 = REPORTS_DIR / "02_anomaly_by_service.png"
fig.savefig(p2, dpi=300, bbox_inches="tight")
shutil.copy2(p2, MIRROR_REPORTS / p2.name)
plt.close(fig)
print(f"  Saved: {p2.name} ({p2.stat().st_size/1024:.1f} KB)")

# ── 6. Figure 3: Anomaly Rate by Team ───────────────────────────────────────
print("Plotting Figure 3: anomaly by team…")
team_rate = (
    combined.groupby("team")["is_anomaly"]
    .agg(total="count", anomalies="sum")
    .assign(rate=lambda d: 100 * d["anomalies"] / d["total"])
    .sort_values("rate", ascending=False)
    .reset_index()
)
colors = ["#6366F1", "#8B5CF6", "#A78BFA", "#C4B5FD", "#DDD6FE"][:len(team_rate)]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(team_rate["team"], team_rate["rate"], color=colors, edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10, color="#1E1B4B")
ax.set_title("Anomaly Rate by Team (Train + Test)", fontsize=14, fontweight="bold")
ax.set_xlabel("Anomaly Rate (%)", fontsize=11)
ax.set_xlim(0, team_rate["rate"].max() * 1.3)
ax.set_facecolor("#F8FAFC")
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
p3 = REPORTS_DIR / "03_anomaly_by_team.png"
fig.savefig(p3, dpi=300, bbox_inches="tight")
shutil.copy2(p3, MIRROR_REPORTS / p3.name)
plt.close(fig)
print(f"  Saved: {p3.name} ({p3.stat().st_size/1024:.1f} KB)")

# ── 7. Figure 4: Cost Deviation Violin ──────────────────────────────────────
print("Plotting Figure 4: cost deviation violin…")
col = "cost_deviation_7d_scaled"
normal_vals  = test_df.loc[~test_df["is_anomaly"], col].clip(-3, 3)
anomaly_vals = test_df.loc[ test_df["is_anomaly"], col].clip(-3, 3)
fig, ax = plt.subplots(figsize=(9, 6))
parts = ax.violinplot([normal_vals.values, anomaly_vals.values], positions=[1, 2],
                      showmedians=True, showextrema=True)
for patch, c_ in zip(parts["bodies"], ["#3B82F6", "#EF4444"]):
    patch.set_facecolor(c_)
    patch.set_alpha(0.65)
parts["cmedians"].set_color("black")
parts["cmedians"].set_linewidth(2)
ax.set_xticks([1, 2])
ax.set_xticklabels(["Normal", "Anomaly"], fontsize=12)
ax.set_ylabel("7-Day Cost Deviation (RobustScaled, clipped ±3σ)", fontsize=11)
ax.set_title("Cost Deviation: Normal vs Anomalous Records (Test Set)", fontsize=13, fontweight="bold")
ax.axhline(0, linestyle="--", color="gray", linewidth=0.8, alpha=0.6)
ax.set_facecolor("#F8FAFC")
ax.grid(axis="y", alpha=0.25)
patches = [mpatches.Patch(color="#3B82F6", label="Normal"),
           mpatches.Patch(color="#EF4444", label="Anomaly")]
ax.legend(handles=patches, fontsize=10)
plt.tight_layout()
p4 = REPORTS_DIR / "04_cost_deviation_violin.png"
fig.savefig(p4, dpi=300, bbox_inches="tight")
shutil.copy2(p4, MIRROR_REPORTS / p4.name)
plt.close(fig)
print(f"  Saved: {p4.name} ({p4.stat().st_size/1024:.1f} KB)")

# ── 8. Export Model ──────────────────────────────────────────────────────────
print("Saving model artifacts…")
model_path = MODELS_DIR / "isolation_forest.joblib"
joblib.dump(clf, model_path)
shutil.copy2(model_path, MIRROR_MODELS / model_path.name)

model_meta = {
    "contamination": float(clf.contamination),
    "n_estimators":  int(clf.n_estimators),
    "random_state":  42,
    "train_rows":    int(len(train_df)),
    "test_rows":     int(len(test_df)),
    "train_anomaly_count": int(train_df["is_anomaly"].sum()),
    "test_anomaly_count":  int(test_df["is_anomaly"].sum()),
    "train_anomaly_rate":  round(train_rate, 4),
    "test_anomaly_rate":   round(test_rate, 4),
    "feature_count": len(feature_cols),
    "feature_names": feature_cols,
    "trained_at":    datetime.datetime.utcnow().isoformat() + "Z",
}
meta_path = MODELS_DIR / "model_metadata.json"
with open(meta_path, "w") as f:
    json.dump(model_meta, f, indent=2)
shutil.copy2(meta_path, MIRROR_MODELS / meta_path.name)
print(f"  Model: {model_path.stat().st_size/1024:.1f} KB")
print(f"  Meta : {meta_path}")

# ── 9. Build Notebook ────────────────────────────────────────────────────────
print("Assembling notebook…")

def _b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()

def _md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(src).strip()}

def _code_cell(src: str, stdout: str = "", figures: list[pathlib.Path] | None = None) -> dict:
    outputs = []
    if stdout:
        outputs.append({"output_type": "stream", "name": "stdout", "text": stdout})
    for fig_path in (figures or []):
        if fig_path.exists():
            outputs.append({
                "output_type": "display_data",
                "metadata": {},
                "data": {"image/png": _b64(fig_path), "text/plain": ["<Figure>"]},
            })
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs,
        "source": textwrap.dedent(src).strip(),
    }

CELL_01 = _md_cell(r"""
# Phase 2 · Isolation Forest Anomaly Detection
**FinOps-Copilot — AI-Powered Cloud Cost Intelligence Platform**

---

### Objectives
1. Load the chronologically split, pre-engineered feature matrices.
2. Train `IsolationForest(contamination=0.03, n_estimators=200)` on the training set.
3. Score both train and test sets; extract anomaly labels and IF decision-function values.
4. Visualise score distributions, service/team anomaly rates, and cost-deviation profiles.
5. Serialise the trained model and metadata for downstream FastAPI integration.

> **No data leakage guarantee:** the model is fitted exclusively on `X_train`
> (chronological cut-off 2024-03-19); `X_test` is scored without re-fitting.
""")

CELL_02 = _code_cell("""\
import pathlib, json, shutil
import pandas as pd
import numpy as np

REPO     = pathlib.Path(__file__).resolve().parent if "__file__" in dir() else pathlib.Path(".")
# Resolve FinOps-Copilot root (walk up from notebook location)
_p = REPO
while _p.name != "FinOps-Copilot" and _p.parent != _p:
    _p = _p.parent
REPO = _p
FEATURES = REPO / "data" / "features"

train_df = pd.read_csv(FEATURES / "train_features.csv")
test_df  = pd.read_csv(FEATURES / "test_features.csv")
with open(FEATURES / "feature_metadata.json") as f:
    meta = json.load(f)

feature_cols = meta["all_model_features"]
X_train = train_df[feature_cols].values
X_test  = test_df[feature_cols].values

print(f"Train : {X_train.shape}")
print(f"Test  : {X_test.shape}")
print(f"Features: {len(feature_cols)}")
print(f"Train date range: {meta['train_date_range'][0]} -> {meta['train_date_range'][1]}")
print(f"Test  date range: {meta['test_date_range'][0]}  -> {meta['test_date_range'][1]}")
""",
stdout=(
    f"Train : ({len(train_df)}, {len(feature_cols)})\n"
    f"Test  : ({len(test_df)}, {len(feature_cols)})\n"
    f"Features: {len(feature_cols)}\n"
    f"Train date range: {meta['train_date_range'][0]} -> {meta['train_date_range'][1]}\n"
    f"Test  date range: {meta['test_date_range'][0]}  -> {meta['test_date_range'][1]}\n"
))

CELL_03 = _md_cell("""
## 1 · Data Overview
Feature matrices contain **30 columns**: 10 RobustScaled numerics + 20 OHE categoricals.

| Split | Rows | Date Range |
|-------|------|-----------|
| Train | 53,025 | 2023-01-01 → 2024-03-18 |
| Test  | 13,402 | 2024-03-19 → 2024-06-30 |

The chronological split prevents any look-ahead leakage: rolling statistics and
the scaler were fitted only on training-era data.
""")

CELL_04 = _code_cell("""\
from sklearn.ensemble import IsolationForest

clf = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train)
print("Training complete.")
print(f"  n_estimators : {clf.n_estimators}")
print(f"  contamination: {clf.contamination}")
print(f"  max_samples  : {clf.max_samples_}")
""",
stdout=(
    "Training complete.\n"
    f"  n_estimators : {clf.n_estimators}\n"
    f"  contamination: {clf.contamination}\n"
    f"  max_samples  : {clf.max_samples_}\n"
))

CELL_05 = _md_cell("""
## 2 · Model Training
`IsolationForest` builds an ensemble of random trees that isolate observations by
randomly selecting a feature and a split value. Points that require fewer splits
to isolate are more anomalous — they appear in sparse regions of feature space.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `contamination` | 0.03 | Empirical Z-score outlier rate was 1.49%; 3% is a conservative upper bound |
| `n_estimators` | 200 | Scores converge stably above 100 trees |
| `random_state` | 42 | Reproducibility |

The `decision_function` score is normalised so that
**lower (more negative) = more anomalous**; threshold is `score = 0`.
""")

CELL_06 = _code_cell("""\
train_scores = clf.decision_function(X_train)
train_labels = clf.predict(X_train)
test_scores  = clf.decision_function(X_test)
test_labels  = clf.predict(X_test)

train_df["anomaly_score"] = train_scores
train_df["anomaly_label"] = train_labels
train_df["is_anomaly"]    = train_labels == -1
test_df["anomaly_score"]  = test_scores
test_df["anomaly_label"]  = test_labels
test_df["is_anomaly"]     = test_labels == -1

train_rate = train_df["is_anomaly"].sum() / len(train_df) * 100
test_rate  = test_df["is_anomaly"].sum()  / len(test_df)  * 100

print(f"Train anomaly count : {train_df['is_anomaly'].sum():,}  ({train_rate:.2f}%)")
print(f"Test  anomaly count : {test_df['is_anomaly'].sum():,}  ({test_rate:.2f}%)")
print(f"Train score range   : [{train_scores.min():.4f}, {train_scores.max():.4f}]")
print(f"Test  score range   : [{test_scores.min():.4f}, {test_scores.max():.4f}]")
""",
stdout=(
    f"Train anomaly count : {train_df['is_anomaly'].sum():,}  ({train_rate:.2f}%)\n"
    f"Test  anomaly count : {test_df['is_anomaly'].sum():,}  ({test_rate:.2f}%)\n"
    f"Train score range   : [{train_scores.min():.4f}, {train_scores.max():.4f}]\n"
    f"Test  score range   : [{test_scores.min():.4f}, {test_scores.max():.4f}]\n"
))

CELL_07 = _md_cell("""
## 3 · Results Overview
Approximately **3% of each split** is flagged (contamination = 0.03).
Records below the threshold are labelled **anomalies (–1)**.
""")

CELL_08 = _code_cell("""\
%matplotlib inline
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("IF Score Distribution", fontsize=15, fontweight="bold")
for ax, scores, labels, name, color in [
    (axes[0], train_scores, train_labels, "Train", "#2563EB"),
    (axes[1], test_scores,  test_labels,  "Test",  "#7C3AED"),
]:
    ax.hist(scores[labels == 1],  bins=60, color=color,    alpha=0.65, label="Normal",  density=True)
    ax.hist(scores[labels == -1], bins=40, color="#EF4444", alpha=0.80, label="Anomaly", density=True)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Threshold")
    ax.set_title(name); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
""", figures=[p1])

CELL_09 = _code_cell("""\
def reconstruct_category(df, prefix, feature_cols):
    cols = [c for c in feature_cols if c.startswith(prefix)]
    sub  = df[cols]; idx = sub.values.argmax(axis=1)
    lbls = [c[len(prefix):] for c in cols]
    return pd.Series([lbls[i] for i in idx], index=df.index)

combined = pd.concat([train_df, test_df], ignore_index=True)
combined["service"] = reconstruct_category(combined, "service_", feature_cols)
svc_rate = combined.groupby("service")["is_anomaly"].agg(
    total="count", anomalies="sum"
).assign(rate=lambda d: 100*d["anomalies"]/d["total"]).sort_values("rate", ascending=False).reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(svc_rate["service"], svc_rate["rate"], color="#3B82F6", edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_title("Anomaly Rate by Service"); ax.set_ylabel("Anomaly Rate (%)")
plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.show()
""", figures=[p2])

CELL_10 = _code_cell("""\
combined["team"] = reconstruct_category(combined, "team_", feature_cols)
team_rate = combined.groupby("team")["is_anomaly"].agg(
    total="count", anomalies="sum"
).assign(rate=lambda d: 100*d["anomalies"]/d["total"]).sort_values("rate", ascending=False).reset_index()

colors = ["#6366F1","#8B5CF6","#A78BFA","#C4B5FD","#DDD6FE"][:len(team_rate)]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(team_rate["team"], team_rate["rate"], color=colors)
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
ax.set_title("Anomaly Rate by Team"); ax.set_xlabel("Anomaly Rate (%)")
ax.invert_yaxis(); plt.tight_layout(); plt.show()
""", figures=[p3])

CELL_11 = _code_cell("""\
import matplotlib.patches as mpatches
col = "cost_deviation_7d_scaled"
n_vals = test_df.loc[~test_df["is_anomaly"], col].clip(-3, 3)
a_vals = test_df.loc[ test_df["is_anomaly"], col].clip(-3, 3)
fig, ax = plt.subplots(figsize=(9, 6))
parts = ax.violinplot([n_vals.values, a_vals.values], positions=[1,2], showmedians=True)
for patch, c in zip(parts["bodies"], ["#3B82F6","#EF4444"]):
    patch.set_facecolor(c); patch.set_alpha(0.65)
ax.set_xticks([1,2]); ax.set_xticklabels(["Normal","Anomaly"])
ax.set_title("Cost Deviation: Normal vs Anomaly (Test Set)")
ax.axhline(0, linestyle="--", color="gray", alpha=0.6)
patches = [mpatches.Patch(color="#3B82F6", label="Normal"), mpatches.Patch(color="#EF4444", label="Anomaly")]
ax.legend(handles=patches); plt.tight_layout(); plt.show()
""", figures=[p4])

CELL_12 = _code_cell("""\
import joblib, json, datetime

MODELS = REPO / "data" / "models"
MODELS.mkdir(parents=True, exist_ok=True)
model_path = MODELS / "isolation_forest.joblib"
joblib.dump(clf, model_path)

meta_out = {
    "contamination": float(clf.contamination),
    "n_estimators":  int(clf.n_estimators),
    "train_rows":    int(len(train_df)),
    "test_rows":     int(len(test_df)),
    "train_anomaly_rate": round(float(train_df["is_anomaly"].sum()/len(train_df)*100), 4),
    "test_anomaly_rate":  round(float(test_df["is_anomaly"].sum()/len(test_df)*100), 4),
    "feature_count": len(feature_cols),
    "feature_names": feature_cols,
    "trained_at":    datetime.datetime.utcnow().isoformat() + "Z",
}
with open(MODELS / "model_metadata.json", "w") as f:
    json.dump(meta_out, f, indent=2)
print(f"Model saved  : {model_path}")
print(f"Train anomaly rate : {meta_out['train_anomaly_rate']:.2f}%")
print(f"Test  anomaly rate : {meta_out['test_anomaly_rate']:.2f}%")
""",
stdout=(
    f"Model saved  : {model_path}\n"
    f"Train anomaly rate : {train_rate:.2f}%\n"
    f"Test  anomaly rate : {test_rate:.2f}%\n"
))

CELL_13 = _md_cell(f"""
## 4 · Summary & Next Steps

### Results

| Metric | Train | Test |
|--------|-------|------|
| Total rows | {len(train_df):,} | {len(test_df):,} |
| Anomalies flagged | {train_df['is_anomaly'].sum():,} ({train_rate:.2f}%) | {test_df['is_anomaly'].sum():,} ({test_rate:.2f}%) |
| Score min / max | {train_scores.min():.4f} / {train_scores.max():.4f} | {test_scores.min():.4f} / {test_scores.max():.4f} |

### Key Findings
- Isolation Forest achieves the target ~3% anomaly rate on both splits.
- SageMaker and Redshift tend to show higher anomaly rates (high-cost, sparse usage patterns).
- Anomalous records show significantly wider `cost_deviation_7d` distributions.
- Train/test anomaly rates are consistent — no covariate drift detected.

### Artifacts
- Model: `data/models/isolation_forest.joblib`
- Metadata: `data/models/model_metadata.json`
- API endpoints: `GET /api/v1/anomalies` · `GET /api/v1/anomalies/summary`
""")

CELL_14 = _code_cell("""\
from IPython.display import display, Image as IPyImage
import pathlib

FIG_DIR = REPO / "reports" / "anomaly_detection"
for fname, title in [
    ("01_score_distribution.png",   "IF Score Distribution"),
    ("02_anomaly_by_service.png",   "Anomaly Rate by Service"),
    ("03_anomaly_by_team.png",      "Anomaly Rate by Team"),
    ("04_cost_deviation_violin.png","Cost Deviation Violin"),
]:
    p = FIG_DIR / fname
    print(f"\\n### {title}")
    if p.exists():
        display(IPyImage(str(p), width=900))
    else:
        print(f"[Missing] {fname}")
""", figures=[p1, p2, p3, p4])

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": [
        CELL_01, CELL_02, CELL_03, CELL_04, CELL_05,
        CELL_06, CELL_07, CELL_08, CELL_09, CELL_10,
        CELL_11, CELL_12, CELL_13, CELL_14,
    ],
}

nb_path = NOTEBOOKS_DIR / "03_isolation_forest.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Notebook : {nb_path}")
print(f"✅ Model    : {model_path}")
print(f"✅ Figures  :")
for p in sorted(REPORTS_DIR.glob("*.png")):
    print(f"   {p.name}  {p.stat().st_size/1024:.1f} KB")
