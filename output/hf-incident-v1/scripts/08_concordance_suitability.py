"""08 — Concordance analysis (silver vs. LATTE refined label) and suitability evaluation.

Study: hf-incident-v1

Concordance: for MAP cases, compare the silver label (first 2-month period with
HF PheCode) to the LATTE-refined incident period (first period where LATTE
probability >= threshold).

Suitability: data density, NLP coverage, washup attrition, sequence length
distribution, and a narrative assessment of pipeline fitness.

Run:
    cd output/hf-incident-v1
    uv run python scripts/08_concordance_suitability.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

out = Path(__file__).resolve().parent.parent
(out / "plots").mkdir(exist_ok=True)

# Incident date detection method:
# - argmax: argmax of probability per patient (bad — 64% of patients have
#   monotone-increasing trajectories, so argmax = last visit)
# - relative_50pct: first period where probability exceeds the patient's
#   midpoint [min + 0.5*(max-min)] — best method (gold-vs-refined median delta = 0)
INCIDENT_METHOD = "relative_50pct"

# ── 1. Load inputs ────────────────────────────────────────────────────────────
print("Loading data...")
map_results = pd.read_parquet(out / "data" / "map_results.parquet")
silver_labels = pd.read_parquet(out / "data" / "silver_labels.parquet")
gemini_results = pd.read_parquet(out / "data" / "gemini_incident_results.parquet")
obs_log = pd.read_parquet(out / "data" / "obs_log.parquet")
obs_log_full = pd.read_parquet(out / "data" / "obs_log_full.parquet")

map_results["patient_id"] = map_results["patient_id"].astype(str)
silver_labels["subject_id"] = silver_labels["subject_id"].astype(str)

meta = json.loads((out / "data" / "once_features_meta.json").read_text())
BASELINE_DATE = meta["baseline_date"]
PERIOD_DAYS = meta["period_days"]

map_cases = set(map_results[map_results["phenotype"] == 1]["patient_id"].astype(str))
print(f"  MAP cases    : {len(map_cases):,}")
print(f"  Silver labels: {len(silver_labels):,}")


# Use tuned CV fold predictions (weight_smooth=0.04, epochs=45 — better AUC 0.844)
# Fall back to baseline CV if tuned results not available
def _load_cv_preds(cv_dir_name):
    parts = []
    for k in range(1, 6):
        p = out / "data" / cv_dir_name / f"fold_{k}" / "predictions.parquet"
        if p.exists():
            fp = pd.read_parquet(p)
            fp["subject_id"] = fp["subject_id"].astype(float).astype(int).astype(str)
            parts.append(fp)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


tuned_exists = (
    out / "data" / "cv_results_tuned" / "fold_1" / "predictions.parquet"
).exists()
cv_dir = "cv_results_tuned" if tuned_exists else "cv_results"
latte_preds = _load_cv_preds(cv_dir)
latte_preds = latte_preds.rename(columns={"visit_T": "T"})
print(
    f"  CV LATTE rows: {len(latte_preds):,}, {latte_preds['subject_id'].nunique()} patients "
    f"({'tuned' if tuned_exists else 'baseline'} CV hold-out)"
)

# ── 2. Derive refined incident date from LATTE predictions ────────────────────
print(f"\nDeriving refined incident dates (method: {INCIDENT_METHOD})...")
# Validated: 64% of patients have monotone-increasing LATTE trajectories, so
# argmax = last visit for most patients. The relative_50pct method (first period
# where prob > min + 0.5*(max-min)) aligns with Gemini gold labels at median delta=0.


def _derive_refined_T(grp):
    probs = grp.sort_values("T")["incident_probability"].values
    ts = grp.sort_values("T")["T"].values
    if INCIDENT_METHOD == "argmax":
        return ts[np.argmax(probs)]
    elif INCIDENT_METHOD == "relative_50pct":
        p_mid = probs.min() + 0.5 * (probs.max() - probs.min())
        above = np.where(probs >= p_mid)[0]
        return ts[above[0]] if len(above) > 0 else ts[-1]
    else:
        raise ValueError(f"Unknown INCIDENT_METHOD: {INCIDENT_METHOD}")


refined_T = (
    latte_preds.groupby("subject_id").apply(_derive_refined_T).rename("T_refined")
)

all_latte_patients = pd.Series(latte_preds["subject_id"].unique(), name="subject_id")
refined_df = all_latte_patients.to_frame().merge(refined_T, on="subject_id", how="left")

print(
    f"  Patients with refined incident date: {refined_df['T_refined'].notna().sum():,}"
)
print(f"  T_refined range: [{refined_T.min():.0f}, {refined_T.max():.0f}]")

# ── 3. Concordance analysis ───────────────────────────────────────────────────
print("\nConcordance analysis (MAP cases only)...")

# Build concordance table: MAP cases with both silver and refined labels
conc = silver_labels[silver_labels["subject_id"].isin(map_cases)].merge(
    refined_df.rename(columns={"subject_id": "subject_id"}),
    on="subject_id",
    how="inner",
)
conc = conc.dropna(subset=["T_silver", "T_refined"])
conc["delta"] = conc["T_refined"] - conc["T_silver"]


def concordance_category(delta):
    if delta == 0:
        return "concordant"
    elif delta < 0:
        return "earlier"
    else:
        return "later"


conc["category"] = conc["delta"].apply(concordance_category)

cat_counts = conc["category"].value_counts()
cat_pct = (conc["category"].value_counts(normalize=True) * 100).round(1)

print(f"\n  Total MAP cases with both labels: {len(conc):,}")
print(
    f"  Concordant (delta=0)  : {cat_counts.get('concordant', 0):>5,}  ({cat_pct.get('concordant', 0):.1f}%)"
)
print(
    f"  Earlier (LATTE < silver): {cat_counts.get('earlier', 0):>5,}  ({cat_pct.get('earlier', 0):.1f}%)"
)
print(
    f"  Later (LATTE > silver)  : {cat_counts.get('later', 0):>5,}  ({cat_pct.get('later', 0):.1f}%)"
)

earlier = conc[conc["category"] == "earlier"]["delta"]
later = conc[conc["category"] == "later"]["delta"]
print(
    f"\n  Median lead (earlier): {-earlier.median():.1f} periods ({-earlier.median() * PERIOD_DAYS / 30:.1f} months)"
)
print(
    f"  Median lag  (later)  : {later.median():.1f} periods ({later.median() * PERIOD_DAYS / 30:.1f} months)"
)

# Gold-label concordance (ground-truth check for labeled subset)
gemini_valid = gemini_results[
    ~gemini_results["parse_error"] & (gemini_results["label"] == 1)
].copy()
gemini_valid["subject_id"] = gemini_valid["subject_id"].astype(str)
gold_conc = (
    gemini_valid[["subject_id", "incident_T"]]
    .rename(columns={"incident_T": "T_gold"})
    .merge(silver_labels, on="subject_id", how="inner")
    .merge(refined_df, on="subject_id", how="inner")
    .dropna(subset=["T_gold", "T_silver", "T_refined"])
)
if len(gold_conc) > 0:
    gold_conc["delta_gold_vs_silver"] = gold_conc["T_gold"] - gold_conc["T_silver"]
    gold_conc["delta_gold_vs_refined"] = gold_conc["T_gold"] - gold_conc["T_refined"]
    print(f"\n  Gold-labeled patients ({len(gold_conc)}):")
    print(
        f"    Gold vs silver  — median delta: {gold_conc['delta_gold_vs_silver'].median():.1f} periods"
    )
    print(
        f"    Gold vs refined — median delta: {gold_conc['delta_gold_vs_refined'].median():.1f} periods"
    )
    gold_conc.to_parquet(out / "data" / "gold_concordance.parquet", index=False)

    fig_gold_delta = px.histogram(
        gold_conc,
        x="delta_gold_vs_refined",
        nbins=30,
        title="Gemini Gold − LATTE Refined Incident Period (2-month periods)",
        labels={"delta_gold_vs_refined": "Δ periods (Gemini − LATTE refined)"},
        color_discrete_sequence=["#AB63FA"],
    )
    fig_gold_delta.add_vline(
        x=0, line_dash="dash", line_color="red", annotation_text="Perfect agreement"
    )
    fig_gold_delta.write_json(out / "plots" / "gemini_vs_latte_delta.json")
    print("  plots/gemini_vs_latte_delta.json")

conc.to_parquet(out / "data" / "concordance.parquet", index=False)

# Plot: concordance bar chart
fig_conc = go.Figure(
    go.Bar(
        x=["Concordant", "Earlier (LATTE)", "Later (LATTE)"],
        y=[
            cat_counts.get("concordant", 0),
            cat_counts.get("earlier", 0),
            cat_counts.get("later", 0),
        ],
        marker_color=["#2CA02C", "#1F77B4", "#FF7F0E"],
        text=[f"{cat_pct.get(c, 0):.1f}%" for c in ["concordant", "earlier", "later"]],
        textposition="outside",
    )
)
fig_conc.update_layout(
    title="Silver vs. LATTE Refined Label Concordance (MAP cases)",
    xaxis_title="Category",
    yaxis_title="Number of patients",
)
fig_conc.write_json(out / "plots" / "concordance_distribution.json")

# Plot: delta histogram
fig_delta = px.histogram(
    conc,
    x="delta",
    nbins=40,
    title="LATTE Refined − Silver Label Delta (2-month periods)",
    labels={"delta": "Δ periods (refined − silver)"},
    color_discrete_sequence=["#636EFA"],
)
fig_delta.add_vline(
    x=0, line_dash="dash", line_color="red", annotation_text="Silver label"
)
fig_delta.write_json(out / "plots" / "label_delta_histogram.json")

# ── 4. Suitability evaluation ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUITABILITY EVALUATION")
print("=" * 60)

# 4a. Data density (obs_log, washup-filtered)
base_dt = pd.to_datetime(BASELINE_DATE)
obs_log["_dt"] = pd.to_datetime(obs_log["datetime"], errors="coerce")
obs_log["T"] = np.floor((obs_log["_dt"] - base_dt).dt.days / PERIOD_DAYS).astype(
    "Int64"
)

features_per_period = (
    obs_log.dropna(subset=["T"])
    .groupby(["subject_id", "T"])["event"]
    .nunique()
    .reset_index(name="n_features")
)
median_feat = features_per_period["n_features"].median()
pct_sparse = (features_per_period["n_features"] < 3).mean() * 100
total_periods = features_per_period.groupby("subject_id")["T"].nunique()

print("\n  Data density:")
print(f"    Median unique features per period: {median_feat:.1f}")
print(f"    % sparse periods (< 3 features)  : {pct_sparse:.1f}%")
print(f"    Median total periods per patient  : {total_periods.median():.1f}")
print(
    f"    p10 / p90 total periods           : {total_periods.quantile(0.1):.0f} / {total_periods.quantile(0.9):.0f}"
)

# 4b. NLP signal — fraction with any CUI event pre-silver-label
nlp_obs = (
    obs_log[obs_log["event_type"] == "cui"].copy()
    if "event_type" in obs_log.columns
    else pd.DataFrame()
)
if not nlp_obs.empty:
    nlp_obs["_dt"] = pd.to_datetime(nlp_obs["datetime"], errors="coerce")
    nlp_obs["T"] = np.floor((nlp_obs["_dt"] - base_dt).dt.days / PERIOD_DAYS).astype(
        "Int64"
    )
    nlp_with_silver = nlp_obs.merge(
        silver_labels[["subject_id", "T_silver"]], on="subject_id", how="inner"
    )
    nlp_pre = nlp_with_silver[nlp_with_silver["T"] < nlp_with_silver["T_silver"]]
    pct_nlp_pre = nlp_pre["subject_id"].nunique() / len(silver_labels) * 100
    print("\n  NLP signal:")
    print(f"    % patients with NLP CUI event PRE-silver-label: {pct_nlp_pre:.1f}%")
else:
    print("\n  NLP signal: no event_type='cui' column found")
    pct_nlp_pre = float("nan")

# 4c. Washup attrition — from exploration
n_raw_hf = 31_369
n_post_washup = len(silver_labels)
pct_excluded = (n_raw_hf - n_post_washup) / n_raw_hf * 100
print("\n  Washup attrition:")
print(f"    Raw HF candidates     : {n_raw_hf:,}")
print(f"    Post-washup cohort    : {n_post_washup:,}")
print(f"    % excluded (prevalent): {pct_excluded:.1f}%")

# 4d. LATTE sequence length coverage (>= 3 total periods)
pct_latte_ready = (total_periods >= 3).mean() * 100
print("\n  LATTE sequence coverage:")
print(f"    % patients with >= 3 periods (LATTE-ready): {pct_latte_ready:.1f}%")
print(
    f"    % patients with >= 5 periods               : {(total_periods >= 5).mean() * 100:.1f}%"
)

# 4e. Silver label timing — time from MIMIC entry to first HF code
print("\n  Silver label timing:")
print(
    f"    Median T_silver: {silver_labels['T_silver'].median():.1f} periods from baseline"
)
print("    n_pre_periods dist:")
print(
    f"    {silver_labels['n_pre_periods'].value_counts().sort_index().head(10).to_string()}"
)

# ── 5. Suitability summary plot ───────────────────────────────────────────────
metrics = {
    "Washup attrition (prevalent excluded, %)": pct_excluded,
    "LATTE-ready patients (≥3 periods, %)": pct_latte_ready,
    "Sparse periods (< 3 features, %)": pct_sparse,
    "NLP pre-silver coverage (%)": pct_nlp_pre,
}
fig_suit = go.Figure(
    go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        marker_color=["#EF553B", "#2CA02C", "#FFA15A", "#AB63FA"],
        text=[f"{v:.1f}%" for v in metrics.values()],
        textposition="outside",
    )
)
fig_suit.update_layout(
    title="Suitability Metrics — MAP + LATTE for HF Incident Timing in MIMIC-IV",
    yaxis_title="%",
    yaxis_range=[0, 110],
)
fig_suit.write_json(out / "plots" / "suitability_summary.json")

# ── 6. Feature density plot ───────────────────────────────────────────────────
fig_density = px.histogram(
    features_per_period,
    x="n_features",
    nbins=50,
    title="Unique Features per 2-Month Period (washup-filtered cohort)",
    labels={"n_features": "Unique features in period"},
    color_discrete_sequence=["#636EFA"],
)
fig_density.write_json(out / "plots" / "data_density.json")

# ── 7. CV summary if available ───────────────────────────────────────────────
print("\n  CV Results:")
for label, path in [
    (
        "Baseline (epochs=35, smooth=0.1)",
        out / "data" / "cv_results" / "cv_summary.csv",
    ),
    (
        "Tuned   (epochs=45, smooth=0.04)",
        out / "data" / "cv_results_tuned" / "cv_summary_tuned.csv",
    ),
]:
    if path.exists():
        df = pd.read_csv(path)
        a = df["auc"].dropna()
        print(
            f"    {label}: {a.mean():.4f} ± {a.std():.4f}  (min={a.min():.4f}, max={a.max():.4f})"
        )

# ── 8. CV AUC bar chart ───────────────────────────────────────────────────────
tuned_cv_path = out / "data" / "cv_results_tuned" / "cv_summary_tuned.csv"
if tuned_cv_path.exists():
    cv_df = pd.read_csv(tuned_cv_path)
    aucs = cv_df["auc"].dropna()
    mean_auc = aucs.mean()
    std_auc = aucs.std()

    fig_auc = go.Figure()
    fig_auc.add_trace(
        go.Bar(
            x=[f"Fold {int(f)}" for f in cv_df["fold"]],
            y=cv_df["auc"],
            marker_color="#636EFA",
            text=[f"{a:.3f}" for a in cv_df["auc"]],
            textposition="outside",
        )
    )
    fig_auc.add_hline(
        y=mean_auc,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean = {mean_auc:.3f} ± {std_auc:.3f}",
        annotation_position="bottom right",
    )
    fig_auc.update_layout(
        title="5-Fold CV AUC — LATTE Incident HF (tuned: epochs=45, weight_smooth=0.04)",
        xaxis_title="Fold",
        yaxis_title="AUC",
        yaxis_range=[0, 1.05],
        showlegend=False,
    )
    fig_auc.write_json(out / "plots" / "cv_auc.json")
    print("  plots/cv_auc.json")

# ── 9. Gemini-vs-silver vs LATTE-vs-silver concordance comparison ─────────────
if len(gold_conc) > 0:
    gold_conc["cat_vs_silver"] = gold_conc["delta_gold_vs_silver"].apply(
        concordance_category
    )
    gem_pct = (gold_conc["cat_vs_silver"].value_counts(normalize=True) * 100).round(1)

    ordered = ["earlier", "concordant", "later"]
    labels = ["Earlier", "Concordant", "Later"]
    latte_vals = [cat_pct.get(c, 0) for c in ordered]
    gemini_vals = [gem_pct.get(c, 0) for c in ordered]

    fig_comp = go.Figure(
        data=[
            go.Bar(
                name=f"LATTE refined vs silver (n={len(conc)})",
                x=labels,
                y=latte_vals,
                marker_color="#636EFA",
                text=[f"{v:.1f}%" for v in latte_vals],
                textposition="outside",
            ),
            go.Bar(
                name=f"Gemini gold vs silver (n={len(gold_conc)})",
                x=labels,
                y=gemini_vals,
                marker_color="#EF553B",
                text=[f"{v:.1f}%" for v in gemini_vals],
                textposition="outside",
            ),
        ]
    )
    fig_comp.update_layout(
        barmode="group",
        title="Concordance vs Silver Label: LATTE refined vs Gemini gold",
        yaxis_title="%",
        yaxis_range=[0, 110],
    )
    fig_comp.write_json(out / "plots" / "concordance_comparison.json")
    print("  plots/concordance_comparison.json")

    gem_earlier_median = -gold_conc[gold_conc["cat_vs_silver"] == "earlier"][
        "delta_gold_vs_silver"
    ].median()
    print(f"\n  Gemini vs silver concordance (n={len(gold_conc)}):")
    print(
        f"    Earlier   : {gem_pct.get('earlier', 0):.1f}%  (median lead: {gem_earlier_median:.1f} periods)"
    )
    print(f"    Concordant: {gem_pct.get('concordant', 0):.1f}%")
    print(f"    Later     : {gem_pct.get('later', 0):.1f}%")

print("\nDone. Plots saved:")
print("  plots/concordance_distribution.json")
print("  plots/label_delta_histogram.json")
print("  plots/suitability_summary.json")
print("  plots/data_density.json")
