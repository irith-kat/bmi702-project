"""00 — Protocol exploration for hf-incident-v1.

Checks key data characteristics that could influence protocol decisions:
  1. Candidate pool size (HF ICD code holders)
  2. Age exclusion impact (< 18)
  3. Washup attrition (< 2 two-month periods before first HF code)
  4. Data-beyond-anchor exclusion (no non-HF signal)
  5. Period-length sensitivity (per-patient sequence lengths)
  6. ONCE file contents (codified feature count, NLP CUI count)
  7. Notes availability in BigQuery

Run:
    cd output/hf-incident-v1
    uv run python scripts/00_explore.py
"""

import glob
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from m4 import execute_query, set_dataset
from m4.config import set_active_backend
from preprocessing.nlp import get_once_features

REPO_ROOT = Path(__file__).resolve().parents[4]
out = Path(__file__).resolve().parent.parent
(out / "data").mkdir(exist_ok=True)
(out / "plots").mkdir(exist_ok=True)

PERIOD_DAYS = 60  # 2-month aggregation window

# ── 0. ONCE files ──────────────────────────────────────────────────────────────
print("=" * 60)
print("ONCE file inspection")
print("=" * 60)
codified_files = sorted(glob.glob(str(REPO_ROOT / "input" / "ONCE_*PheCode*.csv")))
narrative_files = sorted(glob.glob(str(REPO_ROOT / "input" / "ONCE_*_C[0-9]*.csv")))
codified_file = next(
    f for f in codified_files if "428" in f and "heart failure" in f.lower()
)
narrative_file = next(
    f for f in narrative_files if "heart failure" in f.lower() and "C0018802" in f
)
once = get_once_features(codified_file, narrative_file)
print(f"  Codified features : {len(once['codified_list'])}")
print(f"  NLP CUI targets   : {len(once['nlp_target_cuis'])}")
print(f"  Top 10 codified   : {once['codified_list'][:10]}")
print(f"  Top 10 CUIs       : {list(once['nlp_target_cuis'])[:10]}")

# ── 1. Codified data: DuckDB ───────────────────────────────────────────────────
set_active_backend("duckdb")
set_dataset("mimic-iv")

print("\n" + "=" * 60)
print("DuckDB: candidate pool and exclusion cascade")
print("=" * 60)

# 1a. All patients with an HF ICD code (I50% or 428%)
print("\n[1] HF candidate pool...")
hf_candidates = execute_query("""
    SELECT DISTINCT
        CAST(d.subject_id AS VARCHAR) AS subject_id
    FROM mimiciv_hosp.diagnoses_icd d
    WHERE
        (d.icd_version = 10 AND d.icd_code LIKE 'I50%')
        OR
        (d.icd_version = 9  AND d.icd_code LIKE '428%')
""")
n_raw = len(hf_candidates)
print(f"  All patients with HF ICD code: {n_raw:,}")

# 1b. Age at first admission
print("\n[2] Age exclusion (< 18)...")
age_df = execute_query("""
    WITH hf AS (
        SELECT DISTINCT CAST(subject_id AS VARCHAR) AS subject_id
        FROM mimiciv_hosp.diagnoses_icd
        WHERE (icd_version = 10 AND icd_code LIKE 'I50%')
           OR (icd_version = 9  AND icd_code LIKE '428%')
    ),
    first_admit AS (
        SELECT
            CAST(a.subject_id AS VARCHAR) AS subject_id,
            MIN(a.admittime) AS first_admittime
        FROM mimiciv_hosp.admissions a
        GROUP BY a.subject_id
    )
    SELECT
        h.subject_id,
        p.anchor_age,
        fa.first_admittime
    FROM hf h
    JOIN mimiciv_hosp.patients p ON CAST(p.subject_id AS VARCHAR) = h.subject_id
    JOIN first_admit fa ON fa.subject_id = h.subject_id
""")
n_adult = (age_df["anchor_age"] >= 18).sum()
n_ped = n_raw - n_adult
print(f"  Age < 18 excluded: {n_ped:,}  ({n_ped / n_raw * 100:.1f}%)")
print(f"  Adults remaining : {n_adult:,}")

# 1c. Washup analysis — periods before first HF code
print("\n[3] Washup analysis (2-month periods before first HF code)...")
washup_df = execute_query("""
    WITH hf AS (
        SELECT DISTINCT CAST(subject_id AS VARCHAR) AS subject_id
        FROM mimiciv_hosp.diagnoses_icd
        WHERE (icd_version = 10 AND icd_code LIKE 'I50%')
           OR (icd_version = 9  AND icd_code LIKE '428%')
    ),
    patients_adult AS (
        SELECT h.subject_id
        FROM hf h
        JOIN mimiciv_hosp.patients p ON CAST(p.subject_id AS VARCHAR) = h.subject_id
        WHERE p.anchor_age >= 18
    ),
    first_admit AS (
        SELECT
            CAST(a.subject_id AS VARCHAR) AS subject_id,
            MIN(a.admittime) AS first_admittime
        FROM mimiciv_hosp.admissions a
        GROUP BY a.subject_id
    ),
    first_hf_admit AS (
        SELECT
            CAST(d.subject_id AS VARCHAR) AS subject_id,
            MIN(a.admittime) AS first_hf_admittime
        FROM mimiciv_hosp.diagnoses_icd d
        JOIN mimiciv_hosp.admissions a ON d.hadm_id = a.hadm_id
        WHERE (d.icd_version = 10 AND d.icd_code LIKE 'I50%')
           OR (d.icd_version = 9  AND d.icd_code LIKE '428%')
        GROUP BY d.subject_id
    )
    SELECT
        pa.subject_id,
        fa.first_admittime,
        fha.first_hf_admittime,
        DATEDIFF('day', fa.first_admittime, fha.first_hf_admittime) AS days_before_hf
    FROM patients_adult pa
    JOIN first_admit fa     ON fa.subject_id  = pa.subject_id
    JOIN first_hf_admit fha ON fha.subject_id = pa.subject_id
""")

# Compute number of complete 2-month periods before first HF
washup_df["periods_before_hf"] = (washup_df["days_before_hf"] / PERIOD_DAYS).astype(int)
n_pass_washup = (washup_df["periods_before_hf"] >= 2).sum()
n_fail_washup = n_adult - n_pass_washup
print(
    f"  Patients with 0 periods before HF (prevalent at entry): {(washup_df['periods_before_hf'] == 0).sum():,}"
)
print(
    f"  Patients with 1 period before HF                      : {(washup_df['periods_before_hf'] == 1).sum():,}"
)
print(
    f"  Patients with >= 2 periods before HF (pass washup)    : {n_pass_washup:,}  ({n_pass_washup / n_adult * 100:.1f}%)"
)
print(f"  Washup excluded: {n_fail_washup:,}  ({n_fail_washup / n_adult * 100:.1f}%)")

# Distribution of periods before HF (capped at 20 for readability)
period_dist = washup_df["periods_before_hf"].clip(upper=20).value_counts().sort_index()
print("\n  Periods-before-HF distribution (0–20+):")
for k, v in period_dist.items():
    label = str(k) if k < 20 else "20+"
    print(f"    {label:>4}: {v:,}")

# Total MIMIC history length
total_hist = execute_query("""
    WITH hf AS (
        SELECT DISTINCT CAST(subject_id AS VARCHAR) AS subject_id
        FROM mimiciv_hosp.diagnoses_icd
        WHERE (icd_version = 10 AND icd_code LIKE 'I50%')
           OR (icd_version = 9  AND icd_code LIKE '428%')
    ),
    patients_adult AS (
        SELECT h.subject_id
        FROM hf h
        JOIN mimiciv_hosp.patients p ON CAST(p.subject_id AS VARCHAR) = h.subject_id
        WHERE p.anchor_age >= 18
    ),
    history AS (
        SELECT
            CAST(a.subject_id AS VARCHAR) AS subject_id,
            DATEDIFF('day', MIN(a.admittime), MAX(a.dischtime)) AS total_days
        FROM mimiciv_hosp.admissions a
        GROUP BY a.subject_id
    )
    SELECT pa.subject_id, h.total_days
    FROM patients_adult pa
    JOIN history h ON h.subject_id = pa.subject_id
""")
print("\n[4] Patient history length (adult HF candidates):")
print(
    f"  Median total MIMIC history: {total_hist['total_days'].median():.0f} days ({total_hist['total_days'].median() / 30:.1f} months)"
)
print(
    f"  p10 / p90: {total_hist['total_days'].quantile(0.1):.0f} / {total_hist['total_days'].quantile(0.9):.0f} days"
)

# 1d. Data beyond anchor check
print("\n[5] Non-HF signal check (among washup-passing adults)...")
passing_ids = washup_df[washup_df["periods_before_hf"] >= 2]["subject_id"].tolist()
# Check in batches — just count patients who have any non-HF diagnosis
BATCH = 2000
non_hf_counts = []
for i in range(0, len(passing_ids), BATCH):
    batch = passing_ids[i : i + BATCH]
    id_list = ", ".join(f"'{sid}'" for sid in batch)
    res = execute_query(f"""
        SELECT CAST(subject_id AS VARCHAR) AS subject_id, COUNT(*) AS n_non_hf_diag
        FROM mimiciv_hosp.diagnoses_icd
        WHERE CAST(subject_id AS VARCHAR) IN ({id_list})
          AND NOT ((icd_version = 10 AND icd_code LIKE 'I50%')
               OR  (icd_version = 9  AND icd_code LIKE '428%'))
        GROUP BY subject_id
    """)
    non_hf_counts.append(res)
non_hf_df = pd.concat(non_hf_counts, ignore_index=True)
n_with_other = non_hf_df["subject_id"].nunique()
n_without_other = n_pass_washup - n_with_other
print(
    f"  Washup-passing patients with non-HF diagnoses: {n_with_other:,}  ({n_with_other / n_pass_washup * 100:.1f}%)"
)
print(
    f"  Patients with ONLY HF diagnosis codes (to exclude): {n_without_other:,}  ({n_without_other / n_pass_washup * 100:.1f}%)"
)

# 1e. Number of admissions per washup-passing patient
print("\n[6] Admission count per washup-passing patient...")
adm_count = execute_query("""
    WITH hf AS (
        SELECT DISTINCT CAST(subject_id AS VARCHAR) AS subject_id
        FROM mimiciv_hosp.diagnoses_icd
        WHERE (icd_version = 10 AND icd_code LIKE 'I50%')
           OR (icd_version = 9  AND icd_code LIKE '428%')
    ),
    patients_adult AS (
        SELECT h.subject_id
        FROM hf h
        JOIN mimiciv_hosp.patients p ON CAST(p.subject_id AS VARCHAR) = h.subject_id
        WHERE p.anchor_age >= 18
    )
    SELECT
        CAST(a.subject_id AS VARCHAR) AS subject_id,
        COUNT(DISTINCT a.hadm_id) AS n_admissions
    FROM mimiciv_hosp.admissions a
    WHERE CAST(a.subject_id AS VARCHAR) IN (SELECT subject_id FROM patients_adult)
    GROUP BY a.subject_id
""")
# Filter to washup-passing
adm_count_washed = adm_count[adm_count["subject_id"].isin(passing_ids)]
print(
    f"  Median admissions (washup-passing): {adm_count_washed['n_admissions'].median():.1f}"
)
print(f"  1 admission: {(adm_count_washed['n_admissions'] == 1).sum():,}")
print(f"  2 admissions: {(adm_count_washed['n_admissions'] == 2).sum():,}")
print(f"  >= 3 admissions: {(adm_count_washed['n_admissions'] >= 3).sum():,}")

# ── 2. Notes: BigQuery ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BigQuery: notes availability")
print("=" * 60)
set_active_backend("bigquery")
set_dataset("mimic-iv-note")

# Sample of washup-passing patients to check note availability
sample_ids = passing_ids[:400]
id_list = ", ".join(str(sid) for sid in sample_ids)
notes_check = execute_query(f"""
    SELECT
        CAST(subject_id AS STRING) AS subject_id,
        COUNT(*) AS n_notes
    FROM mimiciv_note.discharge
    WHERE subject_id IN ({id_list})
    GROUP BY subject_id
""")
n_with_notes = notes_check["subject_id"].nunique()
pct_notes = n_with_notes / len(sample_ids) * 100
print(f"  Sample: {len(sample_ids)} washup-passing patients")
print(
    f"  Patients with >= 1 discharge note: {n_with_notes} / {len(sample_ids)} ({pct_notes:.1f}%)"
)
print(f"  Median notes per patient: {notes_check['n_notes'].median():.1f}")
print(f"  Max notes per patient: {notes_check['n_notes'].max()}")
print(
    f"  (Extrapolated full cohort estimate: ~{int(n_pass_washup * pct_notes / 100):,} patients with notes)"
)

# ── 3. CONSORT summary table ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONSORT attrition summary")
print("=" * 60)
steps = [
    ("All patients with HF ICD code", n_raw),
    ("After age >= 18 filter", n_adult),
    ("After >= 2 periods washup filter", n_pass_washup),
    (
        "After non-HF signal filter (estimated)",
        n_with_other * n_pass_washup // len(passing_ids[:400])
        if len(passing_ids) > 400
        else n_with_other,
    ),
]
for label, n in steps:
    print(f"  {n:>7,}  {label}")

# ── 4. Save data and plots ─────────────────────────────────────────────────────
summary = {
    "n_raw_hf_candidates": n_raw,
    "n_adult": n_adult,
    "n_peds_excluded": n_ped,
    "n_pass_washup": n_pass_washup,
    "n_fail_washup": n_fail_washup,
    "n_with_non_hf_signal": n_with_other,
    "n_anchor_only": n_without_other,
    "median_days_history": float(total_hist["total_days"].median()),
    "pct_with_notes_sample": pct_notes,
    "period_days": PERIOD_DAYS,
    "once_codified_features": len(once["codified_list"]),
    "once_nlp_cuis": len(once["nlp_target_cuis"]),
}
pd.Series(summary).to_frame("value").to_parquet(
    out / "data" / "explore_summary.parquet"
)

# Plot: washup distribution
washup_dist = (
    washup_df["periods_before_hf"]
    .clip(upper=15)
    .value_counts()
    .sort_index()
    .reset_index()
)
washup_dist.columns = ["periods_before_hf", "n_patients"]
fig_washup = go.Figure(
    go.Bar(
        x=washup_dist["periods_before_hf"].astype(str),
        y=washup_dist["n_patients"],
        marker_color=[
            "#EF553B" if x < 2 else "#636EFA" for x in washup_dist["periods_before_hf"]
        ],
    )
)
fig_washup.update_layout(
    title="Distribution of 2-Month Periods Before First HF Code (adult candidates)",
    xaxis_title="Number of 2-month periods before first HF code",
    yaxis_title="Number of patients",
    annotations=[
        dict(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text="Red = excluded by washup rule (< 2 periods)",
            showarrow=False,
            font=dict(size=11),
        )
    ],
)
fig_washup.write_json(out / "plots" / "washup_distribution.json")
print("\nSaved: plots/washup_distribution.json")

# Plot: CONSORT flow
fig_consort = go.Figure(
    go.Funnel(
        y=[s[0] for s in steps],
        x=[s[1] for s in steps],
        textinfo="value+percent initial",
    )
)
fig_consort.update_layout(title="CONSORT Attrition — HF Candidate Pool (Exploration)")
fig_consort.write_json(out / "plots" / "explore_consort.json")
print("Saved: plots/explore_consort.json")

print("\n" + "=" * 60)
print("Exploration complete.")
print("=" * 60)
for k, v in summary.items():
    print(f"  {k:<35}: {v}")
