---
name: map-phenotyping
description: Run the MAP (Multimodal Automated Phenotyping) algorithm on a prepared feature matrix to produce per-patient phenotype probability scores and binary case/control labels. Use after build-datamart to identify a disease cohort from EHR data.
---

# MAP Phenotyping (Module 2, Part B)

## Goal
Run MAP to produce a per-patient phenotype probability score and binary case/control label. MAP fits a two-component Poisson mixture model per feature (case component = higher code count, control component = lower/zero count) and aggregates posteriors into a single score.

## Prerequisites
- `mat_df` and `note_df` from **build-datamart**
- R installed with the `PheCAP` package: `Rscript -e "install.packages('PheCAP')"`

## Step 1 — Run MAP

```python
from map import run_map

map_results = run_map(
    mat_df=mat_df,          # patients × features (all modalities), index named subject_id
    note_df=note_df,        # patients × note_count, same index as mat_df
    main_icd_col=MAIN_PHECODE,  # anchor PheCode string, e.g. '714.1'
)
# Returns: pd.DataFrame columns=[patient_id, score, phenotype]
# score: 0–1 posterior probability of being a case
# phenotype: 1=case, 0=control (MAP data-driven threshold)
```

**Requirements for MAP to fit correctly:**
- `mat_df.index.name` must be `"subject_id"`
- `main_icd_col` must exactly match a column in `mat_df` (bare PheCode string, no prefix)
- All features must have ≥20 non-zero patients (enforced by `build_map_feature_matrix` + post-join sparse filter)
- `note_df` must have no zero values (enforced by `build_note_proxy`)

## Step 2 — Validate the cohort (three mandatory checks)

### Check 1: Score distribution

```python
import matplotlib.pyplot as plt
from vitrine import show

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(map_results["score"], bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("MAP Score Distribution")
axes[0].set_xlabel("Score"); axes[0].set_ylabel("Patients")

counts = map_results["phenotype"].value_counts().sort_index()
axes[1].bar(["Control (0)", "Case (1)"], counts.values, color=["#aec6cf","#ff7f7f"])
axes[1].set_title("MAP Phenotype Labels")
for i, v in enumerate(counts.values):
    axes[1].text(i, v + 5, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("output/map_scores.png")
show(map_results, title="MAP Results")
```

Expected score distribution: bimodal or strongly right-skewed — most patients near 0 (controls), smaller cluster at higher scores (cases). A flat or unimodal distribution means the features aren't separating cases from controls.

### Check 2: Anchor code burden in cases vs controls

```python
scored = map_results.set_index("patient_id")
cases = scored[scored["phenotype"] == 1].index
controls = scored[scored["phenotype"] == 0].index

anchor_counts = mat_df[MAIN_PHECODE].rename("anchor_count")
case_counts = anchor_counts.reindex(cases).fillna(0)
ctrl_counts = anchor_counts.reindex(controls).fillna(0)

print(f"Cases    n={len(cases):,}  mean={case_counts.mean():.2f}  median={case_counts.median():.0f}")
print(f"Controls n={len(controls):,}  mean={ctrl_counts.mean():.2f}  median={ctrl_counts.median():.0f}")
print(f"% cases with ≥1 anchor code: {(case_counts > 0).mean()*100:.1f}%")
```

Expected: cases have substantially higher anchor counts. RA/MIMIC-IV baseline: cases mean=2.39, controls mean=0.03, 100% of cases had ≥1 RA code.

### Check 3: Prevalence plausibility

```python
n_cases = int((map_results["phenotype"] == 1).sum())
prevalence = n_cases / len(map_results) * 100
print(f"Cases: {n_cases:,} / {len(map_results):,} ({prevalence:.1f}%)")
```

Compare against known disease prevalence in inpatient settings. Note: MAP prevalence is computed within the study population (patients with ≥1 ONCE feature), which is already enriched for disease-relevant patients — so observed prevalence will be higher than population prevalence.

## Step 3 — Inspect top-scoring patients

```python
top = (
    scored[["score", "phenotype"]]
    .join(mat_df[[MAIN_PHECODE]].rename(columns={MAIN_PHECODE: "anchor_count"}))
    .join(note_df)
    .sort_values("score", ascending=False)
    .head(20)
)
show(top, title="Top 20 MAP Cases")
```

High scores with low anchor counts indicate patients where NLP or other modality features drove the assignment — these are exactly the cases that simple code-based filters would miss.

## Step 4 — Save the cohort

```python
import os
os.makedirs("output", exist_ok=True)

cohort = map_results[map_results["phenotype"] == 1][["patient_id", "score"]]
cohort.to_csv("output/cohort_map_cases.csv", index=False)
map_results.to_csv("output/map_results_full.csv", index=False)

print(f"Cohort saved: {len(cohort):,} cases")
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `"Log-likelihood: NA"` in R stderr | Feature with <20 non-zero patients | Ensure `build_map_feature_matrix` and post-join sparse filter ran |
| All scores = 0 | `main_icd_col` not in `mat_df` columns | Use bare PheCode string (e.g. `"714.1"` not `"PheCode:714.1"`) |
| 0 cases labeled | Study population too small or features too sparse | Check `mat_df.shape`; lower `min_nonzero` cautiously |
| R package error | `PheCAP` not installed | `Rscript -e "install.packages('PheCAP')"` |
| `RuntimeError` from `run_map` | R script failed | Read the error `.stderr` in the exception message |
| Non-PheCode columns cause R error | MAP anchor must be a PheCode | Verify `main_icd_col` is a PheCode column, not an RXNORM/LNC/CUI column |

## Output contract

`map_results`: `pd.DataFrame`
- `patient_id`: matches `mat_df.index` values
- `score`: float 0–1, MAP posterior probability of case status
- `phenotype`: int 0 or 1, MAP binary label

Use `phenotype == 1` to define the case cohort for downstream analysis (survival, treatment comparison, etc.).
