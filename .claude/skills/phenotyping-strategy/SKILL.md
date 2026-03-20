---
name: phenotyping-strategy
description: Choose between MAP (Multimodal Automated Phenotyping) and a simple rule-based filter for building a disease cohort. Use when starting a new phenotyping task to decide the right approach before writing code.
---

# Phenotyping Strategy

Two approaches are available. Choose based on dataset size, available co-features, and how much precision matters.

## Decision Guide

| Question | MAP | Rule-based |
|---|---|---|
| Need validated, high-specificity cohort? | ✓ | — |
| Have ONCE co-features for the disease? | ✓ | — |
| Dataset has ≥1,000 candidates? | ✓ | — |
| Want per-patient probability scores? | ✓ | — |
| Simple inclusion/exclusion criteria only? | — | ✓ |
| Small dataset (<200 candidates)? | — | ✓ |
| No co-feature data available? | — | ✓ |
| Exploratory / quick check? | — | ✓ |

## Rule-Based Filter

Filter the observation log directly. No model required.

```python
# ICD-based: patients with ≥N anchor PheCode encounters
anchor_counts = (
    icd_obs[icd_obs["event"] == "PheCode:714.1"]
    .groupby("subject_id").size()
)
case_ids = set(anchor_counts[anchor_counts >= 2].index)

# Multi-criteria: require ICD + medication
dmard_patients = set(rx_obs[rx_obs["event"] == "RXNORM:1734340"]["subject_id"])
confirmed = case_ids & dmard_patients
```

**When to use:** exploratory analysis, pilot runs, simple criteria (e.g. ≥2 ICD codes), or when MAP co-features are too sparse to fit reliably.

## MAP

Fits a Poisson mixture model over all ONCE co-features jointly. Assigns each patient a posterior probability and binary phenotype label.

```python
mat_df, note_df = preprocess_map(obs_log, admissions_df, once_features, main_phecode="714.1")
map_results = run_map(mat_df, note_df, "PheCode:714.1")
case_ids = set(map_results[map_results["phenotype"] == 1]["patient_id"])
```

**When to use:** final cohort for a study, when ICD specificity is known to be low (e.g. RA: ~58% PPV for multiple ICD codes), or when NLP features are available to boost sensitivity.

## MAP Without NLP vs With NLP

| Mode | Effect |
|---|---|
| MAP, no NLP | Specificity gain only — rejects weak ICD cases, cannot find new ones |
| MAP + NLP | Specificity + sensitivity — also finds patients under-coded in ICD |

Without NLP, expect `map_only ≈ 0` (MAP finds almost no cases ICD missed). With NLP (`notes_to_events`), `map_only` grows — these are the real sensitivity gains.

## Minimum Requirements for MAP

- **Anchor PheCode** present in `once_features["codified_list"]`
- **≥20 patients** with each co-feature (otherwise MAP drops it; set `min_nonzero`)
- **R + MAP package** installed (`run_map` shells out to Rscript)
- **Observation log** with at least ICD (PheCodes); RxNorm/CUI improve performance
