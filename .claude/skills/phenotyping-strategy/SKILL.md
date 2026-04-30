---
name: phenotyping-strategy
description: Choose between rule-based filter, MAP (Multimodal Automated Phenotyping), or MAP + LATTE (incident timing) for building a disease cohort. Use when starting a new phenotyping task to decide the right approach before writing code.
---

# Phenotyping Strategy

Three approaches are available. Choose based on the research question (ever/never vs. when), dataset size, and available co-features.

## Decision Guide

| Question | Rule-based | MAP | MAP + LATTE |
|---|---|---|---|
| Need validated, high-specificity cohort? | — | ✓ | ✓ |
| Need to know **when** disease first occurred? | — | — | ✓ |
| Time-to-event / incidence analysis planned? | — | — | ✓ |
| Have ONCE co-features for the disease? | — | ✓ | ✓ |
| Dataset has ≥1,000 candidates? | — | ✓ | ✓ |
| Want per-patient probability scores? | — | ✓ | ✓ |
| Can obtain ~30–200 Gemini gold labels? | — | — | ✓ |
| Simple inclusion/exclusion criteria only? | ✓ | — | — |
| Small dataset (<200 candidates)? | ✓ | — | — |
| No co-feature data available? | ✓ | — | — |
| Exploratory / quick check? | ✓ | — | — |

## Vocabulary:
If you encounter an unknown vocabulary in the database that is not translatable with the preprocessing tools, and you need it to build the cohort, you can stop the research session and ask the user to look for a mapping file (custom, ATHENA...) or create it. Once the new mapping is provide, you can follow the `custom-vocab-mapping` skill to add it to the pipeline and continue with the planning.

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

**When to use:** final cohort for a study, when ICD specificity is known to be low, or when NLP features are available to boost sensitivity.

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

---

## MAP + LATTE (Incident Timing)

Use when the research question is **longitudinal**: not just who has the disease, but when they first developed it, some disease specific event... LATTE is a semi-supervised GRU that learns from ~100–200 gold-labeled patients (cases + controls) and a larger silver-label pool.

**When it adds value:**
- Time-to-event outcomes or survival analysis
- Drug safety studies requiring a pre-disease baseline window
- Early detection: LATTE flags onset before the ICD code appears

**Key parameters:**
- `baseline_date`: fixed study anchor (e.g. `"2100-01-01"` for MIMIC-IV)
- `key_codes`: PheCode anchor(s) — must be in ONCE feature list

**See:** `latte-phenotyping` skill for full implementation details.

### Minimum Requirements for LATTE

- **Gemini API access** (Vertex AI, `global` region for `gemini-3.1-flash-lite-preview`)
- **Discharge notes available**
