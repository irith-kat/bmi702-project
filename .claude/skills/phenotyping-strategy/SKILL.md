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

```
MAP cohort → map_prefilter()  → gold label candidates (cases + optional controls)
                    ↓
          run_gemini_labeling()   ← Gemini LLM reviews discharge notes
          labels_to_latte()       ← per-visit (subject_id, T, Y) labels
                    ↓
          format_latte_input()    ← train/test/unlabeled CSVs
          build_cooccurrence_embeddings()
          run_latte()             ← per-patient, per-visit incident probability
```

**When it adds value:**
- Time-to-event outcomes or survival analysis
- Drug safety studies requiring a pre-disease baseline window
- Early detection: LATTE flags onset before the ICD code appears

**Key parameters:**
- `baseline_date`: fixed study anchor (e.g. `"2100-01-01"` for MIMIC-IV)
- `month_window=3`: 3-month bins (LATTE paper default)
- `key_codes`: PheCode anchor(s) — must be in ONCE feature list
- Gold labels: ~30–150 cases; controls (MAP-rejected ICD patients) strengthen supervised loss but can be 0 if cohort is disease-specific

**See:** `latte-phenotyping` skill for full implementation details.

### Minimum Requirements for LATTE

- **Gemini API access** (Vertex AI, `global` region for `gemini-3.1-flash-lite-preview`)
- **Discharge notes available**
