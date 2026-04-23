## Research Protocol: Heart Failure Incident Cohort — HF_test_run_v1

### Research Question
Can the MAP + LATTE pipeline correctly identify incident Heart Failure (HF) patients
and locate the onset timing from longitudinal MIMIC-IV EHR records, using ~120 Gemini-labeled
gold labels and ONCE-curated codified + NLP features?

This is a test run to validate that all pipeline components work together end-to-end
and to tune LATTE hyperparameters for the HF cohort.

### Study Design
Phenotyping pipeline validation — MAP for cross-sectional case/control assignment,
LATTE for incident timing estimation, Gemini for gold label generation.

### Population
**Dataset:** MIMIC-IV (full), BigQuery backend
**Anchor phenotype:** PheCode:428.1 (Congestive Heart Failure)
**HF ICD codes:** ICD-10 I50.x, ICD-9 428.x

**Inclusion:** Patients with ≥1 HF ICD code AND ≥2 hospital admissions
**Exclusion:** Patients with no ONCE-feature coverage after sparsity filtering (dropped by `preprocess_map`)

### Cohort Definition Approach
**Method:** MAP (Multimodal Automated Phenotyping) → LATTE (incident timing)
**Anchor PheCode:** 428.1 — Congestive Heart Failure
**NLP:** Yes — 115 HF-relevant CUIs extracted from discharge notes via MedSpaCy
**ONCE files:**
  - Codified: `input/ONCE_heart failure_PheCode428.1_cos0.165.csv`
  - Narrative: `input/ONCE_PT_phenotype_heart failure_C0018802_titlecos0.5_titlecut0.3_exactFALSE.csv`
**Features:** 64 codified + 115 NLP CUIs = 179 total
**MAP rationale:** HF ICD specificity is known to be low (coding of HF symptoms vs. true HF);
MAP applies co-feature support to reject probable miscodes.

### Gold Label Strategy
**Gold label count:** ~120 (96 train + 24 test, 80/20 stratified split; or 5-fold CV)
**Method:** Gemini (`gemini-3.1-flash-lite-preview` via Vertex AI)
**Selection:** MAP prefilter (not silver prefilter):
  - Cases pool: MAP phenotype=1 patients
  - Controls pool: MAP phenotype=0 AND icd_coded=True (ICD-present but MAP-rejected)
  - Remaining MAP cohort (~10k patients): unlabeled LATTE input pool
**Rationale for MAP prefilter:** Silver prefilter (LATTE Eq. 1) fails on disease-specific
sub-cohorts — applied to 21k HF patients, everyone has the anchor ICD code, returning 0
high-silver case candidates. MAP posteriors provide reliable case/control signal instead.
**Cache:** `data/gemini_hf_cache.jsonl` — idempotent, re-runs skip already-labeled patients
**Config:** `HF_DECOMP_DISEASE_CONFIG` (recurring decompensation events; criteria: ADHF
diagnosis, EF < 40%, BNP/NT-proBNP, loop diuretics, pulmonary edema, orthopnea, S3 gallop)

### LATTE Configuration (validated via 10-run tuning experiment)
**Baseline date:** 2100-01-01 (study-wide anchor; MIMIC-IV dates are shifted to ~2100–2200)
**Month window:** 3 (LATTE paper default)
**Feature codes:** ONCE feature_codes (64 codified + 115 NLP CUIs)
**Key codes:** LOINC:33762-6, ShortName:BNP (silver label proxy for decompensation)
**Epochs:** 35
**Epoch silver:** 8
**Embedding dim:** 50
**Layers incident:** "80" (single GRU layer — critical for small label sets; see tuning notes)
**weight_unlabel:** 0.015 (scaled to row ratio ≈ 1/72; default 0.2 causes gradient collapse)
**weight_prevalence:** 0.2
**weight_contrastive:** 0.1
**weight_smooth:** 0.1
**weight_additional:** 0.1
**flag_train_augment:** 1
**max_visits:** 25
**min_nonzero:** 20 (MAP feature matrix; set to 5 causes flexmix NaN log-likelihood)

### Hyperparameter Tuning Notes
A 10-run experiment was conducted to tune LATTE for this cohort. Key findings:
1. **Gradient collapse** (Run 1, AUC=0.500): Default `weight_unlabel=0.2` causes 14.4×
   unlabeled gradient dominance (72:1 row ratio with month_window=3). Fix: scale
   `weight_unlabel ≈ n_labeled / n_unlabeled ≈ 0.014`.
2. **LATTE checkpoint bug**: `a_semi_model_final.py` originally saved only the last 2
   epoch checkpoints. Patched to save from `epoch_silver` onward for true best-epoch
   selection. Worth +0.035 AUC.
3. **Single GRU layer** (Run 9, AUC=0.697): Switching from `layers_incident="80,80"` to
   `"80"` gained +0.043 AUC. With only ~30 positive cases the 2-layer GRU overfits.
4. **weight_unlabel and layers interact**: single-layer needs lower weight (0.015);
   dual-layer tolerates higher weight (0.025). Tune jointly.
5. **EPOCHS=35, EPOCH_SILVER=8 optimal**: 50 epochs overfits (best checkpoint at ~43);
   10 silver epochs reduces joint training time with no benefit.

Full tuning results: `logs/SUMMARY.md`

### Analysis Plan
1. `01_cohort_definition.py` — Query MIMIC-IV BigQuery: HF patients + admissions, diagnoses,
   prescriptions, procedures, lab events (ONCE-filtered), discharge notes (batched).
   All tables cache-checked (skipped if parquet already present).
2. `02_notes_nlp.py` — MedSpaCy CUI extraction from discharge notes (up to 3 notes/patient,
   10k chars each). Saves `cui_obs.parquet`. Cache-checked at script start.
3. `03_feature_matrix.py` — Load ONCE features; build obs_log from codified events; merge
   NLP CUIs from `cui_obs.parquet`; run `preprocess_map` (min_nonzero=20); save
   `obs_log.parquet`, `mat_df.parquet`, `once_features_meta.json`.
4. `04_map_phenotyping.py` — Run MAP on mat_df; save `map_results.parquet` with posterior
   scores and phenotype assignments.
5. `05_gold_labels.py` — MAP prefilter to select case/control candidates; fetch notes from
   `notes_raw.parquet` (no BigQuery re-fetch); Gemini labeling with
   `HF_DECOMP_DISEASE_CONFIG`; `decomp_labels_to_latte()` → `gold_labels.parquet`,
   `unlabeled_pool.parquet`.
6. `06_latte_phenotyping.py` — Single 80/20 run: `format_latte_input`, embeddings,
   `run_latte` → `latte_predictions.parquet`.
7. `07_latte_cv.py` — 5-fold stratified CV over all 120 labeled patients. Embeddings
   built once and shared across folds. Per-fold results in `data/cv_results/fold_{k}/`.
   Summary in `data/cv_results/cv_summary.csv`.

### Potential Biases & Limitations
- Gold label set (~120 patients) is smaller than the paper's 172; expect ~0.02–0.05 AUC gap
- Single 24-patient test set (script 06) has high variance; 5-fold CV (script 07) is preferred
- Gemini labeling is approximate LLM-as-clinician, not formal expert adjudication
- MIMIC-IV is a single academic centre; MGB data in the original paper has richer follow-up
- ICD codes are assigned at discharge — not true incidence dates
- Decompensation silver proxy (BNP/NT-proBNP) may be absent in early study years

### Skills Used
- `mimic-preprocessing`: build_obs_log, icd_to_events, drug_to_events, cpt_to_events
- `map-phenotyping`: preprocess_map, run_map
- `latte-phenotyping`: map_prefilter, run_gemini_labeling, decomp_labels_to_latte,
  format_latte_input, build_cooccurrence_embeddings, run_latte
- `notes-nlp`: notes_to_events (MedSpaCy), cui_obs merging
