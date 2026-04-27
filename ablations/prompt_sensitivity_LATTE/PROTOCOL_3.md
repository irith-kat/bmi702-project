## Research Protocol: Heart Failure Incident Cohort — heart-failure-incident-v1

### Research Question
Among all MIMIC-IV patients with ≥1 Heart Failure ICD code who have at least two
3-month observation periods before their first HF code, can the MAP + LATTE pipeline
identify the incident onset date of Heart Failure, and how does that refined timing
compare to the silver label (first ICD code date)?

### Study Design
Cohort identification + incident timing + label comparison + suitability evaluation.

### Population
**Source:** MIMIC-IV (full), BigQuery backend
**Anchor phenotype:** PheCode:428.1 — Congestive Heart Failure
**HF ICD codes:** ICD-10 I50.x, ICD-9 428.x

**Inclusion:**
- ≥1 HF ICD code (I50.x or 428.x) at any admission
- ≥2 distinct 3-month periods containing ≥1 observation event BEFORE the first HF ICD date

**Exclusion:**
- Patients whose first HF code appears at their very first observable period (no look-back)
- Patients excluded by MAP's sparsity filter (min_nonzero=20 feature threshold)

**Silver label (anchor timing):** Earliest `admittime` of any admission carrying I50.x or 428.x,
collapsed to the corresponding 3-month period T_silver.

### Cohort Definition Approach
**Method:** MAP (Multimodal Automated Phenotyping) → LATTE (incident timing)
**Anchor PheCode:** 428.1 — Congestive Heart Failure
**NLP:** Yes — MedSpaCy CUI extraction from discharge summaries
**ONCE files:**
  - Codified: `input/ONCE_heart failure_PheCode428.1_cos0.165.csv`
  - Narrative: `input/ONCE_PT_phenotype_heart failure_C0018802_titlecos0.5_titlecut0.3_exactFALSE.csv`
**MAP rationale:** HF ICD specificity is low (HF symptoms coded vs true clinical HF);
MAP uses ONCE co-feature support to separate confirmed cases from probable miscodes.
**MAP config:** min_nonzero=20; NLP contribution expected moderate (HF CUIs prevalent in notes).

### Gold Label Strategy
**Mode:** Incident timing — identify FIRST HF admission per patient
**Config:** `HF_DISEASE_CONFIG` (incident, not decompensation)
**Gold label count:** ~120 MAP cases labeled by Gemini
**Method:** Gemini (`gemini-3.1-flash-lite-preview` via Vertex AI)
**Selection:** MAP prefilter — cases pool only (MAP phenotype=1 patients with discharge notes)
**Cache:** `data/gemini_hf_incident_cache.jsonl` — idempotent
**Conversion:** `labels_to_latte()` → per-visit (subject_id, T, Y) DataFrame
  - Controls and non-incident windows → Y=0; windows ≥ incident_T → Y=1

### Analysis Plan

**Script 01 — `01_cohort_definition.py`**
Query MIMIC-IV: HF patients (I50.x / 428.x), admissions, diagnoses, prescriptions,
procedures, ONCE-filtered lab events, discharge notes.
Compute per-patient `silver_date` (first HF ICD admittime) and save to data.

**Script 02 — `02_notes_nlp.py`**
MedSpaCy CUI extraction from discharge notes. Save `cui_obs.parquet`.

**Script 03 — `03_feature_matrix.py`**
Build obs_log (structured + NLP CUI). Apply 3-month period inclusion criterion:
retain only patients with ≥2 distinct periods before T_silver. Run `preprocess_map`.
Report CONSORT flow: raw → silver-date joinable → period filter → MAP sparsity filter.

**Script 04 — `04_map_phenotyping.py`**
Run MAP on mat_df. Save `map_results.parquet` with phenotype scores and binary labels.

**Script 05 — `05_gold_labels.py`**
MAP prefilter → sample ~120 MAP cases. Gemini incident labeling with `HF_DISEASE_CONFIG`.
Parse results → `labels_to_latte()` → `gold_labels.parquet`, `unlabeled_pool.parquet`.

**Script 06 — `06_latte_phenotyping.py`**
`format_latte_input` → embeddings → `run_latte`. Save `latte_predictions.parquet`
(per-patient incident probability and predicted incident T).

**Script 07 — `07_label_comparison.py`**
Join LATTE predictions to silver label dates. For each patient compute:
- `T_silver`: 3-month period of first HF ICD code
- `T_latte`: LATTE predicted incident period (argmax of incident_probability)
- Agreement category: concordant (T_latte == T_silver), earlier (T_latte < T_silver),
  later (T_latte > T_silver), or discordant (LATTE label=0 vs ICD label=1)
Produce: agreement bar chart, scatter plot T_latte vs T_silver, summary table.

**Script 08 — `08_suitability_evaluation.py`**
Structured evaluation across four dimensions:
1. **Dataset suitability**: HF prevalence in MIMIC, period coverage distribution,
   look-back window adequacy (how many periods are available before T_silver?)
2. **Label quality**: LATTE AUC on gold-labeled patients; Gemini parse error rate;
   agreement rate with principal-diagnosis rule
3. **Temporal signal**: Distribution of T_latte - T_silver (lead time / lag);
   proportion of concordant vs earlier/later calls
4. **Known limitations**: ICD codes assigned at discharge (not admission);
   single academic centre (BIDMC, Boston); MIMIC date-shifting obscures calendar time;
   BNP/NT-proBNP may be sparse in early study years; look-back restriction
   (≥2 periods) biases toward patients with longer care histories

### Potential Biases & Limitations
- ICD-first date is a discharge diagnosis, not a true onset — LATTE attempts to refine this
  but is also bounded by discharge-assigned codes
- Patients with only 1 observation period before first HF code are excluded — this preferentially
  drops patients with acute-onset HF or short medical histories (selection bias)
- Single centre: results may not generalise beyond academic tertiary-care hospitals
- MIMIC date-shifting means "earlier" LATTE calls may reflect look-back availability,
  not true earlier onset in calendar time
- Gemini gold label set (~120 patients) has high label variance for AUC estimation;
  cross-validation (5-fold) is preferred over a single held-out test split

### Skills to Use
- `mimic-preprocessing`: build_obs_log, rollup_itemid_to_loinc
- `map-phenotyping`: preprocess_map, run_map
- `latte-phenotyping`: map_prefilter, run_gemini_labeling (incident mode),
  labels_to_latte, format_latte_input, build_cooccurrence_embeddings, run_latte
- `mimic-note-preprocessing`: notes_to_events (MedSpaCy CUI extraction)
