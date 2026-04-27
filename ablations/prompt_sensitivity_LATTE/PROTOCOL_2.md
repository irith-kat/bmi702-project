# Research Protocol: Incident Heart Failure Timing in MIMIC-IV

## Research Question

For MIMIC-IV patients with at least one Heart Failure code and sufficient longitudinal data prior to that code (≥2 three-month periods), **when did Heart Failure first clinically occur?** How does the MAP+LATTE-refined incident date compare to the first-code silver label, and is this approach well-suited to this dataset and clinical question?

---

## Study Design

Cohort identification + incident timing + silver-label comparison + suitability evaluation

---

## Population

**Source:** MIMIC-IV (full dataset, ~300K patients)
**Anchor phenotype:** PheCode 428.1 — Congestive Heart Failure / Heart Failure (CUI C0018802)

### Inclusion
- At least one encounter with a Heart Failure ICD code (mapped to PheCode 428 or 428.1)
- At least **two** 3-month periods with any structured EHR data **prior to** the first HF-coded period (prevalent case filter)

### Exclusion
- Age < 18 at first MIMIC encounter
- Patients whose first HF code appears in their first or second data period (insufficient pre-HF observation window)

---

## Cohort Definition Approach

**Method:** MAP (Multimodal Automated Phenotyping) + LATTE (incident timing)

**Anchor:** PheCode 428.1 — Heart Failure

**NLP:** Yes — CUI features extracted from MIMIC-IV discharge notes via MedSpaCy, sourced from BigQuery

**ONCE files:**
- Codified: `input/ONCE_heart failure_PheCode428.1_cos0.165.csv`
- Narrative: `input/ONCE_PT_phenotype_heart failure_C0018802_titlecos0.5_titlecut0.3_exactFALSE.csv`

**Temporal aggregation:** 3-month periods. All visits and features within a calendar quarter are collapsed to a single period row per patient.

**MAP rationale:** HF ICD coding in MIMIC is discharge-assigned and likely has variable specificity (e.g., patients admitted for other reasons who coincidentally carry a HF code). MAP's Poisson mixture model uses co-feature patterns from the ONCE files to separate true cases from incidental coding, producing per-patient posterior probabilities rather than a binary ICD filter.

**LATTE rationale:** The clinical question is *incident timing* — we need to know the first period in which HF onset is detectable from the longitudinal record, which may differ from the first coded period. LATTE's semi-supervised GRU learns from ~100–200 Gemini-labeled patients and predicts per-period onset probability over the full sequence.

**MAP config:** `min_nonzero=10`, NLP contribution expected: moderate-to-high (HF has well-characterized note language including "ejection fraction", "BNP", "dyspnea", "edema")

---

## Silver Label

The **silver label** is the calendar period containing the patient's **first occurrence** of any HF PheCode (428 / 428.1) across all MIMIC encounters. This represents the simplest rule-based anchor for incident HF timing.

---

## Analysis Plan

### Step 1 — Cohort Definition (`scripts/cohort_definition.py`)
- Pull all patients with ≥1 HF ICD code from DuckDB
- Map ICD codes to PheCode 428 / 428.1
- Aggregate all structured EHR events to 3-month periods
- Apply age <18 exclusion at first encounter
- Apply ≥2 pre-HF period filter
- Record silver label (first HF period) per patient
- Save `data/hf_candidates.parquet` and `data/period_index.parquet`

### Step 2 — NLP Feature Extraction (`scripts/nlp_features.py`)
- For each candidate patient, pull discharge notes from BigQuery
- Run MedSpaCy NER to extract CUI mentions
- Aggregate CUI mentions to 3-month periods
- Append CUI rows to observation log
- Save `data/obs_log_with_nlp.parquet`

### Step 3 — MAP Phenotyping (`scripts/map_phenotyping.py`)
- Load observation log + ONCE files
- Run MAP Poisson mixture model
- Assign per-patient MAP scores and binary case/control labels
- Save `data/map_results.parquet`

### Step 4 — LATTE Incident Timing (`scripts/latte_phenotyping.py`)
- Load MAP results, format as 3-month period sequences
- Gemini labels ~100–200 patients (incident period annotation)
- Train semi-supervised GRU; predict per-period onset probability for all cases
- Assign refined incident period = first period with probability ≥ threshold
- Save `data/latte_results.parquet`

### Step 5 — Label Comparison (`scripts/label_comparison.py`)
- Join LATTE refined incident period to silver label period
- Categorize each patient:
  - **Concordant**: refined = silver
  - **Earlier**: refined < silver (LATTE detects onset before first code)
  - **Later**: refined > silver (first code preceded detectable onset signal)
- Compute period-gap distribution for Earlier/Later cases
- Visualize: stacked bar (concordance categories), histogram of period gaps
- Save `data/label_comparison.parquet`

### Step 6 — Characterization + Suitability Evaluation (`scripts/characterization.py`)
- **CONSORT flow**: patients at each pipeline stage
- **Demographics**: age distribution (histogram), sex and race (bar charts), stratified by concordance category
- **Top comorbidities**: top-15 PheCode clusters co-occurring in the cohort
- **Suitability evaluation** (narrative + quantitative):
  - Dataset suitability: MIMIC patient mix (ICU-heavy), discharge-assigned ICD coding, note coverage rate
  - Silver label quality: specificity of first HF code as an incident marker (how often is there a prior hospitalization with HF features but no code?)
  - Method suitability: MAP posterior distribution shape, LATTE convergence, label concordance rate
  - Clinical validity signal: do "Earlier" patients have preceding HF-related features (BNP, echo, diuretic prescriptions) in the pre-code periods?
- Save all plots as JSON to `plots/`

---

## Output Files

| File | Description |
|------|-------------|
| `data/hf_candidates.parquet` | All patients with ≥1 HF code + temporal filter applied |
| `data/period_index.parquet` | Per-patient, per-period feature matrix |
| `data/obs_log_with_nlp.parquet` | Observation log including NLP CUI rows |
| `data/map_results.parquet` | MAP posterior scores + binary labels |
| `data/latte_results.parquet` | Per-patient, per-period LATTE onset probability + refined incident period |
| `data/label_comparison.parquet` | Concordance classification + period gaps |
| `plots/consort_flow.json` | CONSORT-style exclusion counts |
| `plots/age_distribution.json` | Age histogram by concordance category |
| `plots/sex_race_distribution.json` | Demographic bar charts |
| `plots/label_comparison_categories.json` | Concordant/Earlier/Later stacked bar |
| `plots/period_gap_distribution.json` | Period-gap histogram for Earlier/Later patients |
| `plots/top_comorbidities.json` | Top-15 co-occurring PheCode clusters |
| `RESULTS.md` | Final findings summary |

---

## Potential Limitations

1. **ICD coding timing**: MIMIC ICD codes are discharge-assigned, not admission- or onset-timestamped. The "first coded period" may lag true clinical onset.
2. **MIMIC selection bias**: MIMIC over-represents ICU admissions. HF cohort may skew toward advanced/decompensated presentations vs. community-onset HF.
3. **3-month period granularity**: Incident timing is bounded to the quarter; within-quarter onset cannot be resolved.
4. **Prevalent case filter sensitivity**: The ≥2 pre-HF periods filter removes patients with limited MIMIC history, which may exclude patients who entered MIMIC already sick — but creates a systematic right-truncation of the early-in-MIMIC population.
5. **Gemini label quality**: LATTE depends on the quality of ~100–200 human (Gemini) annotations; HF onset timing is a clinically complex judgment call.
6. **Note coverage**: Not all patients have rich discharge notes; NLP CUI signal will be sparse for shorter stays.

---

## Skills Used

| Skill | Purpose |
|-------|---------|
| `mimic-preprocessing` | Build observation log from DuckDB structured EHR |
| `mimic-note-preprocessing` | CUI extraction from BigQuery discharge notes |
| `map-phenotyping` | Poisson mixture model case/control classification |
| `latte-phenotyping` | Semi-supervised GRU incident timing |
| `m4-api` | DuckDB + BigQuery data access |
