# Research Protocol: Heart Failure Incident Timing (MIMIC-IV)

## Research Question
Identify the incident date of Heart Failure (HF) in MIMIC-IV patients who carry an HF diagnosis code, using a semi-supervised approach that refines the silver label (first HF PheCode) with longitudinal EHR signal. Evaluate concordance of the refined incident date with the silver label, and assess the suitability of this approach for MIMIC-IV and this clinical question.

---

## Study Design
Cohort identification with incident timing + label concordance analysis + suitability evaluation.

**Study ID:** `hf-incident-v1`

---

## Population

**Source:** MIMIC-IV (full dataset)

**Anchor phenotype:** PheCode 428.1 — Heart Failure

**Candidate pool:** All patients in MIMIC with at least one HF ICD code (ICD-9: 428%, ICD-10: I50%).

### Exploration findings (script: `scripts/00_explore.py`)
| Metric | Value |
|--------|-------|
| All patients with HF ICD code | 31,369 |
| Age < 18 | 0 (MIMIC-IV is adult-only) |
| Surviving 2-period washup | 9,430 (30.1%) |
| Prevalent at entry (period 0) | 21,283 (67.8%) |
| With non-HF diagnosis codes (of washup-passers) | 9,430 (100%) |
| With ≥1 discharge note (sample, n=400) | 91% |
| Median MIMIC history | 279 days (~4.7 two-month periods) |
| ONCE codified features | 64 |
| ONCE NLP CUI targets | 115 |

### Inclusion Criteria
- At least one HF ICD code recorded in MIMIC (ICD-9: 428%, ICD-10: I50%)

### Exclusion Criteria
1. **Age < 18** at time of first MIMIC admission *(defensive check; exploration confirms 0 excluded)*
2. **Insufficient pre-index data:** fewer than 2 two-month periods with any data recorded *before* the first HF code — removes patients who likely developed HF before entering MIMIC (prevalent cases). *Exploration: this removes 70% of HF-coded patients — expected for an ICU dataset where HF is overwhelmingly a pre-existing comorbidity.*

*Note: the "data beyond anchor only" criterion was dropped after exploration confirmed that 100% of washup-passing patients have non-HF diagnosis codes.*

---

## Temporal Aggregation
All visits and codes are aggregated into **2-month periods**. Each patient's longitudinal record is represented as a sequence of 2-month windows, each containing the set of PheCode, CCS, RxNorm, and NLP CUI features observed in that window.

---

## Cohort Definition Approach

**Method:** MAP (Multimodal Automated Phenotyping) → LATTE (incident timing)

**Silver label:** First 2-month period containing an HF PheCode code (PheCode 428.1)

**ONCE files:**
- Codified: `ONCE_heart failure_PheCode428.1_cos0.165.csv`
- Narrative (NLP CUI): `ONCE_PT_phenotype_heart failure_C0018802_titlecos0.5_titlecut0.3_exactFALSE.csv`

**Data sources:**
- Codified EHR features (PheCode, CCS, RxNorm, LOINC): local DuckDB (MIMIC-IV)
- NLP CUI features from clinical notes: BigQuery

**MAP rationale:** The HF PheCode has moderate specificity in MIMIC — many patients receive the code during acute admissions without a primary HF diagnosis. MAP uses the co-feature structure from ONCE to assign posterior probabilities, separating true incident HF from incidental coding. This is preferable to a rule-based filter for a condition that is frequently coded for comorbidity rather than as the primary encounter reason.

**LATTE rationale:** The silver label (first HF code) is known to overestimate incidence — it does not distinguish between first *documentation* and first *occurrence*. LATTE learns from 120 Gemini-gold-labeled patients to estimate when HF actually began in each patient's sequence.

**MAP config:**
- `min_nonzero` threshold: default (to be tuned on final candidate count)
- NLP contribution: expected high (HF-related CUI mentions — dyspnea, edema, reduced EF — often appear before formal coding)

**Gold labels:** 120 patients labeled by Gemini for LATTE training (~1.3% of the 9,430 washup-passing candidates — workable for LATTE)

---

## Analysis Plan

### Step 1: Cohort Definition
- Pull all patients with HF PheCode from DuckDB
- Pull NLP CUI features from BigQuery
- Build observation log (2-month periods, codified + NLP)
- Apply exclusion criteria (age, washup window, data sufficiency)
- Report CONSORT flow

### Step 2: MAP Phenotyping
- Load ONCE files + observation log
- Run MAP to assign per-patient posterior probability of true HF
- Binary case/control split using MAP posterior threshold
- Save `map_results.parquet`

### Step 3: LATTE Incident Timing
- From MAP cases: run LATTE with 120 Gemini-gold-labeled patients
- Output: per-patient, per-period incident probability
- Assign refined incident date = first period where LATTE probability exceeds threshold
- Save `latte_results.parquet`

### Step 4: Concordance Analysis (Silver vs. Refined Label)
For each MAP case, compare silver label (first HF PheCode period) to LATTE refined incident date:
- **Concordant:** refined = silver (same 2-month period)
- **Earlier:** refined < silver (LATTE detects HF before first code)
- **Later:** refined > silver (LATTE places onset after first code — first code may have been a false positive or remote history code)

Report:
- Distribution of concordance categories (counts + %)
- Median lead/lag in periods for earlier/later groups
- Histogram of delta (refined − silver) in periods

### Step 5: Suitability Evaluation
Assess whether MAP + LATTE is appropriate for MIMIC-IV / HF incident detection:
- **Data density:** median features per period, % sparse periods
- **NLP signal:** fraction of patients with any CUI mention pre-silver-label
- **Washup attrition:** how many candidates are removed by the 2-period rule (proxy for prevalent-case contamination)
- **LATTE label coverage:** fraction of MAP cases with sufficient visit sequences for LATTE (>= 3 periods total)
- **Silver label reliability check:** distribution of time from MIMIC entry to first HF code (very early codes = potentially prevalent)
- Narrative summary of suitability

---

## Outputs

| File | Description |
|------|-------------|
| `data/obs_log.parquet` | Observation log (codified + NLP, 2-month periods) |
| `data/map_results.parquet` | MAP posterior probabilities + case/control labels |
| `data/latte_results.parquet` | Per-patient, per-period LATTE incident probabilities |
| `data/cohort.parquet` | Final cohort with silver label, refined label, concordance category |
| `plots/consort_flow.json` | CONSORT attrition diagram |
| `plots/concordance_distribution.json` | Bar chart: concordant / earlier / later |
| `plots/label_delta_histogram.json` | Histogram: refined − silver (in 2-month periods) |
| `plots/data_density.json` | Feature density per period |
| `plots/suitability_summary.json` | Suitability metric panel |

---

## Potential Limitations
- **MIMIC-IV ICD coding at discharge:** All codes reflect the full stay, not admission time. The silver label (first HF code) represents the first discharge that included HF — not necessarily the first clinical manifestation.
- **Severe prevalent-case attrition (exploration-confirmed):** 70% of HF-coded patients are excluded by the washup rule (21,283 of 31,369 have HF as their first recorded code). The surviving 9,430 are a specific subgroup — patients who were in MIMIC long enough before their first HF code to build a clean pre-index record. This selection effect should be discussed in the suitability evaluation.
- **Short MIMIC histories:** Median history is only 279 days (~4.7 two-month periods). After the washup window consumes ≥2 periods, many patients have very short post-washup sequences for LATTE to learn from. Sequence-length distribution should be reported as part of suitability.
- **LATTE generalization:** LATTE learns from 120 Gemini-labeled patients. With complex HF heterogeneity (HFrEF vs HFpEF, acute decompensation vs chronic), 120 labels may under-represent subtypes.
- **NLP feature lag:** Clinical notes in MIMIC are discharge summaries, not real-time notes — NLP CUI features may reflect post-hoc documentation rather than symptom onset.
- **BigQuery / DuckDB split:** NLP features (BigQuery) and codified features (DuckDB) are joined on `subject_id` + period. Any patient missing from one source will be noted but not excluded (91% expected note coverage).

---

## Skills Pipeline

| Step | Skill | Reason |
|------|-------|--------|
| SQL data access | `m4-api` | Multi-step DuckDB + BigQuery queries |
| Table joins | `mimic-table-relationships` | Correct patient-admission linkage |
| Observation log | `mimic-preprocessing` | Standardized rollup (ICD→PheCode, NDC→RxNorm, etc.) |
| NLP features | `mimic-note-preprocessing` | CUI mentions from BigQuery notes |
| MAP | `map-phenotyping` | Posterior probabilities over ONCE co-features |
| LATTE | `latte-phenotyping` | Incident timing with Gemini gold labels |
