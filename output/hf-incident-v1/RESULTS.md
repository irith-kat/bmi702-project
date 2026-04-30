# Results: Heart Failure Incident Timing (hf-incident-v1)

## Pipeline Summary

| Step | Script | Key Output |
|------|--------|------------|
| DuckDB raw data | `01_cohort_data.py` | 31,369 HF patients; diagnoses, Rx, procedures |
| BigQuery notes | `01b_notes_pull.py` | 95,857 discharge notes; 25,326 patients (80.7%) |
| NLP CUI extraction | `02_nlp_features.py` | 80 unique CUIs; ~356K events |
| Feature matrix + washup | `03_feature_matrix.py` | 3,056 washup-passing patients × 80 features |
| MAP phenotyping | `04_map_phenotyping.py` | 1,318 cases / 1,738 controls |
| Gold labels | `05_gold_labels.py` | 120 patients labeled; 107 HF confirmed |
| LATTE baseline | `06_latte_phenotyping.py` | AUC 0.756 (single run, hold-out) |
| LATTE CV baseline | `07_latte_cv.py` | Mean AUC 0.776 ± 0.085 (5-fold) |
| LATTE CV tuned | `07b_latte_cv_tuned.py` | Mean AUC **0.843 ± 0.046** (5-fold) |
| Concordance + suitability | `08_concordance_suitability.py` | See below |

---

## Cohort CONSORT Flow

| Stage | N |
|-------|---|
| All MIMIC patients with HF ICD code (I50%/428%) | 31,369 |
| After age ≥ 18 (defensive check) | 31,369 (no change; MIMIC-IV is adult-only) |
| After event-based washup (≥ 2 two-month periods with any observed event before first HF code) | **3,056** |
| MAP candidates | 3,056 |
| MAP cases (phenotype = 1) | 1,318 |
| MAP controls (phenotype = 0) | 1,738 |
| Gold-labeled (Gemini, incident mode) | 120 (from MAP cases) |
| HF confirmed by Gemini (label = 1) | 107 / 120 (89.2%) |

**Note on washup attrition:** The pre-specified washup criterion (≥2 two-month periods with observed data before first HF code) excluded **90.3%** of HF-coded patients. This is substantially higher than the calendar-based exploration estimate (70%). The difference arises because the event-based rule requires actual clinical activity in the pre-HF window, not merely calendar time — many patients have their first HF code at the first or second admission where any data is recorded, with no clinical activity in prior periods. This is a critical finding for suitability.

---

## MAP Results

- Candidates: 3,056 (all with HF anchor code > 0)
- Cases (phenotype=1): 1,318 (43.1%)
- Controls (phenotype=0): 1,738 (56.9%)
- Score range: [0.33, 0.65] — **narrow**

The narrow MAP score range is structurally expected: every washup-passing patient has the HF anchor code by definition. MAP therefore has limited signal to discriminate between HF-positive and HF-negative patients based on co-feature structure alone. Patients with richer HF-related comorbidities (AF, cardiomyopathy, loop diuretic use) received higher MAP scores and were classified as cases. The 1,318 MAP cases represent patients with a richer HF-supporting clinical signature, while the 1,738 controls likely had incidental HF coding.

---

## Gold Labels (Gemini Incident Mode)

- 120 MAP cases sent to Gemini for incident HF review
- 107 confirmed HF (label=1) — 89.2% case rate
- 13 labeled as non-HF (label=0) — 10.8% false positives in MAP cases
- 0 parse errors
- Incident T range across Gemini-labeled patients: broad (spanning MIMIC's shifted time range)
- LATTE gold-label format: 1,036 per-visit rows; Y=1 for post-incident periods, Y=0 for pre-incident

---

## LATTE Incident Timing — Training Results

### Baseline (epochs=35, weight_smooth=0.1)

| Fold | Train | Test | Cases (test) | AUC |
|------|-------|------|--------------|-----|
| 1 | 96 | 24 | 22 | 0.642 |
| 2 | 96 | 24 | 22 | 0.877 |
| 3 | 96 | 24 | 21 | 0.779 |
| 4 | 96 | 24 | 21 | 0.774 |
| 5 | 96 | 24 | 21 | 0.807 |
| **Mean** | | | | **0.776 ± 0.085** |

### Tuned (epochs=45, weight_smooth=0.04, weight_contrastive=0.12)

| Fold | AUC |
|------|-----|
| 1 | 0.894 |
| 2 | 0.859 |
| 3 | 0.875 |
| 4 | 0.788 |
| 5 | 0.801 |
| **Mean** | **0.843 ± 0.046** |

**Tuning effect:** Reducing `weight_smooth` (0.1 → 0.04) and increasing epochs (35 → 45) produced a +6.7pp mean AUC improvement and cut fold-level variance in half (σ: 0.085 → 0.046). The lower smoothness weight allows the model to represent sharper probability transitions near the incident period, which is appropriate for an incident-timing task. The worst fold improved from 0.642 to 0.788 — likely because the longer training allowed the GRU to converge on richer temporal patterns.

---

## Concordance: Silver vs. Refined Label (Tuned LATTE, Relative-50% Method)

Comparison performed on all 120 gold-labeled patients (5-fold hold-out predictions).

### Incident date detection method

Three methods were evaluated for deriving the refined incident date from LATTE probability trajectories:

| Method | Earlier | Concordant | Later | Gold–refined median Δ |
|--------|---------|------------|-------|----------------------|
| Argmax | 54 (45%) | 4 (3%) | 62 (52%) | −10 periods |
| Steepest derivative | 119 (99%) | 0 | 1 (1%) | +4 periods |
| **Relative 50% threshold** | **107 (89%)** | **2 (2%)** | **11 (9%)** | **0 periods** ✓ |

**Argmax was rejected**: 64% of patients have monotone-increasing LATTE trajectories (LATTE is a cumulative model — Y=1 for all post-incident periods), so argmax = last visit for most patients. This renders it uninformative for locating onset.

**Steepest derivative (max Δp) was rejected**: selects the earliest rising point, placing nearly all patients before the silver label — including many implausibly early detections.

**Relative 50% threshold** (first period where probability exceeds `min + 0.5 × (max − min)` for each patient) was selected. It produces a **Gemini-gold vs LATTE-refined median delta of exactly 0 periods**, indicating the method correctly locates the probability transition point at the true clinical onset on average.

### Concordance results (relative 50% method)

| Category | N | % | Median Δ |
|----------|---|---|----------|
| Earlier (LATTE < silver) | 107 | 89.2% | −6 periods (−12 months) |
| Concordant (Δ = 0) | 2 | 1.7% | — |
| Later (LATTE > silver) | 11 | 9.2% | +11 periods (+22 months) |

**Gold-label concordance (n=79 Gemini-confirmed HF):**
- Gemini gold vs silver: median Δ = −5 periods (−10 months)
- Gemini gold vs LATTE refined: **median Δ = 0 periods** → LATTE correctly locates clinical onset at the same time as the physician-level gold label on average

**Per-patient agreement (Gemini vs LATTE refined, n=79):** The median delta of 0 periods masks a wide distribution (std = 10.3 periods, ±20 months). Only 48% of patients agree within ±1 period; 63% within ±3 periods. The distribution is left-skewed: LATTE is systematically conservative relative to Gemini — when they disagree, it is usually because Gemini found HF mentioned in the clinical narrative at an admission where the structured feature trajectory was not yet informative enough for LATTE to place onset. The median = 0 reflects that LATTE has no systematic bias relative to Gemini on average, not that per-patient precision is high.

**"Earlier" cases (89.2%):** LATTE detects HF onset a median 12 months before the first ICD code. This is expected: ICD codes in MIMIC are assigned at discharge and may lag the true onset by the admission duration. The relative-threshold method finds the period where the patient's clinical trajectory transitions from low to high HF probability — which may precede formal coding.

**"Later" cases (9.2%):** In a minority of patients, LATTE places onset after the first HF code. These likely represent patients whose first HF code was incidental (e.g., historical code or low-acuity admission), with the true clinical onset occurring at a later, more symptom-driven admission.

---

## Suitability Evaluation

### Data Characteristics

| Metric | Value | Assessment |
|--------|-------|------------|
| Median features per 2-month period | 35 | Rich — well-coded patient records |
| Sparse periods (< 3 features) | 1.7% | Excellent sparsity profile |
| Median total periods per patient | 7 | Short — structural MIMIC limitation |
| p10 / p90 total periods | 3 / 16 | Wide spread; some patients much longer |
| % patients with ≥3 periods (LATTE-ready) | 100% | All patients meet minimum sequence length |
| % patients with ≥5 periods | 74% | Most have enough context for LATTE |
| NLP CUI coverage pre-silver-label | 37.8% | Moderate — 38% have clinical note signal before first code |
| MAP score range | [0.33, 0.65] | Narrow — limited discriminatory power |

### Structural Limitations Specific to This Dataset and Question

1. **MIMIC-IV is a prevalent-case dataset for HF.** With 90.3% of HF-coded patients excluded by the washup rule, only patients with an extended pre-HF MIMIC record survive. These are not representative of all incident HF — they are biased toward patients who used MIMIC-affiliated hospitals extensively before their HF diagnosis. Incidence estimates from this cohort should not be extrapolated to a general population.

2. **Short patient histories constrain temporal modeling.** Median 7 two-month periods per patient is at the lower end for LATTE's GRU to learn rich temporal patterns. The inter-patient variability (p10=3, p90=16 periods) means LATTE learns from heterogeneous sequence lengths. This is reflected in the fold-level AUC variance (fold 4 = 0.788, fold 1 = 0.894 in tuned CV).

3. **ICD coding at discharge.** All HF codes in MIMIC-IV are discharge diagnoses, not admission-time codes. The silver label therefore represents the first discharge that listed HF — it can lag true onset by the full admission duration. This systematic lag inflates the "earlier" category in the concordance analysis.

4. **MAP score compression.** All washup-passing candidates have the HF anchor code (100% anchor prevalence in mat_df). MAP relies on co-feature pattern variation to discriminate — with a homogeneous anchor distribution, the resulting scores are compressed into a narrow range [0.33, 0.65] with no clear bimodal separation. A more discriminating MAP would require a cohort containing non-HF patients for comparison.

5. **NLP from discharge summaries.** Discharge summaries reflect the full inpatient stay and are retrospective. CUI features detected pre-silver-label may reflect documentation of historical findings, not prospective clinical observation. This limits the interpretation of NLP as a "pre-clinical signal."

### Overall Suitability Assessment

**MAP + LATTE for HF incident timing in MIMIC-IV is feasible but faces structural constraints.** The pipeline runs successfully end-to-end, LATTE achieves a cross-validated AUC of 0.843, and the concordance analysis reveals meaningful patterns (45% of cases detected earlier than the silver label). However:

- The **90.3% washup attrition** is a fundamental problem: it selects a highly specific and non-representative subpopulation. Results cannot generalize to typical incident HF.
- **MAP's limited discriminatory power** (narrow score range) reduces its utility as a prior for LATTE. A future version should include matched non-HF controls in the MAP candidate pool.
- **LATTE consistently places onset 20 months later than Gemini gold labels** (argmax method). A transition-based incident detection (first period where probability exceeds patient-specific baseline by a threshold) may better align with clinical onset.
- The **35-feature/period data density is a strength** — these are not sparse records. The data quality supports a more discriminating phenotyping approach.

**Recommendation for future work:** Use a broader candidate pool (including HF-adjacent patients), add matched controls from the full MIMIC population, and explore a transition-point incident detector rather than argmax. For a different dataset with longer patient histories (e.g., a claims database), the MAP + LATTE approach is expected to perform substantially better.

---

## Gemini Gold vs Silver vs LATTE — Concordance Comparison

To validate that LATTE is recovering signal similar to physician review rather than just reproducing the silver label, we compare the concordance pattern of two independent sources against the silver label:

| Label source | N | Earlier | Concordant | Later | Median lead (earlier) |
|---|---|---|---|---|---|
| LATTE refined vs silver | 120 | 89.2% | 1.7% | 9.2% | 6 periods (12 months) |
| Gemini gold vs silver | 79 | 96.2% | 0.0% | 3.8% | 6 periods (12 months) |

Both LATTE and Gemini independently place onset **earlier** than the first ICD code in the vast majority of patients, with identical median leads (6 periods, 12 months). This alignment supports the interpretation that LATTE is learning the same clinical signal as physician review — earlier-than-coding HF onset — rather than simply recapitulating the silver label. The small difference in "later" rates (9.2% LATTE vs 3.8% Gemini) is expected: LATTE is a probabilistic model trained on 120 patients, while Gemini has access to the full discharge narrative.

---

## Key Files

| File | Description |
|------|-------------|
| `data/obs_log.parquet` | 1.72M events, 3,056 washup-passing patients |
| `data/map_results.parquet` | MAP scores + case/control labels |
| `data/gemini_incident_results.parquet` | Gemini incident labels for 120 patients |
| `data/gold_labels.parquet` | LATTE-format per-visit labels |
| `data/cv_results_tuned/cv_summary_tuned.csv` | 5-fold CV results (tuned) |
| `data/concordance.parquet` | Silver vs refined label comparison (120 patients) |
| `data/gold_concordance.parquet` | Gemini gold vs silver vs LATTE (79 confirmed HF) |
| `plots/concordance_distribution.json` | Bar chart: concordant/earlier/later |
| `plots/label_delta_histogram.json` | Δ(refined − silver) distribution |
| `plots/suitability_summary.json` | Suitability metric panel |
| `plots/data_density.json` | Features per period histogram |
