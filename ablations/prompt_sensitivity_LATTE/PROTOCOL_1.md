# Research Protocol: Heart Failure Incident Timing Cohort (MIMIC-IV)

## Research Question
Among all MIMIC-IV patients with at least one Heart Failure ICD code, identify the incident date of HF for each patient using a semi-supervised approach that distinguishes true HF cases from miscoded patients, and characterize the agreement between the refined incident label and the silver standard PheCode label.

---

## Study Design
Cohort identification with incident timing + label comparison + suitability evaluation

**Study ID:** `hf-incident-latte-v1`

---

## Population
**Source:** MIMIC-IV (full)
**Anchor phenotype:** PheCode 428.1 — Congestive Heart Failure
**Inclusion:** All patients with ≥1 ICD-9/ICD-10 Heart Failure code at any encounter
**Exclusion:** Patients aged < 18 at first MIMIC encounter

---

## Temporal Structure
All visits, diagnoses, prescriptions, and NLP features are aggregated into **3-month (quarter) periods** per patient. The timeline runs from a patient's earliest MIMIC encounter to their last. Each period is a row in the feature matrix passed to LATTE.

---

## Cohort Definition Approach
**Method:** MAP (Multimodal Automated Phenotyping) → LATTE (incident timing)

**Anchor:** PheCode 428.1 (silver label — presence of HF code in any quarter)

**NLP:** Yes — CUI features extracted from MIMIC-IV discharge notes via BigQuery + MedSpaCy

**ONCE files (in `input/`):**
- Codified: `ONCE_heart failure_PheCode428.1_cos0.165.csv`
- Narrative: `ONCE_PT_phenotype_heart failure_C0018802_titlecos0.5_titlecut0.3_exactFALSE.csv`

**MAP rationale:** HF is frequently over-coded (e.g., as a secondary finding, rule-out diagnosis, or billing artifact). MAP's mixture model assigns per-patient posterior probabilities that reflect true HF burden across the ONCE co-feature distribution, allowing separation of true cases from miscoded patients.

**LATTE configuration:**
- Gold labels: ~120 patients labeled via Gemini (physician review of discharge notes)
  - True HF: label the **first** 3-month period containing an HF code as Y=1 (incident); all later periods follow naturally
  - Miscoded / not true HF: **all** periods labeled Y=0
- Silver labels: PheCode 428.1 presence in a 3-month period (binary)
- Temporal aggregation: 3-month periods
- Features: codified ONCE features (from DuckDB) + NLP CUI features (from BigQuery)

---

## Outputs

### Primary
1. **Labeled cohort** (`data/cohort.parquet`): one row per patient, with:
   - `subject_id`, `incident_quarter` (LATTE-refined first period), `silver_incident_quarter` (first period with HF PheCode), `map_score`, `latte_label`
   - Period-level dataset (`data/cohort_periods.parquet`): one row per patient × 3-month period, with `Y_latte`, `Y_silver`

### Secondary
2. **Label comparison** (`data/label_comparison.parquet`): for each patient, classify refined vs silver as:
   - **Concordant**: incident quarter matches
   - **Earlier**: LATTE incident quarter is before the first silver-label quarter
   - **Later**: LATTE incident quarter is after the first silver-label quarter
   - **Discordant (no HF)**: LATTE assigns all Y=0 but silver has ≥1 positive period
3. **Suitability evaluation** (`data/suitability_eval.md`): narrative + quantitative assessment covering:
   - HF coding prevalence and heterogeneity in MIMIC-IV
   - Gold label feasibility (how many candidates, labeling burden)
   - Feature coverage (NLP CUI density, codified feature sparsity per period)
   - MAP fit quality (posterior distribution shape, case/control separation)
   - LATTE convergence and calibration against gold labels
   - Limitations and confidence in the incident dates

---

## Scripts (Execution Order)

| Script | Step | Source |
|--------|------|--------|
| `01_cohort_candidates.py` | Pull HF-coded patients, apply age exclusion, build obs_log at period level | DuckDB (codified) |
| `02_nlp_features.py` | Fetch discharge notes from BigQuery, run MedSpaCy NER, append CUI rows to obs_log | BigQuery + MedSpaCy |
| `03_map_phenotyping.py` | Run MAP over ONCE files + obs_log → per-patient HF posteriors | ONCE + obs_log |
| `04_latte_phenotyping.py` | Run LATTE with gold + silver labels → incident timing per period | MAP results + labels |
| `05_label_comparison.py` | Classify concordant / earlier / later / discordant, plot distribution | LATTE + silver labels |
| `06_suitability_evaluation.py` | Quantitative suitability metrics + narrative evaluation | All outputs |

---

## Characterization Plan
1. CONSORT flow: total MIMIC-IV patients → HF-coded → age ≥18 → final case/control/no-HF split
2. Label comparison distribution: concordant / earlier / later / discordant (bar chart)
3. Incident quarter distribution: how many quarters before/after first HF code does LATTE identify?
4. MAP posterior distribution: histogram of HF scores across the full candidate pool
5. Feature coverage per period: median non-zero ONCE features (codified vs NLP) across quarters

---

## Potential Limitations
- ICD codes in MIMIC-IV are assigned at **discharge** — the first coded quarter approximates but does not equal the true clinical onset date
- Gold labeling via Gemini is limited to ~120 patients; model generalization depends on representativeness of labeled set
- NLP CUI coverage varies by note type and availability; patients with few/no notes will rely entirely on codified features
- MIMIC-IV is a single-center ICU-enriched dataset — HF prevalence and coding patterns may not generalize
- 3-month period aggregation may mask short-duration episodes or conflate distinct exacerbations
