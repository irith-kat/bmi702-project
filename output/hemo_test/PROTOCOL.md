# Research Protocol: hemo_test — Hemorrhoid Cohort Identification and Characterization

## Research Question
Which patients in MIMIC-IV-demo have hemorrhoids, as identified by multimodal automated phenotyping (MAP), and what are their key clinical characteristics?

## Study Design
**Cohort characterization** — identify cases via MAP phenotyping, then describe the cohort.

## Population
**Inclusion:** All subjects in MIMIC-IV-demo (n ≈ 100 patients).
**Exclusion:** None — cast the widest net given the small demo cohort.

## ONCE Features
| File | Type | Features |
|------|------|----------|
| `ONCE_hemorrhoids_PheCode455_cos0.165.csv` | Codified (PheCodes + CCS) | PheCode:455, PheCode:578.8, CCS:81, PheCode:578, PheCode:578.2, PheCode:565 |
| `ONCE_PT_phenotype_hemorrhoids_C0019112_titlecos0.5_titlecut0.3_exactFALSE.csv` | Narrative (CUIs) | 50 CUIs incl. Hemorrhoids (C0019112), Hemorrhoidectomy (C0019108), Rectal hemorrhage (C0267596), etc. |

**Anchor PheCode:** `455` (Hemorrhoids — direct ICD mapping)

## Data Sources
| Data type | Source | Backend |
|-----------|--------|---------|
| ICD diagnoses, CPT procedures, demographics | `mimic-iv-demo` | Local DuckDB |
| Discharge summaries (clinical notes) | `mimic-iv-note` | BigQuery |

## Analysis Plan

### Step 1: Cohort definition (`01_cohort_definition.py`)
- Pull all subjects + demographics (age, gender, race) from `mimiciv_hosp.patients` and `mimiciv_hosp.admissions` (local demo).
- Pull all ICD-9/10 diagnoses from `mimiciv_hosp.diagnoses_icd`.
- Pull CPT/procedure codes from `mimiciv_hosp.procedures_icd`.
- Save: `data/subjects.parquet`, `data/diagnoses.parquet`, `data/procedures.parquet`.

### Step 2: Feature matrix construction (`02_feature_matrix.py`)
- Roll up ICD codes → PheCodes using `rollup_icd_to_phecode`.
- Roll up procedure codes → CCS using `rollup_cpt_to_ccs`.
- Parse ONCE files with `get_once_features` → `codified_list` + `nlp_target_cuis`.
- Fetch discharge notes from BigQuery (`mimic-iv-note`, `mimiciv_note.discharge`), filtered to demo subjects.
- Run MedSpaCy NER on notes → CUI feature matrix.
- Assemble `mat_df` (patients × features) and `note_df` (note counts).
- Save: `data/mat_df.parquet`, `data/note_df.parquet`.

### Step 3: MAP phenotyping (`03_map_phenotyping.py`)
- Call `run_map(mat_df, note_df, main_icd_col="455")`.
- Apply threshold p ≥ 0.5 → binary `is_case` label.
- Save: `data/map_scores.parquet`.

### Step 4: Cohort characterization (`04_characterization.py`)
- Compare cases vs. non-cases on:
  - Age distribution (histogram)
  - Gender (bar chart)
  - Top comorbid PheCodes (bar chart)
- Save: plots as JSON, summary stats to `data/characterization.parquet`.

## MAP Configuration
- **Anchor PheCode:** `455`
- **Case threshold:** p ≥ 0.5 (default)
- **Sparse feature cutoff:** drop features with < 5 non-zero patients (relaxed from default 20 given small demo cohort)

## Potential Biases & Limitations
- MIMIC-IV-demo has ~100 patients — MAP's EM algorithm may underfit with very few hemorrhoid cases; results are illustrative, not clinically validated.
- ICD codes are assigned at discharge — case identification reflects documented diagnoses, not prospective screening.
- NLP features from BigQuery notes may be incomplete if the demo patient subset has sparse discharge summaries.
- No control matching — characterization is descriptive only.

## M4 Skills Used
- `preprocessing-strategy`: ICD→PheCode + CPT→CCS rollup
- `build-datamart`: Assemble mat_df + note_df
- `map-phenotyping`: Run MAP algorithm
- `m4-api`: Query local demo (tabular) and BigQuery (notes)
