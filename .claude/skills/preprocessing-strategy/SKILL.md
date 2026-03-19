---
name: preprocessing-strategy
description: Harmonize raw EHR data by rolling up codes to standardized vocabularies (ICD→PheCode, RxNorm→ingredient, LOINC lab tests, CPT/HCPCS→CCS). Use at the start of any phenotyping workflow to prepare multi-modal feature data from MIMIC-IV or other EHR datasets.
---

# Preprocessing Strategy (Module 1)

## Goal
Transform raw EHR data into harmonized, standardized features across all available code modalities. The output feeds directly into **build-datamart** (Module 2). Always query via the M4 API.

## Step 0 — Set up M4 and inspect available tables

```python
from m4 import set_dataset, execute_query, get_schema

set_dataset("mimic-iv")
schema = get_schema()
print(schema['tables'])  # verify tables before querying
```

## Step 1 — Parse ONCE features to know which modalities are needed

ONCE identifies features from multiple vocabularies. Parse the codified list to know which tables to query.

```python
from once import get_once_features, parse_once_by_modality

once = get_once_features(
    codified_file="ONCE_<Disease>_PheCode<code>_cos0.165.csv",
    narrative_file="ONCE_<Disease>_<CUI>_titlecos0.5_titlecut0.3_exactFALSE.csv",
)
modalities = parse_once_by_modality(once)
# modalities keys: 'phecode', 'rxnorm', 'loinc', 'ccs', 'shortname', 'other'
# Only query tables for modalities with non-empty lists
print({k: len(v) for k, v in modalities.items()})
```

## Step 2 — Load and roll up each modality

### ICD Diagnoses → PheCode (query if `modalities['phecode']` non-empty)

```python
from rollup import rollup_icd_to_phecode

# Query raw diagnoses from M4
diagnoses_df = execute_query("""
    SELECT subject_id, hadm_id, icd_code, icd_version
    FROM mimiciv_hosp.diagnoses_icd
""")

# Roll up ICD-9/10 → PheCode
# Note: MIMIC-IV stores codes without dots (e.g. '71410', not '714.10').
# rollup_icd_to_phecode inserts the dot automatically.
phecode_df = rollup_icd_to_phecode(
    diagnoses_df,
    icd_column="icd_code",
    mapping_file="Phecode_map_v1_2_icd9_icd10cm.csv",  # project root
)
# Result adds 'Phecode' and 'PhecodeString' columns; unmatched rows keep Phecode=NaN
```

### Prescriptions → RxNorm Ingredient (query if `modalities['rxnorm']` non-empty)

```python
from rollup import rollup_rxnorm_to_ingredient

# MIMIC-IV prescriptions carry drug name, NDC, and GSN codes.
# RxNorm ingredient mapping is a stub — implement via RxNav API or local table.
prescriptions_df = execute_query("""
    SELECT subject_id, hadm_id, drug, ndc, gsn
    FROM mimiciv_hosp.prescriptions
    WHERE drug IS NOT NULL
""")

# Current stub returns df unchanged; implement ingredient mapping as needed.
rx_df = rollup_rxnorm_to_ingredient(prescriptions_df, rxnorm_column="ndc")
# After implementation: expect an 'rxnorm_ingredient' column to be added.
```

### Lab Tests → LOINC (query if `modalities['loinc']` or `modalities['shortname']` non-empty)

```python
# MIMIC-IV lab items link itemid → LOINC via d_labitems.
# For ShortName features, use the label column instead of loinc_code.
labs_df = execute_query("""
    SELECT le.subject_id, le.hadm_id, dl.loinc_code, dl.label AS lab_name
    FROM mimiciv_hosp.labevents le
    LEFT JOIN mimiciv_hosp.d_labitems dl ON le.itemid = dl.itemid
    WHERE dl.loinc_code IS NOT NULL
       OR dl.label IS NOT NULL
""")
# Filter to ONCE-identified LOINC codes or short names before pivoting
loinc_df = labs_df[labs_df["loinc_code"].isin(modalities["loinc"])]
shortname_df = labs_df[labs_df["lab_name"].isin(modalities["shortname"])]
```

### Procedures → CCS (query if `modalities['ccs']` non-empty)

```python
from rollup import rollup_cpt_to_ccs

# MIMIC-IV: HCPCS Level II codes are in hcpcsevents; ICD-10-PCS are in procedures_icd.
# Use hcpcsevents for AHRQ CCS mapping (closest to CPT in MIMIC).
hcpcs_df = execute_query("""
    SELECT subject_id, hadm_id, hcpcs_cd
    FROM mimiciv_hosp.hcpcsevents
""")
ccs_df = rollup_cpt_to_ccs(
    hcpcs_df,
    cpt_column="hcpcs_cd",
    mapping_file="CCS_Services_Procedures_v2025-1_052425.csv",  # project root
)
# Adds 'ccs_category' and 'ccs_description' columns
```

## Step 3 — Understand unfamiliar codes (optional)

```python
from vocab import get_code_definition

# Clarify what a code means before using it as a feature
get_code_definition("714.1", "ICD10CM")   # → "Rheumatoid arthritis"
get_code_definition("4548-4", "LNC")      # → "Hemoglobin A1c"
get_code_definition("614391", "RXNORM")   # → "abatacept"
```

Requires network access to UMLS API. Use this to understand feature semantics before committing to a feature set.

## Output contract

After preprocessing you have one DataFrame per modality, each containing at minimum:
- `subject_id`: patient identifier
- `hadm_id`: admission identifier (for de-duplication)
- A standardized code column: `Phecode`, `loinc_code`, `lab_name`, `ccs_category`, or the rolled-up RxNorm identifier

All DataFrames feed into **build-datamart** where they are pivoted into the unified patient × feature matrix.

## Handling missing modalities

If a MIMIC-IV table is not available or a rollup function is a stub:
- Skip that modality's table query
- The feature matrix will simply omit that column set
- MAP can still run on whatever modalities are available — more features improve phenotyping quality but are not all required
