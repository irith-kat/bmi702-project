---
name: build-datamart
description: Build the multi-modal patient feature matrix (mat_df) and note count denominator (note_df) required by MAP phenotyping. Use after preprocessing-strategy to construct the study population and feature matrix from ONCE-identified features across all EHR code modalities (PheCodes, RxNorm, LOINC, CCS, CUIs).
---

# Build Data Mart (Module 2, Part A)

## Goal
Construct a wide patient × feature count matrix (`mat_df`) and a per-patient note count denominator (`note_df`) for MAP. The strategy is **broad inclusion first**: capture every patient with any signal for the target phenotype, even if it includes many false positives. MAP separates true cases from controls.

## Prerequisites
- Preprocessed modality DataFrames from **preprocessing-strategy**
- ONCE output files parsed via `get_once_features` + `parse_once_by_modality`
- M4 API available: `from m4 import set_dataset, execute_query`

## Step 1 — Parse ONCE features by modality

```python
from once import get_once_features, parse_once_by_modality

once = get_once_features(
    codified_file="ONCE_<Disease>_PheCode<code>_cos0.165.csv",
    narrative_file="ONCE_<Disease>_<CUI>_titlecos0.5_titlecut0.3_exactFALSE.csv",
)
modalities = parse_once_by_modality(once)

# Choosing the anchor PheCode: use the feature with target_similarity=1.0
# in the ONCE codified output — this is the primary diagnostic code.
MAIN_PHECODE = once["codified"][once["codified"]["target_similarity"] == 1.0]["Variable"].iloc[0]
MAIN_PHECODE = MAIN_PHECODE.replace("PheCode:", "")  # e.g. '714.1' for RA
```

## Step 2 — Build the PheCode feature matrix (anchor modality)

The PheCode modality is the anchor and is always required.

```python
from preprocessing import build_map_feature_matrix

mat_df = build_map_feature_matrix(
    phecode_df=phecode_df,          # from preprocessing-strategy Step 2
    once_phecodes=modalities["phecode"],
    main_phecode=MAIN_PHECODE,
    min_nonzero=20,  # MAP flexmix EM requires ≥20 non-zero patients per feature
)
# mat_df: index=subject_id, columns=PheCodes, values=counts
# Study population is now defined: patients with ≥1 ONCE PheCode feature
```

## Step 3 — Build and join additional modality matrices

For each non-empty modality beyond PheCodes, pivot to wide format and join onto `mat_df`. Each modality follows the same pattern: **pivot → restrict to ONCE features → join**.

### RxNorm ingredients (if `modalities['rxnorm']` non-empty)

```python
# rx_df from preprocessing-strategy; must have subject_id + rxnorm_ingredient column
rx_wide = (
    rx_df[rx_df["rxnorm_ingredient"].isin(modalities["rxnorm"])]
    .pivot_table(index="subject_id", columns="rxnorm_ingredient", aggfunc="size", fill_value=0)
)
rx_wide.columns = [f"RXNORM:{c}" for c in rx_wide.columns]  # namespace columns
mat_df = mat_df.join(rx_wide, how="left").fillna(0)
```

### LOINC lab tests (if `modalities['loinc']` non-empty)

```python
# loinc_df from preprocessing-strategy; has subject_id + loinc_code column
loinc_wide = (
    loinc_df[loinc_df["loinc_code"].isin(modalities["loinc"])]
    .pivot_table(index="subject_id", columns="loinc_code", aggfunc="size", fill_value=0)
)
loinc_wide.columns = [f"LNC:{c}" for c in loinc_wide.columns]
mat_df = mat_df.join(loinc_wide, how="left").fillna(0)
```

### Lab short names (if `modalities['shortname']` non-empty)

```python
# shortname_df from preprocessing-strategy; has subject_id + lab_name column
sn_wide = (
    shortname_df[shortname_df["lab_name"].isin(modalities["shortname"])]
    .pivot_table(index="subject_id", columns="lab_name", aggfunc="size", fill_value=0)
)
sn_wide.columns = [f"ShortName:{c}" for c in sn_wide.columns]
mat_df = mat_df.join(sn_wide, how="left").fillna(0)
```

### CCS procedure categories (if `modalities['ccs']` non-empty)

```python
# ccs_df from preprocessing-strategy; has subject_id + ccs_category column
ccs_wide = (
    ccs_df[ccs_df["ccs_category"].isin(modalities["ccs"])]
    .pivot_table(index="subject_id", columns="ccs_category", aggfunc="size", fill_value=0)
)
ccs_wide.columns = [f"CCS:{c}" for c in ccs_wide.columns]
mat_df = mat_df.join(ccs_wide, how="left").fillna(0)
```

### Post-join sparse feature filter

After joining all modalities, re-apply the sparse feature filter globally. The anchor PheCode is always retained.

```python
MIN_NONZERO = 20
nonzero = (mat_df > 0).sum()
sparse = nonzero[(nonzero < MIN_NONZERO) & (nonzero.index != MAIN_PHECODE)].index
if len(sparse):
    mat_df = mat_df.drop(columns=sparse)
print(f"mat_df: {mat_df.shape}  ({mat_df.shape[1]} features after sparse filter)")
```

## Step 4 — Add NLP (CUI) features (optional but recommended)

NLP features capture phenotype signal in clinical notes that coded data misses. Use if MIMIC-IV-Note is available.

```python
from m4 import set_dataset, execute_query
from note_ner import extract_cui_features, aggregate_features

# Switch to notes dataset — must be separate from the hosp dataset
set_dataset("mimic-iv-note")
notes_df = execute_query("""
    SELECT subject_id, note_id, text
    FROM mimiciv_note.discharge
""")
# Restrict to study population to reduce NLP volume
notes_df = notes_df[notes_df["subject_id"].isin(mat_df.index)].reset_index(drop=True)

# Extract CUI mentions using ONCE narrative vocabulary (MedSpaCy NER)
# Filters out negated, uncertain, and family-history mentions automatically
cui_long = extract_cui_features(
    notes_df,
    text_column="text",
    id_column="subject_id",
    target_cuis=once["nlp_target_cuis"],  # [{term, cui}, ...] from ONCE narrative
)
cui_wide = aggregate_features(cui_long, id_column="subject_id", feature_column="cui")
mat_df = mat_df.join(cui_wide, how="left").fillna(0)
print(f"mat_df with NLP: {mat_df.shape}")

# For large corpora (>100k notes), save cui_long to parquet after the first run:
# cui_long.to_parquet("cui_features_<disease>.parquet", index=False)
# On subsequent runs: cui_long = pd.read_parquet("cui_features_<disease>.parquet")
```

## Step 5 — Build note count denominator

### Option A: Real note counts (preferred when notes available)

```python
note_df = (
    notes_df.groupby("subject_id")
    .size()
    .to_frame("note_count")
    .reindex(mat_df.index)
    .fillna(0)
    .clip(lower=1)
    .astype(int)
)
```

### Option B: Admission count proxy (when notes unavailable)

```python
from m4 import set_dataset, execute_query
from preprocessing import build_note_proxy

set_dataset("mimic-iv")
admissions_df = execute_query("SELECT subject_id, hadm_id FROM mimiciv_hosp.admissions")
note_df = build_note_proxy(admissions_df, mat_df.index)
# Validated in the RA/MIMIC-IV notebook — MAP results remain clinically sensible.
```

## Output contract

- **`mat_df`**: `pd.DataFrame`, index=`subject_id` (named), columns = namespaced feature codes
  - PheCode columns: bare PheCode strings (e.g. `"714.1"`)
  - RxNorm columns: `"RXNORM:<code>"`
  - LOINC columns: `"LNC:<code>"`
  - Lab short names: `"ShortName:<name>"`
  - CCS categories: `"CCS:<category>"`
  - CUI columns: UMLS CUI strings (e.g. `"C0003873"`)
  - Values: integer counts (times the feature was recorded per patient)
- **`note_df`**: `pd.DataFrame`, index=`subject_id`, single column `note_count`, int ≥ 1
- `mat_df.index` and `note_df.index` must be identical

These are the direct inputs to **map-phenotyping** (Module 2, Part B).
