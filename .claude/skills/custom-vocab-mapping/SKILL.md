---
name: custom-vocab-mapping
description: Given a dataset with custom/proprietary vocabulary codes and a crosswalk file
  (custom → ICD, NDC, CPT, or directly to PheCode/RxNorm/CCS), produce a drop-in mapping
  CSV that the preprocessing pipeline's rollup functions can consume via their mapping_file=
  parameters. Use when adapting a non-MIMIC EHR dataset for the observation log / MAP / LATTE
  pipeline.
---

# custom-vocab-mapping

## When to use

- Your EHR system uses hospital-specific formulary codes, local diagnosis codes, or proprietary procedure identifiers
- You have (or can obtain) a crosswalk from your custom codes to a standard vocabulary (ICD-9/10, NDC, CPT/HCPCS, RxNorm, PheCode, CCS)
- You want to feed non-MIMIC data into the observation log / MAP / LATTE pipeline
- You need to understand what fraction of your custom codes successfully map to the standard rollup vocabularies

## Prerequisites

```python
pip install ndclib   # for NDC normalization (drug modality only)
import pandas as pd
```

The observation log pipeline lives in `src/preprocessing/structured/`. The mapping dicts
it consumes are in `mapping_dicts/`.

---

## Vocabulary bridge paths

The pipeline always ends at one of three rollup vocabularies. Your job is to bridge from
your custom codes to the intermediate or final vocab:

```
Custom code
    │
    ├─► ICD-9/10 ──────────► PheCode     (diagnosis modality)
    │
    ├─► NDC (11-digit) ─────► RxNorm     (drug modality, primary path)
    ├─► Drug name ──────────► RxNorm     (drug modality, fallback path)
    │
    ├─► CPT/HCPCS ──────────► CCS        (procedure modality)
    │
    └─► itemid ─────────────► LOINC      (lab modality)
```

If your crosswalk goes directly to the final vocab (e.g., custom → PheCode), skip the
intermediate join and proceed to Step 4 directly.

---

## Step 1: Identify modality and gather files

Ask the user:

1. **Modality** — diagnosis, drug, or procedure?
2. **Custom vocabulary file** — one row per code; must have at least a code column and ideally a label column
3. **Crosswalk file** — maps custom code → standard vocabulary code (ICD, NDC, CPT, RxNorm, etc.)

Confirm both files exist and are readable before proceeding.

---

## Step 2: Inspect schemas

```python
import pandas as pd

vocab_df    = pd.read_csv("path/to/custom_vocab.csv", nrows=5)
crosswalk_df = pd.read_csv("path/to/crosswalk.csv", nrows=5)

print(vocab_df.dtypes)
print(crosswalk_df.dtypes)
```

Identify:
- **Custom code column** in both files (the join key)
- **Bridge code column** in the crosswalk (ICD, NDC, CPT, or direct rollup target)
- Any label/description columns to carry through

---

## Step 3: Choose bridge path

| If crosswalk target is... | Bridge path | Skip to |
|---|---|---|
| ICD-9 or ICD-10 code | custom → ICD → PheCode join | Step 4a |
| NDC (any format) | custom → NDC → normalize → RxNorm join | Step 4b |
| Free-text drug name | custom → drug name → RxNorm join | Step 4c |
| CPT or HCPCS code | custom → CPT → CCS (already handled by rollup) | Step 4d |
| Lab itemid (MIMIC-IV) | custom itemid → LOINC via `rollup_itemid_to_loinc` | Step 4e |
| PheCode directly | emit mapping with `ICD, Phecode, PhecodeString` columns | Step 4a shortcut |
| RxNorm ingredient ID directly | emit mapping with `ndc, ingredient_id, ingredient_name` columns | Step 4b shortcut |

---

## Step 4: Generate the mapping CSV

### 4a — Diagnosis: custom → ICD → PheCode

```python
import pandas as pd

# Load your files
crosswalk_df = pd.read_csv("crosswalk.csv")   # columns: custom_code, icd_code
phecode_map  = pd.read_csv(
    "mapping_dicts/Phecode_map_v1_2_icd9_icd10cm.csv",
    usecols=["ICD", "Phecode", "PhecodeString", "PhecodeCategory"],
    dtype=str,
)

# Normalize ICD format in crosswalk if needed
# If codes already have dots → skip; if not → insert dot at position 3
def add_dot(code):
    if pd.isna(code) or "." in str(code) or len(str(code)) <= 3:
        return code
    return str(code)[:3] + "." + str(code)[3:]

crosswalk_df["icd_code"] = crosswalk_df["icd_code"].apply(add_dot)

# Two-hop join: custom → ICD → PheCode
merged = crosswalk_df.merge(phecode_map, left_on="icd_code", right_on="ICD", how="left")

# Output schema must match Phecode_map_v1_2_icd9_icd10cm.csv exactly
# Use custom_code as the ICD column so rollup_icd_to_phecode() can join on it
output = merged[["custom_code", "Phecode", "PhecodeString", "PhecodeCategory"]].rename(
    columns={"custom_code": "ICD"}
).drop_duplicates()
```

**Save:**
```python
output.to_csv("mapping_dicts/<dataset>_icd_phecode_mapping.csv", index=False)
```

**Use:**
```python
from preprocessing import icd_to_events
obs = icd_to_events(
    df, icd_col="local_diag_code", date_col="encounter_date",
    mapping_file="mapping_dicts/<dataset>_icd_phecode_mapping.csv",
    has_dots=True,   # custom codes already have dots after normalization above
)
```

---

### 4b — Drug: custom → NDC → RxNorm ingredient

NDC codes come in many formats. Use `ndclib` to normalize to the canonical 11-digit
format before joining against `ndc_to_rxnorm_ingredient.csv`.

```python
import pandas as pd
from ndclib import NDC
from ndclib.exceptions import MissingNDCFormatException

crosswalk_df = pd.read_csv("crosswalk.csv")   # columns: custom_code, ndc
ndc_map      = pd.read_csv(
    "mapping_dicts/ndc_to_rxnorm_ingredient.csv",
    dtype={"ndc": str, "ingredient_id": str, "ingredient_name": str},
)

# Normalize unique NDC codes only (not every row) to minimize RxNorm API calls
# ndclib uses the RxNorm web API by default; it handles 10-digit, 11-digit,
# hyphenated and unhyphenated formats correctly.
# If vocabulary > ~500 unique codes, add time.sleep(0.1) between calls.
def normalize_ndc(raw_ndc: str) -> str | None:
    try:
        return NDC(str(raw_ndc)).ndc11
    except MissingNDCFormatException:
        return None   # UPC or unrecognized code — will surface in coverage report

unique_ndcs = crosswalk_df["ndc"].dropna().unique()
ndc_norm_map = {code: normalize_ndc(code) for code in unique_ndcs}
crosswalk_df["ndc11"] = crosswalk_df["ndc"].map(ndc_norm_map)

# Join normalized NDC → RxNorm ingredient
merged = crosswalk_df.merge(ndc_map, left_on="ndc11", right_on="ndc", how="left")

# Output schema must match ndc_to_rxnorm_ingredient.csv exactly
output = merged[["custom_code", "ingredient_id", "ingredient_name"]].rename(
    columns={"custom_code": "ndc"}
).drop_duplicates(subset=["ndc"])
```

**Save:**
```python
output.to_csv("mapping_dicts/<dataset>_ndc_rxnorm_mapping.csv", index=False)
```

**Use:**
```python
from preprocessing import drug_to_events
obs = drug_to_events(
    df, ndc_col="local_drug_code", date_col="rx_date",
    ndc_mapping_file="mapping_dicts/<dataset>_ndc_rxnorm_mapping.csv",
    drug_name_mapping_file=None,   # disable MIMIC-specific drug name fallback
)
```

> **Note on MediSpan:** If you have a Wolters Kluwer Medi-Span license, use the
> `MediSpanProvider` instead of the default RxNorm web API. This avoids rate limiting
> and also handles UPCs:
> ```python
> from ndclib import NDC, MediSpanProvider
> from pathlib import Path
> NDC.set_provider(MediSpanProvider(Path("/path/to/MEDFPLS/USAENG/DB/MEDNDC")))
> ```

---

### 4c — Drug: custom → free-text drug name → RxNorm ingredient

```python
import pandas as pd

crosswalk_df   = pd.read_csv("crosswalk.csv")   # columns: custom_code, drug_name
drug_name_map  = pd.read_csv(
    "mapping_dicts/drug_name_to_rxnorm_ingredient.csv",
    dtype=str,
)

crosswalk_df["drug_name_norm"] = crosswalk_df["drug_name"].str.strip().str.lower()
merged = crosswalk_df.merge(drug_name_map, left_on="drug_name_norm", right_on="drug_name", how="left")

output = merged[["custom_code", "ingredient_id", "ingredient_name"]].rename(
    columns={"custom_code": "drug_name"}
).drop_duplicates(subset=["drug_name"])
```

> **Note:** `drug_name_to_rxnorm_ingredient.csv` was generated from MIMIC drug names.
> If your drug names differ significantly, coverage will be low. Consider building a
> supplemental mapping via the RxNorm API (`/approximateTerm` endpoint) or Athena OMOP.

**Save:**
```python
output.to_csv("mapping_dicts/<dataset>_drugname_rxnorm_mapping.csv", index=False)
```

**Use:**
```python
obs = drug_to_events(
    df, ndc_col="dummy_ndc", date_col="rx_date",
    drug_col="local_drug_name",
    ndc_mapping_file="mapping_dicts/<dataset>_ndc_rxnorm_mapping.csv",
    drug_name_mapping_file="mapping_dicts/<dataset>_drugname_rxnorm_mapping.csv",
)
```

---

### 4d — Procedure: custom → CPT/HCPCS → CCS

`rollup_cpt_to_ccs()` handles the CPT/HCPCS → CCS join internally using range matching.
You only need to produce a file that maps your custom code → the CPT/HCPCS code,
then pass your data with that column.

```python
crosswalk_df = pd.read_csv("crosswalk.csv")   # columns: custom_code, cpt_code

# No further join needed — the crosswalk itself is the bridge.
# Just ensure cpt_code column is clean (uppercase, no spaces).
crosswalk_df["cpt_code"] = crosswalk_df["cpt_code"].str.strip().str.upper()
```

**Use:** merge your procedure table with `crosswalk_df` before calling `cpt_to_events()`:
```python
proc_df = proc_df.merge(crosswalk_df, on="local_proc_code", how="left")
obs = cpt_to_events(proc_df, cpt_col="cpt_code", date_col="proc_date")
```

---

### 4e — Lab: itemid → LOINC

MIMIC-IV `labevents` uses integer `itemid` keys. `rollup_itemid_to_loinc()` joins them
to LOINC codes using `mapping_dicts/d_labitems_to_loinc.csv` (MIT-LCP OMOP mapping,
1400/1630 itemids covered). For non-MIMIC datasets, produce a crosswalk to MIMIC itemids
or directly to LOINC codes and pass it via `mapping_file=`.

```python
from rollup import rollup_itemid_to_loinc
from preprocessing import lab_to_events

labevents_with_loinc = rollup_itemid_to_loinc(
    df            = labevents_df,
    itemid_column = "itemid",
    mapping_file  = "mapping_dicts/d_labitems_to_loinc.csv",
)

lab_obs = lab_to_events(
    df          = labevents_with_loinc,
    loinc_col   = "loinc_code",
    date_col    = "charttime",
    value_col   = "valuenum",
    subject_col = "subject_id",
)
# event format: "LOINC:11555-0"
```

---

## Step 5: Coverage report

Always report coverage before saving. Low coverage means either the bridge codes are
wrong format, or many codes genuinely have no rollup mapping.

```python
total     = len(output)
mapped    = output["Phecode"].notna().sum()   # adjust column name per modality
unmapped  = output["Phecode"].isna().sum()
pct       = 100 * mapped / total if total else 0

print(f"Total codes:    {total}")
print(f"Mapped:         {mapped} ({pct:.1f}%)")
print(f"Unmapped:       {unmapped}")

# Show top unmapped codes for diagnosis
if unmapped:
    top_unmapped = (
        output[output["Phecode"].isna()]
        .merge(crosswalk_df, on="ICD", how="left")[["ICD", "icd_code"]]
        .value_counts()
        .head(20)
    )
    print(top_unmapped.to_string())
```

**Interpreting coverage:**
- > 80%: good — proceed
- 60–80%: investigate top unmapped codes; may be legitimate gaps in the PheCode/RxNorm map
- < 60%: likely a format mismatch (dots, leading zeros, case sensitivity)

---

## Step 6: Use with the preprocessing pipeline

After saving the mapping CSV, pass it via the `mapping_file=` / `ndc_mapping_file=`
parameter of the corresponding function. The pipeline signature is already designed for
this — no code changes required.

| Modality | Function | Parameter |
|---|---|---|
| Diagnosis | `icd_to_events()` or `build_obs_log()` | `mapping_file=` / `icd_mapping_file=` |
| Drug (NDC) | `drug_to_events()` or `build_obs_log()` | `ndc_mapping_file=` |
| Drug (name) | `drug_to_events()` or `build_obs_log()` | `drug_name_mapping_file=` |
| Procedure | `cpt_to_events()` — no custom mapping file needed | (merge upstream) |
| Lab | `rollup_itemid_to_loinc()` → `lab_to_events()` | `mapping_file=` in rollup call |

---

## Generalizability notes

### What works out of the box for non-MIMIC data

| Feature | Status |
|---|---|
| Custom `subject_col` column name | Fully parameterized throughout |
| Custom date column names | Fully parameterized throughout |
| ICD codes with existing dots (TriNetX, OMOP, most EHRs) | `has_dots=True` in `icd_to_events()` / `build_obs_log(icd_has_dots=True)` |
| ICD codes without dots (MIMIC-IV) | `has_dots=False` or auto-detect |
| NDC in any format (hyphenated, 10-digit, 11-digit) | Handled via `ndclib` in this skill; `rollup.py` best-effort normalizes remaining cases |
| MIMIC drug name fallback (`drug_name_to_rxnorm_ingredient.csv`) | Optional; pass `drug_name_mapping_file=None` to disable for non-MIMIC data |

### Known gaps for non-MIMIC datasets

| Gap | Mitigation |
|---|---|
| Custom procedure codes that aren't CPT/HCPCS | Produce a custom → CPT crosswalk and merge upstream (Step 4d) |
| ICD-11 codes | No PheCode v1.2 mapping exists for ICD-11; use ICD-11 → ICD-10 crosswalk first (WHO provides this) |
| SNOMED diagnoses | Map SNOMED → ICD-10 via OMOP `CONCEPT_RELATIONSHIP` before this pipeline |
| ATC drug codes | Map ATC → RxNorm ingredient via OMOP `CONCEPT_RELATIONSHIP`; output `ndc, ingredient_id, ingredient_name` schema |
| Lab / vital sign events | Supported via `rollup_itemid_to_loinc()` → `lab_to_events()`; produces `"loinc"` event_type rows (see Step 4e) |
