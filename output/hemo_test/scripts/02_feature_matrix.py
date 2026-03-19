"""02 — Build MAP feature matrix: PheCodes (local) + CUI NLP features (BigQuery notes)."""

import sys
from pathlib import Path

import pandas as pd
from m4 import set_dataset, execute_query
from m4.config import set_active_backend
from once import get_once_features
from rollup import rollup_icd_to_phecode
from preprocessing import build_map_feature_matrix, build_note_proxy
from note_ner import extract_cui_features, aggregate_features

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "m4-pheno"))

out = Path(__file__).resolve().parent.parent
ONCE_CODIFIED = str(REPO / "ONCE_hemorrhoids_PheCode455_cos0.165.csv")
ONCE_NARRATIVE = str(
    REPO
    / "ONCE_PT_phenotype_hemorrhoids_C0019112_titlecos0.5_titlecut0.3_exactFALSE.csv"
)
PHECODE_MAP = str(REPO / "Phecode_map_v1_2_icd9_icd10cm.csv")
MAIN_PHECODE = "455"

# ── 1. Load upstream data ─────────────────────────────────────────────────────
subjects = pd.read_parquet(out / "data" / "subjects.parquet")
admissions = pd.read_parquet(out / "data" / "admissions.parquet")
diagnoses = pd.read_parquet(out / "data" / "diagnoses.parquet")
subject_ids = set(subjects["subject_id"].tolist())

# ── 2. Parse ONCE features ────────────────────────────────────────────────────
once = get_once_features(ONCE_CODIFIED, ONCE_NARRATIVE)
codified_list = once["codified_list"]
nlp_target_cuis = once["nlp_target_cuis"]

# Strip "PheCode:" prefix for build_map_feature_matrix
once_phecodes = [
    c.replace("PheCode:", "") for c in codified_list if c.startswith("PheCode:")
]
print(f"ONCE PheCodes: {once_phecodes}")
print(f"ONCE CUIs: {len(nlp_target_cuis)}")

# ── 3. ICD → PheCode rollup ───────────────────────────────────────────────────
phecode_df = rollup_icd_to_phecode(diagnoses, "icd_code", mapping_file=PHECODE_MAP)
print(
    f"PheCode rollup rows: {len(phecode_df)}, unmapped: {phecode_df['Phecode'].isna().sum()}"
)

# ── 4. Build codified feature matrix (patients × PheCodes) ────────────────────
mat_df = build_map_feature_matrix(
    phecode_df,
    once_phecodes=once_phecodes,
    main_phecode=MAIN_PHECODE,
    min_nonzero=5,  # relaxed for small demo cohort
)
print(f"Codified mat_df shape: {mat_df.shape}")

# ── 5. Fetch discharge notes from BigQuery ────────────────────────────────────
set_active_backend("bigquery")
set_dataset("mimic-iv-note")
subject_id_list = ", ".join(str(sid) for sid in sorted(subject_ids))
notes_raw = execute_query(f"""
    SELECT subject_id, note_id, text AS note_text
    FROM mimiciv_note.discharge
    WHERE subject_id IN ({subject_id_list})
""")
print(
    f"Notes fetched: {len(notes_raw)} (subjects with notes: {notes_raw['subject_id'].nunique()})"
)

# ── 6. NLP: MedSpaCy NER → CUI feature matrix ────────────────────────────────
if len(notes_raw) > 0:
    cui_long = extract_cui_features(
        notes_raw,
        text_column="note_text",
        id_column="subject_id",
        target_cuis=nlp_target_cuis,
    )
    print(f"CUI mentions extracted: {len(cui_long)}")

    if len(cui_long) > 0:
        cui_wide = aggregate_features(
            cui_long, id_column="subject_id", feature_column="cui"
        )
        print(f"CUI matrix shape: {cui_wide.shape}")

        # Merge codified + NLP matrices
        mat_df = mat_df.join(cui_wide, how="outer").fillna(0)
        print(f"Combined mat_df shape (codified + NLP): {mat_df.shape}")
    else:
        print("No CUI mentions found in notes — using codified features only.")
else:
    print("No notes found for demo subjects — using codified features only.")

# ── 7. Build note_df (note counts per patient as MAP denominator) ──────────────
# Prefer actual note count from BigQuery; fall back to admission proxy
if len(notes_raw) > 0:
    note_df = (
        notes_raw.groupby("subject_id")
        .size()
        .to_frame("note_count")
        .reindex(mat_df.index)
        .fillna(1)
        .clip(lower=1)
        .astype(int)
    )
else:
    # Fallback: admission count as proxy
    set_active_backend("duckdb")
    set_dataset("mimic-iv-demo")
    note_df = build_note_proxy(admissions, mat_df.index)

print(
    f"note_df shape: {note_df.shape}, mean note_count: {note_df['note_count'].mean():.1f}"
)

# ── 8. Save ───────────────────────────────────────────────────────────────────
mat_df.to_parquet(out / "data" / "mat_df.parquet")
note_df.to_parquet(out / "data" / "note_df.parquet")

print("Script 02 complete.")
