"""03 — Run MAP algorithm and assign HF case/control labels.

Study: HF_test_run_v1
Reads mat_df + note_df from data/ (produced by 02_feature_matrix.py).
Outputs map_results.parquet with columns: patient_id, score, phenotype.
"""

from pathlib import Path

import pandas as pd

from map import run_map

out = Path(__file__).resolve().parent.parent

MAIN_PHECODE = "428.1"
ANCHOR_COL = f"PheCode:{MAIN_PHECODE}"

# ── 1. Load feature matrix ─────────────────────────────────────────────────────
print("Loading feature matrix...")
mat_df = pd.read_parquet(out / "data" / "mat_df.parquet")
note_df = pd.read_parquet(out / "data" / "note_df.parquet")
print(f"  mat_df : {mat_df.shape[0]:,} patients × {mat_df.shape[1]} features")
print(f"  note_df: {len(note_df):,} patients")
print(f"  Anchor col '{ANCHOR_COL}' present: {ANCHOR_COL in mat_df.columns}")
print(f"  Patients with anchor > 0: {(mat_df[ANCHOR_COL] > 0).sum():,}")

# ── 2. Run MAP ─────────────────────────────────────────────────────────────────
print("\nRunning MAP (this shells out to Rscript)...")
map_results = run_map(mat_df, note_df, ANCHOR_COL)
print(f"  MAP results: {len(map_results):,} patients")
print(
    f"  Score range: [{map_results['score'].min():.4f}, {map_results['score'].max():.4f}]"
)
print(f"  Cases  (phenotype=1): {(map_results['phenotype'] == 1).sum():,}")
print(f"  Controls (phenotype=0): {(map_results['phenotype'] == 0).sum():,}")

# ── 3. Join with anchor ICD flag ───────────────────────────────────────────────
anchor_flag = (mat_df[ANCHOR_COL] > 0).rename("icd_coded").reset_index()
anchor_flag.columns = ["patient_id", "icd_coded"]

# MAP (via R) returns patient_id as int; mat_df index may be str (BigQuery ids).
# Normalise both to str before merging.
map_results["patient_id"] = map_results["patient_id"].astype(str)
anchor_flag["patient_id"] = anchor_flag["patient_id"].astype(str)

map_results = map_results.merge(anchor_flag, on="patient_id", how="left")

icd_only = ((map_results["icd_coded"]) & (map_results["phenotype"] == 0)).sum()
map_only = (~map_results["icd_coded"]) & (map_results["phenotype"] == 1)
map_only_n = map_only.sum()

print(f"\n  ICD-coded HF patients       : {map_results['icd_coded'].sum():,}")
print(f"  MAP cases                   : {(map_results['phenotype'] == 1).sum():,}")
print(f"  ICD-only (MAP rejected)     : {icd_only:,}  ← probable false positives")
print(f"  MAP-only (not ICD-coded)    : {map_only_n:,}  ← requires NLP to be non-zero")

# ── 4. Score distribution summary ─────────────────────────────────────────────
high_conf = (map_results["score"] > 0.8).sum()
low_conf = (map_results["score"] < 0.2).sum()
uncertain = ((map_results["score"] >= 0.2) & (map_results["score"] <= 0.8)).sum()
print(f"\n  Score > 0.8 (high-confidence cases)   : {high_conf:,}")
print(f"  Score < 0.2 (high-confidence controls) : {low_conf:,}")
print(f"  Score 0.2–0.8 (uncertain)              : {uncertain:,}")

# ── 5. Save ────────────────────────────────────────────────────────────────────
map_results.to_parquet(out / "data" / "map_results.parquet", index=False)
print("\nDone. Saved map_results.parquet")
print(f"  Columns: {list(map_results.columns)}")
