"""04 — Run MAP algorithm to assign HF case/control labels.

Study: hf-incident-v1
Reads : data/mat_df.parquet, data/note_df.parquet  (from 03_feature_matrix.py)
Writes: data/map_results.parquet

Run:
    cd output/hf-incident-v1
    uv run python scripts/04_map_phenotyping.py
"""

from pathlib import Path

import pandas as pd
from map import run_map

out = Path(__file__).resolve().parent.parent

MAIN_PHECODE = "428.1"
ANCHOR_COL = f"PheCode:{MAIN_PHECODE}"

print("Loading feature matrix...")
mat_df = pd.read_parquet(out / "data" / "mat_df.parquet")
note_df = pd.read_parquet(out / "data" / "note_df.parquet")
print(f"  mat_df : {mat_df.shape[0]:,} patients × {mat_df.shape[1]} features")
print(f"  note_df: {len(note_df):,} patients")
print(f"  Anchor '{ANCHOR_COL}' present: {ANCHOR_COL in mat_df.columns}")
print(f"  Patients with anchor > 0: {(mat_df[ANCHOR_COL] > 0).sum():,}")

print("\nRunning MAP...")
map_results = run_map(mat_df, note_df, ANCHOR_COL)
print(f"  MAP results: {len(map_results):,} patients")
print(
    f"  Score range: [{map_results['score'].min():.4f}, {map_results['score'].max():.4f}]"
)
print(f"  Cases    (phenotype=1): {(map_results['phenotype'] == 1).sum():,}")
print(f"  Controls (phenotype=0): {(map_results['phenotype'] == 0).sum():,}")

# Join ICD flag for cross-check
anchor_flag = (mat_df[ANCHOR_COL] > 0).rename("icd_coded").reset_index()
anchor_flag.columns = ["patient_id", "icd_coded"]
map_results["patient_id"] = map_results["patient_id"].astype(str)
anchor_flag["patient_id"] = anchor_flag["patient_id"].astype(str)
map_results = map_results.merge(anchor_flag, on="patient_id", how="left")

icd_only = ((map_results["icd_coded"]) & (map_results["phenotype"] == 0)).sum()
print(f"\n  ICD-coded patients  : {map_results['icd_coded'].sum():,}")
print(
    f"  ICD-only (MAP=0)    : {icd_only:,}  ← probable false positives filtered by MAP"
)

high_conf = (map_results["score"] > 0.8).sum()
low_conf = (map_results["score"] < 0.2).sum()
print(f"  High-confidence cases (score>0.8): {high_conf:,}")
print(f"  Low-confidence (score<0.2)        : {low_conf:,}")

map_results.to_parquet(out / "data" / "map_results.parquet", index=False)
print("\nSaved map_results.parquet")
