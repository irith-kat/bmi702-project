"""03 — Run MAP phenotyping on the hemorrhoid feature matrix."""

import sys
from pathlib import Path

import pandas as pd
from map import run_map

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "m4-pheno"))

out = Path(__file__).resolve().parent.parent
MAIN_PHECODE = "455"
THRESHOLD = 0.5

# ── 1. Load feature matrices ──────────────────────────────────────────────────
mat_df = pd.read_parquet(out / "data" / "mat_df.parquet")
note_df = pd.read_parquet(out / "data" / "note_df.parquet")

print(f"mat_df shape: {mat_df.shape}")
print(f"note_df shape: {note_df.shape}")
print(f"Anchor column '{MAIN_PHECODE}' present: {MAIN_PHECODE in mat_df.columns}")
print(f"Patients with PheCode 455 > 0: {(mat_df[MAIN_PHECODE] > 0).sum()}")

# ── 2. Run MAP ────────────────────────────────────────────────────────────────
print("\nRunning MAP algorithm (R subprocess)...")
scores_df = run_map(mat_df, note_df, main_icd_col=MAIN_PHECODE)
print(f"MAP output shape: {scores_df.shape}")
print(f"MAP output columns: {scores_df.columns.tolist()}")
print(scores_df.head(10))

# ── 3. Apply threshold → binary case labels ───────────────────────────────────
score_col = "score" if "score" in scores_df.columns else scores_df.columns[1]
scores_df["is_case"] = (scores_df[score_col] >= THRESHOLD).astype(int)

n_cases = scores_df["is_case"].sum()
n_total = len(scores_df)
print(
    f"\nCases (p >= {THRESHOLD}): {n_cases} / {n_total} ({100 * n_cases / n_total:.1f}%)"
)
print("\nScore distribution:")
print(scores_df[score_col].describe())

# ── 4. Save ───────────────────────────────────────────────────────────────────
scores_df.to_parquet(out / "data" / "map_scores.parquet", index=False)
print("\nScript 03 complete.")
