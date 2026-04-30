"""09 — MAP accuracy evaluation against Gemini gold labels.

Evaluates MAP binary classification accuracy using a balanced sample of
MAP-positive and MAP-negative patients, all reviewed by Gemini.

The existing gold labels (gemini_incident_results.parquet) cover only
MAP-positive patients.  This script samples MAP-negative patients from the
unlabeled pool and labels them via Gemini using a SEPARATE cache so the
main pipeline files are not modified.

Outputs
-------
  data/gemini_neg_sample_cache.jsonl     — Gemini API cache for MAP-neg patients
  data/gemini_neg_sample_results.parquet — parsed Gemini results for MAP-neg
  plots/09_map_accuracy.png              — confusion matrix + breakdown + metrics

Run
---
  cd output/hf-incident-v1
  uv run python scripts/09_map_accuracy.py
"""

import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from latte.gemini import (
    get_cached_subject_ids,
    parse_gemini_results,
    run_gemini_labeling,
)
from latte.labeler_utils import (
    HF_DISEASE_CONFIG,
    build_result_record,
    build_system_instruction,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

out = Path(__file__).resolve().parent.parent

# ── Config ─────────────────────────────────────────────────────────────────────
N_POS_SAMPLE = 50  # MAP-positive patients to subsample from gold labels
N_NEG_SAMPLE = 50  # MAP-negative patients to label (separate cache)
SEED = 42
BASELINE_DATE = "2100-01-01"
MONTH_WINDOW = 2

NEG_CACHE_JSONL = str(out / "data" / "gemini_neg_sample_cache.jsonl")
NEG_RESULTS_PATH = out / "data" / "gemini_neg_sample_results.parquet"
PLOT_PATH = out / "plots" / "09_map_accuracy.png"

PROJECT_ID = "just-duality-438820-n4"
LOCATION = "global"
MODEL = "publishers/google/models/gemini-3.1-flash-lite-preview"

# ── 1. Load existing data ──────────────────────────────────────────────────────
print("Loading data...")
map_results = pd.read_parquet(out / "data" / "map_results.parquet")
map_results["patient_id"] = map_results["patient_id"].astype(str)

gem_pos = pd.read_parquet(out / "data" / "gemini_incident_results.parquet")
gem_pos = gem_pos[~gem_pos["parse_error"] & (gem_pos["label"] != -1)].copy()
gem_pos["subject_id"] = gem_pos["subject_id"].astype(str)

print(f"  MAP results      : {len(map_results):,} patients")
print(f"  Existing gold    : {len(gem_pos)} MAP-positive patients")
print(f"    label=1 (HF)   : {(gem_pos['label'] == 1).sum()}")
print(f"    label=0 (no HF): {(gem_pos['label'] == 0).sum()}")

# ── 2. Subsample N_POS_SAMPLE MAP-positive patients ────────────────────────────
rng = np.random.default_rng(SEED)
n_pos = min(N_POS_SAMPLE, len(gem_pos))
pos_ids = rng.choice(gem_pos["subject_id"].values, size=n_pos, replace=False)
pos_sample = gem_pos[gem_pos["subject_id"].isin(pos_ids)].copy()
print(f"\nSubsampled {len(pos_sample)} MAP-positive patients (seed={SEED})")
print(
    f"  label=1: {(pos_sample['label'] == 1).sum()}  label=0: {(pos_sample['label'] == 0).sum()}"
)

# ── 3. Sample MAP-negative candidates ─────────────────────────────────────────
map_neg_ids = map_results[map_results["phenotype"] == 0]["patient_id"].tolist()
print(f"\nMAP-negative patients in cohort: {len(map_neg_ids):,}")

# Restrict to patients with discharge notes
notes_meta = pd.read_parquet(out / "data" / "notes_raw.parquet", columns=["subject_id"])
valid_sids = set(notes_meta["subject_id"].astype(str).unique())
neg_candidates = [p for p in map_neg_ids if p in valid_sids]
print(f"  With discharge notes: {len(neg_candidates):,}")

# Prefer already-cached patients to avoid unnecessary API calls
cached_neg_sids = get_cached_subject_ids(NEG_CACHE_JSONL)
print(f"  Already in neg cache: {len(cached_neg_sids)}")


def _sample_with_pref(candidates: list, n: int, preferred: set, rng) -> list:
    pref = [p for p in candidates if p in preferred]
    others = [p for p in candidates if p not in preferred]
    chosen = pref[:n]
    gap = n - len(chosen)
    if gap > 0 and others:
        chosen += rng.choice(others, size=min(gap, len(others)), replace=False).tolist()
    return chosen


neg_sample_ids = _sample_with_pref(neg_candidates, N_NEG_SAMPLE, cached_neg_sids, rng)
print(f"  Selected {len(neg_sample_ids)} MAP-negative patients for labeling")

# ── 4. Load notes for MAP-negative sample ─────────────────────────────────────
print("\nLoading notes for MAP-negative sample (may take a moment)...")
notes_all = pd.read_parquet(out / "data" / "notes_raw.parquet")
neg_id_set = set(neg_sample_ids)
notes_neg = notes_all[notes_all["subject_id"].astype(str).isin(neg_id_set)].copy()
notes_neg = notes_neg.sort_values(["subject_id", "charttime"]).reset_index(drop=True)
neg_with_notes = [
    s for s in neg_sample_ids if s in set(notes_neg["subject_id"].astype(str))
]
print(f"  Notes found for {len(neg_with_notes)} / {len(neg_sample_ids)} patients")

# ── 5. Label MAP-negative patients with Gemini (separate cache) ───────────────
print(f"\nRunning Gemini labeling for {len(neg_with_notes)} MAP-negative patients...")
print(f"  Cache: {NEG_CACHE_JSONL}")
n_new = run_gemini_labeling(
    notes_df=notes_neg,
    subject_ids=neg_with_notes,
    cache_jsonl=NEG_CACHE_JSONL,
    config=HF_DISEASE_CONFIG,
    model_name=MODEL,
    project_id=PROJECT_ID,
    location=LOCATION,
    max_notes_per_patient=60,
    retry_delay_seconds=5.0,
    record_builder=build_result_record,
    system_instruction_builder=build_system_instruction,
)
print(f"  Newly labeled by Gemini: {n_new}")

# ── 6. Parse MAP-negative results ─────────────────────────────────────────────
print("\nParsing MAP-negative Gemini results...")
gem_neg_all = parse_gemini_results(
    cache_jsonl=NEG_CACHE_JSONL,
    baseline_date=BASELINE_DATE,
    month_window=MONTH_WINDOW,
)
gem_neg_all["subject_id"] = gem_neg_all["subject_id"].astype(str)
gem_neg = (
    gem_neg_all[gem_neg_all["subject_id"].isin([str(s) for s in neg_with_notes])]
    .copy()
    .reset_index(drop=True)
)
gem_neg = gem_neg[~gem_neg["parse_error"] & (gem_neg["label"] != -1)].copy()
print(f"  Valid results : {len(gem_neg)} MAP-negative patients")
print(f"  label=1 (HF) : {(gem_neg['label'] == 1).sum()}")
print(f"  label=0 (no HF): {(gem_neg['label'] == 0).sum()}")
gem_neg.to_parquet(NEG_RESULTS_PATH, index=False)
print(f"  Saved → {NEG_RESULTS_PATH.name}")

# ── 7. Build comparison table ──────────────────────────────────────────────────
print("\nBuilding TP/TN/FP/FN table...")
pos_df = pos_sample[["subject_id", "label"]].copy()
pos_df["map"] = 1
neg_df = gem_neg[["subject_id", "label"]].copy()
neg_df["map"] = 0

comparison = pd.concat([pos_df, neg_df], ignore_index=True)
comparison = comparison.rename(columns={"label": "gemini"})

comparison["outcome"] = comparison.apply(
    lambda r: "TP"
    if r["map"] == 1 and r["gemini"] == 1
    else "FP"
    if r["map"] == 1 and r["gemini"] == 0
    else "TN"
    if r["map"] == 0 and r["gemini"] == 0
    else "FN",
    axis=1,
)

counts = comparison["outcome"].value_counts()
TP = int(counts.get("TP", 0))
TN = int(counts.get("TN", 0))
FP = int(counts.get("FP", 0))
FN = int(counts.get("FN", 0))
N = len(comparison)

accuracy = (TP + TN) / N
precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
recall = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
specificity = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else float("nan")
)

print(f"\n{'=' * 40}")
print(f"  MAP vs Gemini — Balanced Sample (N={N})")
print(f"  MAP+ : {len(pos_df)}    MAP- : {len(neg_df)}")
print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
print(f"  Accuracy    : {accuracy:.3f}")
print(f"  Precision   : {precision:.3f}")
print(f"  Recall      : {recall:.3f}")
print(f"  Specificity : {specificity:.3f}")
print(f"  F1 Score    : {f1:.3f}")
print(f"{'=' * 40}\n")

# ── 8. Plot ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45, left=0.06, right=0.97)

# ── 8a. Confusion matrix ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
# rows = Gemini (actual), cols = MAP (predicted)
cm = np.array([[TN, FP], [FN, TP]])
im = ax1.imshow(cm, cmap="Blues", vmin=0, vmax=max(TP, TN, FP, FN) + 2)

ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(["MAP Neg (0)", "MAP Pos (1)"], fontsize=10)
ax1.set_yticklabels(["Gemini Neg\n(label=0)", "Gemini Pos\n(label=1)"], fontsize=10)
ax1.set_xlabel("MAP Predicted Label", fontsize=11, labelpad=8)
ax1.set_ylabel("Gemini Gold Label", fontsize=11, labelpad=8)
ax1.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=10)

cell_labels = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
threshold = cm.max() * 0.55
for (i, j), lbl in cell_labels.items():
    val = cm[i, j]
    color = "white" if val > threshold else "black"
    ax1.text(
        j,
        i,
        f"{lbl}\n{val}",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=color,
    )

plt.colorbar(im, ax=ax1, shrink=0.75, pad=0.04)

# ── 8b. Bar chart ──────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
bar_labels = ["TP", "TN", "FP", "FN"]
bar_vals = [TP, TN, FP, FN]
bar_colors = ["#27ae60", "#2980b9", "#e74c3c", "#e67e22"]
bars = ax2.bar(
    bar_labels, bar_vals, color=bar_colors, edgecolor="white", linewidth=1.5, width=0.55
)
ax2.set_ylabel("Count", fontsize=11)
ax2.set_title(
    "Classification Breakdown\n(Balanced Sample)",
    fontsize=13,
    fontweight="bold",
    pad=10,
)
ax2.set_ylim(0, max(bar_vals) * 1.3)
for bar, val in zip(bars, bar_vals):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(bar_vals) * 0.03,
        str(val),
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(axis="x", labelsize=12)

# ── 8c. Metrics table ──────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.axis("off")

rows = [
    ["Accuracy", f"{accuracy:.3f}"],
    ["Precision (PPV)", f"{precision:.3f}"],
    ["Recall (Sens.)", f"{recall:.3f}"],
    ["Specificity", f"{specificity:.3f}"],
    ["F1 Score", f"{f1:.3f}"],
    ["", ""],
    ["N (total)", str(N)],
    ["MAP-positive", str(len(pos_df))],
    ["MAP-negative", str(len(neg_df))],
    ["TP / TN / FP / FN", f"{TP}/{TN}/{FP}/{FN}"],
]

tbl = ax3.table(
    cellText=rows,
    colLabels=["Metric", "Value"],
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif rows[row - 1][0] == "":
        cell.set_facecolor("#f8f8f8")
    elif row % 2 == 0:
        cell.set_facecolor("#eaf2fb")
    else:
        cell.set_facecolor("#ffffff")

ax3.set_title("Summary Metrics", fontsize=13, fontweight="bold", pad=12)

fig.suptitle(
    f"MAP Binary Phenotype Accuracy vs Gemini Gold Labels  ·  hf-incident-v1\n"
    f"Balanced sample: {len(pos_df)} MAP-positive + {len(neg_df)} MAP-negative  "
    f"(MAP-negative labeled separately, seed={SEED})",
    fontsize=12,
    y=1.03,
)

PLOT_PATH.parent.mkdir(exist_ok=True)
fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Plot saved → {PLOT_PATH}")
plt.close()
