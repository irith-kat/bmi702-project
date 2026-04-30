# Ablation Experiments

## Overview

The goal of this ablation study is to isolate the contributions of preprocessing (A1 -> A2), multimodality (A2 -> A3), and phenotyping (A3 -> A4). We define the following 4 configurations testing Claude Code's cohort-building performance with progressively richer skill sets.

| Setting | Skills | Python Modules | Data Directories |
|--------|--------|----------------|-----------------|
| Baseline | `clinical-research-session`, `m4-api`, `mimic-table-relationships` |   | `mimiciv/` |
| Structured | + `mimic-preprocessing` | `src/preprocessing/structured/` | + `mapping_dicts/` |
| Multimodal | + `mimic-note-preprocessing` | + `src/preprocessing/nlp/` | + `input/` (ONCE files) |
| Full | + `phenotyping-strategy`, `map-phenotyping`, `latte-phenotyping` | + `src/map/` + `src/latte/` + `src/LATTE-main/` |  |

IMPORTANT: `clinical-research-session`, which is our orchestrator skill, must change according to each ablation setup and waht other skills are (not) available. The current `clinical-research-session` supports the full ablation. When creating each ablation config, edit and prune the `clinical-research-session` as necessary.

## Setup

Ablation configs live outside the main git repo at `../ablation-configs/` (sibling to `bmi702-project/`).

If not already done, install Claude to run on terminal: `curl -fsSL https://claude.ai/install.sh | bash` (Mac). To launch and authenticate, type `claude` and follow the instructions.

## Running Experiments

**DISCLAIMER: The following commands use `--dangerously-skip-permissions` to automatically run experiments. Remove that if you are uncomfortable with bypassing permissions.**

Open new terminal for each ablation config. From root folder containing `bmi702-project` and `ablation-configs`:

```bash
# A1 — base
cd ablation-configs/A1-base
claude --dangerously-skip-permissions --add-dir ../../bmi702-project/mimiciv

# A2 — + structured preprocessing
cd ablation-configs/A2-structured
claude --dangerously-skip-permissions \
  --add-dir ../../bmi702-project/mimiciv \
  --add-dir ../../bmi702-project/mapping_dicts \
  --add-dir ../../bmi702-project/src/preprocessing/structured

# A3 — + NLP
cd ablation-configs/A3-nlp
claude --dangerously-skip-permissions \
  --add-dir ../../bmi702-project/mimiciv \
  --add-dir ../../bmi702-project/mapping_dicts \
  --add-dir ../../bmi702-project/input \
  --add-dir ../../bmi702-project/src/preprocessing/structured \
  --add-dir ../../bmi702-project/src/preprocessing/nlp

# A4 — full
cd ablation-configs/A4-full
claude --dangerously-skip-permissions \
  --add-dir ../../bmi702-project/mimiciv \
  --add-dir ../../bmi702-project/mapping_dicts \
  --add-dir ../../bmi702-project/input \
  --add-dir ../../bmi702-project/src/preprocessing/structured \
  --add-dir ../../bmi702-project/src/preprocessing/nlp \
  --add-dir ../../bmi702-project/src/map \
  --add-dir ../../bmi702-project/src/latte \
  --add-dir ../../bmi702-project/src/LATTE-main
```

## Prompt

Use the same prompt for all four configs:

```
/clinical-research-session Build a cohort of patients with heart failure from MIMIC-IV.

When finished, save the following to a file called OUTPUT_COHORT.json in the current directory:
{
  "n_cases": <int>,
  "n_controls": <int>,
  "case_ids": [<list of subject_id>],
  "control_ids": [<list of subject_id>],
  "map_scores": {<subject_id: float>}, # ONLY FOR FULL ABLATION
  "icd_only_count": <int for ICD-coded but MAP-rejected>, # ONLY FOR FULL ABLATION
  "map_only_count": <int for MAP-found, ICD-missed>, # ONLY FOR FULL ABLATION
  "feature_types_used": [<"icd", "rx", "lab", "cpt", "nlp">],
  "consort_flow": [
    {"stage": "<e.g. total patients>", "n": <int>},
    {"stage": "<e.g. anchor ICD candidates>", "n": <int>},
    {"stage": "<...>", "n": <int>},
    {"stage": "final cases", "n": <int>},
    {"stage": "final controls", "n": <int>}
  ],
  "top_driving_codes": [<up to 20 codes with weights>],
  "comorbidity_prevalence_cases": {
    "hypertension": <float>,
    "ckd": <float>,
    "diabetes": <float>,
    "atrial_fibrillation": <float>,
    "coronary_artery_disease": <float>,
    <add any other comorbidities identified as clinically relevant during cohort construction>
  },
  "methodology_summary": "<4-5 sentences covering: (1) which feature types were available and used (ICD, labs, medications, NLP CUIs, MAP scores) and why each was or was not included; (2) how cases and controls were defined, including any code-frequency thresholds or MAP score cutoffs applied; (3) any preprocessing steps taken (rollups, vocabulary mapping, ONCE feature extraction) and how they shaped the final observation log; (4) known limitations of this configuration — what signal was unavailable and how that may have affected case identification or label quality; (5) any notable assumptions or judgment calls made during cohort construction specific to heart failure.>"
}
```

After running each ablation, save the Claude Code transcript using the slash command `\export` and then selecting save to file!

## Notes

- The `docs`, `notebooks`, `tests`, and `figures` directories are excluded via `.claudeignore` in each ablation config
