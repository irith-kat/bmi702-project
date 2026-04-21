# Ablation Experiments

## Overview

Four configurations testing Claude Code cohort-building performance with progressively richer skill sets.

| Config | Skills | Python Modules | Data Directories |
|--------|--------|----------------|-----------------|
| A1-base | `clinical-research-session`, `m4-api`, `mimic-table-relationships` |   | `mimiciv/` |
| A2-structured | + `mimic-preprocessing` | `src/preprocessing/structured/` | + `mapping_dicts/` |
| A3-nlp | + `mimic-note-preprocessing` | + `src/preprocessing/nlp/` | + `input/` (ONCE files) |
| A4-full | + `phenotyping-strategy`, `map-phenotyping`, `latte-phenotyping` | + `src/map/` + `src/latte/` |  |

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
  --add-dir ../../bmi702-project/src/latte
```

## Prompt

Use the same prompt for all four configs:

```
/clinical-research-session Build a cohort of patients with heart failure from MIMIC-IV.
```

## Notes

- `docs/`, `notebooks/`, and `figures/` are excluded via `.claudeignore` in each ablation config
