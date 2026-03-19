---
name: clinical-research-session
description: Start a structured clinical research session. Use when users describe research goals, want to analyze cohorts, investigate hypotheses, or need a rigorous research plan. Interviews the user, then produces a research protocol.
---

# Clinical Research Workflow

Structured clinical research from hypothesis through analysis. All work is organized as a **study** — one research question, one output directory, spanning one or more conversations.

## When This Skill Activates

- User invokes `/research` command
- User describes research intent: "I want to study...", "Can we analyze...", "What's the mortality rate for..."
- User mentions cohort analysis, hypothesis testing, or comparative studies

## Study Setup

Every research session is organized as a **study**. Set the study identifier at the start of every session:

```python
from pathlib import Path

STUDY = "early-vasopressors-sepsis-v1"
output_dir = Path("output") / STUDY
(output_dir / "scripts").mkdir(parents=True, exist_ok=True)
(output_dir / "data").mkdir(parents=True, exist_ok=True)
(output_dir / "plots").mkdir(parents=True, exist_ok=True)
```

**Continuing a study:** Check the existing `output_dir` for `PROTOCOL.md` and completed scripts to re-orient. Use terminal conversation to mark a new conversation within an ongoing study — not a new study.

**Branching:** Create a new version (`v2`) when a different analytical approach is needed.

### Output Structure

Scripts ARE the science. Create this structure at study start:

```
output/{study}/
├── PROTOCOL.md
├── RESULTS.md
├── scripts/
│   ├── 01_cohort_definition.py
│   ├── 02_baseline_characteristics.py
│   ├── 03_outcome_analysis.py
│   └── ...
├── data/
│   ├── cohort.parquet
│   ├── baseline_table.parquet
│   └── ...
└── plots/
    ├── age_distribution.json
    ├── kaplan_meier.json
    └── ...
```

### Script-First Workflow

Every analysis step that produces a result MUST be executed from a stored script. The script IS the analysis — not a retrospective summary of interactive work.

**The pattern — for every analysis step:**
1. **Write** the script to `scripts/NN_name.py`
2. **Run** it — outputs (parquets, JSON plots) land in `data/` and `plots/`
3. **Report** results by loading and summarizing the script's outputs in the terminal

Interactive exploration (checking schemas, small test queries to understand data shape) is fine — not everything needs a script. But the moment you produce a result to communicate to the researcher, it must come from a stored script.

**Script requirements:**
- **Self-contained**: imports, `set_dataset()`, SQL strings, analysis code, output writes — everything to run `python scripts/01_cohort_definition.py` from the study directory
- **Relative paths**: use `out = Path(__file__).resolve().parent.parent` to locate `data/` and `plots/`
- **Saves outputs**: `.parquet` to `data/`, Plotly figures as `.json` to `plots/` via `fig.write_json()` (never `.html` or `.png`)
- **Plotly reload**: `plotly.io.from_json(open("plots/fig.json").read())` to reconstruct a `Figure` from disk
- **Independent**: each script runs on its own; later scripts load earlier outputs from `data/`

**Example — one analysis step, start to finish:**

```python
# 1. Write the script
(output_dir / "scripts" / "01_cohort_definition.py").write_text('''\
"""01 — Define sepsis cohort from MIMIC-IV."""
from pathlib import Path
from m4 import execute_query, set_dataset

set_dataset("mimic-iv")
out = Path(__file__).resolve().parent.parent

sql = """
SELECT s.stay_id, s.subject_id, i.admission_age,
       a.hospital_expire_flag
FROM mimiciv_derived.sepsis3 s
INNER JOIN mimiciv_derived.icustay_detail i ON s.stay_id = i.stay_id
INNER JOIN mimiciv_hosp.admissions a ON s.hadm_id = a.hadm_id
WHERE i.first_icu_stay = true AND i.admission_age >= 18
"""
cohort = execute_query(sql)
cohort.to_parquet(out / "data" / "cohort.parquet")
print(f"Cohort: {len(cohort)} patients")
''')

# 2. Run it (via Bash tool)

# 3. Report results in terminal
import pandas as pd
cohort = pd.read_parquet(output_dir / "data" / "cohort.parquet")
print(cohort.describe())
```

---

## Phase 1: Research Interview

Collect study parameters through structured terminal conversation. The interview is **adaptive** — ask only what you cannot infer from the user's initial description.

**Guidelines:**
- **Skip questions** the user already answered in their prompt
- **Add questions** not in the library if the research question demands it
- **Ask in batches** where possible — not one question at a time

### Question Library

Use `AskUserQuestion` to collect structured answers. Compose from these standard questions, and add custom ones as needed:

**Research question type:**
- Association study (Is variable X associated with outcome Y?)
- Prediction model (Can we predict outcome Y from variables X?)
- Cohort characterization (What are the characteristics of population P?)
- Exploratory / hypothesis-generating

**Study design:**
- Descriptive — characterize a cohort
- Comparative — compare groups (treatment vs control)
- Predictive — build or validate a model
- Exploratory — clustering, pattern discovery

**Primary outcome:**
- In-hospital mortality (`hospital_expire_flag`)
- 28-day or 90-day mortality (`dod` relative to admission)
- ICU length of stay (`los` in `icustays` — survivor bias risk)
- Hospital length of stay (`dischtime` minus `admittime`)
- Ventilator-free days (28 minus days on MV)
- AKI incidence (KDIGO stage 2+ after exposure window)

**Exposure / intervention:**
- Treatment timing (early vs late initiation)
- Treatment dose / intensity (high vs low)
- Treatment received vs not (binary, any use within a window)
- Severity score / biomarker (continuous or categorical)
- None (descriptive study)

**Base population:**
- If there is any pre-defined population (e.g., sepsis, AKI, heart failure), identify it here. If not, MAP phenotyping will be used to identify the disease cohort from the full population.

**Exclusion criteria** (multiple allowed):
- First ICU stay only
- Age < 18
- ICU stay < 24h
- Early death within N hours of admission
- Missing key variables

**Confounders** (multiple allowed):
- Age, sex
- Illness severity (SOFA, APACHE III, SAPS-II)
- Charlson / Elixhauser comorbidities
- Admission type (medical vs surgical)
- Baseline labs (lactate, creatinine, bilirubin)
- Mechanical ventilation at baseline

**Dataset:**
- `mimic-iv` (full)
- `mimic-iv-demo` (100 patients, for testing)
- `eicu` (multi-center)

### After the Interview

Review answers in the terminal. Key refinements to consider:
- **Research question** — Make specific and answerable: "Are sicker patients dying more?" → "Is day-1 SOFA independently associated with 30-day mortality in sepsis?"
- **Outcome** — Confirm operationalization (table/column, survivor bias for LOS, follow-up window for X-free days)
- **Exposure** — Nail down time window, comparator, and immortal time bias risk
- **Confounders** — Check for mediators on the causal path (should NOT be adjusted for); consider propensity scores for treatment comparisons

---

## Phase 2: Research Protocol

Draft a structured protocol. Save to `output_dir / "PROTOCOL.md"` and show the researcher for approval before proceeding.

```markdown
## Research Protocol: [Title]

### Research Question
[Specific, answerable question]

### Study Design
[Descriptive/Comparative/Predictive/Exploratory/...]

### Population
**Inclusion:** [criteria]
**Exclusion:** [criteria with rationale]

### Variables (adapt the section as suitable for the study)
**Primary Outcome:** [definition and measurement]
**Exposure:** [definition and timing]
**Covariates:** [list with definitions]

### Analysis Plan
1. [Step with rationale]
2. ...

### Potential Biases & Limitations
- [Known limitation]

### Skills to Use
- [Skill]: [Why]
```

---

## Scientific Integrity Guardrails

Apply throughout the analysis.

### Bias Prevention

**Immortal Time Bias**
- Define exposure at a FIXED time point (admission, 24h, 48h)
- Never use "ever received during stay" for treatments
- Use landmark analysis when appropriate

**Selection Bias**
- Report all exclusions with counts (CONSORT flow)
- Analyze whether excluded patients differ systematically
- Avoid conditioning on post-treatment variables

**Information Leakage**
- ICD codes are assigned at DISCHARGE — don't use for admission predictions
- Length of stay is only known at discharge
- Labs/vitals must be timestamped appropriately

**Confounding by Indication**
- Treatments are given to sicker patients
- Always adjust for severity (SOFA, APACHE, SAPS)
- Consider propensity scores for treatment comparisons

### Statistical Rigor

- Pre-specify primary outcome; apply Bonferroni/FDR for secondary analyses
- Report cohort sizes at each step; be cautious with small subgroups
- Report missingness; consider imputation vs complete case; perform sensitivity analyses

### Visualizations

Use plots liberally — a chart often reveals what a table hides.

**Distributions → Plotly figures, not raw tables.** Categorical variables (race, gender, admission type) → horizontal bar chart. Continuous variables (age, LOS, SOFA) → histogram. Reserve tabular summaries for small counts (n, median, IQR).

**Every plot must have an explanation.** When reporting a figure to the researcher, always include a 1–4 sentence interpretation: what the plot shows and why it matters.

Save all Plotly figures as JSON:
```python
fig.write_json(output_dir / "plots" / "age_distribution.json")
# Reload later with:
import plotly.io as pio
fig = pio.from_json(open(output_dir / "plots" / "age_distribution.json").read())
```

Preferred plot types: Kaplan-Meier for survival, forest plots for effect sizes, covariate balance after matching, CONSORT flow diagrams.

### Reproducibility

- **Write → run → report.** Never report results from throwaway interactive code.
- **Iterate on scripts, not inline.** If a step needs fixing, edit the script and re-run.
- **Later scripts read earlier outputs.** Step 03 loads `data/cohort.parquet` from step 01 — not by re-running the query.

---

## M4 Skills Reference

### Phenotyping Pipeline (use in this order for disease cohort identification)

When the goal is to identify patients with a specific disease from EHR data, invoke these three skills **sequentially**. Each skill's output is the next skill's input.

| Step | Skill | What it does | Input | Output |
|------|-------|-------------|-------|--------|
| 1 | `preprocessing-strategy` | Parse ONCE files; roll up ICD→PheCode, CPT→CCS, etc. | Raw EHR tables (diagnoses, procedures, labs, prescriptions) | Per-modality DataFrames (phecode_df, ccs_df, …) |
| 2 | `build-datamart` | Pivot modality DataFrames + NLP CUI extraction into unified patient × feature matrix | Modality DataFrames + ONCE features + discharge notes | `mat_df`, `note_df` |
| 3 | `map-phenotyping` | Run MAP mixture model → per-patient probability scores + binary case/control labels | `mat_df`, `note_df`, anchor PheCode | `map_results` with `score` and `phenotype` columns |

**Trigger rule:** If the user mentions ONCE files, MAP phenotyping, or wants to identify patients with a disease condition, always start with `preprocessing-strategy` before touching any feature matrix code. If the ONCE files are not present, refer the user to https://shiny.parse-health.org/ONCE/ and make him download the NLP dictionary and the codified features, and place them in the repo root. Recommend applying a filter to phenotyping_features = True to reduce noise in the selection.

**Script naming convention for phenotyping:**
```
01_cohort_definition.py    ← pull raw EHR tables (m4-api)
02_feature_matrix.py       ← preprocessing-strategy + build-datamart
03_map_phenotyping.py      ← map-phenotyping
04_characterization.py     ← describe the identified cohort
```

**Critical: backend switching for notes**

Tabular EHR data (diagnoses, admissions, etc.) uses the DuckDB backend; clinical notes use BigQuery. Always switch backends explicitly when crossing data sources:

```python
from m4.config import set_active_backend
from m4 import set_dataset

# Tabular EHR (local DuckDB)
set_active_backend("duckdb")
set_dataset("mimic-iv-demo")  # or "mimic-iv"

# ... query diagnoses, procedures, etc. ...

# Switch to BigQuery for clinical notes
set_active_backend("bigquery")
set_dataset("mimic-iv-note")

# ... fetch discharge summaries, run NER ...
```

**min_nonzero guidance:** Always apply a global sparsity filter to ALL mat_df columns before running MAP — not just PheCodes. MAP's flexmix EM crashes ("Log-likelihood: NA") on any column with too few non-zero patients, regardless of dataset size. Sparsity is driven by term rarity, not cohort size: even in full MIMIC-IV, device codes or highly specific exam findings from the ONCE NLP dictionary may appear in <20 patients. Use `min_nonzero=20` (default) for full datasets, `5` for demo/pilot runs. The anchor PheCode is always retained.

### Data & Methodology
| Skill | When to Use |
|-------|-------------|
| `m4-api` | Multi-step analysis, statistical tests |
| `mimic-table-relationships` | Understanding joins |
---

## Example Flows

### Phenotyping study

**User:** "Find all patients with hemorrhoids using the ONCE files and MAP"

**What you already know:** disease = hemorrhoids, method = MAP phenotyping, ONCE files present. Skip most interview questions; ask only about dataset and goal (identification only vs. characterization).

**After response:**
- Confirm ONCE codified and narrative files exist in the repo root (glob `ONCE_*.csv`)
- Identify anchor PheCode from the codified file (`target_similarity == 1.0`)
- Note whether notes are needed (always yes for MAP — CUIs improve recall)

**Protocol → approval → execute (4 scripts):**
```
01_cohort_definition.py     ← m4-api: pull subjects, diagnoses, procedures (DuckDB)
02_feature_matrix.py        ← preprocessing-strategy + build-datamart:
                               rollup ICD→PheCode, fetch notes (BigQuery), NER, assemble mat_df + note_df
03_map_phenotyping.py       ← map-phenotyping: run MAP, apply threshold, save scored cohort
04_characterization.py      ← describe cases: demographics, comorbidities, top features
```

**Key checks before MAP:**
- Apply global sparsity filter (`min_nonzero`) to ALL columns after joining NLP features, not just PheCodes
- Verify anchor PheCode column exists in mat_df (bare string, e.g. `"455"`, not `"PheCode:455"`)
- Verify note_df has no zero values

---

### Comparative/outcomes study

**User:** "I want to study if early vasopressor use affects mortality in sepsis"

**What you already know:** comparative design, sepsis population, vasopressor exposure, mortality-related. Skip those questions.

**Interview:** Ask only what's missing — outcome definition (28-day vs in-hospital), "early" window, exclusion criteria, dataset. Ask all at once.

**After response, refine in terminal:**
- Anchor the time window to suspected infection onset (`suspicion-of-infection` skill), not ICU admission
- Recommend excluding death within the exposure window to avoid immortal time bias
- For treatment comparison, suggest propensity score matching over simple regression

**Protocol → approval → execute:**
- Save protocol to `output_dir / "PROTOCOL.md"`, present to researcher and wait for approval
- Each phase = write `scripts/NN_name.py` → run → report from `data/` and `plots/` outputs
- Ask for approval at decision points (cohort size, variable definitions, analysis method)
- Save final `RESULTS.md` and summarize findings

---

## Red Flags

Stop and reconsider if you see:
- **"Patients who survived to receive..."** → Immortal time bias
- **"Using ICD codes to identify patients at admission"** → Information leakage
- **"Complete cases only (N drops from X to Y)"** → Selection bias risk
- **"Treatment group had higher mortality"** → Confounding by indication
- **"47 significant associations"** → Multiple comparisons
- **"Small sample size but p < 0.05"** → Likely false positive

---

## After Completion

1. Save `RESULTS.md` to `output_dir` with findings, effect sizes, and confidence intervals
2. Summarize key findings in the terminal
3. Acknowledge limitations explicitly — include a limitations section in `RESULTS.md`
4. Suggest validation on independent data (e.g., eICU if MIMIC was used)
