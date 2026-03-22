---
name: clinical-research-session
description: Start a structured clinical research session for cohort building (rule-based or MAP-based phenotyping). Use when users describe research goals, want to analyze cohorts, investigate hypotheses, or need a rigorous research plan. Interviews the user, decides the phenotyping approach, then produces a research protocol.
---

# Clinical Research Workflow

Structured cohort building from research question through characterization. All work is organized as a **study** — one research question, one output directory, spanning one or more conversations.

## When This Skill Activates

- User invokes `/research` command
- User describes research intent: "I want to study...", "Can we analyze...", "Find patients with..."
- User mentions cohort analysis, cohort characterization, or disease identification

## Study Setup

Every research session is organized as a **study**. Set the study identifier at the start of every session:

```python
from pathlib import Path

STUDY = "hemorrhoids-cohort-v1"
output_dir = Path("output") / STUDY
(output_dir / "scripts").mkdir(parents=True, exist_ok=True)
(output_dir / "data").mkdir(parents=True, exist_ok=True)
(output_dir / "plots").mkdir(parents=True, exist_ok=True)
```

**Continuing a study:** Check the existing `output_dir` for `PROTOCOL.md` and completed scripts to re-orient. Use terminal conversation to mark a new conversation within an ongoing study — not a new study.

**Branching:** Create a new version (`v2`) when a different analytical approach is needed.

### Output Structure

Scripts ARE the science. Create this folder structure at study start:

```
output/{study}/
├── PROTOCOL.md
├── scripts/
│   ├── cohort_definition.py
│   ├── nlp_features.py
│   ├── map_phenotyping.py
│   └── characterization.py
├── data/
│   ├── cohort.parquet
│   └── ...
└── plots/
    ├── age_distribution.json
    └── ...
```

Scripts must be self-contained and independent — each loads what it needs from `data/`.

### Script-First Workflow

Every analysis step that produces a result MUST be executed from a stored script. The script IS the analysis — not a retrospective summary of interactive work.

**The pattern — for every analysis step:**
1. **Write** the script to `scripts/`
2. **Run** it — outputs (parquets, JSON plots) land in `data/` and `plots/`
3. **Report** results by loading and summarizing the script's outputs in the terminal

Interactive exploration (checking schemas, small test queries to understand data shape) is fine — not everything needs a script. But the moment you produce a result to communicate to the researcher, it must come from a stored script.

**Script requirements:**
- **Self-contained**: imports, `set_dataset()`, SQL strings, analysis code, output writes — everything to run `python scripts/cohort_definition.py` from the study directory
- **Relative paths**: use `out = Path(__file__).resolve().parent.parent` to locate `data/` and `plots/`
- **Saves outputs**: `.parquet` to `data/`, Plotly figures as `.json` to `plots/` via `fig.write_json()` (never `.html` or `.png`)
- **Plotly reload**: `plotly.io.from_json(open("plots/fig.json").read())` to reconstruct a `Figure` from disk
- **Independent**: each script runs on its own; later scripts load earlier outputs from `data/`

---

## Phase 1: Research Interview

Collect study parameters through structured terminal conversation. The interview is **adaptive** — ask only what you cannot infer from the user's initial description.

**Guidelines:**
- **Skip questions** the user already answered in their prompt
- **Add questions** not in the library if the research question demands it
- **Ask in batches** where possible — not one question at a time

### Question Library

Use `AskUserQuestion` to collect structured answers. Compose from these standard questions, and add custom ones as needed:

**Target phenotype / disease:**
- What condition or phenotype are you trying to identify?
- Is there a pre-existing ICD or PheCode to anchor on? If unknown, a lookup can help.

**Scope of the session:**
- Cohort identification only (who has the condition?)
- Cohort identification + characterization (who has it, and what do they look like?)
- Downstream analysis planned? (flag for a future session, not this one)

**Dataset:**
- `mimic-iv` (full)
- `mimic-iv-demo` (100 patients, for testing/development)

**NLP / notes:**
- Should clinical notes be used for NLP CUI features? (improves sensitivity; adds extra runtime — see `mimic-note-preprocessing`)

**ONCE files:**
- Are ONCE files available in `input/`? (required for MAP; generate at https://shiny.parse-health.org/ONCE/ if not)

**Exclusion criteria** (multiple allowed):
- Age < 18
- Missing key variables
- Specific admission types to exclude

**Characterization goals** (if cohort characterization is in scope):
- Demographics (age, sex, race)
- Comorbidities (top ICD clusters, Charlson/Elixhauser)
- Top ONCE features present in cases vs controls
- Temporal patterns (admission trends, seasonality)

### After the Interview

Review answers in the terminal. Key refinements to consider:
- **Anchor ICD/PheCode** — Confirm it exists before proceeding
- **Dataset** — Recommend demo for first runs; switch to full MIMIC-IV once the pipeline validates
- **NLP decision** — Confirm whether notes will be used before writing feature-matrix code

**Decide phenotyping approach** using the `phenotyping-strategy` skill criteria:

| Signal | Points toward |
|--------|--------------|
| ONCE files available | MAP |
| ≥1,000 candidates expected | MAP |
| ICD specificity known to be low (e.g. RA ~58% PPV) | MAP |
| Need per-patient probability scores | MAP |
| No ONCE files / exploratory / pilot | Rule-based |
| Small dataset (<200 candidates) | Rule-based |
| Simple inclusion criteria only | Rule-based |

State the chosen approach and reason explicitly before drafting the protocol.

---

## Phase 2: Research Protocol

Draft a structured protocol. Save to `output_dir / "PROTOCOL.md"` and show the researcher for approval before proceeding.

```markdown
## Research Protocol: [Title]

### Research Question
[Specific, answerable question — e.g. "Identify and characterize patients with hemorrhoids in MIMIC-IV using ICD-based cohort definition"]

### Study Design
[Cohort identification / Cohort identification + characterization]

### Population
**Source:** [MIMIC-IV full / demo]
**Anchor phenotype:** [PheCode + disease name]
**Inclusion:** [criteria]
**Exclusion:** [criteria with rationale]

### Cohort Definition Approach
**Method:** [Rule-based ICD filter | MAP (Multimodal Automated Phenotyping)]
**Anchor ICD/PheCode:** [e.g. 455 — Hemorrhoids]
**NLP:** [Yes — clinical notes / No — structured EHR only]
**ONCE files:** [codified file name, narrative file name, or N/A]
**MAP rationale:** [why MAP over rule-based — e.g. "RA ICD specificity ~58%; ONCE files available; ≥1k candidates"]
**MAP config:** [min_nonzero threshold; NLP contribution expected: high / low / unknown]

### Characterization Plan
1. [Demographics — age distribution, sex, race]
2. [Top comorbidities]
3. [Top ONCE features in cases vs controls]
4. [Other planned summaries]

### Potential Limitations
- [Known limitation]

### Skills to Use
- [Skill]: [Why]
```

---

## Scientific Integrity Guardrails

Apply throughout the analysis.

### Bias Prevention

**Selection Bias**
- Report all exclusions with counts (CONSORT flow) — how many patients entered, how many excluded at each step, how many remain
- If patients are excluded for missing data, check whether they differ from included patients (age, sex, severity)
- Avoid conditioning on post-phenotyping variables when building the case/control split

**Information Leakage (ICD codes)**
- ICD codes in MIMIC-IV are assigned at **discharge**, not admission — they reflect the full stay
- This is generally fine for phenotyping (you want the discharge diagnosis), but flag it if downstream analysis requires admission-time features
- Do not use ICD codes to define the anchor phenotype AND as features simultaneously without acknowledging circularity

### Statistical Rigor

- Report cohort sizes at each pipeline step (raw → after exclusions → final case/control split)
- Report feature sparsity before and after filtering
- Be cautious with small subgroups; suppress characterization tables for cells n < 5

### Visualizations

Use plots liberally — a chart often reveals what a table hides.

**Distributions → Plotly figures, not raw tables.** Categorical variables (race, gender, admission type) → horizontal bar chart. Continuous variables (age, LOS) → histogram. Reserve tabular summaries for small counts (n, median, IQR).

**Every plot must have an explanation.** When reporting a figure to the researcher, always include a 1–4 sentence interpretation: what the plot shows and why it matters.

Save all Plotly figures as JSON:
```python
fig.write_json(output_dir / "plots" / "age_distribution.json")
# Reload later with:
import plotly.io as pio
fig = pio.from_json(open(output_dir / "plots" / "age_distribution.json").read())
```

Preferred plot types for cohort characterization: CONSORT flow (annotated diagram or table), demographic bar charts, feature prevalence comparison (cases vs controls).

**CONSORT flow stages** (include only stages that apply to the study):
1. Total patients in dataset
2. After structured EHR filtering (anchor phenotype candidates identified)
3. After exclusion criteria applied (age, missing data, etc.)
4. After NLP/notes filtering (if enabled)
5. After MAP phenotyping → final case and control counts

### Reproducibility

- **Write → run → report.** Never report results from throwaway interactive code.
- **Iterate on scripts, not inline.** If a step needs fixing, edit the script and re-run.
- **Later scripts read earlier outputs.** The characterization script loads `data/cohort.parquet` from the cohort definition script — not by re-running the query.

---

## M4 Skills Reference

### Data Preparation Pipeline

Steps must run in order — each step depends on the output of the previous.

| Step | Skill | What it does | Input | Output |
|------|-------|-------------|-------|--------|
| 1 | `mimic-preprocessing` | Roll up ICD→PheCode, CPT→CCS, NDC→RxNorm; assemble observation log | Raw EHR tables | `obs_log` |
| 2 | `mimic-note-preprocessing` | Extract CUI mentions from discharge notes via MedSpaCy; append to obs_log | `obs_log` + ONCE narrative CUIs + notes | `obs_log` + `event_type="cui"` rows |
| 3 | `map-phenotyping` | Fit Poisson mixture model over ONCE co-features; assign posterior probabilities and binary labels | `obs_log` + ONCE files | `map_results` with `phenotype` column |

**ONCE files:** Required for MAP. Place codified and narrative feature files in `input/`. Generate at https://shiny.parse-health.org/ONCE/ if not present. Enable `phenotyping_features = True` to reduce noise.

### Data & Methodology
| Skill | When to Use |
|-------|-------------|
| `m4-api` | Writing SQL queries, multi-step data access |
| `mimic-table-relationships` | Understanding joins, avoiding duplicates |
| `phenotyping-strategy` | Deciding MAP vs rule-based at study start |

---

## Example Flow

### Hemorrhoid cohort identification and characterization

**User:** "Find all patients with hemorrhoids using ICD codes, then characterize the cohort"

**What you already know:** disease = hemorrhoids, method = rule-based ICD filter, goal = identification + characterization. Skip most interview questions.

**Interview (ask in one batch):**
- Which dataset — `mimic-iv` or `mimic-iv-demo` for development?
- Should clinical notes be used for NLP CUI features?
- Any exclusion criteria (e.g. age < 18)?
- Are ONCE files available in `input/`?

**After response:**
- Confirm the anchor ICD/PheCode exists in the data
- Note NLP decision

**Draft protocol → show to researcher → wait for approval**

**Execute — three scripts:**

```
cohort_definition.py  ← m4-api: pull diagnoses_icd, procedures, prescriptions;
                         build obs_log via mimic-preprocessing;
                         save obs_log.parquet

nlp_features.py       ← [if NLP enabled]: load obs_log.parquet; filter candidates,
                         fetch notes, run MedSpaCy NER via mimic-note-preprocessing,
                         append CUI rows; save obs_log_with_nlp.parquet

map_phenotyping.py    ← load obs_log.parquet (or obs_log_with_nlp.parquet); load ONCE files;
                         run map-phenotyping (preprocess_map + run_map) → map_results.parquet;
                         save cohort.parquet (cases + controls)

characterization.py   ← load cohort.parquet; age histogram, sex/race bar charts,
                         top-10 comorbidities, top ONCE features cases vs controls
```

**After cohort is defined:**
- Report CONSORT flow: total subjects → candidates → cases → controls
- Compare top features between cases and controls — feature prevalence bar chart

---

## Red Flags

Stop and reconsider if you see:
- **"Using ICD codes to identify patients at admission"** → ICD codes are assigned at discharge in MIMIC-IV; flag circularity or timing issues
- **"Complete cases only (N drops from X to Y)"** → Selection bias risk; check if excluded patients differ
- **"47 significant associations"** → Multiple comparisons; apply FDR correction
- **"Small sample (n=12) but p < 0.05"** → Likely false positive; report with caution

---

## After Completion

1. Save a brief `RESULTS.md` to `output_dir` with: cohort size, key characterization findings
2. Summarize key findings in the terminal
3. Acknowledge limitations — especially ICD coverage gaps and NLP sensitivity/specificity tradeoffs
4. If downstream analysis is planned, suggest it as a new study version (`v2`) or a separate session
