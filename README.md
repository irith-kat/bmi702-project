# Autonomous Phenotyping for Agentic EHR-Based Clinical Research Workflows

An autonomous, LLM-driven phenotyping framework for end-to-end construction of clinical cohorts from EHRs. The system integrates structured data harmonization, clinical NLP, and probabilistic phenotyping ([MAP](https://celehs.github.io/MAP/) and [LATTE](https://github.com/celehs/LATTE)) within an agentic workflow that converts natural language research queries into state-of-the-art cohort definitions.

From a single prompt, the agent conducts a structured interview with the researcher, drafts a research protocol that can be iteratively refined, then autonomously builds features, runs phenotyping, and outputs a results package with as few as two human interventions occurring in the first 10 minutes. The full workflow typically runs in about 2–4 hours, depending on the nature of the request.

The framework builds on [M4](https://github.com/hannesill/m4) for natural-language SQL access to EHR databases, and extends it with a multimodal preprocessing and phenotyping layer orchestrated through Claude Code skills. It is designed to be modular and adaptable to any SQL-queryable EHR dataset, not just MIMIC-IV.

## Getting Started

### 1. Install dependencies

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Install uv if you don't already have it, then run the following to install all project dependencies:

```bash
uv sync
```

The pipeline also requires R with the MAP package installed. Install [R](https://www.r-project.org/), then run the following to install the MAP package:

```r
install.packages("remotes")
remotes::install_github("celehs/MAP")
```

### 2. Unzip the mapping dictionaries

Unzip the vocabulary mapping files in `mapping_dicts/`:

```bash
unzip mapping_dicts/mapping_dicts_csvs.zip -d mapping_dicts/
```

The Athena NDC/RxNorm file can be regenerated with:

```bash
uv run python mapping_dicts/build_ndc_to_rxnorm.py
```

### 3. Set up M4 for database access

This pipeline delegates all SQL query generation and EHR database access to [M4](https://github.com/hannesill/m4). Follow the M4 setup instructions to configure a connection to your MIMIC-IV instance (local CSV files or BigQuery).

### 4. Set up GCP for the LLM gold labeling pipeline

The LATTE phenotyping pipeline uses an LLM (Gemini via GCP) to generate gold-standard labels for LATTE. Once you have a Google Cloud project with the Vertex AI API enabled, run the following to authenticate:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Default model: `gemini-2.5-flash`

### 5. Set up ONCE feature dictionaries

This pipeline uses [ONCE](https://shiny.parse-health.org/ONCE/) to select the codified concepts and NLP CUIs most relevant to your disease of interest.

1. Go to [https://shiny.parse-health.org/ONCE/](https://shiny.parse-health.org/ONCE/)
2. Enter the disease name
3. Download the two output CSV files for codified features and NLP CUI features
4. Place both files in `input/`

It is recommended to briefly review and prune the downloaded dictionaries for your specific clinical question before starting the pipeline.



## Usage with Claude Code

The `.claude/skills/` directory contains all agent skills. In Claude Code, start a session with our orchestrator skill:

```
/clinical-research-session Generate a cohort...
```

The agent will first interview you about your research question. Then it will generate a fully reproducible study protocol that you can conversationally refine. Once approved, it will autonomously invoke the other skills (`mimic-preprocessing`, `map-phenotyping`, `latte-phenotyping`, etc.) as needed to define your desired cohort. All generated scripts, protocols, and results are written to `output/<study-name>/`.

## Miscellaneous

### Test suite

```bash
uv run pytest
```

### Pre-commit hooks

```bash
pre-commit install
```
