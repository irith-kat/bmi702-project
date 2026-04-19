# Autonomous Phenotyping for Clinical Research Workflows
The goal of this project is to build a "phenotyping-as-a-service" MCP server that operationalizes KOMAP and LATTE as MCP tools to enable better autonomous workflows for end-to-end clinical research.

Accurate cohort definition remains a major challenge in EHR-based research studies, as diagnostic billing codes alone often fail to reflect true clinical phenotypes. While [KOMAP](https://github.com/celehs/KOMAP) (multimodal weakly supervised phenotyping) and [LATTE](https://github.com/celehs/LATTE) (longitudinal incident phenotyping) address these limitations, they currently require manual configuration and technical expertise. By exposing these algorithms as callable tools, this system enables LLM-based agents to perform multimodal and longitudinal phenotyping beyond simple code-based filtering, supporting more accurate and reproducible end-to-end clinical research workflows. This project will use the [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) dataset and build off of the data access MCP infrastructure that [M4](https://github.com/hannesill/m4/tree/main) has established, and extend the agent's capabilities from heuristic cohort selection to advanced phenotyping.

## Using non-MIMIC datasets

The preprocessing pipeline (`src/preprocessing/structured/`) is designed to work beyond MIMIC-IV. Two capabilities support this:

**Custom vocabulary mapping** — if your EHR uses hospital-specific formulary codes, local diagnosis codes, or proprietary procedure identifiers, use the `custom-vocab-mapping` agent skill to produce a drop-in mapping CSV from your crosswalk file. The skill bridges custom codes → ICD/NDC/CPT → the pipeline's standard rollup vocabularies (PheCode, RxNorm ingredient, CCS), and uses [`ndclib`](https://pypi.org/project/ndclib/) for correct NDC format normalization across 10-digit, 11-digit, and hyphenated inputs. The resulting file is passed via the `mapping_file=` parameter of `icd_to_events()`, `drug_to_events()`, or `build_obs_log()` — no code changes required.

**ICD dot handling** — most EHRs outside MIMIC-IV store ICD codes with the decimal dot already present (e.g. `E11.9`). Pass `has_dots=True` to `icd_to_events()` or `build_obs_log(icd_has_dots=True)` to skip dot insertion. The default (`None`) auto-detects the format from the data.

## Running the tests

```bash
uv run pytest               # run the full suite
uv run pytest tests/ -v     # verbose output
uv run pytest tests/unit/test_note_ner.py   # single module
```

Tests live in `tests/` and cover `m4-pheno/note_ner.py` and `m4-pheno/once.py`. MedSpaCy is mocked so no model download is required.

## Installing pre-commit hooks
```bash
pre-commit install
```
