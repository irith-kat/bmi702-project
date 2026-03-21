# Results: MS Cohort v1

**Date completed:** 2026-03-21
**Environment:** WSL2 (Linux 6.6.87.2), 18 CPU cores, 15 GB RAM, Python 3.12
**Dataset:** MIMIC-IV full (DuckDB local backend) + MIMIC-IV-Note (BigQuery backend)

---

## Execution Times

| Step | Script | Wall time | Notes |
|---|---|---|---|
| 01 — EHR pull + obs_log | `01_cohort_definition.py` | ~30 min | DuckDB local; diagnoses + Rx + HCPCS; ICD→PheCode, NDC→RxNorm, CPT→CCS rollups |
| 02a — BigQuery note fetch | (inside `02_nlp_features.py`) | ~8 min | 66 batches × 400 patients; 63,757 notes from `mimiciv_note.discharge` |
| 02b — MedSpaCy NER | (inside `02_nlp_features.py`) | **61.5 min** | 18 parallel workers (fork), ~2,016 notes/worker; all chunks completed simultaneously |
| 03 — MAP phenotyping | `03_map_phenotyping.py` | ~6 min | `preprocess_map` + R MAP algorithm via subprocess |
| 04 — Characterization | `04_characterization.py` | ~1 min | Plots + summary table |
| **Total** | | **~107 min** | Dominated by MedSpaCy NLP (~58% of runtime) |

### NLP throughput details
- Notes fetched from BigQuery: 63,757 (18,924 patients)
- Notes processed by MedSpaCy (after `notes_per_patient=3` filter): ~36,300 (3 most recent per patient)
- Workers: 18 (one per CPU core, Python `multiprocessing.Pool`, `fork` context)
- Each worker loaded its own MedSpaCy model independently (required to avoid spaCy's TargetRule msgpack serialization crash when using `n_process > 1`)
- Aggregate throughput: ~9.8 notes/sec (~0.55 notes/sec/worker); `max_note_chars=10_000`
- tqdm wall time: `18/18 [1:01:30<00:00, 205.01s/chunk]` — all workers finished within the same tqdm tick

### Key engineering notes for teammates
- **Backend switch required between scripts 01 and 02:** structured EHR uses DuckDB (local), notes use BigQuery. Forgetting to call `set_active_backend("bigquery")` before note queries silently fails.
- **BigQuery batch size = 400 subject IDs per query** to stay within the M4 10k-token IN-clause limit. 26,002 candidates → 66 batches.
- **Import order matters:** `from preprocessing import ...` must come immediately after `sys.path.insert()` calls, before any other imports (`m4`, `pandas`). Otherwise Python caches `src/preprocessing/` as an empty namespace package and the import fails.
- **`min_nonzero` raised from 20→100:** 9 ONCE features had only 25–78 patients with non-zero counts in the 26,002-patient candidate pool, causing the flexmix EM algorithm to produce NaN log-likelihood. Raising the threshold dropped these features; 27 of 36 were retained. See feature nonzero counts below.
- **Notes cache:** `notes_df.parquet` is cached; script 02 skips BigQuery re-fetch if the file exists, saving ~8 min on reruns.

---

## CONSORT Flow

| Step | N |
|---|---|
| MIMIC-IV adults (age ≥18) | 364,627 |
| Candidates (≥1 ONCE codified event) | 26,002 |
| With discharge notes (BigQuery) | 18,924 |
| With ≥1 NLP CUI mention | 13,457 |
| ICD anchor PheCode:335 present | 1,362 |
| **MAP cases (phenotype=1)** | **605** |
| MAP controls (phenotype=0) | 25,397 |

MAP accepted 605 of 1,362 ICD-coded patients (~44%) as high-confidence MS cases after co-feature validation.

---

## MAP Model Details

- **Anchor:** PheCode:335 (Multiple Sclerosis); nonzero in 1,362 / 26,002 patients (5.2%)
- **Features retained (27):** 13 ONCE NLP CUIs + 13 PheCodes + 1 anchor
- **`min_nonzero` = 100** (raised from default 20; see engineering notes above)
- **MAP score range:** 0.0000 – 0.6846
- **Threshold:** MAP default (posterior ≥ 0.5 → case)

### Feature nonzero counts (all 36 candidates, pre-filter)

| Feature | Nonzero patients | Kept? |
|---|---|---|
| PheCode:295 (Psychosis) | 25 | Dropped |
| PheCode:275.2 | 36 | Dropped |
| PheCode:377.1 | 36 | Dropped |
| CUI:C0423551 | 42 | Dropped |
| PheCode:711.3 | 44 | Dropped |
| PheCode:704.2 | 57 | Dropped |
| PheCode:368.1 | 65 | Dropped |
| CUI:C1529600 | 77 | Dropped |
| PheCode:199.4 | 78 | Dropped |
| CUI:C0011304 | 131 | Kept |
| PheCode:709.7 | 133 | Kept |
| CUI:C0528175 | 155 | Kept |
| CUI:C0066677 | 156 | Kept |
| CUI:C0029134 | 168 | Kept |
| PheCode:341 (Other CNS disorders) | 196 | Kept |
| PheCode:798.1 | 221 | Kept |
| PheCode:728.7 | 223 | Kept |
| PheCode:303.3 | 347 | Kept |
| PheCode:377.3 | 349 | Kept |
| CUI:C0035020 | 368 | Kept |
| CUI:C0231170 | 442 | Kept |
| PheCode:709.2 | 541 | Kept |
| CUI:C0011581 | 550 | Kept |
| PheCode:303.4 | 595 | Kept |
| CUI:C0344315 | 649 | Kept |
| PheCode:301 | 763 | Kept |
| CUI:C0026769 (Multiple sclerosis) | 804 | Kept |
| PheCode:335 (MS anchor) | 1,362 | Kept (always) |
| PheCode:596.5 | 2,024 | Kept |
| PheCode:295.3 | 3,097 | Kept |
| CUI:C0004268 | 3,336 | Kept |
| PheCode:295.1 | 4,023 | Kept |
| CUI:C3714552 | 5,249 | Kept |
| PheCode:300.9 | 5,711 | Kept |
| CUI:C1457887 | 10,270 | Kept |
| PheCode:290.1 | 10,410 | Kept |

---

## Characterization Summary

| Metric | Cases (n=605) | Controls (n=25,397) |
|---|---|---|
| Age median | 52 | 65 |
| Age IQR | 39–63 | 43–81 |
| Age mean ± SD | 51.3 ± 15.2 | 61.4 ± 22.2 |
| % Female | 74.7% | 57.1% |
| % White | 75.5% | 67.4% |
| Mean LOS (days) | 4.6 | 5.5 |
| Mean admissions | 4.3 | 4.1 |

### Top 5 Comorbidities (Cases)

1. Tobacco use disorder — 42.5%
2. Essential hypertension — 38.2%
3. Urinary tract infection — 35.2%
4. Anxiety disorder — 29.1%
5. Hyperlipidemia — 28.6%

### NLP CUI extraction

- Unique CUIs found: 21 of 34 target CUIs
- NLP observation rows: 54,400 (13,457 patients with ≥1 CUI)
- obs_log event breakdown: rxnorm 15,649,801 · phecode 5,943,457 · ccs 106,853 · cui 54,400

---

## Key Findings

- **Female predominance (74.7%)** is consistent with known MS epidemiology (~3:1 F:M ratio) — strong internal validity signal.
- **Younger age at first admission** (median 52 vs 65 for controls) is consistent with MS onset in young adulthood.
- **UTI as a top comorbidity** (35%) reflects neurogenic bladder, a common MS complication.
- **Anxiety disorder prevalence** (29%) reflects known psychiatric comorbidity in MS.

## Limitations

- MAP score range (0–0.68) is compressed relative to typical MAP runs; with only 5.2% anchor prevalence in the candidate pool, the posterior is diluted by the large control mass.
- NLP applied to 3 most recent discharge notes per patient only; earlier notes may contain relevant CUI mentions.
- 9 ONCE features were too sparse for MAP EM convergence and were excluded; raising `min_nonzero` is a data-driven decision with potential signal loss.
- MIMIC-IV ICD codes are assigned at discharge — appropriate for phenotyping but creates potential temporal leakage for admission-time features.

---

## Output Files

| File | Size | Description |
|---|---|---|
| `data/obs_log.parquet` | 106 MB | 21.7M rows, structured EHR events (PheCode, RxNorm, CCS) |
| `data/notes_df.parquet` | 376 MB | 63,757 discharge notes for 18,924 patients |
| `data/obs_log_with_nlp.parquet` | 106 MB | 21.75M rows, obs_log + 54,400 NLP CUI events |
| `data/admissions_df.parquet` | 14 MB | Admission-level table with LOS, race, admit year |
| `data/patients_df.parquet` | 2.8 MB | Patient demographics |
| `data/cohort.parquet` | 364 KB | 26,002 patients with MAP score + phenotype + demographics |
| `data/characterization_summary.parquet` | 2.4 KB | Wide summary table (cases vs controls) |
| `plots/map_score_distribution.json` | MAP posterior histogram | |
| `plots/consort_flow.json` | CONSORT waterfall chart | |
| `plots/age_distribution.json` | Age histograms, cases vs controls | |
| `plots/sex_distribution.json` | Sex breakdown | |
| `plots/race_distribution.json` | Race/ethnicity (top 8) | |
| `plots/top_comorbidities.json` | Top 15 comorbidities, cases vs controls | |
| `plots/once_feature_prevalence.json` | Top 20 ONCE features, cases vs controls | |
| `plots/admission_trend.json` | Admission year trend | |
| `plots/los_distribution.json` | Mean LOS distribution | |
