# Gold Label Generation — Gemini API Pipeline

This is the active pipeline for generating gold-standard phenotype labels from discharge summaries using an LLM-as-clinician approach.

---

## Concept

The model is prompted to act as a senior physician conducting a structured chart review.  For each patient, it receives their complete admission history (all discharge summaries in chronological order) and is asked to:

1. Work through each admission sequentially, extracting relevant clinical evidence
2. Assign a per-admission status (disease present / absent) with a confidence rating
3. Identify the **incident visit** — the first admission where the disease is newly documented
4. Output a structured JSON object with a binary label, per-admission timeline, and reasoning

This produces richer signal than ICD-code lookup: the model can distinguish prevalent from incident disease, handle ambiguous language, and reason across multiple admissions.

---

## Input

| Field | Description |
|-------|-------------|
| `notes_df` | DataFrame with columns `subject_id`, `hadm_id`, `charttime`, `text` — one row per discharge note |
| `subject_ids` | List of patient IDs to label |
| `config` | `DiseaseConfig` object with disease name, ICD codes, diagnostic criteria, and incident definition |
| `cache_jsonl` | Path to the local JSONL cache file (created if absent) |

`notes_df` can come from `parse_discharge_summaries()` (local text file) or directly from a BigQuery query against `mimiciv_note.discharge`.

The `DiseaseConfig` controls everything disease-specific in the prompt.  `HF_DISEASE_CONFIG` is provided as a ready-to-use Heart Failure config.  For other diseases, instantiate `DiseaseConfig` with appropriate `diagnostic_criteria`, `incident_definition`, and `key_codes` (the latter used only for silver pre-filtering, never shown to the model).

---

## Output

`parse_gemini_results()` returns a DataFrame with one row per patient:

| Column | Description |
|--------|-------------|
| `subject_id` | Patient identifier |
| `label` | `1` = disease present, `0` = never present, `-1` = parse error |
| `incident_hadm_id` | `hadm_id` of the first admission where disease was present |
| `incident_charttime` | Charttime of the incident admission |
| `incident_T` | LATTE time-window index from baseline (if `baseline_date` provided) |
| `timeline_json` | Raw JSON array of per-admission model judgments |
| `parse_error` | `True` if the model response could not be parsed |

To convert to LATTE's per-visit `(subject_id, T, Y)` format, pass the results through `labels_to_latte()`.

---

## Idempotency

Every successful API response is appended to the local cache JSONL immediately.  On re-run, `run_gemini_labeling()` reads the cache first and skips any `subject_id` already present.  This means:

- Interrupted runs resume from where they stopped — no duplicate API calls
- The cache is the source of truth; `parse_gemini_results()` reads only the cache
- To re-label a patient, delete their line from the cache file

---

## Model and region

Default model: `publishers/google/models/gemini-3.1-flash-lite-preview`
Default location: `global`

The 3.1 preview model is not yet available in `us-central1` — use `global`, which routes to whichever region has it deployed.  Stable models like `gemini-2.5-flash-lite` work in `us-central1` if you prefer a pinned region.

---

## Running the integration test

The test sends two synthetic patients (one clear HF case, one control) through the full pipeline and asserts both labels are correct.

```bash
uv run python tests/integration/test_gemini_labeler.py
```

The test caches results to `/tmp/gemini_test_cache.jsonl`.  Re-runs are free — it skips patients already in the cache.  To force a fresh API call:

```bash
rm /tmp/gemini_test_cache.jsonl && uv run python tests/integration/test_gemini_labeler.py
```

---

## Troubleshooting

**404 on `generateContent` despite model appearing in the catalog**
The model is region-restricted.  Try `location="global"` — this is the required location for `gemini-3.1-flash-lite-preview`.

**Both labels come back as `parse_error=True`**
The model returned a non-JSON response.  Check the raw cache file:
```bash
cat /tmp/gemini_test_cache.jsonl
```
If responses are empty, check that your ADC credentials are valid (`gcloud auth application-default login`).

**`subject_id` in results shows `10001919` instead of real patient IDs**
This is a stale cache issue from before a bug fix — the model was echoing the schema example ID.  Delete the cache and re-run.

**`ModuleNotFoundError: google.genai`**
```bash
uv add google-cloud-aiplatform
```

**Slow / rate-limited**
Add `retry_delay_seconds=10` to `run_gemini_labeling()`.  The flash-lite model has generous quota but very long prompts (patients with many admissions) can occasionally hit per-minute token limits.
