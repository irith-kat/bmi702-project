# MedGemma Exploration

MedGemma is a medically specialised open-weights model from Google, available via Vertex AI Model Garden.  This document records what we explored and why we paused the path in favour of the Gemini API approach.

---

## What is MedGemma?

MedGemma-27B is a 27-billion-parameter instruction-tuned model pre-trained on medical literature, clinical notes, and biomedical corpora.  Unlike general-purpose Gemini models, it has explicit grounding in clinical terminology, ICD coding conventions, and discharge summary structure — making it well-suited to tasks like chart review and phenotyping.

The variant used here is `google/medgemma-27b-text-it`: text-only, instruction-tuned, no image modality.

---

## Benefits

**Medical specialisation** — MedGemma is trained on clinical data and understands the language of discharge summaries, assessment-and-plan sections, and ICD codes without needing extensive prompt engineering to orient it.  It tends to produce more reliable per-admission evidence extraction than general models at the same parameter count.

**Open weights** — The model weights can be downloaded and run locally (on sufficient hardware).  This is significant for HIPAA-sensitive data: a local deployment has no data leaving your infrastructure, unlike any hosted API.

**Local option** — You can run MedGemma on a machine with 4× 24GB GPUs (e.g. 4× A10G or 4× L4) using vLLM, with no network dependency after the initial weight download.  This makes it viable for environments where sending patient data to an external endpoint is prohibited.

---

## Downsides: infrastructure overhead

The primary friction is the Vertex AI BatchPredictionJob setup:

- Requires a GCS bucket in the same region as the job for both input and output JSONL
- Machine provisioning (g2-standard-48 + 4× L4) adds cold-start latency of 10–20 minutes before any inference runs
- The wire format is `@requestFormat: chatCompletions` (vLLM-compatible), which differs from native Gemini API format
- Jobs run asynchronously — you submit, poll, then parse; no streaming or interactive feedback
- Per-job overhead makes it inefficient for small batches (fewer than ~50 patients)
- Requires accepting MedGemma Terms of Service in Model Garden before API access is granted

Running locally avoids GCS and job submission overhead but requires significant GPU hardware and manual vLLM serving setup.

---

## Labeling pipeline overview

The pipeline in `medgemma.py` follows five steps:

1. **Silver pre-filter** (`labeler_utils.silver_prefilter`) — Uses LATTE's Equation 1 to score all patients by their key-code density.  Samples from the high-score tail (likely cases) and low-score tail (likely controls) to produce a balanced set for gold labeling.

2. **JSONL preparation** (`medgemma.build_patient_jsonl`) — For each selected patient, retrieves all discharge summaries, sorts them chronologically, formats them into a single structured prompt, and writes one JSONL line per patient using the `@requestFormat: chatCompletions` wire format expected by vLLM on Vertex AI.

3. **Batch job submission** (`medgemma.submit_medgemma_batch`) — Uploads the JSONL to GCS and submits a Vertex AI `BatchPredictionJob` targeting the MedGemma-27B model with 4× L4 GPUs.  Returns a job resource name.

4. **Result polling** (`medgemma.wait_for_batch_job`) — Polls the job state until it reaches a terminal state (`SUCCEEDED`, `FAILED`, or `CANCELLED`).

5. **Result parsing** (`medgemma.parse_medgemma_results`) — Reads output JSONL from GCS, extracts the model's JSON response per patient, and returns a DataFrame with `subject_id`, `label`, `incident_hadm_id`, `incident_charttime`, and `incident_T` ready for LATTE.

Idempotency is handled via GCS: `get_completed_subjects()` inspects the output prefix before building the JSONL, so re-runs skip already-labeled patients.

HAS NOT BEEN TESTED END TO END YET.

---

## Why we paused this path

The infrastructure setup (GCS, Vertex AI batch jobs, GPU provisioning, terms acceptance) is a significant lift relative to the Gemini API approach, which achieves equivalent labeling quality for this cohort size with a direct API call per patient and a local cache file.  MedGemma remains the right choice if data residency requirements prohibit sending notes to an external API.

See `gold_label_generation_gemini.md` for the active pipeline.
