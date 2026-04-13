"""
medgemma.py
-----------
Gold-label generation using MedGemma-27B via Vertex AI BatchPredictionJob.

Wire format: @requestFormat: chatCompletions (MedGemma / vLLM on Vertex AI).

Workflow
--------
1. build_patient_jsonl()    – write one JSONL line per patient (chatCompletions format)
2. upload_jsonl_to_gcs()    – push input JSONL to GCS
3. submit_medgemma_batch()  – launch a BatchPredictionJob
4. wait_for_batch_job()     – poll until terminal state
5. parse_medgemma_results() – read GCS output, return LATTE gold label DataFrame

GCS is the source of truth for checkpointing: get_completed_subjects() inspects
already-written output lines so a re-run safely skips finished patients.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from .labeler_utils import (
    DiseaseConfig,
    build_user_content,
    build_result_record,
    error_record,
    extract_json_from_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL preparation (chatCompletions wire format)
# ---------------------------------------------------------------------------


def build_patient_jsonl(
    notes_df: pd.DataFrame,
    subject_ids: list[str],
    output_jsonl: str,
    config: DiseaseConfig,
    completed_subjects: set[str] | None = None,
    max_chars_per_note: int | None = None,
) -> int:
    """
    Write one JSONL line per patient using the chatCompletions wire format
    expected by MedGemma on Vertex AI Model Garden / vLLM.

    Returns the number of patients written.
    """
    completed = completed_subjects or set()
    written = skipped_completed = skipped_no_notes = 0

    with open(output_jsonl, "w", encoding="utf-8") as fh:
        for sid in subject_ids:
            sid_str = str(sid)
            if sid_str in completed:
                skipped_completed += 1
                continue
            try:
                system_instruction, user_content = build_user_content(
                    sid_str, notes_df, config, max_chars_per_note
                )
            except ValueError:
                logger.warning("No notes found for subject_id=%s — skipping.", sid_str)
                skipped_no_notes += 1
                continue

            line: dict[str, Any] = {
                "@requestFormat": "chatCompletions",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 4096,
                "temperature": 0,
                "_subject_id": sid_str,
            }
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1

    logger.info(
        "Wrote %d patients to %s  (skipped: %d completed, %d no-notes).",
        written,
        output_jsonl,
        skipped_completed,
        skipped_no_notes,
    )
    return written


# ---------------------------------------------------------------------------
# GCS checkpointing helpers
# ---------------------------------------------------------------------------


def get_completed_subjects(output_gcs_prefix: str) -> set[str]:
    """
    Inspect GCS output JSONL files and return subject_ids that already have results.
    Returns an empty set if the prefix does not exist or GCS is unavailable.
    """
    try:
        from google.cloud import storage as gcs
    except ImportError:
        logger.warning(
            "google-cloud-storage not installed; cannot check GCS completions."
        )
        return set()

    prefix_stripped = output_gcs_prefix.removeprefix("gs://")
    bucket_name, _, blob_prefix = prefix_stripped.partition("/")

    try:
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=blob_prefix))
    except Exception as exc:
        logger.warning("Could not list GCS blobs at %s: %s", output_gcs_prefix, exc)
        return set()

    completed_set: set[str] = set()
    for blob in blobs:
        if not blob.name.endswith(".jsonl"):
            continue
        try:
            content = blob.download_as_text(encoding="utf-8")
            for raw_line in content.splitlines():
                if not raw_line.strip():
                    continue
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                sid = obj.get("_subject_id") or obj.get("subject_id")
                if not sid:
                    pred = obj.get("prediction") or obj.get("response") or {}
                    if isinstance(pred, dict):
                        sid = pred.get("subject_id")
                    elif isinstance(pred, str):
                        try:
                            sid = json.loads(pred).get("subject_id")
                        except json.JSONDecodeError:
                            pass
                if sid:
                    completed_set.add(str(sid))
        except Exception as exc:
            logger.warning("Could not read blob %s: %s", blob.name, exc)

    logger.info(
        "Found %d already-completed subjects in %s.",
        len(completed_set),
        output_gcs_prefix,
    )
    return completed_set


def upload_jsonl_to_gcs(local_path: str, gcs_path: str) -> None:
    """Upload a local JSONL file to GCS."""
    from google.cloud import storage as gcs

    prefix_stripped = gcs_path.removeprefix("gs://")
    bucket_name, _, blob_name = prefix_stripped.partition("/")

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    logger.info("Uploaded %s → %s", local_path, gcs_path)


# ---------------------------------------------------------------------------
# Batch job submission
# ---------------------------------------------------------------------------


def submit_medgemma_batch(
    project_id: str,
    location: str,
    input_gcs_path: str,
    output_gcs_prefix: str,
    job_display_name: str = "medgemma_gold_label_job",
    model_name: str = "google/medgemma-27b-text-it",
    machine_type: str = "g2-standard-48",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 4,
) -> str:
    """
    Submit a Vertex AI BatchPredictionJob for MedGemma-27B.

    Returns the resource_name of the created job.
    """
    try:
        from google.cloud import aiplatform
    except ImportError as exc:
        raise ImportError(
            "google-cloud-aiplatform is required. "
            "Install with: pip install google-cloud-aiplatform"
        ) from exc

    aiplatform.init(project=project_id, location=location, api_transport="rest")
    logger.info(
        "Submitting BatchPredictionJob: model=%s, input=%s, output=%s",
        model_name,
        input_gcs_path,
        output_gcs_prefix,
    )
    batch_job = aiplatform.BatchPredictionJob.create(
        job_display_name=job_display_name,
        model_name=model_name,
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source=[input_gcs_path],
        gcs_destination_prefix=output_gcs_prefix,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
    )
    logger.info(
        "BatchPredictionJob submitted: %s  (state=%s)",
        batch_job.resource_name,
        batch_job.state,
    )
    return batch_job.resource_name


def wait_for_batch_job(resource_name: str, poll_interval_seconds: int = 60) -> str:
    """
    Block until a BatchPredictionJob reaches a terminal state.

    Returns the final state string, e.g. "JOB_STATE_SUCCEEDED".
    """
    import time
    from google.cloud import aiplatform

    batch_job = aiplatform.BatchPredictionJob(resource_name)
    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
    while True:
        state = batch_job.state.name
        logger.info("BatchPredictionJob %s state: %s", resource_name, state)
        if state in terminal_states:
            return state
        time.sleep(poll_interval_seconds)
        batch_job._sync_gca_resource()


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _extract_response_text(outer: dict) -> str | None:
    """Extract model text from various Vertex AI output envelopes."""
    pred = outer.get("prediction")
    if isinstance(pred, dict):
        choices = pred.get("choices") or pred.get("predictions", {}).get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content")
        if "content" in pred:
            return pred["content"]
    choices = outer.get("choices")
    if choices:
        return choices[0].get("message", {}).get("content")
    if isinstance(pred, str):
        return pred
    candidates = outer.get("candidates") or (
        outer.get("prediction", {}).get("candidates")
        if isinstance(outer.get("prediction"), dict)
        else None
    )
    if candidates:
        try:
            return candidates[0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            pass
    return None


def parse_medgemma_results(
    output_gcs_prefix: str,
    baseline_date: str | None = None,
    month_window: int = 3,
) -> pd.DataFrame:
    """
    Read GCS BatchPredictionJob output and return a LATTE gold label DataFrame.

    Columns: subject_id, label, incident_hadm_id, incident_charttime,
             incident_T, timeline_json, parse_error
    """
    try:
        from google.cloud import storage as gcs
    except ImportError as exc:
        raise ImportError("google-cloud-storage required for result parsing.") from exc

    prefix_stripped = output_gcs_prefix.removeprefix("gs://")
    bucket_name, _, blob_prefix = prefix_stripped.partition("/")

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=blob_prefix))

    records: list[dict] = []

    for blob in blobs:
        if not blob.name.endswith(".jsonl"):
            continue
        content = blob.download_as_text(encoding="utf-8")
        for raw_line in content.splitlines():
            if not raw_line.strip():
                continue
            try:
                outer = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                logger.warning("Could not parse output line: %s", exc)
                records.append(error_record())
                continue

            subject_id = str(outer.get("_subject_id", ""))
            response_text = _extract_response_text(outer)

            if response_text is None:
                logger.warning(
                    "Could not find response text for subject_id=%s", subject_id
                )
                records.append(error_record(subject_id))
                continue

            parsed = extract_json_from_response(response_text)
            if parsed is None:
                logger.warning(
                    "Could not parse model JSON for subject_id=%s.  Raw: %.200s",
                    subject_id,
                    response_text,
                )
                records.append(error_record(subject_id))
                continue

            records.append(
                build_result_record(parsed, subject_id, baseline_date, month_window)
            )

    result_df = pd.DataFrame(records)
    if result_df.empty:
        logger.warning("No results parsed from %s.", output_gcs_prefix)
        return result_df

    n_parsed = (~result_df["parse_error"]).sum()
    n_errors = result_df["parse_error"].sum()
    logger.info(
        "Parsed %d/%d results successfully (%d errors).  "
        "Gold labels: %d cases, %d controls.",
        n_parsed,
        len(result_df),
        n_errors,
        (result_df["label"] == 1).sum(),
        (result_df["label"] == 0).sum(),
    )
    return result_df
