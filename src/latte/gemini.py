"""
gemini.py
---------
Gold-label generation using the Gemini API via Vertex AI (native Gemini format).

Unlike the MedGemma path (batch job + GCS), this module makes individual API calls
and caches every response to a local JSONL file.  Re-running is safe: subjects
already present in the cache are skipped.

Workflow
--------
1. run_gemini_labeling()  – iterate subject_ids, skip cached, call API, append to cache
2. parse_gemini_results() – read the cache JSONL and return a LATTE gold label DataFrame

Default model: publishers/google/models/gemini-3.1-flash-lite-preview
"""

from __future__ import annotations

import json
import logging
import time

import pandas as pd

from .labeler_utils import (
    DiseaseConfig,
    HF_DISEASE_CONFIG,
    build_user_content,
    build_result_record,
    error_record,
    extract_json_from_response,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "publishers/google/models/gemini-3.1-flash-lite-preview"
DEFAULT_LOCATION = "global"  # 3.1 preview not yet available in us-central1


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def get_cached_subject_ids(cache_jsonl: str) -> set[str]:
    """Return the set of subject_ids already present in a labeling cache file."""
    return _load_cache(cache_jsonl)


def _load_cache(cache_jsonl: str) -> set[str]:
    """Return the set of subject_ids already present in the cache file."""
    completed: set[str] = set()
    try:
        with open(cache_jsonl, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sid = obj.get("subject_id")
                    if sid:
                        completed.add(str(sid))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return completed


def _append_cache(cache_jsonl: str, record: dict) -> None:
    """Append a single result record to the cache file."""
    with open(cache_jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Main labeling loop
# ---------------------------------------------------------------------------


def run_gemini_labeling(
    notes_df: pd.DataFrame,
    subject_ids: list[str],
    cache_jsonl: str,
    config: DiseaseConfig = HF_DISEASE_CONFIG,
    model_name: str = DEFAULT_MODEL,
    project_id: str | None = None,
    location: str = DEFAULT_LOCATION,
    max_chars_per_note: int | None = None,
    max_notes_per_patient: int | None = None,
    retry_delay_seconds: float = 5.0,
    max_retries: int = 3,
    record_builder=None,
    system_instruction_builder=None,
) -> int:
    """
    Label patients using the Gemini API, caching results to a local JSONL file.

    Parameters
    ----------
    notes_df : pd.DataFrame
        Output of parse_discharge_summaries().
    subject_ids : list[str]
        Patients to label.
    cache_jsonl : str
        Local file path for the result cache.  Appended to on each successful call.
        Subjects already in the cache are skipped (idempotent).
    config : DiseaseConfig
        Disease-specific prompt parameters.
    model_name : str
        Vertex AI model resource name.
    project_id : str | None
        GCP project ID.  If None, uses the ADC project.
    location : str
        Vertex AI region.
    max_chars_per_note : int | None
        Truncate each note to this many characters before building the prompt.
    retry_delay_seconds : float
        Seconds to wait between retries on transient API errors.
    max_retries : int
        Maximum number of retries per patient before writing an error record.

    Returns
    -------
    int
        Number of patients newly labeled (cache misses processed).
    """
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise ImportError(
            "google-genai is required. Install with: pip install google-genai"
        ) from exc

    client = genai.Client(vertexai=True, project=project_id, location=location)

    completed = _load_cache(cache_jsonl)
    logger.info(
        "Cache has %d completed subjects.  %d subjects requested.",
        len(completed),
        len(subject_ids),
    )

    from .labeler_utils import build_system_instruction

    _si_builder = system_instruction_builder or build_system_instruction
    _rec_builder = record_builder or build_result_record

    system_instruction = _si_builder(config)
    gen_config = genai_types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0,
        max_output_tokens=16384,  # chain-of-thought for ~60 admissions needs ~10-12K tokens
    )

    newly_labeled = 0
    skipped_cached = 0
    skipped_no_notes = 0

    for sid in subject_ids:
        sid_str = str(sid)
        if sid_str in completed:
            skipped_cached += 1
            continue

        try:
            _, user_content = build_user_content(
                sid_str, notes_df, config, max_chars_per_note, max_notes_per_patient
            )
        except ValueError:
            logger.warning("No notes found for subject_id=%s — skipping.", sid_str)
            skipped_no_notes += 1
            continue

        response_text = _call_with_retry(
            client,
            model_name,
            gen_config,
            user_content,
            sid_str,
            retry_delay_seconds,
            max_retries,
        )

        if response_text is None:
            logger.error(
                "All retries failed for subject_id=%s — writing error record.", sid_str
            )
            _append_cache(cache_jsonl, {**error_record(sid_str), "subject_id": sid_str})
            newly_labeled += 1
            continue

        parsed = extract_json_from_response(response_text)
        if parsed is None:
            logger.warning(
                "Could not parse model JSON for subject_id=%s.  Raw: %.200s",
                sid_str,
                response_text,
            )
            _append_cache(cache_jsonl, {**error_record(sid_str), "subject_id": sid_str})
        else:
            record = _rec_builder(parsed, sid_str)
            _append_cache(cache_jsonl, record)

        newly_labeled += 1

    logger.info(
        "Gemini labeling complete: %d newly labeled, %d cached, %d no-notes.",
        newly_labeled,
        skipped_cached,
        skipped_no_notes,
    )
    return newly_labeled


def _call_with_retry(
    client,
    model_name: str,
    gen_config,
    user_content: str,
    sid_str: str,
    retry_delay: float,
    max_retries: int,
) -> str | None:
    """Call the Gemini model via google-genai, retrying on transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_content,
                config=gen_config,
            )
            text = response.text  # raises ValueError if finish_reason != STOP
            if not text:
                raise ValueError("Empty response text")
            return text
        except Exception as exc:
            logger.warning(
                "Gemini API error for subject_id=%s (attempt %d/%d): %s",
                sid_str,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(retry_delay)

    return None


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_gemini_results(
    cache_jsonl: str,
    baseline_date: str | None = None,
    month_window: int = 3,
) -> pd.DataFrame:
    """
    Read the local cache JSONL and return a LATTE gold label DataFrame.

    Columns: subject_id, label, incident_hadm_id, incident_charttime,
             incident_T, timeline_json, parse_error
    """
    records: list[dict] = []
    try:
        with open(cache_jsonl, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Could not parse cache line: %s", exc)
                    records.append(error_record())
                    continue

                # Cache lines are already in result-record format from run_gemini_labeling.
                # Re-compute incident_T if a baseline_date is supplied and wasn't at label time.
                if (
                    baseline_date
                    and obj.get("incident_charttime")
                    and obj.get("incident_T") is None
                ):
                    import numpy as np

                    try:
                        base_dt = pd.to_datetime(baseline_date)
                        ct = pd.to_datetime(obj["incident_charttime"])
                        obj["incident_T"] = int(
                            np.floor((ct - base_dt).days / (30.44 * month_window))
                        )
                    except Exception:
                        pass

                # Re-hydrate types that were serialised to string by json.dumps
                if obj.get("incident_charttime"):
                    obj["incident_charttime"] = pd.to_datetime(
                        obj["incident_charttime"], errors="coerce"
                    )

                records.append(obj)
    except FileNotFoundError:
        logger.warning("Cache file not found: %s", cache_jsonl)

    result_df = pd.DataFrame(records)
    if result_df.empty:
        logger.warning("No results in cache %s.", cache_jsonl)
        return result_df

    n_parsed = (~result_df["parse_error"]).sum()
    n_errors = result_df["parse_error"].sum()
    logger.info(
        "Parsed %d/%d results from cache (%d errors).  "
        "Gold labels: %d cases, %d controls.",
        n_parsed,
        len(result_df),
        n_errors,
        (result_df["label"] == 1).sum(),
        (result_df["label"] == 0).sum(),
    )
    return result_df


def parse_gemini_recurring_results(
    cache_jsonl: str,
    baseline_date: str,
    month_window: int = 3,
) -> pd.DataFrame:
    """
    Read a recurring-event cache JSONL and return a DataFrame ready for
    recurring_labels_to_latte().

    Columns: subject_id, label, event_Ts (list[int]), parse_error.
    event_Ts contains the integer time-window indices at which events occurred,
    computed from baseline_date and month_window.
    """
    import numpy as np

    base_dt = pd.to_datetime(baseline_date)
    records: list[dict] = []

    try:
        with open(cache_jsonl, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Could not parse cache line: %s", exc)
                    records.append(
                        {
                            "subject_id": "",
                            "label": -1,
                            "event_Ts": [],
                            "parse_error": True,
                        }
                    )
                    continue

                if obj.get("parse_error"):
                    records.append(
                        {
                            "subject_id": obj.get("subject_id", ""),
                            "label": -1,
                            "event_Ts": [],
                            "parse_error": True,
                        }
                    )
                    continue

                event_cts = obj.get("event_charttimes") or []
                event_Ts: list[int] = []
                for ct_str in event_cts:
                    try:
                        ct = pd.to_datetime(ct_str)
                        t = int(np.floor((ct - base_dt).days / (30.44 * month_window)))
                        event_Ts.append(t)
                    except Exception:
                        pass

                records.append(
                    {
                        "subject_id": str(obj.get("subject_id", "")),
                        "label": int(obj.get("label", -1)),
                        "event_Ts": event_Ts,
                        "parse_error": False,
                    }
                )
    except FileNotFoundError:
        logger.warning("Cache file not found: %s", cache_jsonl)

    result_df = pd.DataFrame(records)
    if result_df.empty:
        logger.warning("No results in recurring-event cache %s.", cache_jsonl)
        return result_df

    n_parsed = (~result_df["parse_error"]).sum()
    n_events = result_df["event_Ts"].apply(len).sum()
    logger.info(
        "Recurring-event results: %d/%d patients parsed, %d total events identified.",
        n_parsed,
        len(result_df),
        n_events,
    )
    return result_df
