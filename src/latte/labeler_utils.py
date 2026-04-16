"""
labeler_utils.py
----------------
Shared utilities for gold-label generation: disease configuration, prompt building,
discharge summary parsing, silver pre-filtering, and JSON response parsing.

Used by both medgemma.py (Vertex AI BatchPredictionJob) and gemini.py (direct API).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disease configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class DiseaseConfig:
    """
    All disease-specific parameters needed to build the labeler prompt.

    Attributes
    ----------
    name : str
        Human-readable disease name used in the prompt, e.g. "Heart Failure (HF)".
    icd_codes : str
        ICD-10 / ICD-9 codes shown to the model as context.
        Use standard ICD codes only — PheCode identifiers are internal constructs
        that do not appear in clinical notes and should NOT be included here.
    diagnostic_criteria : str
        Bulleted clinical criteria the model should look for.
    incident_definition : str
        Definition of what counts as the INCIDENT visit.
    key_codes : list[str]
        Prefixed feature codes for silver label pre-filtering ONLY.
        Never shown to the LLM.  E.g. ["PheCode:428", "RXNORM:4603"].
    silver_alpha : float
        Penalisation weight for high health-utilisation in the silver label formula.
    silver_tau : float
        Temperature for the silver label sigmoid.
    output_schema_example : str
        A JSON example shown in the prompt so the model knows the exact output shape.
    """

    name: str
    icd_codes: str
    diagnostic_criteria: str
    incident_definition: str
    key_codes: list[str]
    silver_alpha: float = 0.5
    silver_tau: float = 1.0
    output_schema_example: str = field(default="")

    def __post_init__(self):
        if not self.output_schema_example:
            self.output_schema_example = _DEFAULT_SCHEMA_EXAMPLE


_DEFAULT_SCHEMA_EXAMPLE = """{
  "subject_id": "10001919",
  "reasoning": "Admission 1 (2124-04-21): ...[per-visit analysis]...",
  "timeline": [
    {
      "visit_id": 1,
      "hadm_id": "29897682",
      "charttime": "2124-04-21",
      "status": 0,
      "confidence": "high",
      "evidence": "No HF-related findings. Chief complaint is gastric cancer."
    }
  ],
  "incident_visit_id": null,
  "label": 0
}"""

_RECURRING_SCHEMA_EXAMPLE = """{
  "subject_id": "10001919",
  "reasoning": "Admission 1 (2124-04-21): Stable follow-up. No event signs.\\nAdmission 2 (2124-07-15): Active event: BNP 1200 pg/mL, IV furosemide initiated.\\nAdmission 3 (2124-11-03): Elective procedure. No event documented.",
  "timeline": [
    {
      "visit_id": 1,
      "hadm_id": "29897682",
      "charttime": "2124-04-21",
      "status": 0,
      "confidence": "high",
      "evidence": "Stable maintenance visit. No active event criteria met."
    },
    {
      "visit_id": 2,
      "hadm_id": "38871234",
      "charttime": "2124-07-15",
      "status": 1,
      "confidence": "high",
      "evidence": "Active event: BNP 1200 pg/mL, bilateral crackles, IV furosemide started."
    },
    {
      "visit_id": 3,
      "hadm_id": "41023456",
      "charttime": "2124-11-03",
      "status": 0,
      "confidence": "high",
      "evidence": "Elective procedure. No event criteria met."
    }
  ],
  "event_visit_ids": [2],
  "label": 1
}"""


# ---------------------------------------------------------------------------
# Default Heart Failure configuration
# ---------------------------------------------------------------------------

HF_DISEASE_CONFIG = DiseaseConfig(
    name="Heart Failure (HF)",
    icd_codes="ICD-10: I50.x  |  ICD-9: 428.x",
    diagnostic_criteria="""
- Ejection fraction < 40 % (systolic HF) or documented diastolic dysfunction
- Explicit diagnosis: "acute decompensated heart failure", "congestive heart failure",
  "systolic heart failure", "diastolic heart failure", or "HFrEF" / "HFpEF"
- New initiation of loop diuretics for volume overload (furosemide / Lasix,
  bumetanide, torsemide) — not pre-existing chronic use
- BNP > 400 pg/mL or NT-proBNP > 900 pg/mL with compatible clinical picture
- Pulmonary oedema or bilateral pleural effusions attributed to fluid overload
  on imaging or clinical assessment
- Signs/symptoms: orthopnoea, PND, bilateral pitting oedema attributed to HF,
  S3 gallop, jugular venous distension""",
    incident_definition=(
        "The INCIDENT visit is the FIRST hospital admission at which Heart Failure "
        "is definitively present AND was NOT present or documented in any earlier "
        "admission in this patient's history.  If HF is documented at the very first "
        "available admission it may be prevalent (pre-existing) rather than incident; "
        "mark it as status=1 and note 'possibly prevalent' in the evidence field.  "
        "If HF is never documented across all admissions, set incident_visit_id to "
        "null and label to 0."
    ),
    key_codes=["PheCode:428", "PheCode:428.1"],
)

HF_DECOMP_DISEASE_CONFIG = DiseaseConfig(
    name="Heart Failure (HF) Decompensation",
    icd_codes="ICD-10: I50.x  |  ICD-9: 428.x",
    diagnostic_criteria="""
- Initiation or escalation of IV loop diuretics (furosemide/Lasix, bumetanide, torsemide)
  for volume overload — not routine oral maintenance
- BNP > 400 pg/mL or NT-proBNP > 900 pg/mL documented in notes
- Explicit language: "acute decompensated heart failure", "ADHF", "acute on chronic HF",
  "decompensated CHF", "volume overload requiring admission"
- Pulmonary oedema on imaging attributed to fluid overload
- Clinical signs of congestion: bilateral crackles, peripheral oedema, orthopnoea, PND,
  JVD — all in the context of known HF""",
    incident_definition=(
        "A DECOMPENSATION EVENT is any hospital admission at which the patient presents "
        "with acute worsening of known Heart Failure requiring IV diuresis or urgent "
        "intervention, with clinical or laboratory evidence of volume overload.  "
        "A patient may have MULTIPLE decompensation events across different admissions — "
        "mark EVERY admission that meets the criterion as status=1.  "
        "Routine chronic HF follow-up, outpatient-equivalent care, or admissions "
        "primarily for other conditions are status=0."
    ),
    key_codes=[
        "LOINC:33762-6",
        "ShortName:BNP",
    ],  # NT-proBNP / BNP → silver label anchor
    output_schema_example=_RECURRING_SCHEMA_EXAMPLE,
)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION_TEMPLATE = """\
You are a senior clinical physician with expertise in {disease_name} conducting a \
structured chart review of longitudinal hospital discharge summaries.

## Your Task
Determine whether {disease_name} was ever diagnosed in this patient, \
and if so, identify the INCIDENT (first-occurrence) admission.

For reference, the relevant ICD codes are: {icd_codes}.
These codes may appear in assessment/problem-list sections of notes, but do not \
rely on code presence alone — base your judgment on the full clinical picture.

## Diagnostic Criteria
Look for the following findings to support a {disease_name} diagnosis:
{diagnostic_criteria}

## Incident Visit Definition
{incident_definition}

## Instructions — Chain-of-Thought Required
You MUST work through the admissions in strict chronological order.  Do NOT jump \
straight to the JSON output.  Follow these steps:

### STEP 1 — Per-Admission Evidence Extraction
For EACH admission (in the order they appear), write a brief structured note covering:
  a) Relevant diagnoses or clinical impressions mentioning {disease_name}
  b) Relevant medications started or changed at this admission
  c) Relevant laboratory values (with dates if available)
  d) Relevant imaging or procedure findings
  e) Whether {disease_name} language appears explicitly in the text

### STEP 2 — Per-Admission Status Judgment
After extracting evidence for each admission, assign a status:
  status=1 if {disease_name} is DEFINITIVELY present at this admission
  status=0 if {disease_name} is ABSENT or NOT DOCUMENTED at this admission
  Also assign a confidence: "high", "moderate", or "low"

### STEP 3 — Incident Determination
Review your per-admission judgments chronologically.  Identify the first admission \
where status=1 and all prior admissions have status=0.  That is the incident visit.
If no admission meets this criterion, set incident_visit_id to null.

### STEP 4 — Final JSON Output
Output ONLY valid JSON (no markdown fences, no trailing text) using this exact schema:
{output_schema_example}

IMPORTANT:
- The "reasoning" field should contain your Step 1–3 work (plain text, may be long).
- The "timeline" array must have one entry per admission in chronological order.
- visit_id is a 1-based integer (1 = earliest admission).
- "status" must be 0 or 1 (integer).
- "label" is 1 if ANY admission has status=1, else 0.
- Output ONLY the JSON object — do not add any text before or after it.\
"""

USER_TURN_TEMPLATE = """\
## Patient {subject_id} — Complete Admission History ({n_admissions} admissions)

{formatted_notes}\
"""


RECURRING_SYSTEM_INSTRUCTION_TEMPLATE = """\
You are a senior clinical physician with expertise in {disease_name} conducting a \
structured chart review of longitudinal hospital discharge summaries.

## Your Task
Identify EVERY admission in this patient's history that qualifies as a \
{disease_name} event.  A patient may have zero, one, or many such events.

For reference, the relevant ICD codes are: {icd_codes}.
Do not rely on code presence alone — base your judgment on the full clinical picture.

## Event Criteria
{diagnostic_criteria}

## Event Definition
{incident_definition}

## Instructions — Chain-of-Thought Required
Work through the admissions in strict chronological order.

### STEP 1 — Per-Admission Evidence Extraction
For EACH admission (in the order they appear), write a brief structured note covering:
  a) Relevant diagnoses or clinical impressions
  b) Relevant medications started, escalated, or changed
  c) Relevant laboratory values or biomarkers
  d) Relevant imaging or procedure findings
  e) Whether event-specific language appears explicitly in the text

### STEP 2 — Per-Admission Status Judgment
After extracting evidence, assign:
  status=1 if this admission qualifies as a {disease_name} event (meets criteria above)
  status=0 if this is a routine or unrelated admission
  Also assign a confidence: "high", "moderate", or "low"

### STEP 3 — Event List
List the visit_ids (1-based integers) of ALL admissions with status=1.
If no admission qualifies, set event_visit_ids to [].
Set label=1 if the list is non-empty, else label=0.

### STEP 4 — Final JSON Output
Output ONLY valid JSON (no markdown fences, no trailing text) using this exact schema:
{output_schema_example}

IMPORTANT:
- The "reasoning" field should contain your Step 1–2 work (plain text, may be long).
- The "timeline" array must have one entry per admission in chronological order.
- visit_id is a 1-based integer (1 = earliest admission).
- "status" must be 0 or 1 (integer).
- "event_visit_ids" is a list of integers (may be empty []).
- "label" is 1 if event_visit_ids is non-empty, else 0.
- Output ONLY the JSON object — do not add any text before or after it.\
"""


def build_system_instruction(config: DiseaseConfig) -> str:
    return SYSTEM_INSTRUCTION_TEMPLATE.format(
        disease_name=config.name,
        icd_codes=config.icd_codes,
        diagnostic_criteria=config.diagnostic_criteria.strip(),
        incident_definition=config.incident_definition.strip(),
        output_schema_example=config.output_schema_example,
    )


def build_system_instruction_recurring(config: DiseaseConfig) -> str:
    return RECURRING_SYSTEM_INSTRUCTION_TEMPLATE.format(
        disease_name=config.name,
        icd_codes=config.icd_codes,
        diagnostic_criteria=config.diagnostic_criteria.strip(),
        incident_definition=config.incident_definition.strip(),
        output_schema_example=config.output_schema_example,
    )


def format_patient_notes(notes: pd.DataFrame) -> str:
    """
    Format a patient's admission notes into a readable chronological string.

    `notes` must have columns: hadm_id, charttime (datetime), text.
    Admissions are numbered starting from 1 in chronological order.
    """
    notes = notes.sort_values("charttime").reset_index(drop=True)
    parts: list[str] = []
    for i, row in notes.iterrows():
        ct = row["charttime"]
        date_str = (
            ct.strftime("%Y-%m-%d") if isinstance(ct, pd.Timestamp) else str(ct)[:10]
        )
        header = (
            f"=== Admission {i + 1} of {len(notes)} | "
            f"hadm_id={row['hadm_id']} | charttime={date_str} ==="
        )
        parts.append(header)
        parts.append(str(row["text"]).strip())
        parts.append("")
    return "\n".join(parts)


def build_user_content(
    sid_str: str,
    notes_df: pd.DataFrame,
    config: DiseaseConfig,
    max_chars_per_note: int | None = None,
    max_notes_per_patient: int | None = None,
) -> tuple[str, str]:
    """
    Build (system_instruction, user_content) strings for a single patient.

    Returns a tuple; caller decides how to package them for the target API.
    Raises ValueError if the patient has no notes.

    max_notes_per_patient keeps the most recent N notes, which is usually the
    most clinically informative window and avoids context-length failures on
    patients with very long admission histories.
    """
    patient_notes = notes_df[notes_df["subject_id"] == sid_str].copy()
    if patient_notes.empty:
        raise ValueError(f"No notes found for subject_id={sid_str}")

    patient_notes["charttime"] = pd.to_datetime(
        patient_notes["charttime"], errors="coerce"
    )
    patient_notes = patient_notes.sort_values("charttime").reset_index(drop=True)

    if max_notes_per_patient is not None and len(patient_notes) > max_notes_per_patient:
        logger.info(
            "subject_id=%s has %d notes; keeping first %d (chronological).",
            sid_str,
            len(patient_notes),
            max_notes_per_patient,
        )
        patient_notes = patient_notes.head(max_notes_per_patient).reset_index(drop=True)

    if max_chars_per_note is not None:
        patient_notes = patient_notes.copy()
        patient_notes["text"] = patient_notes["text"].str[:max_chars_per_note]

    formatted_notes = format_patient_notes(patient_notes)
    user_content = USER_TURN_TEMPLATE.format(
        subject_id=sid_str,
        n_admissions=len(patient_notes),
        formatted_notes=formatted_notes,
    )
    system_instruction = build_system_instruction(config)
    return system_instruction, user_content


# ---------------------------------------------------------------------------
# Parse discharge_summaries.txt
# ---------------------------------------------------------------------------

_NOTE_HEADER_RE = re.compile(
    r"---\s*note_id=(\S+)\s+subject_id=(\S+)\s+hadm_id=(\S+)\s+charttime=([\d\- :]+?)\s*---"
)


def parse_discharge_summaries(notes_file: str) -> pd.DataFrame:
    """
    Parse the project's discharge_summaries.txt into a DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        note_id, subject_id, hadm_id, charttime (pd.Timestamp), text
    """
    with open(notes_file, encoding="utf-8") as fh:
        raw = fh.read()

    records: list[dict] = []
    chunks = _NOTE_HEADER_RE.split(raw)
    i = 1
    while i + 4 <= len(chunks):
        note_id, subject_id, hadm_id, charttime_raw, note_text = chunks[i : i + 5]
        try:
            charttime = pd.to_datetime(charttime_raw.strip(), utc=False)
        except Exception:
            charttime = pd.NaT
        records.append(
            {
                "note_id": note_id.strip(),
                "subject_id": subject_id.strip(),
                "hadm_id": hadm_id.strip(),
                "charttime": charttime,
                "text": note_text.strip(),
            }
        )
        i += 5

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No notes parsed from %s — check file format.", notes_file)
    else:
        logger.info(
            "Parsed %d notes for %d unique patients from %s",
            len(df),
            df["subject_id"].nunique(),
            notes_file,
        )
    return df


# ---------------------------------------------------------------------------
# Silver pre-filter (LATTE Equation 1)
# ---------------------------------------------------------------------------


def map_prefilter(
    map_results: pd.DataFrame,
    n_cases: int = 150,
    n_controls: int = 150,
    patient_col: str = "patient_id",
    phenotype_col: str = "phenotype",
    icd_coded_col: str = "icd_coded",
    seed: int = 42,
    valid_sids: set[str] | None = None,
    preferred_sids: set[str] | None = None,
) -> dict[str, list]:
    """
    Use MAP posterior scores to select gold label candidates for LATTE.

    This is the preferred prefilter when MAP has already been run.  It produces
    better candidates than ``silver_prefilter`` for disease-specific cohorts
    (where everyone already has the anchor ICD code and silver scores carry no
    discriminative signal).

    Parameters
    ----------
    map_results : pd.DataFrame
        Output of ``run_map()``.  Must have columns:
        - ``patient_col``  — patient identifier
        - ``phenotype_col`` — 0/1 MAP label
        - ``icd_coded_col`` — bool; True if patient has ≥1 anchor ICD event
    n_cases : int
        Number of MAP-confirmed cases to sample for gold labeling.
    n_controls : int
        Number of MAP-rejected ICD patients to sample for gold labeling.
        These are patients ICD-coded for the disease but rejected by MAP
        (probable miscodes / billing artifacts) — high-value negatives.
    patient_col, phenotype_col, icd_coded_col : str
        Column names in ``map_results``.
    seed : int
        Random seed for reproducible sampling.
    valid_sids : set[str] | None
        If provided, restrict sampling to this set of patient IDs.
        Use to guarantee selected patients have discharge notes before
        making BigQuery/API calls (avoids silent skips downstream).
    preferred_sids : set[str] | None
        If provided, sample from these IDs first (where they qualify),
        then fill remaining slots from other candidates.
        Use to prioritise already-cached patients so Gemini API calls
        are minimised across re-runs.

    Returns
    -------
    dict with keys:
        ``cases_pool``     — list[str] patient IDs sampled from MAP phenotype=1
        ``controls_pool``  — list[str] patient IDs sampled from MAP phenotype=0
                             AND icd_coded=True (MAP-rejected ICD cases)
        ``unlabeled_pool`` — list[str] all remaining patient IDs (LATTE mid-tier)
    """
    rng = np.random.default_rng(seed)

    df = map_results.copy()
    df[patient_col] = df[patient_col].astype(str)

    case_candidates = df[df[phenotype_col] == 1][patient_col].tolist()
    # Controls: patients ICD-coded for the disease but MAP rejected them —
    # these are the informative hard negatives for LATTE's supervised loss.
    control_candidates = df[(df[phenotype_col] == 0) & df[icd_coded_col]][
        patient_col
    ].tolist()

    if valid_sids is not None:
        valid_str = {str(s) for s in valid_sids}
        n_cases_before, n_controls_before = (
            len(case_candidates),
            len(control_candidates),
        )
        case_candidates = [p for p in case_candidates if p in valid_str]
        control_candidates = [p for p in control_candidates if p in valid_str]
        logger.info(
            "valid_sids filter: cases %d → %d, controls %d → %d",
            n_cases_before,
            len(case_candidates),
            n_controls_before,
            len(control_candidates),
        )

    def _sample_with_preference(candidates: list, n: int, preferred: set[str]) -> list:
        """Fill up to n slots from preferred first, then randomly from the rest."""
        if not preferred:
            return rng.choice(
                candidates, size=min(n, len(candidates)), replace=False
            ).tolist()
        pref = [p for p in candidates if p in preferred]
        others = [p for p in candidates if p not in preferred]
        chosen = pref[:n]
        still_needed = n - len(chosen)
        if still_needed > 0 and others:
            chosen += rng.choice(
                others, size=min(still_needed, len(others)), replace=False
            ).tolist()
        return chosen

    preferred_str = {str(s) for s in preferred_sids} if preferred_sids else set()
    cases_sampled = _sample_with_preference(case_candidates, n_cases, preferred_str)
    controls_sampled = _sample_with_preference(
        control_candidates, n_controls, preferred_str
    )

    gold_set = set(cases_sampled) | set(controls_sampled)
    unlabeled_pool = [p for p in df[patient_col].tolist() if p not in gold_set]

    logger.info(
        "MAP prefilter: %d case candidates (phenotype=1), "
        "%d control candidates (ICD-only, MAP-rejected).  "
        "Sampled %d cases + %d controls.  Unlabeled pool: %d patients.",
        len(case_candidates),
        len(control_candidates),
        len(cases_sampled),
        len(controls_sampled),
        len(unlabeled_pool),
    )
    return {
        "cases_pool": cases_sampled,
        "controls_pool": controls_sampled,
        "unlabeled_pool": unlabeled_pool,
    }


def silver_prefilter(
    obs_log: pd.DataFrame,
    config: DiseaseConfig,
    n_cases: int = 150,
    n_controls: int = 150,
    high_threshold: float = 0.7,
    low_threshold: float = 0.2,
    subject_col: str = "subject_id",
    seed: int = 42,
) -> dict[str, pd.Series]:
    """
    Compute LATTE silver labels (Eq. 1) and sample case / control pools.

    .. note::
        **Prefer** ``map_prefilter`` when MAP has already been run.
        ``silver_prefilter`` is designed for a *full mixed population* where
        most patients have zero anchor-ICD events.  Applied to a
        disease-specific sub-cohort (where everyone already has anchor codes),
        the formula loses discriminative power and may return zero case
        candidates.  Use ``silver_prefilter`` only when MAP results are not
        yet available.

    Returns dict with keys:
        "silver_scores", "cases_pool", "controls_pool", "unlabeled_pool"
    """
    rng = np.random.default_rng(seed)

    key_mask = obs_log["event"].isin(config.key_codes)
    key_counts = obs_log[key_mask].groupby(subject_col).size().rename("key_count")
    total_counts = obs_log.groupby(subject_col).size().rename("total_count")

    summary = pd.concat([key_counts, total_counts], axis=1).fillna(0)
    log_key = np.log1p(summary["key_count"])
    log_total = np.log1p(summary["total_count"])
    score_raw = (log_key - config.silver_alpha * log_total) / config.silver_tau
    silver_scores = 1.0 / (1.0 + np.exp(-score_raw))

    high_pool = silver_scores[silver_scores > high_threshold].index.tolist()
    low_pool = silver_scores[silver_scores < low_threshold].index.tolist()
    mid_pool = silver_scores[
        (silver_scores >= low_threshold) & (silver_scores <= high_threshold)
    ].index.tolist()

    n_cases_sample = min(n_cases, len(high_pool))
    n_controls_sample = min(n_controls, len(low_pool))

    cases_sampled = rng.choice(high_pool, size=n_cases_sample, replace=False).tolist()
    controls_sampled = rng.choice(
        low_pool, size=n_controls_sample, replace=False
    ).tolist()

    logger.info(
        "Silver filter: %d high (>%.2f), %d low (<%.2f), %d mid.  "
        "Sampled %d cases + %d controls for gold labeling.",
        len(high_pool),
        high_threshold,
        len(low_pool),
        low_threshold,
        len(mid_pool),
        n_cases_sample,
        n_controls_sample,
    )
    return {
        "silver_scores": silver_scores,
        "cases_pool": cases_sampled,
        "controls_pool": controls_sampled,
        "unlabeled_pool": mid_pool,
    }


# ---------------------------------------------------------------------------
# Shared result parsing helpers
# ---------------------------------------------------------------------------


def extract_json_from_response(text: str) -> dict | None:
    """
    Extract the JSON object from the model's response text.

    Strips markdown fences if present and finds the last {...} block.
    """
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        text = fence_match.group(1)
    brace_match = re.search(r"\{[\s\S]*\}", text, re.DOTALL)
    if not brace_match:
        return None
    try:
        return json.loads(brace_match.group(0))
    except json.JSONDecodeError:
        return None


def build_result_record(
    parsed: dict,
    subject_id_fallback: str,
    baseline_date: str | None = None,
    month_window: int = 3,
) -> dict:
    """
    Convert a parsed model JSON dict into a LATTE gold label record.

    Used by both medgemma.py and gemini.py after extracting response text.
    """
    # Always use the known subject_id we queried for — the model may echo back
    # the schema example ID instead of the real one.
    sid = subject_id_fallback or str(parsed.get("subject_id", ""))
    label = int(parsed.get("label", -1))
    timeline = parsed.get("timeline", [])
    incident_vid = parsed.get("incident_visit_id")

    incident_hadm_id = None
    incident_charttime = None
    incident_T = None

    if incident_vid is not None:
        incident_visit = next(
            (v for v in timeline if v.get("visit_id") == incident_vid), None
        )
        if incident_visit:
            incident_hadm_id = str(incident_visit.get("hadm_id", ""))
            ct_raw = incident_visit.get("charttime")
            try:
                incident_charttime = pd.to_datetime(ct_raw)
            except Exception:
                incident_charttime = None

            if incident_charttime is not None and baseline_date is not None:
                try:
                    base_dt = pd.to_datetime(baseline_date)
                    delta_days = (incident_charttime - base_dt).days
                    incident_T = int(np.floor(delta_days / (30.44 * month_window)))
                except Exception:
                    incident_T = None

    return {
        "subject_id": sid,
        "label": label,
        "incident_hadm_id": incident_hadm_id,
        "incident_charttime": incident_charttime,
        "incident_T": incident_T,
        "timeline_json": json.dumps(timeline),
        "parse_error": False,
    }


def build_result_record_recurring(
    parsed: dict,
    subject_id_fallback: str,
) -> dict:
    """
    Convert a parsed recurring-event model JSON into a cache record.

    Extracts event_visit_ids from the response and resolves their charttimes
    from the timeline.  baseline_date / month_window are NOT applied here —
    they are deferred to parse_gemini_recurring_results so the cache remains
    timezone-agnostic and reusable with different study anchors.
    """
    sid = subject_id_fallback or str(parsed.get("subject_id", ""))
    label = int(parsed.get("label", -1))
    timeline = parsed.get("timeline", [])
    event_vids = parsed.get("event_visit_ids") or []

    vid_to_charttime: dict[int, str] = {}
    for v in timeline:
        vid = v.get("visit_id")
        ct = v.get("charttime")
        if vid is not None and ct:
            vid_to_charttime[int(vid)] = str(ct)

    event_charttimes = [
        vid_to_charttime[int(vid)] for vid in event_vids if int(vid) in vid_to_charttime
    ]

    return {
        "subject_id": sid,
        "label": label,
        "event_visit_ids": event_vids,
        "event_charttimes": event_charttimes,
        "timeline_json": json.dumps(timeline),
        "parse_error": False,
    }


def error_record(subject_id: str = "") -> dict:
    return {
        "subject_id": subject_id,
        "label": -1,
        "incident_hadm_id": None,
        "incident_charttime": None,
        "incident_T": None,
        "timeline_json": "[]",
        "parse_error": True,
    }


# ---------------------------------------------------------------------------
# Convert labeler results → LATTE per-visit labels
# ---------------------------------------------------------------------------


def labels_to_latte(
    results_df: pd.DataFrame,
    obs_log: pd.DataFrame,
    baseline_date: str,
    month_window: int = 3,
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """
    Convert parsed labeler results to LATTE's per-visit binary label format.

    LATTE's supervised signal needs a label for every (patient, T) pair:
      - Controls (label=0): all visit windows → Y = 0
      - Cases (label=1): windows < incident_T → Y = 0; windows >= incident_T → Y = 1

    Returns
    -------
    pd.DataFrame with columns [subject_id, T, Y]
    """
    base_dt = pd.to_datetime(baseline_date)

    obs = obs_log.copy()
    obs["_dt"] = pd.to_datetime(obs["datetime"], errors="coerce")
    obs["T"] = np.floor((obs["_dt"] - base_dt).dt.days / (30.44 * month_window)).astype(
        "Int64"
    )

    valid = results_df[~results_df["parse_error"]]
    labeled_sids = set(valid[subject_col].astype(str))
    obs_labeled = obs[obs[subject_col].astype(str).isin(labeled_sids)]

    all_visit_pairs = (
        obs_labeled[[subject_col, "T"]]
        .drop_duplicates()
        .rename(columns={subject_col: "subject_id"})
        .copy()
    )
    all_visit_pairs["subject_id"] = all_visit_pairs["subject_id"].astype(str)

    label_map = valid.set_index("subject_id")[["label", "incident_T"]]
    all_visit_pairs = all_visit_pairs.merge(label_map, on="subject_id", how="left")

    def _compute_Y(row):
        if row["label"] == 0:
            return 0
        if row["label"] == 1:
            inc_T = row["incident_T"]
            if pd.isna(inc_T):
                return 1
            return int(row["T"] >= int(inc_T))
        return -1

    all_visit_pairs["Y"] = all_visit_pairs.apply(_compute_Y, axis=1)
    return (
        all_visit_pairs[["subject_id", "T", "Y"]]
        .sort_values(["subject_id", "T"])
        .reset_index(drop=True)
    )


def recurring_labels_to_latte(
    results_df: pd.DataFrame,
    obs_log: pd.DataFrame,
    baseline_date: str,
    month_window: int = 3,
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """
    Convert recurring-event Gemini results to LATTE per-visit binary labels.

    For each labeled patient, Y=1 at any visit window that contains a
    qualifying event; Y=0 at all other windows.  Patients with label=0
    (no events found) have Y=0 at every window.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of parse_gemini_recurring_results().
        Must have columns: subject_id, label, event_Ts (list[int]), parse_error.
    obs_log : pd.DataFrame
        Observation log; used to enumerate all (patient, T) pairs.
    baseline_date : str
        Study anchor date (YYYY-MM-DD); must match the value used in all scripts.
    month_window : int
        Size of each time window in months.

    Returns
    -------
    pd.DataFrame with columns [subject_id, T, Y]
    """
    base_dt = pd.to_datetime(baseline_date)

    obs = obs_log.copy()
    obs["_dt"] = pd.to_datetime(obs["datetime"], errors="coerce")
    obs["T"] = np.floor((obs["_dt"] - base_dt).dt.days / (30.44 * month_window)).astype(
        "Int64"
    )

    valid = results_df[~results_df["parse_error"]]
    labeled_sids = set(valid[subject_col].astype(str))
    obs_labeled = obs[obs[subject_col].astype(str).isin(labeled_sids)]

    all_visit_pairs = (
        obs_labeled[[subject_col, "T"]]
        .drop_duplicates()
        .rename(columns={subject_col: "subject_id"})
        .copy()
    )
    all_visit_pairs["subject_id"] = all_visit_pairs["subject_id"].astype(str)

    # Build a map: subject_id → set of event T values
    event_T_map: dict[str, set] = {}
    for _, row in valid.iterrows():
        sid = str(row[subject_col])
        ts = row.get("event_Ts") or []
        event_T_map[sid] = set(int(t) for t in ts if pd.notna(t))

    def _compute_Y(row):
        sid = row["subject_id"]
        t = row["T"]
        if pd.isna(t):
            return -1
        return int(int(t) in event_T_map.get(sid, set()))

    all_visit_pairs["Y"] = all_visit_pairs.apply(_compute_Y, axis=1)
    return (
        all_visit_pairs[["subject_id", "T", "Y"]]
        .sort_values(["subject_id", "T"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Validation: cross-check against principal-diagnosis rule
# ---------------------------------------------------------------------------


def validate_against_principal_dx(
    results_df: pd.DataFrame,
    diagnoses_df: pd.DataFrame,
    config: DiseaseConfig,
    subject_col: str = "subject_id",
    icd_col: str = "icd_code",
    seq_col: str = "seq_num",
    admittime_col: str = "admittime",
) -> pd.DataFrame:
    """
    Compare gold labels against a rule-based baseline: the patient's earliest
    admission where a key disease ICD code appears as principal diagnosis (seq_num=1).

    Returns
    -------
    pd.DataFrame with columns:
        subject_id, gold_label, rule_label, agreement,
        gold_incident_charttime, rule_incident_admittime
    """
    principal_dx = diagnoses_df[diagnoses_df[seq_col] == 1].copy()
    raw_codes = [c.split(":")[-1] for c in config.key_codes]
    key_mask = principal_dx[icd_col].str.startswith(tuple(raw_codes))
    first_match = (
        principal_dx[key_mask]
        .sort_values(admittime_col)
        .groupby(subject_col)
        .first()[[admittime_col]]
        .rename(columns={admittime_col: "rule_incident_admittime"})
        .reset_index()
    )
    first_match["rule_label"] = 1

    valid = results_df[~results_df["parse_error"]].copy()
    valid["subject_id"] = valid["subject_id"].astype(str)
    first_match["subject_id"] = first_match[subject_col].astype(str)

    merged = valid.merge(first_match, on="subject_id", how="left")
    merged["rule_label"] = merged["rule_label"].fillna(0).astype(int)
    merged["agreement"] = merged["label"] == merged["rule_label"]

    agreement_rate = merged["agreement"].mean()
    logger.info(
        "Gold vs principal-dx agreement: %.1f%%  (%d/%d patients)",
        100 * agreement_rate,
        merged["agreement"].sum(),
        len(merged),
    )

    return merged[
        [
            "subject_id",
            "label",
            "rule_label",
            "agreement",
            "incident_charttime",
            "rule_incident_admittime",
        ]
    ].rename(columns={"label": "gold_label"})
