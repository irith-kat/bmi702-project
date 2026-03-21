"""
General EHR preprocessing utilities — algorithm-agnostic observation log construction.

All public functions produce a standardized long-format "observation log" DataFrame
with exactly five columns:

    subject_id  (int | str)    — patient identifier
    event_type  (str)          — vocabulary / modality label
                                 one of: "phecode", "rxnorm", "ccs", "cui", "loinc"
    event       (str)          — vocabulary-prefixed code value
                                 e.g. "PheCode:714.1", "RXNORM:1049630",
                                      "CCS:3", "LNC:4548-4", "CUI:C0003873"
    value       (float | None) — numeric measurement result; None for categorical events
                                 (diagnoses, medications, procedures, NLP mentions).
                                 Populated for lab events: e.g. 8.2 for HbA1c %.
                                 Including this column in the core schema keeps the
                                 format forward-compatible with lab data without a
                                 future schema change.
    datetime    (datetime)     — when the observation occurred (admission, prescription,
                                 procedure, or note date)

event_type and event are intentionally redundant: event_type enables fast
modality-level filtering (e.g. obs_log[obs_log["event_type"] == "phecode"])
without parsing string prefixes, which matters for rule-based and temporal
algorithms that reason differently per modality (e.g., LATTE).

This schema is algorithm-agnostic: the same observation log can feed MAP (via
preprocess_map in map/map.py), simple rule-based phenotyping (filter by
event_type / event / value), or temporal algorithms like LATTE (use
datetime + event_type).
"""

import pandas as pd
from note_ner import extract_cui_features
from rollup import rollup_icd_to_phecode, rollup_cpt_to_ccs, rollup_ndc_to_ingredient


# icd_to_events
# -------------
# Roll up ICD-9/10 diagnosis codes to PheCodes and return rows in the
# observation log format. Rows with no PheCode mapping are silently dropped.
#
# Args:
#   df          (pd.DataFrame) : Diagnoses table. Must contain subject_col, icd_col,
#                                and date_col. Example: MIMIC-IV diagnoses_icd joined
#                                with admissions for admit dates.
#   icd_col     (str)          : Column of raw ICD-9/10 codes (without dots is fine;
#                                rollup.py inserts dots automatically).
#                                Example: "icd_code"
#   date_col    (str)          : Column of event dates.
#                                Example: "admittime"
#   subject_col  (str)          : Patient ID column. Default: "subject_id"
#   mapping_file (str)          : Path to the PheCode mapping CSV.
#                                 Default: "Phecode_map_v1_2_icd9_icd10cm.csv"
#                                 Pass an absolute path to avoid CWD dependency.
#
# Returns:
#   pd.DataFrame : Observation log rows with columns
#                  [subject_col, "event_type", "event", "value", "datetime"].
#                  event_type: "phecode"
#                  event format: "PheCode:<code>"  e.g. "PheCode:714.1"
#                  value: None (diagnoses are categorical)
def icd_to_events(
    df: pd.DataFrame,
    icd_col: str,
    date_col: str,
    subject_col: str = "subject_id",
    mapping_file: str = "Phecode_map_v1_2_icd9_icd10cm.csv",
) -> pd.DataFrame:
    rolled = rollup_icd_to_phecode(df, icd_col, mapping_file=mapping_file)
    events = rolled.dropna(subset=["Phecode"]).copy()
    events["event_type"] = "phecode"
    events["event"] = "PheCode:" + events["Phecode"]
    events["value"] = None
    return (
        events[[subject_col, "event_type", "event", "value", date_col]]
        .rename(columns={date_col: "datetime"})
        .reset_index(drop=True)
    )


# drug_to_events
# --------------
# Roll up NDC codes to RxNorm ingredient level and return observation log rows.
# Internally calls rollup_ndc_to_ingredient() from rollup.py; no pre-processing
# of the prescriptions table is required before calling this function.
#
# Filter to drug_type == 'MAIN' before passing to avoid double-counting
# multi-component drugs (MIMIC rows also contain BASE and ADDITIVE entries).
#
# Args:
#   df                     (pd.DataFrame) : Prescriptions table. Must contain
#                                           subject_col, ndc_col, and date_col.
#                                           Example: MIMIC-IV prescriptions.
#   ndc_col                (str)          : Column of 11-digit NDC strings.
#                                           Example: "ndc"
#   date_col               (str)          : Column of prescription start dates.
#                                           Example: "starttime"
#   subject_col            (str)          : Patient ID column. Default: "subject_id"
#   drug_col               (str | None)   : Optional free-text drug name column used
#                                           as a case-insensitive fallback when NDC
#                                           lookup fails (via gcpt_drug_ndc mapping).
#                                           Example: "drug". Default: None
#   ndc_mapping_file       (str)          : Path to ndc_to_rxnorm_ingredient.csv.
#   drug_name_mapping_file (str)          : Path to drug_name_to_rxnorm_ingredient.csv.
#
# Returns:
#   pd.DataFrame : Observation log rows with columns
#                  [subject_col, "event_type", "event", "value", "datetime"].
#                  Rows where neither NDC nor drug-name lookup resolves are dropped.
#                  event_type: "rxnorm"
#                  event format: "RXNORM:<ingredient_id>"  e.g. "RXNORM:956874"
#                  value: None (medication presence is categorical)
def drug_to_events(
    df: pd.DataFrame,
    ndc_col: str,
    date_col: str,
    subject_col: str = "subject_id",
    drug_col: str | None = None,
    ndc_mapping_file: str = "mapping_dicts/ndc_to_rxnorm_ingredient.csv",
    drug_name_mapping_file: str = "mapping_dicts/drug_name_to_rxnorm_ingredient.csv",
) -> pd.DataFrame:
    rolled = rollup_ndc_to_ingredient(
        df,
        ndc_column=ndc_col,
        drug_column=drug_col,
        ndc_mapping_file=ndc_mapping_file,
        drug_name_mapping_file=drug_name_mapping_file,
    )
    events = rolled.dropna(subset=["rxnorm_ingredient_id"]).copy()
    events["event_type"] = "rxnorm"
    events["event"] = "RXNORM:" + events["rxnorm_ingredient_id"].astype(str)
    events["value"] = None
    return (
        events[[subject_col, "event_type", "event", "value", date_col]]
        .rename(columns={date_col: "datetime"})
        .reset_index(drop=True)
    )


# cpt_to_events
# -------------
# Roll up CPT/HCPCS procedure codes to AHRQ CCS categories and return
# observation log rows. Rows with no CCS mapping are silently dropped.
#
# Args:
#   df           (pd.DataFrame) : Procedures table. Must contain subject_col,
#                                 cpt_col, and date_col.
#                                 Example: MIMIC-IV hcpcsevents or procedures_icd.
#   cpt_col      (str)          : Column of CPT/HCPCS codes.
#                                 Example: "hcpcs_cd"
#   date_col     (str)          : Column of procedure dates.
#                                 Example: "chartdate"
#   subject_col  (str)          : Patient ID column. Default: "subject_id"
#   mapping_file (str)          : Path to the AHRQ CCS mapping CSV.
#                                 Default: "CCS_Services_Procedures_v2025-1.csv"
#
# Returns:
#   pd.DataFrame : Observation log rows with columns
#                  [subject_col, "event_type", "event", "value", "datetime"].
#                  event_type: "ccs"
#                  event format: "CCS:<category>"  e.g. "CCS:3"
#                  value: None (procedures are categorical)
def cpt_to_events(
    df: pd.DataFrame,
    cpt_col: str,
    date_col: str,
    subject_col: str = "subject_id",
    mapping_file: str = "CCS_Services_Procedures_v2025-1.csv",
) -> pd.DataFrame:
    rolled = rollup_cpt_to_ccs(df, cpt_col, mapping_file)
    events = rolled.dropna(subset=["ccs_category"]).copy()
    events["event_type"] = "ccs"
    events["event"] = "CCS:" + events["ccs_category"].astype(str)
    events["value"] = None
    return (
        events[[subject_col, "event_type", "event", "value", date_col]]
        .rename(columns={date_col: "datetime"})
        .reset_index(drop=True)
    )


# notes_to_events
# ---------------
# Extract CUI mentions from clinical notes using MedSpaCy and return
# observation log rows. Each row represents one confirmed CUI mention in
# one note. Negated, uncertain, and family-history mentions are excluded.
#
# Args:
#   notes_df       (pd.DataFrame) : Clinical notes table. One row per note.
#                                   Must contain subject_col, text_col, date_col.
#                                   Example: MIMIC-IV-Note discharge notes.
#   text_col       (str)          : Column of raw note text. Example: "text"
#   date_col       (str)          : Column of note dates. Example: "chartdate"
#   target_cuis    (list[dict])   : CUIs to search for. Each dict has "term" and "cui".
#                                   Example: [{"term": "rheumatoid arthritis", "cui": "C0003873"}]
#                                   Use get_once_features()["nlp_target_cuis"] to build this.
#   subject_col    (str)          : Patient ID column. Default: "subject_id"
#   max_note_chars     (int | None)   : Truncate each note to this many characters before NLP.
#                                       Recommended: 10_000 for a good speed/recall tradeoff.
#                                       Default: None (no truncation).
#   notes_per_patient  (int | None)   : Keep only the N most recent notes per patient before
#                                       running NLP. Ensures all patients get coverage when
#                                       notes_df contains many notes per patient.
#                                       Default: None (use all notes as provided).
#   n_process          (int)          : Workers for nlp.pipe(). 1 = single-threaded (safe in
#                                       Jupyter). -1 = all cores. Default: 1
#
# Returns:
#   pd.DataFrame : Observation log rows with columns
#                  [subject_col, "event_type", "event", "value", "datetime"].
#                  event_type: "cui"
#                  event format: "CUI:<cui>"  e.g. "CUI:C0003873"
#                  value: None (NLP mentions are categorical)
def notes_to_events(
    notes_df: pd.DataFrame,
    text_col: str,
    date_col: str,
    target_cuis: list[dict],
    subject_col: str = "subject_id",
    max_note_chars: int | None = None,
    notes_per_patient: int | None = None,
    n_process: int = 1,
) -> pd.DataFrame:
    if notes_per_patient is not None:
        notes_df = (
            notes_df.sort_values(date_col, ascending=False)
            .groupby(subject_col)
            .head(notes_per_patient)
            .reset_index(drop=True)
        )
    cui_df = extract_cui_features(
        notes_df,
        text_column=text_col,
        id_column=subject_col,
        target_cuis=target_cuis,
        date_column=date_col,
        max_note_chars=max_note_chars,
        n_process=n_process,
    )
    cui_df["event_type"] = "cui"
    cui_df["event"] = "CUI:" + cui_df["cui"]
    cui_df["value"] = None
    return cui_df[
        [subject_col, "event_type", "event", "value", "datetime"]
    ].reset_index(drop=True)


# build_obs_log
# -------------
# Build a unified observation log from one or more EHR data modalities.
# Each provided modality is converted to the standard
# (subject_id, event_type, event, value, datetime) schema and concatenated
# vertically. Any subset of modalities may be supplied; omit irrelevant
# ones by leaving them as None.
#
# Args:
#   icd_df          (pd.DataFrame | None) : Diagnoses table for ICD → PheCode rollup.
#                                           Requires icd_col and icd_date_col.
#   icd_col         (str | None)          : ICD code column in icd_df.
#                                           Example: "icd_code"
#   icd_date_col    (str | None)          : Date column in icd_df.
#                                           Example: "admittime"
#
#   drug_df                (pd.DataFrame | None) : Prescriptions table.
#                                                  Requires drug_ndc_col and drug_date_col.
#                                                  Filter to drug_type == 'MAIN' before passing.
#   drug_ndc_col           (str | None)          : 11-digit NDC column.
#                                                  Example: "ndc"
#   drug_date_col          (str | None)          : Date column in drug_df.
#                                                  Example: "starttime"
#   drug_col               (str | None)          : Optional free-text drug name column for
#                                                  fallback NDC lookup. Example: "drug"
#   ndc_mapping_file       (str)                 : Path to ndc_to_rxnorm_ingredient.csv.
#   drug_name_mapping_file (str)                 : Path to drug_name_to_rxnorm_ingredient.csv.
#
#   cpt_df          (pd.DataFrame | None) : Procedures table for CPT → CCS rollup.
#                                           Requires cpt_col and cpt_date_col.
#   cpt_col         (str | None)          : CPT/HCPCS code column.
#                                           Example: "hcpcs_cd"
#   cpt_date_col    (str | None)          : Date column in cpt_df.
#                                           Example: "chartdate"
#
#   notes_df        (pd.DataFrame | None) : Clinical notes table for CUI extraction.
#                                           Requires notes_text_col, notes_date_col,
#                                           and target_cuis.
#   notes_text_col  (str | None)          : Note text column.
#                                           Example: "text"
#   notes_date_col  (str | None)          : Note date column.
#                                           Example: "chartdate"
#   target_cuis     (list[dict] | None)   : CUIs to extract (term + cui dicts).
#                                           Use get_once_features()["nlp_target_cuis"].
#
#   subject_col      (str)                : Patient ID column shared across all tables.
#                                           Default: "subject_id"
#   icd_mapping_file (str)                : Path to PheCode mapping CSV for ICD rollup.
#                                           Default: "Phecode_map_v1_2_icd9_icd10cm.csv"
#   cpt_mapping_file (str)                : Path to AHRQ CCS mapping CSV for CPT rollup.
#                                           Default: "CCS_Services_Procedures_v2025-1.csv"
#
# Returns:
#   pd.DataFrame : Observation log with columns:
#                  - subject_col  (int | str)    : patient identifier
#                  - "event_type" (str)           : modality label — "phecode", "rxnorm",
#                                                  "ccs", or "cui"
#                                                  (prescriptions always emit "rxnorm")
#                  - "event"      (str)           : prefixed code, e.g. "PheCode:714.1"
#                  - "value"      (float | None)  : numeric result; None for categorical
#                                                  events; populated for future lab events
#                  - "datetime"   (datetime)      : when the observation occurred
def build_obs_log(
    icd_df: pd.DataFrame | None = None,
    icd_col: str | None = None,
    icd_date_col: str | None = None,
    drug_df: pd.DataFrame | None = None,
    drug_ndc_col: str | None = None,
    drug_date_col: str | None = None,
    drug_col: str | None = None,
    cpt_df: pd.DataFrame | None = None,
    cpt_col: str | None = None,
    cpt_date_col: str | None = None,
    notes_df: pd.DataFrame | None = None,
    notes_text_col: str | None = None,
    notes_date_col: str | None = None,
    target_cuis: list[dict] | None = None,
    subject_col: str = "subject_id",
    icd_mapping_file: str = "Phecode_map_v1_2_icd9_icd10cm.csv",
    cpt_mapping_file: str = "CCS_Services_Procedures_v2025-1.csv",
    ndc_mapping_file: str = "mapping_dicts/ndc_to_rxnorm_ingredient.csv",
    drug_name_mapping_file: str = "mapping_dicts/drug_name_to_rxnorm_ingredient.csv",
    max_note_chars: int | None = None,
    notes_per_patient: int | None = None,
    n_process: int = 1,
) -> pd.DataFrame:
    parts = []

    if icd_df is not None:
        if icd_col is None or icd_date_col is None:
            raise ValueError(
                "icd_col and icd_date_col are required when icd_df is provided."
            )
        parts.append(
            icd_to_events(icd_df, icd_col, icd_date_col, subject_col, icd_mapping_file)
        )

    if drug_df is not None:
        if drug_ndc_col is None or drug_date_col is None:
            raise ValueError(
                "drug_ndc_col and drug_date_col are required when drug_df is provided."
            )
        parts.append(
            drug_to_events(
                drug_df,
                drug_ndc_col,
                drug_date_col,
                subject_col,
                drug_col=drug_col,
                ndc_mapping_file=ndc_mapping_file,
                drug_name_mapping_file=drug_name_mapping_file,
            )
        )

    if cpt_df is not None:
        if cpt_col is None or cpt_date_col is None:
            raise ValueError(
                "cpt_col and cpt_date_col are required when cpt_df is provided."
            )
        parts.append(
            cpt_to_events(cpt_df, cpt_col, cpt_date_col, subject_col, cpt_mapping_file)
        )

    if notes_df is not None:
        if notes_text_col is None or notes_date_col is None or target_cuis is None:
            raise ValueError(
                "notes_text_col, notes_date_col, and target_cuis are required when notes_df is provided."
            )
        parts.append(
            notes_to_events(
                notes_df,
                notes_text_col,
                notes_date_col,
                target_cuis,
                subject_col,
                max_note_chars=max_note_chars,
                notes_per_patient=notes_per_patient,
                n_process=n_process,
            )
        )

    if not parts:
        raise ValueError(
            "At least one modality (icd_df, drug_df, cpt_df, or notes_df) must be provided."
        )

    return pd.concat(parts, ignore_index=True)
