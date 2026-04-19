import os
import re

import pandas as pd


def rollup_icd_to_phecode(
    df: pd.DataFrame,
    icd_column: str,
    mapping_file: str = "Phecode_map_v1_2_icd9_icd10cm.csv",
    has_dots: bool | None = None,
) -> pd.DataFrame:
    """
    Map ICD-9/10 codes to PheCodes using the standard mapping file.

    Args:
        df (pd.DataFrame): Input DataFrame containing ICD codes.
        icd_column (str): Name of the column containing ICD codes.
        mapping_file (str): Path to the Phecode mapping CSV.
        has_dots (bool | None): Whether ICD codes already contain a decimal dot.
            None (default) → auto-detect from the first non-null value.
            False → insert dot at position 3 (MIMIC-IV raw format).
            True → codes already dotted; skip insertion.

    Returns:
        pd.DataFrame: DataFrame with added 'Phecode' and 'PhecodeString' columns.
    """
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    # Auto-detect: check first non-null code for presence of '.'
    if has_dots is None:
        sample = df[icd_column].dropna()
        has_dots = ("." in str(sample.iloc[0])) if not sample.empty else True

    def _add_icd_dot(code: str) -> str:
        if pd.isna(code):
            return code
        code = str(code)
        if "." not in code and len(code) > 3:
            return code[:3] + "." + code[3:]
        return code

    map_df = pd.read_csv(
        mapping_file,
        usecols=["ICD", "Phecode", "PhecodeString"],
        dtype={"ICD": str, "Phecode": str},
    )

    df = df.copy()
    if not has_dots:
        df[icd_column] = df[icd_column].apply(_add_icd_dot)

    result_df = df.merge(map_df, left_on=icd_column, right_on="ICD", how="left")

    if icd_column != "ICD":
        result_df.drop(columns=["ICD"], inplace=True)

    return result_df


def rollup_ndc_to_ingredient(
    df: pd.DataFrame,
    ndc_column: str,
    drug_column: str | None = None,
    ndc_mapping_file: str = "mapping_dicts/ndc_to_rxnorm_ingredient.csv",
    drug_name_mapping_file: str = "mapping_dicts/drug_name_to_rxnorm_ingredient.csv",
) -> pd.DataFrame:
    """
    Map NDC codes to RxNorm ingredient-level concepts.

    Primary lookup: 11-digit NDC → RxNorm ingredient via OMOP Athena mapping
    (NDC → Clinical/Branded Drug → Ingredient via CONCEPT_ANCESTOR).

    Fallback: when NDC lookup fails and drug_column is provided, attempts a
    case-insensitive match against gcpt_drug_ndc-derived drug name mapping.

    Adds two columns to the returned DataFrame:
      - rxnorm_ingredient_id   : RxNorm concept_id of the ingredient
      - rxnorm_ingredient_name : human-readable ingredient name

    Args:
        df: Input DataFrame containing NDC codes.
        ndc_column: Column name with 11-digit NDC strings.
        drug_column: Optional column name with drug free-text names (used as
            fallback via gcpt_drug_ndc when NDC lookup misses).
        ndc_mapping_file: Path to ndc_to_rxnorm_ingredient.csv.
        drug_name_mapping_file: Path to drug_name_to_rxnorm_ingredient.csv.

    Returns:
        DataFrame with 'rxnorm_ingredient_id' and 'rxnorm_ingredient_name' added.
        Rows with no match in either lookup have NaN in those columns.
    """
    if not os.path.exists(ndc_mapping_file):
        raise FileNotFoundError(f"NDC mapping file not found: {ndc_mapping_file}")

    def _normalize_ndc(ndc: str) -> str:
        """Normalize NDC to 11-digit format.
        Strips non-digit characters (hyphens, spaces), then prepends '0' when
        the result is 10 digits — the common case when a labeler code was stored
        with 4 digits instead of 5 (4-4-2 → 5-4-2 canonical form).
        Already-11-digit codes are returned unchanged.
        For ambiguous 10-digit inputs from non-MIMIC sources, use ndclib in an
        offline mapping-generation step (custom-vocab-mapping skill) to get the
        correct segment-aware normalization before this runtime path is reached."""
        if pd.isna(ndc):
            return ndc
        digits = re.sub(r"\D", "", str(ndc))
        if len(digits) == 10:
            return "0" + digits
        return digits

    result = df.copy()
    result[ndc_column] = result[ndc_column].apply(_normalize_ndc)

    ndc_map = pd.read_csv(
        ndc_mapping_file,
        dtype={"ndc": str, "ingredient_id": str, "ingredient_name": str},
    ).rename(
        columns={
            "ingredient_id": "rxnorm_ingredient_id",
            "ingredient_name": "rxnorm_ingredient_name",
        }
    )

    result = result.merge(ndc_map, left_on=ndc_column, right_on="ndc", how="left")
    # Drop the mapping's join key only if it differs from the input column name
    if ndc_column != "ndc":
        result.drop(columns=["ndc"], inplace=True)

    # Fallback: drug name lookup for rows NDC lookup missed
    if drug_column is not None and os.path.exists(drug_name_mapping_file):
        drug_map = pd.read_csv(
            drug_name_mapping_file,
            dtype={"drug_name": str, "ingredient_id": str, "ingredient_name": str},
        ).rename(
            columns={
                "ingredient_id": "rxnorm_ingredient_id_fb",
                "ingredient_name": "rxnorm_ingredient_name_fb",
            }
        )

        missed = result["rxnorm_ingredient_id"].isna()
        if missed.any():
            normalized = result.loc[missed, drug_column].str.strip().str.lower()
            fb = normalized.to_frame("drug_name").merge(
                drug_map, on="drug_name", how="left"
            )
            result.loc[missed, "rxnorm_ingredient_id"] = fb[
                "rxnorm_ingredient_id_fb"
            ].values
            result.loc[missed, "rxnorm_ingredient_name"] = fb[
                "rxnorm_ingredient_name_fb"
            ].values

    return result


def rollup_itemid_to_loinc(
    df: pd.DataFrame,
    itemid_column: str,
    mapping_file: str = "mapping_dicts/d_labitems_to_loinc.csv",
) -> pd.DataFrame:
    """
    Map MIMIC-IV lab itemids to LOINC codes using the MIT-LCP OMOP mapping.

    Source: github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/mapping

    Only rows where the OMOP vocabulary is 'LOINC' are kept in the mapping;
    itemids that map to a non-LOINC concept (or have no mapping) get NaN in
    the added columns — consistent with how other rollup functions handle misses.

    Args:
        df: Input DataFrame containing MIMIC-IV lab itemids.
        itemid_column: Column name holding integer itemids.
        mapping_file: Path to d_labitems_to_loinc.csv.

    Returns:
        DataFrame with 'loinc_code' and 'loinc_label' columns added.
    """
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    map_df = pd.read_csv(
        mapping_file,
        usecols=[
            "itemid (omop_source_code)",
            "omop_concept_code",
            "omop_concept_name",
            "omop_vocabulary_id",
        ],
        dtype={"itemid (omop_source_code)": str, "omop_concept_code": str},
    ).rename(
        columns={
            "itemid (omop_source_code)": "itemid",
            "omop_concept_code": "loinc_code",
            "omop_concept_name": "loinc_label",
        }
    )

    # Keep only LOINC-vocabulary rows; non-LOINC concepts become misses
    map_df = (
        map_df[map_df["omop_vocabulary_id"] == "LOINC"]
        .drop(columns=["omop_vocabulary_id"])
        .rename(columns={"itemid": "_map_itemid"})
    )

    result = df.copy()
    result["_itemid_str"] = result[itemid_column].astype(str)
    result = result.merge(
        map_df, left_on="_itemid_str", right_on="_map_itemid", how="left"
    )
    result.drop(columns=["_itemid_str", "_map_itemid"], inplace=True)

    return result


def rollup_cpt_to_ccs(
    df: pd.DataFrame,
    cpt_column: str,
    mapping_file: str = "CCS_Services_Procedures_v2025-1.csv",
) -> pd.DataFrame:
    """
    Roll up CPT/HCPCS codes to AHRQ CCS categories using alphanumeric range
    matching from the AHRQ mapping file.

    Args:
        df (pd.DataFrame): Input DataFrame containing CPT/HCPCS codes.
        cpt_column (str): Name of the column containing raw CPT codes.
        mapping_file (str): Path to the AHRQ CSV mapping file.

    Returns:
        pd.DataFrame: Original DataFrame with 'ccs_category' and
                      'ccs_description' columns added.
    """
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    # The AHRQ CCS file has a metadata title row before the actual header.
    mapping_df = pd.read_csv(mapping_file, skiprows=1)
    mapping_df.columns = [c.lower().replace(" ", "_") for c in mapping_df.columns]

    # The file uses a combined "Code Range" column (e.g. "'0735T-0735T'") instead
    # of separate start/end columns. Strip surrounding single quotes then split.
    mapping_df["code_range"] = mapping_df["code_range"].astype(str).str.strip("'")
    mapping_df["start_code"] = (
        mapping_df["code_range"].str.split("-").str[0].str.zfill(5)
    )
    mapping_df["end_code"] = (
        mapping_df["code_range"].str.split("-").str[-1].str.zfill(5)
    )

    df_result = df.copy()
    df_result[cpt_column] = (
        df_result[cpt_column].astype(str).str.strip().str.upper().str.zfill(5)
    )

    def find_ccs(code):
        if len(code) != 5:
            return None, None
        match = mapping_df[
            (mapping_df["start_code"] <= code) & (mapping_df["end_code"] >= code)
        ]
        if not match.empty:
            return match.iloc[0]["ccs"], match.iloc[0]["ccs_label"]
        return None, None

    results = df_result[cpt_column].map(find_ccs)
    df_result[["ccs_category", "ccs_description"]] = pd.DataFrame(
        results.tolist(), index=df_result.index
    )

    return df_result
