import os

import pandas as pd


def rollup_icd_to_phecode(
    df: pd.DataFrame,
    icd_column: str,
    mapping_file: str = "Phecode_map_v1_2_icd9_icd10cm.csv",
) -> pd.DataFrame:
    """
    Map ICD-9/10 codes to PheCodes using the standard mapping file.

    Args:
        df (pd.DataFrame): Input DataFrame containing ICD codes.
        icd_column (str): Name of the column containing ICD codes.
        mapping_file (str): Path to the Phecode mapping CSV.

    Returns:
        pd.DataFrame: DataFrame with added 'Phecode' and 'PhecodeString' columns.
    """
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    def _add_icd_dot(code: str) -> str:
        """Insert a '.' at position 3 if the ICD code does not already contain one.
        ICD codes in raw MIMIC-IV data are stored without the dot."""
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

    ndc_map = pd.read_csv(
        ndc_mapping_file,
        dtype={"ndc": str, "ingredient_id": str, "ingredient_name": str},
    ).rename(
        columns={
            "ingredient_id": "rxnorm_ingredient_id",
            "ingredient_name": "rxnorm_ingredient_name",
        }
    )

    result = df.copy()
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

    mapping_df = pd.read_csv(mapping_file)
    mapping_df.columns = [c.lower().replace(" ", "_") for c in mapping_df.columns]

    # Ensure range boundary columns are 5-char strings — pandas may infer purely
    # numeric ranges (e.g. 00100 → 100) as int64, which breaks string comparison.
    mapping_df["start_code"] = mapping_df["start_code"].astype(str).str.zfill(5)
    mapping_df["end_code"] = mapping_df["end_code"].astype(str).str.zfill(5)

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
            return match.iloc[0]["ccs_category"], match.iloc[0][
                "ccs_category_description"
            ]
        return None, None

    results = df_result[cpt_column].map(find_ccs)
    df_result[["ccs_category", "ccs_description"]] = pd.DataFrame(
        results.tolist(), index=df_result.index
    )

    return df_result
