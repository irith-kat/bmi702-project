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


def rollup_rxnorm_to_ingredient(df: pd.DataFrame, rxnorm_column: str) -> pd.DataFrame:
    """
    Roll up RxNorm codes to ingredient-level.

    Note: Placeholder for integration with an RxNorm API or local RxNav-style
    mapping table. Currently returns the DataFrame unchanged.
    """
    # TODO: Implement RxNorm to ingredient mapping (e.g., using RxNav API or local DB)
    return df


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
