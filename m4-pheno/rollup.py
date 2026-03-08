import pandas as pd
import os


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

    # Load mapping - focusing on ICD and Phecode columns
    map_df = pd.read_csv(
        mapping_file,
        usecols=["ICD", "Phecode", "PhecodeString"],
        dtype={"ICD": str, "Phecode": str},
    )

    # Merge with input data
    # We use a left join to keep all original rows
    result_df = df.merge(map_df, left_on=icd_column, right_on="ICD", how="left")

    # Clean up the extra 'ICD' column from merge if it's different from icd_column
    if icd_column != "ICD":
        result_df.drop(columns=["ICD"], inplace=True)

    return result_df


def rollup_rxnorm_to_ingredient(df: pd.DataFrame, rxnorm_column: str) -> pd.DataFrame:
    """
    Roll up RxNorm codes to ingredient-level.

    Note: This currently serves as a placeholder for integration with an RxNorm API
     или a local RxNav-style mapping table.
    """
    # TODO: Implement RxNorm to Ingredient mapping (e.g., using RxNav API or local DB)
    return df


def rollup_cpt_to_ccs(
    df: pd.DataFrame,
    cpt_column: str,
    mapping_file: str = "CCS_Services_Procedures_v2025-1.csv",
) -> pd.DataFrame:
    """
    Rolls up a column of CPT/HCPCS codes to their CCS categories using
    alphanumeric range matching from the AHRQ mapping file.

    Args:
        df (pd.DataFrame): The research cohort DataFrame (e.g., from M4).
        cpt_column (str): The name of the column containing raw CPT codes.
        mapping_file (str): Path to the AHRQ CSV mapping file.

    Returns:
        pd.DataFrame: The original DataFrame with 'ccs_category' and
                      'ccs_description' columns added.
    """
    # 1. Load the AHRQ translation table
    # The file contains ranges for 5-character alphanumeric strings [cite: 868, 912]
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    mapping_df = pd.read_csv(mapping_file)

    # Standardize column names from the mapping file for consistent access
    # Assumes headers like 'start_code', 'end_code', 'ccs_category', 'ccs_description'
    mapping_df.columns = [c.lower().replace(" ", "_") for c in mapping_df.columns]

    # Ensure range boundary columns are 5-char strings — pandas may infer them as int64
    # for purely numeric ranges (e.g. 00100 → 100), which breaks the string comparison.
    mapping_df["start_code"] = mapping_df["start_code"].astype(str).str.zfill(5)
    mapping_df["end_code"] = mapping_df["end_code"].astype(str).str.zfill(5)

    # 2. Pre-process the input column
    # CPT codes must be strings and 5 characters long for correct range comparison [cite: 912]
    df_result = df.copy()
    df_result[cpt_column] = (
        df_result[cpt_column].astype(str).str.strip().str.upper().str.zfill(5)
    )

    def find_ccs(code):
        if len(code) != 5:
            return None, None

        # Perform alphanumeric range check: Start_Code <= Code <= End_Code [cite: 879]
        # This handles Category II (ending in F) and III (ending in T) correctly [cite: 910]
        match = mapping_df[
            (mapping_df["start_code"] <= code) & (mapping_df["end_code"] >= code)
        ]

        if not match.empty:
            # Guidelines state specific categories take precedence over general ones [cite: 818]
            return match.iloc[0]["ccs_category"], match.iloc[0][
                "ccs_category_description"
            ]
        return None, None

    # 3. Apply the mapping logic
    # We use a vectorized-friendly zip/map approach to populate both new columns
    results = df_result[cpt_column].map(find_ccs)
    df_result[["ccs_category", "ccs_description"]] = pd.DataFrame(
        results.tolist(), index=df.index
    )

    return df_result
