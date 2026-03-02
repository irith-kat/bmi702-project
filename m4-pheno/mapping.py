import pandas as pd
import os

def rollup_icd_to_phecode(df: pd.DataFrame, icd_column: str, mapping_file: str = "Phecode_map_v1_2_icd9_icd10cm.csv") -> pd.DataFrame:
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
    map_df = pd.read_csv(mapping_file, usecols=['ICD', 'Phecode', 'PhecodeString'], dtype={'ICD': str, 'Phecode': str})
    
    # Merge with input data
    # We use a left join to keep all original rows
    result_df = df.merge(map_df, left_on=icd_column, right_on='ICD', how='left')
    
    # Clean up the extra 'ICD' column from merge if it's different from icd_column
    if icd_column != 'ICD':
        result_df.drop(columns=['ICD'], inplace=True)
        
    return result_df

