import pandas as pd

def extract_cuis(notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract clinical concepts (CUIs) from raw clinical notes.
    
    This function processes a DataFrame of notes (usually from M4) and returns 
    a 'long-format' DataFrame of patient IDs and their associated CUIs.
    
    Args:
        notes_df (pd.DataFrame): DataFrame with at least columns for 'patient_id' and 'note_text'.
        
    Returns:
        pd.DataFrame: A long-format DataFrame with columns like ['patient_id', 'cui'].
    """
    # TODO: Implement NILE logic for CUI extraction
    # This should interface with the NILE algorithm
    results = pd.DataFrame(columns=['patient_id', 'cui'])
    return results
