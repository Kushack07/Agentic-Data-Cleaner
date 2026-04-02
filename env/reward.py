import pandas as pd
from typing import Tuple

def compute_reward(action_type: str, before_df: pd.DataFrame, after_df: pd.DataFrame, error_msg: str = None) -> Tuple[float, str]:
    """
    Computes dense reward based on the action taken and its effect on the dataset.
    Returns (reward, info_string)
    """
    if error_msg:
        # Invalid / harmful action (e.g. hallucinating column names, bad parsing)
        return -2.0, f"Error executed action: {error_msg}"
        
    if before_df.equals(after_df) and action_type != "submit":
        # The action did absolutely nothing (wasted step)
        return -0.5, "Action resulted in no changes to the dataset."
        
    if action_type == "submit":
        # The agent thinks it's done. 
        # A generic 0.0 reward; the main score comes from the grader.
        return 0.0, "Submission received."

    reward = 0.0
    info = ""

    before_duplicates = before_df.duplicated().sum()
    after_duplicates = after_df.duplicated().sum()
    
    before_missing = before_df.isna().sum().sum()
    after_missing = after_df.isna().sum().sum()

    if action_type == "drop_duplicates":
        if after_duplicates < before_duplicates:
            reward += 1.0
            info = f"Successfully removed {before_duplicates - after_duplicates} duplicate rows."
        else:
            reward -= 1.0
            info = "Attempted to drop duplicates, but none were present or incorrectly executed."

    elif action_type == "fill_missing":
        if after_missing < before_missing:
            reward += 0.5
            info = f"Successfully filled {before_missing - after_missing} missing values."
        else:
            reward -= 1.0
            info = "Attempted to fill missing values, but no valid changes were made."

    elif action_type == "remove_outliers":
        if len(after_df) < len(before_df):
            # A simplistic heuristic: outliers were dropped, row count decreased.
            # Beware that dropping valid rows is bad, but for intermediate rewards, this is a proxy.
            reward += 0.3
            info = f"Removed {len(before_df) - len(after_df)} rows based on outlier detection."
        else:
            reward -= 1.0
            info = "Attempted to remove outliers, but no rows were removed."

    elif action_type == "normalize_column":
        # Assumes normalization changes data distribution without introducing nans
        if after_missing > before_missing:
            reward -= 1.0
            info = "Normalization introduced missing values. Incorrect transformation."
        else:
            reward += 0.3
            info = "Normalization applied successfully."

    elif action_type == "convert_data_type":
        if after_missing > before_missing:
             reward -= 1.0
             info = "Data type conversion introduced missing values (coercion error)."
        else:
            # We assume it was successful if it didn't create new NaNs
            reward += 0.3
            info = "Data type conversion applied successfully."

    else:
        # Unknown action
        reward -= 2.0
        info = f"Unknown action: {action_type}"

    # Global penalty for catastrophic loss of data
    if len(after_df) == 0:
         return -5.0, "Catastrophic error: All rows were deleted."
    if len(after_df.columns) < len(before_df.columns):
         return -1.0, "Catastrophic error: Columns were deleted."

    return reward, info
