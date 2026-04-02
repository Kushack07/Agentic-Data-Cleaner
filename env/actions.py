import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Tuple

class Observation(BaseModel):
    dataset_preview: str = Field(description="First 5 and last 5 rows of the dataset in markdown format.")
    missing_value_counts: Dict[str, int] = Field(description="Number of missing values per column.")
    duplicate_counts: int = Field(description="Total number of exact duplicate rows in the dataset.")
    column_types: Dict[str, str] = Field(description="Data type of each column.")
    summary_statistics: Dict[str, Any] = Field(description="Basic statistical summaries for numeric columns (mean, std, min, max).")

class Action(BaseModel):
    action_type: str = Field(
        description="The cleaning action to perform. Must be one of: 'fill_missing', 'drop_duplicates', 'remove_outliers', 'normalize_column', 'convert_data_type', 'submit'."
    )
    column: Optional[str] = Field(
        default=None,
        description="The target column for the action. Required for all actions except 'drop_duplicates' and 'submit'."
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional parameters for the action, e.g., {'fill_value': 'mean'} for fill_missing or {'target_type': 'float'} for convert_data_type."
    )

class RewardSchema(BaseModel):
    value: float = Field(description="The scalar reward obtained from the last action.")
    reason: str = Field(description="Explanation for the reward given.")

def apply_action(df: pd.DataFrame, action: Action) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Applies the specified action to the dataframe.
    Returns the modified dataframe and an error message if the action failed.
    """
    df_copy = df.copy()
    try:
        if action.action_type == "drop_duplicates":
            df_copy = df_copy.drop_duplicates(ignore_index=True)
            return df_copy, None
            
        if action.action_type == "submit":
            return df_copy, None

        if not action.column or action.column not in df_copy.columns:
            return df_copy, f"Column '{action.column}' not found or not specified."

        col = action.column
        params = action.parameters or {}

        if action.action_type == "fill_missing":
            fill_value = params.get("fill_value", "median")
            if str(fill_value).lower() == "mean" and pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            elif str(fill_value).lower() == "median" and pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif str(fill_value).lower() == "mode":
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            else:
                df_copy[col] = df_copy[col].fillna(fill_value)

        elif action.action_type == "remove_outliers":
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                return df_copy, f"Column '{col}' is not numeric; cannot remove outliers."
            q1 = df_copy[col].quantile(0.25)
            q3 = df_copy[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
            df_copy = df_copy.reset_index(drop=True)

        elif action.action_type == "normalize_column":
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                return df_copy, f"Column '{col}' is not numeric; cannot normalize."
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            if max_val > min_val:
                df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)

        elif action.action_type == "convert_data_type":
            target_type = params.get("target_type", "float")
            try:
                if target_type == "numeric":
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                elif target_type == "datetime":
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                else:
                    df_copy[col] = df_copy[col].astype(target_type)
            except Exception as e:
                return df_copy, f"Failed to cast to '{target_type}': {str(e)}"

        else:
            return df_copy, f"Unknown action type '{action.action_type}'"

        return df_copy, None

    except Exception as e:
        return df_copy, f"Unexpected error applying action: {str(e)}"
