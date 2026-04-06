from typing import Any, Dict, Optional

from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class DataPrepAction(Action):
    action_type: str = Field(
        ...,
        description=(
            "Cleaning action to perform. Supported values: fill_missing, "
            "drop_duplicates, remove_outliers, normalize_column, "
            "convert_data_type, submit."
        ),
    )
    column: Optional[str] = Field(
        default=None,
        description="Target column for the action when applicable.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional action parameters such as fill strategy or target type.",
    )


class DataPrepObservation(Observation):
    dataset_preview: str = Field(
        ...,
        description="Head/tail preview of the current dataset in markdown format.",
    )
    missing_value_counts: Dict[str, int] = Field(
        ...,
        description="Number of missing values per column.",
    )
    duplicate_counts: int = Field(
        ...,
        description="Number of exact duplicate rows in the current dataset.",
    )
    column_types: Dict[str, str] = Field(
        ...,
        description="Current data type of each column.",
    )
    summary_statistics: Dict[str, Any] = Field(
        ...,
        description="Summary statistics for numeric columns.",
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional information about the last transition.",
    )
