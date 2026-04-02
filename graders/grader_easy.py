import pandas as pd
from tasks.task_easy import get_ground_truth

def grade(final_dataset: pd.DataFrame) -> float:
    """
    Grades the agent's final state for the easy task.
    Returns a score between 0.0 and 1.0.
    """
    if final_dataset is None or len(final_dataset) == 0:
        return 0.0
        
    ground_truth = get_ground_truth()
    
    # Sort both to allow direct comparison regardless of row order
    gt_sorted = ground_truth.sort_values(by='id').reset_index(drop=True)
    
    try:
        final_sorted = final_dataset.sort_values(by='id').reset_index(drop=True)
    except KeyError:
        return 0.0 # Missing the ID column means they deleted something important
        
    if len(final_sorted) != len(gt_sorted):
        # Penalty for wrong number of rows
        max_len = max(len(final_sorted), len(gt_sorted))
        diff = abs(len(final_sorted) - len(gt_sorted))
        score = max(0.0, 1.0 - (diff / max_len))
        # Cap score at 0.5 if lengths don't match exactly
        return min(score, 0.5)
        
    # Check if dataframes match perfectly
    try:
        if final_sorted.equals(gt_sorted):
            return 1.0
        else:
            # Partial score if sizes match but values don't
            matches = (final_sorted == gt_sorted).sum().sum()
            total = gt_sorted.size
            return float(matches / total)
    except Exception:
        return 0.0
