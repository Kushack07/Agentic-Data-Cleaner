import pandas as pd
from tasks.task_medium import get_ground_truth

def grade(final_dataset: pd.DataFrame) -> float:
    """
    Grades the agent's final state for the medium task.
    Returns a score between 0.0 and 1.0.
    """
    if final_dataset is None or len(final_dataset) == 0:
        return 0.0
        
    ground_truth = get_ground_truth()
    
    # Sort both to allow direct comparison
    gt_sorted = ground_truth.sort_values(by='id').reset_index(drop=True)
    
    try:
        final_sorted = final_dataset.sort_values(by='id').reset_index(drop=True)
    except KeyError:
        return 0.0 # ID missing
        
    score = 0.0
    
    # 1. Did they maintain the shape mostly?
    if len(final_sorted) == len(gt_sorted):
         score += 0.2
         
    # 2. Check types
    if pd.api.types.is_numeric_dtype(final_sorted['salary']):
         score += 0.2
         
    # 3. Check NaNs
    if final_sorted.isna().sum().sum() == 0:
         score += 0.3
         
    # 4. Check actual values (normalized distance relative to gt)
    try:
        # Match scalar values where possible
        matches = (final_sorted == gt_sorted).sum().sum()
        total = gt_sorted.size
        value_score = (matches / total) * 0.3
        score += value_score
    except Exception:
        pass
        
    return min(1.0, max(0.0, score))
