import pandas as pd
from tasks.task_hard import get_ground_truth

def grade(final_dataset: pd.DataFrame) -> float:
    """
    Grades the agent's final state for the hard task.
    Returns a score between 0.0 and 1.0.
    """
    if final_dataset is None or len(final_dataset) == 0:
        return 0.0
        
    ground_truth = get_ground_truth()
    
    try:
        final_sorted = final_dataset.sort_values(by='id').reset_index(drop=True)
    except KeyError:
        return 0.0
        
    score = 0.0
    
    # 1. Missing Values (20%)
    if final_sorted.isna().sum().sum() == 0:
         score += 0.2
         
    # 2. Duplicates Removed (20%)
    if final_sorted.duplicated().sum() == 0:
         score += 0.2
         
    # 3. Formatting Parsed / Types Correct (30%)
    if pd.api.types.is_numeric_dtype(final_sorted['salary']):
         # Also check if it looks parsed safely
         if final_sorted['salary'].max() > 1000:
             score += 0.3
             
    # 4. Outliers Addressed (did they remove the extreme salaries?)
    if 'salary' in final_sorted.columns and pd.api.types.is_numeric_dtype(final_sorted['salary']):
        if final_sorted['salary'].max() < 200000: # We injected 10x
            score += 0.2
            
    # 5. Length match check (10%)
    if len(final_sorted) == len(ground_truth):
        score += 0.1
        
    return min(1.0, max(0.0, score))
