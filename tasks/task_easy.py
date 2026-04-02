import pandas as pd
import numpy as np

def generate_base_dataset(seed=42):
    """Generates a clean base dataset of 100 rows."""
    np.random.seed(seed)
    data = {
        'id': range(1, 101),
        'age': np.random.randint(20, 60, 100),
        'salary': np.random.normal(60000, 15000, 100).round(2),
        'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], 100)
    }
    return pd.DataFrame(data)

def generate_dataset(seed=100):
    """Generates a dataset with only duplicate issues for Task Easy."""
    df = generate_base_dataset(seed)
    
    # Inject 10 exact duplicate rows
    duplicates = df.sample(n=10, random_state=seed)
    df_messy = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle so they aren't all at the end
    df_messy = df_messy.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_messy

def get_ground_truth(seed=100):
    """Returns the perfectly clean version of the easy dataset."""
    df = generate_dataset(seed)
    df_clean = df.drop_duplicates(ignore_index=True)
    # The clean one should exactly match the sorted unique values
    df_clean = df_clean.sort_values(by='id').reset_index(drop=True)
    return df_clean
