import pandas as pd
import numpy as np

def generate_base_dataset(seed=42):
    """Generates a clean base dataset of 200 rows."""
    np.random.seed(seed)
    data = {
        'id': range(1, 201),
        'age': np.random.randint(20, 60, 200),
        'salary': np.random.normal(60000, 15000, 200).round(2),
        'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], 200)
    }
    return pd.DataFrame(data)

def generate_dataset(seed=200):
    """Generates a dataset with missing values and data type issues."""
    np.random.seed(seed)
    df = generate_base_dataset(seed)
    
    # 1. Inject missing values
    age_null_indices = np.random.choice(df.index, size=15, replace=False)
    salary_null_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[age_null_indices, 'age'] = np.nan
    df.loc[salary_null_indices, 'salary'] = np.nan
    
    # 2. Corrupt types (turn numeric to string with 'k')
    string_indices = np.random.choice(df.index, size=10, replace=False)
    # df['salary'] was float, we'll cast it to object by adding string values
    df['salary'] = df['salary'].astype(object)
    for idx in string_indices:
        if pd.notna(df.at[idx, 'salary']):
             val = df.at[idx, 'salary']
             df.at[idx, 'salary'] = str(int(val // 1000)) + 'k'
             
    # 3. Inject missing department
    dept_null_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[dept_null_indices, 'department'] = np.nan

    return df

def get_ground_truth(seed=200):
    """Returns perfectly clean dataframe (target output format) for grading."""
    # We load the messy one, fill appropriately. 
    # For a determinist grader, we'll define 'clean' as:
    # missing numeric -> median
    # missing category -> mode
    # corrupted type -> float
    df_messy = generate_dataset(seed)
    
    # Clean types first
    def parse_salary(val):
        if pd.isna(val): return val
        if isinstance(val, str) and val.endswith('k'):
             return float(val.replace('k', '')) * 1000
        return float(val)
        
    df_messy['salary'] = df_messy['salary'].apply(parse_salary)
    
    # Fill missing based on simple stats (excluding nans for the stat)
    age_median = df_messy['age'].median()
    df_messy['age'] = df_messy['age'].fillna(age_median)
    
    salary_median = df_messy['salary'].median()
    df_messy['salary'] = df_messy['salary'].fillna(salary_median)
    
    dept_mode = df_messy['department'].mode()[0]
    df_messy['department'] = df_messy['department'].fillna(dept_mode)
    
    return df_messy
