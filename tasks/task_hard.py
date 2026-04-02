import pandas as pd
import numpy as np

def generate_base_dataset(seed=42):
    """Generates a clean base dataset of 300 rows."""
    np.random.seed(seed)
    data = {
        'id': range(1, 301),
        'age': np.random.randint(20, 60, 300),
        'salary': np.random.normal(80000, 20000, 300).round(2),
        'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing', 'Finance'], 300),
        'performance_score': np.random.uniform(1.0, 5.0, 300).round(1)
    }
    return pd.DataFrame(data)

def generate_dataset(seed=300):
    """Generates a complex dataset with all issues for Task Hard."""
    np.random.seed(seed)
    df = generate_base_dataset(seed)
    
    # 1. Missing values
    df.loc[np.random.choice(df.index, size=25, replace=False), 'age'] = np.nan
    df.loc[np.random.choice(df.index, size=30, replace=False), 'salary'] = np.nan
    
    # 2. Outliers
    outlier_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_indices, 'salary'] = df.loc[outlier_indices, 'salary'] * 10 
    
    # 3. Corrupt Types & Formatting
    df['salary'] = df['salary'].astype(object)
    format_indices = np.random.choice(df.index, size=15, replace=False)
    for idx in format_indices:
        if pd.notna(df.at[idx, 'salary']):
             val = df.at[idx, 'salary']
             df.at[idx, 'salary'] = f"${val:,.2f}"
             
    # 4. Duplicates
    duplicates = df.sample(n=15, random_state=seed)
    df_messy = pd.concat([df, duplicates], ignore_index=True)
    df_messy = df_messy.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df_messy

def get_ground_truth(seed=300):
    df_messy = generate_dataset(seed)
    
    # Drop duplicates
    df_clean = df_messy.drop_duplicates(ignore_index=True)
    
    # Fix types and formatting
    def parse_salary(val):
        if pd.isna(val): return val
        if isinstance(val, str) and val.startswith('$'):
             return float(val.replace('$', '').replace(',', ''))
        return float(val)
        
    df_clean['salary'] = df_clean['salary'].apply(parse_salary)
    
    # Remove outliers (IQR method on salary)
    q1 = df_clean['salary'].quantile(0.25)
    q3 = df_clean['salary'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_clean = df_clean[(df_clean['salary'] >= lower_bound) & (df_clean['salary'] <= upper_bound)]
    df_clean = df_clean.reset_index(drop=True)
    
    # Fill missing values
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    df_clean['salary'] = df_clean['salary'].fillna(df_clean['salary'].median())
    
    return df_clean
