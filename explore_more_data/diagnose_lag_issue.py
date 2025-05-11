# Script to diagnose why lagged predictors become NaN for animals after the first one

import pandas as pd
import numpy as np

# Load the data
exp_df = pd.read_csv('/home/rlab/raghavendra/ddm_data/outExp.csv')

# Remove wrong rows as in the original script
exp_df = exp_df[~((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3))].copy()

# Comparable batch
all_df = exp_df[(exp_df['batch_name'] == 'Comparable')].copy()

# Define session and trial column names
SESSION_COL = 'session'
TRIAL_COL = 'trial'

# Function to add lagged columns (same as in glm_v3.py)
def add_k_previous_trial_value_explicit_lookup(df, value_col_name, k, session_col='session', trial_col='trial'):
    working_df = df.copy()
    new_col_name = f"{value_col_name}_{k}"
    
    working_df['__target_prev_trial_id__'] = working_df[trial_col] - k
    
    df_source_values = df[[session_col, trial_col, value_col_name]].copy()
    df_source_values = df_source_values.rename(columns={
        value_col_name: new_col_name, 
        trial_col: '__source_actual_trial_id__'
    })
    
    df_source_values = df_source_values.drop_duplicates(
        subset=[session_col, '__source_actual_trial_id__'], 
        keep='first'
    )
    
    merged_df = pd.merge(
        working_df,
        df_source_values,
        left_on=[session_col, '__target_prev_trial_id__'],
        right_on=[session_col, '__source_actual_trial_id__'],
        how='left'
    )
    
    columns_to_drop = ['__target_prev_trial_id__']
    if '__source_actual_trial_id__' in merged_df.columns:
        columns_to_drop.append('__source_actual_trial_id__')
    
    result_df = merged_df.drop(columns=columns_to_drop)
    
    return result_df

# Process data for each animal separately and check for NaN issues
def process_animal(animal_id, T_max_lag=1):
    print(f"\n=== Processing Animal {animal_id}, T_max_lag={T_max_lag} ===\n")
    
    # Filter for the specific animal
    animal_df = all_df[all_df['animal'] == animal_id].copy()
    
    # Basic stats
    print(f"Total rows for animal {animal_id}: {len(animal_df)}")
    
    # Create derived columns
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
    animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
    animal_df['abs_ILD'] = animal_df['ILD'].abs()
    
    # Normalize trial number within each session
    animal_df['norm_trial'] = animal_df.groupby(SESSION_COL)[TRIAL_COL].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
    )
    
    # Create lagged variables
    lagged_vars_to_create = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']
    
    # Check if all base columns exist
    for base_var_name in lagged_vars_to_create:
        if base_var_name not in animal_df.columns:
            print(f"Warning: Base column '{base_var_name}' not found in animal_df for animal {animal_id}")
    
    # Track lagged columns created
    lagged_columns = []
    
    for k_lag in range(1, T_max_lag + 1):
        for base_var_name in lagged_vars_to_create:
            # Determine the final name for the new lagged column
            final_lagged_col_name = f'{base_var_name}_{k_lag}'
            if base_var_name == 'is_abort':
                final_lagged_col_name = f'abort_{k_lag}'
            
            # Skip if base column doesn't exist
            if base_var_name not in animal_df.columns:
                print(f"Skipping lag for '{base_var_name}' as it doesn't exist")
                animal_df[final_lagged_col_name] = np.nan
                lagged_columns.append(final_lagged_col_name)
                continue
            
            # Create the lagged column
            temp_df_with_lag = add_k_previous_trial_value_explicit_lookup(
                df=animal_df, 
                value_col_name=base_var_name, 
                k=k_lag,
                session_col=SESSION_COL,
                trial_col=TRIAL_COL
            )
            
            # Extract the new column and add it to animal_df with the correct final name
            created_col_name_by_func = f"{base_var_name}_{k_lag}"
            if created_col_name_by_func in temp_df_with_lag.columns:
                animal_df[final_lagged_col_name] = temp_df_with_lag[created_col_name_by_func]
                lagged_columns.append(final_lagged_col_name)
            else:
                print(f"Warning: Expected new column '{created_col_name_by_func}' not found after lagging.")
                animal_df[final_lagged_col_name] = np.nan
                lagged_columns.append(final_lagged_col_name)
    
    # Define predictor columns
    current_predictor_cols = []
    for k_lag in range(1, T_max_lag + 1):
        for var_name_prefix in ['rewarded', 'abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']:
            current_predictor_cols.append(f'{var_name_prefix}_{k_lag}')
    
    # Add norm_trial as a predictor
    current_predictor_cols.append('norm_trial')
    
    # Check NaN values in lagged columns
    lagged_nan_counts = animal_df[lagged_columns].isna().sum()
    print("\nNaN counts in lagged columns:")
    for col, count in lagged_nan_counts.items():
        print(f"{col}: {count} NaNs ({count/len(animal_df)*100:.2f}%)")
    
    # Count rows with NaN in any predictor column
    rows_with_nan = animal_df[current_predictor_cols].isna().any(axis=1).sum()
    print(f"\nRows with NaN in any predictor column: {rows_with_nan} ({rows_with_nan/len(animal_df)*100:.2f}%)")
    
    # Count rows that are abort trials
    abort_rows = (animal_df['is_abort'] == 1).sum()
    print(f"Abort trials: {abort_rows} ({abort_rows/len(animal_df)*100:.2f}%)")
    
    # Check rows that would be kept after filtering
    animal_df_cleaned = animal_df.dropna(subset=current_predictor_cols).reset_index(drop=True)
    print(f"Rows after NaN drop: {len(animal_df_cleaned)} ({len(animal_df_cleaned)/len(animal_df)*100:.2f}%)")
    
    # Filter for abort trials
    animal_df_cleaned = animal_df_cleaned[animal_df_cleaned['is_abort'] == 1].copy()
    print(f"Rows after NaN drop and abort filter: {len(animal_df_cleaned)} ({len(animal_df_cleaned)/len(animal_df)*100:.2f}%)")
    
    # Check if intended_fix column exists
    if 'intended_fix' not in animal_df.columns:
        print(f"\nWARNING: 'intended_fix' column does not exist for animal {animal_id}")
        # Check all columns to see what might be similar
        print(f"Available columns: {animal_df.columns.tolist()}")

# Process each animal for T_max_lag=1
for animal_id in sorted(all_df['animal'].unique()):
    process_animal(animal_id, T_max_lag=1)

# Test the fix: Process with a modified function that includes animal_id in the join
def add_k_previous_trial_value_fixed(df, value_col_name, k, animal_col='animal', session_col='session', trial_col='trial'):
    working_df = df.copy()
    new_col_name = f"{value_col_name}_{k}"
    
    working_df['__target_prev_trial_id__'] = working_df[trial_col] - k
    
    df_source_values = df[[animal_col, session_col, trial_col, value_col_name]].copy()
    df_source_values = df_source_values.rename(columns={
        value_col_name: new_col_name, 
        trial_col: '__source_actual_trial_id__'
    })
    
    df_source_values = df_source_values.drop_duplicates(
        subset=[animal_col, session_col, '__source_actual_trial_id__'], 
        keep='first'
    )
    
    merged_df = pd.merge(
        working_df,
        df_source_values,
        left_on=[animal_col, session_col, '__target_prev_trial_id__'],
        right_on=[animal_col, session_col, '__source_actual_trial_id__'],
        how='left'
    )
    
    columns_to_drop = ['__target_prev_trial_id__']
    if '__source_actual_trial_id__' in merged_df.columns:
        columns_to_drop.append('__source_actual_trial_id__')
    
    result_df = merged_df.drop(columns=columns_to_drop)
    
    return result_df

print("\n\n=== Testing the fixed function ===\n")

# Process the first two animals with the fixed function
animals_to_test = sorted(all_df['animal'].unique())[:2]
for animal_id in animals_to_test:
    print(f"\n=== Processing Animal {animal_id} with fixed function ===\n")
    
    # Filter for the specific animal
    animal_df = all_df[all_df['animal'] == animal_id].copy()
    
    # Create derived columns
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
    
    # Test with just one lagged variable
    base_var_name = 'is_abort'
    k_lag = 1
    final_lagged_col_name = f'abort_{k_lag}'
    
    # Create the lagged column with the fixed function
    temp_df_with_lag = add_k_previous_trial_value_fixed(
        df=animal_df, 
        value_col_name=base_var_name, 
        k=k_lag,
        animal_col='animal',
        session_col=SESSION_COL,
        trial_col=TRIAL_COL
    )
    
    # Extract the new column and add it to animal_df with the correct final name
    created_col_name_by_func = f"{base_var_name}_{k_lag}"
    if created_col_name_by_func in temp_df_with_lag.columns:
        animal_df[final_lagged_col_name] = temp_df_with_lag[created_col_name_by_func]
        # Check NaN values
        nan_count = animal_df[final_lagged_col_name].isna().sum()
        print(f"{final_lagged_col_name}: {nan_count} NaNs ({nan_count/len(animal_df)*100:.2f}%)")
    else:
        print(f"Warning: Expected new column '{created_col_name_by_func}' not found after lagging.")
