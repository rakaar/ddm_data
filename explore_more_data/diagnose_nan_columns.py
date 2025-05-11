# Script to diagnose which columns are causing NaN issues in the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Load the data
exp_df = pd.read_csv('/home/rlab/raghavendra/ddm_data/outExp.csv')

# Remove wrong rows as in the original script
count = ((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3)).sum()
print(f"Number of rows where TotalFixTime is NaN and abort_event == 3: {count}")
exp_df = exp_df[~((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3))].copy()

# Comparable batch
all_df = exp_df[(exp_df['batch_name'] == 'Comparable')].copy()

# Function to add lagged columns (simplified from original)
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

# Create a PDF to save diagnostic plots
pdf = PdfPages('nan_diagnostics.pdf')

# Function to analyze NaN values for a specific animal and T_max_lag
def analyze_animal_nans(animal_id, T_max_lag):
    print(f"\n=== Analyzing Animal {animal_id}, T_max_lag={T_max_lag} ===\n")
    
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
    animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
    )
    
    # Check NaN values in base columns
    base_columns = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']
    base_nan_counts = animal_df[base_columns].isna().sum()
    print("\nNaN counts in base columns:")
    for col, count in base_nan_counts.items():
        print(f"{col}: {count} NaNs ({count/len(animal_df)*100:.2f}%)")
    
    # Create lagged variables
    lagged_vars_to_create = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']
    
    # Track lagged columns created
    lagged_columns = []
    
    for k_lag in range(1, T_max_lag + 1):
        for base_var_name in lagged_vars_to_create:
            # Determine the final name for the new lagged column
            final_lagged_col_name = f'{base_var_name}_{k_lag}'
            if base_var_name == 'is_abort':
                final_lagged_col_name = f'abort_{k_lag}'
            
            # Create the lagged column
            temp_df_with_lag = add_k_previous_trial_value_explicit_lookup(
                df=animal_df, 
                value_col_name=base_var_name, 
                k=k_lag
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
    
    # Create a heatmap of NaN values
    plt.figure(figsize=(12, 8))
    sns.heatmap(animal_df[current_predictor_cols].isna(), cmap='viridis', cbar=False)
    plt.title(f"NaN values in predictor columns - Animal {animal_id}, T_max_lag={T_max_lag}")
    plt.xlabel("Predictor Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Create a bar chart of NaN percentages
    plt.figure(figsize=(14, 8))
    nan_percentages = (animal_df[current_predictor_cols].isna().sum() / len(animal_df) * 100).sort_values(ascending=False)
    sns.barplot(x=nan_percentages.index, y=nan_percentages.values)
    plt.title(f"Percentage of NaN values per predictor column - Animal {animal_id}, T_max_lag={T_max_lag}")
    plt.xlabel("Predictor Columns")
    plt.ylabel("Percentage of NaN values")
    plt.xticks(rotation=90)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Analyze which combinations of columns are causing the most NaNs
    # Focus on rows that are abort trials
    abort_df = animal_df[animal_df['is_abort'] == 1].copy()
    
    # Check which combinations of NaN columns are most common
    nan_patterns = abort_df[current_predictor_cols].isna().apply(lambda x: tuple(x), axis=1)
    pattern_counts = nan_patterns.value_counts().head(10)  # Top 10 patterns
    
    print("\nTop NaN patterns in abort trials:")
    for pattern, count in pattern_counts.items():
        cols_with_nan = [current_predictor_cols[i] for i, is_nan in enumerate(pattern) if is_nan]
        if cols_with_nan:  # Only print patterns with at least one NaN
            print(f"Pattern with {count} occurrences ({count/len(abort_df)*100:.2f}%):")
            print(f"  Columns with NaN: {', '.join(cols_with_nan)}")
    
    # Return the cleaned dataframe size for summary
    return len(animal_df_cleaned)

# Analyze each animal for each T_max_lag
results = []

for T_max_lag in [1, 2, 3]:
    for animal_id in all_df['animal'].unique():
        rows_after_filtering = analyze_animal_nans(animal_id, T_max_lag)
        results.append({
            'animal': animal_id,
            'T_max_lag': T_max_lag,
            'rows_after_filtering': rows_after_filtering
        })

# Create a summary table
results_df = pd.DataFrame(results)
pivot_df = results_df.pivot(index='animal', columns='T_max_lag', values='rows_after_filtering')
pivot_df.columns = [f'T_max_lag={col}' for col in pivot_df.columns]

print("\n=== Summary of rows remaining after filtering ===\n")
print(pivot_df)

# Create a summary plot
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, annot=True, fmt='g', cmap='YlGnBu')
plt.title('Number of rows remaining after NaN drop and abort filter')
plt.tight_layout()
pdf.savefig()
plt.close()

# Close the PDF
pdf.close()

print("\nDiagnostic results saved to nan_diagnostics.pdf")
