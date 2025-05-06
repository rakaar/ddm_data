# %%
import numpy as np
import pandas as pd

# Load the data
exp_df = pd.read_csv('../outExp.csv')

# Remove wrong rows as in the original script
count = ((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3)).sum()
print("Number of rows where TotalFixTime is NaN and abort_event == 3:", count)
exp_df = exp_df[~((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3))].copy()

# Comparable batch
all_df = exp_df[(exp_df['batch_name'] == 'Comparable')]

# %%
# Function to add lagged column (same as in glm_all_T.py)
def add_lagged_column(df, col, k):
    def get_lagged(session_df):
        trial_to_val = session_df.set_index('trial')[col]
        return session_df['trial'].map(lambda t: trial_to_val.get(t - k, np.nan))
    return df.groupby('session', group_keys=False).apply(get_lagged)

# %%
# Analyze all animals
print("\n==== ANALYZING ALL ANIMALS ====\n")

all_animals_summary = []

for animal in all_df['animal'].unique():
    print(f"\nAnalyzing animal: {animal}")
    
    # Prepare the dataframe for the animal
    animal_df = all_df[all_df['animal'] == animal].copy()
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
    animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
    animal_df['LED_trial'] = animal_df['LED_trial'].fillna(0).astype(int)
    
    # Add all the lagged columns
    T = 1  # We're only looking at lag 1 for simplicity
    lagged_vars = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'ILD', 'ABL', 'LED_trial', 'TotalFixTime', 'CNPTime', 'MT']
    for k in range(1, T + 1):
        for var in lagged_vars:
            colname = f'{var}_{k}' if var != 'is_abort' else f'abort_{k}'
            base_col = var
            animal_df[colname] = add_lagged_column(animal_df, base_col, k)
    
    # Get the lag column names
    lag_cols = [f'{var}_{k}' for k in range(1, T + 1)
                for var in ['rewarded', 'abort', 'short_poke', 'intended_fix', 'ILD', 'ABL', 'LED_trial', 'TotalFixTime', 'CNPTime', 'MT']]
    
    # Identify abort rows before any filtering
    abort_rows_before = animal_df[animal_df['is_abort'] == 1].copy()
    
    # Count rows where previous trial was abort
    abort_followed_by_abort = ((abort_rows_before['is_abort'] == 1) & (abort_rows_before['abort_1'] == 1)).sum()
    
    # Get rows where abort_1 = 1
    abort_1_rows = abort_rows_before[abort_rows_before['abort_1'] == 1].copy()
    
    # Check which of these rows would be removed by dropna
    abort_1_rows_with_nan = abort_1_rows[abort_1_rows[lag_cols].isna().any(axis=1)].copy()
    
    # Count NaN values in each column for these rows
    nan_counts = abort_1_rows_with_nan[lag_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    
    # Calculate percentages
    nan_percentages = (nan_counts / len(abort_1_rows_with_nan) * 100).round(2)
    
    # Create a summary of which columns are causing rows to be dropped
    nan_summary = pd.DataFrame({
        'count': nan_counts,
        'percentage': nan_percentages
    })
    
    # Add to summary for all animals
    all_animals_summary.append({
        'animal': animal,
        'total_abort_rows': len(abort_rows_before),
        'abort_1_rows': len(abort_1_rows),
        'abort_1_rows_with_nan': len(abort_1_rows_with_nan),
        'percentage_removed': round(len(abort_1_rows_with_nan) / len(abort_1_rows) * 100, 2) if len(abort_1_rows) > 0 else 0,
        'nan_columns': nan_summary
    })
    
    # Print summary for this animal
    print(f"Total abort rows: {len(abort_rows_before)}")
    print(f"Rows where previous trial was abort (abort_1 = 1): {len(abort_1_rows)}")
    print(f"Rows with abort_1 = 1 that would be removed by dropna: {len(abort_1_rows_with_nan)}")
    print(f"Percentage of abort_1 = 1 rows removed: {round(len(abort_1_rows_with_nan) / len(abort_1_rows) * 100, 2)}% (if any)")
    
    if len(abort_1_rows_with_nan) > 0:
        print("\nColumns causing rows to be dropped (NaN counts):")
        print(nan_summary)

# %%
# Print overall summary
print("\n==== OVERALL SUMMARY ====\n")

# Create a DataFrame from the summary
summary_df = pd.DataFrame([
    {
        'animal': item['animal'],
        'total_abort_rows': item['total_abort_rows'],
        'abort_1_rows': item['abort_1_rows'],
        'abort_1_rows_with_nan': item['abort_1_rows_with_nan'],
        'percentage_removed': item['percentage_removed']
    } for item in all_animals_summary
])

print("Summary statistics for all animals:")
print(summary_df)

print("\nAverage percentage of abort_1 = 1 rows removed across all animals:")
print(f"{summary_df['percentage_removed'].mean():.2f}%")

# %%
# Analyze which columns are most commonly causing rows to be dropped
print("\n==== COLUMNS CAUSING ROWS TO BE DROPPED ====\n")

# Combine nan_columns from all animals
all_nan_counts = pd.Series(dtype='int64')

for item in all_animals_summary:
    if len(item['nan_columns']) > 0:
        all_nan_counts = all_nan_counts.add(item['nan_columns']['count'], fill_value=0)

# Sort and display
all_nan_counts = all_nan_counts.sort_values(ascending=False)
print("NaN counts across all animals:")
print(all_nan_counts)

# Calculate percentages relative to total removed rows
total_removed_rows = summary_df['abort_1_rows_with_nan'].sum()
all_nan_percentages = (all_nan_counts / total_removed_rows * 100).round(2)

print("\nPercentage of removed rows with NaN in each column:")
print(all_nan_percentages)

# %%
# Create a more detailed explanation
print("\n==== DETAILED EXPLANATION ====\n")

print(f"Total abort_1 = 1 rows across all animals: {summary_df['abort_1_rows'].sum()}")
print(f"Total abort_1 = 1 rows removed: {summary_df['abort_1_rows_with_nan'].sum()} ({summary_df['abort_1_rows_with_nan'].sum() / summary_df['abort_1_rows'].sum() * 100:.2f}%)")

# Get top 3 columns causing removals
top_columns = all_nan_counts.head(3).index.tolist()
top_counts = all_nan_counts.head(3).values
top_percentages = all_nan_percentages.head(3).values

print("\nMain columns causing rows to be removed:")
for i, col in enumerate(top_columns):
    print(f"{col}: {top_counts[i]} rows ({top_percentages[i]}% of removed rows)")

print("\nSummary explanation:")
print(f"{summary_df['abort_1_rows_with_nan'].sum()} rows where abort_1 = 1 were removed because:")

for i, col in enumerate(top_columns):
    print(f"- {col} was NaN in {top_counts[i]} rows ({top_percentages[i]}% of removed rows)")
