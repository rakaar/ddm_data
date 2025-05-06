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
# Pick a random animal to analyze
animals = all_df['animal'].unique()
print(f"Available animals: {animals}")

# Let's analyze the first animal
selected_animal = animals[0]
print(f"\nAnalyzing animal: {selected_animal}")

# %%
# Prepare the dataframe for the selected animal
animal_df = all_df[all_df['animal'] == selected_animal].copy()
animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
animal_df['LED_trial'] = animal_df['LED_trial'].fillna(0).astype(int)

# Add the lagged 'is_abort' column
animal_df['abort_1'] = add_lagged_column(animal_df, 'is_abort', 1)

# %%
# Now let's analyze the dataset after filtering for aborts only
print("\nStep 1: Original dataset statistics:")
print(f"Total rows: {len(animal_df)}")
print(f"Abort rows (is_abort=1): {animal_df['is_abort'].sum()}")
print(f"Non-abort rows (is_abort=0): {(animal_df['is_abort'] == 0).sum()}")

# %%
# Check the distribution of abort_1 in the original dataset
print("\nStep 2: Distribution of abort_1 (previous trial was abort) in original dataset:")
print(animal_df['abort_1'].value_counts(dropna=False))

# %%
# Now filter for only abort trials and check the distribution again
abort_only_df = animal_df[animal_df['is_abort'] == 1].copy()
print("\nStep 3: Distribution of abort_1 in abort-only dataset:")
print(abort_only_df['abort_1'].value_counts(dropna=False))
print(f"Percentage where previous trial was abort (1): {abort_only_df['abort_1'].mean() * 100:.2f}%")
print(f"Percentage where previous trial was not abort (0): {(1 - abort_only_df['abort_1'].mean()) * 100:.2f}%")

# %%
# Check if we have any consecutive aborts (current and previous trial both aborts)
print("\nStep 4: Checking for consecutive aborts:")
consecutive_aborts = ((animal_df['is_abort'] == 1) & (animal_df['abort_1'] == 1)).sum()
total_aborts = (animal_df['is_abort'] == 1).sum()
print(f"Consecutive aborts: {consecutive_aborts} out of {total_aborts} aborts ({consecutive_aborts/total_aborts*100:.2f}%)")

# %%
# Check if there are any NaN values in abort_1 for abort trials
print("\nStep 5: Checking for NaN values in abort_1 for abort trials:")
nan_abort_1 = abort_only_df['abort_1'].isna().sum()
print(f"NaN values in abort_1 for abort trials: {nan_abort_1} out of {len(abort_only_df)} ({nan_abort_1/len(abort_only_df)*100:.2f}%)")

# %%
# Let's check what happens after we drop NaN values in the lagged columns
print("\nStep 6: After dropping NaN values in abort_1:")
abort_only_no_nan_df = abort_only_df.dropna(subset=['abort_1']).copy()
print(f"Rows remaining: {len(abort_only_no_nan_df)} out of {len(abort_only_df)} original abort rows")
print("Distribution of abort_1 after dropping NaNs:")
print(abort_only_no_nan_df['abort_1'].value_counts(dropna=False))

# %%
# Check if abort_1 is constant after all the filtering steps in the original script
print("\nStep 7: Simulating the filtering steps in glm_all_T.py:")

# Add all the lagged columns as in the original script
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

# Let's analyze what's happening in the filtering steps
# First, identify abort rows before any filtering
abort_rows_before = animal_df[animal_df['is_abort'] == 1].copy()
print(f"Abort rows before filtering: {len(abort_rows_before)}")
print("Distribution of abort_1 in abort rows before filtering:")
print(abort_rows_before['abort_1'].value_counts(dropna=False))

# Now check what happens after dropping NaN values
after_dropna = animal_df.dropna(subset=lag_cols).copy()
abort_rows_after_dropna = after_dropna[after_dropna['is_abort'] == 1].copy()
print("\nAbort rows after dropping NaN values:", len(abort_rows_after_dropna))
print("Distribution of abort_1 in abort rows after dropping NaN values:")
print(abort_rows_after_dropna['abort_1'].value_counts(dropna=False))

# Check which abort rows were removed by the dropna operation
removed_abort_rows = len(abort_rows_before) - len(abort_rows_after_dropna)
print("\nNumber of abort rows removed by dropna:", removed_abort_rows, 
      f"({removed_abort_rows/len(abort_rows_before)*100:.2f}%)")

# Check if rows where abort_1 == 1 were disproportionately removed
abort_followed_by_abort_before = ((abort_rows_before['is_abort'] == 1) & (abort_rows_before['abort_1'] == 1)).sum()
abort_followed_by_abort_after = ((abort_rows_after_dropna['is_abort'] == 1) & (abort_rows_after_dropna['abort_1'] == 1)).sum()
removed_abort_followed_by_abort = abort_followed_by_abort_before - abort_followed_by_abort_after

print(f"Abort rows where previous trial was also abort (before dropna): {abort_followed_by_abort_before}")
print(f"Abort rows where previous trial was also abort (after dropna): {abort_followed_by_abort_after}")
print(f"Removed abort rows where previous trial was also abort: {removed_abort_followed_by_abort}", 
      f"({removed_abort_followed_by_abort/abort_followed_by_abort_before*100:.2f}% of such rows)")

# Check which columns have the most NaN values in abort rows
print("\nColumns with most NaN values in abort rows (top 5):")
nan_counts = abort_rows_before[lag_cols].isna().sum().sort_values(ascending=False)
print(nan_counts.head())

# Apply the same filtering as in glm_all_T.py to get the final result
filtered_df = after_dropna[after_dropna['is_abort'] == 1].copy()

print("\nFinal rows after filtering:", len(filtered_df), "out of", len(animal_df), "original rows")

# Check if abort_1 is constant
unique_abort_1 = filtered_df['abort_1'].nunique()
print(f"Number of unique values in abort_1 after filtering: {unique_abort_1}")
if unique_abort_1 <= 1:
    print(f"abort_1 is CONSTANT with value: {filtered_df['abort_1'].iloc[0]}")
else:
    print("abort_1 is NOT constant. Values:")
    print(filtered_df['abort_1'].value_counts())

# %%
# Check other columns that might be constant
print("\nStep 8: Checking all columns for constant values:")
constant_cols = [col for col in lag_cols if filtered_df[col].nunique() <= 1]
if constant_cols:
    print(f"Constant columns: {constant_cols}")
    for col in constant_cols:
        print(f"  {col} has constant value: {filtered_df[col].iloc[0]}")
else:
    print("No constant columns found.")

# %%
# Analyze the sessions to understand why this might be happening
print("\nStep 9: Session analysis:")
session_stats = []
for session in filtered_df['session'].unique():
    session_df = filtered_df[filtered_df['session'] == session]
    session_stats.append({
        'session': session,
        'total_rows': len(session_df),
        'abort_1_values': session_df['abort_1'].unique(),
        'abort_1_is_constant': session_df['abort_1'].nunique() <= 1
    })

session_stats_df = pd.DataFrame(session_stats)
print(f"Sessions with constant abort_1: {session_stats_df['abort_1_is_constant'].sum()} out of {len(session_stats_df)}")

# %%
# Look at the first few trials of each session to understand better
print("\nStep 10: Looking at the first few trials of each session:")
for session in animal_df['session'].unique()[:3]:  # Just look at first 3 sessions
    session_full_df = animal_df[animal_df['session'] == session].sort_values('trial')
    print(f"\nSession {session} first 10 trials:")
    print(session_full_df[['trial', 'is_abort', 'abort_1']].head(10))
