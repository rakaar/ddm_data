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
# Check if MT is NaN for abort trials
print("\n==== CHECKING MT VALUES FOR ABORT TRIALS ====\n")

# Create is_abort column
all_df['is_abort'] = (all_df['abort_event'] == 3).astype(int)

# Count total abort trials
total_aborts = all_df['is_abort'].sum()
print(f"Total abort trials: {total_aborts}")

# Count abort trials where MT is NaN
aborts_with_nan_mt = ((all_df['is_abort'] == 1) & (all_df['MT'].isna())).sum()
print(f"Abort trials with NaN MT: {aborts_with_nan_mt}")
print(f"Percentage of abort trials with NaN MT: {aborts_with_nan_mt / total_aborts * 100:.2f}%")

# Count abort trials where MT is not NaN
aborts_with_mt = ((all_df['is_abort'] == 1) & (~all_df['MT'].isna())).sum()
print(f"Abort trials with non-NaN MT: {aborts_with_mt}")
print(f"Percentage of abort trials with non-NaN MT: {aborts_with_mt / total_aborts * 100:.2f}%")

# %%
# Check distribution of MT values for non-abort vs abort trials
print("\n==== MT VALUE DISTRIBUTION ====\n")

# For non-abort trials
non_abort_mt = all_df[all_df['is_abort'] == 0]['MT']
print(f"Non-abort trials MT statistics:")
print(f"  Count: {len(non_abort_mt)}")
print(f"  NaN count: {non_abort_mt.isna().sum()} ({non_abort_mt.isna().mean() * 100:.2f}%)")
if non_abort_mt.notna().any():
    print(f"  Mean: {non_abort_mt.mean():.4f}")
    print(f"  Min: {non_abort_mt.min():.4f}")
    print(f"  Max: {non_abort_mt.max():.4f}")

# For abort trials
abort_mt = all_df[all_df['is_abort'] == 1]['MT']
print(f"\nAbort trials MT statistics:")
print(f"  Count: {len(abort_mt)}")
print(f"  NaN count: {abort_mt.isna().sum()} ({abort_mt.isna().mean() * 100:.2f}%)")
if abort_mt.notna().any():
    print(f"  Mean: {abort_mt.mean():.4f}")
    print(f"  Min: {abort_mt.min():.4f}")
    print(f"  Max: {abort_mt.max():.4f}")
    
    # Show a few examples of abort trials with non-NaN MT
    print("\nSample of abort trials with non-NaN MT:")
    sample_rows = all_df[(all_df['is_abort'] == 1) & (~all_df['MT'].isna())].head(5)
    print(sample_rows[['animal', 'session', 'trial', 'abort_event', 'MT']])

# %%
# Check by animal
print("\n==== MT NaN STATUS BY ANIMAL ====\n")

animal_stats = []
for animal in all_df['animal'].unique():
    animal_df = all_df[all_df['animal'] == animal]
    
    # Count abort trials
    animal_aborts = animal_df['is_abort'].sum()
    
    # Count abort trials with NaN MT
    animal_aborts_nan_mt = ((animal_df['is_abort'] == 1) & (animal_df['MT'].isna())).sum()
    
    # Calculate percentage
    if animal_aborts > 0:
        nan_percentage = animal_aborts_nan_mt / animal_aborts * 100
    else:
        nan_percentage = 0
    
    animal_stats.append({
        'animal': animal,
        'total_aborts': animal_aborts,
        'aborts_with_nan_mt': animal_aborts_nan_mt,
        'nan_percentage': nan_percentage
    })

# Create DataFrame and display
animal_stats_df = pd.DataFrame(animal_stats)
print(animal_stats_df)

# %%
# Check relationship with other columns
print("\n==== RELATIONSHIP WITH OTHER COLUMNS ====\n")

# Check if there's a pattern with other columns
columns_to_check = ['TotalFixTime', 'CNPTime', 'MT']

for col in columns_to_check:
    nan_count = ((all_df['is_abort'] == 1) & (all_df[col].isna())).sum()
    percentage = nan_count / total_aborts * 100
    print(f"{col} is NaN in {nan_count} abort trials ({percentage:.2f}%)")

# Check for correlations between NaN patterns
print("\nCorrelation between NaN patterns in abort trials:")
abort_df = all_df[all_df['is_abort'] == 1].copy()

for col1 in columns_to_check:
    for col2 in columns_to_check:
        if col1 != col2:
            # Count cases where both columns are NaN
            both_nan = ((abort_df[col1].isna()) & (abort_df[col2].isna())).sum()
            # Count cases where only first column is NaN
            only_col1_nan = ((abort_df[col1].isna()) & (~abort_df[col2].isna())).sum()
            # Count cases where only second column is NaN
            only_col2_nan = ((~abort_df[col1].isna()) & (abort_df[col2].isna())).sum()
            # Count cases where neither column is NaN
            neither_nan = ((~abort_df[col1].isna()) & (~abort_df[col2].isna())).sum()
            
            print(f"{col1} vs {col2}:")
            print(f"  Both NaN: {both_nan} ({both_nan / len(abort_df) * 100:.2f}%)")
            print(f"  Only {col1} NaN: {only_col1_nan} ({only_col1_nan / len(abort_df) * 100:.2f}%)")
            print(f"  Only {col2} NaN: {only_col2_nan} ({only_col2_nan / len(abort_df) * 100:.2f}%)")
            print(f"  Neither NaN: {neither_nan} ({neither_nan / len(abort_df) * 100:.2f}%)")
