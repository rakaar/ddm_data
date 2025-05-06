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

# Print available columns
print("\nAvailable columns in the dataframe:")
print(all_df.columns.tolist())

# Check for 'rewarded' column specifically
if 'rewarded' in all_df.columns:
    print("\n'rewarded' column exists")
else:
    print("\n'rewarded' column does NOT exist")
    # Look for similarly named columns
    reward_cols = [col for col in all_df.columns if 'reward' in col.lower()]
    print(f"Columns related to reward: {reward_cols}")

# %%
# Check for the first animal
animal = all_df['animal'].unique()[0]
print(f"\nChecking columns for animal {animal}")
animal_df = all_df[all_df['animal'] == animal].copy()

# Print the first few rows to see what's available
print("\nFirst few rows of animal dataframe:")
print(animal_df.head())

# Check for abort_event and related columns
print("\nUnique values of abort_event:")
print(animal_df['abort_event'].unique())

# Check if we need to create the is_abort column
if 'is_abort' not in animal_df.columns:
    print("\nCreating is_abort column from abort_event")
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    print(f"is_abort value counts: {animal_df['is_abort'].value_counts()}")
