# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# Check if LED_trial is always 0 before abort trials
print("\n==== CHECKING IF LED_TRIAL IS ALWAYS 0 BEFORE ABORT TRIALS ====\n")

results = []

for animal in all_df['animal'].unique():
    print(f"\nAnalyzing animal: {animal}")
    
    # Prepare the dataframe for the animal
    animal_df = all_df[all_df['animal'] == animal].copy()
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    animal_df['LED_trial'] = animal_df['LED_trial'].fillna(0).astype(int)
    
    # Add previous trial's LED_trial value
    animal_df['prev_LED_trial'] = add_lagged_column(animal_df, 'LED_trial', 1)
    
    # Filter to only keep abort trials
    abort_df = animal_df[animal_df['is_abort'] == 1].copy()
    
    # Check the distribution of previous LED_trial values for abort trials
    prev_led_counts = abort_df['prev_LED_trial'].value_counts(dropna=False)
    prev_led_percent = abort_df['prev_LED_trial'].value_counts(normalize=True, dropna=False) * 100
    
    # Calculate statistics
    total_abort_trials = len(abort_df)
    prev_led_is_zero = (abort_df['prev_LED_trial'] == 0).sum()
    prev_led_is_zero_percent = prev_led_is_zero / total_abort_trials * 100 if total_abort_trials > 0 else 0
    prev_led_is_nan = abort_df['prev_LED_trial'].isna().sum()
    prev_led_is_nan_percent = prev_led_is_nan / total_abort_trials * 100 if total_abort_trials > 0 else 0
    prev_led_is_nonzero = total_abort_trials - prev_led_is_zero - prev_led_is_nan
    prev_led_is_nonzero_percent = prev_led_is_nonzero / total_abort_trials * 100 if total_abort_trials > 0 else 0
    
    # Store results
    animal_result = {
        'animal': animal,
        'total_abort_trials': total_abort_trials,
        'prev_led_is_zero': prev_led_is_zero,
        'prev_led_is_zero_percent': prev_led_is_zero_percent,
        'prev_led_is_nan': prev_led_is_nan,
        'prev_led_is_nan_percent': prev_led_is_nan_percent,
        'prev_led_is_nonzero': prev_led_is_nonzero,
        'prev_led_is_nonzero_percent': prev_led_is_nonzero_percent,
        'prev_led_value_counts': prev_led_counts.to_dict()
    }
    results.append(animal_result)
    
    # Print results for this animal
    print(f"Total abort trials: {total_abort_trials}")
    print(f"Previous LED_trial is 0: {prev_led_is_zero} ({prev_led_is_zero_percent:.2f}%)")
    print(f"Previous LED_trial is NaN: {prev_led_is_nan} ({prev_led_is_nan_percent:.2f}%)")
    print(f"Previous LED_trial is non-zero: {prev_led_is_nonzero} ({prev_led_is_nonzero_percent:.2f}%)")
    print("\nDetailed value counts:")
    for value, count in prev_led_counts.items():
        print(f"  {value}: {count} ({count/total_abort_trials*100:.2f}%)")
    
    # Check if all non-NaN values are 0
    if prev_led_is_nonzero == 0 and prev_led_is_zero > 0:
        print("\nALL non-NaN previous LED_trial values are 0 for abort trials!")
    elif prev_led_is_nonzero > 0:
        print("\nSome abort trials have non-zero previous LED_trial values.")
    
    # If there are any non-zero values, show examples
    if prev_led_is_nonzero > 0:
        non_zero_examples = abort_df[abort_df['prev_LED_trial'] > 0].head(5)
        print("\nExamples of abort trials with non-zero previous LED_trial:")
        print(non_zero_examples[['session', 'trial', 'prev_LED_trial', 'is_abort']])

# %%
# Create a summary dataframe
summary_df = pd.DataFrame(results)

# Print overall summary
print("\n==== OVERALL SUMMARY ====\n")
print(f"Total animals analyzed: {len(summary_df)}")

# Calculate overall statistics
total_abort_trials = summary_df['total_abort_trials'].sum()
total_prev_led_zero = summary_df['prev_led_is_zero'].sum()
total_prev_led_nan = summary_df['prev_led_is_nan'].sum()
total_prev_led_nonzero = summary_df['prev_led_is_nonzero'].sum()

print(f"Total abort trials across all animals: {total_abort_trials}")
print(f"Previous LED_trial is 0: {total_prev_led_zero} ({total_prev_led_zero/total_abort_trials*100:.2f}%)")
print(f"Previous LED_trial is NaN: {total_prev_led_nan} ({total_prev_led_nan/total_abort_trials*100:.2f}%)")
print(f"Previous LED_trial is non-zero: {total_prev_led_nonzero} ({total_prev_led_nonzero/total_abort_trials*100:.2f}%)")

# Check if all animals have only 0 values for previous LED_trial (excluding NaNs)
all_zero = (summary_df['prev_led_is_nonzero'] == 0).all()
if all_zero:
    print("\nFor ALL animals, ALL non-NaN previous LED_trial values are 0 for abort trials!")
else:
    animals_with_nonzero = summary_df[summary_df['prev_led_is_nonzero'] > 0]['animal'].tolist()
    print(f"\nSome animals have non-zero previous LED_trial values: {animals_with_nonzero}")

# %%
# Check if LED_trial is always 0 for all trials, not just before aborts
print("\n==== CHECKING IF LED_TRIAL IS ALWAYS 0 FOR ALL TRIALS ====\n")

for animal in all_df['animal'].unique():
    animal_df = all_df[all_df['animal'] == animal].copy()
    animal_df['LED_trial'] = animal_df['LED_trial'].fillna(0).astype(int)
    
    led_counts = animal_df['LED_trial'].value_counts()
    led_percent = animal_df['LED_trial'].value_counts(normalize=True) * 100
    
    print(f"\nAnimal {animal} - LED_trial value counts:")
    for value, count in led_counts.items():
        print(f"  {value}: {count} ({led_percent[value]:.2f}%)")
    
    if led_counts.shape[0] == 1 and 0 in led_counts:
        print(f"LED_trial is ALWAYS 0 for animal {animal}!")
    else:
        print(f"LED_trial has {led_counts.shape[0]} unique values for animal {animal}")

# %%
# Check if LED_trial is actually used in the experiment
print("\n==== CHECKING IF LED_TRIAL IS USED IN THE EXPERIMENT ====\n")

# Count unique values of LED_trial across all animals
led_counts_all = all_df['LED_trial'].value_counts(dropna=False)
led_percent_all = all_df['LED_trial'].value_counts(normalize=True, dropna=False) * 100

print("LED_trial value counts across all animals:")
for value, count in led_counts_all.items():
    print(f"  {value}: {count} ({led_percent_all[value]:.2f}%)")

# Check if LED_trial is mentioned in other columns
led_columns = [col for col in all_df.columns if 'led' in col.lower()]
print(f"\nColumns related to LED: {led_columns}")

for col in led_columns:
    unique_values = all_df[col].nunique()
    print(f"\n{col} has {unique_values} unique values")
    if unique_values > 0 and unique_values < 10:  # Only show if there are a reasonable number of unique values
        value_counts = all_df[col].value_counts(dropna=False)
        print(f"{col} value counts:")
        for value, count in value_counts.items():
            print(f"  {value}: {count} ({count/len(all_df)*100:.2f}%)")
