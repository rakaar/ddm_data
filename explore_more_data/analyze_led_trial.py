# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# Analyze LED_trial and LED_trial_1 for all animals
print("\n==== ANALYZING LED_TRIAL AND LED_TRIAL_1 FOR ALL ANIMALS ====\n")

# Define the variables we want to lag
T = 1  # We're only looking at lag 1 for simplicity

results = []

for animal in all_df['animal'].unique():
    print(f"\nAnalyzing animal: {animal}")
    
    # Prepare the dataframe for the animal
    animal_df = all_df[all_df['animal'] == animal].copy()
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
    animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
    animal_df['LED_trial'] = animal_df['LED_trial'].fillna(0).astype(int)
    
    # Add lagged columns
    animal_df['LED_trial_1'] = add_lagged_column(animal_df, 'LED_trial', 1)
    
    # Basic statistics for LED_trial
    led_trial_stats = {
        'animal': animal,
        'total_rows': len(animal_df),
        'led_trial_unique_values': animal_df['LED_trial'].nunique(),
        'led_trial_value_counts': animal_df['LED_trial'].value_counts().to_dict(),
        'led_trial_nan_count': animal_df['LED_trial'].isna().sum(),
        'led_trial_nan_percent': animal_df['LED_trial'].isna().mean() * 100
    }
    
    # Filter to only keep abort trials
    abort_df = animal_df[animal_df['is_abort'] == 1].copy()
    
    # Statistics for LED_trial in abort trials
    led_trial_abort_stats = {
        'abort_rows': len(abort_df),
        'led_trial_abort_unique_values': abort_df['LED_trial'].nunique(),
        'led_trial_abort_value_counts': abort_df['LED_trial'].value_counts().to_dict(),
        'led_trial_abort_nan_count': abort_df['LED_trial'].isna().sum(),
        'led_trial_abort_nan_percent': abort_df['LED_trial'].isna().mean() * 100 if len(abort_df) > 0 else 0
    }
    
    # Statistics for LED_trial_1 in abort trials
    led_trial_1_abort_stats = {
        'led_trial_1_abort_unique_values': abort_df['LED_trial_1'].nunique(),
        'led_trial_1_abort_value_counts': abort_df['LED_trial_1'].value_counts().to_dict(),
        'led_trial_1_abort_nan_count': abort_df['LED_trial_1'].isna().sum(),
        'led_trial_1_abort_nan_percent': abort_df['LED_trial_1'].isna().mean() * 100 if len(abort_df) > 0 else 0
    }
    
    # Now filter to keep only abort trials with no NaN in LED_trial_1
    abort_no_nan_df = abort_df.dropna(subset=['LED_trial_1']).copy()
    
    # Statistics after dropping NaNs
    led_trial_1_no_nan_stats = {
        'abort_no_nan_rows': len(abort_no_nan_df),
        'led_trial_1_no_nan_unique_values': abort_no_nan_df['LED_trial_1'].nunique(),
        'led_trial_1_no_nan_value_counts': abort_no_nan_df['LED_trial_1'].value_counts().to_dict(),
        'percent_rows_remaining': len(abort_no_nan_df) / len(abort_df) * 100 if len(abort_df) > 0 else 0
    }
    
    # Combine all stats
    animal_stats = {**led_trial_stats, **led_trial_abort_stats, **led_trial_1_abort_stats, **led_trial_1_no_nan_stats}
    results.append(animal_stats)
    
    # Print summary for this animal
    print(f"Total rows: {len(animal_df)}")
    print(f"Abort rows: {len(abort_df)}")
    print(f"LED_trial unique values: {animal_df['LED_trial'].nunique()}")
    print(f"LED_trial value counts: {animal_df['LED_trial'].value_counts().to_dict()}")
    print(f"LED_trial_1 in abort trials - unique values: {abort_df['LED_trial_1'].nunique()}")
    print(f"LED_trial_1 in abort trials - value counts: {abort_df['LED_trial_1'].value_counts().to_dict()}")
    print(f"LED_trial_1 in abort trials - NaN count: {abort_df['LED_trial_1'].isna().sum()} ({abort_df['LED_trial_1'].isna().mean() * 100:.2f}%)")
    
    # Check if LED_trial_1 becomes constant after filtering
    if len(abort_no_nan_df) > 0:
        if abort_no_nan_df['LED_trial_1'].nunique() == 1:
            constant_value = abort_no_nan_df['LED_trial_1'].iloc[0]
            print(f"LED_trial_1 becomes CONSTANT with value {constant_value} after filtering!")
            print(f"Rows remaining after filtering: {len(abort_no_nan_df)} out of {len(abort_df)} ({len(abort_no_nan_df) / len(abort_df) * 100:.2f}%)")
        else:
            print(f"LED_trial_1 remains variable after filtering with {abort_no_nan_df['LED_trial_1'].nunique()} unique values")
            print(f"LED_trial_1 value counts after filtering: {abort_no_nan_df['LED_trial_1'].value_counts().to_dict()}")

# %%
# Create a summary dataframe
summary_df = pd.DataFrame(results)

# Print overall summary
print("\n==== OVERALL SUMMARY ====\n")
print(f"Total animals analyzed: {len(summary_df)}")

# Check if LED_trial_1 becomes constant for any animals
constant_animals = summary_df[summary_df['led_trial_1_no_nan_unique_values'] == 1]
if len(constant_animals) > 0:
    print(f"\nAnimals where LED_trial_1 becomes constant after filtering: {len(constant_animals)} out of {len(summary_df)}")
    for idx, row in constant_animals.iterrows():
        value = list(row['led_trial_1_no_nan_value_counts'].keys())[0]
        print(f"  Animal {row['animal']}: constant value = {value}, rows remaining = {row['abort_no_nan_rows']} ({row['percent_rows_remaining']:.2f}%)")
else:
    print("\nLED_trial_1 does not become constant for any animal after filtering")

# %%
# Visualize the distribution of LED_trial and LED_trial_1
plt.figure(figsize=(15, 10))

# Create subplots for each animal
for i, animal in enumerate(all_df['animal'].unique()):
    animal_data = summary_df[summary_df['animal'] == animal].iloc[0]
    
    # Extract value counts
    led_trial_counts = pd.Series(animal_data['led_trial_value_counts'])
    led_trial_1_counts = pd.Series(animal_data['led_trial_1_abort_value_counts'])
    
    # Plot
    plt.subplot(2, len(all_df['animal'].unique()), i+1)
    led_trial_counts.plot(kind='bar', color='blue', alpha=0.7)
    plt.title(f'LED_trial - Animal {animal}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    
    plt.subplot(2, len(all_df['animal'].unique()), i+1+len(all_df['animal'].unique()))
    led_trial_1_counts.plot(kind='bar', color='green', alpha=0.7)
    plt.title(f'LED_trial_1 (abort trials) - Animal {animal}')
    plt.xlabel('Value')
    plt.ylabel('Count')

plt.tight_layout()
plt.savefig('led_trial_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Create a correlation matrix for each animal to check LED_trial_1 correlations
for animal in all_df['animal'].unique():
    print(f"\nCorrelation matrix for animal {animal}:")
    
    # Prepare the dataframe for the animal
    animal_df = all_df[all_df['animal'] == animal].copy()
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
    animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
    animal_df['LED_trial'] = animal_df['LED_trial'].fillna(0).astype(int)
    
    # Add all lagged columns
    lagged_vars = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'ILD', 'ABL', 'LED_trial', 'TotalFixTime', 'CNPTime']
    for k in range(1, T + 1):
        for var in lagged_vars:
            colname = f'{var}_{k}' if var != 'is_abort' else f'abort_{k}'
            base_col = var
            animal_df[colname] = add_lagged_column(animal_df, base_col, k)
    
    # Get the lag column names
    lag_cols = [f'{var}_{k}' for k in range(1, T + 1)
                for var in ['rewarded', 'abort', 'short_poke', 'intended_fix', 'ILD', 'ABL', 'LED_trial', 'TotalFixTime', 'CNPTime']]
    
    # Filter to only keep abort trials and drop NaN values
    animal_df = animal_df.dropna(subset=lag_cols).reset_index(drop=True)
    animal_df = animal_df[animal_df['is_abort'] == 1].copy()
    
    # Add norm_trial
    animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() if x.max() > x.min() else 1))
    
    # Add predictor_cols
    predictor_cols = lag_cols + ['norm_trial']
    
    # Check if we have enough data
    if len(animal_df) < 10:
        print(f"Not enough data for animal {animal} after filtering. Skipping correlation matrix.")
        continue
    
    # Print LED_trial_1 unique values after filtering
    print(f"LED_trial_1 unique values after filtering: {animal_df['LED_trial_1'].nunique()}")
    print(f"LED_trial_1 value counts after filtering: {animal_df['LED_trial_1'].value_counts().to_dict()}")
    
    # Calculate correlation matrix
    corr_matrix = animal_df[predictor_cols].corr()
    
    # Print LED_trial_1 correlations
    print("\nLED_trial_1 correlations with other predictors:")
    led_correlations = corr_matrix['LED_trial_1'].sort_values(ascending=False)
    print(led_correlations)
    
    # Plot correlation matrix without masking to see all correlations
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f')
    
    plt.title(f'Full Correlation Matrix for Animal {animal}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'full_correlation_matrix_animal_{animal}.png', dpi=300, bbox_inches='tight')
    plt.show()
