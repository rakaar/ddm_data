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
    # Removed include_groups parameter which isn't available in this pandas version
    return df.groupby('session', group_keys=False).apply(get_lagged)

# %%
# Create correlation matrices for each animal
T = 1  # We're only looking at lag 1 for simplicity
lagged_vars = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'ILD', 'ABL', 'LED_trial', 'TotalFixTime', 'CNPTime']

for animal in all_df['animal'].unique():
    print(f"\nAnalyzing animal: {animal}")
    
    # Prepare the dataframe for the animal
    animal_df = all_df[all_df['animal'] == animal].copy()
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
    animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
    animal_df['LED_trial'] = animal_df['LED_trial'].fillna(0).astype(int)
    animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() if x.max() > x.min() else 1))
    
    # Add lagged columns
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
    
    # Add norm_trial to the predictors
    predictor_cols = lag_cols + ['norm_trial']
    
    # Check if we have enough data
    if len(animal_df) < 10:
        print(f"Not enough data for animal {animal} after filtering. Skipping.")
        continue
    
    # Calculate and plot correlation matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = animal_df[predictor_cols].corr()
    
    # Plot heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={"shrink": .5})
    
    plt.title(f'Correlation Matrix of Predictors for Animal {animal}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'correlation_matrix_animal_{animal}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Identify highly correlated predictors (>0.8)
    high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix.abs() < 1.0)
    if high_corr.any().any():
        print("\nHighly correlated pairs (|r| > 0.8):")
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if high_corr.iloc[i, j]:
                    print(f"  {corr_matrix.columns[i]} & {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")

# %%
# Create a combined correlation matrix for all animals
print("\nCreating combined correlation matrix for all animals...")

# Prepare combined dataframe
combined_df = pd.DataFrame()

for animal in all_df['animal'].unique():
    # Prepare the dataframe for the animal
    animal_df = all_df[all_df['animal'] == animal].copy()
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
    animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
    animal_df['LED_trial'] = animal_df['LED_trial'].fillna(0).astype(int)
    
    # Add lagged columns
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
    
    # Add norm_trial and animal columns
    animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() if x.max() > x.min() else 1))
    animal_df['animal'] = animal
    
    # Add to combined dataframe
    combined_df = pd.concat([combined_df, animal_df])

# Add norm_trial to the predictors
predictor_cols = lag_cols + ['norm_trial']

# Calculate and plot correlation matrix
plt.figure(figsize=(14, 12))
corr_matrix = combined_df[predictor_cols].corr()

# Plot heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={"shrink": .5})

plt.title('Combined Correlation Matrix of Predictors Across All Animals', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_matrix_all_animals.png', dpi=300, bbox_inches='tight')
plt.show()

# Identify highly correlated predictors (>0.8)
high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix.abs() < 1.0)
if high_corr.any().any():
    print("\nHighly correlated pairs (|r| > 0.8):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if high_corr.iloc[i, j]:
                print(f"  {corr_matrix.columns[i]} & {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")

# %%
# Create a more detailed analysis of the relationship between TotalFixTime_1 and other predictors
print("\nAnalyzing relationship between TotalFixTime_1 and other predictors...")

# Create scatter plots for TotalFixTime_1 vs other continuous predictors
continuous_predictors = ['ILD_1', 'ABL_1', 'TotalFixTime_1', 'CNPTime_1', 'norm_trial']

# Set up the figure
plt.figure(figsize=(15, 10))

# Create scatter plots
for i, pred in enumerate(continuous_predictors):
    if pred != 'TotalFixTime_1':
        plt.subplot(2, 2, i if i < 3 else i-1)
        sns.scatterplot(data=combined_df, x='TotalFixTime_1', y=pred, hue='animal', alpha=0.6)
        plt.title(f'TotalFixTime_1 vs {pred}')
        plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('totalfixtime_relationships.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Create boxplots for categorical predictors vs TotalFixTime_1
categorical_predictors = ['rewarded_1', 'abort_1', 'short_poke_1', 'intended_fix_1', 'LED_trial_1']

# Set up the figure
plt.figure(figsize=(15, 10))

# Create boxplots
for i, pred in enumerate(categorical_predictors):
    plt.subplot(2, 3, i+1)
    sns.boxplot(data=combined_df, x=pred, y='TotalFixTime_1')
    plt.title(f'{pred} vs TotalFixTime_1')
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('totalfixtime_vs_categorical.png', dpi=300, bbox_inches='tight')
plt.show()
