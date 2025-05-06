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
# Function to add lagged column (same as in glm_all_T.py, but without include_groups parameter)
def add_lagged_column(df, col, k):
    def get_lagged(session_df):
        trial_to_val = session_df.set_index('trial')[col]
        return session_df['trial'].map(lambda t: trial_to_val.get(t - k, np.nan))
    return df.groupby('session', group_keys=False).apply(get_lagged)

# %%
# Set T value (as in glm_all_T.py)
T = 1

# Create a figure to plot correlation matrices for each animal
fig, axes = plt.subplots(4, 3, figsize=(20, 24))
axes = axes.flatten()

# Store correlation results for all animals
all_correlations = []

# Define the variables to use based on what's available in the dataframe
lagged_vars = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'ILD', 'ABL', 'TotalFixTime', 'CNPTime']

print("Creating 'short_poke' column based on is_abort=1 and TotalFixTime < 0.3")

for i, animal in enumerate(all_df['animal'].unique()):
    print(f"\nAnalyzing animal: {animal}")
    
    # Prepare the dataframe for the animal
    animal_df = all_df[all_df['animal'] == animal].copy()
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
    
    # Create short_poke column as defined in glm_all_T.py
    animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
    
    # Add normalized trial number
    animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )
    
    # Create lagged variables (excluding MT and LED_trial)
    for k in range(1, T + 1):
        for var in lagged_vars:
            colname = f'{var}_{k}' if var != 'is_abort' else f'abort_{k}'
            base_col = var if var != 'is_abort' else 'is_abort'
            animal_df[colname] = add_lagged_column(animal_df, base_col, k)
    
    # Create lag_cols list
    lag_cols = [f'{var}_{k}' for k in range(1, T + 1)
                for var in ['rewarded', 'abort', 'short_poke', 'intended_fix', 'ILD', 'ABL', 'TotalFixTime', 'CNPTime']]
    
    # Drop rows with NaN in lagged columns
    animal_no_nan_df = animal_df.dropna(subset=lag_cols).reset_index(drop=True)
    
    # Filter to only keep abort trials
    abort_no_nan_df = animal_no_nan_df[animal_no_nan_df['is_abort'] == 1].copy()
    
    # Define predictor columns
    predictor_cols = [f'{var}_{k}' for k in range(1, T + 1)
                     for var in ['rewarded', 'abort', 'short_poke', 'intended_fix', 'ILD', 'ABL', 'TotalFixTime', 'CNPTime']]
    predictor_cols += ['norm_trial']
    
    # Check for constant columns
    constant_cols = []
    for col in predictor_cols:
        if col in abort_no_nan_df.columns and abort_no_nan_df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Constant columns for animal {animal}: {constant_cols}")
        # Print the constant values
        for col in constant_cols:
            value = abort_no_nan_df[col].iloc[0] if not abort_no_nan_df[col].isna().all() else np.nan
            print(f"  {col} = {value}")
    else:
        print(f"No constant columns for animal {animal}")
    
    # Calculate correlation matrix
    corr_matrix = abort_no_nan_df[predictor_cols].corr()
    
    # Store correlation data
    corr_data = {
        'animal': animal,
        'correlation_matrix': corr_matrix,
        'constant_columns': constant_cols,
        'num_rows': len(abort_no_nan_df)
    }
    all_correlations.append(corr_data)
    
    # Plot correlation matrix
    ax = axes[i]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                center=0, square=True, linewidths=.5, annot=False, ax=ax)
    ax.set_title(f'Animal {animal} - Correlation Matrix\n(Rows: {len(abort_no_nan_df)})')
    
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('correlation_matrices_fixed_v2.png', dpi=300, bbox_inches='tight')

# %%
# Find the highest correlations across all animals
print("\n==== HIGHEST CORRELATIONS ACROSS ALL ANIMALS ====\n")

# Combine all correlation matrices, weighted by number of rows
weighted_corr = None
total_rows = 0

for corr_data in all_correlations:
    if weighted_corr is None:
        weighted_corr = corr_data['correlation_matrix'] * corr_data['num_rows']
    else:
        weighted_corr += corr_data['correlation_matrix'] * corr_data['num_rows']
    total_rows += corr_data['num_rows']

weighted_corr /= total_rows

# Create a heatmap of the weighted correlation matrix
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(weighted_corr, dtype=bool))
sns.heatmap(weighted_corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1,
            center=0, square=True, linewidths=.5, annot=False)
plt.title('Weighted Average Correlation Matrix Across All Animals')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('weighted_correlation_matrix_fixed_v2.png', dpi=300, bbox_inches='tight')

# Get the correlations in a different way
print("Top 20 highest absolute correlations:")
# Convert to pandas DataFrame for easier manipulation
weighted_corr_df = pd.DataFrame(weighted_corr, index=predictor_cols, columns=predictor_cols)

# Get the lower triangle of the correlation matrix (excluding diagonal)
tri_mask = np.triu(np.ones(weighted_corr_df.shape), k=0).astype(bool)
weighted_corr_df_masked = weighted_corr_df.mask(tri_mask)

# Stack the DataFrame to get pairs of variables and their correlation
corr_pairs = weighted_corr_df_masked.stack().reset_index()
corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']

# Sort by absolute correlation
corr_pairs['Abs_Correlation'] = corr_pairs['Correlation'].abs()
corr_pairs_sorted = corr_pairs.sort_values('Abs_Correlation', ascending=False)

# Print top 20
for _, row in corr_pairs_sorted.head(20).iterrows():
    print(f"{row['Variable 1']} & {row['Variable 2']}: {row['Correlation']:.4f}")
