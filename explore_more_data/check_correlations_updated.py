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
# Set T value (as in glm_all_T.py)
T = 1

# Create a figure to plot correlation matrices for each animal
fig, axes = plt.subplots(4, 3, figsize=(20, 24))
axes = axes.flatten()

# Store correlation results for all animals
all_correlations = []

for i, animal in enumerate(all_df['animal'].unique()):
    print(f"\nAnalyzing animal: {animal}")
    
    # Prepare the dataframe for the animal
    animal_df = all_df[all_df['animal'] == animal].copy()
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    
    # Add normalized trial number
    animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )
    
    # Create lagged variables (excluding MT and LED_trial)
    lagged_vars = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'ILD', 'ABL', 'TotalFixTime', 'CNPTime']
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
plt.savefig('correlation_matrices_updated.png', dpi=300, bbox_inches='tight')

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

# Create a mask for the upper triangle and diagonal
mask = np.triu(np.ones_like(weighted_corr, dtype=bool))
np.fill_diagonal(mask, True)  # Also mask the diagonal

# Get the correlations as a Series
corr_series = weighted_corr.stack()[~mask.stack()]

# Sort by absolute correlation value
sorted_corrs = corr_series.abs().sort_values(ascending=False)

# Print the top 20 highest absolute correlations
print("Top 20 highest absolute correlations:")
for (var1, var2), corr_val in sorted_corrs.iloc[:20].items():
    print(f"{var1} & {var2}: {weighted_corr.loc[var1, var2]:.4f}")

# %%
# Create a heatmap of the weighted correlation matrix
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(weighted_corr, dtype=bool))
sns.heatmap(weighted_corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1,
            center=0, square=True, linewidths=.5, annot=False)
plt.title('Weighted Average Correlation Matrix Across All Animals')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('weighted_correlation_matrix_updated.png', dpi=300, bbox_inches='tight')

# %%
# Check for multicollinearity using Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("\n==== VARIANCE INFLATION FACTOR (VIF) ANALYSIS ====\n")

# Create a dataframe to store VIF values for each animal
vif_results = []

for animal in all_df['animal'].unique():
    # Prepare the dataframe for the animal
    animal_df = all_df[all_df['animal'] == animal].copy()
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    
    # Add normalized trial number
    animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )
    
    # Create lagged variables (excluding MT and LED_trial)
    lagged_vars = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'ILD', 'ABL', 'TotalFixTime', 'CNPTime']
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
    
    # Remove constant columns
    non_constant_cols = []
    for col in predictor_cols:
        if col in abort_no_nan_df.columns and abort_no_nan_df[col].nunique() > 1:
            non_constant_cols.append(col)
    
    if len(non_constant_cols) < len(predictor_cols):
        print(f"Removed {len(predictor_cols) - len(non_constant_cols)} constant columns for animal {animal}")
    
    # Skip if there are too few rows or columns
    if len(abort_no_nan_df) <= len(non_constant_cols) or len(non_constant_cols) == 0:
        print(f"Skipping VIF for animal {animal}: not enough data")
        continue
    
    # Calculate VIF
    X = abort_no_nan_df[non_constant_cols]
    
    try:
        # Add a small amount of noise to prevent perfect multicollinearity
        X = X + np.random.normal(0, 1e-6, X.shape)
        
        # Calculate VIF for each predictor
        vif_data = pd.DataFrame()
        vif_data["Variable"] = non_constant_cols
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        # Store results
        vif_data["animal"] = animal
        vif_results.append(vif_data)
        
        # Print high VIF values
        high_vif = vif_data[vif_data["VIF"] > 5].sort_values("VIF", ascending=False)
        if len(high_vif) > 0:
            print(f"\nHigh VIF values for animal {animal}:")
            print(high_vif)
    except Exception as e:
        print(f"Error calculating VIF for animal {animal}: {e}")

# Combine VIF results
if vif_results:
    all_vif = pd.concat(vif_results)
    
    # Calculate average VIF per variable
    avg_vif = all_vif.groupby("Variable")["VIF"].mean().sort_values(ascending=False)
    
    print("\nAverage VIF across all animals:")
    print(avg_vif.head(10))  # Show top 10 highest VIF
    
    # Plot average VIF
    plt.figure(figsize=(12, 8))
    avg_vif.plot(kind='bar')
    plt.axhline(y=5, color='r', linestyle='--', label='VIF=5 threshold')
    plt.axhline(y=10, color='darkred', linestyle='--', label='VIF=10 threshold')
    plt.title('Average Variance Inflation Factor (VIF) Across All Animals')
    plt.ylabel('VIF')
    plt.xlabel('Predictor')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('average_vif_updated.png', dpi=300, bbox_inches='tight')
else:
    print("No valid VIF results to display")
