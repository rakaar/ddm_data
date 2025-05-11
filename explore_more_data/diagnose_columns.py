# Script to diagnose which columns are causing NaN issues for animals 38-41

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Load the data
exp_df = pd.read_csv('/home/rlab/raghavendra/ddm_data/outExp.csv')

# Remove wrong rows as in the original script
exp_df = exp_df[~((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3))].copy()

# Comparable batch
all_df = exp_df[(exp_df['batch_name'] == 'Comparable')].copy()

# Define session and trial column names
SESSION_COL = 'session'
TRIAL_COL = 'trial'

# Create a PDF to save diagnostic plots
pdf = PdfPages('column_diagnostics.pdf')

# Function to analyze columns for a specific animal
def analyze_animal_columns(animal_id):
    print(f"\n=== Analyzing Animal {animal_id} ===\n")
    
    # Filter for the specific animal
    animal_df = all_df[all_df['animal'] == animal_id].copy()
    
    # Basic stats
    print(f"Total rows for animal {animal_id}: {len(animal_df)}")
    
    # Check NaN values in base columns
    base_columns = ['rewarded', 'abort_event', 'intended_fix', 'ILD', 'ABL', 'TotalFixTime', 'CNPTime', 'MT']
    
    # Add derived columns that will be used in the analysis
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
    animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
    animal_df['abs_ILD'] = animal_df['ILD'].abs()
    
    # Add derived columns to the list
    derived_columns = ['is_abort', 'short_poke', 'rewarded', 'abs_ILD']
    all_columns = base_columns + derived_columns
    
    # Check NaN values in all columns
    nan_counts = animal_df[all_columns].isna().sum()
    print("\nNaN counts in columns:")
    for col, count in nan_counts.items():
        print(f"{col}: {count} NaNs ({count/len(animal_df)*100:.2f}%)")
    
    # Create a bar chart of NaN percentages
    plt.figure(figsize=(12, 6))
    nan_percentages = (animal_df[all_columns].isna().sum() / len(animal_df) * 100).sort_values(ascending=False)
    sns.barplot(x=nan_percentages.index, y=nan_percentages.values)
    plt.title(f"Percentage of NaN values per column - Animal {animal_id}")
    plt.xlabel("Columns")
    plt.ylabel("Percentage of NaN values")
    plt.xticks(rotation=90)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Check for columns with 100% NaN
    cols_all_nan = [col for col in all_columns if animal_df[col].isna().all()]
    if cols_all_nan:
        print(f"\nColumns with 100% NaN: {cols_all_nan}")
    
    # Check for columns with high NaN percentage (>50%)
    cols_high_nan = [col for col in all_columns if animal_df[col].isna().mean() > 0.5 and animal_df[col].isna().mean() < 1.0]
    if cols_high_nan:
        print(f"Columns with >50% NaN: {cols_high_nan}")
    
    # Now let's create lagged columns for T_max_lag=1 and see if they have NaNs
    print("\nCreating lagged columns for T_max_lag=1...")
    
    # Normalize trial number within each session
    animal_df['norm_trial'] = animal_df.groupby(SESSION_COL)[TRIAL_COL].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
    )
    
    # Function to add lagged columns
    def add_lagged_column(df, value_col_name, k=1):
        df_copy = df.copy()
        new_col_name = f"{value_col_name}_{k}"
        
        # Group by session and shift the values
        df_copy[new_col_name] = df_copy.groupby(SESSION_COL)[value_col_name].shift(k)
        
        return df_copy[new_col_name]
    
    # Create lagged columns
    lagged_vars_to_create = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']
    
    for base_var_name in lagged_vars_to_create:
        # Skip if base column doesn't exist or is all NaN
        if base_var_name not in animal_df.columns or animal_df[base_var_name].isna().all():
            print(f"Skipping lag for '{base_var_name}' as it doesn't exist or is all NaN")
            continue
        
        # Create the lagged column
        final_lagged_col_name = f'{base_var_name}_1'
        if base_var_name == 'is_abort':
            final_lagged_col_name = 'abort_1'
        
        animal_df[final_lagged_col_name] = add_lagged_column(animal_df, base_var_name, k=1)
    
    # Define predictor columns for T_max_lag=1
    predictor_cols = [
        'rewarded_1', 'abort_1', 'short_poke_1', 'intended_fix_1', 
        'abs_ILD_1', 'ABL_1', 'TotalFixTime_1', 'CNPTime_1', 'norm_trial'
    ]
    
    # Check which predictor columns exist
    existing_predictors = [col for col in predictor_cols if col in animal_df.columns]
    
    # Check NaN values in lagged columns
    if existing_predictors:
        lagged_nan_counts = animal_df[existing_predictors].isna().sum()
        print("\nNaN counts in predictor columns:")
        for col, count in lagged_nan_counts.items():
            print(f"{col}: {count} NaNs ({count/len(animal_df)*100:.2f}%)")
        
        # Count rows with NaN in any predictor column
        rows_with_nan = animal_df[existing_predictors].isna().any(axis=1).sum()
        print(f"\nRows with NaN in any predictor column: {rows_with_nan} ({rows_with_nan/len(animal_df)*100:.2f}%)")
        
        # Check if there are any rows without NaNs in predictors
        rows_without_nan = len(animal_df) - rows_with_nan
        print(f"Rows without NaN in any predictor column: {rows_without_nan} ({rows_without_nan/len(animal_df)*100:.2f}%)")
        
        # Count abort trials
        abort_rows = (animal_df['is_abort'] == 1).sum()
        print(f"Abort trials: {abort_rows} ({abort_rows/len(animal_df)*100:.2f}%)")
        
        # Check rows that would be kept after filtering
        animal_df_cleaned = animal_df.dropna(subset=existing_predictors).reset_index(drop=True)
        print(f"Rows after NaN drop: {len(animal_df_cleaned)} ({len(animal_df_cleaned)/len(animal_df)*100:.2f}%)")
        
        # Filter for abort trials
        animal_df_cleaned = animal_df_cleaned[animal_df_cleaned['is_abort'] == 1].copy()
        print(f"Rows after NaN drop and abort filter: {len(animal_df_cleaned)} ({len(animal_df_cleaned)/len(animal_df)*100:.2f}%)")
        
        # Create a heatmap of NaN values in predictor columns
        plt.figure(figsize=(12, 8))
        sns.heatmap(animal_df[existing_predictors].isna(), cmap='viridis', cbar=False)
        plt.title(f"NaN values in predictor columns - Animal {animal_id}")
        plt.xlabel("Predictor Columns")
        plt.ylabel("Rows")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    else:
        print("No predictor columns exist for this animal!")
    
    # Check for 'intended_fix' column specifically
    if 'intended_fix' not in animal_df.columns:
        print(f"\nWARNING: 'intended_fix' column does not exist for animal {animal_id}")
        # Check all columns to see what might be similar
        print(f"Available columns: {animal_df.columns.tolist()}")
    
    # Return a summary of the analysis
    return {
        'animal': animal_id,
        'total_rows': len(animal_df),
        'abort_rows': (animal_df['is_abort'] == 1).sum(),
        'rows_with_predictor_nan': rows_with_nan if 'rows_with_nan' in locals() else len(animal_df),
        'rows_after_nan_drop': len(animal_df_cleaned) if 'animal_df_cleaned' in locals() else 0
    }

# Analyze each animal
results = []

for animal_id in sorted(all_df['animal'].unique()):
    results.append(analyze_animal_columns(animal_id))

# Create a summary table
results_df = pd.DataFrame(results)
print("\n=== Summary of analysis ===\n")
print(results_df)

# Create a summary plot
plt.figure(figsize=(10, 6))
results_df['percent_rows_after_nan_drop'] = results_df['rows_after_nan_drop'] / results_df['total_rows'] * 100
plt.bar(results_df['animal'].astype(str), results_df['percent_rows_after_nan_drop'])
plt.title('Percentage of rows remaining after NaN drop and abort filter')
plt.xlabel('Animal ID')
plt.ylabel('Percentage of rows remaining')
plt.tight_layout()
pdf.savefig()
plt.close()

# Close the PDF
pdf.close()

print("\nDiagnostic results saved to column_diagnostics.pdf")
