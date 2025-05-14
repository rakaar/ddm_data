import os
import pandas as pd

batch_csv_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_csv_dir) if f.endswith('.csv')]

# Merge all CSVs into one DataFrame
all_dfs = []
for batch_file in batch_files:
    batch_path = os.path.join(batch_csv_dir, batch_file)
    df = pd.read_csv(batch_path)
    all_dfs.append(df)

merged_df = pd.concat(all_dfs, ignore_index=True)

# Save the merged DataFrame to a new CSV
merged_csv_path = os.path.join(batch_csv_dir, 'merged_batches.csv')
merged_df.to_csv(merged_csv_path, index=False)

print(f"Merged {len(batch_files)} files into {merged_csv_path} with {len(merged_df)} rows.")

# %% qq plots for each abs_ILD
