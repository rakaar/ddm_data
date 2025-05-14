# %%
import os
import pandas as pd

batch_csv_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_csv_dir) if f.endswith('.csv')]

for batch_file in batch_files:
    batch_path = os.path.join(batch_csv_dir, batch_file)
    df = pd.read_csv(batch_path)
    unique_abl = sorted(df['ABL'].dropna().unique())
    unique_ild = sorted(df['ILD'].dropna().unique())
    print(f"{batch_file}:")
    print(f"  Unique ABL: {unique_abl}")
    print(f"  Unique ILD: {unique_ild}\n")
