# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Load data (adjust path if needed)
exp_df = pd.read_csv('../outExp.csv')

# Remove wrong rows 
count = ((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3)).sum()
print("Number of rows where TotalFixTime is NaN and abort_event == 3:", count)
exp_df = exp_df[~((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3))].copy()

# Comparable batch
all_df = exp_df[exp_df['batch_name'] == 'Comparable'].copy()

# Prepare lagged predictors (T=1 as in glm_all_T.py)
def add_lagged_column(df, col, k):
    return df.groupby('session')[col].shift(k)

T = 1
lagged_vars = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']

all_df.loc[:, 'is_abort'] = (all_df['abort_event'] == 3).astype(int)
all_df.loc[:, 'short_poke'] = ((all_df['is_abort'] == 1) & (all_df['TotalFixTime'] < 0.3)).astype(int)
all_df.loc[:, 'rewarded'] = (all_df['success'] == 1).astype(int)
all_df.loc[:, 'abs_ILD'] = all_df['ILD'].abs()
all_df.loc[:, 'norm_trial'] = all_df.groupby('session')['trial'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() if x.max() > x.min() else 1))

for k in range(1, T + 1):
    for var in lagged_vars:
        colname = f'{var}_{k}' if var != 'is_abort' else f'abort_{k}'
        base_col = var
        all_df[colname] = add_lagged_column(all_df, base_col, k)

lag_cols = [f'{var}_{k}' for k in range(1, T + 1)
            for var in ['rewarded', 'abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']]

# Function to plot and save correlations for each animal
def plot_predictor_correlations_per_animal(df, lag_cols, animal_col='animal', pdf_name='animal_predictor_correlations.pdf'):
    """
    For each animal, plot the correlation matrix of predictors and save each as a page in a single PDF.
    """
    animals = df[animal_col].unique()
    with PdfPages(pdf_name) as pdf:
        for animal in animals:
            animal_df = df[df[animal_col] == animal]
            sub_df = animal_df[lag_cols].dropna()
            if len(sub_df) == 0:
                continue
            corr = sub_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
            plt.title(f'Predictor Correlations: Animal {animal}')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f'PDF saved as {pdf_name}')

if __name__ == "__main__":
    plot_predictor_correlations_per_animal(all_df, lag_cols)

    # --- New: Correlation between predictors and short_poke per animal ---
    pdf_name = 'predictor_vs_short_poke_corr_per_animal.pdf'
    with PdfPages(pdf_name) as pdf:
        animals = all_df['animal'].unique()
        for animal in animals:
            animal_df = all_df[all_df['animal'] == animal]
            # Only keep predictors and short_poke
            cols_for_corr = lag_cols + ['norm_trial', 'short_poke']
            sub_df = animal_df[cols_for_corr].dropna()
            if len(sub_df) == 0:
                continue
            corr_with_short_poke = sub_df.corr()['short_poke'].drop('short_poke')
            plt.figure(figsize=(10, 6))
            corr_with_short_poke.sort_values().plot(kind='barh')
            plt.title(f'Correlation of Predictors with short_poke\nAnimal {animal}')
            plt.xlabel('Correlation coefficient')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f'PDF with predictor-short_poke correlations saved as {pdf_name}')

# %%
