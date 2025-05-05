import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- USER: Set your animal number here ---
import random
batch_name = 'Comparable'  # or set as needed
ANIMAL_ID = 41
DATA_FILE = '../outExp.csv'

# Load and preprocess as in refactor script
exp_df = pd.read_csv(DATA_FILE)
exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

exp_df_batch = exp_df[
    (exp_df['batch_name'] == batch_name) &
    (exp_df['LED_trial'].isin([np.nan, 0]))
].copy()

exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)

# Valid and aborts, as in refactor
df_valid_and_aborts = exp_df_batch[
    (exp_df_batch['success'].isin([1, -1])) |
    (exp_df_batch['abort_event'] == 3)
].copy()

# Animal selection
animal_ids = df_valid_and_aborts['animal'].unique()
if ANIMAL_ID not in animal_ids:
    raise ValueError(f"Animal {ANIMAL_ID} not found in filtered data. Available: {animal_ids}")
df_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == ANIMAL_ID].copy()

# --- Extract arrays ---
ABL_arr = np.sort(df_animal['ABL'].unique())
ILD_arr = np.sort(df_animal['ILD'].unique())

# --- Psychometric function from animal_wise_fit_3_models_script.py ---
def plot_psycho_script(df_1, ILD_arr):
    prob_choice_dict = {}
    all_ABL = np.sort(df_1['ABL'].unique())
    for abl in all_ABL:
        filtered_df = df_1[df_1['ABL'] == abl]
        prob_choice_dict[abl] = [
            sum(filtered_df[filtered_df['ILD'] == ild]['choice'] == 1) / len(filtered_df[filtered_df['ILD'] == ild])
            if len(filtered_df[filtered_df['ILD'] == ild]) > 0 else np.nan
            for ild in ILD_arr
        ]
    return prob_choice_dict

# --- Psychometric function from animal_wise_plotting_utils.py (refactor) ---
def plot_psycho_refactor(df_1, ILD_arr):
    prob_choice_dict = {}
    all_ABL = np.sort(df_1['ABL'].unique())
    for abl in all_ABL:
        filtered_df = df_1[df_1['ABL'] == abl]
        prob_choice_dict[abl] = [
            np.nan if len(filtered_df[filtered_df['ILD'] == ild]) == 0 else np.mean(filtered_df[filtered_df['ILD'] == ild]['choice'] == 1)
            for ild in ILD_arr
        ]
    return prob_choice_dict

# --- Calculate psychometrics ---
psycho_script = plot_psycho_script(df_animal, ILD_arr)
psycho_refactor = plot_psycho_refactor(df_animal, ILD_arr)

# --- Plot ---
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(ABL_arr)))
for i, abl in enumerate(ABL_arr):
    plt.plot(ILD_arr, psycho_script[abl], '-', color=colors[i], linewidth=1, label=f'Script ABL={abl}')
    plt.plot(ILD_arr, psycho_refactor[abl], ':', color=colors[i], linewidth=2, label=f'Refactor ABL={abl}')
plt.xlabel('ILD')
plt.ylabel('P(choice==1)')
plt.title(f'Psychometric Curve Comparison for Animal {ANIMAL_ID}')
plt.legend()
plt.tight_layout()
plt.show()
