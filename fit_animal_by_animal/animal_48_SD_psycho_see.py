# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def plot_psycho(df_1, ILD_arr):
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

batch_name = 'SD'
session_type = 1
exp_df = pd.read_csv('../outExp.csv')
exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

exp_df_batch = exp_df[
        (exp_df['batch_name'] == batch_name) &
        (exp_df['LED_trial'].isin([np.nan, 0])) &
        (exp_df['session_type'] == session_type)
    ].copy()

exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)

df_valid_and_aborts = exp_df_batch[
    (exp_df_batch['success'].isin([1, -1])) |
    (exp_df_batch['abort_event'] == 3)
].copy()

animal_ids = np.sort(df_valid_and_aborts['animal'].unique())
n_animals = len(animal_ids)
n_cols = 4
n_rows = int(np.ceil(n_animals / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey=True)
axs = axs.flatten()
for idx, animal in enumerate(animal_ids):
    ax = axs[idx]
    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1, -1])]
    df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]
    if len(df_valid_animal_less_than_1) == 0:
        ax.set_title(f'Animal {animal}\n(No data)')
        ax.axis('off')
        continue
    ILD_arr = np.sort(df_valid_animal_less_than_1['ILD'].unique())
    psycho_dict = plot_psycho(df_valid_animal_less_than_1, ILD_arr)
    for key in psycho_dict.keys():
        ax.plot(ILD_arr, psycho_dict[key], 'o', label=f'ABL={key}')
    ax.set_ylim(0, 1)
    ax.set_title(f'Animal {animal}')
    ax.legend(fontsize=8)

# %%

# --- Plotting function for all animals in subplots using exp_df_batch ---
def plot_all_animals_psycho(exp_df, session_type, batch_name, fig_title):
    exp_df_batch = exp_df[
        (exp_df['batch_name'] == batch_name) &
        (exp_df['LED_trial'].isin([np.nan, 0])) &
        (exp_df['session_type'] == session_type)
    ].copy()

    exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
    exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)

    df_valid_and_aborts = exp_df_batch[
        (exp_df_batch['success'].isin([1, -1])) |
        (exp_df_batch['abort_event'] == 3)
    ].copy()

    animal_ids = np.sort(df_valid_and_aborts['animal'].unique())
    n_animals = len(animal_ids)
    n_cols = 4
    n_rows = int(np.ceil(n_animals / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey=True)
    axs = axs.flatten()
    for idx, animal in enumerate(animal_ids):
        ax = axs[idx]
        df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
        df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1, -1])]
        df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]
        if len(df_valid_animal_less_than_1) == 0:
            ax.set_title(f'Animal {animal}\n(No data)')
            ax.axis('off')
            continue
        ILD_arr = np.sort(df_valid_animal_less_than_1['ILD'].unique())
        psycho_dict = plot_psycho(df_valid_animal_less_than_1, ILD_arr)
        for key in psycho_dict.keys():
            ax.plot(ILD_arr, psycho_dict[key], 'o', label=f'ABL={key}')
        ax.set_ylim(0, 1)
        ax.set_title(f'Animal {animal}')
        ax.legend(fontsize=8)
    # Hide unused subplots
    for j in range(idx + 1, len(axs)):
        axs[j].axis('off')
    fig.suptitle(fig_title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # Plot for session_type 1
    plot_all_animals_psycho(exp_df, session_type=1, batch_name=batch_name, fig_title='Psychometric (session_type=1)')
    # Plot for session_type 7
    plot_all_animals_psycho(exp_df, session_type=7, batch_name=batch_name, fig_title='Psychometric (session_type=7)')
