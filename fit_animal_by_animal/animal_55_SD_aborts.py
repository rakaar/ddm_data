# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random



batch_name = 'SD'
session_type = [1,7]
exp_df = pd.read_csv('../outExp.csv')
exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

exp_df_batch = exp_df[
        (exp_df['batch_name'] == batch_name) &
        (exp_df['LED_trial'].isin([np.nan, 0])) &
        (exp_df['session_type'].isin(session_type))
    ].copy()

exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)

df_valid_and_aborts = exp_df_batch[
    (exp_df_batch['success'].isin([1, -1])) |
    (exp_df_batch['abort_event'] == 3)
].copy()

animal_ids = np.sort(df_valid_and_aborts['animal'].unique())
animal_idx = -1

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]
animal = animal_ids[animal_idx]
df_aborts_animal = df_aborts[df_aborts['abort_event'] == 3]

plt.hist(df_aborts_animal['TotalFixTime'], bins=np.arange(0, 2, 0.02));
plt.axvline(0.3)

print(animal)
