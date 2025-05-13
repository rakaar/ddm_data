import pandas as pd
import pickle
import numpy as np
import random
###### Cases ###########
# ../outExp.csv
# batch_name == Comparable, session_type isin [6]
# batch_name == SD, session_type isin [1,7]
# batch_name == LED1, session_type isin [1]
# batch_name == LED2, session_type isin [1,2]
# batch_name = LED34, session_type isin [1,2], animal isin [45,57,59,61,63]
# 


# Set batch_name to one of: 'LED7', 'Comparable', 'SD', 'LED1', 'LED2', 'LED34'
batch_name = 'LED7'  # Change this as needed

if batch_name == 'LED7':
    exp_df = pd.read_csv('../out_LED.csv')
else:
    exp_df = pd.read_csv('../outExp.csv')

if 'timed_fix' in exp_df.columns:
    exp_df.loc[:, 'RTwrtStim'] = exp_df['timed_fix'] - exp_df['intended_fix']
    exp_df = exp_df.rename(columns={'timed_fix': 'TotalFixTime'})
exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

# in some cases, response_poke is nan, but succcess and ILD are present
import os

batch_names = ['LED7', 'Comparable', 'SD', 'LED1', 'LED2', 'LED34']

for batch_name in batch_names:
    if batch_name == 'LED7':
        exp_df = pd.read_csv('../out_LED.csv')
    else:
        exp_df = pd.read_csv('../outExp.csv')

    if 'timed_fix' in exp_df.columns:
        exp_df.loc[:, 'RTwrtStim'] = exp_df['timed_fix'] - exp_df['intended_fix']
        exp_df = exp_df.rename(columns={'timed_fix': 'TotalFixTime'})
    exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

    # in some cases, response_poke is nan, but succcess and ILD are present
    # reconstruct response_poke from success and ILD 
    # if success and ILD are present, a response should also be present
    mask_nan = exp_df['response_poke'].isna()
    mask_success_1 = (exp_df['success'] == 1)
    mask_success_neg1 = (exp_df['success'] == -1)
    mask_ild_pos = (exp_df['ILD'] > 0)
    mask_ild_neg = (exp_df['ILD'] < 0)

    # For success == 1
    exp_df.loc[mask_nan & mask_success_1 & mask_ild_pos, 'response_poke'] = 3
    exp_df.loc[mask_nan & mask_success_1 & mask_ild_neg, 'response_poke'] = 2

    # For success == -1
    exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_pos, 'response_poke'] = 2
    exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_neg, 'response_poke'] = 3

    # Filtering for each batch
    if batch_name == 'LED7':
        exp_df_batch = exp_df[
            (exp_df['LED_trial'].isin([np.nan, 0])) &
            (exp_df['session_type'].isin([7])) &
            (exp_df['training_level'].isin([16])) &
            (exp_df['repeat_trial'].isin([0,2,np.nan]))
        ].copy()
    elif batch_name == 'Comparable':
        exp_df_batch = exp_df[
            (exp_df['batch_name'] == batch_name) &
            (exp_df['LED_trial'].isin([np.nan, 0])) &
            (exp_df['session_type'].isin([6]))
        ].copy()
    elif batch_name == 'SD':
        exp_df_batch = exp_df[
            (exp_df['batch_name'] == batch_name) &
            (exp_df['LED_trial'].isin([np.nan, 0])) &
            (exp_df['session_type'].isin([1,7]))
        ].copy()
    elif batch_name == 'LED1':
        exp_df_batch = exp_df[
            (exp_df['batch_name'] == batch_name) &
            (exp_df['LED_trial'].isin([np.nan, 0])) &
            (exp_df['session_type'].isin([1]))
        ].copy()
    elif batch_name == 'LED2':
        exp_df_batch = exp_df[
            (exp_df['batch_name'] == batch_name) &
            (exp_df['LED_trial'].isin([np.nan, 0])) &
            (exp_df['session_type'].isin([1,2]))
        ].copy()
    elif batch_name == 'LED34':
        allowed_animals = [45,57,59,61,63]
        exp_df_batch = exp_df[
            (exp_df['batch_name'] == batch_name) &
            (exp_df['LED_trial'].isin([np.nan, 0])) &
            (exp_df['session_type'].isin([1,2])) &
            (exp_df['animal'].isin(allowed_animals))
        ].copy()
    else:
        raise ValueError(f"Unknown batch_name: {batch_name}")

    # aborts don't have choice, so assign random 
    exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
    # 1 or 0 if the choice was correct or not
    exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)

    # DF - valid and aborts
    df_valid_and_aborts = exp_df_batch[
        (exp_df_batch['success'].isin([1,-1])) |
        (exp_df_batch['abort_event'] == 3)
    ].copy()

    df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]

    animal_ids = df_valid_and_aborts['animal'].unique()
    all_animal_stats = {}
    for animal_idx in range(len(animal_ids)):
        animal = animal_ids[animal_idx]
        df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]

        df_aborts_animal = df_aborts[df_aborts['animal'] == animal]
        df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]
        df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]

        all_animal_stats[animal] = {
            'aborts_len': len(df_aborts_animal),
            'valid_len': len(df_valid_animal_less_than_1)
        }

    # Save all stats to a single pickle file
    outdir = 'animal_stats_pickles'
    pickle_filename = os.path.join(outdir, f'animal_stats_{batch_name}.pkl')
    os.makedirs(outdir, exist_ok=True)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(all_animal_stats, f)
    print(f"Saved animal stats to {pickle_filename}")
    print(all_animal_stats)
