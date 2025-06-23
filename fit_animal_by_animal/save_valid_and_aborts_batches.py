# %%
import os
import pandas as pd
import random

# List of batches to process (excluding 'Comparable' and any LED.csv)

# Map batch to file
batch_file_map = {
    'SD': '../outExp.csv',
    'LED1': '../outExp.csv',
    'LED2': '../outExp.csv',
    'LED34': '../outExp.csv',
    'Comparable': '../outExp.csv',
    'LED6': '../outExp.csv',
    'LED7': '../out_LED.csv',
    'LED8': '../outLED8.csv'
}

batch_names = batch_file_map.keys()
# batch_names = ['SD', 'LED1', 'LED2', 'LED34', 'Comparable', 'LED6', 'LED7']

# Output directory
output_dir = 'batch_csvs'
os.makedirs(output_dir, exist_ok=True)

for batch_name in batch_names:
    print(f'Processing batch: {batch_name}')
    exp_df = pd.read_csv(batch_file_map[batch_name])

    # If needed, reconstruct RTwrtStim and rename timed_fix
    if 'timed_fix' in exp_df.columns:
        exp_df.loc[:, 'RTwrtStim'] = exp_df['timed_fix'] - exp_df['intended_fix']
        exp_df = exp_df.rename(columns={'timed_fix': 'TotalFixTime'})
    # Remove problematic rows
    exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

    # Reconstruct response_poke if missing
    mask_nan = exp_df['response_poke'].isna()
    mask_success_1 = (exp_df['success'] == 1)
    mask_success_neg1 = (exp_df['success'] == -1)
    mask_ild_pos = (exp_df['ILD'] > 0)
    mask_ild_neg = (exp_df['ILD'] < 0)
    exp_df.loc[mask_nan & mask_success_1 & mask_ild_pos, 'response_poke'] = 3
    exp_df.loc[mask_nan & mask_success_1 & mask_ild_neg, 'response_poke'] = 2
    exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_pos, 'response_poke'] = 2
    exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_neg, 'response_poke'] = 3

    # Add choice and accuracy columns to exp_df (applies to all batches)
    exp_df['choice'] = exp_df['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
    exp_df['accuracy'] = (exp_df['ILD'] * exp_df['choice']).apply(lambda x: 1 if x > 0 else 0)

    # Apply batch-specific filters (copied from make_a_dict_with_num_of_animals.py)
    if batch_name == 'SD':
        exp_df_batch = exp_df[
            (exp_df['batch_name'] == batch_name) &
            ((exp_df['LED_trial'].isin([float('nan'), 0]) | exp_df['LED_trial'].isna())) &
            (exp_df['session_type'].isin([1,7]))
        ].copy()
    elif batch_name == 'LED1':
        exp_df_batch = exp_df[
            (exp_df['batch_name'] == batch_name) &
            ((exp_df['LED_trial'].isin([float('nan'), 0]) | exp_df['LED_trial'].isna())) &
            (exp_df['session_type'].isin([1]))
        ].copy()
    elif batch_name == 'LED2':
        exp_df_batch = exp_df[
            (exp_df['batch_name'] == batch_name) &
            ((exp_df['LED_trial'].isin([float('nan'), 0]) | exp_df['LED_trial'].isna())) &
            (exp_df['session_type'].isin([1,2]))
        ].copy()
    elif batch_name == 'LED34':
        allowed_animals = [45,57,59,61,63]
        exp_df_batch = exp_df[
            (exp_df['batch_name'] == batch_name) &
            ((exp_df['LED_trial'].isin([float('nan'), 0]) | exp_df['LED_trial'].isna())) &
            (exp_df['session_type'].isin([1,2])) &
            (exp_df['animal'].isin(allowed_animals))
        ].copy()
    elif batch_name =='Comparable':
        exp_df_batch = exp_df[
            (exp_df['batch_name'] == batch_name) &
            ((exp_df['LED_trial'].isin([float('nan'), 0]) | exp_df['LED_trial'].isna())) &
            (exp_df['session_type'].isin([6]))
        ].copy()
    elif batch_name == 'LED6':
        exp_df_batch = exp_df[
            (exp_df['batch_name'] == batch_name) &
            ((exp_df['LED_trial'].isin([float('nan'), 0]) | exp_df['LED_trial'].isna())) &
            (exp_df['session_type'].isin([1,2]))
        ].copy()      
    elif batch_name == 'LED7':
        exp_df_batch = exp_df[
            ((exp_df['LED_trial'].isin([float('nan'), 0]) | exp_df['LED_trial'].isna())) &
            (exp_df['session_type'].isin([7])) &
            (exp_df['training_level'].isin([16])) &
            (exp_df['repeat_trial'].isin([0,2]) | exp_df['repeat_trial'].isna())
        ].copy() 
        # add batch_name column
        exp_df_batch['batch_name'] = batch_name     
    elif batch_name == 'LED8':
        exp_df_batch = exp_df[
            ((exp_df['LED_trial'].isin([float('nan'), 0]) | exp_df['LED_trial'].isna())) &
            (exp_df['session_type'].isin([1])) &
            (exp_df['training_level'].isin([16])) &
            (exp_df['repeat_trial'].isin([0,2]) | exp_df['repeat_trial'].isna())

        ].copy() 
        # add batch_name column
        exp_df_batch['batch_name'] = batch_name     
    else:
        raise ValueError(f"Unknown batch_name: {batch_name}")

    # Get only valid and aborts
    df_valid_and_aborts = exp_df_batch[
        (exp_df_batch['success'].isin([1, -1])) |
        (exp_df_batch['abort_event'] == 3)
    ].copy()

    # Save to CSV
    out_path = os.path.join(output_dir, f'batch_{batch_name}_valid_and_aborts.csv')
    df_valid_and_aborts.to_csv(out_path, index=False)
    print(f'Saved {out_path} ({len(df_valid_and_aborts)} rows)')

print('Done!')
