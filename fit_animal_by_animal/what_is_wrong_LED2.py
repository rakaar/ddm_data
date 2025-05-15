# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Load the data
exp_df = pd.read_csv('../outExp.csv')
if 'timed_fix' in exp_df.columns:
    exp_df.loc[:, 'RTwrtStim'] = exp_df['timed_fix'] - exp_df['intended_fix']
    exp_df = exp_df.rename(columns={'timed_fix': 'TotalFixTime'})
# 
# remove rows where abort happened, and RT is nan
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

exp_df_batch = exp_df[
    (exp_df['batch_name'] == 'LED2') &
    (exp_df['LED_trial'].isin([np.nan, 0])) &
    (exp_df['session_type'].isin([1,2]))
].copy()

# aborts don't have choice, so assign random 
exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else random.choice([1, -1])))
# 1 or 0 if the choice was correct or not
exp_df_batch['accuracy'] = (exp_df_batch['ILD'] * exp_df_batch['choice']).apply(lambda x: 1 if x > 0 else 0)

# %%
### DF - valid and aborts ###
df_valid_and_aborts = exp_df_batch[
    (exp_df_batch['success'].isin([1,-1])) |
    (exp_df_batch['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]

# %%
df_valid = df_valid_and_aborts[df_valid_and_aborts['success'].isin([1,-1])].copy()
