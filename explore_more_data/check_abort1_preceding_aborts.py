import pandas as pd

# Load the same data as in glm_all_T.py
exp_df = pd.read_csv('/home/rlab/raghavendra/ddm_data/outExp.csv')
exp_df = exp_df[~((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3))].copy()
all_df = exp_df[(exp_df['batch_name'] == 'Comparable')]

results = []

for animal in all_df['animal'].unique():
    animal_df = all_df[all_df['animal'] == animal].copy()
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
    animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
    animal_df['abs_ILD'] = animal_df['ILD'].abs()
    animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() if x.max() > x.min() else 1))
    
    # Create all lagged columns as in glm_all_T.py
    lagged_vars = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime', 'MT']
    for var in lagged_vars:
        colname = f'{var}_1' if var != 'is_abort' else 'abort_1'
        base_col = var
        animal_df[colname] = animal_df.groupby('session')[base_col].shift(1)

    # Before dropping any NaNs
    aborts_with_prev_abort = animal_df[(animal_df['is_abort'] == 1) & (animal_df['abort_1'] == 1)]
    total_aborts = (animal_df['is_abort'] == 1).sum()
    n_aborts_with_prev_abort = len(aborts_with_prev_abort)

    lag_cols = [f'{var}_1' if var != 'is_abort' else 'abort_1' for var in lagged_vars]
    animal_df_lagged = animal_df.dropna(subset=lag_cols).copy()

    # After dropping NaNs
    aborts_with_prev_abort_post = animal_df_lagged[(animal_df_lagged['is_abort'] == 1) & (animal_df_lagged['abort_1'] == 1)]
    total_aborts_post = (animal_df_lagged['is_abort'] == 1).sum()
    n_aborts_with_prev_abort_post = len(aborts_with_prev_abort_post)

    results.append({
        'animal': animal,
        'total_aborts': total_aborts,
        'aborts_with_prev_abort': n_aborts_with_prev_abort,
        'total_aborts_post': total_aborts_post,
        'aborts_with_prev_abort_post': n_aborts_with_prev_abort_post
    })

results_df = pd.DataFrame(results)
print(results_df)
print("\nIf 'aborts_with_prev_abort' > 0 but 'aborts_with_prev_abort_post' == 0, those rows are being dropped by dropna.")
