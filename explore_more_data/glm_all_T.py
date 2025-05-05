# %%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

exp_df = pd.read_csv('../outExp.csv')

# Remove wrong rows 
count = ((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3)).sum()
print("Number of rows where RTwrtStim is NaN and abort_event == 3:", count)
exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

# Comparable batch
all_df = exp_df[(exp_df['batch_name'] == 'Comparable')]

# Prepare figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for idx, T in enumerate([1, 2, 3]):
    results = []
    for animal in all_df['animal'].unique():
        animal_df = all_df[all_df['animal'] == animal].copy()
        animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
        animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
        animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
        animal_df['LED_trial'] = animal_df['LED_trial'].fillna(0).astype(int)
        animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() if x.max() > x.min() else 1))
        
        for k in range(1, T + 1):
            animal_df[f'rewarded_{k}'] = animal_df.groupby('session')['rewarded'].shift(k)
            animal_df[f'abort_{k}'] = animal_df.groupby('session')['is_abort'].shift(k)
            animal_df[f'short_poke_{k}'] = animal_df.groupby('session')['short_poke'].shift(k)
            animal_df[f'intended_fix_{k}'] = animal_df.groupby('session')['intended_fix'].shift(k)
            animal_df[f'ILD_{k}'] = animal_df.groupby('session')['ILD'].shift(k)
            animal_df[f'ABL_{k}'] = animal_df.groupby('session')['ABL'].shift(k)
            animal_df[f'LED_trial_{k}'] = animal_df.groupby('session')['LED_trial'].shift(k)
            animal_df[f'RTwrtStim_{k}'] = animal_df.groupby('session')['RTwrtStim'].shift(k)
        
        lag_cols = [f'{var}_{k}' for k in range(1, T + 1)
                    for var in ['rewarded', 'abort', 'short_poke', 'intended_fix', 'ILD', 'ABL', 'LED_trial', 'RTwrtStim']]
        
        animal_df = animal_df.dropna(subset=lag_cols).reset_index(drop=True)
        animal_df = animal_df[animal_df['is_abort'] == 1].copy()
        keep_cols = lag_cols + ['short_poke', 'session', 'animal', 'norm_trial']
        animal_df = animal_df[keep_cols].copy()
        
        sessions = animal_df['session'].unique()
        np.random.shuffle(sessions)
        n_sessions = len(sessions)
        n_train = int(n_sessions * 0.7)
        n_test = n_sessions - n_train
        train_sessions = sessions[:n_train]
        test_sessions = sessions[n_train:]
        
        train_df = animal_df[animal_df['session'].isin(train_sessions)].copy()
        test_df = animal_df[animal_df['session'].isin(test_sessions)].copy()
        
        predictor_cols = [f'{var}_{k}' for k in range(1, T + 1)
                          for var in ['rewarded', 'abort', 'short_poke', 'intended_fix', 'ILD', 'ABL', 'LED_trial', 'RTwrtStim']]
        predictor_cols += ['norm_trial']
        
        X_train = train_df[predictor_cols]
        y_train = train_df['short_poke']
        X_test = test_df[predictor_cols]
        y_test = test_df['short_poke']
        
        model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, class_weight='balanced')
        model.fit(X_train, y_train)
        
        y_train_prob = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
        
        test_auc = None
        if len(test_df) > 0:
            y_test_prob = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_test_prob)
        
        results.append({'animal': animal, 'train_auc': train_auc, 'test_auc': test_auc})
    
    # === Plotting in subplot ===
    df_res = pd.DataFrame(results)
    animals = df_res['animal'].astype(str)
    train_aucs = df_res['train_auc']
    test_aucs = df_res['test_auc']
    
    x = np.arange(len(animals))
    width = 0.35
    
    ax = axs[idx]
    ax.bar(x - width/2, train_aucs, width, label='Train ROC-AUC')
    ax.bar(x + width/2, test_aucs, width, label='Test ROC-AUC')
    
    ax.set_xlabel('Animal')
    ax.set_ylabel('ROC-AUC')
    ax.set_title(f'T = {T}')
    ax.set_xticks(x)
    ax.set_xticklabels(animals)
    ax.set_ylim(0, 1)
    if idx == 0:
        ax.legend()

plt.suptitle('Train and Test ROC-AUC per Animal (T=1,2,3)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
