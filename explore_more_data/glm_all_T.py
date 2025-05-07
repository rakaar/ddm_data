# TODO
# 1. correlations
# 2. corr btn input and output
# 3. check code and see

# %%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

exp_df = pd.read_csv('../outExp.csv')
# %%

# Remove wrong rows 
count = ((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3)).sum()
print("Number of rows where TotalFixTime is NaN and abort_event == 3:", count)
exp_df = exp_df[~((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3))].copy()

# Comparable batch
all_df = exp_df[(exp_df['batch_name'] == 'Comparable')]

# Prepare figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for idx, T in enumerate([1]):
    results = []
    for animal in all_df['animal'].unique():
        animal_df = all_df[all_df['animal'] == animal].copy()
        animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
        animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
        animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
        animal_df['abs_ILD'] = animal_df['ILD'].abs()

        animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() if x.max() > x.min() else 1))
        
        # Robust lagged variable creation using trial numbers within each session
        def add_lagged_column(df, col, k):
            # Use groupby().shift() to create lagged columns efficiently and warning-free
            return df.groupby('session')[col].shift(k)

        # Removed MT from lagged variables
        # MT was removed to prevent systematic removal of rows where abort_1 = 1
        lagged_vars = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']
        for k in range(1, T + 1):
            for var in lagged_vars:
                colname = f'{var}_{k}' if var != 'is_abort' else f'abort_{k}'
                base_col = var
                animal_df[colname] = add_lagged_column(animal_df, base_col, k)

        
        # Updated lag_cols to match the updated lagged_vars list (removed MT)
        lag_cols = [f'{var}_{k}' for k in range(1, T + 1)
                    for var in ['rewarded', 'abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']]
        
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
        
        # Updated predictor_cols to remove MT
        predictor_cols = [f'{var}_{k}' for k in range(1, T + 1)
                          for var in ['rewarded', 'abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']]
        predictor_cols += ['norm_trial']
        
        X_train = train_df[predictor_cols]
        y_train = train_df['short_poke']
        X_test = test_df[predictor_cols]
        y_test = test_df['short_poke']
        
        # Check constant columns (zero variance)
        constant_cols = [col for col in X_train.columns if X_train[col].nunique() <= 1]
        if constant_cols:
            print(f"Animal {animal} → Dropping constant columns: {constant_cols}")
            X_train = X_train.drop(columns=constant_cols)
            X_test = X_test.drop(columns=constant_cols, errors='ignore')

        # Check highly correlated predictors (>0.95)
        corr_matrix = X_train.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        if high_corr_cols:
            print(f"Animal {animal} → Dropping highly correlated columns: {high_corr_cols}")
            X_train = X_train.drop(columns=high_corr_cols)
            X_test = X_test.drop(columns=high_corr_cols, errors='ignore')

        # After dropping constant and highly correlated columns, check if X_train is empty
        if X_train.shape[1] == 0 or X_train.shape[0] == 0:
            print(f"Animal {animal} → Skipping model fit: no non-constant/correlated predictors remain after filtering.")
            results.append({'animal': animal, 'train_auc': None, 'test_auc': None})
            continue

        # Always use penalty='l2' with very large C for no regularization (for sklearn compatibility)
        model = LogisticRegression(penalty='l2', C=1e12, solver='lbfgs', max_iter=5000, class_weight='balanced', fit_intercept=True)
        model.fit(X_train, y_train)

        # Fit statsmodels Logit for significance with L2 regularization (very small alpha)
        X_train_sm = sm.add_constant(X_train)
        logit_model = sm.Logit(y_train, X_train_sm)
        try:
            result = logit_model.fit_regularized(method='l1', alpha=1e-12)
            sm_coefs = result.params
        except Exception as e:
            print(f"Statsmodels Logit failed for animal {animal}: {e}")
            sm_coefs = None

        # Prepare side-by-side table
        print(f'\nAnimal: {animal}')
        print('Predictor                 | Sklearn Coef   | Statsmodels Coef')
        print('--------------------------+---------------+-------------------')
        # Intercept/const first
        print(f'{"Intercept/const":26} | {model.intercept_[0]:13.6f} | {sm_coefs["const"] if sm_coefs is not None else "NA":17}')
        # Now all predictors
        for name, coef in zip(predictor_cols, model.coef_[0]):
            sm_val = sm_coefs[name] if (sm_coefs is not None and name in sm_coefs) else "NA"
            print(f'{name:26} | {coef:13.6f} | {sm_val:17}')
        if sm_coefs is not None:
            print("\nStatsmodels summary:")
            print(result.summary())

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


# %%
