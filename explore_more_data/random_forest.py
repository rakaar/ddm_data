import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

exp_df = pd.read_csv('../outExp.csv')

# Remove wrong rows 
count = ((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3)).sum()
print("Number of rows where RTwrtStim is NaN and abort_event == 3:", count)
exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

# Comparable batch
all_df = exp_df[(exp_df['batch_name'] == 'Comparable')]

results = []
for animal in all_df['animal'].unique():
    animal_df = all_df[all_df['animal'] == animal].copy()
    animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
    animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
    animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
    animal_df['LED_trial'] = animal_df['LED_trial'].fillna(0).astype(int)
    animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() if x.max() > x.min() else 1))
    
    T = 10
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
    
    # Keep only abort trials AFTER lagging
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
    
    # 🔥 RANDOM FOREST MODEL
    model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced',  # handles imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    
    acc = accuracy_score(y_train, y_train_pred)
    auc = roc_auc_score(y_train, y_train_prob)
    precision = precision_score(y_train, y_train_pred)
    recall = recall_score(y_train, y_train_pred)
    f1 = f1_score(y_train, y_train_pred)
    cm = confusion_matrix(y_train, y_train_pred)
    
    print("=" * 30)
    print(f"Animal {animal} | Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Train → Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print(f"Train Confusion Matrix:\n{cm}")
    
    test_acc, test_auc, test_precision, test_recall, test_f1, test_cm = [None] * 6
    if len(test_df) > 0:
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        test_acc = accuracy_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_prob)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
        
        print(f"Test → Accuracy: {test_acc:.3f}, ROC-AUC: {test_auc:.3f}, Precision: {test_precision:.3f}, Recall: {test_recall:.3f}, F1: {test_f1:.3f}")
        print(f"Test Confusion Matrix:\n{test_cm}")
    
    results.append({
        'animal': animal,
        'train_acc': acc,
        'train_auc': auc,
        'train_precision': precision,
        'train_recall': recall,
        'train_f1': f1,
        'train_cm': cm,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_cm': test_cm,
        'feature_importances': model.feature_importances_
    })
