# %%
# 1. correlations
# 2. corr btn input and output
# 3. check code and see

import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# %%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

# --- Helper function moved to top level --- 
def add_k_previous_trial_value_explicit_lookup(df: pd.DataFrame, 
                                               value_col_name: str, 
                                               k: int, 
                                               session_col: str = 'session', 
                                               trial_col: str = 'trial') -> pd.DataFrame:
    """
    Adds a new column with the value of 'value_col_name' from the trial whose
    ID is (current_trial_id - k) within the same session, using an explicit lookup.

    If a trial with ID (current_trial_id - k) does not exist in that session,
    or if the value_col_name for that trial is NaN, the new column's value 
    will be NaN.

    Args:
        df (pd.DataFrame): Input DataFrame.
        value_col_name (str): The name of the column whose past value is needed.
        k (int): The number of trials to look back (e.g., k=1 means current_trial_id - 1).
                 Must be a positive integer.
        session_col (str): The name of the column identifying the session.
                           Defaults to 'session'.
        trial_col (str): The name of the column identifying the trial number.
                         Defaults to 'trial'.

    Returns:
        pd.DataFrame: DataFrame with the new column added.
                      The new column will be named f"{value_col_name}_{k}".
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not all(col in df.columns for col in [session_col, trial_col, value_col_name]):
        raise ValueError(f"Ensure '{session_col}', '{trial_col}', and '{value_col_name}' are columns in the DataFrame.")
    if not isinstance(k, int) or k < 1:
        raise ValueError("'k' must be a positive integer.")

    # Make a copy to work on. This preserves the original DataFrame's index.
    working_df = df.copy()
    new_col_name = f"{value_col_name}_{k}"

    # 1. In our working_df, create a column that represents the trial ID we are looking for.
    working_df['__target_prev_trial_id__'] = working_df[trial_col] - k

    # 2. Prepare a "source" DataFrame. This contains the actual trial IDs and their corresponding values.
    #    We will merge this back onto our working_df.
    #    - Select only necessary columns: session, trial, and the value we want.
    #    - Rename value_col_name to new_col_name so it appears correctly after merge.
    #    - Rename trial_col to something unique to avoid clashes during merge.
    df_source_values = df[[session_col, trial_col, value_col_name]].copy()
    df_source_values = df_source_values.rename(columns={
        value_col_name: new_col_name, 
        trial_col: '__source_actual_trial_id__'
    })
    
    # Crucial: Ensure the source for lookup is unique on its merge keys (session, __source_actual_trial_id__)
    # If multiple rows exist for the same (session, trial_id) in the original data,
    # this drop_duplicates will pick one (the first encountered).
    # This prevents row explosion if a target_prev_trial_id matches multiple source rows.
    df_source_values = df_source_values.drop_duplicates(
        subset=[session_col, '__source_actual_trial_id__'], 
        keep='first'
    )

    # 3. Perform a left merge.
    #    We merge working_df (left) with df_source_values (right).
    #    - Match on session.
    #    - Match working_df's '__target_prev_trial_id__' with df_source_values's '__source_actual_trial_id__'.
    merged_df = pd.merge(
        working_df,
        df_source_values, # Contains {session_col, '__source_actual_trial_id__', new_col_name}
        left_on=[session_col, '__target_prev_trial_id__'],
        right_on=[session_col, '__source_actual_trial_id__'],
        how='left' # Keep all rows from working_df; add new_col_name where match found
    )
    
    # 4. Clean up.
    #    The `merged_df` now has all original columns from `working_df`, 
    #    plus `__target_prev_trial_id__` (from `working_df`),
    #    plus `new_col_name` (populated by the merge from `df_source_values`),
    #    plus `__source_actual_trial_id__` (from `df_source_values`).
    #    We need to drop the helper columns.

    columns_to_drop = ['__target_prev_trial_id__']
    # Check if '__source_actual_trial_id__' exists before trying to drop (it might not if all merges failed)
    if '__source_actual_trial_id__' in merged_df.columns:
        columns_to_drop.append('__source_actual_trial_id__')
    
    result_df = merged_df.drop(columns=columns_to_drop)

    return result_df

# For backwards compatibility with the original function
def add_lagged_column(df, col, k):
    result_df = add_k_previous_trial_value_explicit_lookup(df, col, k)
    # Extract only the new column to match the behavior of the original function
    return result_df[f"{col}_{k}"]

# --- Correlation plotting function ---
def plot_predictor_correlations_per_animal(df, lag_cols, animal_col='animal', pdf_name='animal_predictor_correlations.pdf'):
    """
    For each animal, plot the correlation matrix of predictors and save each as a page in a single PDF.
    """
    animals = df[animal_col].unique()
    with PdfPages(pdf_name) as pdf:
        for animal in animals:
            animal_df = df[df[animal_col] == animal]
            # Drop rows with NA in predictors
            sub_df = animal_df[lag_cols].dropna()
            if len(sub_df) == 0:
                continue
            corr = sub_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
            plt.title(f'Predictor Correlations: Animal {animal}')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

def run_analysis():
    # Fix path to use absolute path within project
    exp_df = pd.read_csv('/home/rlab/raghavendra/ddm_data/outExp.csv')
    # %%
    # Remove wrong rows 
    count = ((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3)).sum()
    print("Number of rows where TotalFixTime is NaN and abort_event == 3:", count)
    exp_df = exp_df[~((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3))].copy()

    # Comparable batch
    all_df = exp_df[(exp_df['batch_name'] == 'Comparable')]

    # Prepare figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    from matplotlib.backends.backend_pdf import PdfPages

    pdf = PdfPages('animal_logit_results.pdf')

    for idx, T in enumerate([1, 2, 3]):
        results = []
        animal_summaries = []  # Store for PDF
        for animal in all_df['animal'].unique():
            animal_df = all_df[all_df['animal'] == animal].copy()
            animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
            animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
            animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
            animal_df['abs_ILD'] = animal_df['ILD'].abs()

            animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() if x.max() > x.min() else 1))
            
            # CNPTime added as per request, TotalFixTime already included (replaced RTwrtStim)
            lagged_vars = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']
            for k in range(1, T + 1):
                for var in lagged_vars:
                    colname = f'{var}_{k}' if var != 'is_abort' else f'abort_{k}'
                    base_col = var
                    # Use the new function to generate the lagged column
                    result_df = add_k_previous_trial_value_explicit_lookup(animal_df, base_col, k)
                    # Extract the new column and add it to animal_df with the correct name
                    animal_df[colname] = result_df[f"{base_col}_{k}"]

            
            # Updated lag_cols to match the updated lagged_vars list (includes CNPTime)
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
            
            # Updated predictor_cols to include CNPTime
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
            # Always use penalty='l2' with very large C for no regularization (for sklearn compatibility)
            model = LogisticRegression(penalty='l2', C=1e12, solver='lbfgs', max_iter=5000, class_weight='balanced', fit_intercept=True)
            model.fit(X_train, y_train)

            # Fit statsmodels Logit for significance without regularization to avoid convergence warnings
            X_train_sm = sm.add_constant(X_train)
            logit_model = sm.Logit(y_train, X_train_sm)
            try:
                # Use regular MLE fit instead of regularized fit
                # With improved convergence parameters
                result = logit_model.fit(method='newton', maxiter=1000, disp=0, tol=1e-8)
                sm_coefs = result.params
            except Exception as e:
                print(f"Statsmodels Logit failed for animal {animal}: {e}")
                sm_coefs = None

            # Prepare side-by-side table (save for PDF)
            coef_table = []
            coef_table.append(["Predictor", "Sklearn Coef", "Statsmodels Coef"])
            coef_table.append(["Intercept/const", f"{model.intercept_[0]:.6f}", f"{sm_coefs['const'] if sm_coefs is not None else 'NA'}"])
            for name, coef in zip(predictor_cols, model.coef_[0]):
                sm_val = sm_coefs[name] if (sm_coefs is not None and name in sm_coefs) else "NA"
                coef_table.append([name, f"{coef:.6f}", f"{sm_val}"])

            # Save coefficients table to PDF (page 1 for animal)
            fig, ax = plt.subplots(figsize=(8.5, 0.5+0.3*len(coef_table)))
            ax.axis('off')
            table = ax.table(cellText=coef_table, loc='center', cellLoc='center', colLabels=None, edges='horizontal')
            ax.set_title(f"Animal {animal} | T={T}\nLogistic Regression Coefficients", fontsize=14, pad=20)
            pdf.savefig(fig)
            plt.close(fig)

            # Save statsmodels summary to PDF (page 2 for animal)
            if sm_coefs is not None:
                summary_text = result.summary().as_text()
            else:
                summary_text = "Statsmodels Logit failed."
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            fig.text(0.01, 0.98, f"Animal {animal} | T={T} | Statsmodels Summary", fontsize=14, va='top', ha='left')
            fig.text(0.01, 0.95, summary_text, fontsize=9, va='top', ha='left', family='monospace')
            pdf.savefig(fig)
            plt.close(fig)

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

    # Save the ROC-AUC plot to the PDF as the last page
    fig = plt.gcf()
    pdf.savefig(fig)
    pdf.close()

    plt.suptitle('Train and Test ROC-AUC per Animal (T=1,2,3)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    run_analysis()
    # If you want to also generate the correlation plots when running directly:
    # print("Generating correlation plots...")
    # temp_df_for_corr = pd.read_csv('/home/rlab/raghavendra/ddm_data/outExp.csv')
    # temp_df_for_corr = temp_df_for_corr[temp_df_for_corr['batch_name'] == 'Comparable']
    # # Define lag_cols appropriately here if needed, e.g., for T=1 or a fixed T
    # # This part needs careful definition of lag_cols based on what you want to plot by default
    # # Example for T=1:
    # lag_cols_example = [f'{var}_1' for var in ['rewarded', 'abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']]
    # plot_predictor_correlations_per_animal(temp_df_for_corr, lag_cols_example)
    # print("Correlation plots saved to animal_predictor_correlations.pdf")
    pass # Placeholder if you don't want to run correlations by default
