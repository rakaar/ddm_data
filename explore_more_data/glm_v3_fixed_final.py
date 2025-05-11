# %%
# 1. correlations
# 2. corr btn input and output
# 3. check code and see

import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# %%
import numpy as np
import pandas as pd
import random # random is imported but not used. Consider removing if not needed elsewhere.
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

# --- Helper function: add_k_previous_trial_value_explicit_lookup ---
# This is the new function you provided, directly integrated here.
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
        # Check if value_col_name exists, if not, it might be an issue with upstream data prep
        if value_col_name not in df.columns:
             print(f"Warning: value_col_name '{value_col_name}' not found in DataFrame columns: {df.columns.tolist()}")
        # Continue with other checks or raise specific error
        missing_cols = [col for col in [session_col, trial_col, value_col_name] if col not in df.columns]
        if missing_cols: # Raise error if any critical column is missing
            raise ValueError(f"Ensure '{session_col}', '{trial_col}', and '{value_col_name}' are columns in the DataFrame. Missing: {missing_cols}")

    if not isinstance(k, int) or k < 1:
        raise ValueError("'k' must be a positive integer.")

    # Make a copy to work on. This preserves the original DataFrame's index.
    working_df = df.copy()
    new_col_name = f"{value_col_name}_{k}"

    working_df['__target_prev_trial_id__'] = working_df[trial_col] - k

    df_source_values = df[[session_col, trial_col, value_col_name]].copy()
    df_source_values = df_source_values.rename(columns={
        value_col_name: new_col_name, 
        trial_col: '__source_actual_trial_id__'
    })
    
    df_source_values = df_source_values.drop_duplicates(
        subset=[session_col, '__source_actual_trial_id__'], 
        keep='first'
    )

    merged_df = pd.merge(
        working_df,
        df_source_values,
        left_on=[session_col, '__target_prev_trial_id__'],
        right_on=[session_col, '__source_actual_trial_id__'],
        how='left'
    )
    
    columns_to_drop = ['__target_prev_trial_id__']
    if '__source_actual_trial_id__' in merged_df.columns:
        columns_to_drop.append('__source_actual_trial_id__')
    
    result_df = merged_df.drop(columns=columns_to_drop)

    return result_df

# --- Simpler function to add lagged columns using groupby and shift ---
def add_lagged_column(df, value_col_name, k=1, session_col='session'):
    df_copy = df.copy()
    new_col_name = f"{value_col_name}_{k}"
    
    # Group by session and shift the values
    df_copy[new_col_name] = df_copy.groupby(session_col)[value_col_name].shift(k)
    
    return df_copy[new_col_name]

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
            if len(sub_df) < 2: # Need at least 2 samples to compute correlation
                print(f"Skipping correlation plot for Animal {animal}: Not enough data after dropping NAs (samples: {len(sub_df)}).")
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
    
    # Remove wrong rows 
    count = ((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3)).sum()
    print(f"Number of rows where TotalFixTime is NaN and abort_event == 3: {count}")
    exp_df = exp_df[~((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3))].copy()

    # Comparable batch
    all_df = exp_df[(exp_df['batch_name'] == 'Comparable')].copy() # Use .copy() to avoid SettingWithCopyWarning later

    # Prepare figure for ROC-AUC plots
    fig_roc_auc, axs_roc_auc = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Define PDF for detailed results per animal
    pdf_animal_details = PdfPages('animal_logit_results_T123.pdf')

    # Define session and trial column names to be used
    SESSION_COL = 'session'
    TRIAL_COL = 'trial'

    for idx, T_max_lag in enumerate([1, 2, 3]): # Renamed T to T_max_lag for clarity
        results_auc = []
        
        for animal_id in all_df['animal'].unique():
            print(f"\nProcessing animal {animal_id} with T_max_lag={T_max_lag}")
            
            # Filter for the specific animal
            animal_df = all_df[all_df['animal'] == animal_id].copy()
            print(f"  Total rows for animal {animal_id}: {len(animal_df)}")
            
            animal_df['is_abort'] = (animal_df['abort_event'] == 3).astype(int)
            animal_df['short_poke'] = ((animal_df['is_abort'] == 1) & (animal_df['TotalFixTime'] < 0.3)).astype(int)
            animal_df['rewarded'] = (animal_df['success'] == 1).astype(int)
            animal_df['abs_ILD'] = animal_df['ILD'].abs()

            # Normalize trial number within each session
            # Check for sessions with only one trial to avoid division by zero or NaN issues
            animal_df['norm_trial'] = animal_df.groupby(SESSION_COL)[TRIAL_COL].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
            )
            
            # Use the simpler add_lagged_column function which uses groupby and shift
            # This is more reliable and creates fewer NaNs
            lagged_vars_to_create = ['rewarded', 'is_abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']
            
            for k_lag in range(1, T_max_lag + 1):
                for base_var_name in lagged_vars_to_create:
                    # Determine the final name for the new lagged column
                    final_lagged_col_name = f'{base_var_name}_{k_lag}'
                    if base_var_name == 'is_abort':
                        final_lagged_col_name = f'abort_{k_lag}'
                    
                    # Ensure base_var_name exists before trying to lag it
                    if base_var_name not in animal_df.columns:
                        print(f"Warning: Base column '{base_var_name}' not found in animal_df for animal {animal_id}, lag k={k_lag}. Skipping this lag.")
                        animal_df[final_lagged_col_name] = np.nan # Add as NaN to maintain column structure
                        continue

                    # Use the simpler function to generate the lagged column
                    animal_df[final_lagged_col_name] = add_lagged_column(
                        df=animal_df, 
                        value_col_name=base_var_name, 
                        k=k_lag, 
                        session_col=SESSION_COL
                    )

            # Define predictor columns based on the final_lagged_col_name logic
            current_predictor_cols = []
            for k_lag in range(1, T_max_lag + 1):
                for var_name_prefix in ['rewarded', 'abort', 'short_poke', 'intended_fix', 'abs_ILD', 'ABL', 'TotalFixTime', 'CNPTime']:
                    # 'abort' is the prefix for lagged 'is_abort'
                    current_predictor_cols.append(f'{var_name_prefix}_{k_lag}')
            
            # Add norm_trial as a predictor
            current_predictor_cols.append('norm_trial')

            # Drop rows with any NaN in the essential predictor columns before splitting
            # This ensures train/test sets are built from complete cases for these predictors
            animal_df_cleaned = animal_df.dropna(subset=current_predictor_cols).reset_index(drop=True)
            print(f"  Rows after NaN drop: {len(animal_df_cleaned)} ({len(animal_df_cleaned)/len(animal_df)*100:.2f}%)")
            
            # Filter for abort trials *after* creating lags (so lags can be from non-abort trials)
            # but *before* splitting and training, if the model is only for abort trials.
            animal_df_cleaned = animal_df_cleaned[animal_df_cleaned['is_abort'] == 1].copy()
            print(f"  Rows after NaN drop and abort filter: {len(animal_df_cleaned)} ({len(animal_df_cleaned)/len(animal_df)*100:.2f}%)")
                
            if animal_df_cleaned.empty:
                print(f"Animal {animal_id}, T_max_lag={T_max_lag} u2192 No data remaining after NaN drop and abort filter. Skipping.")
                results_auc.append({'animal': animal_id, 'train_auc': np.nan, 'test_auc': np.nan})
                continue
            
            # Define keep_cols based on current_predictor_cols and other necessary ones
            keep_cols = current_predictor_cols + ['short_poke', SESSION_COL, 'animal'] # 'animal' is animal_id
            # Ensure all keep_cols exist in animal_df_cleaned
            final_cols_to_keep = [col for col in keep_cols if col in animal_df_cleaned.columns]
            animal_df_model_ready = animal_df_cleaned[final_cols_to_keep].copy()
            
            sessions = animal_df_model_ready[SESSION_COL].unique()
            if len(sessions) < 2 and len(sessions) > 0 : # Need at least 2 sessions for a split, or handle single session case
                 print(f"Animal {animal_id}, T_max_lag={T_max_lag} u2192 Only one session ({sessions[0]}) with data. Using all for training, no test set.")
                 train_sessions = sessions
                 test_sessions = [] # No sessions for testing
            elif len(sessions) == 0:
                 print(f"Animal {animal_id}, T_max_lag={T_max_lag} u2192 No sessions with data. Skipping.")
                 results_auc.append({'animal': animal_id, 'train_auc': np.nan, 'test_auc': np.nan})
                 continue
            else:
                np.random.shuffle(sessions) # Shuffle for random split
                n_sessions = len(sessions)
                n_train = int(n_sessions * 0.7)
                # n_test = n_sessions - n_train # Unused
                train_sessions = sessions[:n_train]
                test_sessions = sessions[n_train:]
            
            train_df = animal_df_model_ready[animal_df_model_ready[SESSION_COL].isin(train_sessions)].copy()
            test_df = animal_df_model_ready[animal_df_model_ready[SESSION_COL].isin(test_sessions)].copy()
            
            X_train = train_df[current_predictor_cols]
            y_train = train_df['short_poke']
            
            if X_train.empty or y_train.empty:
                print(f"Animal {animal_id}, T_max_lag={T_max_lag} u2192 Training data is empty. Skipping model fit.")
                results_auc.append({'animal': animal_id, 'train_auc': np.nan, 'test_auc': np.nan})
                continue
            if y_train.nunique() < 2:
                print(f"Animal {animal_id}, T_max_lag={T_max_lag} u2192 Training target 'short_poke' has only one class. Skipping model fit.")
                results_auc.append({'animal': animal_id, 'train_auc': np.nan, 'test_auc': np.nan})
                continue


            # Prepare X_test, y_test (can be empty if no test_sessions)
            X_test = pd.DataFrame() # Initialize as empty
            y_test = pd.Series(dtype='int') # Initialize as empty

            if not test_df.empty:
                X_test = test_df[current_predictor_cols]
                y_test = test_df['short_poke']
                if y_test.empty or y_test.nunique() < 2: # If test set target is empty or single class
                    print(f"Animal {animal_id}, T_max_lag={T_max_lag} u2192 Test target 'short_poke' is empty or has only one class. Test AUC will be NaN.")
                    # X_test and y_test will be used, but roc_auc_score will handle it (often by error or NaN)

            
            # Check constant columns (zero variance) in X_train
            cols_to_drop_train = [col for col in X_train.columns if X_train[col].nunique(dropna=False) <= 1]
            if cols_to_drop_train:
                print(f"Animal {animal_id}, T_max_lag={T_max_lag} u2192 Dropping constant columns from X_train: {cols_to_drop_train}")
                X_train = X_train.drop(columns=cols_to_drop_train)
                if not X_test.empty:
                    X_test = X_test.drop(columns=cols_to_drop_train, errors='ignore')

            # Check highly correlated predictors (>0.95) in X_train
            if not X_train.empty:
                corr_matrix = X_train.corr().abs()
                upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_cols_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
                if high_corr_cols_to_drop:
                    print(f"Animal {animal_id}, T_max_lag={T_max_lag} u2192 Dropping highly correlated columns from X_train: {high_corr_cols_to_drop}")
                    X_train = X_train.drop(columns=high_corr_cols_to_drop)
                    if not X_test.empty:
                         X_test = X_test.drop(columns=high_corr_cols_to_drop, errors='ignore')
            
            if X_train.shape[1] == 0: # No predictors left
                print(f"Animal {animal_id}, T_max_lag={T_max_lag} u2192 No predictors remaining after filtering. Skipping model fit.")
                results_auc.append({'animal': animal_id, 'train_auc': np.nan, 'test_auc': np.nan})
                continue
            
            # Sklearn Logistic Regression
            # Using large C for minimal regularization, similar to statsmodels default
            model = LogisticRegression(penalty='l2', C=1e9, solver='lbfgs', max_iter=5000, class_weight='balanced', fit_intercept=True)
            model.fit(X_train, y_train)

            # Statsmodels Logit for p-values and detailed summary
            sm_coefs = None
            sm_pvalues = None
            sm_summary_text = "Statsmodels Logit failed or not run."
            try:
                X_train_sm = sm.add_constant(X_train, has_constant='add') # Add intercept
                logit_model_sm = sm.Logit(y_train, X_train_sm)
                logit_results_sm = logit_model_sm.fit(method='newton', maxiter=100, disp=0) # disp=0 to suppress convergence messages
                sm_coefs = logit_results_sm.params
                sm_pvalues = logit_results_sm.pvalues
                sm_summary_text = logit_results_sm.summary().as_text()
            except Exception as e:
                print(f"Statsmodels Logit failed for animal {animal_id}, T_max_lag={T_max_lag}: {e}")
                # Retain sm_summary_text as "Statsmodels Logit failed."

            # Prepare coefficient table for PDF
            coef_table_data = []
            coef_table_data.append(["Predictor", "Sklearn Coef", "Statsmodels Coef", "Statsmodels P-Value"])
            
            # Intercept / const
            sklearn_intercept = model.intercept_[0] if hasattr(model, 'intercept_') else np.nan
            sm_const_coef = sm_coefs['const'] if sm_coefs is not None and 'const' in sm_coefs else np.nan
            sm_const_pval = sm_pvalues['const'] if sm_pvalues is not None and 'const' in sm_pvalues else np.nan
            coef_table_data.append(["Intercept", f"{sklearn_intercept:.4f}", f"{sm_const_coef:.4f}", f"{sm_const_pval:.4f}"])

            # Coefficients for predictors
            for i, pred_name in enumerate(X_train.columns):
                sklearn_coef = model.coef_[0][i] if hasattr(model, 'coef_') and model.coef_.size > i else np.nan
                sm_pred_coef = sm_coefs[pred_name] if sm_coefs is not None and pred_name in sm_coefs else np.nan
                sm_pred_pval = sm_pvalues[pred_name] if sm_pvalues is not None and pred_name in sm_pvalues else np.nan
                coef_table_data.append([pred_name, f"{sklearn_coef:.4f}", f"{sm_pred_coef:.4f}", f"{sm_pred_pval:.4f}"])

            # Save coefficients table to PDF
            fig_table, ax_table = plt.subplots(figsize=(10, max(4, 0.3 * len(coef_table_data) + 1))) # Adjust height
            ax_table.axis('off')
            table_obj = ax_table.table(cellText=coef_table_data, loc='center', cellLoc='left', colWidths=[0.4, 0.2, 0.2, 0.2])
            table_obj.auto_set_font_size(False)
            table_obj.set_fontsize(8)
            table_obj.scale(1, 1.2) # Adjust scale
            ax_table.set_title(f"Animal {animal_id} | T_max_lag={T_max_lag}\nLogistic Regression Coefficients", fontsize=12, pad=15)
            pdf_animal_details.savefig(fig_table, bbox_inches='tight')
            plt.close(fig_table)

            # Save statsmodels summary to PDF
            fig_summary, ax_summary = plt.subplots(figsize=(8.5, 11)) # Standard page size
            ax_summary.axis('off')
            fig_summary.text(0.05, 0.95, f"Animal {animal_id} | T_max_lag={T_max_lag} | Statsmodels Summary", fontsize=12, va='bottom', ha='left')
            fig_summary.text(0.05, 0.05, sm_summary_text, fontsize=8, va='bottom', ha='left', family='monospace', wrap=True)
            pdf_animal_details.savefig(fig_summary)
            plt.close(fig_summary)

            # Calculate ROC-AUC
            train_auc = np.nan
            if not y_train.empty and y_train.nunique() >=2:
                 y_train_prob = model.predict_proba(X_train)[:, 1]
                 train_auc = roc_auc_score(y_train, y_train_prob)

            test_auc = np.nan
            if not X_test.empty and not y_test.empty and y_test.nunique() >= 2:
                y_test_prob = model.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_test_prob)
            
            results_auc.append({'animal': animal_id, 'train_auc': train_auc, 'test_auc': test_auc})
        
        # Plotting ROC-AUC for current T_max_lag
        df_res_auc = pd.DataFrame(results_auc)
        if not df_res_auc.empty:
            animal_labels = df_res_auc['animal'].astype(str) # Ensure animal IDs are strings for labels
            train_aucs = df_res_auc['train_auc']
            test_aucs = df_res_auc['test_auc']
            
            x_indices = np.arange(len(animal_labels))
            bar_width = 0.35
            
            ax_current_roc = axs_roc_auc[idx]
            ax_current_roc.bar(x_indices - bar_width/2, train_aucs, bar_width, label='Train ROC-AUC', color='skyblue')
            ax_current_roc.bar(x_indices + bar_width/2, test_aucs, bar_width, label='Test ROC-AUC', color='salmon')
            
            ax_current_roc.set_xlabel('Animal ID')
            ax_current_roc.set_ylabel('ROC-AUC')
            ax_current_roc.set_title(f'Max Lag T = {T_max_lag}')
            ax_current_roc.set_xticks(x_indices)
            ax_current_roc.set_xticklabels(animal_labels, rotation=45, ha="right")
            ax_current_roc.set_ylim(0, 1.05)
            ax_current_roc.axhline(0.5, color='grey', linestyle='--', linewidth=0.8) # Chance level
            if idx == 0: # Add legend only to the first subplot
                ax_current_roc.legend(loc='lower right')
        else:
            axs_roc_auc[idx].text(0.5, 0.5, "No data to plot", ha='center', va='center')
            axs_roc_auc[idx].set_title(f'Max Lag T = {T_max_lag}')


    # Save the main ROC-AUC plot (fig_roc_auc) to the PDF as the last summary page
    fig_roc_auc.suptitle('Train and Test ROC-AUC per Animal (Max Lags T=1,2,3)', fontsize=16)
    fig_roc_auc.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    pdf_animal_details.savefig(fig_roc_auc) # Save the figure with all subplots
    
    # Close the PDF file
    pdf_animal_details.close()
    print(f"Detailed results saved to animal_logit_results_T123.pdf")

    # Display the ROC-AUC plot
    plt.show()

if __name__ == "__main__":
    run_analysis()
