# %%
import pandas as pd
import numpy as np

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

# --- Helper for running tests (can be defined globally or in a test class) ---
global_test_passed_count = 0
global_test_failed_count = 0

def run_test(condition, test_name):
    global global_test_passed_count, global_test_failed_count
    if condition:
        print(f"PASS: {test_name}")
        global_test_passed_count += 1
    else:
        print(f"FAIL: {test_name}")
        global_test_failed_count += 1
    return condition

# --- Test Data Setup ---
# Mock data designed to test explicit lookup, gaps, and session boundaries.
# Session 3 has trials out of numerical order in the raw data to test robustness.
exp_df = pd.read_csv('/home/rlab/raghavendra/ddm_data/outExp.csv')
# %%
# Remove wrong rows 
count = ((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3)).sum()
print("Number of rows where TotalFixTime is NaN and abort_event == 3:", count)
exp_df = exp_df[~((exp_df['TotalFixTime'].isna()) & (exp_df['abort_event'] == 3))].copy()

# Comparable batch
all_df = exp_df[(exp_df['batch_name'] == 'Comparable')]
original_test_df = all_df.copy()

# Define column names to use in tests
SESSION_COL_NAME = 'session'
TRIAL_COL_NAME = 'trial'
VALUE_COL_NAME = 'TotalFixTime'

# --- Apply the function for k=1 ---
K_VALUE_1 = 1
df_processed_k1 = add_k_previous_trial_value_explicit_lookup(
    original_test_df.copy(), VALUE_COL_NAME, K_VALUE_1, SESSION_COL_NAME, TRIAL_COL_NAME
)
NEW_COL_K1 = f"{VALUE_COL_NAME}_{K_VALUE_1}"
print(f"\n--- Processed DataFrame (k={K_VALUE_1}) ---")
print(df_processed_k1.sort_values([SESSION_COL_NAME, TRIAL_COL_NAME])) # Sort for display

# --- Apply the function for k=2 ---
K_VALUE_2 = 2
df_processed_k2 = add_k_previous_trial_value_explicit_lookup(
    original_test_df.copy(), VALUE_COL_NAME, K_VALUE_2, SESSION_COL_NAME, TRIAL_COL_NAME
)
NEW_COL_K2 = f"{VALUE_COL_NAME}_{K_VALUE_2}"
print(f"\n--- Processed DataFrame (k={K_VALUE_2}) ---")
print(df_processed_k2.sort_values([SESSION_COL_NAME, TRIAL_COL_NAME])) # Sort for display


print("\n--- Running Tests for Explicit Lookup Function ---")
global_test_passed_count = 0 # Reset counters for this test run
global_test_failed_count = 0

# Test 1: New column existence
run_test(NEW_COL_K1 in df_processed_k1.columns, f"Test 1.1 (k=1): New column '{NEW_COL_K1}' exists.")
run_test(NEW_COL_K2 in df_processed_k2.columns, f"Test 1.2 (k=2): New column '{NEW_COL_K2}' exists.")

# Test 2: DataFrame shape and original columns preservation
run_test(len(df_processed_k1) == len(original_test_df), "Test 2.1 (k=1): DataFrame length is preserved.")
run_test(set(original_test_df.columns).issubset(set(df_processed_k1.columns)), "Test 2.2 (k=1): Original columns are preserved.")
run_test(len(df_processed_k1.columns) == len(original_test_df.columns) + 1, "Test 2.3 (k=1): Exactly one new column added.")

# Test 3: Verification by random sampling (k=1)
print("\n--- Test 3: Random Sample Verification (k=1) ---")
num_samples = min(5, len(df_processed_k1))
if num_samples > 0:
    # Sample based on index to ensure we can reliably get the row from processed_df
    sampled_indices = original_test_df.sample(n=num_samples, random_state=42).index 
    
    for original_idx in sampled_indices:
        processed_row = df_processed_k1.loc[original_idx] # Get corresponding row by index
        current_session = processed_row[SESSION_COL_NAME]
        current_trial = processed_row[TRIAL_COL_NAME]
        value_in_new_col = processed_row[NEW_COL_K1]
        
        target_prev_trial_id = current_trial - K_VALUE_1
        
        # Look for the target_prev_trial_id in the *original* DataFrame
        source_rows = original_test_df[
            (original_test_df[SESSION_COL_NAME] == current_session) & 
            (original_test_df[TRIAL_COL_NAME] == target_prev_trial_id)
        ]
        
        expected_value_from_source = np.nan
        if not source_rows.empty:
            # If multiple source rows, function logic takes the first. Test should reflect this.
            expected_value_from_source = source_rows[VALUE_COL_NAME].iloc[0] 
            
        test_desc = (f"Random Sample (k=1): S{current_session}, T{current_trial} -> '{NEW_COL_K1}' "
                     f"is ({value_in_new_col:.3f})" if pd.notna(value_in_new_col) else \
                     f"is (NaN)")
        test_desc += (f", Expected from T{target_prev_trial_id} "
                      f"is ({expected_value_from_source:.3f})" if pd.notna(expected_value_from_source) else \
                      f"is (NaN)")

        # Comparison (handle NaNs and float precision)
        if pd.isna(value_in_new_col) and pd.isna(expected_value_from_source):
            run_test(True, test_desc)
        elif pd.notna(value_in_new_col) and pd.notna(expected_value_from_source):
            run_test(np.isclose(value_in_new_col, expected_value_from_source), test_desc)
        else: # One is NaN, the other is not
            run_test(False, test_desc)
else:
    print("Skipping Test 3: DataFrame is empty or too small for sampling.")

# Test 4: Trials that are "first" (lowest trial ID in session) should have NaN
print("\n--- Test 4: Lowest Trial ID in Session (k=1) ---")
# Find rows with the minimum trial_id for each session in the original df
min_trial_indices = original_test_df.loc[original_test_df.groupby(SESSION_COL_NAME)[TRIAL_COL_NAME].idxmin()].index

all_min_trials_nan_k1 = True
for original_idx in min_trial_indices:
    processed_row = df_processed_k1.loc[original_idx]
    session_id = processed_row[SESSION_COL_NAME]
    trial_id = processed_row[TRIAL_COL_NAME] # This is the lowest trial_id for this session
    val_in_new_col = processed_row[NEW_COL_K1]

    # For this lowest trial T_min, T_min - K_VALUE_1 should not exist as a trial in that session.
    # Thus, the looked-up value should be NaN.
    if not pd.isna(val_in_new_col):
        all_min_trials_nan_k1 = False
        print(f"FAIL Detail (Test 4, k=1): For S{session_id}, Lowest Trial {trial_id}, "
              f"'{NEW_COL_K1}' is {val_in_new_col}, expected NaN.")
        break 
run_test(all_min_trials_nan_k1, f"Test 4 (k=1): For lowest trial ID in each session, '{NEW_COL_K1}' is NaN.")

# Test 5: Specific known cases based on mock data
print("\n--- Test 5: Specific Known Case Verification ---")
# Case 5.1 (k=1): Session 1, Trial 11. Target prev trial is 10. Expected value: 1.0
row_s1t11_k1 = df_processed_k1[(df_processed_k1[SESSION_COL_NAME] == 1) & (df_processed_k1[TRIAL_COL_NAME] == 11)]
if not row_s1t11_k1.empty:
    actual = row_s1t11_k1[NEW_COL_K1].iloc[0]
    expected = 1.0
    run_test(np.isclose(actual, expected), f"Test 5.1 (k=1): S1, T11 -> {actual:.1f}, Expected {expected:.1f}")
else: print("Skipping Test 5.1: S1, T11 not found.")

# Case 5.2 (k=1): Session 1, Trial 13. Target prev trial is 12 (does not exist). Expected: NaN
row_s1t13_k1 = df_processed_k1[(df_processed_k1[SESSION_COL_NAME] == 1) & (df_processed_k1[TRIAL_COL_NAME] == 13)]
if not row_s1t13_k1.empty:
    actual = row_s1t13_k1[NEW_COL_K1].iloc[0]
    run_test(pd.isna(actual), f"Test 5.2 (k=1): S1, T13 -> NaN (target T12 non-existent)")
else: print("Skipping Test 5.2: S1, T13 not found.")

# Case 5.3 (k=1): Session 3, Trial 100. Target prev trial is 99. Expected: 3.4 (handles original data order)
row_s3t100_k1 = df_processed_k1[(df_processed_k1[SESSION_COL_NAME] == 3) & (df_processed_k1[TRIAL_COL_NAME] == 100)]
if not row_s3t100_k1.empty:
    actual = row_s3t100_k1[NEW_COL_K1].iloc[0]
    expected = 3.4 # Value from S3, T99
    run_test(np.isclose(actual, expected), f"Test 5.3 (k=1): S3, T100 -> {actual:.1f}, Expected {expected:.1f}")
else: print("Skipping Test 5.3: S3, T100 not found.")

# Case 5.4 (k=2): Session 1, Trial 13. Target prev trial is 11. Expected value: 1.1
row_s1t13_k2 = df_processed_k2[(df_processed_k2[SESSION_COL_NAME] == 1) & (df_processed_k2[TRIAL_COL_NAME] == 13)]
if not row_s1t13_k2.empty:
    actual = row_s1t13_k2[NEW_COL_K2].iloc[0]
    expected = 1.1 # Value from S1, T11
    run_test(np.isclose(actual, expected), f"Test 5.4 (k=2): S1, T13 -> {actual:.1f}, Expected {expected:.1f}")
else: print("Skipping Test 5.4: S1, T13 not found.")

# Case 5.5 (k=2): Session 2, Trial 4. Target prev trial is 2. Expected: 2.2 (gap handling)
row_s2t4_k2 = df_processed_k2[(df_processed_k2[SESSION_COL_NAME] == 2) & (df_processed_k2[TRIAL_COL_NAME] == 4)]
if not row_s2t4_k2.empty:
    actual = row_s2t4_k2[NEW_COL_K2].iloc[0]
    expected = 2.2 # Value from S2, T2
    run_test(np.isclose(actual, expected), f"Test 5.5 (k=2): S2, T4 -> {actual:.1f}, Expected {expected:.1f}")
else: print("Skipping Test 5.5: S2, T4 not found.")

# Case 5.6 (k=2): Session 2, Trial 2. Target prev trial is 0 (does not exist). Expected: NaN
row_s2t2_k2 = df_processed_k2[(df_processed_k2[SESSION_COL_NAME] == 2) & (df_processed_k2[TRIAL_COL_NAME] == 2)]
if not row_s2t2_k2.empty:
    actual = row_s2t2_k2[NEW_COL_K2].iloc[0]
    run_test(pd.isna(actual), f"Test 5.6 (k=2): S2, T2 -> NaN (target T0 non-existent)")
else: print("Skipping Test 5.6: S2, T2 not found.")

# Case 5.7 (k=1): Session 4, Trial 3 (lowest trial ID in this session, even if data unordered). Target T2 (non-existent). Expected: NaN
row_s4t3_k1 = df_processed_k1[(df_processed_k1[SESSION_COL_NAME] == 4) & (df_processed_k1[TRIAL_COL_NAME] == 3)]
if not row_s4t3_k1.empty:
    actual = row_s4t3_k1[NEW_COL_K1].iloc[0]
    run_test(pd.isna(actual), f"Test 5.7 (k=1): S4, T3 (lowest) -> NaN (target T2 non-existent)")
else: print("Skipping Test 5.7: S4, T3 not found.")


# --- Final Summary ---
print("\n--- Test Summary (Explicit Lookup Function) ---")
print(f"Total Tests Passed: {global_test_passed_count}")
print(f"Total Tests Failed: {global_test_failed_count}")

if global_test_failed_count == 0:
    print("\nAll explicit lookup tests passed successfully!")
else:
    print("\nSome explicit lookup tests failed.")