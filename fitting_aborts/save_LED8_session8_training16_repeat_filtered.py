# %%
from pathlib import Path
import subprocess

import pandas as pd


# %%
############ Parameters (edit here) ############
batch_name = "LED8"
mat_table_name = "totalout"
session_type = 8
training_level = 16
allowed_repeat_trials = [0, 2]
allowed_abort_events = [3, 4]
assumed_led_duration_sec = 0.97777

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
input_mat_path = REPO_ROOT / "outMatrix_LED8.mat"
output_csv_path = REPO_ROOT / "LED8_session8_training16_repeat_filtered.csv"


# %%
############ Export filtered totalout table to CSV using MATLAB ############
if not input_mat_path.exists():
    raise FileNotFoundError(f"Could not find input MAT file: {input_mat_path}")

matlab_input_path = str(input_mat_path).replace("'", "''")
matlab_output_path = str(output_csv_path).replace("'", "''")
allowed_repeat_trials_text = " ".join(str(value) for value in allowed_repeat_trials)

matlab_cmd = (
    f"S = load('{matlab_input_path}','{mat_table_name}'); "
    f"T = S.{mat_table_name}; "
    f"mask = T.session_type == {session_type} & "
    f"T.training_level == {training_level} & "
    f"(isnan(T.repeat_trial) | ismember(T.repeat_trial,[{allowed_repeat_trials_text}])); "
    "T = T(mask,:); "
    f"writetable(T,'{matlab_output_path}'); "
    "fprintf('Exported rows: %d\\n', height(T)); "
    "fprintf('Exported columns: %d\\n', width(T));"
)

subprocess.run(["matlab", "-batch", matlab_cmd], check=True)


# %%
############ Add standard derived columns and save final CSV ############
df = pd.read_csv(output_csv_path)
df = df[
    df["success"].isin([1, -1]) | df["abort_event"].isin(allowed_abort_events)
].copy()
df["batch_name"] = batch_name
df["RTwrtStim"] = df["timed_fix"] - df["intended_fix"]
df["t_LED"] = df["intended_fix"] - df["LED_onset_time"]
df["LED_time_wrt_fix"] = df["LED_onset_time"] - df["intended_fix"]
df["abs_ILD"] = df["ILD"].abs()

remove_led_on_after_light = df["LED_trial"].eq(1) & (
    (df["t_LED"] + assumed_led_duration_sec) < df["timed_fix"]
)

n_remove_led_on_after_light = int(remove_led_on_after_light.sum())

df = df.loc[~remove_led_on_after_light].copy()
df.to_csv(output_csv_path, index=False)


# %%
############ Quick summary from the saved CSV ############
print(f"Saved {output_csv_path}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Animals: {sorted(df['animal'].dropna().astype(int).unique().tolist())}")
print(f"Sessions: {df['session'].dropna().nunique()}")
print(f"Removed LED-on rows after assumed light window: {n_remove_led_on_after_light}")

print("\nsuccess counts:")
print(df["success"].value_counts(dropna=False).sort_index().to_string())

print("\nabort_event counts:")
print(df["abort_event"].value_counts(dropna=False).sort_index().to_string())

print("\nrepeat_trial counts:")
print(df["repeat_trial"].value_counts(dropna=False).sort_index().to_string())

print("\nLED_trial counts:")
print(df["LED_trial"].value_counts(dropna=False).sort_index().to_string())

print("\nLED_onset_time summary:")
print(df["LED_onset_time"].describe().to_string())

print("\nt_LED summary:")
print(df["t_LED"].describe().to_string())
