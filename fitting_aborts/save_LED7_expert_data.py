# %%
from pathlib import Path

import pandas as pd


# %%
############ Parameters (edit here) ############
batch_name = "LED7"
session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]
allowed_abort_events = [3, 4]
max_rtwrtstim_for_fit = 1.0

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
input_csv_path = REPO_ROOT / "out_LED.csv"
output_csv_path = REPO_ROOT / f"{batch_name}_expert_data.csv"


# %%
############ Build LED7 expert dataset once and save it ############
if not input_csv_path.exists():
    raise FileNotFoundError(f"Could not find input CSV: {input_csv_path}")

exp_df = pd.read_csv(input_csv_path)
exp_df["batch_name"] = batch_name
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df["t_LED"] = exp_df["intended_fix"] - exp_df["LED_onset_time"]
exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})

exp_df = exp_df[exp_df["RTwrtStim"] < max_rtwrtstim_for_fit].copy()
exp_df = exp_df[~((exp_df["RTwrtStim"].isna()) & (exp_df["abort_event"] == 3))].copy()

mask_nan = exp_df["response_poke"].isna()
mask_success_1 = exp_df["success"] == 1
mask_success_neg1 = exp_df["success"] == -1
mask_ild_pos = exp_df["ILD"] > 0
mask_ild_neg = exp_df["ILD"] < 0
exp_df.loc[mask_nan & mask_success_1 & mask_ild_pos, "response_poke"] = 3
exp_df.loc[mask_nan & mask_success_1 & mask_ild_neg, "response_poke"] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_pos, "response_poke"] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_neg, "response_poke"] = 3

mask_led_off = (exp_df["LED_trial"] == 0) | (exp_df["LED_trial"].isna())
mask_repeat = exp_df["repeat_trial"].isin(allowed_repeat_trials) | exp_df["repeat_trial"].isna()
led7_expert_df = exp_df[
    mask_led_off
    & mask_repeat
    & exp_df["session_type"].isin([session_type])
    & exp_df["training_level"].isin([training_level])
].copy()

led7_expert_df = led7_expert_df[
    (led7_expert_df["success"].isin([1, -1]))
    | (led7_expert_df["abort_event"].isin(allowed_abort_events))
].copy()
led7_expert_df = led7_expert_df[
    led7_expert_df["RTwrtStim"] < max_rtwrtstim_for_fit
].copy()
led7_expert_df["abs_ILD"] = led7_expert_df["ILD"].abs()

led7_expert_df.to_csv(output_csv_path, index=False)

print(f"Saved {output_csv_path}")
print(f"Rows: {len(led7_expert_df)}")
print(f"Columns: {len(led7_expert_df.columns)}")
print(f"Animals: {sorted(led7_expert_df['animal'].dropna().astype(int).unique().tolist())}")
print(f"ABLs: {sorted(led7_expert_df['ABL'].dropna().astype(float).unique().tolist())}")
print(f"abs_ILDs: {sorted(led7_expert_df['abs_ILD'].dropna().astype(float).unique().tolist())}")
