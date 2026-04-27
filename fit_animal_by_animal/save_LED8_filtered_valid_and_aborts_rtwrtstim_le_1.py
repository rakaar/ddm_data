# %%
from pathlib import Path

import pandas as pd


# %%
LED8_BATCH_FILE = Path(__file__).resolve().parent.parent / "outLED8.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "batch_csvs"
OUTPUT_CSV = OUTPUT_DIR / "LED8_session1_training16_repeat0_2_nan_LED0_1_nan_valid_aborts_RTwrtStim_le_1.csv"

SESSION_TYPES = [1]
TRAINING_LEVELS = [16]
REPEAT_TRIAL_VALUES = [0, 2]
LED_TRIAL_VALUES = [0, 1]
ABORT_EVENTS = [3, 4]
SUCCESS_VALUES = [1, -1]
MAX_RTWRTSTIM = 1


# %%
OUTPUT_DIR.mkdir(exist_ok=True)

exp_df = pd.read_csv(LED8_BATCH_FILE)

# if "RTwrtStim" not in exp_df.columns:
#     exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]

# filtered_df = exp_df[
#     exp_df["training_level"].isin(TRAINING_LEVELS)
#     & (exp_df["repeat_trial"].isin(REPEAT_TRIAL_VALUES) | exp_df["repeat_trial"].isna())
#     & exp_df["session_type"].isin(SESSION_TYPES)
#     & (exp_df["LED_trial"].isin(LED_TRIAL_VALUES) | exp_df["LED_trial"].isna())
#     & (exp_df["abort_event"].isin(ABORT_EVENTS) | exp_df["success"].isin(SUCCESS_VALUES))
#     & (exp_df["RTwrtStim"] <= MAX_RTWRTSTIM)
# ].copy()

# filtered_df.to_csv(OUTPUT_CSV, index=False)

# print(f"Read: {LED8_BATCH_FILE}")
# print(f"Saved: {OUTPUT_CSV}")
# print(f"Rows saved: {len(filtered_df)}")
# print(f"session_type unique: {filtered_df['session_type'].unique().tolist()}")
# print(f"LED_trial unique: {filtered_df['LED_trial'].unique().tolist()}")
# print(f"repeat_trial unique: {sorted(filtered_df['repeat_trial'].dropna().unique().tolist())}")
# print(f"abort_event unique: {sorted(filtered_df['abort_event'].dropna().unique().tolist())}")
# print(f"success unique: {sorted(filtered_df['success'].dropna().unique().tolist())}")
# print(f"RTwrtStim max: {filtered_df['RTwrtStim'].max()}")
# %%
exp_df['LED_trial'].unique()