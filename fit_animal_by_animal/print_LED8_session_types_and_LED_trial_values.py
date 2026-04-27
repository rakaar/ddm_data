# %%
from pathlib import Path

import pandas as pd


# %%
LED8_BATCH_FILE = Path(__file__).resolve().parent.parent / "outLED8.csv"


# %%
df = pd.read_csv(LED8_BATCH_FILE)

print(f"Reading LED8 batch file: {LED8_BATCH_FILE}")
print()

session_types = sorted(df["session_type"].dropna().unique().tolist())

for session_type in session_types:
    session_df = df[df["session_type"] == session_type]
    led_trial_values = session_df["LED_trial"].unique().tolist()

    print(f"session_type {session_type}")
    print(f"LED_trial unique values: {led_trial_values}")
    print()
