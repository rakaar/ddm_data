# %%
from pathlib import Path

import pandas as pd


# %%
CSV_PATH = Path("outMatrix_LED8_converted.csv")
SESSION_TYPE = 1
TRAINING_LEVEL = 16
REPEAT_TRIAL_VALUES = [0, 2]


# %%
led8 = pd.read_csv(CSV_PATH)

needed_columns = ["session_type", "training_level", "repeat_trial", "LED_trial"]
missing_columns = [col for col in needed_columns if col not in led8.columns]
if missing_columns:
    raise SystemExit(
        f"{CSV_PATH} does not have these needed columns: {missing_columns}\n"
        f"Available columns are: {list(led8.columns)}\n"
        "This CSV is the long-format event export from outMatrix_LED8.mat, "
        "not a trial-level dataframe with session/training/repeat/LED_trial columns."
    )


# %%
mask = (
    (led8["session_type"] == SESSION_TYPE)
    & (led8["training_level"] == TRAINING_LEVEL)
    & (led8["repeat_trial"].isin(REPEAT_TRIAL_VALUES) | led8["repeat_trial"].isna())
)

df_8_expert = led8.loc[mask].copy()

print(f"CSV: {CSV_PATH}")
print(
    "Rows with "
    f"session_type == {SESSION_TYPE}, "
    f"training_level == {TRAINING_LEVEL}, "
    f"repeat_trial in {REPEAT_TRIAL_VALUES} or NaN: {len(df_8_expert):,}"
)

print("\nrepeat_trial counts in filtered rows:")
print(df_8_expert["repeat_trial"].value_counts(dropna=False).sort_index().to_string())

print("\nRows by animal:")
print(df_8_expert["animal"].value_counts().sort_index().to_string())


# %%
led_off = df_8_expert["LED_trial"].eq(0) | df_8_expert["LED_trial"].isna()
led_on = df_8_expert["LED_trial"].eq(1)

n_led_off = led_off.sum()
n_led_on = led_on.sum()

print("\nLED_trial counts in df_8_expert:")
print(f"LED off, LED_trial == 0 or NaN: {n_led_off:,}")
print(f"LED on, LED_trial == 1: {n_led_on:,}")
if n_led_on:
    print(f"LED off / LED on ratio: {n_led_off / n_led_on:.4f}")
else:
    print("LED off / LED on ratio: undefined, LED on count is 0")

if n_led_off:
    print(f"LED on / LED off ratio: {n_led_on / n_led_off:.4f}")
else:
    print("LED on / LED off ratio: undefined, LED off count is 0")
