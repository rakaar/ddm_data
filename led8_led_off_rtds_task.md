# LED8 Session 8 LED-Off RTD Plot Task

Create a new Python script in this repo to plot RTDs from the exported LED8 CSV.

## Input data

Use:

`/home/rlab/raghavendra/ddm_data/LED8_session8_training16_repeat_filtered.csv`

This is already the filtered LED8 expert dataset for `session_type == 8`.

## Goal

Make RTD plots for LED-off trials only.

## Required filtering

Start from the CSV and then apply these filters in this order:

1. Keep only valid-choice trials where `success` is in `[1, -1]`.
2. Keep only rows where `RTwrtStim` is between `0` and `1` inclusive.
3. Keep only LED-off trials, meaning `LED_trial` is either `0` or `NaN`.

## Plot requirements

Create a `3 x 5` grid of subplots:

- rows = `ABL`
- columns = `abs_ILD`

For each `(ABL, abs_ILD)` combination:

- take the filtered trials for that panel
- plot the RT distribution using `RTwrtStim`
- a simple histogram is fine
- use the same x-axis range across panels: `[0, 1]`

## Titles

For each subplot title, include only:

- `ABL`
- `abs ILD`

Example format:

`ABL=40, abs ILD=4`

Do not include trial counts or any other text in subplot titles.

Add a figure suptitle:

`Session type 8, LED off trials, RTDs`

## Style / repo conventions

Follow the style used in this repo:

- use `# %%` cell blocks
- keep editable parameters near the top in a dedicated `# %%` section
- avoid `main()` / `if __name__ == "__main__":`
- keep the script readable and inline rather than over-abstracting

## Output

The script should:

- load the CSV
- apply the filters
- make the `3 x 5` plot
- show the figure
- optionally save a PNG next to the script

## Suggested filename

A good filename would be:

`led8_session8_led_off_rtds.py`

## Run command

Use the repo virtualenv interpreter:

`.venv/bin/python led8_session8_led_off_rtds.py`

## Sanity checks

Before plotting, it is reasonable to confirm:

- there are exactly 3 unique `ABL` values after filtering
- there are exactly 5 unique `abs_ILD` values after filtering

If not, raise an error with the values found.
