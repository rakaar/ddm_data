# Publication Plot Payload Handoff

This note is for any agent/script that wants to reuse the two publication-grade plots produced by:

`fitting_aborts/fit_only_LED_off_with_LED_ON_fit_diagnostics_truncate_not_censor_ABL_delay.py`

Do not rerun the Monte Carlo just to make these plots again. Load the saved `.pkl` payloads and plot from those.

## Output directory

Both payloads are saved under:

`fitting_aborts/norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay/diagnostics/`

For the current script, the two relevant payload files are:

1. ABL theory + data overlay payload

`diag_norm_tied_batch_LED7_aggregate_ledoff_truncated_<TRUNC_TAG>_rtwrtstim_by_ABL_publication_truncate_not_censor_ABL_delay.pkl`

Example for `TRUNC_TAG=100ms`:

`diag_norm_tied_batch_LED7_aggregate_ledoff_truncated_100ms_rtwrtstim_by_ABL_publication_truncate_not_censor_ABL_delay.pkl`

2. Delay bar plot payload

`diag_norm_tied_batch_LED7_aggregate_ledoff_delay_bars_truncate_not_censor_ABL_delay.pkl`

## Payload 1: ABL theory + data overlay

### What it contains

A compact plotting payload for the single-panel publication figure with:

- x-axis in both seconds and ms
- data histogram densities for each ABL
- theory densities for each ABL
- labels/config needed for plotting

### Top-level keys

- `config`
- `t_pts_truncated`
- `t_pts_truncated_ms`
- `data_hist_centers_truncated`
- `data_hist_centers_truncated_ms`
- `abl_curves`

### `config`

- `batch_name`: string
- `truncate_rt_wrt_stim_s`: float
- `truncate_rt_wrt_stim_ms`: float
- `supported_ABL_values`: list of ints, usually `[20, 40, 60]`
- `xlabel`: string
- `ylabel`: string

### `abl_curves`

This is a dict keyed by integer ABL values.

For each ABL key (`20`, `40`, `60`), the value is a dict with:

- `data_density_truncated`: numpy array
- `theory_density_truncated_norm`: numpy array

### Units

- `t_pts_truncated`: seconds
- `t_pts_truncated_ms`: milliseconds
- `data_hist_centers_truncated`: seconds
- `data_hist_centers_truncated_ms`: milliseconds
- densities are the y-values used directly for plotting

### Minimal loader example

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

payload_path = (
    "fitting_aborts/"
    "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay/"
    "diagnostics/"
    "diag_norm_tied_batch_LED7_aggregate_ledoff_truncated_100ms_rtwrtstim_"
    "by_ABL_publication_truncate_not_censor_ABL_delay.pkl"
)

with open(payload_path, "rb") as f:
    payload = pickle.load(f)

t_ms = np.asarray(payload["t_pts_truncated_ms"], dtype=float)
bin_centers_ms = np.asarray(payload["data_hist_centers_truncated_ms"], dtype=float)
supported_abls = payload["config"]["supported_ABL_values"]

abl_colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

fig, ax = plt.subplots(figsize=(6.2, 4.0))
for abl in supported_abls:
    curves = payload["abl_curves"][int(abl)]
    data_y = np.asarray(curves["data_density_truncated"], dtype=float)
    theory_y = np.asarray(curves["theory_density_truncated_norm"], dtype=float)

    ax.step(bin_centers_ms, data_y, where="mid", color=abl_colors[int(abl)], lw=1.2)
    ax.plot(t_ms, theory_y, color=abl_colors[int(abl)], lw=2.5)

ax.set_xlim(0, payload["config"]["truncate_rt_wrt_stim_ms"])
ax.set_xlabel(payload["config"]["xlabel"])
ax.set_ylabel(payload["config"]["ylabel"])
plt.show()
```

## Payload 2: delay bar plot

### What it contains

A compact plotting payload for the delay bar figure.

### Top-level keys

- `labels`
- `values_ms`
- `batch_name`
- `pdf_path`
- `png_path`

### Meaning

- `labels`: mathtext-ready strings for the five delay categories
- `values_ms`: numpy array of length 5, in milliseconds

The five entries are ordered as:

1. `delta_a - delta_LED`
2. `delta_LED + delta_m`
3. `delta_e^20`
4. `delta_e^40`
5. `delta_e^60`

### Minimal loader example

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

payload_path = (
    "fitting_aborts/"
    "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay/"
    "diagnostics/"
    "diag_norm_tied_batch_LED7_aggregate_ledoff_delay_bars_"
    "truncate_not_censor_ABL_delay.pkl"
)

with open(payload_path, "rb") as f:
    payload = pickle.load(f)

labels = payload["labels"]
values_ms = np.asarray(payload["values_ms"], dtype=float)
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.bar(x, values_ms, width=0.68, color="#808080", edgecolor="black", linewidth=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_position(("data", 0.0))
ax.set_ylabel("Delay (ms)")
ax.set_xticks(x)
ax.set_xticklabels(["", *labels[1:]])
ax.set_yticks([-60, 0, 60])

first_label_y = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
ax.text(x[0], first_label_y, labels[0], ha="center", va="bottom")

plt.show()
```

## Recommendation to the next agent

If you only need these two figures inside another multi-panel figure:

- load the two `.pkl` files
- do not import or run the original fitting script
- do not recompute any MC samples
- treat the payloads as final plotting data
