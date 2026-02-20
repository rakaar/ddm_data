# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)

# %%
# =============================================================================
# Parameters (edit here)
# =============================================================================
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "out_LED.csv")
SESSION_TYPE = 7
TRAINING_LEVEL = 16
T_TRUNC = 0.3

# None -> all available animals after filtering
ANIMALS = None

BIN_WIDTH = 0.005
BINS_WRT_LED = np.arange(-3.0, 3.0 + BIN_WIDTH, BIN_WIDTH)
X_LIM = (-0.1, 0.2)
FIGSIZE = (13, 4.8)
SAVE_EXT = "pdf"
SHOW_PLOTS = True

# True -> histogram area equals abort fraction (density * abort fraction)
# False -> plain density of abort-time distribution
AREA_WEIGHTED = True


def _scaled_hist(values: np.ndarray, n_total: int, bins: np.ndarray, area_weighted: bool) -> tuple[np.ndarray, float]:
    if n_total <= 0 or len(values) == 0:
        return np.zeros(len(bins) - 1), 0.0
    hist, _ = np.histogram(values, bins=bins, density=True)
    frac = len(values) / n_total
    if area_weighted:
        return hist * frac, frac
    return hist, frac


def _load_and_filter_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["repeat_trial"].isin([0, 2]) | df["repeat_trial"].isna()]
    df = df[df["session_type"].isin([SESSION_TYPE])]
    df = df[df["training_level"].isin([TRAINING_LEVEL])]
    df = df.dropna(subset=["intended_fix", "LED_onset_time", "timed_fix"])
    df = df[(df["abort_event"] == 3) | (df["success"].isin([1, -1]))]
    return df


def _build_fit_df(df_animal: pd.DataFrame) -> pd.DataFrame:
    df_on = df_animal[df_animal["LED_trial"] == 1]
    df_off = df_animal[(df_animal["LED_trial"] == 0) | (df_animal["LED_trial"].isna())]

    df_on_fit = pd.DataFrame(
        {
            "RT": df_on["timed_fix"].values,
            "t_stim": df_on["intended_fix"].values,
            "t_LED": (df_on["intended_fix"] - df_on["LED_onset_time"]).values,
            "LED_trial": 1,
        }
    )
    df_off_fit = pd.DataFrame(
        {
            "RT": df_off["timed_fix"].values,
            "t_stim": df_off["intended_fix"].values,
            "t_LED": (df_off["intended_fix"] - df_off["LED_onset_time"]).values,
            "LED_trial": 0,
        }
    )
    return pd.concat([df_on_fit, df_off_fit], ignore_index=True)


def _apply_abort_truncation(fit_df: pd.DataFrame, t_trunc: float) -> pd.DataFrame:
    # Match "filter out aborts < T_trunc" behavior.
    remove_mask = (fit_df["RT"] < fit_df["t_stim"]) & (fit_df["RT"] < t_trunc)
    return fit_df.loc[~remove_mask].copy()


def _compute_panel_curves(fit_df: pd.DataFrame, bins: np.ndarray, area_weighted: bool) -> dict:
    on_mask = fit_df["LED_trial"] == 1
    off_mask = fit_df["LED_trial"] == 0
    on_abort_mask = on_mask & (fit_df["RT"] < fit_df["t_stim"])
    off_abort_mask = off_mask & (fit_df["RT"] < fit_df["t_stim"])

    on_vals = (fit_df.loc[on_abort_mask, "RT"] - fit_df.loc[on_abort_mask, "t_LED"]).to_numpy()
    off_vals = (fit_df.loc[off_abort_mask, "RT"] - fit_df.loc[off_abort_mask, "t_LED"]).to_numpy()

    n_total_on = int(on_mask.sum())
    n_total_off = int(off_mask.sum())
    hist_on, frac_on = _scaled_hist(on_vals, n_total_on, bins, area_weighted)
    hist_off, frac_off = _scaled_hist(off_vals, n_total_off, bins, area_weighted)

    return {
        "hist_on": hist_on,
        "hist_off": hist_off,
        "n_total_on": n_total_on,
        "n_total_off": n_total_off,
        "n_aborts_on": int(on_abort_mask.sum()),
        "n_aborts_off": int(off_abort_mask.sum()),
        "frac_on": frac_on,
        "frac_off": frac_off,
    }


def _plot_grid(df: pd.DataFrame, animals_to_plot: np.ndarray) -> str:
    n_animals = len(animals_to_plot)
    bin_centers = (BINS_WRT_LED[1:] + BINS_WRT_LED[:-1]) / 2
    fig_height = max(FIGSIZE[1], 2.7 * n_animals)
    fig, axes = plt.subplots(n_animals, 2, figsize=(FIGSIZE[0], fig_height), sharex=True, sharey=True)
    if n_animals == 1:
        axes = np.array([axes])

    for i, animal in enumerate(animals_to_plot):
        df_animal = df[df["animal"] == animal]
        fit_df_no_trunc = _build_fit_df(df_animal)
        fit_df_trunc = _apply_abort_truncation(fit_df_no_trunc, T_TRUNC)
        panel_defs = [
            ("No truncation", fit_df_no_trunc),
            (f"With truncation (remove aborts RT < {T_TRUNC:.3f}s)", fit_df_trunc),
        ]

        for j, (panel_title, fit_df_curr) in enumerate(panel_defs):
            ax = axes[i, j]
            stats = _compute_panel_curves(fit_df_curr, BINS_WRT_LED, AREA_WEIGHTED)
            area_on = np.trapz(stats["hist_on"], bin_centers)
            area_off = np.trapz(stats["hist_off"], bin_centers)

            on_label = "LED ON" if (i == 0 and j == 0) else None
            off_label = "LED OFF" if (i == 0 and j == 0) else None
            led_label = "LED onset" if (i == 0 and j == 0) else None

            ax.plot(bin_centers, stats["hist_on"], color="red", lw=1.8, alpha=0.85, label=on_label)
            ax.plot(bin_centers, stats["hist_off"], color="blue", lw=1.8, alpha=0.85, label=off_label)
            ax.axvline(0, color="k", ls="--", alpha=0.5, label=led_label)
            ax.set_xlim(*X_LIM)
            ax.set_title(f"Animal {animal} | {panel_title}\nA_on={area_on:.3f}, A_off={area_off:.3f}", fontsize=10)

            if i == n_animals - 1:
                ax.set_xlabel("RT - t_LED (s)")
            if j == 0:
                y_label = "Abort rate (area = abort fraction)" if AREA_WEIGHTED else "Density"
                ax.set_ylabel(y_label)

    axes[0, 0].legend(fontsize=8, loc="upper right")
    y_label = "Abort rate (area = abort fraction)" if AREA_WEIGHTED else "Density"
    fig.suptitle(f"RT wrt LED by animal: no truncation vs truncation (n={n_animals})\nY={y_label}", y=1.003)
    fig.tight_layout()

    out_name = f"data_rt_wrt_led_trunc_vs_no_trunc_grid_{n_animals}x2.{SAVE_EXT}"
    out_path = os.path.join(SCRIPT_DIR, out_name)
    fig.savefig(out_path, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return out_path


# %%
# =============================================================================
# Run per animal
# =============================================================================
df = _load_and_filter_data(DATA_PATH)
animals_available = np.sort(df["animal"].dropna().unique())
if len(animals_available) == 0:
    raise RuntimeError("No animals found after filtering.")

if ANIMALS is None:
    animals_to_plot = animals_available
else:
    animals_to_plot = np.array(ANIMALS)
    missing = [a for a in animals_to_plot if a not in set(animals_available)]
    if missing:
        raise ValueError(f"Animals not found after filtering: {missing}. Available: {animals_available}")

print(f"Animals to plot ({len(animals_to_plot)}): {animals_to_plot}")
print(f"Area-weighted mode: {AREA_WEIGHTED}")
path = _plot_grid(df, animals_to_plot)
print(f"Saved: {path}")
