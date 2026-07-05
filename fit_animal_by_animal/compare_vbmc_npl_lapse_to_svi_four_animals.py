# %%
"""
Compare refreshed old VBMC NPL+lapse fits against early-best and random-100k SVI.

The old VBMC fit has no alpha and uses one scalar t_E_aff. SVI has alpha and
condition-wise t_E_aff. This script compares shared parameters directly and
reports t_E_aff as VBMC scalar versus SVI average across conditions.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
VBMC_ROOT = Path(
    os.environ.get(
        "VBMC_NPL_LAPSE_ROOT",
        str(SCRIPT_DIR / "vbmc_npl_lapse_four_animal_rerun"),
    )
).expanduser()
EARLY_SVI_ROOT = SCRIPT_DIR / "numpyro_svi_npl_alpha_lapse_condition_delay_low_lr_patience12_min12k_restore_best_reruns"
RANDOM_100K_SVI_ROOT = SCRIPT_DIR / "numpyro_svi_npl_alpha_lapse_condition_delay_random_plausible_low_lr_100k_earlybest_reruns"

ANIMALS = [
    ("LED7", 98),
    ("LED7", 100),
    ("LED34_even", 52),
    ("LED34_even", 60),
]

PARAMS = [
    "rate_lambda",
    "T_0",
    "theta_E",
    "w",
    "del_go",
    "rate_norm_l",
    "lapse_prob",
    "lapse_prob_right",
    "t_E_aff",
]

DISPLAY_SCALE = {
    "T_0": 1000.0,
    "del_go": 1000.0,
    "t_E_aff": 1000.0,
    "lapse_prob": 100.0,
}

DISPLAY_NAME = {
    "T_0": "T_0_ms",
    "del_go": "del_go_ms",
    "t_E_aff": "t_E_aff_ms",
    "lapse_prob": "lapse_pct",
}

OUT_DIR = VBMC_ROOT / "comparison_to_svi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "vbmc_npl_lapse_vs_svi_parameter_comparison.csv"
OUT_SUMMARY = OUT_DIR / "vbmc_npl_lapse_vs_svi_closeness_summary.csv"


# %%
def read_summary(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).set_index("parameter")


def svi_param_mean(summary_df, param):
    if param == "t_E_aff":
        rows = summary_df.loc[[idx for idx in summary_df.index if str(idx).startswith("t_E_aff_")]]
        return float(rows["mean"].mean())
    return float(summary_df.loc[param, "mean"])


# %%
# =============================================================================
# Compare
# =============================================================================
rows = []
run_rows = []
for batch_name, animal in ANIMALS:
    key = f"{batch_name}_{animal}"
    vbmc_dir = VBMC_ROOT / key
    vbmc_summary_path = vbmc_dir / "vbmc_norm_lapse_posterior_summary.csv"
    if not vbmc_summary_path.exists():
        print(f"Skipping {batch_name}/{animal}: missing {vbmc_summary_path}")
        continue
    vbmc_summary = read_summary(vbmc_summary_path)
    early_summary = read_summary(EARLY_SVI_ROOT / key / "main_fullrank_posterior_summary.csv")
    random_summary = read_summary(RANDOM_100K_SVI_ROOT / key / "main_fullrank_posterior_summary.csv")

    run_summary_path = vbmc_dir / "vbmc_norm_lapse_run_summary.csv"
    if run_summary_path.exists():
        run_summary = pd.read_csv(run_summary_path).iloc[0].to_dict()
        run_rows.append(run_summary)

    for param in PARAMS:
        vbmc_mean = float(vbmc_summary.loc[param, "mean"])
        vbmc_q025 = float(vbmc_summary.loc[param, "q025"])
        vbmc_q975 = float(vbmc_summary.loc[param, "q975"])
        early_mean = svi_param_mean(early_summary, param)
        random_mean = svi_param_mean(random_summary, param)
        delta_early = abs(vbmc_mean - early_mean)
        delta_random = abs(vbmc_mean - random_mean)
        if delta_early < delta_random:
            closer = "early_svi"
        elif delta_random < delta_early:
            closer = "random100k_svi"
        else:
            closer = "tie"

        scale = DISPLAY_SCALE.get(param, 1.0)
        display_param = DISPLAY_NAME.get(param, param)
        rows.append(
            {
                "batch_name": batch_name,
                "animal": animal,
                "parameter": param,
                "display_parameter": display_param,
                "vbmc_mean": vbmc_mean,
                "vbmc_q025": vbmc_q025,
                "vbmc_q975": vbmc_q975,
                "early_svi_mean": early_mean,
                "random100k_svi_mean": random_mean,
                "abs_delta_early": delta_early,
                "abs_delta_random100k": delta_random,
                "closer_to": closer,
                "vbmc_display_mean": vbmc_mean * scale,
                "early_display_mean": early_mean * scale,
                "random100k_display_mean": random_mean * scale,
                "abs_delta_early_display": delta_early * scale,
                "abs_delta_random100k_display": delta_random * scale,
            }
        )

comparison_df = pd.DataFrame(rows)
if comparison_df.empty:
    raise RuntimeError(f"No VBMC summaries found under {VBMC_ROOT}")
comparison_df.to_csv(OUT_CSV, index=False)

closeness_df = (
    comparison_df.groupby(["parameter", "display_parameter", "closer_to"])
    .size()
    .reset_index(name="n_animals")
    .sort_values(["parameter", "closer_to"])
)
closeness_df.to_csv(OUT_SUMMARY, index=False)

print("VBMC run summaries:")
if run_rows:
    run_df = pd.DataFrame(run_rows)
    print(
        run_df[
            [
                "batch_name",
                "animal",
                "n_valid_fit_trials",
                "final_stable",
                "final_elbo",
                "final_elbo_sd",
                "final_iter",
            ]
        ].to_string(index=False)
    )
else:
    print("No run summaries found.")

print("\nParameter comparison means in display units:")
display_cols = [
    "batch_name",
    "animal",
    "display_parameter",
    "vbmc_display_mean",
    "early_display_mean",
    "random100k_display_mean",
    "abs_delta_early_display",
    "abs_delta_random100k_display",
    "closer_to",
]
print(comparison_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6g}"))

print("\nCloseness summary:")
print(closeness_df.to_string(index=False))
print(f"\nSaved comparison CSV: {OUT_CSV}")
print(f"Saved closeness summary: {OUT_SUMMARY}")

# %%
