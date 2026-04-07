# %%
"""
Compare two model fits (115ms vs 130ms truncation, both fixN=OFF)
against data truncated to 130ms.

Layout: 1×4 (ABL 20, ABL 40, ABL 60, All ABLs combined).
"""

# %%
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DIAG_DIR = (
    SCRIPT_DIR
    / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay_no_choice"
    / "diagnostics"
)

# %%
############ Paths ############
pkl_115 = DIAG_DIR / (
    "diag_norm_tied_batch_LED7_aggregate_ledoff_raw_rtwrtstim_"
    "truncate_not_censor_ABL_delay_no_choice_trunc115ms_allvalid.pkl"
)
pkl_130 = DIAG_DIR / (
    "diag_norm_tied_batch_LED7_aggregate_ledoff_raw_rtwrtstim_"
    "truncate_not_censor_ABL_delay_no_choice_trunc130ms_allvalid.pkl"
)

for p in [pkl_115, pkl_130]:
    if not p.exists():
        raise FileNotFoundError(f"Missing payload: {p}")

# %%
############ Load payloads ############
with open(pkl_115, "rb") as f:
    pay_115 = pickle.load(f)
with open(pkl_130, "rb") as f:
    pay_130 = pickle.load(f)

supported_abls = (20, 40, 60)
abl_colors = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

# Data comes from the 130ms payload (truncated to 130ms)
data_bin_centers = pay_130["data_hist_centers_truncated"]
data_bin_widths = np.diff(pay_130["data_hist_edges_truncated"])

# Theory grids
t_theory_115 = pay_115["t_pts_truncated"]
t_theory_130 = pay_130["t_pts_truncated"]

trunc_s = pay_130["config"]["truncate_rt_wrt_stim_s"]  # 0.130

# Extract per-ABL delays from both fits
params_115 = pay_115["parameter_snapshot"]["normalized_tied_means"]
params_130 = pay_130["parameter_snapshot"]["normalized_tied_means"]
t_E_aff_115 = {20: params_115["t_E_aff_20"], 40: params_115["t_E_aff_40"], 60: params_115["t_E_aff_60"]}
t_E_aff_130 = {20: params_130["t_E_aff_20"], 40: params_130["t_E_aff_40"], 60: params_130["t_E_aff_60"]}

# %%
############ Plot ############
fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharex=True, sharey=True)
combined_max = 0.0

for ax, abl in zip(axes[:3], supported_abls):
    abl_int = int(abl)
    color = abl_colors[abl_int]

    # Data from 130ms payload
    data_d = pay_130["abl_split_truncated"][abl_int]["data_density_truncated"]
    # Theory from 115ms fit
    theory_115 = pay_115["abl_split_truncated"][abl_int]["theory_density_truncated_norm"]
    # Theory from 130ms fit
    theory_130 = pay_130["abl_split_truncated"][abl_int]["theory_density_truncated_norm"]

    n_rows = pay_130["abl_split_truncated"][abl_int]["n_rows"]
    n_trunc = pay_130["abl_split_truncated"][abl_int]["n_truncated_points"]

    d115 = t_E_aff_115[abl_int] * 1e3
    d130 = t_E_aff_130[abl_int] * 1e3

    ax.step(
        data_bin_centers, data_d,
        where="mid", lw=1, color=color, alpha=0.5, label="Data (130ms)",
    )
    ax.plot(
        t_theory_115, theory_115,
        lw=1, color="crimson", linestyle="-",
        label=f"115ms fit (\u03b4={d115:.1f}ms)",
    )
    ax.plot(
        t_theory_130, theory_130,
        lw=1, color="black", linestyle="-",
        label=f"130ms fit (\u03b4={d130:.1f}ms)",
    )
    ax.axvline(x=0.115, color="crimson", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.set_title(f"ABL {abl_int}  (n={n_rows}, trunc={n_trunc})")
    ax.set_xlim(0.06, trunc_s)
    ax.set_xlabel("RT - t_stim (s)")
    if ax is axes[0]:
        ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    combined_max = max(combined_max, float(np.max(data_d)),
                       float(np.max(theory_115)), float(np.max(theory_130)))

# Combined panel
ax_c = axes[3]
for abl in supported_abls:
    abl_int = int(abl)
    color = abl_colors[abl_int]
    data_d = pay_130["abl_split_truncated"][abl_int]["data_density_truncated"]
    theory_115 = pay_115["abl_split_truncated"][abl_int]["theory_density_truncated_norm"]
    theory_130 = pay_130["abl_split_truncated"][abl_int]["theory_density_truncated_norm"]

    ax_c.step(
        data_bin_centers, data_d,
        where="mid", lw=1.4, color=color, alpha=0.45, label=f"ABL{abl_int} data",
    )
    ax_c.plot(
        t_theory_115, theory_115,
        lw=1.8, color=color, linestyle="-", alpha=0.9, label=f"ABL{abl_int} 115ms",
    )
    ax_c.plot(
        t_theory_130, theory_130,
        lw=1.8, color=color, linestyle="--", alpha=0.9, label=f"ABL{abl_int} 130ms",
    )

ax_c.axvline(x=0.115, color="crimson", linestyle=":", linewidth=1.0, alpha=0.5)
ax_c.set_title("All ABLs")
ax_c.set_xlim(0.06, trunc_s)
ax_c.set_xlabel("RT - t_stim (s)")
ax_c.legend(fontsize=7, ncol=2)

if combined_max > 0:
    axes[0].set_ylim(0.0, combined_max * 1.1)

fig.suptitle(
    "Data (130ms) vs Model 115ms-fit  vs Model 130ms-fit   |  fixN=OFF",
    y=1.02,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out_base = DIAG_DIR / "compare_trunc115_vs_trunc130_allvalid"
fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
fig.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved: {out_base}.pdf / .png")
