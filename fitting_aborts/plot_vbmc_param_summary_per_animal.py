# %%
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np


# %%
# =============================================================================
# Parameters
# =============================================================================
SCRIPT_DIR = os.path.dirname(__file__)
ANIMALS = None  # e.g. ["92", "93", "98"]; None => discover from pkl files
N_POSTERIOR_SAMPLES = int(1e5)
PARAM_LABELS = [
    "V_A_base",
    "V_A_post_LED",
    "theta_A",
    "del_a_minus_del_LED",
    "del_m_plus_del_LED",
]
DELAY_PARAM_INDICES = {3, 4}


# %%
# =============================================================================
# Helpers
# =============================================================================
def discover_animals_from_pkls(folder: str) -> list[str]:
    pattern = re.compile(r"^vbmc_real_(.+)_CORR_ID_fit\.pkl$")
    animals = []
    for filename in os.listdir(folder):
        m = pattern.match(filename)
        if m is not None:
            animals.append(m.group(1))
    return sorted(set(animals), key=animal_sort_key)


def animal_sort_key(animal_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(animal_id))
    except ValueError:
        return (1, animal_id)


def load_param_stats(vp_path: str, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(vp_path, "rb") as f:
        vp = pickle.load(f)
    vp_samples = vp.sample(n_samples)[0]
    mean = np.mean(vp_samples, axis=0)
    q025 = np.quantile(vp_samples, 0.025, axis=0)
    q975 = np.quantile(vp_samples, 0.975, axis=0)
    return mean, q025, q975


# %%
# =============================================================================
# Load per-animal parameter summaries
# =============================================================================
if ANIMALS is None:
    animals = discover_animals_from_pkls(SCRIPT_DIR)
else:
    animals = sorted((str(a) for a in ANIMALS), key=animal_sort_key)

if len(animals) == 0:
    raise RuntimeError("No per-animal VBMC pkl files found.")

print(f"Animals to plot ({len(animals)}): {animals}")

means_all = []
q025_all = []
q975_all = []
animals_loaded = []

for animal in animals:
    vp_path = os.path.join(SCRIPT_DIR, f"vbmc_real_{animal}_CORR_ID_fit.pkl")
    if not os.path.exists(vp_path):
        print(f"Warning: missing {vp_path}, skipping animal {animal}")
        continue

    mean, q025, q975 = load_param_stats(vp_path, N_POSTERIOR_SAMPLES)
    means_all.append(mean)
    q025_all.append(q025)
    q975_all.append(q975)
    animals_loaded.append(animal)

if len(animals_loaded) == 0:
    raise RuntimeError("No animals could be loaded from pkl files.")

animals_loaded = np.array(animals_loaded, dtype=str)
means_all = np.array(means_all)
q025_all = np.array(q025_all)
q975_all = np.array(q975_all)


# %%
# =============================================================================
# Plot one figure per parameter
# =============================================================================
for p_idx, p_name in enumerate(PARAM_LABELS):
    scale = 1000.0 if p_idx in DELAY_PARAM_INDICES else 1.0
    unit = "ms" if p_idx in DELAY_PARAM_INDICES else ""

    y = means_all[:, p_idx] * scale
    y_low = q025_all[:, p_idx] * scale
    y_high = q975_all[:, p_idx] * scale
    yerr = np.vstack([y - y_low, y_high - y])

    x_pos = np.arange(len(animals_loaded))

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(animals_loaded)), 4.5))
    ax.errorbar(
        x_pos,
        y,
        yerr=yerr,
        fmt="o",
        color="k",
        ecolor="tab:blue",
        elinewidth=2,
        capsize=4,
        ms=6,
    )
    ax.set_xlabel("Animal")
    y_label = f"{p_name} ({unit})" if unit else p_name
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} per animal (mean with 2.5-97.5 percentile)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(animals_loaded)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, f"vbmc_{p_name}_per_animal_errorbar.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()
