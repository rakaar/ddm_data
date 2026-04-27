# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "batch_csvs"

DATASETS = {
    "LED7": DATA_DIR / "batch_LED7_valid_and_aborts.csv",
    "LED8": DATA_DIR / "batch_LED8_valid_and_aborts.csv",
}

OUTPUT_PATHS = {
    "LED7": SCRIPT_DIR / "intended_fix_distribution_LED7.png",
    "LED8": SCRIPT_DIR / "intended_fix_distribution_LED8.png",
}

TARGET_COLUMN = "intended_fix"
NUM_BINS = 80


# %%
def shifted_exponential_pdf(t, a, tau):
    pdf = np.zeros_like(t, dtype=float)
    valid_mask = t >= a
    pdf[valid_mask] = np.exp(-(t[valid_mask] - a) / tau) / tau
    return pdf


# %%
figure_paths = {}

for batch_name, csv_path in DATASETS.items():
    df = pd.read_csv(csv_path)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"{TARGET_COLUMN} not found in {csv_path}")

    intended_fix = df[TARGET_COLUMN].dropna().to_numpy(dtype=float)

    a_hat = intended_fix.min()
    tau_hat = np.mean(intended_fix - a_hat)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(
        intended_fix,
        bins=NUM_BINS,
        density=True,
        alpha=0.55,
        color="tab:blue",
        edgecolor="white",
        label="Histogram",
    )

    t_grid = np.linspace(a_hat, intended_fix.max(), 500)
    ax.plot(
        t_grid,
        shifted_exponential_pdf(t_grid, a_hat, tau_hat),
        color="tab:red",
        linewidth=2.5,
        label=r"$(1/\tau)\exp(-(t-a)/\tau)$ fit",
    )

    ax.set_xlabel("intended_fix (s)")
    ax.set_ylabel("Density")
    ax.set_title(f"{batch_name}: a = {a_hat:.4f}, tau = {tau_hat:.4f}")
    ax.legend()
    fig.tight_layout()

    output_path = OUTPUT_PATHS[batch_name]
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    figure_paths[batch_name] = output_path.resolve()


# %%
for batch_name, output_path in figure_paths.items():
    print(f"{batch_name}: {output_path}")
