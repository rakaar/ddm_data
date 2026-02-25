# %%
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
# %%
og_df = pd.read_csv('fit_animal_by_animal/batch_csvs/batch_LED8_valid_and_aborts.csv')
df_valid = og_df[
    og_df['success'].isin([1, -1]) &
    ((og_df['LED_trial'] == 0) | og_df['LED_trial'].isna())
].copy()

# %%
df_valid['timed_RT'].hist(bins=100)
# %%
ABL_unique = df_valid['ABL'].unique()
# %%
ABL_colors = {20: 'blue', 40: 'orange', 60: 'green'}
for ABL in ABL_unique:
    df_valid_ABL = df_valid[df_valid['ABL'] == ABL]
    # rt_abl_ms = 1000.0 * (df_valid_ABL['timed_fix'] - df_valid_ABL['intended_fix'])
    rt_abl_ms = 1000.0 * df_valid_ABL['RTwrtStim']
    rt_abl_ms = rt_abl_ms[rt_abl_ms >= 0]
    if rt_abl_ms.empty:
        continue
    max_edge = max(700, int(np.ceil(rt_abl_ms.max() / 20.0) * 20))
    plt.hist(
        rt_abl_ms,
        bins=np.arange(0, 1000, 10),
        density=True,
        color=ABL_colors.get(ABL, 'black'),
        alpha=0.5,
        histtype='step',
        label=f'ABL={ABL}'
    )
plt.xlim(0, 700)
plt.legend()
plt.title('RT Distribution by ABL')
# %%
def epanechnikov_kde(x_grid, samples, bandwidth):
    """Return KDE values at x_grid using an Epanechnikov kernel."""
    samples = np.asarray(samples)
    u = (x_grid[:, None] - samples[None, :]) / bandwidth
    kernel_vals = 0.75 * (1 - u**2)
    kernel_vals[np.abs(u) > 1] = 0.0
    return kernel_vals.sum(axis=1) / (samples.size * bandwidth)


def silverman_bandwidth(samples):
    """Robust Silverman bandwidth; falls back to a small default."""
    n = samples.size
    if n < 2:
        return 0.05
    std = np.std(samples, ddof=1)
    iqr = np.subtract(*np.percentile(samples, [75, 25]))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.05
    return max(1.06 * sigma * (n ** (-1 / 5)), 0.01)


x_grid = np.linspace(0, 1000, 1000)
for ABL in ABL_unique:
    df_valid_ABL = df_valid[df_valid['ABL'] == ABL]
    # rt_abl = (1000.0 * (df_valid_ABL['timed_fix'] - df_valid_ABL['intended_fix'])).dropna().to_numpy()
    rt_abl = (1000.0 * df_valid_ABL['RTwrtStim']).dropna().to_numpy()
    rt_abl = rt_abl[rt_abl >= 0]
    if rt_abl.size == 0:
        continue

    bw = silverman_bandwidth(rt_abl)
    kde_vals = epanechnikov_kde(x_grid, rt_abl, bw)
    plt.plot(
        x_grid,
        kde_vals,
        color=ABL_colors.get(ABL, 'black'),
        linewidth=2,
        label=f'ABL={ABL} KDE'
    )

plt.xlim(0, 700)
plt.xticks(np.arange(0, 701, 100))
plt.xlabel('RT (ms)')
plt.ylabel('Density')
plt.title('RT Distribution by ABL (Epanechnikov KDE)')
plt.legend()

# %%
