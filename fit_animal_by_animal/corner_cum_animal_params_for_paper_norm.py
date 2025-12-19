# %%
"""
Corner-style cumulative animal-parameter plot for the Norm TIED model.

For each animal (across selected batches), load vbmc Norm TIED results,
compute the mean and std of each parameter's samples, and plot a
lower-triangular corner matrix where each off-diagonal panel shows
per-animal means for (param_x, param_y). The diagonal shows histograms
of per-animal means for that parameter.

Defaults are aligned with compare_animal_params_for_paper.py logic.

Example:
    python corner_cum_animal_params_for_paper.py \
        --params rate_lambda T_0 theta_E rate_norm_l \
        --outfile corner_cum_norm_tied_params.pdf

Outputs:
- A PDF saved next to this script under the provided --outfile name.
- Optionally, a CSV with per-animal means/stds via --csv-out <path>.
"""
import os
import argparse
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as MplEllipse
import importlib.util
import sys
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde
try:
    # Prefer shared figure style if available via package-style import
    from fit_animal_by_animal.figure_template import STYLE  # type: ignore
    STYLE.apply()
except Exception:
    # Fallback applied later once RESULTS_DIR is known
    STYLE = None  # type: ignore

# Directory containing the results (same as this file). When run as a #%% cell,
# __file__ may be undefined; fall back to CWD or project subdir.
try:
    RESULTS_DIR = os.path.dirname(__file__)
except NameError:
    _cwd = os.getcwd()
    _cand = os.path.join(_cwd, 'fit_animal_by_animal')
    RESULTS_DIR = _cand if os.path.isdir(_cand) else _cwd

# Fallback: apply style by loading figure_template.py directly if import above failed
if 'STYLE' not in globals() or STYLE is None:  # type: ignore
    _ft_path = os.path.join(RESULTS_DIR, 'figure_template.py')
    if os.path.exists(_ft_path):
        try:
            _spec = importlib.util.spec_from_file_location('figure_template_style', _ft_path)
            if _spec and _spec.loader:
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
                if hasattr(_mod, 'STYLE'):
                    _mod.STYLE.apply()
        except Exception:
            pass

# Default batches to consider (keep in sync with compare_animal_params_for_paper.py)
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']

# Simple, high-contrast colors per batch (keep in sync with compare_animal_params_for_paper.py)
BATCH_COLORS = {
    'Comparable': 'red',
    'SD': '#87CEEB',       # sky blue
    'LED2': 'green',
    'LED1': 'orange',
    'LED34': 'purple',
    'LED7': 'black',
    'LED34_even': 'blue',
}

# Norm TIED model: mapping from clean param label -> sample key in PKL
NORM_TIED_PARAM_KEYMAP = {
    'rate_lambda': 'rate_lambda_samples',
    'T_0': 'T_0_samples',
    'theta_E': 'theta_E_samples',
    'w': 'w_samples',
    't_E_aff': 't_E_aff_samples',
    'del_go': 'del_go_samples',
    'rate_norm_l': 'rate_norm_l_samples',
}

# Pretty LaTeX-style labels per parameter (MathText, no external LaTeX needed)
PARAM_TEX_LABELS: Dict[str, str] = {
    'rate_lambda': r'$\lambda$',
    'T_0': r'$T_0$',
    'theta_E': r'$\theta_E$',
    'w': r'$w$',
    't_E_aff': r'$t^{\mathrm{aff}}_E$',
    'del_go': r'$\Delta_{go}$',
    'rate_norm_l': r'$\ell$',
}

MODEL_KEY = 'vbmc_norm_tied_results'

# Optional default axis ranges to keep ticks simple (endpoints only)
DEFAULT_AXIS_RANGES: Dict[str, Tuple[float, float]] = {
    'rate_lambda': (0.9, 3.3),
    'T_0': (0.04, 0.25),
    'theta_E': (1.2, 3.4),
    'rate_norm_l': (0.8, 1.0),
}


def discover_animals(results_dir: str, desired_batches: List[str]) -> List[Tuple[str, int]]:
    """Return sorted list of (batch, animal_id) that have PKL results files.

    Tries CSV discovery first (valid trials), then falls back to scanning PKLs.
    """
    batch_dir = os.path.join(results_dir, 'batch_csvs')
    batch_files = [f'batch_{b}_valid_and_aborts.csv' for b in desired_batches]
    dfs = []
    for fname in batch_files:
        fpath = os.path.join(batch_dir, fname)
        if os.path.exists(fpath):
            try:
                dfs.append(pd.read_csv(fpath))
            except Exception as e:
                print(f"Warning: failed reading {fpath}: {e}")
    batch_animal_pairs: List[Tuple[str, str]] = []
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged_valid = merged[merged['success'].isin([1, -1])].copy()
        pairs = merged_valid[['batch_name', 'animal']].drop_duplicates().values
        batch_animal_pairs = sorted(list(map(tuple, pairs)))
        print(f"Found {len(batch_animal_pairs)} batch-animal pairs from CSVs.")
    else:
        print('Warning: No batch CSVs found for DESIRED_BATCHES. Falling back to scanning PKL files in RESULTS_DIR.')

    animal_batch_tuples: List[Tuple[str, int]] = []
    if batch_animal_pairs:
        for (batch, animal) in batch_animal_pairs:
            try:
                animal_id = int(animal)
            except Exception:
                # Skip non-integer animal identifiers
                continue
            pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
            if os.path.exists(os.path.join(results_dir, pkl_fname)):
                animal_batch_tuples.append((batch, animal_id))
    else:
        # Fallback: scan directory for PKL files per desired batch
        for fname in os.listdir(results_dir):
            if not (fname.startswith('results_') and fname.endswith('.pkl')):
                continue
            for batch in desired_batches:
                prefix = f'results_{batch}_animal_'
                if fname.startswith(prefix):
                    try:
                        animal_id = int(fname.split('_')[-1].replace('.pkl', ''))
                        animal_batch_tuples.append((batch, animal_id))
                    except Exception:
                        pass

    return sorted(animal_batch_tuples, key=lambda x: (x[0], x[1]))


def load_means_stds_for_norm_tied(
    results_dir: str,
    animal_batch_tuples: List[Tuple[str, int]],
    params: List[str],
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[str], List[str]]:
    """Load vbmc_norm_tied_results and compute per-animal means/stds for given params.

    Returns:
        means_dict: param -> list of per-animal means
        stds_dict:  param -> list of per-animal stds
        labels:     list of 'BATCH-ID' labels per animal (same order across params)
        colors:     list of colors per animal based on batch
    """
    means_dict: Dict[str, List[float]] = {p: [] for p in params}
    stds_dict: Dict[str, List[float]] = {p: [] for p in params}
    labels: List[str] = []
    colors: List[str] = []

    for batch, animal_id in animal_batch_tuples:
        pkl_path = os.path.join(results_dir, f'results_{batch}_animal_{animal_id}.pkl')
        if not os.path.exists(pkl_path):
            continue
        try:
            with open(pkl_path, 'rb') as f:
                results = pickle.load(f)
        except Exception as e:
            print(f"Warning: failed loading {pkl_path}: {e}")
            continue

        if MODEL_KEY not in results:
            continue
        model_blob = results[MODEL_KEY]

        ok = True
        per_param_stats = {}
        for p in params:
            skey = NORM_TIED_PARAM_KEYMAP.get(p)
            if skey is None or skey not in model_blob:
                ok = False
                break
            s = np.asarray(model_blob[skey])
            if s.size == 0 or not np.isfinite(s).any():
                ok = False
                break
            per_param_stats[p] = (float(np.nanmean(s)), float(np.nanstd(s)))
        if not ok:
            continue

        # Keep same order across params
        for p in params:
            m, sd = per_param_stats[p]
            means_dict[p].append(m)
            stds_dict[p].append(sd)
        labels.append(f"{batch}-{animal_id}")
        colors.append(BATCH_COLORS.get(batch, 'gray'))

    return means_dict, stds_dict, labels, colors


def load_samples_flat_for_norm_tied(
    results_dir: str,
    animal_batch_tuples: List[Tuple[str, int]],
    params: List[str],
    n_samples_per_animal: int = 1000,
    seed: int = 0,
) -> Tuple[Dict[str, List[float]], List[str], List[str]]:
    """Return flattened joint samples across animals for given params.

    For each animal, draws up to `n_samples_per_animal` indices from the
    posterior sample arrays and uses the SAME indices across all parameters
    to preserve joint correlations. The per-animal samples are then appended
    into one flat list per parameter.

    Returns:
        samples_dict: param -> concatenated list of samples across animals
        labels:       list of labels per animal (unchanged)
        colors:       list of colors per POINT (same color across all points)
    """
    rng = np.random.default_rng(seed)
    samples_dict: Dict[str, List[float]] = {p: [] for p in params}
    labels: List[str] = []
    colors: List[str] = []

    # Use a single color for every point
    point_color = '#1f77b4'

    for batch, animal_id in animal_batch_tuples:
        pkl_path = os.path.join(results_dir, f'results_{batch}_animal_{animal_id}.pkl')
        if not os.path.exists(pkl_path):
            continue
        try:
            with open(pkl_path, 'rb') as f:
                results = pickle.load(f)
        except Exception as e:
            print(f"Warning: failed loading {pkl_path}: {e}")
            continue

        if MODEL_KEY not in results:
            continue
        model_blob = results[MODEL_KEY]

        # Verify all requested params exist and get lengths
        arrs = []
        ok = True
        for p in params:
            skey = NORM_TIED_PARAM_KEYMAP.get(p)
            if skey is None or skey not in model_blob:
                ok = False
                break
            s = np.asarray(model_blob[skey]).reshape(-1)
            if s.size == 0 or not np.isfinite(s).any():
                ok = False
                break
            arrs.append(s)
        if not ok:
            continue

        L = min(a.size for a in arrs)
        k = min(n_samples_per_animal, L) if L > 0 else 0
        if k == 0:
            continue
        # sample indices with replacement only if necessary
        replace = L < k
        idx = rng.choice(L, size=k, replace=replace)

        # Append samples per parameter
        for p, s in zip(params, arrs):
            samples_dict[p].extend(np.asarray(s[idx], dtype=float).tolist())

        labels.append(f"{batch}-{animal_id}")
        colors.extend([point_color] * k)

    return samples_dict, labels, colors


def load_samples_grouped_for_norm_tied(
    results_dir: str,
    animal_batch_tuples: List[Tuple[str, int]],
    params: List[str],
    n_samples_per_animal: int = 1000,
    seed: int = 0,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str], List[str]]:
    """Return grouped joint samples per animal for given params.

    Returns:
        grouped: label -> {param -> samples (1D np.ndarray)}
        labels:  list of labels (order corresponds to colors list)
        colors:  list of one color per label (batch-based)
    """
    rng = np.random.default_rng(seed)
    grouped: Dict[str, Dict[str, np.ndarray]] = {}
    labels: List[str] = []
    colors: List[str] = []

    for batch, animal_id in animal_batch_tuples:
        pkl_path = os.path.join(results_dir, f'results_{batch}_animal_{animal_id}.pkl')
        if not os.path.exists(pkl_path):
            continue
        try:
            with open(pkl_path, 'rb') as f:
                results = pickle.load(f)
        except Exception as e:
            print(f"Warning: failed loading {pkl_path}: {e}")
            continue

        if MODEL_KEY not in results:
            continue
        model_blob = results[MODEL_KEY]

        # Verify arrays and pick joint indices
        arrs = []
        ok = True
        for p in params:
            skey = NORM_TIED_PARAM_KEYMAP.get(p)
            if skey is None or skey not in model_blob:
                ok = False
                break
            s = np.asarray(model_blob[skey]).reshape(-1)
            if s.size == 0 or not np.isfinite(s).any():
                ok = False
                break
            arrs.append(s)
        if not ok:
            continue

        L = min(a.size for a in arrs)
        k = min(n_samples_per_animal, L) if L > 0 else 0
        if k == 0:
            continue
        replace = L < k
        idx = rng.choice(L, size=k, replace=replace)

        label = f"{batch}-{animal_id}"
        grouped[label] = {}
        for p, s in zip(params, arrs):
            grouped[label][p] = np.asarray(s[idx], dtype=float)
        labels.append(label)
        colors.append(BATCH_COLORS.get(batch, '#1f77b4'))

    return grouped, labels, colors

def _axis_limits(values: List[float], pad_frac: float = 0.05) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if vmin == vmax:
        if vmin == 0:
            return -0.5, 0.5
        span = abs(vmin) * 0.1
        return vmin - span, vmax + span
    pad = (vmax - vmin) * pad_frac
    return vmin - pad, vmax + pad


def corner_plot(
    means: Dict[str, List[float]],
    labels: List[str],
    colors: List[str],
    params: List[str],
    title: str,
    outfile: str,
    point_size: float = 30.0,
    alpha: float = 0.9,
    show_legend: bool = True,
    overlay_means: Optional[Dict[str, List[float]]] = None,
    overlay_point_size: Optional[float] = None,
    overlay_color: str = '#8B0000',
    overlay_edgecolor: str = 'k',
    overlay_alpha: float = 0.95,
    plot_scatter: bool = True,
    ellipses_from_grouped: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ellipse_colors_by_label: Optional[Dict[str, str]] = None,
    ellipse_quantile: float = 0.95,
    ellipse_alpha: float = 1.0,
    ellipse_linewidth: float = 1.0,
    ellipse_edgecolor: str = '#2b6cb0',
    use_kde: bool = False,
    kde_bw: Optional[float] = None,
    kde_n: int = 1000,
    diag_ranked: bool = False,
    diag_ranked_ci: float = 95.0,
    tick_labelsize: Optional[float] = None,
    label_fontsize: Optional[float] = None,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
):
    """Make lower-triangular corner-style scatter matrix of per-animal means.

    Diagonal panels show histograms of per-animal means for each parameter.
    """
    n = len(params)
    if n == 0:
        print('No parameters to plot. Exiting.')
        return

    # Precompute limits per param
    lims: Dict[str, Tuple[float, float]] = {}
    for p in params:
        if axis_ranges is not None and p in axis_ranges:
            lims[p] = tuple(axis_ranges[p])  # type: ignore[assignment]
        else:
            lims[p] = _axis_limits(means.get(p, []))

    fig, axes = plt.subplots(n, n, figsize=(3.2*n, 3.2*n), squeeze=False)

    # Fixed sizes unless explicitly overridden
    eff_tick = float(tick_labelsize) if tick_labelsize is not None else 20.0
    eff_label = float(label_fontsize) if label_fontsize is not None else 25.0

    for i, py in enumerate(params):
        for j, px in enumerate(params):
            ax = axes[i, j]
            if i == j:
                # Diagonal panel modes
                if diag_ranked and ellipses_from_grouped is not None and len(ellipses_from_grouped) > 0:
                    # Ranked per-animal posterior means with CI from percentiles
                    stats = []  # (label, mean, lo, hi)
                    ci = float(diag_ranked_ci)
                    ci = min(max(ci, 0.0), 100.0)
                    lo_p = 0.5 * (100.0 - ci)
                    hi_p = 100.0 - lo_p
                    for lab, pdata in ellipses_from_grouped.items():
                        arr = np.asarray(pdata.get(px, []), dtype=float)
                        arr = arr[np.isfinite(arr)]
                        if arr.size < 2:
                            continue
                        m = float(np.mean(arr))
                        try:
                            lo = float(np.percentile(arr, lo_p))
                            hi = float(np.percentile(arr, hi_p))
                        except Exception:
                            # Fallback to mean +/- 1.96*std if percentile fails
                            sd = float(np.std(arr)) if arr.size > 1 else 0.0
                            lo, hi = m - 1.96 * sd, m + 1.96 * sd
                        stats.append((lab, m, lo, hi))
                    if len(stats) > 0:
                        stats.sort(key=lambda t: t[1], reverse=True)  # descending by mean
                        # Plot horizontal error bars; x is value, y is rank index
                        for k, (lab, m, lo, hi) in enumerate(stats):
                            # Fixed red color for all animals (no per-animal/batch coloring)
                            col = '#8B0000'
                            xerr_low = max(0.0, m - lo)
                            xerr_high = max(0.0, hi - m)
                            ax.errorbar(
                                m, k,
                                xerr=[[xerr_low], [xerr_high]],
                                fmt='o', color=col, ecolor=col,
                                elinewidth=1.4, capsize=0, markersize=4,
                                markeredgecolor='k', linewidth=1.0, alpha=0.95,
                            )
                        ax.set_xlim(lims[px])
                        ax.set_ylim(-0.5, len(stats) - 0.5)
                        # Hide animal ticks (y); show two x ticks at endpoints
                        ax.set_yticks([])
                        _xt = np.linspace(lims[px][0], lims[px][1], 2)
                        ax.set_xticks(_xt)
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    else:
                        ax.axis('off')
                else:
                    # Histogram (default) or KDE of per-animal means for this param
                    vals = np.asarray(means.get(px, []), dtype=float)
                    vals = vals[np.isfinite(vals)]
                    drew_kde = False
                    if use_kde and vals.size > 1:
                        try:
                            bw = kde_bw if kde_bw is not None else None
                            kde = gaussian_kde(vals, bw_method=bw)
                            _kde_n = int(kde_n) if kde_n is not None else 256
                            _kde_n = max(64, min(10000, _kde_n))
                            xs = np.linspace(lims[px][0], lims[px][1], _kde_n)
                            ys = kde(xs)
                            ax.plot(xs, ys, color='#2b6cb0', linewidth=1.5)
                            drew_kde = True
                        except Exception:
                            drew_kde = False
                    if not drew_kde:
                        if vals.size > 0:
                            ax.hist(
                                vals,
                                bins=min(20, max(5, int(np.sqrt(vals.size)))),
                                histtype='step',
                                color='#2b6cb0',
                                linewidth=1.2,
                            )
                    ax.set_xlim(lims[px])
                    # No y-ticks on histogram/KDE panels
                    ax.set_yticks([])
                    # Only two x-ticks on histogram/KDE panels
                    _xt = np.linspace(lims[px][0], lims[px][1], 2)
                    ax.set_xticks(_xt)
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            elif i > j:
                # Lower triangle: scatter of (px mean, py mean)
                xs = means.get(px, [])
                ys = means.get(py, [])
                if plot_scatter and len(xs) == len(ys) == len(colors):
                    ax.scatter(xs, ys, s=point_size, c=colors, alpha=alpha, edgecolor='k', linewidths=0.3)
                # Draw filled covariance ellipses per animal if requested (only when not plotting scatter)
                if (ellipses_from_grouped is not None and len(ellipses_from_grouped) > 0 and not plot_scatter):
                    # chi2 quantile for df=2 has closed form: s = -2 ln(1 - q)
                    q = float(ellipse_quantile)
                    if not (0.0 < q < 1.0):
                        q = 0.95
                    s_chi2 = -2.0 * np.log(max(1e-12, 1.0 - q))
                    for lab, pdata in ellipses_from_grouped.items():
                        x = np.asarray(pdata.get(px, []), dtype=float)
                        y = np.asarray(pdata.get(py, []), dtype=float)
                        if x.size < 2 or y.size < 2:
                            continue
                        x = x[np.isfinite(x)]
                        y = y[np.isfinite(y)]
                        m_x = float(np.mean(x)) if x.size else np.nan
                        m_y = float(np.mean(y)) if y.size else np.nan
                        if not (np.isfinite(m_x) and np.isfinite(m_y)):
                            continue
                        # 2x2 covariance
                        cov = np.cov(np.vstack([x, y]))
                        if not np.all(np.isfinite(cov)):
                            continue
                        try:
                            evals, evecs = np.linalg.eigh(cov)
                        except np.linalg.LinAlgError:
                            continue
                        order = np.argsort(evals)[::-1]
                        evals = np.maximum(evals[order], 0.0)
                        evecs = evecs[:, order]
                        # ellipse axes (width/height) are 2*sqrt(s*lambda)
                        width = 2.0 * float(np.sqrt(s_chi2 * evals[0])) if evals.size > 0 else 0.0
                        height = 2.0 * float(np.sqrt(s_chi2 * evals[1])) if evals.size > 1 else 0.0
                        if width == 0.0 or height == 0.0:
                            continue
                        angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
                        # Single-color, outline-only ellipse
                        col = ellipse_edgecolor
                        patch = MplEllipse(
                            (m_x, m_y), width=width, height=height, angle=angle,
                            facecolor='none', edgecolor=col, linewidth=ellipse_linewidth,
                            alpha=ellipse_alpha, zorder=4,
                        )
                        ax.add_patch(patch)
                # Overlay per-animal means on top (e.g., when background are samples)
                if overlay_means is not None:
                    xs_ov = overlay_means.get(px, [])
                    ys_ov = overlay_means.get(py, [])
                    if len(xs_ov) > 0 and len(xs_ov) == len(ys_ov):
                        s_ov = overlay_point_size if overlay_point_size is not None else point_size * 1.6
                        ax.scatter(
                            xs_ov,
                            ys_ov,
                            s=s_ov,
                            c=overlay_color,
                            alpha=overlay_alpha,
                            edgecolor=overlay_edgecolor,
                            linewidths=0.6,
                            zorder=5,
                        )
                ax.set_xlim(lims[px])
                ax.set_ylim(lims[py])
                # Only two ticks on both axes
                _xt = np.linspace(lims[px][0], lims[px][1], 2)
                _yt = np.linspace(lims[py][0], lims[py][1], 2)
                ax.set_xticks(_xt)
                ax.set_yticks(_yt)
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            else:
                # Upper triangle: hide
                ax.axis('off')
                continue

            # Axes labels only on left col and bottom row
            if i == n - 1:
                ax.set_xlabel(PARAM_TEX_LABELS.get(px, px))
            else:
                ax.set_xticklabels([])
            if j == 0:
                # For diagonal ranked panels (horizontal), suppress y label (no animal ticks/label)
                if not (i == j and diag_ranked):
                    ax.set_ylabel(PARAM_TEX_LABELS.get(py, py))
            else:
                ax.set_yticklabels([])

            # Apply larger tick label size
            ax.tick_params(axis='both', which='major', labelsize=eff_tick)
            # Apply larger axis label size
            try:
                ax.xaxis.label.set_size(eff_label)
                ax.yaxis.label.set_size(eff_label)
            except Exception:
                pass

            # Uniform subplot borders: avoid double-drawn shared edges
            spine_lw = 1.0
            for side in ('left', 'bottom', 'right', 'top'):
                ax.spines[side].set_linewidth(spine_lw)
                ax.spines[side].set_visible(True)
            # Hide right spine only if the right neighbor is a visible panel (i >= j+1)
            if j < n - 1 and i >= j + 1:
                ax.spines['right'].set_visible(False)
            # Hide top spine only if the upper neighbor is a visible panel ((i-1) >= j)
            if i > 0 and (i - 1) >= j:
                ax.spines['top'].set_visible(False)

            ax.grid(True, linestyle=':', alpha=0.4)

    if show_legend:
        # Build legend mapping batches -> color from the per-animal colors list
        batch_to_color: Dict[str, str] = {}
        for lab, col in zip(labels, colors):  # labels usually per-animal
            if '-' in lab:
                bname = lab.split('-', 1)[0]
                if bname not in batch_to_color:
                    batch_to_color[bname] = col
        if batch_to_color:
            handles = []
            labels_ = []
            for b, c in batch_to_color.items():
                h = plt.Line2D([0], [0], marker='o', color='w', label=b,
                               markerfacecolor=c, markeredgecolor='k', markersize=7)
                handles.append(h)
                labels_.append(b)
            fig.legend(handles, labels_, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
    outpath = os.path.join(RESULTS_DIR, outfile)
    fig.savefig(outpath)
    # plt.close(fig)
    print(f'Saved corner plot: {outpath}')


def save_csv(
    csv_path: str,
    means: Dict[str, List[float]],
    stds: Dict[str, List[float]],
    labels: List[str],
):
    rows = []
    for idx, label in enumerate(labels):
        row = {'label': label}
        for p in means.keys():
            row[f'{p}_mean'] = means[p][idx] if idx < len(means[p]) else np.nan
            row[f'{p}_std'] = stds[p][idx] if idx < len(stds[p]) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f'Saved CSV: {csv_path}')


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI args, tolerating unknown flags (e.g., Jupyter/VSCode kernel args).

    Using parse_known_args allows running this file as a #%% cell where the
    kernel injects flags like `--f=...`.
    """
    parser = argparse.ArgumentParser(description='Corner-style plot of Norm TIED per-animal parameter means', allow_abbrev=False)
    parser.add_argument('--batches', nargs='*', default=DESIRED_BATCHES,
                        help='Batches to include (default: %(default)s)')
    parser.add_argument('--params', nargs='*', default=[],
                        help='Explicit parameter names to include; if omitted, uses --param-set')
    parser.add_argument('--param-set', choices=['imp', 'all'], default='imp',
                        help='Quick selection: imp=[rate_lambda,T_0,theta_E,rate_norm_l]; all=all Norm-TIED params')
    parser.add_argument('--outfile', default='corner_cum_animal_params_norm_tied.pdf',
                        help='Output PDF filename (saved in this directory)')
    parser.add_argument('--csv-out', default='', help='Optional CSV path to save per-animal means/stds')
    parser.add_argument('--point-size', type=float, default=30.0, help='Scatter point size')
    parser.add_argument('--alpha', type=float, default=0.9, help='Scatter alpha')
    parser.add_argument('--mode', choices=['mean', 'samples'], default='samples',
                        help="'mean' plots per-animal means; 'samples' plots flattened posterior samples")
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Number of joint samples per animal when --mode samples')
    parser.add_argument('--fit-ellipses', action='store_true', default=True,
                        help='In samples mode, draw filled covariance ellipses per animal instead of sample points (default: on)')
    parser.add_argument('--no-ellipses', action='store_false', dest='fit_ellipses',
                        help='Disable ellipses and show sample scatter points instead')
    parser.add_argument('--ellipse-quantile', type=float, default=0.95,
                        help='Ellipse confidence quantile (df=2 chi-square), e.g., 0.95')
    parser.add_argument('--ellipse-alpha', type=float, default=1.0,
                        help='Ellipse edge alpha (transparency)')
    parser.add_argument('--ellipse-linewidth', type=float, default=1.0,
                        help='Ellipse edge linewidth')
    parser.add_argument('--ellipse-color', type=str, default='#2b6cb0',
                        help='Single edge color for all ellipses')
    # Diagonal KDE option
    parser.add_argument('--kde', action='store_true',
                        help='Use KDE instead of histogram on diagonal panels', default=True)
    parser.add_argument('--kde-bw', default='',
                        help='KDE bandwidth: numeric factor or "scott"/"silverman" (default: auto)')
    parser.add_argument('--kde-n', type=int, default=1000,
                        help='Number of evaluation points for KDE on diagonals (default: 1000)')
    parser.add_argument('--diag-ranked', action='store_true', default=True,
                        help='On diagonal, show ranked per-animal posterior means with CI instead of histogram/KDE (default: on)')
    parser.add_argument('--no-diag-ranked', action='store_false', dest='diag_ranked',
                        help='Disable ranked diagonals and fall back to histogram/KDE')
    parser.add_argument('--diag-ci', type=float, default=95.0,
                        help='CI percentage for --diag-ranked (e.g., 95)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling indices')
    # When run inside Jupyter/VSCode cells, sys.argv may contain kernel flags like: -f /path/kernel.json
    # Clean these out so boolean flags like --fit-ellipses don't accidentally consume them.
    if argv is None:
        argv = sys.argv[1:]
    cleaned: List[str] = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in ('-f', '--f'):
            skip_next = True  # drop the following kernel json path
            continue
        # Sometimes the kernel path may appear without the -f prefix; drop it heuristically
        if tok.endswith('.json') and ('/jupyter/runtime/' in tok or tok.startswith('/run/user/')):
            continue
        cleaned.append(tok)

    args, _unknown = parser.parse_known_args(cleaned)
    return args


def main(argv=None):
    args = parse_args(argv)

    # Select params: explicit --params overrides --param-set
    if getattr(args, 'params', None):
        params = [p for p in args.params if p in NORM_TIED_PARAM_KEYMAP]
    else:
        if getattr(args, 'param_set', 'imp') == 'all':
            params = list(NORM_TIED_PARAM_KEYMAP.keys())
        else:
            params = ['rate_lambda', 'T_0', 'theta_E', 'rate_norm_l']
    if not params:
        raise ValueError('No valid parameters selected. Valid: %s' % ', '.join(NORM_TIED_PARAM_KEYMAP.keys()))

    # Discover animals
    animal_tuples = discover_animals(RESULTS_DIR, args.batches)

    # Parse KDE bandwidth once
    kde_bw_parsed = None
    try:
        _bw = getattr(args, 'kde_bw', '')
        if isinstance(_bw, (int, float)):
            kde_bw_parsed = float(_bw)
        else:
            _s = str(_bw).strip()
            if _s:
                _sl = _s.lower()
                if _sl in ('scott', 'silverman'):
                    kde_bw_parsed = _sl
                else:
                    try:
                        kde_bw_parsed = float(_s)
                    except Exception:
                        print(f"Warning: invalid --kde-bw value '{_s}', using default.")
                        kde_bw_parsed = None
    except Exception:
        kde_bw_parsed = None

    if args.mode == 'samples':
        # Load flattened joint samples across animals
        values, labels, colors = load_samples_flat_for_norm_tied(
            RESULTS_DIR, animal_tuples, params, n_samples_per_animal=args.n_samples, seed=args.seed
        )
        if all(len(values.get(p, [])) == 0 for p in params):
            print('No samples found for selected batches/params.')
            return
        # Compute per-animal means for overlay
        means_overlay, _stds_overlay, _labels_overlay, _colors_overlay = load_means_stds_for_norm_tied(
            RESULTS_DIR, animal_tuples, params
        )
        if args.fit_ellipses:
            # Grouped samples per animal to fit ellipses (colors by batch)
            grouped, labels_grp, colors_grp = load_samples_grouped_for_norm_tied(
                RESULTS_DIR, animal_tuples, params, n_samples_per_animal=args.n_samples, seed=args.seed
            )
            ellipse_color_map = {lab: col for lab, col in zip(labels_grp, colors_grp)}
            title = f'Norm TIED: posterior ellipses per animal (q={args.ellipse_quantile:.2f}, n={args.n_samples})'
            corner_plot(
                means=values,  # used for axis limits
                labels=labels,
                colors=colors,
                params=params,
                title=title,
                outfile=args.outfile,
                point_size=args.point_size,
                alpha=args.alpha,
                show_legend=False,
                overlay_means=means_overlay,  # overlay red mean dots
                plot_scatter=False,
                ellipses_from_grouped=grouped,
                ellipse_colors_by_label=None,
                ellipse_quantile=args.ellipse_quantile,
                ellipse_alpha=args.ellipse_alpha,
                ellipse_linewidth=args.ellipse_linewidth,
                ellipse_edgecolor=args.ellipse_color,
                use_kde=args.kde,
                kde_bw=kde_bw_parsed,
                kde_n=args.kde_n,
                diag_ranked=args.diag_ranked,
                diag_ranked_ci=args.diag_ci,
                axis_ranges=DEFAULT_AXIS_RANGES,
            )
        else:
            title = f'Norm TIED: posterior samples per animal (n={args.n_samples} each)'
            grouped_for_diag = None
            if args.diag_ranked:
                grouped_for_diag, _lab_g, _col_g = load_samples_grouped_for_norm_tied(
                    RESULTS_DIR, animal_tuples, params, n_samples_per_animal=args.n_samples, seed=args.seed
                )
            corner_plot(
                means=values,
                labels=labels,
                colors=colors,  # one color per point
                params=params,
                title=title,
                outfile=args.outfile,
                point_size=args.point_size,
                alpha=args.alpha,
                show_legend=False,
                overlay_means=means_overlay,
                ellipses_from_grouped=grouped_for_diag,
                use_kde=args.kde,
                kde_bw=kde_bw_parsed,
                kde_n=args.kde_n,
                diag_ranked=args.diag_ranked,
                diag_ranked_ci=args.diag_ci,
                axis_ranges=DEFAULT_AXIS_RANGES,
            )
    else:
        # Default mean mode
        means, stds, labels, colors = load_means_stds_for_norm_tied(RESULTS_DIR, animal_tuples, params)
        if len(labels) == 0:
            print('No animals with Norm TIED results found for selected batches/params.')
            return
        # Use a single color for all animals
        colors = ['#8B0000'] * len(labels)
        title = 'Norm TIED: per-animal parameter means'
        grouped_for_diag = None
        if args.diag_ranked:
            grouped_for_diag, _lab_g, _col_g = load_samples_grouped_for_norm_tied(
                RESULTS_DIR, animal_tuples, params, n_samples_per_animal=args.n_samples, seed=args.seed
            )
        corner_plot(
            means=means,
            labels=labels,
            colors=colors,
            params=params,
            title=title,
            outfile=args.outfile,
            point_size=args.point_size,
            alpha=args.alpha,
            show_legend=False,
            ellipses_from_grouped=grouped_for_diag,
            use_kde=args.kde,
            kde_bw=kde_bw_parsed,
            kde_n=args.kde_n,
            diag_ranked=args.diag_ranked,
            diag_ranked_ci=args.diag_ci,
            axis_ranges=DEFAULT_AXIS_RANGES,
        )
        if args.csv_out:
            csv_path = args.csv_out
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(RESULTS_DIR, csv_path)
            save_csv(csv_path, means, stds, labels)


def save_corner_data_to_pkl(
    outpath: str = 'norm_model_fit_params_corner_plot_all_animals.pkl',
    batches: List[str] = DESIRED_BATCHES,
    params: Optional[List[str]] = None,
    n_samples: int = 5000,
    seed: int = 0,
    ellipse_quantile: float = 0.95,
):
    """Save all data needed for the corner plot to a pickle file.
    
    This allows the corner plot to be recreated without re-loading all
    individual animal result files.
    """
    if params is None:
        params = ['rate_lambda', 'T_0', 'theta_E', 'rate_norm_l']
    
    animal_tuples = discover_animals(RESULTS_DIR, batches)
    
    # Load per-animal means/stds for overlay
    means_overlay, stds_overlay, labels_overlay, colors_overlay = load_means_stds_for_norm_tied(
        RESULTS_DIR, animal_tuples, params
    )
    
    # Load flattened samples (for axis limits)
    values_flat, labels_flat, colors_flat = load_samples_flat_for_norm_tied(
        RESULTS_DIR, animal_tuples, params, n_samples_per_animal=n_samples, seed=seed
    )
    
    # Load grouped samples per animal (for ellipses and diag ranked)
    grouped, labels_grp, colors_grp = load_samples_grouped_for_norm_tied(
        RESULTS_DIR, animal_tuples, params, n_samples_per_animal=n_samples, seed=seed
    )
    
    corner_data = {
        'params': params,
        'means_overlay': means_overlay,
        'stds_overlay': stds_overlay,
        'labels_overlay': labels_overlay,
        'colors_overlay': colors_overlay,
        'values_flat': values_flat,
        'labels_flat': labels_flat,
        'colors_flat': colors_flat,
        'grouped': grouped,
        'labels_grp': labels_grp,
        'colors_grp': colors_grp,
        'axis_ranges': DEFAULT_AXIS_RANGES,
        'ellipse_quantile': ellipse_quantile,
        'n_samples': n_samples,
        'seed': seed,
        'batches': batches,
        'PARAM_TEX_LABELS': PARAM_TEX_LABELS,
    }
    
    save_path = os.path.join(RESULTS_DIR, outpath)
    with open(save_path, 'wb') as f:
        pickle.dump(corner_data, f)
    print(f'Saved corner plot data to: {save_path}')
    return corner_data


if __name__ == '__main__':
    # Save pickle data for combined figure
    save_corner_data_to_pkl()
    # Also generate the standalone corner plot
    main(['--param-set', 'imp', '--outfile', 'corner_imp.pdf'])
