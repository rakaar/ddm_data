# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sem
from sklearn.metrics import r2_score
import figure_template as ft
from collections import defaultdict
from matplotlib.patches import Ellipse as MplEllipse
from matplotlib.ticker import FormatStrFormatter

# Helper functions for unpickling the quantile data
def _create_innermost_dict():
    return {'empirical': [], 'theoretical': []}

def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)

# --- Data Loading ---
def load_data():
    """Loads all necessary data from pickle files."""
    with open('norm_psy_fig2_data.pkl', 'rb') as f:
        psy_data = pickle.load(f)
    with open('norm_quant_fig2_data.pkl', 'rb') as f:
        quant_data = pickle.load(f)
    with open('../fit_each_condn/norm_gamma_fig2_data.pkl', 'rb') as f:
        gamma_data = pickle.load(f)
    with open('norm_slopes_fig2_data.pkl', 'rb') as f:
        slopes_data = pickle.load(f)
    with open('norm_model_fit_params_corner_plot_all_animals.pkl', 'rb') as f:
        corner_data = pickle.load(f)
    return psy_data, quant_data, gamma_data, slopes_data, corner_data

# --- Figure 4 Plotting Functions (from fig4_all_using_template.py) ---
def plot_psychometric(ax, data):
    """Plots the psychometric curves with both empirical and theoretical fits."""
    empirical_agg = data['empirical_agg']
    theory_agg = data['theory_agg']
    ILD_arr = data['ILD_arr']
    
    colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
    
    for abl in [20, 40, 60]:
        emp = empirical_agg[abl]
        theo = theory_agg[abl]
        emp_mean = np.nanmean(emp, axis=0)
        theo_mean = np.nanmean(theo, axis=0)
        ilds = np.array(ILD_arr)
        theo_mean = np.array(theo_mean)
        
        # Empirical data points with error bars
        n_emp = np.sum(~np.isnan(emp), axis=0)
        emp_sem = np.nanstd(emp, axis=0) / np.sqrt(np.maximum(n_emp - 1, 1))
        ax.errorbar(ilds, emp_mean, yerr=emp_sem, fmt='o', color=colors[abl], 
                   capsize=0, label=f'Data ABL={abl}', markersize=8)
        
        # Logistic fit to theory: solid line
        valid_idx = ~np.isnan(theo_mean)
        if np.sum(valid_idx) >= 4:
            try:
                def sigmoid(x, upper, lower, x0, k):
                    return lower + (upper - lower) / (1 + np.exp(-k*(x-x0)))
                p0 = [1.0, 0.0, 0.0, 1.0]  # upper, lower, x0, k
                bounds = ([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf])
                popt, _ = curve_fit(sigmoid, ilds[valid_idx], theo_mean[valid_idx], p0=p0, bounds=bounds)
                ilds_smooth = np.linspace(min(ilds), max(ilds), 200)
                fit_curve = sigmoid(ilds_smooth, *popt)
                ax.plot(ilds_smooth, fit_curve, linestyle='-', color=colors[abl], 
                       label=f'Theory fit ABL={abl}')
            except Exception as e:
                print(f"Could not fit logistic for ABL={abl}: {e}")
    
    ax.set_xlabel('ILD (dB)', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel('P(choice = right)', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.axvline(0, alpha=0.5, color='grey', linestyle='--')
    ax.axhline(0.5, alpha=0.5, color='grey', linestyle='--')
    ax.set_ylim(-0.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_box_aspect(1)  # Square panel

def plot_quantiles(ax, data):
    """Plots the RT quantiles with both empirical and theoretical curves."""
    plot_data = data['plot_data']
    continuous_plot_data = data.get('continuous_plot_data', None)
    continuous_abs_ild = data.get('continuous_abs_ild', None)
    QUANTILES_TO_PLOT = data['QUANTILES_TO_PLOT']
    abs_ild_sorted = data['abs_ild_sorted']
    ABL_arr = data['ABL_arr']
    
    for q_idx, q in enumerate(QUANTILES_TO_PLOT):
        emp_means, emp_sems = [], []
        theo_means, theo_sems = [], []
        theo_abs_ild_plot = []  # x-axis for theory

        # --- Aggregate empirical (discrete ILD) ---
        for abs_ild in abs_ild_sorted:
            all_abl_emp_quantiles = np.concatenate([
                np.array(plot_data[abl][abs_ild]['empirical'])[:, q_idx] for abl in ABL_arr
            ])
            emp_means.append(np.nanmean(all_abl_emp_quantiles))
            emp_sems.append(sem(all_abl_emp_quantiles, nan_policy='omit'))

        # --- Aggregate theoretical (continuous ILD if available) ---
        if continuous_plot_data is not None and continuous_abs_ild is not None:
            for abs_ild in continuous_abs_ild:
                all_abl_theo_q = []
                for abl in ABL_arr:
                    if len(continuous_plot_data[abl][abs_ild]['theoretical']) > 0:
                        all_abl_theo_q.extend(np.array(continuous_plot_data[abl][abs_ild]['theoretical'])[:, q_idx])
                if len(all_abl_theo_q) > 0:
                    theo_abs_ild_plot.append(abs_ild)
                    theo_means.append(np.nanmean(all_abl_theo_q))
                    theo_sems.append(sem(all_abl_theo_q, nan_policy='omit'))
        else:
            # Fallback to discrete theoretical (old behaviour)
            for abs_ild in abs_ild_sorted:
                all_abl_theo_quantiles = np.concatenate([
                    np.array(plot_data[abl][abs_ild]['theoretical'])[:, q_idx] for abl in ABL_arr
                ])
                theo_abs_ild_plot.append(abs_ild)
                theo_means.append(np.nanmean(all_abl_theo_quantiles))
                theo_sems.append(sem(all_abl_theo_quantiles, nan_policy='omit'))

        # Plot empirical with error bars (discrete points)
        ax.errorbar(abs_ild_sorted, emp_means, yerr=emp_sems, fmt='o', color='black',
                    markersize=8, capsize=0, label=f'Data q={q:.2f}' if q_idx == 0 else "_nolegend_")

        # Plot theoretical continuous curve + SEM shading
        if len(theo_abs_ild_plot) > 0:
            ax.plot(theo_abs_ild_plot, theo_means, '-', color='tab:red', linewidth=1.5,
                    label=f'Theory q={q:.2f}' if q_idx == 0 else "_nolegend_")
            ax.fill_between(theo_abs_ild_plot,
                             np.array(theo_means) - np.array(theo_sems),
                             np.array(theo_means) + np.array(theo_sems),
                             color='tab:red', alpha=0.2, linewidth=0)

    ax.set_xlabel('|ILD| (dB)', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel('RT Quantile (s)', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xscale('log', base=2)
    ax.set_xticks(abs_ild_sorted)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis='both', which='major', labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_box_aspect(1)  # Square panel

def plot_gamma(ax, data):
    """Plots the gamma curves with condition fit and norm model."""
    all_ABL = data['all_ABL']
    gamma_cond_by_cond_fit_all_animals = data['gamma_cond_by_cond_fit_all_animals']
    all_ILD_sorted = data['all_ILD_sorted']
    batch_animal_pairs = data['batch_animal_pairs']
    ILD_pts = data['ILD_pts']
    gamma_norm_model_fit_theoretical_all_animals = data['gamma_norm_model_fit_theoretical_all_animals']

    # Plot condition by condition fit gamma
    for ABL in all_ABL:
        # Calculate mean and standard error of mean for condition fit
        mean_gamma = np.nanmean(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
        sem_gamma = np.nanstd(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
        
        # Plot condition fit as scatter points with error bars
        ax.errorbar(all_ILD_sorted, mean_gamma, yerr=sem_gamma, fmt='o', 
                   color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                   label=f'ABL={ABL} (cond fit)', capsize=0, markersize=8)

    # Plot theoretical norm model gamma
    for ABL in all_ABL:
        # Get gamma values for this ABL
        gamma_for_ABL = np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan)
        for animal_idx in range(len(batch_animal_pairs)):
            gamma_for_ABL[animal_idx] = gamma_norm_model_fit_theoretical_all_animals[animal_idx]
        
        mean_gamma = np.nanmean(gamma_for_ABL, axis=0)
        sem_gamma = np.nanstd(gamma_for_ABL, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_for_ABL), axis=0))
        
        ax.plot(ILD_pts, mean_gamma, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                label=f'ABL={ABL} (norm)', linestyle='--')
        ax.fill_between(ILD_pts, mean_gamma - sem_gamma, mean_gamma + sem_gamma, 
                        color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', alpha=0.2)

    ax.set_xlabel('ILD', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel('Gamma', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([-2, 0, 2])
    ax.set_ylim(-3, 3)
    ax.set_box_aspect(1)  # Square panel

def plot_slopes(ax, data):
    """Plots the slopes scatter plot comparing data vs model."""
    data_means = data['data_means']
    norm_means = data['norm_means']
    
    ax.scatter(data_means, norm_means, marker='o', s=64, facecolors='w', edgecolors='k', linewidths=1.5)
    ax.set_xlabel('Data', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel('Model', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.1, 0.9)
    ax.tick_params(axis='both', labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot([0.1, 0.9], [0.1, 0.9], color='grey', alpha=0.5, linestyle='--', linewidth=2, zorder=0)
    ax.set_box_aspect(1)  # Square panel
    
    # Calculate and display R²
    r2_norm = r2_score(data_means, norm_means)
    print(f"R² for norm model: {r2_norm:.2f}")

# --- Corner Plot Helper Functions (from corner_cum_animal_params_for_paper_norm.py) ---
def _axis_limits(values, pad_frac=0.05):
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

def plot_corner(axes, corner_data, tick_labelsize=15, label_fontsize=None):
    """Plot the corner plot on a pre-allocated grid of axes.
    
    axes: 2D array of matplotlib axes (n x n)
    corner_data: dict loaded from pickle
    """
    # Use same font sizes as fig4 plots (from figure_template)
    if tick_labelsize is None:
        tick_labelsize = ft.STYLE.TICK_FONTSIZE
    if label_fontsize is None:
        label_fontsize = ft.STYLE.LABEL_FONTSIZE
    
    params = corner_data['params']
    means = corner_data['values_flat']
    means_overlay = corner_data['means_overlay']
    grouped = corner_data['grouped']
    axis_ranges = corner_data['axis_ranges']
    PARAM_TEX_LABELS = corner_data['PARAM_TEX_LABELS']
    ellipse_quantile = corner_data['ellipse_quantile']
    
    n = len(params)
    
    # Precompute limits per param
    lims = {}
    for p in params:
        if axis_ranges is not None and p in axis_ranges:
            lims[p] = tuple(axis_ranges[p])
        else:
            lims[p] = _axis_limits(means.get(p, []))
    
    # Settings
    ellipse_linewidth = 1.0
    ellipse_alpha = 1.0
    ellipse_edgecolor = '#2b6cb0'
    overlay_color = '#8B0000'
    overlay_point_size = 48.0
    diag_ranked = True
    diag_ranked_ci = 95.0
    
    for i, py in enumerate(params):
        for j, px in enumerate(params):
            ax = axes[i, j]
            if i == j:
                # Diagonal: ranked per-animal posterior means with CI
                if diag_ranked and grouped is not None and len(grouped) > 0:
                    stats = []
                    ci = float(diag_ranked_ci)
                    ci = min(max(ci, 0.0), 100.0)
                    lo_p = 0.5 * (100.0 - ci)
                    hi_p = 100.0 - lo_p
                    for lab, pdata in grouped.items():
                        arr = np.asarray(pdata.get(px, []), dtype=float)
                        arr = arr[np.isfinite(arr)]
                        if arr.size < 2:
                            continue
                        m = float(np.mean(arr))
                        try:
                            lo = float(np.percentile(arr, lo_p))
                            hi = float(np.percentile(arr, hi_p))
                        except Exception:
                            sd = float(np.std(arr)) if arr.size > 1 else 0.0
                            lo, hi = m - 1.96 * sd, m + 1.96 * sd
                        stats.append((lab, m, lo, hi))
                    if len(stats) > 0:
                        stats.sort(key=lambda t: t[1], reverse=True)
                        for k, (lab, m, lo, hi) in enumerate(stats):
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
                        ax.set_yticks([])
                        # Offset ticks inward to prevent overlap with adjacent plots
                        x_range = lims[px][1] - lims[px][0]
                        _xt = [lims[px][0] + 0.1*x_range, lims[px][1] - 0.1*x_range]
                        ax.set_xticks(_xt)
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                        # Add "Rat ID" ylabel to topmost diagonal box
                        if i == 0:
                            ax.set_ylabel('Rat ID', fontsize=label_fontsize, labelpad=50)
                    else:
                        ax.axis('off')
                else:
                    ax.axis('off')
            elif i > j:
                # Lower triangle: ellipses + overlay means
                # Draw covariance ellipses
                if grouped is not None and len(grouped) > 0:
                    q = float(ellipse_quantile)
                    if not (0.0 < q < 1.0):
                        q = 0.95
                    s_chi2 = -2.0 * np.log(max(1e-12, 1.0 - q))
                    for lab, pdata in grouped.items():
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
                        width = 2.0 * float(np.sqrt(s_chi2 * evals[0])) if evals.size > 0 else 0.0
                        height = 2.0 * float(np.sqrt(s_chi2 * evals[1])) if evals.size > 1 else 0.0
                        if width == 0.0 or height == 0.0:
                            continue
                        angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
                        patch = MplEllipse(
                            (m_x, m_y), width=width, height=height, angle=angle,
                            facecolor='none', edgecolor=ellipse_edgecolor, linewidth=ellipse_linewidth,
                            alpha=ellipse_alpha, zorder=4,
                        )
                        ax.add_patch(patch)
                
                # Overlay per-animal means
                if means_overlay is not None:
                    xs_ov = means_overlay.get(px, [])
                    ys_ov = means_overlay.get(py, [])
                    if len(xs_ov) > 0 and len(xs_ov) == len(ys_ov):
                        ax.scatter(
                            xs_ov,
                            ys_ov,
                            s=overlay_point_size,
                            c=overlay_color,
                            alpha=0.95,
                            edgecolor='k',
                            linewidths=0.6,
                            zorder=5,
                        )
                ax.set_xlim(lims[px])
                ax.set_ylim(lims[py])
                # Offset ticks inward to prevent overlap with adjacent plots
                x_range = lims[px][1] - lims[px][0]
                y_range = lims[py][1] - lims[py][0]
                _xt = [lims[px][0] + 0.1*x_range, lims[px][1] - 0.1*x_range]
                _yt = [lims[py][0] + 0.1*y_range, lims[py][1] - 0.1*y_range]
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
                ax.set_xlabel(PARAM_TEX_LABELS.get(px, px), fontsize=label_fontsize)
            else:
                ax.set_xticklabels([])
            if j == 0:
                if not (i == j and diag_ranked):
                    ax.set_ylabel(PARAM_TEX_LABELS.get(py, py), fontsize=label_fontsize)
            else:
                ax.set_yticklabels([])
            
            ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
            
            # Uniform subplot borders
            spine_lw = 1.0
            for side in ('left', 'bottom', 'right', 'top'):
                ax.spines[side].set_linewidth(spine_lw)
                ax.spines[side].set_visible(True)
            if j < n - 1 and i >= j + 1:
                ax.spines['right'].set_visible(False)
            if i > 0 and (i - 1) >= j:
                ax.spines['top'].set_visible(False)
            
            ax.grid(False)
            ax.set_box_aspect(1)


# Load data
psy_data, quant_data, gamma_data, slopes_data, corner_data = load_data()

# Create figure with GridSpec
# Left: 2x2 for fig4 plots
# Right: 4x4 for corner plot
n_params = len(corner_data['params'])  # should be 4

# Figure layout: left half for fig4 (2x2), right half for corner (4x4)
# Horizontal layout as requested
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(22, 10))

# Left section for fig4 (2x2)
gs_left = GridSpec(2, 2, figure=fig, left=0.05, right=0.35, top=0.95, bottom=0.08, 
                   hspace=0.15, wspace=0.9)

# Right section for corner (4x4) - aligned with fig4 grid
# Narrow the region so cells are naturally square (height=0.73*10=7.3", width should match)
gs_right = GridSpec(n_params, n_params, figure=fig, left=0.43, right=0.76, top=0.88, bottom=0.15,
                    hspace=0.08, wspace=0.08)

# --- Add fig4 plots to the left 2x2 grid ---
ax_psych = fig.add_subplot(gs_left[0, 0])
plot_psychometric(ax_psych, psy_data)

ax_quant = fig.add_subplot(gs_left[0, 1])
plot_quantiles(ax_quant, quant_data)

ax_slopes = fig.add_subplot(gs_left[1, 0])
plot_slopes(ax_slopes, slopes_data)

ax_gamma = fig.add_subplot(gs_left[1, 1])
plot_gamma(ax_gamma, gamma_data)

# --- Add corner plot to the right 4x4 grid ---
corner_axes = np.empty((n_params, n_params), dtype=object)
for i in range(n_params):
    for j in range(n_params):
        corner_axes[i, j] = fig.add_subplot(gs_right[i, j])

plot_corner(corner_axes, corner_data)

# Save
fig.savefig('figure_4_with_corner.png', dpi=300, bbox_inches='tight')
fig.savefig('figure_4_with_corner.pdf', dpi=300, bbox_inches='tight')
print("Saved figure_4_with_corner.png and figure_4_with_corner.pdf")

# %%
