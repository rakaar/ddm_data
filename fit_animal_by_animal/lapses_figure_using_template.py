# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import figure_template as ft

BASE_DIR = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'

# --- Data Loading ---
def load_data():
    """Loads all necessary data from pickle files."""
    with open(os.path.join(BASE_DIR, 'supp_lapses_distr_plot.pkl'), 'rb') as f:
        lapse_distr_data = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'lapse_rate_loglike_diff_data.pkl'), 'rb') as f:
        ll_diff_data = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'gamma_sep_by_median_lapse_rate_data.pkl'), 'rb') as f:
        gamma_cond_data = pickle.load(f)
    with open(os.path.join(os.path.dirname(BASE_DIR), 'fit_each_condn', 'norm_gamma_fig2_data.pkl'), 'rb') as f:
        norm_gamma_data = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'params_npl_npl_plus_lapse_plot_data.pkl'), 'rb') as f:
        params_data = pickle.load(f)
    return lapse_distr_data, ll_diff_data, gamma_cond_data, norm_gamma_data, params_data


# --- Plotting Functions ---

def plot_lapse_distribution(ax, data):
    """Panel 1: Lapse rate vs animal index (sorted ascending)."""
    lapse_rates = data['lapse_rates']
    median_lapse_rate = data['median_lapse_rate']

    lapse_rates_sorted = np.sort(lapse_rates)
    n_animals = len(lapse_rates_sorted)
    animal_indices = np.arange(1, n_animals + 1)

    ax.scatter(animal_indices, lapse_rates_sorted, color='k', s=50, alpha=0.7)
    ax.axhline(median_lapse_rate, color='gray', linestyle='--', linewidth=2,
               label=f'Median={median_lapse_rate:.2f}%')

    ax.set_xlabel('Animal', fontsize=ft.STYLE.TICK_FONTSIZE)
    ax.set_ylabel('Lapse Rate (%)', fontsize=ft.STYLE.TICK_FONTSIZE)
    ax.legend(fontsize=ft.STYLE.LEGEND_FONTSIZE, frameon=False, loc='best')
    ax.set_xticks([])
    ax.set_ylim(0, 25)
    ax.set_yticks([0, 25])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.10g}'))
    ax.tick_params(axis='y', labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_box_aspect(1)


def plot_ll_diff(ax, data, key):
    """Panels 2-3: Log-likelihood difference scatter plots."""
    sub = data[key]
    lapse_rate_pct = sub['lapse_rate_pct']
    loglike_diff = sub['loglike_diff']

    colors = ['green' if d > 0 else 'red' for d in loglike_diff]
    ax.scatter(lapse_rate_pct, loglike_diff, c=colors, alpha=0.7, s=45,
               edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    median_lr = float(np.median(lapse_rate_pct)) if len(lapse_rate_pct) > 0 else np.nan
    if np.isfinite(median_lr):
        ax.axvline(x=median_lr, color='black', linestyle=':', linewidth=1)

    ax.set_xlabel(sub['x_label'], fontsize=ft.STYLE.TICK_FONTSIZE)
    ax.set_xticks([0, 6])
    ax.tick_params(axis='both', labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_box_aspect(1)
    return ax


def plot_gamma_combined(ax, gamma_cond_data, norm_gamma_data):
    """Panel 4: Combined gamma — condition-fit data vs NPL theoretical."""
    all_ABL = gamma_cond_data['all_ABL']
    all_ILD_sorted = gamma_cond_data['all_ILD_sorted']
    gamma_low_lapse = gamma_cond_data['gamma_low_lapse']
    gamma_high_lapse = gamma_cond_data['gamma_high_lapse']
    gamma_norm_no_lapse = norm_gamma_data['gamma_norm_model_fit_theoretical_all_animals']
    norm_no_lapse_ild_pts = norm_gamma_data['ILD_pts']

    # Combine low+high lapse groups across all ABLs
    gamma_lapse_by_abl = []
    for ABL in all_ABL:
        low_values = gamma_low_lapse[str(ABL)]
        high_values = gamma_high_lapse[str(ABL)]
        if low_values.size == 0:
            combined_values = high_values
        elif high_values.size == 0:
            combined_values = low_values
        else:
            combined_values = np.vstack([low_values, high_values])
        gamma_lapse_by_abl.append(combined_values)

    if gamma_lapse_by_abl:
        gamma_lapse_all = np.vstack(gamma_lapse_by_abl)
        mean_gamma_lapse = np.nanmean(gamma_lapse_all, axis=0)
        sem_gamma_lapse = np.nanstd(gamma_lapse_all, axis=0) / np.sqrt(
            np.sum(~np.isnan(gamma_lapse_all), axis=0))

        ax.errorbar(all_ILD_sorted, mean_gamma_lapse, yerr=sem_gamma_lapse,
                     fmt='o', color='red', capsize=0, markersize=8)

    if gamma_norm_no_lapse is not None and len(gamma_norm_no_lapse) > 0:
        mean_gamma_no_lapse = np.nanmean(gamma_norm_no_lapse, axis=0)
        sem_gamma_no_lapse = np.nanstd(gamma_norm_no_lapse, axis=0) / np.sqrt(
            np.sum(~np.isnan(gamma_norm_no_lapse), axis=0))

        mean_gamma_no_lapse_interp = np.interp(
            all_ILD_sorted, norm_no_lapse_ild_pts, mean_gamma_no_lapse)
        sem_gamma_no_lapse_interp = np.interp(
            all_ILD_sorted, norm_no_lapse_ild_pts, sem_gamma_no_lapse)

        ax.errorbar(all_ILD_sorted, mean_gamma_no_lapse_interp,
                     yerr=sem_gamma_no_lapse_interp,
                     fmt='o', color='black', capsize=0, markersize=8)

    ax.set_xlabel('ILD', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel(r'$\Gamma$', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(axis='both', labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_box_aspect(1)


def plot_param_comparison(ax, params_data, param_name):
    """Panels 5-8: NPL vs NPL+lapse parameter comparison, ordered by lapse prob."""
    x_pos = params_data['x_pos']
    median_lapse_x = params_data['median_lapse_x']
    tick_map = params_data['tick_map']
    p = params_data['params'][param_name]

    ax.errorbar(
        x_pos, p['norm_means'],
        yerr=[p['norm_means'] - p['norm_low'], p['norm_high'] - p['norm_means']],
        fmt='o', color='black', alpha=0.7, capsize=0, label='NPL', markersize=6, linewidth=1.5,
    )
    ax.errorbar(
        x_pos, p['norm_lapse_means'],
        yerr=[p['norm_lapse_means'] - p['norm_lapse_low'],
              p['norm_lapse_high'] - p['norm_lapse_means']],
        fmt='s', color='red', alpha=0.7, capsize=0, label='NPL + lapse', markersize=6, linewidth=1.5,
    )
    ax.axvline(median_lapse_x, color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel('Rat', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel(p['label'], fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_xticks([])
    if param_name in tick_map:
        ax.set_yticks(tick_map[param_name])
    ax.tick_params(axis='both', labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_box_aspect(1)


# %%
# --- Main ---
print("Loading data from pickle files...")
lapse_distr_data, ll_diff_data, gamma_cond_data, norm_gamma_data, params_data = load_data()
print("Data loaded successfully!")

# Fig2 is figsize=(9,9) with 2x2 → each panel ~4.5in.
# New figure: 2x4 → width=18, height=9 to keep same panel size.
print("Creating figure...")
builder = ft.FigureBuilder(
    sup_title="",
    n_rows=2, n_cols=4,
    figsize=(18, 9),
    hspace=0.4, wspace=0.8,
)

# --- Top row ---
# Panel (0,0): Lapse distribution
ax1 = builder.fig.add_subplot(builder.gs[0, 0])
plot_lapse_distribution(ax1, lapse_distr_data)

# Panel (0,1): NPL - (IPL+lapses) LL diff
ax2 = builder.fig.add_subplot(builder.gs[0, 1])
plot_ll_diff(ax2, ll_diff_data, 'norm_minus_vanilla_lapse')
ax2.set_ylabel(r'$\Delta$LL (NPL $-$ IPL$_L$)', fontsize=ft.STYLE.TICK_FONTSIZE)
ax2.set_yticks([-300, 0, 300])

# Panel (0,2): (NPL+lapses) - (IPL+lapses) LL diff
ax3 = builder.fig.add_subplot(builder.gs[0, 2])
plot_ll_diff(ax3, ll_diff_data, 'norm_lapse_minus_vanilla_lapse')
ax3.set_ylabel(r'$\Delta$LL (NPL$_L$ $-$ IPL$_L$)', fontsize=ft.STYLE.TICK_FONTSIZE)
ax3.set_yticks([0, 300])

# Panel (0,3): Gamma combined
ax4 = builder.fig.add_subplot(builder.gs[0, 3])
plot_gamma_combined(ax4, gamma_cond_data, norm_gamma_data)

# --- Bottom row ---
param_order = ['rate_norm_l', 'rate_lambda', 'theta_E', 'T_0']
tick_overrides = {'rate_lambda': [1.5, 2.5]}
bottom_axes = []
for col_idx, param_name in enumerate(param_order):
    ax = builder.fig.add_subplot(builder.gs[1, col_idx])
    plot_param_comparison(ax, params_data, param_name)
    if param_name in tick_overrides:
        ax.set_yticks(tick_overrides[param_name])
    bottom_axes.append(ax)

# Shift last column left to reduce empty space on the right
ft.shift_axes([ax4, bottom_axes[-1]], dx=-0.03)

# --- Save ---
print("Finalizing figure...")
fig = builder.finish()

output_png = os.path.join(BASE_DIR, 'lapses_supp_figure_2x4.png')
output_pdf = os.path.join(BASE_DIR, 'lapses_supp_figure_2x4.pdf')
fig.savefig(output_png, dpi=300, bbox_inches='tight')
fig.savefig(output_pdf, format='pdf', bbox_inches='tight')

print(f"\nFigure saved to:")
print(f"  - {output_png}")
print(f"  - {output_pdf}")

plt.show()
print("\nDone!")

# %%
