# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sem
from sklearn.metrics import r2_score
import figure_template as ft
from collections import defaultdict

# Helper functions for unpickling the quantile data
def _create_innermost_dict():
    return {'empirical': [], 'theoretical': []}

def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)

# --- Data Loading ---
def load_data():
    """Loads all necessary data from pickle files."""
    with open('vanilla_psy_fig2_data.pkl', 'rb') as f:
        psy_data = pickle.load(f)
    with open('vanilla_quant_fig2_data.pkl', 'rb') as f:
        quant_data = pickle.load(f)
    with open('../fit_each_condn/vanilla_gamma_fig2_data.pkl', 'rb') as f:
        gamma_data = pickle.load(f)
    with open('vanilla_slopes_fig2_data.pkl', 'rb') as f:
        slopes_data = pickle.load(f)
    return psy_data, quant_data, gamma_data, slopes_data

# --- Plotting Functions ---
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

def plot_gamma(ax, data):
    """Plots the gamma curves with condition fit and vanilla model."""
    all_ABL = data['all_ABL']
    gamma_cond_by_cond_fit_all_animals = data['gamma_cond_by_cond_fit_all_animals']
    all_ILD_sorted = data['all_ILD_sorted']
    batch_animal_pairs = data['batch_animal_pairs']
    ILD_pts = data['ILD_pts']
    gamma_vanilla_model_fit_theoretical_all_animals = data['gamma_vanilla_model_fit_theoretical_all_animals']

    # Plot condition by condition fit gamma
    for ABL in all_ABL:
        # Calculate mean and standard error of mean for condition fit
        mean_gamma = np.nanmean(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
        sem_gamma = np.nanstd(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
        
        # Plot condition fit as scatter points with error bars
        ax.errorbar(all_ILD_sorted, mean_gamma, yerr=sem_gamma, fmt='o', 
                   color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                   label=f'ABL={ABL} (cond fit)', capsize=0, markersize=8)

    # Plot theoretical vanilla model gamma
    for ABL in all_ABL:
        # Get gamma values for this ABL
        gamma_for_ABL = np.full((len(batch_animal_pairs), len(ILD_pts)), np.nan)
        for animal_idx in range(len(batch_animal_pairs)):
            gamma_for_ABL[animal_idx] = gamma_vanilla_model_fit_theoretical_all_animals[animal_idx]
        
        mean_gamma = np.nanmean(gamma_for_ABL, axis=0)
        sem_gamma = np.nanstd(gamma_for_ABL, axis=0) / np.sqrt(np.sum(~np.isnan(gamma_for_ABL), axis=0))
        
        ax.plot(ILD_pts, mean_gamma, color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                label=f'ABL={ABL} (vanilla)', linestyle='--')
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

def plot_slopes(ax, data):
    """Plots the slopes scatter plot comparing data vs model."""
    data_means = data['data_means']
    vanilla_means = data['vanilla_means']
    
    ax.scatter(data_means, vanilla_means, marker='o', s=64, facecolors='w', edgecolors='k', linewidths=1.5)
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
    
    # Calculate and display R²
    r2_vanilla = r2_score(data_means, vanilla_means)
    # ax.legend([f'$R^2$ = {r2_vanilla:.2f}'], loc='upper left', frameon=False, fontsize=15)

# %%
# Load data and create figure
psy_data, quant_data, gamma_data, slopes_data = load_data()

builder = ft.FigureBuilder(
    sup_title="",
    n_rows=2, n_cols=2, 
    figsize=(9, 9),
    hspace=0.4, wspace=0.4
)

# --- Add plots to the 2x2 grid ---
ax_psych = builder.fig.add_subplot(builder.gs[0, 0])
ax_psych.set_box_aspect(1)
plot_psychometric(ax_psych, psy_data)

ax_quant = builder.fig.add_subplot(builder.gs[0, 1])
ax_quant.set_box_aspect(1)
plot_quantiles(ax_quant, quant_data)

ax_slopes = builder.fig.add_subplot(builder.gs[1, 0])
ax_slopes.set_box_aspect(1)
plot_slopes(ax_slopes, slopes_data)

ax_gamma = builder.fig.add_subplot(builder.gs[1, 1])
ax_gamma.set_box_aspect(1)
plot_gamma(ax_gamma, gamma_data)

fig = builder.finish()
fig.tight_layout()
fig.savefig('fig2_final_figure.png', dpi=300, bbox_inches='tight')
fig.savefig('fig2_final_figure.pdf', dpi=300, bbox_inches='tight')

# %%
