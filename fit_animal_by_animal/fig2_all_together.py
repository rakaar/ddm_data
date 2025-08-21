# %%
# Vanilla Psychometric
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from collections import defaultdict

with open('vanila_psy_fig2_data.pkl', 'rb') as f:
    plot_data = pickle.load(f)

empirical_agg = plot_data['empirical_agg']
theory_agg = plot_data['theory_agg']
ILD_arr = plot_data['ILD_arr']

colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
plt.figure(figsize=(4, 3))  # Smaller figure for publication
for abl in [20, 40, 60]:
    emp = empirical_agg[abl]
    theo = theory_agg[abl]
    emp_mean = np.nanmean(emp, axis=0)
    theo_mean = np.nanmean(theo, axis=0)
    ilds = np.array(ILD_arr)
    theo_mean = np.array(theo_mean)
    # Empirical: dotted line
    n_emp = np.sum(~np.isnan(emp), axis=0)
    print(f'emp n valid: {n_emp}')
    emp_sem = np.nanstd(emp, axis=0) / np.sqrt(np.maximum(n_emp - 1, 1))
    print(f'denominator: {np.sqrt(np.maximum(n_emp - 1, 1))}')
    plt.errorbar(ilds, emp_mean, yerr=emp_sem, fmt='o', color=colors[abl], label=f'Data ABL={abl}', capsize=0, markersize=4)
    # Logistic fit to theory: solid line
    valid_idx = ~np.isnan(theo_mean)
    if np.sum(valid_idx) >= 4:
        try:
            def sigmoid(x, upper, lower, x0, k):
                return lower + (upper - lower) / (1 + np.exp(-k*(x-x0)))
            p0 = [1.0, 0.0, 0.0, 1.0] # upper, lower, x0, k
            bounds = ([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf])
            popt, _ = curve_fit(sigmoid, ilds[valid_idx], theo_mean[valid_idx], p0=p0, bounds=bounds)
            ilds_smooth = np.linspace(min(ilds), max(ilds), 200)
            fit_curve = sigmoid(ilds_smooth, *popt)
            plt.plot(ilds_smooth, fit_curve, linestyle='-', color=colors[abl], label=f'Theory fit ABL={abl}', lw=0.5)
        except Exception as e:
            print(f"Could not fit logistic for ABL={abl}: {e}")
    else:
        print(f"Not enough valid theory points for ABL={abl} to fit.")
plt.xlabel('ILD (dB)', fontsize=16)
plt.ylabel('P(choice = right)', fontsize=16)
# plt.title('Average Psychometric Curves (All ABLs)')
plt.xticks([-15,-5,5,15], fontsize=14)
plt.yticks([0, 0.5, 1], fontsize=14)
plt.axvline(0, alpha=0.5, color='grey', linestyle='--')
plt.axhline(0.5, alpha=0.5, color='grey', linestyle='--')

plt.ylim(-0.05, 1.05)
# Remove top and right spines for publication style
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# %%
# Quantiles
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

# Helper functions for unpickling defaultdict-based structure
def _create_innermost_dict():
    return {'empirical': [], 'theoretical': []}

def _create_inner_defaultdict():
    return defaultdict(_create_innermost_dict)

with open('vanila_quant_fig2_data.pkl', 'rb') as f:
    quantile_plot_data = pickle.load(f)

plot_data = quantile_plot_data['plot_data']
QUANTILES_TO_PLOT = quantile_plot_data['QUANTILES_TO_PLOT']
abs_ild_sorted = quantile_plot_data['abs_ild_sorted']
ABL_arr = quantile_plot_data['ABL_arr']
MODEL_TYPE = quantile_plot_data['MODEL_TYPE']

LABEL_FONTSIZE: int = 25
TICK_FONTSIZE: int = 24

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for q_idx, q in enumerate(QUANTILES_TO_PLOT):
    emp_means, emp_sems = [], []
    theo_means, theo_sems = [], []

    for abs_ild in abs_ild_sorted:
        # Aggregate across all ABLs for this ILD and quantile
        all_abl_emp_quantiles = np.concatenate([
            np.array(plot_data[abl][abs_ild]['empirical'])[:, q_idx] for abl in ABL_arr
        ])
        all_abl_theo_quantiles = np.concatenate([
            np.array(plot_data[abl][abs_ild]['theoretical'])[:, q_idx] for abl in ABL_arr
        ])

        emp_means.append(np.nanmean(all_abl_emp_quantiles))
        emp_sems.append(sem(all_abl_emp_quantiles, nan_policy='omit'))
        
        theo_means.append(np.nanmean(all_abl_theo_quantiles))
        theo_sems.append(sem(all_abl_theo_quantiles, nan_policy='omit'))

    # Plot empirical with error bars
    ax.errorbar(abs_ild_sorted, emp_means, yerr=emp_sems, fmt='o-', color='black', markersize=4, capsize=0, label=f'Data q={q:.2f}' if q_idx == 0 else "_nolegend_")
    # Plot theoretical with error bars
    ax.errorbar(abs_ild_sorted, theo_means, yerr=theo_sems, fmt='^-', color='tab:red', markersize=4, capsize=0, label=f'Theory q={q:.2f}' if q_idx == 0 else "_nolegend_")

ax.set_xlabel('|ILD| (dB)', fontsize=LABEL_FONTSIZE)
ax.set_ylabel('RT Quantile (s)', fontsize=LABEL_FONTSIZE)
ax.set_xscale('log', base=2)
ax.set_xticks(abs_ild_sorted)
ax.set_yticks([0.1, 0.2, 0.3, 0.4])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
# ax.legend()

plt.tight_layout()
plt.savefig(f'quantile_plot_with_errorbars_{MODEL_TYPE}.png', dpi=300)
plt.show()

# %% 
# Gamma plot
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('../fit_each_condn/vanilla_gamma_fig2_data.pkl', 'rb') as f:
    gamma_plot_data = pickle.load(f)

all_ABL = gamma_plot_data['all_ABL']
gamma_cond_by_cond_fit_all_animals = gamma_plot_data['gamma_cond_by_cond_fit_all_animals']
all_ILD_sorted = gamma_plot_data['all_ILD_sorted']
batch_animal_pairs = gamma_plot_data['batch_animal_pairs']
ILD_pts = gamma_plot_data['ILD_pts']
gamma_vanilla_model_fit_theoretical_all_animals = gamma_plot_data['gamma_vanilla_model_fit_theoretical_all_animals']

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# Plot condition by condition fit gamma
for ABL in all_ABL:
    # Calculate mean and standard error of mean for condition fit
    mean_gamma = np.nanmean(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0)
    sem_gamma = np.nanstd(gamma_cond_by_cond_fit_all_animals[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_cond_by_cond_fit_all_animals[str(ABL)]), axis=0))
    
    # Plot condition fit as scatter points with error bars
    ax.errorbar(all_ILD_sorted, mean_gamma, yerr=sem_gamma, fmt='o', color=f'tab:{["blue", "orange", "green"][ABL//20-1]}', 
                  label=f'ABL={ABL} (cond fit)', capsize=0)

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

# ax.set_title('Vanilla', fontsize=24)
ax.set_xlabel('ILD', fontsize=25)
ax.set_ylabel('Gamma', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks([-15, -5, 5, 15])
ax.set_yticks([-2, 0, 2])
ax.set_ylim(-3, 3)

# ax.legend()
plt.tight_layout()
plt.savefig('gamma_cond_fit_vs_vanilla_model.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Slope psychometric
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

with open('vanila_slopes_fig2_data.pkl', 'rb') as f:
    slope_plot_data = pickle.load(f)

data_means = slope_plot_data['data_means']
vanilla_means = slope_plot_data['vanilla_means']

# --- Figure 1: Data vs Vanilla ---
fig_vanilla, ax_vanilla = plt.subplots(figsize=(4, 4))
ax_vanilla.scatter(data_means, vanilla_means, color='k', marker='X', s=60, alpha=0.7)
ax_vanilla.set_xlabel('Data', fontsize=20)
ax_vanilla.set_ylabel('Model', fontsize=20)
ax_vanilla.set_xticks([0.1, 0.5, 0.9])
ax_vanilla.set_yticks([0.1, 0.5, 0.9])
ax_vanilla.set_xlim(0.1, 0.9)
ax_vanilla.set_ylim(0.1, 0.9)
ax_vanilla.tick_params(axis='both', labelsize=18)
ax_vanilla.spines['top'].set_visible(False)
ax_vanilla.spines['right'].set_visible(False)
ax_vanilla.plot([0.1, 0.9], [0.1, 0.9], color='grey', alpha=0.5, linestyle='--', linewidth=2, zorder=0)
r2_vanilla = r2_score(data_means, vanilla_means)
# ax_vanilla.legend([f'$R^2$ = {r2_vanilla:.2f}'], loc='upper left', frameon=False, fontsize=15)
plt.show()