# %%

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# --- Sigmoid function (must match the one in the aggregation script) ---
def sigmoid(x, upper, lower, x0, k):
    """Sigmoid function with explicit upper and lower asymptotes."""
    return lower + (upper - lower) / (1 + np.exp(-k*(x-x0)))

# --- Load data ---
with open('fig1_plot_data.pkl', 'rb') as f:
    plot_data = pickle.load(f)

# --- Extract data ---
ABLS = plot_data['ABLS']
COLORS = plot_data['COLORS']
black_plot_as = plot_data['black_plot_as']
ilds_dict = plot_data['ilds_dict']
mean_params_dict = plot_data['mean_params_dict']
mean_sigmoid_dict = plot_data['mean_sigmoid_dict']
x_smooth_dict = plot_data['x_smooth_dict']
unique_animal_identifiers = plot_data['unique_animal_identifiers']
merged_valid = plot_data['merged_valid']
all_sigmoid_curves_dict = plot_data['all_sigmoid_curves_dict']

# --- Prepare figure using GridSpec for complex layout ---
fig = plt.figure(figsize=(25, 30))
gs = GridSpec(5, 5, figure=fig, hspace=0.4, wspace=0.3)

# Create axes for the psychometric plots in the second row (index 1)
ax_psych_1 = fig.add_subplot(gs[1, 0])
ax_psych_2 = fig.add_subplot(gs[1, 1], sharey=ax_psych_1)
ax_psych_3 = fig.add_subplot(gs[1, 2], sharey=ax_psych_1)
ax_psych_4 = fig.add_subplot(gs[1, 3], sharey=ax_psych_1)

# Group axes for easy iteration in the existing plotting loop
axes = [ax_psych_1, ax_psych_2, ax_psych_3, ax_psych_4]

# Hide y-tick labels for shared axes to avoid clutter
for ax in [ax_psych_2, ax_psych_3, ax_psych_4]:
    plt.setp(ax.get_yticklabels(), visible=False)

# --- Recreate first 3 plots ---
for idx, (abl, color) in enumerate(zip(ABLS, COLORS)):
    ax = axes[idx]
    ilds = ilds_dict[abl]
    x_smooth = x_smooth_dict[abl]

    # Plot individual animal fits from the loaded data
    if abl in all_sigmoid_curves_dict:
        for y_fit in all_sigmoid_curves_dict[abl]:
            ax.plot(x_smooth, y_fit, color=color, alpha=0.3, linewidth=1)

    # Plot the main average sigmoid curve (black one)
    if black_plot_as == "mean_of_params" and abl in mean_params_dict:
        mean_params = mean_params_dict[abl]
        y_mean_sigmoid = sigmoid(x_smooth, *mean_params)
        ax.plot(x_smooth, y_mean_sigmoid, color='black', linewidth=3, label='Avg sigmoid fit')
    elif black_plot_as == "mean_of_sigmoids" and abl in mean_sigmoid_dict:
        mean_sigmoid = mean_sigmoid_dict[abl]
        ax.plot(x_smooth, mean_sigmoid, color='black', linewidth=3, label='Avg sigmoid fit')

    # Re-calculate and plot average data points and std
    all_psycho_points = []
    for batch, animal in unique_animal_identifiers:
        animal_df = merged_valid[(merged_valid['batch_name'] == batch) & (merged_valid['animal'] == animal) & (merged_valid['ABL'] == abl)]
        psycho_allowed = []
        for ild in ilds:
            sub = animal_df[animal_df['ILD'] == ild]
            psycho_allowed.append(np.mean(sub['choice'] == 1) if len(sub) > 0 else np.nan)
        all_psycho_points.append(np.array(psycho_allowed))
    
    mean_psycho = np.nanmean(all_psycho_points, axis=0)
    std_psycho = np.nanstd(all_psycho_points, axis=0)
    ax.errorbar(ilds, mean_psycho, yerr=std_psycho, fmt='o', color=color, capsize=0, markersize=8.5, label='Mean ± std')

    # --- Formatting for first 3 plots ---
    ax.set_title(f'ABL = {abl}', fontsize=18)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel('ILD', fontsize=18)
    if idx == 0:
        ax.set_ylabel('P(Right)', fontsize=18)
        ax.spines['left'].set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='y', colors='black')
    else:
        ax.spines['left'].set_color('#bbbbbb')
        ax.yaxis.label.set_color('#bbbbbb')
        ax.tick_params(axis='y', colors='#bbbbbb')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- 4th plot: All ABLs together ---
ax4 = axes[3]
for abl, color in zip(ABLS, COLORS):
    ilds = ilds_dict[abl]
    
    # Re-calculate mean and std for error bars
    all_psycho_points = []
    for batch, animal in unique_animal_identifiers:
        animal_df = merged_valid[(merged_valid['batch_name'] == batch) & (merged_valid['animal'] == animal) & (merged_valid['ABL'] == abl)]
        psycho = [np.mean(animal_df[animal_df['ILD'] == ild]['choice'] == 1) if len(animal_df[animal_df['ILD'] == ild]) > 0 else np.nan for ild in ilds]
        all_psycho_points.append(psycho)
    
    mean_psycho = np.nanmean(all_psycho_points, axis=0)
    std_psycho = np.nanstd(all_psycho_points, axis=0)
    ax4.errorbar(ilds, mean_psycho, yerr=std_psycho, fmt='o', color=color, capsize=0, markersize=8.5, label=f'ABL={abl} mean')

    # Plot sigmoid using mean parameters or mean-of-sigmoids
    if black_plot_as == "mean_of_params" and abl in mean_params_dict:
        mean_params = mean_params_dict[abl]
        x_smooth = x_smooth_dict[abl]
        y_mean_sigmoid = sigmoid(x_smooth, *mean_params)
        ax4.plot(x_smooth, y_mean_sigmoid, color=color, linewidth=2, label=f'ABL={abl} curve')
    elif black_plot_as == "mean_of_sigmoids" and abl in mean_sigmoid_dict:
        mean_sigmoid = mean_sigmoid_dict[abl]
        x_smooth = x_smooth_dict[abl]
        if mean_sigmoid is not None and x_smooth is not None:
            ax4.plot(x_smooth, mean_sigmoid, color=color, linewidth=2, label=f'ABL={abl} curve')

# --- Formatting for 4th plot ---
ax4.set_title('All ABLs', fontsize=18)
ax4.axvline(0, color='gray', linestyle='--', alpha=0.7)
ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
ax4.set_ylim(0, 1)
ax4.set_xticks([-15, -5, 5, 15])
ax4.set_yticks([0, 0.5, 1])
ax4.tick_params(axis='both', which='major', labelsize=16)
ax4.set_xlabel('ILD', fontsize=18)
ax4.spines['left'].set_color('#bbbbbb')
ax4.yaxis.label.set_color('#bbbbbb')
ax4.tick_params(axis='y', colors='#bbbbbb')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# --- Final layout adjustments for psychometric plots ---
for ax in axes:
    legend = ax.get_legend()
    if legend:
        legend.prop.set_size(16)

# --- CHRONOMETRIC PLOTS --- 

# --- Load Chronometric Data ---
try:
    with open('animal_specific_chronometric_plots/fig1_chrono_plot_data.pkl', 'rb') as f:
        chrono_data = pickle.load(f)

    # --- Extract data ---
    plot_abls = chrono_data['plot_abls']
    all_chrono_data_df = chrono_data['all_chrono_data_df']
    grand_means_data = chrono_data['grand_means_data']
    abl_colors = chrono_data['abl_colors']
    abs_ild_ticks = chrono_data['abs_ild_ticks']

    # --- Create axes for the chronometric plots in the third row (index 2) ---
    ax_chrono_1 = fig.add_subplot(gs[2, 0])
    ax_chrono_2 = fig.add_subplot(gs[2, 1], sharey=ax_chrono_1)
    ax_chrono_3 = fig.add_subplot(gs[2, 2], sharey=ax_chrono_1)
    ax_chrono_4 = fig.add_subplot(gs[2, 3], sharey=ax_chrono_1)
    chrono_axes = [ax_chrono_1, ax_chrono_2, ax_chrono_3, ax_chrono_4]

    # Hide y-tick labels for shared axes
    for ax in [ax_chrono_2, ax_chrono_3, ax_chrono_4]:
        plt.setp(ax.get_yticklabels(), visible=False)

    # --- Recreate first 3 chronometric plots ---
    for i, abl in enumerate(plot_abls):
        ax = chrono_axes[i]
        abl_df = all_chrono_data_df[all_chrono_data_df['ABL'] == abl]

        # Plot individual animal lines
        for (batch_name, animal_id), animal_df in abl_df.groupby(['batch_name', 'animal_id']):
            animal_df = animal_df.sort_values('abs_ILD')
            ax.plot(animal_df['abs_ILD'], animal_df['mean'], color='gray', alpha=0.4, linewidth=1.5)

        # Plot grand mean with SEM
        grand_mean_stats = grand_means_data[abl]
        ax.errorbar(
            x=grand_mean_stats['abs_ILD'], y=grand_mean_stats['mean'], yerr=grand_mean_stats['sem'],
            fmt='o-', color='black', linewidth=2.5, markersize=8.5, capsize=0
        )

        # Formatting
        ax.set_xlabel('|ILD| (dB)', fontsize=18)
        if i == 0:
            ax.set_ylabel('Mean RT (s)', fontsize=18)
            ax.spines['left'].set_color('black')
            ax.tick_params(axis='y', colors='black')
        else:
            ax.spines['left'].set_color('#bbbbbb')
            ax.tick_params(axis='y', colors='#bbbbbb')
        ax.set_xscale('log')
        ax.set_xticks(abs_ild_ticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim(0.1, 0.45)
        ax.set_yticks([0.1, 0.2, 0.3, 0.4])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # --- Recreate 4th chronometric plot ---
    ax4_chrono = chrono_axes[3]
    for abl, stats_data in grand_means_data.items():
        ax4_chrono.errorbar(
            x=stats_data['abs_ILD'], y=stats_data['mean'], yerr=stats_data['sem'],
            fmt='o-', color=abl_colors[abl], label=f'{int(abl)} dB',
            linewidth=2.5, markersize=8.5, capsize=0
        )
    # Formatting
    ax4_chrono.set_xlabel('|ILD| (dB)', fontsize=18)
    ax4_chrono.set_xscale('log')
    ax4_chrono.set_xticks(abs_ild_ticks)
    ax4_chrono.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax4_chrono.xaxis.set_minor_locator(plt.NullLocator())
    ax4_chrono.tick_params(axis='both', which='major', labelsize=12)
    ax4_chrono.spines['top'].set_visible(False)
    ax4_chrono.spines['right'].set_visible(False)
    ax4_chrono.spines['left'].set_color('#bbbbbb')
    ax4_chrono.tick_params(axis='y', colors='#bbbbbb')

except FileNotFoundError:
    print("\nChronometric data file not found. Skipping chronometric plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting chronometric data: {e}")


# --- RTD QUANTILE PLOTS (UNSCALED) ---
try:
    with open('fig1_quantiles_plot_data.pkl', 'rb') as f:
        quantile_data = pickle.load(f)

    # --- Extract data ---
    ABL_arr = quantile_data['ABL_arr']
    abs_ILD_arr = quantile_data['abs_ILD_arr']
    plotting_quantiles = quantile_data['plotting_quantiles']
    mean_unscaled = quantile_data['mean_unscaled']
    sem_unscaled = quantile_data['sem_unscaled']
    abl_colors_quant = quantile_data['abl_colors']

    # --- Create axes for the quantile plots in the fourth row (index 3) ---
    ax_quant_1 = fig.add_subplot(gs[3, 0])
    ax_quant_2 = fig.add_subplot(gs[3, 1], sharey=ax_quant_1)
    ax_quant_3 = fig.add_subplot(gs[3, 2], sharey=ax_quant_1)
    quantile_axes = [ax_quant_1, ax_quant_2, ax_quant_3]

    # Hide y-tick labels for shared axes
    for ax in [ax_quant_2, ax_quant_3]:
        plt.setp(ax.get_yticklabels(), visible=False)

    # --- Recreate quantile plots ---
    for col, abl in enumerate(ABL_arr):
        ax = quantile_axes[col]
        q_mat = mean_unscaled[abl]
        sem_mat = sem_unscaled[abl]
        for q_idx, q_level in enumerate(plotting_quantiles):
            ax.errorbar(abs_ILD_arr, q_mat[q_idx, :], yerr=sem_mat[q_idx, :], marker='o',
                        linestyle='-', color=abl_colors_quant[col])

        # Formatting
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xscale('log')
        ax.set_xticks(abs_ILD_arr)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim(0, 0.6)
        ax.set_yticks([0, 0.25, 0.5])
        ax.set_xlabel('|ILD| (dB)', fontsize=18)

        if col == 0:
            ax.set_ylabel('Mean RT(s)', fontsize=18)

    # --- Create and format the scaled quantile overlay plot ---
    ax_overlay = fig.add_subplot(gs[3, 3])
    mean_scaled = quantile_data['mean_scaled']
    sem_scaled = quantile_data['sem_scaled']

    for col, abl in enumerate(ABL_arr):
        q_mat = mean_scaled[abl]
        sem_mat = sem_scaled[abl]
        for q_idx, q_level in enumerate(plotting_quantiles):
            ax_overlay.errorbar(
                abs_ILD_arr,
                q_mat[q_idx, :],
                yerr=sem_mat[q_idx, :],
                marker='o',
                linestyle='-',
                color=abl_colors_quant[col]
            )

    # Formatting for the overlay plot
    ax_overlay.spines['right'].set_visible(False)
    ax_overlay.spines['top'].set_visible(False)
    ax_overlay.set_xscale('log')
    ax_overlay.set_xticks(abs_ILD_arr)
    ax_overlay.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_overlay.xaxis.set_minor_locator(plt.NullLocator())
    ax_overlay.tick_params(axis='both', which='major', labelsize=12)
    ax_overlay.set_ylim(0, 0.4)
    ax_overlay.set_yticks([0, 0.2, 0.4])
    ax_overlay.set_xlabel('|ILD| (dB)', fontsize=18)
    ax_overlay.set_ylabel('Scaled RT (s)', fontsize=18)

except FileNotFoundError:
    print("\nQuantile data file not found. Skipping quantile plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting quantile data: {e}")


# --- Q-Q PLOTS (ABL 40 Baseline) ---
try:
    with open('fig1_qq_plot_data.pkl', 'rb') as f:
        qq_data = pickle.load(f)

    # --- Extract data ---
    abs_ILD_arr_qq = qq_data['abs_ILD_arr']
    avg_quantiles = qq_data['avg_quantiles']
    sem_quantiles = qq_data['sem_quantiles']
    min_RT_cut_by_ILD = qq_data['min_RT_cut_by_ILD']
    global_min_val = qq_data['global_min_val']
    global_max_val = qq_data['global_max_val']

    # --- Create axes for the Q-Q plots in the fifth row (index 4) ---
    qq_axes = [fig.add_subplot(gs[4, i]) for i in range(5)]

    for i, abs_ild in enumerate(abs_ILD_arr_qq):
        ax = qq_axes[i]
        q_20_avg = avg_quantiles[20][:, i]
        q_40_avg = avg_quantiles[40][:, i]
        q_60_avg = avg_quantiles[60][:, i]
        lower_lim = min_RT_cut_by_ILD[abs_ild]

        # Plot ABL 20 vs 40
        valid_40_20 = ~np.isnan(q_40_avg) & ~np.isnan(q_20_avg)
        if np.any(valid_40_20):
            x_data, y_data = q_40_avg[valid_40_20], q_20_avg[valid_40_20]
            x_sem = sem_quantiles[40][:, i][valid_40_20]
            y_sem = sem_quantiles[20][:, i][valid_40_20]
            mask = (x_data >= lower_lim) & (y_data >= lower_lim)
            ax.errorbar(x_data[mask], y_data[mask], xerr=x_sem[mask], yerr=y_sem[mask], marker='o', linestyle='none', color='tab:blue', capsize=2)
            if np.sum(mask) > 1:
                m, c = np.polyfit(x_data[mask], y_data[mask], 1)
                fit_x = np.array([lower_lim, 0.5])
                ax.plot(fit_x, m*fit_x + c, color='tab:blue')

        # Plot ABL 60 vs 40
        valid_40_60 = ~np.isnan(q_40_avg) & ~np.isnan(q_60_avg)
        if np.any(valid_40_60):
            x_data, y_data = q_40_avg[valid_40_60], q_60_avg[valid_40_60]
            x_sem = sem_quantiles[40][:, i][valid_40_60]
            y_sem = sem_quantiles[60][:, i][valid_40_60]
            mask = (x_data >= lower_lim) & (y_data >= lower_lim)
            ax.errorbar(x_data[mask], y_data[mask], xerr=x_sem[mask], yerr=y_sem[mask], marker='o', linestyle='none', color='tab:green', capsize=2)
            if np.sum(mask) > 1:
                m, c = np.polyfit(x_data[mask], y_data[mask], 1)
                fit_x = np.array([lower_lim, 0.5])
                ax.plot(fit_x, m*fit_x + c, color='tab:green')

        # Formatting
        ax.plot([global_min_val, global_max_val], [global_min_val, global_max_val], color='k', linestyle='--', alpha=0.7, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        upper_lim = 0.5
        ax.set_xlim(lower_lim, upper_lim)
        ax.set_xticks([lower_lim, upper_lim])
        ax.set_ylim(lower_lim, upper_lim)
        ax.set_yticks([lower_lim, upper_lim])
        ax.set_xlabel('RT Quantiles (ABL 40)', fontsize=14)
        if i == 0:
            ax.set_ylabel('RT Quantiles (ABL 20/60)', fontsize=14)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        ax.grid(False)

except FileNotFoundError:
    print("\nQ-Q plot data file not found. Skipping Q-Q plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting Q-Q data: {e}")


# --- Final global figure adjustments ---
plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0, w_pad=2.0)
fig.suptitle('Figure 1', fontsize=24)
plt.savefig('fig1_from_pickle.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure saved as fig1_from_pickle.png")
