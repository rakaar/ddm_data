# %%

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
# Add padding when saving figures to create whitespace around the entire figure
mpl.rcParams["savefig.pad_inches"] = 0.6

# Helper to nudge a list of axes horizontally (dx in figure coordinates)

def shift_axes(ax_list, dx):
    """Shift axes in ax_list horizontally by dx (figure coordinate fraction)."""
    for ax in ax_list:
        pos = ax.get_position()
        ax.set_position([pos.x0 + dx, pos.y0, pos.width, pos.height])
# --- Plotting Configuration ---
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 14
SUPTITLE_FONTSIZE = 24
# Increase gap between axis tick labels and axis titles
plt.rcParams['axes.labelpad'] = 12

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
# Add larger margins around the entire figure to avoid elements touching the edges
fig.subplots_adjust(left=0.06, right=0.97, top=0.96, bottom=0.06)
gs = GridSpec(5, 6, figure=fig, hspace=0.3, wspace=0.0, width_ratios=[1, 1, 1, 1, 1, 1])

# Create axes for the psychometric plots in the second row (index 1)
ax_psych_1 = fig.add_subplot(gs[1, 0])
ax_psych_2 = fig.add_subplot(gs[1, 1], sharey=ax_psych_1)
ax_psych_3 = fig.add_subplot(gs[1, 2], sharey=ax_psych_1)
ax_psych_4 = fig.add_subplot(gs[1, 3], sharey=ax_psych_1)

# Group axes for easy iteration in the existing plotting loop
axes = [ax_psych_1, ax_psych_2, ax_psych_3, ax_psych_4]

# Ensure each psychometric plot is square shaped regardless of data limits
for ax in axes:
    # `set_box_aspect(1)` (Matplotlib ≥3.4) forces the axes box to be square.
    # If running on an older Matplotlib version, fall back to an equal aspect ratio
    # with the box adjustable so the axis limits remain unchanged.
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect(1)
    else:
        ax.set_aspect('equal', adjustable='box')

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
    ax.set_title(f'ABL = {abl}', fontsize=TITLE_FONTSIZE)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xticks([-15, -5, 5, 15])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.set_xlabel('ILD (dB)', fontsize=LABEL_FONTSIZE)
    if idx == 0:
        ax.set_ylabel('P(Right)', fontsize=LABEL_FONTSIZE)
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
ax4.set_title('All ABLs', fontsize=TITLE_FONTSIZE)
ax4.axvline(0, color='gray', linestyle='--', alpha=0.7)
ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
ax4.set_ylim(0, 1)
ax4.set_xticks([-15, -5, 5, 15])
ax4.set_yticks([0, 0.5, 1])
ax4.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
ax4.set_xlabel('ILD (dB)', fontsize=LABEL_FONTSIZE)
ax4.spines['left'].set_color('#bbbbbb')
ax4.yaxis.label.set_color('#bbbbbb')
ax4.tick_params(axis='y', colors='#bbbbbb')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# --- Final layout adjustments for psychometric plots ---
for ax in axes:
    legend = ax.get_legend()
    if legend:
        legend.prop.set_size(LEGEND_FONTSIZE)

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

    # Ensure each chronometric subplot is square shaped
    for ax in chrono_axes:
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect(1)
        else:
            ax.set_aspect('equal', adjustable='box')

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
        ax.set_xlabel('|ILD| (dB)', fontsize=LABEL_FONTSIZE)
        if i == 0:
            ax.set_ylabel('Mean RT (s)', fontsize=LABEL_FONTSIZE)
            ax.spines['left'].set_color('black')
            ax.tick_params(axis='y', colors='black')
        else:
            ax.spines['left'].set_color('#bbbbbb')
            ax.tick_params(axis='y', colors='#bbbbbb')
        ax.set_xscale('log')
        ax.set_xticks(abs_ild_ticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
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
    ax4_chrono.set_xlabel('|ILD| (dB)', fontsize=LABEL_FONTSIZE)
    ax4_chrono.set_xscale('log')
    ax4_chrono.set_xticks(abs_ild_ticks)
    ax4_chrono.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax4_chrono.xaxis.set_minor_locator(plt.NullLocator())
    ax4_chrono.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax4_chrono.spines['top'].set_visible(False)
    ax4_chrono.spines['right'].set_visible(False)
    ax4_chrono.spines['left'].set_color('#bbbbbb')
    ax4_chrono.tick_params(axis='y', colors='#bbbbbb')

    # --- Add summary chronometric plots (RT vs |ILD| and RT vs ABL) ---
    rt_vs_ild = chrono_data['rt_vs_ild']
    rt_vs_abl = chrono_data['rt_vs_abl']

    # --- Add summary chronometric plots (RT vs |ILD| and RT vs ABL) ---
    rt_vs_ild = chrono_data['rt_vs_ild']
    rt_vs_abl = chrono_data['rt_vs_abl']

    # Place summary axes in columns 5 and 6 so they match the size of the main chronometric panels
    ax_ild = fig.add_subplot(gs[2, 4])
    ax_abl = fig.add_subplot(gs[2, 5], sharey=ax_ild)

    # Make them square like the other chronometric axes
    for ax_sum in (ax_ild, ax_abl):
        if hasattr(ax_sum, 'set_box_aspect'):
            ax_sum.set_box_aspect(1)
        else:
            ax_sum.set_aspect('equal', adjustable='box')

    # Nudge both left a touch to minimise the col-4/5 gap
    shift_axes([ax_ild, ax_abl], dx=0.05)

    # Mean RT vs |ILD|
    ax_ild.errorbar(
        x=rt_vs_ild['abs_ILD'], y=rt_vs_ild['mean'], yerr=rt_vs_ild['sem'],
        fmt='o', color='k', capsize=0, markersize=6, linewidth=2
    )
    ax_ild.set_xlabel('|ILD|', fontsize=LABEL_FONTSIZE)
    ax_ild.set_ylabel('Mean RT (s)', fontsize=LABEL_FONTSIZE)
    ax_ild.set_xscale('log')
    ax_ild.set_xticks(abs_ild_ticks)
    ax_ild.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_ild.xaxis.set_minor_locator(plt.NullLocator())
    ax_ild.spines['top'].set_visible(False)
    ax_ild.spines['right'].set_visible(False)
    ax_ild.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

    # Mean RT vs ABL
    ax_abl.errorbar(
        x=range(len(rt_vs_abl)), y=rt_vs_abl['mean'], yerr=rt_vs_abl['sem'],
        fmt='o', linestyle='None', color='k', capsize=0, markersize=6
    )
    ax_abl.set_xticks(range(len(rt_vs_abl)))
    ax_abl.set_xticklabels(rt_vs_abl['ABL'].astype(int))
    ax_abl.set_xlabel('ABL', fontsize=LABEL_FONTSIZE)
    plt.setp(ax_abl.get_yticklabels(), visible=False)
    ax_abl.spines['top'].set_visible(False)
    ax_abl.spines['right'].set_visible(False)
    ax_abl.tick_params(axis='x', which='major', labelsize=TICK_FONTSIZE)

    # --- Final y-axis configuration as per user instruction ---
    # Configure left plot (which controls the shared y-axis)
    ax_ild.set_ylim(0.15, 0.30) # Adjusted ylim to better fit data
    ax_ild.set_yticks([0.15, 0.3])
    ax_ild.set_yticklabels(['0.15', '0.3'])
    ax_ild.tick_params(axis='y', labelleft=True)

    # Keep the right-hand plot label-free
    plt.setp(ax_abl.get_yticklabels(), visible=False)

    ax_abl.spines['top'].set_visible(False)
    ax_abl.spines['right'].set_visible(False)
    ax_abl.tick_params(axis='x', which='major', labelsize=TICK_FONTSIZE)

except FileNotFoundError:
    print("\nChronometric data file not found. Skipping chronometric plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting psychometric data: {e}")

# --- SLOPES AND HISTOGRAMS PLOTS ---
try:
    with open('fig1_slopes_hists_data.pkl', 'rb') as f:
        slope_data = pickle.load(f)

    # --- Extract data ---
    slopes = slope_data['slopes']
    ABLS = slope_data['ABLS']
    animals = slope_data['animals']
    diff_within = slope_data['diff_within']
    diff_across = slope_data['diff_across']
    bins_absdiff = slope_data['bins_absdiff']
    hist_xlim = slope_data['hist_xlim']
    hist_ylim = slope_data['hist_ylim']
    plot_colors = slope_data['plot_colors']

    # --- Create a nested GridSpec for the new layout in cell gs[1, 5] ---
    # This nested grid has 2 rows (for slopes and histograms) and 2 columns (for the two histograms)
    gs_nested = gs[1, 5].subgridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.2)

    # --- 1. Slopes scatter plot (top row of nested grid, spanning both columns) ---
    ax_slopes = fig.add_subplot(gs_nested[0, :])
    for idx, abl in enumerate(ABLS):
        color = plot_colors[idx]
        y = [slopes[abl].get(animal, np.nan) for animal in animals]
        ax_slopes.scatter(range(len(animals)), y, color=color, s=40)

    # Formatting for slopes plot
    ax_slopes.set_xticks([])
    ax_slopes.set_ylabel('Slope (k)', fontsize=LEGEND_FONTSIZE)
    ax_slopes.set_yticks([0, 2])
    ax_slopes.spines['top'].set_visible(False)
    ax_slopes.spines['right'].set_visible(False)
    ax_slopes.set_title('Slopes', fontsize=LABEL_FONTSIZE)

    # --- 2. Histograms (bottom row of nested grid, side-by-side) ---
    ax_hist1 = fig.add_subplot(gs_nested[1, 0]) # Left histogram
    ax_hist2 = fig.add_subplot(gs_nested[1, 1]) # Right histogram
    # Reduce gap between column 4 and 5 for first row by nudging nested axes left
    shift_axes([ax_slopes, ax_hist1, ax_hist2], dx=-0.05)

    # Left histogram: Within-rat differences
    ax_hist1.hist(diff_within, bins=bins_absdiff, color='grey', alpha=0.7, density=True)
    ax_hist1.set_xlabel(r'$\mu_{ABL} - \mu_{rat}$', fontsize=LABEL_FONTSIZE)

    # Right histogram: Across-rat differences
    ax_hist2.hist(diff_across, bins=bins_absdiff, color='grey', alpha=0.7, density=True)
    ax_hist2.set_xlabel(r'$\mu_{rat} - \mu_{grand}$', fontsize=LABEL_FONTSIZE)

    # --- Common formatting for both histograms ---
    hist_xticks = [-0.4,  0,  0.4]
    hist_xticklabels = ['-0.4', '0', '0.4']

    for ax_hist, title in [(ax_hist1, 'Within-animal'), (ax_hist2, 'Across-animal')]:
        ax_hist.set_title(title, fontsize=LEGEND_FONTSIZE)
        ax_hist.set_ylim(0, hist_ylim)
        ax_hist.set_xlim(-0.4, 0.4)
        ax_hist.axvline(0, color='black', linestyle=':', linewidth=1)
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        ax_hist.set_xticks(hist_xticks)
        ax_hist.set_xticklabels(hist_xticklabels)

    # --- Specific y-axis formatting ---
    ax_hist1.set_ylabel('Density', fontsize=LABEL_FONTSIZE)
    ax_hist1.set_yticks([0, hist_ylim])
    ax_hist2.set_yticks([])  # Remove y-ticks from the right plot for a cleaner look

except FileNotFoundError:
    print("\nSlope/histogram data file not found. Skipping these plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting slopes/histograms: {e}")


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

    # Ensure each quantile subplot is square shaped
    for ax in quantile_axes:
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect(1)
        else:
            ax.set_aspect('equal', adjustable='box')

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
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        ax.set_ylim(0, 0.6)
        ax.set_yticks([0, 0.25, 0.5])
        ax.set_xlabel('|ILD| (dB)', fontsize=LABEL_FONTSIZE)

        if col == 0:
            ax.set_ylabel('Mean RT(s)', fontsize=LABEL_FONTSIZE)

    # --- Create and format the scaled quantile overlay plot ---
    ax_overlay = fig.add_subplot(gs[3, 4])
    # Make overlay plot square as well for consistency
    if hasattr(ax_overlay, 'set_box_aspect'):
        ax_overlay.set_box_aspect(1)
    else:
        ax_overlay.set_aspect('equal', adjustable='box')
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
    ax_overlay.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax_overlay.set_ylim(0, 0.4)
    ax_overlay.set_yticks([0, 0.2, 0.4])
    ax_overlay.set_xlabel('|ILD| (dB)', fontsize=LABEL_FONTSIZE)
    ax_overlay.set_ylabel('Scaled RT (s)', fontsize=LABEL_FONTSIZE)

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
    # We want the five Q–Q panels to sit flush together (no spacer gap).
    # Build a nested 1×5 GridSpec inside the parent cell that spans columns 0–4.
    gs_qq = gs[4, 0:5].subgridspec(1, 5, wspace=0.3)
    qq_axes = [fig.add_subplot(gs_qq[0, i]) for i in range(5)]

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

        # --- Formatting for Q-Q plots ---
        ax.set_aspect('equal', adjustable='box')
        ax.plot([global_min_val, global_max_val], [global_min_val, global_max_val], 'k--', alpha=0.7, zorder=0) # Identity line
        ax.set_xlim(global_min_val, global_max_val)
        ax.set_ylim(global_min_val, global_max_val)
        ax.set_title(f'|ILD| = {abs_ild}', fontsize=LABEL_FONTSIZE)
        ax.set_xlabel('RT Quantiles (ABL 40)', fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Set ticks to be the same for x and y
        ticks = np.linspace(round(global_min_val, 1), round(global_max_val, 1), 3)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    # Common y-label for the first plot
    qq_axes[0].set_ylabel('RT Quantiles (ABL 20/60)', fontsize=LABEL_FONTSIZE)

    # Hide y-tick labels for other plots
    for ax in qq_axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

except FileNotFoundError:
    print("\nQ-Q plot data file not found. Skipping Q-Q plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting Q-Q data: {e}")


# --- Final global figure adjustments ---
plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0, w_pad=2.0)
fig.suptitle('Figure 1', fontsize=SUPTITLE_FONTSIZE)
plt.savefig('fig1_from_pickle.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure saved as fig1_from_pickle.png")
