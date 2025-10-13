# %%

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.font_manager as fm
import os
# Add padding when saving figures to create whitespace around the entire figure
mpl.rcParams["savefig.pad_inches"] = 0.6
# Use Helvetica or its open-source equivalent font throughout the figure.
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'TeX Gyre Heros', 'Arial', 'sans-serif']

font_path = fm.findfont(mpl.font_manager.FontProperties(family=mpl.rcParams['font.sans-serif']))
print(f"The font being used is: {font_path}") 

# Create output directory for individual subfigures
OUTPUT_DIR = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/crsy_25_figs_new/fig1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper to nudge a list of axes horizontally (dx in figure coordinates)

def shift_axes(ax_list, dx=0, dy=0):
    """Shift axes in ax_list by dx and dy (figure coordinate fractions)."""
    for ax in ax_list:
        pos = ax.get_position()
        new_pos = [pos.x0 + dx, pos.y0 + dy, pos.width, pos.height]
        ax.set_position(new_pos)

def save_multiple_subfigures(axes_list, filename_prefix, extra_artists=None, expand_left=1.25, expand_right=1.25):
    """Save multiple axes as a single combined figure in multiple formats.
    
    Args:
        axes_list: List of axes to save
        filename_prefix: Base name for output files
        extra_artists: Optional list of extra artists (e.g., text labels) to include
        expand_left: Expansion factor for left side (default 1.25)
        expand_right: Expansion factor for right side (default 1.25)
    """
    # Get bounding box containing all axes
    bboxes = []
    for ax in axes_list:
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        bboxes.append(bbox)
    
    # Include extra artists if provided
    if extra_artists:
        for artist in extra_artists:
            bbox = artist.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            bboxes.append(bbox)
    
    # Calculate combined bounding box
    x0 = min(bbox.x0 for bbox in bboxes)
    y0 = min(bbox.y0 for bbox in bboxes)
    x1 = max(bbox.x1 for bbox in bboxes)
    y1 = max(bbox.y1 for bbox in bboxes)
    
    from matplotlib.transforms import Bbox
    combined_bbox = Bbox([[x0, y0], [x1, y1]])
    
    # Expand the bounding box asymmetrically to include padding
    # Calculate expansion in inches
    width = combined_bbox.width
    height = combined_bbox.height
    
    left_pad = width * (expand_left - 1) / 2
    right_pad = width * (expand_right - 1) / 2
    vert_pad = height * 0.175  # 35% total vertical expansion (17.5% each side)
    
    expanded_bbox = Bbox([
        [x0 - left_pad, y0 - vert_pad],
        [x1 + right_pad, y1 + vert_pad]
    ])
    
    # Save in all requested formats
    formats = {
        'pdf': {'format': 'pdf'},
        'eps': {'format': 'eps'},
        'svg': {'format': 'svg'},
        'png': {'dpi': 600, 'format': 'png'}
    }
    
    for ext, kwargs in formats.items():
        filepath = os.path.join(OUTPUT_DIR, f"{filename_prefix}.{ext}")
        fig.savefig(filepath, bbox_inches=expanded_bbox, facecolor='white', **kwargs)
        print(f"Saved: {filepath}")

# --- Plotting Configuration ---
TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 25
TICK_FONTSIZE = 24
LEGEND_FONTSIZE = 16
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
gs = GridSpec(
    5, 6,
    figure=fig,
    hspace=0.3,
    wspace=0.0,
    width_ratios=[1, 1, 1, 1, 1, 1],
    # Make psychometric (row 1) and chronometric (row 2) panels shorter
    height_ratios=[1, 0.5, 0.5, 0.5, 0]
)

# Create a nested GridSpec for psychometric plots (row 1) with extra column spacing
gs_psych = gs[1, 0:4].subgridspec(1, 4, wspace=0.25)
ax_psych_1 = fig.add_subplot(gs_psych[0, 0])
ax_psych_2 = fig.add_subplot(gs_psych[0, 1], sharey=ax_psych_1)
ax_psych_3 = fig.add_subplot(gs_psych[0, 2], sharey=ax_psych_1)
ax_psych_4 = fig.add_subplot(gs_psych[0, 3], sharey=ax_psych_1)

# Group axes for easy iteration in the existing plotting loop
axes = [ax_psych_1, ax_psych_2, ax_psych_3, ax_psych_4]
# Nudge the fourth psychometric panel right to align with overlay
shift_axes([ax_psych_4], dx=0.04)

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
    
    # Convert to numpy array for easier calculations
    all_psycho_points = np.array(all_psycho_points, dtype=float)
    mean_psycho = np.nanmean(all_psycho_points, axis=0)
    # Standard Error of the Mean: std / sqrt(N) where N is number of non-nan observations per ILD
    n_points = np.sum(~np.isnan(all_psycho_points), axis=0)
    sem_psycho = np.nanstd(all_psycho_points, axis=0) / np.sqrt(n_points)
    ax.errorbar(ilds, mean_psycho, yerr=sem_psycho, fmt='o', color=color, capsize=0, markersize=8.5, label='Mean ± SEM')

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
    
    # Convert to numpy array for easier calculations
    all_psycho_points = np.array(all_psycho_points, dtype=float)
    mean_psycho = np.nanmean(all_psycho_points, axis=0)
    n_points = np.sum(~np.isnan(all_psycho_points), axis=0)
    sem_psycho = np.nanstd(all_psycho_points, axis=0) / np.sqrt(n_points)
    ax4.errorbar(ilds, mean_psycho, yerr=sem_psycho, fmt='o', color=color, capsize=0, markersize=8.5, label=f'ABL={abl} mean ± SEM')

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

# --- Save psychometric figure (all 4 panels) ---
print("\n--- Saving figure 1: Psychometric (4 panels) ---")
fig.canvas.draw()  # Ensure layout is finalized before saving
save_multiple_subfigures(axes, "fig1_psychometric_4panels")

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
    # Chronometric plots (row 2) – nested grid to add space between first three columns
    gs_chrono_main = gs[2, 0:4].subgridspec(1, 4, wspace=0.25)
    ax_chrono_1 = fig.add_subplot(gs_chrono_main[0, 0])
    ax_chrono_2 = fig.add_subplot(gs_chrono_main[0, 1], sharey=ax_chrono_1)
    ax_chrono_3 = fig.add_subplot(gs_chrono_main[0, 2], sharey=ax_chrono_1)
    ax_chrono_4 = fig.add_subplot(gs_chrono_main[0, 3], sharey=ax_chrono_1)
    chrono_axes = [ax_chrono_1, ax_chrono_2, ax_chrono_3, ax_chrono_4]
    # Nudge the fourth chronometric panel right to align with overlay
    shift_axes([ax_chrono_4], dx=0.04)

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
            ax.plot(animal_df['abs_ILD'], animal_df['mean'], color=abl_colors[abl], alpha=0.4, linewidth=1.5)

        # Plot grand mean with SEM
        grand_mean_stats = grand_means_data[abl]
        # Plot colored dots, but black line and error bars
        ax.errorbar(
            x=grand_mean_stats['abs_ILD'], y=grand_mean_stats['mean'], yerr=grand_mean_stats['sem'],
            fmt='o', color=abl_colors[abl], markersize=8.5, capsize=0, linewidth=0, zorder=3
        )
        ax.plot(
            grand_mean_stats['abs_ILD'], grand_mean_stats['mean'], color='black', linewidth=2.5, zorder=2
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
        # Plot colored dots, but black line and error bars
        ax4_chrono.errorbar(
            x=stats_data['abs_ILD'], y=stats_data['mean'], yerr=stats_data['sem'],
            fmt='o-', color=abl_colors[abl], label=f'{int(abl)} dB', markersize=8.5, capsize=0, linewidth=0, zorder=3
        )
        ax4_chrono.plot(
            stats_data['abs_ILD'], stats_data['mean'], color=abl_colors[abl], linewidth=2.5, zorder=2
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

    # Stack the two summary axes vertically spanning columns 4 & 5
    gs_summary = gs[2, 5].subgridspec(2, 1, hspace=0.05)
    ax_ild = fig.add_subplot(gs_summary[0, 0])
    ax_abl = fig.add_subplot(gs_summary[1, 0])

    # Let Matplotlib decide aspect; we only keep same width by position

   

    # Mean RT vs |ILD|
    ax_ild.errorbar(
        x=rt_vs_ild['abs_ILD'], y=rt_vs_ild['mean'], yerr=rt_vs_ild['sem'],
        fmt='o', color='k', capsize=0, markersize=6, linewidth=2
    )
    ax_ild.set_xlabel('|ILD|', fontsize=LABEL_FONTSIZE, ha='right', x=1.4)
    # Move xlabel slightly closer to the axis (higher on figure)
    ax_ild.xaxis.set_label_coords(1.4, 0.1)
    # ax_ild.set_ylabel('Mean RT (s)', fontsize=LABEL_FONTSIZE)
    ax_ild.set_xscale('log')
    ax_ild.set_xticks(abs_ild_ticks)
    ax_ild.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_ild.xaxis.set_minor_locator(plt.NullLocator())
    ax_ild.spines['top'].set_visible(False)
    ax_ild.spines['right'].set_visible(False)
    ax_ild.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    # after formatting ax_ild …

    # Mean RT vs ABL (with ABL-wise coloring)
    for i, row in rt_vs_abl.iterrows():
        abl = row['ABL']
        color = abl_colors.get(abl, 'k')  # Use black as a fallback
        ax_abl.errorbar(
            x=i, y=row['mean'], yerr=row['sem'],
            fmt='o', linestyle='None', color=color, capsize=0, markersize=8.5
        )
    ax_abl.set_xticks(range(len(rt_vs_abl)))
    ax_abl.set_xticklabels(rt_vs_abl['ABL'].astype(int))
    # ax_abl.set_ylabel('Mean RT (s)', fontsize=LABEL_FONTSIZE)
    ax_abl.set_xlabel('ABL', fontsize=LABEL_FONTSIZE, ha='right', x=1.4)
    ax_abl.xaxis.set_label_coords(1.4, 0.1)

    plt.setp(ax_abl.get_yticklabels(), visible=False)
    ax_abl.spines['top'].set_visible(False)
    ax_abl.spines['right'].set_visible(False)
    ax_abl.tick_params(axis='x', which='major', labelsize=TICK_FONTSIZE)

    # --- Final y-axis configuration as per user instruction ---
    # Configure left plot (which controls the shared y-axis)
    ax_ild.set_ylim(0.15, 0.26) # Adjusted ylim to better fit data
    ax_ild.set_yticks([0.15, 0.25])
    ax_ild.set_yticklabels(['0.15', '0.25'])
    ax_ild.tick_params(axis='y', labelleft=True, length=0)

    ax_abl.set_ylim(0.15, 0.30) # Adjusted ylim to better fit data
    ax_abl.set_yticks([0.15, 0.3])
    ax_abl.set_yticklabels(['0.15', '0.3'])
    ax_abl.tick_params(axis='y', labelleft=True)
    

    # Keep the right-hand plot label-free
    # plt.setp(ax_abl.get_yticklabels(), visible=False)

    ax_abl.spines['top'].set_visible(False)
    ax_abl.spines['right'].set_visible(False)
    ax_abl.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE, length=0)

    # ----------------- Align summary chronometric axes with main chronometric row -----------------
    # Align bottom of ax_abl with bottom of ax_chrono_1
    fig.canvas.draw()
    chrono_baseline = ax_chrono_1.get_position().y0
    summary_bottom = ax_abl.get_position().y0
    dy_summary = chrono_baseline - summary_bottom
    shift_axes([ax_ild, ax_abl], dy=dy_summary)

    # Align top of ax_ild with top of ax_chrono_1
    fig.canvas.draw()
    chrono_top = ax_chrono_1.get_position().y1
    summary_top = ax_ild.get_position().y1
    dh_summary = summary_top - chrono_top
    if dh_summary > 0:
        for ax in (ax_ild, ax_abl):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, pos.width, pos.height - dh_summary])

    # Reduce gap between the two summary plots
    fig.canvas.draw()
    gap_now_summ = ax_ild.get_position().y0 - ax_abl.get_position().y1
    gap_desired_summ = 0.02  # figure fraction
    delta_summ = gap_now_summ - gap_desired_summ
    if delta_summ > 0:
        shift_axes([ax_ild], dy=-(delta_summ/2))
        shift_axes([ax_abl], dy=+(delta_summ/2))

    # Nudge down slightly to match vertical placement; keep in last column
    shift_axes([ax_ild, ax_abl], dx=0.05, dy=-0.01)

    # -----------------------------------------------------------
    # Reduce the width of the summary axes (last column) while
    # keeping their left edges fixed.  A factor of ~0.6–0.7 looks
    # visually balanced; tweak here if needed.
    # -----------------------------------------------------------
    width_factor = 0.65  # 1 → no change, <1 shrinks width
    for ax in (ax_ild, ax_abl):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * width_factor, pos.height])

    # -----------------------------------------------------------
    # Add a common y-label centred between the two summary axes.
    # -----------------------------------------------------------
    fig.canvas.draw()
    left_edge = ax_ild.get_position().x0 - 0.02
    center_y = 0.5 * (ax_ild.get_position().y1 + ax_abl.get_position().y0)
    mean_rt_ylabel = fig.text(left_edge - 0.03, center_y, 'Mean RT (s)', rotation='vertical',
             ha='center', va='center', fontsize=LABEL_FONTSIZE)
    # Keep x-labels aligned if available (helps vertical alignment only)
    # try:
    #     fig.align_xlabels(chrono_axes + [ax_abl])
    # except Exception:
    #     pass

    # --- Save chronometric figures ---
    print("\n--- Saving figure 2: Chronometric (4 panels) ---")
    fig.canvas.draw()  # Ensure layout is finalized before saving
    save_multiple_subfigures(chrono_axes, "fig2_chronometric_4panels")
    
    print("\n--- Saving figure 5: Mean RT summary (2 panels) ---")
    # Include the ylabel text artist and add extra left padding to capture it
    save_multiple_subfigures([ax_ild, ax_abl], "fig5_mean_RT_summary_2panels", 
                           extra_artists=[mean_rt_ylabel], expand_left=1.5, expand_right=1.15)

except FileNotFoundError:
    print("\nChronometric data file not found. Skipping chronometric plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting psychometric data: {e}")

# --- JND PLOTS (REPLACING SLOPES) ---
try:
    with open('jnd_analysis_data.pkl', 'rb') as f:
        jnd_data = pickle.load(f)

    # --- Extract data ---
    jnds = jnd_data['jnds']
    mean_jnd = jnd_data['mean_jnd']
    grand_mean_jnd = jnd_data['grand_mean_jnd']
    ABLS = jnd_data['ABLS']
    animals_with_mean = jnd_data['animals_with_mean']
    mean_jnds = jnd_data['mean_jnds']
    diff_within = jnd_data['diff_within']
    plot_colors = ['tab:blue', 'tab:orange', 'tab:green']

    # --- Create a nested GridSpec for two vertical plots in the same row as psychometric ---
    gs_nested = gs[1, 5].subgridspec(2, 1, hspace=-0.2)

    # --- PLOT 1: JNDs per animal (top half) ---
    gs_jnd_plot = gs_nested[0, 0].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax1_main = fig.add_subplot(gs_jnd_plot[0, 0])
    ax1_hist = fig.add_subplot(gs_jnd_plot[0, 1], sharey=ax1_main)

    sorted_animal_indices = np.argsort(mean_jnds)
    sorted_animals = [animals_with_mean[i] for i in sorted_animal_indices]

    for i, animal_id in enumerate(sorted_animals):
        ax1_main.plot(i, mean_jnd[animal_id], 'k_', markersize=6, mew=1.5)
        for j, abl in enumerate(ABLS):
            if animal_id in jnds[abl]:
                ax1_main.plot(i, jnds[abl][animal_id], 'o', color=plot_colors[j], markersize=4, alpha=0.5, linewidth=2)
    
    ax1_main.axhline(grand_mean_jnd, color='k', linestyle=':', linewidth=1)
    print(f'grand_mean_jnd: {grand_mean_jnd}')
    ax1_main.set_xticks([])
    ax1_main.set_ylabel('JND', fontsize=LABEL_FONTSIZE)
    # ax1_main.set_title('JNDs per Animal', fontsize=LABEL_FONTSIZE)
    ax1_main.spines['top'].set_visible(False)
    ax1_main.spines['right'].set_visible(False)
    ax1_main.spines['bottom'].set_visible(False)
    ax1_main.tick_params(axis='y', labelsize=TICK_FONTSIZE, length=0)
    ax1_main.set_ylim(1, 5)
    ax1_main.set_yticks([1, 5])

    # Replace histogram with a vertical bar spanning ±1 SD (length = 2 SD)
    mu_mean = np.mean(mean_jnds)
    sd_mean = np.std(mean_jnds)
    # draw bar at x=0.5 (arbitrary small width panel)
    x_bar = 0.05  # position of 2-SD bar within the tiny hist panel (0=left,1=right)
    ax1_hist.plot([x_bar, x_bar], [mu_mean - sd_mean, mu_mean + sd_mean],
                  color='grey', linewidth=3, solid_capstyle='butt')
    ax1_hist.set_xlim(0, 1)
    ax1_hist.axis('off')

    # --- PLOT 2: Within-animal JND variability (bottom half) ---
    gs_var_plot = gs_nested[1, 0].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax2_main = fig.add_subplot(gs_var_plot[0, 0])
    ax2_hist = fig.add_subplot(gs_var_plot[0, 1], sharey=ax2_main)

    for i, animal_id in enumerate(sorted_animals):
        jnd0 = mean_jnd[animal_id]
        for j, abl in enumerate(ABLS):
            if animal_id in jnds[abl]:
                diff = jnds[abl][animal_id] - jnd0
                ax2_main.plot(i, diff, 'o', color=plot_colors[j], markersize=5, alpha=0.5)

    ax2_main.axhline(0, color='k', linestyle='-', linewidth=1)
    ax2_main.set_xticks([])
    ax2_main.set_ylabel(r'J$_{\text{ABL}}$ - J$_{\mu}$', fontsize=LABEL_FONTSIZE)
    # ax2_main.set_title('Within-Animal Variability', fontsize=LABEL_FONTSIZE)
    ax2_main.spines['top'].set_visible(False)
    ax2_main.spines['right'].set_visible(False)
    ax2_main.spines['bottom'].set_visible(False)
    ax2_main.tick_params(axis='y', labelsize=TICK_FONTSIZE, length=0)
    ax2_main.set_ylim(-2, 2)
    ax2_main.set_yticks([-2, 0, 2])
    ax2_main.set_yticklabels(['-2', '0', '2'])

    # Replace histogram with 2-SD bar centred at mean difference (which is ~0)
    mu_diff = np.mean(diff_within)
    sd_diff = np.std(diff_within)
    # ax2_hist.plot([0.5, 0.5], [mu_diff - sd_diff, mu_diff + sd_diff],
    #               color='green', linewidth=3, solid_capstyle='butt')
    ax2_hist.plot([x_bar, x_bar], [mu_diff - sd_diff, mu_diff + sd_diff],
                  color='grey', linewidth=3, solid_capstyle='butt')
    ax2_hist.set_xlim(0, 1)
    ax2_hist.axis('off')

    # --- Align the bottom JND axis baseline with the psychometric ILD x-axis automatically ---
    # First make sure layout is computed so that get_position() returns final coordinates
    fig.canvas.draw()
    # Baseline (bottom edge) of the psychometric panels – use the first one as reference
    psycho_baseline = ax_psych_1.get_position().y0
    # Current baseline of the bottom JND axis
    jnd_baseline = ax2_main.get_position().y0
    dy_align = psycho_baseline - jnd_baseline
    # Keep the original slight horizontal nudge for aesthetics while vertically aligning
    shift_axes([ax1_main, ax1_hist, ax2_main, ax2_hist], dx=0.05, dy=dy_align)

    # ----------------- Top-edge alignment -----------------
    # Re-draw to ensure new positions are registered
    fig.canvas.draw()
    psycho_top = ax_psych_1.get_position().y1
    jnd_top = ax1_main.get_position().y1
    dh = jnd_top - psycho_top
    if dh > 0:
        # Reduce heights of both JND sub-plots and their histograms by dh
        for ax in (ax1_main, ax1_hist, ax2_main, ax2_hist):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, pos.width, pos.height - dh])

    # ----------------- Reduce inter-plot gap -----------------
    fig.canvas.draw()
    # Gap between bottom edge of top axis and top edge of bottom axis
    gap_now = ax1_main.get_position().y0 - ax2_main.get_position().y1
    gap_desired = 0.05/2  # figure fraction
    delta_gap = gap_now - gap_desired
    if delta_gap > 0:
        # move top down and bottom up equally
        shift_axes([ax1_main, ax1_hist], dy=-(delta_gap / 2))
        shift_axes([ax2_main, ax2_hist], dy=+(delta_gap / 2))

    # --- Save JND figure (2 plots) ---
    print("\n--- Saving figure 4: JND (2 panels) ---")
    fig.canvas.draw()  # Ensure layout is finalized before saving
    # Add extra left padding to capture y-axis labels
    save_multiple_subfigures([ax1_main, ax1_hist, ax2_main, ax2_hist], "fig4_JND_2panels",
                           expand_left=1.5, expand_right=1.35)

except FileNotFoundError:
    print("\nJND data file ('jnd_analysis_data.pkl') not found. Skipping these plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting JNDs/histograms: {e}")


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
    # Use a nested 1×4 sub-GridSpec to match psychometric/chronometric sizing
    gs_quant = gs[3, 0:4].subgridspec(1, 4, wspace=0.25)
    ax_quant_1 = fig.add_subplot(gs_quant[0, 0])
    ax_quant_2 = fig.add_subplot(gs_quant[0, 1], sharey=ax_quant_1)
    ax_quant_3 = fig.add_subplot(gs_quant[0, 2], sharey=ax_quant_1)
    quantile_axes = [ax_quant_1, ax_quant_2, ax_quant_3]
    # No horizontal nudge; keep alignment with psychometric and chronometric rows
    # The 4th slot in gs_quant will hold the scaled overlay later

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
            ax.set_ylabel('RT(s)', fontsize=LABEL_FONTSIZE)

    # --- Create and format the scaled quantile overlay plot ---
    ax_overlay = fig.add_subplot(gs_quant[0, 3])
    # Shift overlay plot right for extra gap
    shift_axes([ax_overlay], dx=0.04)
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

    # --- Save quantile figure (4 panels) ---
    print("\n--- Saving figure 3: Quantile (4 panels) ---")
    fig.canvas.draw()  # Ensure layout is finalized before saving
    save_multiple_subfigures(quantile_axes + [ax_overlay], "fig3_quantile_4panels")

except FileNotFoundError:
    print("\nQuantile data file not found. Skipping quantile plots.")
except Exception as e:
    print(f"\nAn error occurred while plotting quantile data: {e}")

######################################################
###### NOTE: NOT IN THE FIGURE ANY MORE ##############
######################################################

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
        ax.set_xticks([0.08, 0.5])
        ax.set_yticks([0.06, 0.5])

    # Common y-label for the first plot
    qq_axes[0].set_ylabel('RT Quantiles (ABL 20/60)', fontsize=LABEL_FONTSIZE)

    # Hide y-tick labels for other plots
    for ax in qq_axes:
        ax.set_visible(False)

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
# Save as PNG (high resolution) and PDF (vector format)
plt.savefig('fig1_from_pickle.png', dpi=300, bbox_inches='tight')
plt.savefig('fig1_from_pickle.pdf', bbox_inches='tight', format='pdf')
plt.show()

print("\nFigure saved as fig1_from_pickle.png and fig1_from_pickle.pdf")
print(f"\n=== All 5 subfigures saved to: {OUTPUT_DIR} ===")
print("1. fig1_psychometric_4panels - 4 psychometric plots")
print("2. fig2_chronometric_4panels - 4 chronometric plots") 
print("3. fig3_quantile_4panels - 4 quantile plots")
print("4. fig4_JND_2panels - 2 JND plots")
print("5. fig5_mean_RT_summary_2panels - 2 mean RT summary plots")
print("\nEach figure saved in 4 formats: .pdf, .eps, .svg, and .png (600 dpi)")
