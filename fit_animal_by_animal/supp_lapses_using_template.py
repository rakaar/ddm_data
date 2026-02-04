# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse as MplEllipse
import figure_template as ft

# --- Data Loading ---
def load_data():
    """Loads all necessary data from pickle files saved by lapses_supp_figure_save_data.py"""
    with open('supp_lapses_distr_plot.pkl', 'rb') as f:
        lapse_distr_data = pickle.load(f)
    with open('gamma_sep_by_median_lapse_rate_data.pkl', 'rb') as f:
        gamma_data = pickle.load(f)
    with open('rate_norm_l_vs_lapse_prob_data.pkl', 'rb') as f:
        scatter_data = pickle.load(f)
    return lapse_distr_data, gamma_data, scatter_data

# --- Plotting Functions ---
def plot_lapse_distribution(ax, data):
    """
    Plot lapse rate vs animal index (sorted in ascending order of lapse rate).
    Shows scatter plot with no connecting lines and median line.
    """
    lapse_rates = data['lapse_rates']
    median_lapse_rate = data['median_lapse_rate']
    
    # Sort lapse rates in ascending order
    lapse_rates_sorted = np.sort(lapse_rates)
    n_animals = len(lapse_rates_sorted)
    animal_indices = np.arange(1, n_animals + 1)
    
    # Plot as scatter
    ax.scatter(animal_indices, lapse_rates_sorted, color='k', s=50, alpha=0.7)
    
    # Add median horizontal line
    ax.axhline(median_lapse_rate, color='gray', linestyle='--', linewidth=2, label=f'Median={median_lapse_rate:.2f}%')
    
    # Labels
    ax.set_xlabel('Animal', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel('Lapse Rate (%)', fontsize=ft.STYLE.LABEL_FONTSIZE)
    
    # Add legend without box
    ax.legend(fontsize=ft.STYLE.LEGEND_FONTSIZE, frameon=False, loc='best')
    
    # Remove x-axis ticks (no need for animal names)
    ax.set_xticks([])
    ax.set_ylim(0, 25)

    ax.set_yticks([0 ,25])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.10g}'))
    # Y-axis ticks
    ax.tick_params(axis='y', labelsize=ft.STYLE.TICK_FONTSIZE)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make plot square
    ax.set_box_aspect(1)


def plot_gamma_by_median_lapse(axes, data):
    """
    Plot gamma separated by median lapse rate - 3 separate panels, one for each ABL.
    Black: animals below median lapse rate
    Red: animals above or equal to median lapse rate
    
    Parameters:
    -----------
    axes : list of 3 matplotlib axes
        One axis for each ABL (20, 40, 60)
    data : dict
        Dictionary containing gamma data
    """
    all_ABL = data['all_ABL']
    all_ILD_sorted = data['all_ILD_sorted']
    gamma_low_lapse = data['gamma_low_lapse']
    gamma_high_lapse = data['gamma_high_lapse']
    low_lapse_animals = data['low_lapse_animals']
    high_lapse_animals = data['high_lapse_animals']

    print('Gamma colors: black = low lapse, red = high lapse')
    
    for abl_idx, ABL in enumerate(all_ABL):
        ax_gamma = axes[abl_idx]
        
        # Low lapse group (below median) - BLACK
        mean_gamma_low = np.nanmean(gamma_low_lapse[str(ABL)], axis=0)
        sem_gamma_low = np.nanstd(gamma_low_lapse[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_low_lapse[str(ABL)]), axis=0))
        
        # High lapse group (at or above median) - RED
        mean_gamma_high = np.nanmean(gamma_high_lapse[str(ABL)], axis=0)
        sem_gamma_high = np.nanstd(gamma_high_lapse[str(ABL)], axis=0) / np.sqrt(np.sum(~np.isnan(gamma_high_lapse[str(ABL)]), axis=0))
        
        # Add legend labels only for first plot
        if abl_idx == 0:
            label_low = 'Low lapse rate'
            label_high = 'High lapse rate'
        else:
            label_low = None
            label_high = None
        
        ax_gamma.errorbar(all_ILD_sorted, mean_gamma_low, yerr=sem_gamma_low, fmt='o', 
                            color='k', label=label_low, capsize=0, alpha=0.7, markersize=8)
        ax_gamma.errorbar(all_ILD_sorted, mean_gamma_high, yerr=sem_gamma_high, fmt='o', 
                            color='red', label=label_high, capsize=0, alpha=0.7, markersize=8)
        
        # Add title with ABL value
        ax_gamma.set_title(f'ABL={ABL}', fontsize=ft.STYLE.TITLE_FONTSIZE)
        
        # Labels
        ax_gamma.set_xlabel('ILD', fontsize=ft.STYLE.LABEL_FONTSIZE)
        if abl_idx == 0:
            ax_gamma.set_ylabel('Gamma', fontsize=ft.STYLE.LABEL_FONTSIZE)
        
        # Set specific ticks
        ax_gamma.set_xticks([-15, -5, 5, 15])
        ax_gamma.set_yticks([-2, 0, 2])
        ax_gamma.tick_params(axis='both', labelsize=ft.STYLE.TICK_FONTSIZE)
        
        # Remove top and right spines
        ax_gamma.spines['top'].set_visible(False)
        ax_gamma.spines['right'].set_visible(False)
        
        # Make plot square
        ax_gamma.set_box_aspect(1)


def plot_rate_norm_l_vs_lapse(ax, data):
    """
    Create scatter plot with samples from each animal's posterior.
    Shows rate_norm_l vs lapse_prob with linear fit, correlation, and covariance ellipses.
    """
    all_rate_norm_l = data['all_rate_norm_l']
    all_lapse_prob_pct = data['all_lapse_prob_pct']
    ellipses = data['ellipses']
    linear_fit = data['linear_fit']
    animals_by_color = data.get('animals_by_color', False)
    animal_colors = data.get('animal_colors', None)
    ellipse_color = data.get('ellipse_color', '#2b6cb0')
    
    # Plot samples
    if animals_by_color and animal_colors is not None:
        # Plot with unique colors per animal
        all_animal_indices = data['all_animal_indices']
        n_animals = len(data['animal_data'])
        for idx in range(n_animals):
            mask = all_animal_indices == idx
            ax.scatter(all_rate_norm_l[mask], all_lapse_prob_pct[mask], 
                      alpha=0.01, s=20, c=[animal_colors[idx]], edgecolors='none')
    else:
        # Plot all samples with same color
        ax.scatter(all_rate_norm_l, all_lapse_prob_pct, alpha=0.15, s=20, 
                   c='steelblue', edgecolors='none')
    
    # Plot covariance ellipses for each animal
    for ellipse_data in ellipses:
        m_x = ellipse_data['mean_x']
        m_y = ellipse_data['mean_y']
        width = ellipse_data['width']
        height = ellipse_data['height']
        angle = ellipse_data['angle']
        
        # Select color
        if animals_by_color and animal_colors is not None:
            # Find the index for this animal
            animal_idx = None
            for idx, animal_data in enumerate(data['animal_data']):
                if (animal_data['batch'] == ellipse_data['batch'] and 
                    animal_data['animal'] == ellipse_data['animal']):
                    animal_idx = idx
                    break
            current_ellipse_color = animal_colors[animal_idx] if animal_idx is not None else ellipse_color
        else:
            current_ellipse_color = ellipse_color
        
        # Create ellipse patch
        ellipse = MplEllipse(
            (m_x, m_y), width=width, height=height, angle=angle,
            facecolor='none', edgecolor=current_ellipse_color, linewidth=1.5,
            alpha=0.8, zorder=4,
        )
        ax.add_patch(ellipse)
    
    # Plot linear fit
    x_line = linear_fit['x_line']
    y_line = linear_fit['y_line']
    ax.plot(x_line, y_line, color='gray', linestyle='--', linewidth=2, alpha=0.8)
    
    # Labels
    ax.set_xlabel(r'$\ell$', fontsize=ft.STYLE.LABEL_FONTSIZE)
    ax.set_ylabel('Lapse rate (%)', fontsize=ft.STYLE.LABEL_FONTSIZE)
    
    # Set specific ticks
    ax.set_xticks([0.8, 1.0])
    ax.set_yticks([0, 6])
    ax.tick_params(axis='both', labelsize=ft.STYLE.TICK_FONTSIZE)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make plot square
    ax.set_box_aspect(1)


# --- Main Execution ---
if __name__ == '__main__':
    # Load all data
    print("Loading data from pickle files...")
    lapse_distr_data, gamma_data, scatter_data = load_data()
    print("Data loaded successfully!")
    
    # Create figure with 1 row x 7 columns 
    # (1 lapse dist + gap + 3 gamma panels + gap + 1 scatter)
    # Groups: a (lapse dist), b (3 gammas), c (scatter)
    # Width ratios sum = 5.3, so each unit ≈ 28/5.3 ≈ 5.3 inches
    # Height ≈ 5.5 makes panels roughly square
    print("Creating figure...")
    builder = ft.FigureBuilder(
        sup_title="",
        n_rows=1,
        n_cols=7,
        width_ratios=[1, 0.15, 1, 1, 1, 0.15, 1],  # Gaps between groups a-b and b-c
        height_ratios=[1],
        hspace=0.3,
        wspace=0.3,
        # figsize=(28, 5.5)
        figsize=(17.5, 4.35)

    )
    
    # Panel a: Lapse distribution histogram
    print("Plotting panel a: Lapse distribution...")
    ax1 = builder.fig.add_subplot(builder.gs[0, 0])
    plot_lapse_distribution(ax1, lapse_distr_data)
    
    # Column 1 is a gap
    
    # Panels b: Gamma by median lapse rate (3 separate panels for each ABL)
    print("Plotting panels b: Gamma by median lapse rate (3 ABLs)...")
    ax2 = builder.fig.add_subplot(builder.gs[0, 2])
    ax3 = builder.fig.add_subplot(builder.gs[0, 3])
    ax4 = builder.fig.add_subplot(builder.gs[0, 4])
    plot_gamma_by_median_lapse([ax2, ax3, ax4], gamma_data)
    
    # Column 5 is a gap
    
    # Panel c: rate_norm_l vs lapse scatter
    print("Plotting panel c: rate_norm_l vs lapse rate...")
    ax5 = builder.fig.add_subplot(builder.gs[0, 6])
    plot_rate_norm_l_vs_lapse(ax5, scatter_data)
    
    # Finalize and save
    print("Finalizing figure...")
    builder.finish()
    
    # Save figure
    output_png = 'supp_lapses_figure_1x5.png'
    output_pdf = 'supp_lapses_figure_1x5.pdf'
    builder.fig.savefig(output_png, dpi=300, bbox_inches='tight')
    builder.fig.savefig(output_pdf, format='pdf', bbox_inches='tight')
    
    print(f"\nFigure saved to:")
    print(f"  - {output_png}")
    print(f"  - {output_pdf}")
    
    plt.show()
    print("\nDone!")

# %%
