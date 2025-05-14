# %%
import pandas as pd


# %%
merged_data = pd.read_csv('batch_csvs/merged_batches.csv')
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

# %%
print(merged_valid['ABL'].unique())
print(merged_valid['ILD'].unique())

# %%
# add abs_ILD column
merged_valid['abs_ILD'] = merged_valid['ILD'].abs()

# %%
print(merged_valid['abs_ILD'].unique())

# %%
check_ILD_10 = merged_valid[merged_valid['abs_ILD'] == 10]
print(check_ILD_10['ABL'].unique())
print(f'len(check_ILD_10): {len(check_ILD_10)}')

check_ILD_6 = merged_valid[merged_valid['abs_ILD'] == 6]
print(check_ILD_6['ABL'].unique())
print(f'len(check_ILD_6): {len(check_ILD_6)}')

check_ABL_50 = merged_valid[merged_valid['ABL'] == 50]
print(check_ABL_50['abs_ILD'].unique())
print(f'len(check_ABL_50): {len(check_ABL_50)}')


# abs ILD 10,6 are very low, just remove them
merged_valid = merged_valid[merged_valid['abs_ILD'] != 10]
merged_valid = merged_valid[merged_valid['abs_ILD'] != 6]

# ABL 50 is comparitively low, just remove it
# 20,40,60 are the standard ABLs.
merged_valid = merged_valid[merged_valid['ABL'].isin([20,40,60])]

# %%
########################################
## TEMP: remove LED1 and LED2 batches ##
#################   #######################
merged_valid = merged_valid[(merged_valid['batch_name'] != 'LED1') & (merged_valid['batch_name'] != 'LED2')]

# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# Q-Q plot parameters
pdf_path = "qq_percentile_per_absILD_all_batches.pdf"
percentiles = np.arange(5, 100, 10)

with PdfPages(pdf_path) as pdf:
    # Get all unique abs_ILDs across the entire dataset
    abs_ILDs = np.sort(merged_valid['abs_ILD'].unique())
    n_abs_ILDs = len(abs_ILDs)
    
    if n_abs_ILDs == 0:
        print("No valid abs_ILD values found in the dataset.")
    else:
        # Create one figure with subplots for each abs_ILD
        fig, axes = plt.subplots(n_abs_ILDs, 1, figsize=(6, 5*n_abs_ILDs), squeeze=False)
        fig.suptitle(f'QQ Plots by abs(ILD)', fontsize=16)
        
        # Collect all diffs for y-axis limits
        all_diffs = []
        
        for j, abs_ILD in enumerate(abs_ILDs):
            ax = axes[j, 0]
            # Get all data for this abs_ILD, regardless of batch or animal
            abs_ILD_df = merged_valid[merged_valid['abs_ILD'] == abs_ILD]
            
            if abs_ILD_df.empty:
                ax.set_visible(False)
                continue
                
            # Filter for valid reaction times and success values
            RTwrtStim_pos = abs_ILD_df[(abs_ILD_df['RTwrtStim'] > 0) & (abs_ILD_df['success'].isin([1,-1]))]
            
            if RTwrtStim_pos.empty:
                ax.set_visible(False)
                continue
                
            ABLs = np.sort(RTwrtStim_pos['ABL'].unique())
            
            if len(ABLs) == 0:
                ax.set_visible(False)
                continue
            
            # Compute percentiles for each ABL
            q_dict = {}
            for abl in ABLs:
                q_dict[abl] = np.percentile(RTwrtStim_pos[RTwrtStim_pos['ABL'] == abl]['RTwrtStim'], percentiles)
                
            abl_highest = ABLs.max()
            Q_highest = q_dict[abl_highest]
            
            # Calculate diffs for y-axis limits
            for abl in ABLs:
                if abl == abl_highest:
                    continue
                diff = q_dict[abl] - Q_highest
                all_diffs.extend(diff)
                
                # Plot for each ABL
                n_rows = len(RTwrtStim_pos[RTwrtStim_pos['ABL'] == abl])
                ax.plot(Q_highest, diff, marker='o', label=f'ABL {abl} (N={n_rows})')
                
            ax.axhline(0, color='k', linestyle='--', linewidth=1)
            ax.set_xlabel(f'Percentiles of RTwrtStim (ABL={abl_highest})')
            ax.set_ylabel('Q_ABL - Q_highest_ABL')
            # Count total number of rows for this abs_ILD
            total_rows = len(RTwrtStim_pos)
            ax.set_title(f'abs(ILD): {abs_ILD} (N={total_rows})')
            ax.legend(title='ABL')
            
            # Set fixed x-axis limits as requested
            ax.set_xlim(0, 0.4)
        
        # Set consistent y-axis limits across all plots
        if all_diffs:
            global_y_min = min(all_diffs)
            global_y_max = max(all_diffs)
            
            # Add some padding to y-axis
            y_padding = (global_y_max - global_y_min) * 0.1
            global_y_min -= y_padding
            global_y_max += y_padding
            
            # Apply y-axis limits to all subplots
            for j, abs_ILD in enumerate(abs_ILDs):
                if axes[j, 0].get_visible():
                    axes[j, 0].set_ylim(global_y_min, global_y_max)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

print(f"PDF saved to {pdf_path}")

# %%
# for each abs_ILD, calculate the number of rows , and also see for each ABL
for abs_ILD in abs_ILDs:
    abs_ILD_df = merged_valid[merged_valid['abs_ILD'] == abs_ILD]
    print(f'abs(ILD): {abs_ILD} (N={len(abs_ILD_df)})')
    for abl in np.sort(abs_ILD_df['ABL'].unique()):
        abl_df = abs_ILD_df[abs_ILD_df['ABL'] == abl]
        print(f'ABL: {abl} (N={len(abl_df)})')

# %%
# fit straight lines to ABL vs highest ABL to obtain scaling parameter and R²
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
# Q-Q plot parameters
pdf_path = "line_fit_qq_percentile_per_absILD_all_batches.pdf"
percentiles = np.arange(5, 100, 10)
# Minimum reaction time in seconds
MIN_RT = 0.1  # 100 ms

# Dictionary to store fit parameters for each abs_ILD and ABL
fit_params = {}

with PdfPages(pdf_path) as pdf:
    # Get all unique abs_ILDs across the entire dataset
    abs_ILDs = np.sort(merged_valid['abs_ILD'].unique())
    n_abs_ILDs = len(abs_ILDs)
    
    if n_abs_ILDs == 0:
        print("No valid abs_ILD values found in the dataset.")
    else:
        # Create one figure with subplots for each abs_ILD
        fig, axes = plt.subplots(n_abs_ILDs, 1, figsize=(6, 5*n_abs_ILDs), squeeze=False)
        fig.suptitle(f'QQ Plots by abs(ILD)', fontsize=16)
        
        # Collect all y values for axis limits
        all_y_values = []
        
        for j, abs_ILD in enumerate(abs_ILDs):
            ax = axes[j, 0]
            # Get all data for this abs_ILD, regardless of batch or animal
            abs_ILD_df = merged_valid[merged_valid['abs_ILD'] == abs_ILD]
            
            if abs_ILD_df.empty:
                ax.set_visible(False)
                continue
                
            # Filter for valid reaction times and success values
            RTwrtStim_pos = abs_ILD_df[(abs_ILD_df['RTwrtStim'] > MIN_RT) & (abs_ILD_df['success'].isin([1,-1]))]
            
            if RTwrtStim_pos.empty:
                ax.set_visible(False)
                continue
                
            ABLs = np.sort(RTwrtStim_pos['ABL'].unique())
            
            if len(ABLs) == 0:
                ax.set_visible(False)
                continue
            
            # Compute percentiles for each ABL
            q_dict = {}
            for abl in ABLs:
                q_dict[abl] = np.percentile(RTwrtStim_pos[RTwrtStim_pos['ABL'] == abl]['RTwrtStim'], percentiles)
                
            abl_highest = ABLs.max()
            Q_highest = q_dict[abl_highest]
            
            # For each ABL, plot Q_ABL vs Q_highest and fit a straight line
            # Initialize dictionary for this abs_ILD if it doesn't exist
            if abs_ILD not in fit_params:
                fit_params[abs_ILD] = {}
                
            for abl in ABLs:
                if abl == abl_highest:
                    continue
                    
                Q_abl = q_dict[abl]
                all_y_values.extend(Q_abl)
                
                # Number of data points for this ABL
                n_rows = len(RTwrtStim_pos[RTwrtStim_pos['ABL'] == abl])
                
                # Plot Q_ABL vs Q_highest
                ax.scatter(Q_highest, Q_abl, marker='o', label=f'ABL {abl} (N={n_rows})')
                
                # Fit a straight line using weighted least squares
                # Reshape for sklearn - no need to subtract MIN_RT
                X = Q_highest.reshape(-1, 1)
                y = Q_abl
                
                # Use simple linear regression (equivalent to weighted with equal weights)
                model = LinearRegression()
                model.fit(X, y)
                
                # Get predictions and R²
                y_pred = model.predict(X)
                r_squared = model.score(X, y)
                
                # Store the fit parameters in the dictionary
                fit_params[abs_ILD][abl] = {
                    'slope': float(model.coef_[0]),
                    'intercept': float(model.intercept_),
                    'r_squared': float(r_squared),
                    'reference_abl': int(abl_highest)
                }
                if int(abl_highest) != 60:
                    raise ValueError(f'Reference ABL is not 60')
                    
                
                # Plot the fitted line
                ax.plot(Q_highest, y_pred, linestyle='-', 
                        label=f'ABL {abl} fit: y={model.coef_[0]:.3f}x+{model.intercept_:.3f}, R²={r_squared:.3f}')
            
            # Add reference line (y=x)
            # min_val = min(np.min(Q_highest), np.min(all_y_values)) if all_y_values else MIN_RT
            # max_val = max(np.max(Q_highest), np.max(all_y_values)) if all_y_values else 0.4
            min_val = MIN_RT  # Start from minimum RT of 90ms
            max_val = 0.4
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            ax.set_xlabel(f'Percentiles of RTwrtStim (ABL={abl_highest})')
            ax.set_ylabel('Percentiles of RTwrtStim (other ABLs)')
            # Count total number of rows for this abs_ILD
            total_rows = len(RTwrtStim_pos)
            ax.set_title(f'abs(ILD): {abs_ILD} (N={total_rows})')
            ax.legend(title='ABL', fontsize='small')
            
            # Set fixed x-axis limits as requested
            ax.set_xlim(0, 0.4)
        
        # Set consistent y-axis limits across all plots
        if all_y_values:
            global_y_min = MIN_RT  # Start at minimum RT of 90ms
            global_y_max = max(all_y_values)
            
            # Add some padding to y-axis
            y_padding = global_y_max * 0.1
            global_y_max += y_padding
            
            # Apply y-axis limits to all subplots
            for j, abs_ILD in enumerate(abs_ILDs):
                if axes[j, 0].get_visible():
                    axes[j, 0].set_ylim(global_y_min, global_y_max)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

print(f"PDF saved to {pdf_path}")


# %%
# Print fit parameters as a formatted table
import pandas as pd

# Convert nested dictionary to a list of dictionaries for pandas DataFrame
table_data = []
for abs_ild in fit_params:
    for abl in fit_params[abs_ild]:
        table_data.append({
            'ABL': abl,
            'abs ILD': abs_ild,
            'slope': fit_params[abs_ild][abl]['slope'],
            'intercept': fit_params[abs_ild][abl]['intercept'],
            'R^2': fit_params[abs_ild][abl]['r_squared']
        })

# Create DataFrame and sort by abs_ILD and ABL
df_params = pd.DataFrame(table_data)
df_params = df_params.sort_values(by=['abs ILD', 'ABL'])

# Format the table with proper column headers
print("\nFit Parameters Table:\n")

# Group by ABL and print each group separately
for abl in [20, 40]:  # Print ABL 20 first, then ABL 40
    abl_data = df_params[df_params['ABL'] == abl].sort_values(by='abs ILD')
    if not abl_data.empty:
        print(f"\nABL {abl}:\n")
        print(abl_data[['abs ILD', 'slope', 'intercept', 'R^2']].to_string(index=False, float_format=lambda x: f'{x:.3f}'))


# %%
# Create RTD histograms and scaled RTDs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde

# Parameters for the RTD plots
pdf_path = "rtd_and_scaled_rtd_kde.pdf"
bin_width = 0.01  # 20 ms bins
max_rt = 1  # Maximum RT to plot (1 second)
RT_MIN = 0.1  # Minimum RT threshold (90 ms) to exclude anticipatory responses
bins = np.arange(RT_MIN, max_rt + bin_width, bin_width)
kde_points = np.linspace(RT_MIN, max_rt, 300)  # Points for KDE evaluation

with PdfPages(pdf_path) as pdf:
    # Get all unique abs_ILDs across the entire dataset
    abs_ILDs = np.sort(merged_valid['abs_ILD'].unique())
    n_abs_ILDs = len(abs_ILDs)
    
    if n_abs_ILDs == 0:
        print("No valid abs_ILD values found in the dataset.")
    else:
        # Create a figure with two rows: original RTDs and scaled RTDs
        fig, axes = plt.subplots(2, n_abs_ILDs, figsize=(15, 8), squeeze=False)
        fig.suptitle('Reaction Time Distributions and Scaled RTDs', fontsize=16)
        
        # Colors for different ABLs
        colors = {20: 'green', 40: 'red', 60: 'blue'}
        
        # Store max density for consistent y-axis scaling
        max_density_original = 0
        max_density_scaled = 0
        
        for j, abs_ILD in enumerate(abs_ILDs):
            # Get all data for this abs_ILD
            abs_ILD_df = merged_valid[merged_valid['abs_ILD'] == abs_ILD]
            
            if abs_ILD_df.empty:
                axes[0, j].set_visible(False)
                axes[1, j].set_visible(False)
                continue
                
            # Filter for valid reaction times and success values
            # Use RT_MIN to exclude anticipatory responses as suggested
            RTwrtStim_pos = abs_ILD_df[(abs_ILD_df['RTwrtStim'] >= RT_MIN) & 
                                       (abs_ILD_df['RTwrtStim'] <= max_rt) & 
                                       (abs_ILD_df['success'].isin([1,-1]))]
            
            if RTwrtStim_pos.empty:
                axes[0, j].set_visible(False)
                axes[1, j].set_visible(False)
                continue
                
            ABLs = np.sort(RTwrtStim_pos['ABL'].unique())
            
            if len(ABLs) == 0:
                axes[0, j].set_visible(False)
                axes[1, j].set_visible(False)
                continue
            
            # Reference ABL (highest)
            abl_highest = ABLs.max()
            
            # Plot original RTDs (top row)
            for abl in ABLs:
                # Get RT data for this ABL
                rt_data = RTwrtStim_pos[RTwrtStim_pos['ABL'] == abl]['RTwrtStim'].values
                
                if len(rt_data) > 1:  # Need at least 2 points for KDE
                    # Use KDE instead of histogram for smoother density estimates
                    kde = gaussian_kde(rt_data)
                    density = kde(kde_points)
                    
                    # Update max density for consistent y-axis
                    max_density_original = max(max_density_original, np.max(density))
                    
                    # Plot in the top row
                    axes[0, j].plot(kde_points, density, color=colors.get(abl, 'black'), 
                                   label=f'ABL {abl}')
                
                # No combined plot storage needed
            
            # Plot scaled RTDs (bottom row)
            for abl in ABLs:
                if abl == abl_highest:
                    # Reference ABL doesn't need scaling
                    rt_data = RTwrtStim_pos[RTwrtStim_pos['ABL'] == abl]['RTwrtStim'].values
                    
                    if len(rt_data) > 1:  # Need at least 2 points for KDE
                        # Use KDE instead of histogram
                        kde = gaussian_kde(rt_data)
                        density = kde(kde_points)
                        axes[1, j].plot(kde_points, density, color=colors.get(abl, 'black'), 
                                       label=f'ABL {abl}')
                    
                    # No combined plot storage needed
                else:
                    # Scale the RT data using the slope from fit_params
                    if abs_ILD in fit_params and abl in fit_params[abs_ILD]:
                        slope = fit_params[abs_ILD][abl]['slope']
                        intercept = fit_params[abs_ILD][abl]['intercept']
                        
                        # Get RT data for this ABL
                        rt_data = RTwrtStim_pos[RTwrtStim_pos['ABL'] == abl]['RTwrtStim'].values
                        
                        # Scale the data: (x - intercept) / slope
                        scaled_rt_data = (rt_data - intercept) / slope
                        
                        # Filter out negative values after scaling
                        # scaled_rt_data = scaled_rt_data[scaled_rt_data > 0]
                        # scaled_rt_data = scaled_rt_data[scaled_rt_data <= max_rt]
                        
                        if len(scaled_rt_data) > 1:  # Need at least 2 points for KDE
                            # Use KDE for scaled data - don't truncate the data before density estimation
                            kde = gaussian_kde(scaled_rt_data)
                            density = kde(kde_points)
                            
                            # Update max density for consistent y-axis
                            max_density_scaled = max(max_density_scaled, np.max(density))
                            
                            # Plot in the bottom row
                            axes[1, j].plot(kde_points, density, color=colors.get(abl, 'black'), 
                                           label=f'ABL {abl} scaled')
                        
                        # No combined plot storage needed
            
            # Set titles and labels
            axes[0, j].set_title(f'|ILD| = {abs_ILD} dB')
            axes[0, j].set_xlabel('RT (s)')
            
            # Only add y-axis label and legend to the first column
            if j == 0:
                axes[0, j].set_ylabel('Density')
                axes[0, j].legend(fontsize='small', title='Original RTDs')
            
            axes[0, j].set_xlim(0, max_rt)
            
            axes[1, j].set_xlabel('RT (s)')
            
            # Only add y-axis label and legend to the first column
            if j == 0:
                axes[1, j].set_ylabel('Density')
                axes[1, j].legend(fontsize='small', title='Scaled RTDs')
            
            axes[1, j].set_xlim(0, max_rt)
        
        # No combined plot
        
        # Add R² inset to the bottom right plot of the last abs_ILD
        r2_inset = axes[1, -1].inset_axes([0.65, 0.65, 0.3, 0.3])
        
        # Collect R² values for each abs_ILD and ABL
        abs_ild_labels = []
        r2_values = {}
        
        for abs_ild in sorted(fit_params.keys()):
            abs_ild_labels.append(abs_ild)
            for abl in sorted(fit_params[abs_ild].keys()):
                if abl not in r2_values:
                    r2_values[abl] = []
                r2_values[abl].append(fit_params[abs_ild][abl]['r_squared'])
        
        # No combined label needed
        
        # Plot R² values as bar chart
        bar_width = 0.2
        x = np.arange(len(abs_ild_labels))
        
        for i, (abl, values) in enumerate(sorted(r2_values.items())):
            # Plot just the actual values
            r2_inset.bar(x + (i - 1) * bar_width, values, bar_width, 
                        color=colors.get(abl, 'black'), label=f'ABL {abl}')
        
        r2_inset.set_title('R²', fontsize=10)
        r2_inset.set_ylim(0.99, 1.0)  # Set y-axis from 0.99 to 1.0 to highlight differences
        r2_inset.set_xticks(x)
        r2_inset.set_xticklabels(abs_ild_labels, fontsize=8, rotation=45)
        r2_inset.tick_params(axis='both', which='major', labelsize=8)
        
        # Set consistent y-axis limits across all plots
        for j in range(n_abs_ILDs):
            if not axes[0, j].get_visible():
                continue
            axes[0, j].set_ylim(0, max_density_original * 1.1)
            axes[1, j].set_ylim(0, max_density_scaled * 1.1)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

print(f"RTD plots saved to {pdf_path}")


# %%
# Plot psychometric function (probability of choice == 1 vs ILD) for each ABL
pdf_path = "psychometric_merged.pdf"

# Create a figure
plt.figure(figsize=(10, 6))

# Get unique ABLs
ABLs = np.sort(merged_valid['ABL'].unique())

# Define colors for each ABL
colors = {20: 'blue', 40: 'green', 60: 'red'}

# Get unique ILDs (including sign)
ILDs = np.sort(merged_valid['ILD'].unique())

# For each ABL, calculate probability of choice == 1 for each ILD
for abl in ABLs:
    # Filter data for this ABL
    abl_data = merged_valid[merged_valid['ABL'] == abl]
    
    # Initialize arrays to store probabilities and ILDs
    probs = []
    ild_values = []
    
    # Calculate probability for each ILD
    for ild in ILDs:
        ild_data = abl_data[abl_data['ILD'] == ild]
        if len(ild_data) > 0:
            # Calculate probability of choice == 1
            prob = np.mean(ild_data['choice'] == 1)
            probs.append(prob)
            ild_values.append(ild)
    
    # Convert to numpy arrays
    probs = np.array(probs)
    ild_values = np.array(ild_values)
    
    # Plot scatter points
    plt.scatter(ild_values, probs, color=colors.get(abl, 'black'), 
                alpha=0.7, label=f'ABL {abl}')
    
    # Connect points with lines
    plt.plot(ild_values, probs, color=colors.get(abl, 'black'), alpha=0.5)

# Add reference line at 0.5 probability
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

# Add reference line at ILD = 0
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)

# Set labels and title
plt.xlabel('ILD (dB)')
plt.ylabel('P(choice = 1)')
plt.title('Psychometric Function by ABL')
plt.grid(alpha=0.3)
plt.legend()

# Set y-axis limits
plt.ylim(0, 1)

# Save figure
plt.tight_layout()
plt.savefig(pdf_path)
plt.close()

print(f"Psychometric function plot saved to {pdf_path}")

# %%
# Plot tachometric curves (probability of accuracy == 1 vs RTwrtStim) for each abs_ILD
pdf_path = "tachometric_merged.pdf"

# Define bin edges (from 0 to 1 seconds, in steps of 20ms)
bw = 0.05
bin_edges = np.arange(0, 1 + bw, bw)

# Helper function to calculate tachometric curve
def plot_tacho(df, bins):
    # Make a copy of the dataframe
    df = df.copy()
    
    # Bin reaction times
    df.loc[:,'RT_bin'] = pd.cut(df['RTwrtStim'], bins=bins, include_lowest=True)
    
    # Group by RT bin and calculate mean accuracy and count
    grouped_by_rt_bin = df.groupby('RT_bin', observed=False)['accuracy'].agg(['mean', 'count'])
    
    # Get bin midpoints for plotting
    grouped_by_rt_bin['bin_mid'] = grouped_by_rt_bin.index.map(lambda x: x.mid)
    
    return grouped_by_rt_bin['bin_mid'], grouped_by_rt_bin['mean']

# Get unique abs_ILDs and ABLs
abs_ILDs = np.sort(merged_valid['abs_ILD'].unique())
ABLs = np.sort(merged_valid['ABL'].unique())

# Define colors for each ABL
colors = {20: 'blue', 40: 'green', 60: 'red'}

# Create figure with subplots for each abs_ILD
fig, axes = plt.subplots(1, len(abs_ILDs), figsize=(15, 5), sharey=True)
fig.suptitle('Tachometric Curves: Accuracy vs Reaction Time', fontsize=16)

# For each abs_ILD
for i, abs_ild in enumerate(abs_ILDs):
    ax = axes[i]
    
    # Filter data for this abs_ILD
    ild_data = merged_valid[merged_valid['abs_ILD'] == abs_ild]
    
    # For each ABL
    for abl in ABLs:
        # Filter data for this ABL
        abl_data = ild_data[ild_data['ABL'] == abl]
        
        # Skip if no data
        if len(abl_data) == 0:
            continue
        
        # Filter for valid reaction times
        rt_data = abl_data[abl_data['RTwrtStim'] > 0]
        
        # Skip if no valid RTs
        if len(rt_data) == 0:
            continue
        
        # Calculate tachometric curve
        bin_mids, accuracies = plot_tacho(rt_data, bin_edges)
        
        # Plot accuracy vs RT for this ABL
        ax.plot(bin_mids, accuracies, color=colors.get(abl, 'black'), 
                alpha=0.7, label=f'ABL {abl}')
    
    # Set title and labels
    ax.set_title(f'abs(ILD): {abs_ild}')
    ax.set_xlabel('Reaction Time (s)')
    if i == 0:
        ax.set_ylabel('Accuracy')
    
    # Add reference line at 0.5 accuracy (chance level)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add grid
    ax.grid(alpha=0.3)
    
    # Add legend if it's the first plot
    if i == 0:
        ax.legend()

# Set y-axis limits
for ax in axes:
    ax.set_ylim(0.4, 1)
    ax.set_xlim(0, 0.7)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure
plt.savefig(pdf_path)
plt.close(fig)

print(f"Tachometric curves saved to {pdf_path}")

# %%
print(merged_valid['batch_name'].unique())

# %%
# Plot psychometric function for each batch separately
pdf_path = "psychometric_function_by_batch.pdf"

# Get unique batches
batches = np.sort(merged_valid['batch_name'].unique())

# Define colors for each ABL
colors = {20: 'blue', 40: 'green', 60: 'red'}

# Create figure with subplots for each batch
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten for easier indexing
fig.suptitle('Psychometric Function by Batch', fontsize=16)

# For each batch
for i, batch in enumerate(batches):
    ax = axes[i]
    
    # Filter data for this batch
    batch_data = merged_valid[merged_valid['batch_name'] == batch]
    
    # Get unique ABLs in this batch
    batch_ABLs = np.sort(batch_data['ABL'].unique())
    
    # Get unique ILDs (including sign)
    ILDs = np.sort(batch_data['ILD'].unique())
    
    # For each ABL, calculate probability of choice == 1 for each ILD
    for abl in batch_ABLs:
        # Filter data for this ABL
        abl_data = batch_data[batch_data['ABL'] == abl]
        
        # Initialize arrays to store probabilities and ILDs
        probs = []
        ild_values = []
        
        # Calculate probability for each ILD
        for ild in ILDs:
            ild_data = abl_data[abl_data['ILD'] == ild]
            if len(ild_data) > 0:
                # Calculate probability of choice == 1
                prob = np.mean(ild_data['choice'] == 1)
                probs.append(prob)
                ild_values.append(ild)
        
        # Convert to numpy arrays
        probs = np.array(probs)
        ild_values = np.array(ild_values)
        
        # Plot scatter points
        ax.scatter(ild_values, probs, color=colors.get(abl, 'black'), 
                  alpha=0.7, label=f'ABL {abl}')
        
        # Connect points with lines
        ax.plot(ild_values, probs, color=colors.get(abl, 'black'), alpha=0.5)
    
    # Set title and labels
    ax.set_title(f'Batch: {batch}')
    ax.set_xlabel('ILD (dB)')
    ax.set_ylabel('P(choice = 1)')
    
    # Add reference lines
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add grid
    ax.grid(alpha=0.3)
    
    # Add legend
    ax.legend()

# Set y-axis limits
for ax in axes:
    ax.set_ylim(0, 1)

# Hide any unused subplots
for j in range(len(batches), len(axes)):
    axes[j].set_visible(False)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure
plt.savefig(pdf_path)
plt.close(fig)

print(f"Psychometric function by batch saved to {pdf_path}")

# %%
#a psychometric, without LED1 and LED2 batch
merged_valid_abl_60_good = merged_valid[(merged_valid['batch_name'] != 'LED1') & (merged_valid['batch_name'] != 'LED2')]
# merged_valid_abl_60_good = merged_valid[merged_valid['batch_name'] == 'LED2']

# Plot psychometric function for filtered dataset (excluding LED1 and LED2 batches)
pdf_path = "psychometric_function_good_60.pdf"

# Create a figure
plt.figure(figsize=(10, 6))

# Get unique ABLs
ABLs = np.sort(merged_valid_abl_60_good['ABL'].unique())

# Define colors for each ABL
colors = {20: 'blue', 40: 'green', 60: 'red'}

# Get unique ILDs (including sign)
ILDs = np.sort(merged_valid_abl_60_good['ILD'].unique())

# For each ABL, calculate probability of choice == 1 for each ILD
for abl in ABLs:
    # Filter data for this ABL
    abl_data = merged_valid_abl_60_good[merged_valid_abl_60_good['ABL'] == abl]
    
    # Initialize arrays to store probabilities and ILDs
    probs = []
    ild_values = []
    
    # Calculate probability for each ILD
    for ild in ILDs:
        ild_data = abl_data[abl_data['ILD'] == ild]
        if len(ild_data) > 0:
            # Calculate probability of choice == 1
            prob = np.mean(ild_data['choice'] == 1)
            probs.append(prob)
            ild_values.append(ild)
    
    # Convert to numpy arrays
    probs = np.array(probs)
    ild_values = np.array(ild_values)
    
    # Plot scatter points
    plt.scatter(ild_values, probs, color=colors.get(abl, 'black'), 
                alpha=0.7, label=f'ABL {abl}')
    
    # Connect points with lines
    plt.plot(ild_values, probs, color=colors.get(abl, 'black'), alpha=0.5)

# Add reference line at 0.5 probability
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

# Add reference line at ILD = 0
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)

# Set labels and title
plt.xlabel('ILD (dB)')
plt.ylabel('P(choice = 1)')
plt.title('Psychometric Function (Excluding LED1 and LED2 batches)')
plt.grid(alpha=0.3)
plt.legend()

# Set y-axis limits
plt.ylim(0, 1)

# Save figure
plt.tight_layout()
plt.savefig(pdf_path)
plt.close()

print(f"Filtered psychometric function plot saved to {pdf_path}")


# %%
# psychometric per animal of LED2
# Filter for LED2 batch animals
led2_data = merged_valid[merged_valid['batch_name'] == 'LED2']

# Get unique animals in LED2 batch
led2_animals = np.sort(led2_data['animal'].unique())

# Create a PDF to save all plots
pdf_path = "psychometric_function_LED2_per_animal.pdf"

# Define colors for each ABL
colors = {20: 'blue', 40: 'green', 60: 'red'}

# Create a figure with subplots for each animal
fig, axes = plt.subplots(len(led2_animals), 1, figsize=(10, 5*len(led2_animals)), squeeze=False)
fig.suptitle('Psychometric Function per Animal (LED2 Batch)', fontsize=16)

# Get unique ILDs (including sign)
ILDs = np.sort(led2_data['ILD'].unique())

# For each animal in LED2 batch
for i, animal in enumerate(led2_animals):
    ax = axes[i, 0]
    
    # Filter data for this animal
    animal_data = led2_data[led2_data['animal'] == animal]
    
    # Get unique ABLs for this animal
    animal_ABLs = np.sort(animal_data['ABL'].unique())
    
    # For each ABL, calculate probability of choice == 1 for each ILD
    for abl in animal_ABLs:
        # Filter data for this ABL
        abl_data = animal_data[animal_data['ABL'] == abl]
        
        # Initialize arrays to store probabilities and ILDs
        probs = []
        ild_values = []
        
        # Calculate probability for each ILD
        for ild in ILDs:
            ild_data = abl_data[abl_data['ILD'] == ild]
            if len(ild_data) > 0:
                # Calculate probability of choice == 1
                prob = np.mean(ild_data['choice'] == 1)
                probs.append(prob)
                ild_values.append(ild)
        
        # Convert to numpy arrays
        probs = np.array(probs)
        ild_values = np.array(ild_values)
        
        # Plot scatter points
        ax.scatter(ild_values, probs, color=colors.get(abl, 'black'), 
                    alpha=0.7, label=f'ABL {abl}')
        
        # Connect points with lines
        ax.plot(ild_values, probs, color=colors.get(abl, 'black'), alpha=0.5)
    
    # Add reference line at 0.5 probability
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add reference line at ILD = 0
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('ILD (dB)')
    ax.set_ylabel('P(choice = 1)')
    ax.set_title(f'Animal {animal}')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Set y-axis limits
    ax.set_ylim(0, 1)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure
plt.savefig(pdf_path)
plt.close(fig)

print(f"LED2 psychometric function plots per animal saved to {pdf_path}")
