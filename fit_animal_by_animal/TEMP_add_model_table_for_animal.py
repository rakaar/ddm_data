# %% 
# read pickle file
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pkl_file = "/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_Comparable_animal_41.pkl"
try:
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Error: Pickle file not found at {pkl_file}. Please check the file path and try again.")

def render_df_to_pdf(df, title, pdf):
    fig, ax = plt.subplots(figsize=(min(20, 2 + 0.7 * len(df.columns)), 1.5 + 0.4 * len(df)))
    ax.axis('off')
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center',
                     colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# --- Function to create Abort Results Table ---
def create_abort_table(abort_results):
    if not abort_results or not isinstance(abort_results, dict):
        print("Abort results data not found or invalid.")
        return None

    data = {'Parameter': [], 'Mean Value': []}
    scalar_data = {}

    for key, value in abort_results.items():
        if key.endswith('_samples') or key == 't_A_aff_samp': # Include potential typo
            param_name = key.replace('_samples', '').replace('_samp', '')
            if isinstance(value, np.ndarray) and value.size > 0:
                data['Parameter'].append(param_name)
                data['Mean Value'].append(np.mean(value))
            else:
                data['Parameter'].append(param_name)
                data['Mean Value'].append('N/A') # Handle non-array or empty
        elif key in ['elbo', 'elbo_sd', 'loglike']:
            scalar_data[key] = value

    df = pd.DataFrame(data)
    # Add scalar values as separate rows
    for key, value in scalar_data.items():
         df.loc[len(df)] = [key, value]

    # Add message if exists
    if 'message' in abort_results:
         df.loc[len(df)] = ['message', abort_results['message']]

    print("--- Abort Model Results ---")
    print(df.to_string(index=False))
    print("\n")
    return df

# --- Function to create Tied Parameters Table ---
def create_tied_table(all_results):
    tied_keys = [k for k in all_results.keys() if k.startswith('vbmc_') and k.endswith('_tied_results')]
    if not tied_keys:
        print("No tied parameter results found.")
        return None

    # Collect all unique parameter names (base name without _samples)
    all_params = set()
    for key in tied_keys:
        for param_key in all_results[key].keys():
            if param_key.endswith('_samples'):
                all_params.add(param_key.replace('_samples', ''))

    # Use user-specified order
    desired_order = [
        'rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go',
        'rate_norm_l', 'bump_height', 'bump_width', 'dip_height', 'dip_width'
    ]
    # Only keep params that are present
    present_params = [p for p in desired_order if p in all_params]
    param_headers = []
    for p in present_params:
        if p == 'T_0':
            param_headers.append('T_0 (ms)')
        else:
            param_headers.append(p)

    table_data = {'Model': []}
    scalar_cols = ['ELBO', 'ELBO SD', 'Log Likelihood']
    for col in scalar_cols + param_headers:
        table_data[col] = []

    # Populate table data
    for key in tied_keys:
        model_name = key.replace('vbmc_', '').replace('_tied_results', '').replace('_', ' ').title()
        table_data['Model'].append(model_name)
        model_results = all_results[key]

        # Add scalar values, rounded
        for col in ['elbo', 'elbo_sd', 'loglike']:
            val = model_results.get(col, '-')
            if isinstance(val, float):
                table_data[{'elbo': 'ELBO', 'elbo_sd': 'ELBO SD', 'loglike': 'Log Likelihood'}[col]].append(f"{val:.3f}")
            else:
                table_data[{'elbo': 'ELBO', 'elbo_sd': 'ELBO SD', 'loglike': 'Log Likelihood'}[col]].append(val)

        # Add parameter means, rounded, and T_0 scaled
        for i, param in enumerate(present_params):
            param_key_samples = f"{param}_samples"
            header = param_headers[i]
            if param_key_samples in model_results and isinstance(model_results[param_key_samples], np.ndarray):
                mean_val = np.mean(model_results[param_key_samples])
                if param == 'T_0':
                    mean_val = mean_val * 1000  # convert to ms
                table_data[header].append(f"{mean_val:.3f}")
            else:
                table_data[header].append('-')

    df = pd.DataFrame(table_data)
    print("--- Tied Models Comparison ---")
    print(df.to_string(index=False))
    print("\n")
    return df

# --- Main Execution --- 
if __name__ == "__main__":
    pdf_path = "model_tables.pdf"
    abort_df = None
    tied_df = None

    if 'vbmc_aborts_results' in results:
        abort_df = create_abort_table(results['vbmc_aborts_results'])
    else:
        print("Key 'vbmc_aborts_results' not found in the results dictionary.")

    tied_df = create_tied_table(results)

    # Save both tables to PDF
    with PdfPages(pdf_path) as pdf:
        if abort_df is not None:
            render_df_to_pdf(abort_df, "Abort Model Results", pdf)
        if tied_df is not None:
            render_df_to_pdf(tied_df, "Tied Models Comparison", pdf)
    print(f"\nTables saved to PDF: {pdf_path}\n")
