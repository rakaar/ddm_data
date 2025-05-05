import os
from PyPDF2 import PdfReader, PdfWriter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from animal_wise_plotting_utils import plot_abort_diagnostic
from time_vary_norm_utils import rho_A_t_VEC_fn, cum_A_t_fn

# Settings (adjust as needed)
batch_name = 'Comparable'  # or set dynamically
N_theory = int(1e3)
T_trunc = 0.3

import os
import pickle
from PyPDF2 import PdfReader, PdfWriter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from animal_wise_plotting_utils import plot_abort_diagnostic
from time_vary_norm_utils import rho_A_t_VEC_fn, cum_A_t_fn

batch_name = 'Comparable'  # or set dynamically
N_theory = int(1e3)
T_trunc = 0.3

# Load your dataframes (not parameters) as before
from debug_proactive_diag import df_valid_and_aborts, df_aborts

for animal in df_valid_and_aborts['animal'].unique():
    # --- Read model parameters from the corresponding pickle file ---
    pkl_path = f'results_{batch_name}_animal_{animal}.pkl'
    if not os.path.exists(pkl_path):
        print(f"Pickle file {pkl_path} not found, skipping animal {animal}.")
        continue
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    # Try to get abort model parameters from the pickle
    # Try common keys: 'vbmc_aborts_results', or fallback to direct keys
    if 'vbmc_aborts_results' in results:
        abort_results = results['vbmc_aborts_results']
        V_A = abort_results['V_A_samples'].mean()
        theta_A = abort_results['theta_A_samples'].mean()
        t_A_aff = abort_results['t_A_aff_samp'].mean()
    else:
        # fallback: try direct keys
        V_A = results.get('V_A', None)
        theta_A = results.get('theta_A', None)
        t_A_aff = results.get('t_A_aff', None)
        if None in (V_A, theta_A, t_A_aff):
            print(f"Could not find abort model params in {pkl_path}, skipping animal {animal}.")
            continue

    orig_pdf = f'results_{batch_name}_animal_{animal}.pdf'
    new_pdf = f'results_{batch_name}_animal_{animal}_diagnostic_fixed_v2.pdf'
    if not os.path.exists(orig_pdf):
        print(f"Original PDF {orig_pdf} not found, skipping.")
        continue
    # Read all pages from original PDF
    reader = PdfReader(orig_pdf)
    writer = PdfWriter()
    num_pages = len(reader.pages)

    # Print for debug
    print(f"Animal {animal}: V_A={V_A:.4f}, theta_A={theta_A:.4f}, t_A_aff={t_A_aff:.4f}")

    for i in range(num_pages):
        if i == 3:
            # Replace page 4 with new diagnostic
            temp_diag_pdf = f'temp_diag_{animal}.pdf'
            with PdfPages(temp_diag_pdf) as temp_pdf:
                plot_abort_diagnostic(
                    pdf_pages=temp_pdf,
                    df_aborts_animal=df_aborts[df_aborts['animal'] == animal],
                    df_valid_and_aborts=df_valid_and_aborts,
                    N_theory=N_theory,
                    V_A=V_A,
                    theta_A=theta_A,
                    t_A_aff=t_A_aff,
                    T_trunc=T_trunc,
                    rho_A_t_VEC_fn=rho_A_t_VEC_fn,
                    cum_A_t_fn=cum_A_t_fn,
                    title=f'Abort Model RTD Diagnostic (fixed, animal {animal})'
                )
            diag_reader = PdfReader(temp_diag_pdf)
            writer.add_page(diag_reader.pages[0])
            os.remove(temp_diag_pdf)
        else:
            writer.add_page(reader.pages[i])
    with open(new_pdf, 'wb') as f_out:
        writer.write(f_out)
    print(f"Patched PDF written: {new_pdf}")
