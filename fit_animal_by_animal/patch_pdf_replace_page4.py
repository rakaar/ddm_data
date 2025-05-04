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

# Load your dataframes and model params as needed here
# For demonstration, assume df_valid_and_aborts, df_aborts, V_A, theta_A, t_A_aff are available
from debug_proactive_diag import df_valid_and_aborts, df_aborts, V_A, theta_A, t_A_aff

for animal in df_valid_and_aborts['animal'].unique():
    orig_pdf = f'results_{batch_name}_animal_{animal}.pdf'
    new_pdf = f'results_{batch_name}_animal_{animal}_diagnostic_fixed.pdf'
    if not os.path.exists(orig_pdf):
        print(f"Original PDF {orig_pdf} not found, skipping.")
        continue
    # Read all pages from original PDF
    reader = PdfReader(orig_pdf)
    writer = PdfWriter()
    num_pages = len(reader.pages)

    # Copy pages 1-3 (0-based: 0,1,2)
    for i in range(num_pages):
        if i == 3:
            # Replace page 4 with new diagnostic
            # Generate new diagnostic as a temp PDF
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
            # Insert the new diagnostic page
            diag_reader = PdfReader(temp_diag_pdf)
            writer.add_page(diag_reader.pages[0])
            os.remove(temp_diag_pdf)
        else:
            # Copy original page
            writer.add_page(reader.pages[i])
    # Write to new PDF
    with open(new_pdf, 'wb') as f_out:
        writer.write(f_out)
    print(f"Patched PDF written: {new_pdf}")
