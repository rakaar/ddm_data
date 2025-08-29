# %%
# param vs animal wise pdfs

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import shutil
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import subprocess

# Directory containing the results
RESULTS_DIR = os.path.dirname(__file__)
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']


# Define simple, high-contrast colors for each batch
BATCH_COLORS = {
    'Comparable': 'red',
    'SD': '#87CEEB',  # sky blue
    'LED2': 'green',
    'LED1': 'orange',
    'LED34': 'purple',
    'LED7': 'black',
    'LED34_even': 'blue',
}

# %%
# Build animal list from CSVs for DESIRED_BATCHES
batch_dir = os.path.join(RESULTS_DIR, 'batch_csvs')
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]
dfs = []
for fname in batch_files:
    fpath = os.path.join(batch_dir, fname)
    if os.path.exists(fpath):
        dfs.append(pd.read_csv(fpath))
if len(dfs) > 0:
    merged_data = pd.concat(dfs, ignore_index=True)
    merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
    batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))
    print(f"Found {len(batch_animal_pairs)} batch-animal pairs from CSVs.")
else:
    print('Warning: No batch CSVs found for DESIRED_BATCHES. Falling back to scanning PKL files in RESULTS_DIR.')
    batch_animal_pairs = []

# Build animal tuples from CSV-derived set if available; otherwise fallback to directory scan
animal_batch_tuples = []  # List of (batch, animal_number)
pkl_files = []  # List of (batch, animal_number, filename)
if batch_animal_pairs:
    for (batch, animal) in batch_animal_pairs:
        try:
            animal_id = int(animal)
        except Exception:
            # Skip non-integer animal identifiers as PKL files use integer IDs
            continue
        fname = f'results_{batch}_animal_{animal_id}.pkl'
        pkl_path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(pkl_path):
            animal_batch_tuples.append((batch, animal_id))
            pkl_files.append((batch, animal_id, fname))
else:
    for fname in os.listdir(RESULTS_DIR):
        if fname.startswith('results_') and fname.endswith('.pkl'):
            for batch in DESIRED_BATCHES:
                prefix = f'results_{batch}_animal_'
                if fname.startswith(prefix):
                    try:
                        animal_id = int(fname.split('_')[-1].replace('.pkl', ''))
                        animal_batch_tuples.append((batch, animal_id))
                        pkl_files.append((batch, animal_id, fname))
                    except Exception:
                        continue

# Sort by batch then animal number
animal_batch_tuples = sorted(animal_batch_tuples, key=lambda x: (x[0], x[1]))
# %%
# Model configs: (model_key, param_keys, param_labels, plot_title)
model_configs = [
    ('vbmc_aborts_results',
        ['V_A_samples', 'theta_A_samples', 't_A_aff_samp'],
        ['V_A', 'theta_A', 't_A_aff'],
        'Aborts Model'),
    ('vbmc_vanilla_tied_results',
        ['rate_lambda_samples', 'T_0_samples', 'theta_E_samples', 'w_samples', 't_E_aff_samples', 'del_go_samples'],
        ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go'],
        'Vanilla TIED Model'),
    ('vbmc_norm_tied_results',
        ['rate_lambda_samples', 'T_0_samples', 'theta_E_samples', 'w_samples', 't_E_aff_samples', 'del_go_samples', 'rate_norm_l_samples'],
        ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go', 'rate_norm_l'],
        'Norm TIED Model'),
    ('vbmc_time_vary_norm_tied_results',
        ['rate_lambda_samples', 'T_0_samples', 'theta_E_samples', 'w_samples', 't_E_aff_samples', 'del_go_samples', 'rate_norm_l_samples', 'bump_height_samples', 'bump_width_samples', 'dip_height_samples', 'dip_width_samples'],
        ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go', 'rate_norm_l', 'bump_height', 'bump_width', 'dip_height', 'dip_width'],
        'Time-Varying Norm TIED Model'),
]

overall_param_means = {}

for model_key, param_keys, param_labels, plot_title in model_configs:
    means = {param: [] for param in param_keys}
    ci_lows = {param: [] for param in param_keys}   # 2.5th percentile
    ci_highs = {param: [] for param in param_keys}  # 97.5th percentile
    valid_animals = []  # Will store (batch, animal_id)
    valid_labels = []   # Will store strings like 'LED7-92'
    batch_colors = []   # Color for each animal
    # First pass: gather means and 95% CI (nonparametric)
    for batch, animal_id in animal_batch_tuples:
        pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
        pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key not in results:
            continue
        valid_animals.append((batch, animal_id))
        valid_labels.append(f'{batch}-{animal_id}')
        batch_colors.append(BATCH_COLORS.get(batch, 'gray'))
        for param in param_keys:
            samples = np.asarray(results[model_key][param])
            mean = np.mean(samples)
            ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
            means[param].append(mean)
            ci_lows[param].append(ci_lower)
            ci_highs[param].append(ci_upper)

    # Aggregate mean across animals per parameter for this model
    overall_param_means[plot_title] = {}
    for p_key, p_label in zip(param_keys, param_labels):
        vals = np.array(means[p_key], dtype=float)
        overall_param_means[plot_title][p_label] = float(np.nanmean(vals)) if vals.size > 0 else float('nan')

    from matplotlib.backends.backend_pdf import PdfPages
    outname = f'compare_animals_all_batches_{model_key}.pdf'
    with PdfPages(os.path.join(RESULTS_DIR, outname)) as pdf:
        for i, param in enumerate(param_keys):
            # Sort by mean value (descending)
            zipped = list(zip(means[param], ci_lows[param], ci_highs[param], valid_labels, batch_colors, valid_animals))
            if len(zipped) == 0:
                # No data for this parameter in this model; skip plotting
                continue
            zipped.sort(key=lambda x: x[0], reverse=True)
            sorted_means, sorted_ci_lows, sorted_ci_highs, sorted_labels, sorted_colors, sorted_animals = zip(*zipped)
            y_pos = np.arange(len(sorted_labels))
            fig, ax = plt.subplots(figsize=(7, 6))
            for idx in range(len(sorted_labels)):
                # Plot CI as a horizontal line
                ax.hlines(y=y_pos[idx], xmin=sorted_ci_lows[idx], xmax=sorted_ci_highs[idx], color=sorted_colors[idx], linewidth=3, alpha=0.7)
                # Plot mean as a point
                ax.plot(sorted_means[idx], y_pos[idx], 'o', color=sorted_colors[idx])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_labels)
            # Color y-tick labels by batch
            for ticklabel, (batch, _) in zip(ax.get_yticklabels(), sorted_animals):
                ticklabel.set_color(BATCH_COLORS.get(batch, 'gray'))
            ax.set_xlabel(param_labels[i])
            ax.set_ylabel('Batch-Animal')
            ax.set_title(f'{plot_title}: {param_labels[i]} (mean, 95% CI)')
            ax.axvspan(0, np.max(np.array(sorted_ci_highs)), color='#b7e4c7', alpha=0.2, zorder=-1)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f'Saved: {outname}')

# %%
# elbo, loglike

fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)

# Model order for this plot (to match rows)
model_order = [
    ('vbmc_aborts_results', 'Aborts Model'),
    ('vbmc_vanilla_tied_results', 'Vanilla TIED Model'),
    ('vbmc_norm_tied_results', 'Norm TIED Model'),
    ('vbmc_time_vary_norm_tied_results', 'Time-Varying Norm TIED Model'),
]

# First, collect all TIED model values to compute global y-limits
all_tied_elbos = []
all_tied_elbo_sds = []
all_tied_loglikes = []
for row, (model_key, _) in enumerate(model_order[1:], 1):
    for batch, animal_id in animal_batch_tuples:
        pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
        pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key not in results:
            continue
        all_tied_elbos.append(results[model_key].get('elbo', np.nan))
        all_tied_elbo_sds.append(results[model_key].get('elbo_sd', 0.0))
        all_tied_loglikes.append(results[model_key].get('loglike', np.nan))

tied_vals = np.array(all_tied_elbos + all_tied_loglikes)
tied_finite = tied_vals[np.isfinite(tied_vals)]
if len(tied_finite) > 0:
    tied_min, tied_max = np.min(tied_finite), np.max(tied_finite)
    tied_pad = 0.05 * (tied_max - tied_min)
    tied_min -= tied_pad
    tied_max += tied_pad
else:
    tied_min, tied_max = None, None

for row, (model_key, model_title) in enumerate(model_order):
    elbos = []
    elbo_sds = []
    loglikes = []
    valid_animals = []  # (batch, animal_id)
    valid_labels = []   # e.g. LED7-92
    for batch, animal_id in animal_batch_tuples:
        pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
        pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key not in results:
            continue
        elbos.append(results[model_key].get('elbo', np.nan))
        elbo_sds.append(results[model_key].get('elbo_sd', 0.0))
        loglikes.append(results[model_key].get('loglike', np.nan))
        valid_animals.append((batch, animal_id))
        valid_labels.append(f'{batch}-{animal_id}')
    x = np.arange(len(valid_labels))
    # Get y-limits for aborts (row 0) and use tied_min/tied_max for others
    if row == 0:
        all_vals = np.array(elbos + loglikes)
        finite_vals = all_vals[np.isfinite(all_vals)]
        if len(finite_vals) > 0:
            min_y, max_y = np.min(finite_vals), np.max(finite_vals)
            pad = 0.05 * (max_y - min_y)
            min_y -= pad
            max_y += pad
        else:
            min_y, max_y = None, None
    else:
        min_y, max_y = tied_min, tied_max
    # ELBO bar plot with error bars
    ax_elbo = axes[row, 0]
    ax_elbo.bar(x, elbos, yerr=elbo_sds, color='royalblue', alpha=0.8, capsize=6, edgecolor='black')
    ax_elbo.set_ylabel(model_title)
    ax_elbo.set_title('ELBO vs Batch-Animal')
    ax_elbo.set_xticks(x)
    ax_elbo.set_xticklabels(valid_labels, rotation=45, ha='right')
    if row == 3:
        ax_elbo.set_xlabel('Batch-Animal')
    if min_y is not None and max_y is not None:
        ax_elbo.set_ylim([min_y, max_y])
    # loglike bar plot
    ax_loglike = axes[row, 1]
    ax_loglike.bar(x, loglikes, color='firebrick', alpha=0.8, edgecolor='black')
    ax_loglike.set_title('Loglike vs Batch-Animal')
    ax_loglike.set_xticks(x)
    ax_loglike.set_xticklabels(valid_labels, rotation=45, ha='right')
    if row == 3:
        ax_loglike.set_xlabel('Batch-Animal')
    if min_y is not None and max_y is not None:
        ax_loglike.set_ylim([min_y, max_y])
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'compare_animals_elbo_loglike_all_batches.pdf'))
print('Saved: compare_animals_elbo_loglike_all_batches.pdf')
plt.close(fig)

# %%
# Print overall mean of each parameter across animals for each model
print("\n=== Mean parameter values across animals (by model) ===")
for model_title, param_means in overall_param_means.items():
    print(f"[{model_title}]")
    for label, value in param_means.items():
        try:
            print(f"  {label}: {value:.6g}")
        except Exception:
            print(f"  {label}: {value}")

# %%
# ELBO percent increase (Norm TIED vs Vanilla TIED) per animal
vanilla_key = 'vbmc_vanilla_tied_results'
norm_key = 'vbmc_norm_tied_results'

labels = []
pct_increase = []  # 100 * (Norm - Vanilla) / Vanilla

for batch, animal_id in animal_batch_tuples:
    pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
    pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
    if not os.path.exists(pkl_path):
        continue
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    if (vanilla_key not in results) or (norm_key not in results):
        continue
    elbo_v = results[vanilla_key].get('elbo', np.nan)
    elbo_n = results[norm_key].get('elbo', np.nan)
    if not np.isfinite(elbo_v) or not np.isfinite(elbo_n) or elbo_v == 0:
        continue
    labels.append(f'{batch}-{animal_id}')
    pct_increase.append(100.0 * (elbo_n - elbo_v) / elbo_v)

if len(labels) > 0:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(labels)), 4.5))
    ax.bar(x, pct_increase, color='black', edgecolor='black', alpha=0.9)
    ax.axhline(0, color='k', linewidth=1)
    # No x-axis labels or ticks
    ax.set_xticks([])
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    # Y ticks and clean spines
    ax.set_yticks([-4, 0, 12])
    ax.tick_params(axis='y', labelsize=24)
    # Ensure these ticks are within the visible range
    ymin = min([-4] + pct_increase)
    ymax = max([12] + pct_increase)
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # No titles or labels
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['bottom'].set_visible(False)
    ax.set_title('ELBO increase (%) = 100 * (Norm - Vanilla) / Vanilla')
    plt.tight_layout()
    outname = 'compare_animals_elbo_percent_increase_norm_vs_vanilla.pdf'
    plt.savefig(os.path.join(RESULTS_DIR, outname))

    print(f'Saved: {outname}')
    # plt.close(fig)
else:
    print('No overlapping animals found with both Vanilla TIED and Norm TIED ELBOs to compute percent increase.')

# %%
# LOG-LIKELIHOOD percent increase (Norm TIED vs Vanilla TIED) per animal
vanilla_key = 'vbmc_vanilla_tied_results'
norm_key = 'vbmc_norm_tied_results'

labels = []
pct_increase = []  # 100 * (Norm - Vanilla) / Vanilla

for batch, animal_id in animal_batch_tuples:
    pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
    pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
    if not os.path.exists(pkl_path):
        continue
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    if (vanilla_key not in results) or (norm_key not in results):
        continue
    ll_v = results[vanilla_key].get('loglike', np.nan)
    ll_n = results[norm_key].get('loglike', np.nan)
    if not np.isfinite(ll_v) or not np.isfinite(ll_n) or ll_v == 0:
        continue
    labels.append(f'{batch}-{animal_id}')
    pct_increase.append(100.0 * (ll_n - ll_v) / ll_v)

if len(labels) > 0:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(labels)), 4.5))
    ax.bar(x, pct_increase, color='black', edgecolor='black', alpha=0.9)
    ax.axhline(0, color='k', linewidth=1)
    # No x-axis labels or ticks
    ax.set_xticks([])
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    # Y ticks and clean spines
    ax.set_yticks([0, 30])
    ax.tick_params(axis='y', labelsize=24)
    # Ensure these ticks are within the visible range
    ymin = min([0] + pct_increase)
    ymax = max([30] + pct_increase)
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # No titles or labels
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('log-likelihood increase (%) = 100 * (Norm - Vanilla) / Vanilla')
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    outname = 'compare_animals_loglike_percent_increase_norm_vs_vanilla.pdf'
    plt.savefig(os.path.join(RESULTS_DIR, outname))
    print(f'Saved: {outname}')
    # plt.close(fig)
else:
    print('No overlapping animals found with both Vanilla TIED and Norm TIED LOG-LIKELIHOODs to compute percent increase.')

# %%
# Build per-model parameter tables (CSV): rows = Rat 1, Rat 2, ...; columns = params with "mean ± SD"
def _format_mean_sd(mean, sd, sig=3):
    try:
        if not np.isfinite(mean) or not np.isfinite(sd):
            return ''
    except Exception:
        return ''
    # Use fixed 3 decimals
    return f"{mean:.3f} ± {sd:.3f}"

def _save_params_table_pdf(df: pd.DataFrame, title: str, out_pdf_path: str) -> None:
    """Save a styled table PDF for the given DataFrame using ReportLab.

    - A4 landscape page
    - Repeating header row on page breaks
    - Alternating row backgrounds
    - Center-aligned text with thin grid
    """
    # Page geometry
    page_size = landscape(A4)
    left_margin = right_margin = top_margin = bottom_margin = 18  # 0.25 inch margins
    page_w, _ = page_size
    usable_w = page_w - left_margin - right_margin

    # Column widths and font sizing
    n_cols = len(df.columns)
    if n_cols <= 8:
        font_size = 10
    elif n_cols <= 10:
        font_size = 9
    elif n_cols <= 12:
        font_size = 8
    elif n_cols <= 14:
        font_size = 7
    else:
        font_size = 6
    col_width = usable_w / max(1, n_cols)
    col_widths = [col_width] * n_cols

    # Build table data (ensure strings)
    header = list(df.columns)
    body = [["" if (v is None) else str(v) for v in row] for row in df.values.tolist()]
    data = [header] + body

    styles = getSampleStyleSheet()
    title_para = Paragraph(f"<b>{title}</b>", styles['Title'])

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.black),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor('#EEEEEE')]),
    ]))

    doc = SimpleDocTemplate(
        out_pdf_path,
        pagesize=page_size,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
    )
    doc.build([title_para, Spacer(1, 6), table])

def _latex_header_for_param(label: str) -> str:
    """Map internal parameter labels to LaTeX-friendly column headers."""
    mapping = {
        'rate_lambda': '$\\lambda$',
        'theta_E': '$\\vartheta_e$',
        'T_0': '$T_0$ (ms)',
        't_E_aff': '$t_{E,aff}$ (ms)',
        'del_go': '$\\Delta_{go}$ (ms)',
        'w': 'w',
        'rate_norm_l': '$\\ell$',
        'bump_height': 'bump\\_height',
        'bump_width': 'bump\\_width',
        'dip_height': 'dip\\_height',
        'dip_width': 'dip\\_width',
        'V_A': '$V_A$',
        'theta_A': '$\\theta_A$',
        't_A_aff': '$t_{A,aff}$ (ms)',
    }
    return mapping.get(label, label.replace('_', '\\_'))

def _format_mean_sd_tex(mean, sd, sig=3) -> str:
    """Return a LaTeX math-mode mean ± sd string or empty if invalid."""
    try:
        if not np.isfinite(mean) or not np.isfinite(sd):
            return ''
    except Exception:
        return ''
    return f"${mean:.3f} \\pm {sd:.3f}$"

def _save_params_table_latex(df: pd.DataFrame, model_key: str, param_labels: list, plot_title: str, compile_pdf: bool = True) -> None:
    """Write a standalone LaTeX file for the table and optionally compile to PDF.

    Produces two files in RESULTS_DIR:
    - table_params_{model_key}.tex
    - table_params_{model_key}_latex.pdf (if pdflatex is available)
    """
    # Build header row labels (use LaTeX mapping for parameter columns)
    header_cells = ['']  # blank top-left header similar to sample style
    for p in param_labels:
        header_cells.append(_latex_header_for_param(p))

    # Build table rows
    lines = []
    ms_params = set()  # All time params (T_0, t_E_aff, del_go) are already in ms in df
    for _, row in df.iterrows():
        cells = [str(row['Rat'])]
        for p in param_labels:
            val = row.get(p, '')
            if isinstance(val, str) and '±' in val:
                try:
                    left, right = val.split('±')
                    m = float(left.strip())
                    s = float(right.strip())
                    # No further rescaling here; values are already in desired display units
                    cell = _format_mean_sd_tex(m, s)
                except Exception:
                    cell = val.replace('±', '$ \\pm $')
            else:
                cell = str(val)
            cells.append(cell)
        lines.append(' & '.join(cells) + ' \\\\')

    col_spec = 'l' + 'c' * len(param_labels)
    # Use ASCII-safe em-dash for pdfLaTeX
    caption = f"Model parameters --- {plot_title}"
    label = f"tab:params_{model_key}"

    tex = []
    tex.append(r"\documentclass{article}")
    tex.append(r"\usepackage[margin=1in]{geometry}")
    tex.append(r"\usepackage[utf8]{inputenc}")
    tex.append(r"\usepackage[T1]{fontenc}")
    tex.append(r"\usepackage{graphicx}")
    tex.append(r"\usepackage{amsmath}")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\begin{document}")
    tex.append(r"\begin{table}[h!]")
    tex.append(r"\centering")
    tex.append(r"\resizebox{\textwidth}{!}{%")
    tex.append(fr"\begin{{tabular}}{{{col_spec}}}")
    tex.append(r"\toprule")
    tex.append(' & '.join(header_cells) + ' \\\\')
    tex.append(r"\midrule")
    tex.extend(lines)
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"}")
    tex.append(fr"\caption{{{caption}}}")
    tex.append(fr"\label{{{label}}}")
    tex.append(r"\end{table}")
    tex.append(r"\end{document}")

    tex_name = f"table_params_{model_key}.tex"
    tex_path = os.path.join(RESULTS_DIR, tex_name)
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex))
    print(f"Saved: {tex_name}")

    if compile_pdf and shutil.which('pdflatex') is not None:
        try:
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex_name],
                cwd=RESULTS_DIR,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            base = os.path.splitext(tex_name)[0]
            # Rename/move resulting PDF to have _latex suffix to distinguish from ReportLab
            src_pdf = os.path.join(RESULTS_DIR, base + '.pdf')
            dst_pdf = os.path.join(RESULTS_DIR, f"table_params_{model_key}_latex.pdf")
            if os.path.exists(src_pdf):
                if src_pdf != dst_pdf:
                    os.replace(src_pdf, dst_pdf)
                print(f"Saved: {os.path.basename(dst_pdf)}")
        except Exception as e:
            print(f"pdflatex compilation failed: {e}")

for model_key, param_keys, param_labels, plot_title in model_configs:
    table_rows = []
    per_param_means_across_rats = {label: [] for label in param_labels}
    rat_counter = 0
    for batch, animal_id in animal_batch_tuples:
        pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
        pkl_path = os.path.join(RESULTS_DIR, pkl_fname)
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        if model_key not in results:
            continue
        rat_counter += 1
        row = {'Rat': f'Rat {rat_counter}'}
        for p_key, p_label in zip(param_keys, param_labels):
            samples = np.asarray(results[model_key][p_key]).astype(float).ravel()
            samples = samples[np.isfinite(samples)]
            if samples.size == 0:
                mean_val, sd_val = np.nan, np.nan
            else:
                mean_val = float(np.mean(samples))
                sd_val = float(np.std(samples))
            # Convert time params to ms for output tables (T_0, t_E_aff, del_go)
            if p_label in ('T_0', 't_E_aff', 'del_go'):
                if np.isfinite(mean_val):
                    mean_val *= 1000.0
                if np.isfinite(sd_val):
                    sd_val *= 1000.0
            row[p_label] = _format_mean_sd(mean_val, sd_val)
            if np.isfinite(mean_val):
                per_param_means_across_rats[p_label].append(mean_val)
        table_rows.append(row)
    # Append an 'All' row summarizing across rats (mean ± SD of per-rat means)
    if len(table_rows) > 0:
        all_row = {'Rat': 'Avg'}
        for p_label in param_labels:
            vals = np.array(per_param_means_across_rats[p_label], dtype=float)
            if vals.size > 0:
                all_row[p_label] = _format_mean_sd(float(np.nanmean(vals)), float(np.nanstd(vals)))
            else:
                all_row[p_label] = ''
        table_rows.append(all_row)
        df = pd.DataFrame(table_rows, columns=['Rat'] + param_labels)
        csv_name = f"table_params_{model_key}.csv"
        df.to_csv(os.path.join(RESULTS_DIR, csv_name), index=False)
        print(f"Saved: {csv_name}")
        # Also save a nicely formatted table as PDF per model
        pdf_name = f"table_params_{model_key}.pdf"
        _save_params_table_pdf(df, title=plot_title, out_pdf_path=os.path.join(RESULTS_DIR, pdf_name))
        print(f"Saved: {pdf_name}")
        # And save a LaTeX table (and compile to PDF if pdflatex exists)
        _save_params_table_latex(df, model_key=model_key, param_labels=param_labels, plot_title=plot_title, compile_pdf=True)
    else:
        print(f"No animals found for model {model_key}; no CSV written.")
