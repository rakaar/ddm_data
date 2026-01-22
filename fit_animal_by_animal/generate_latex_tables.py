import pandas as pd
import numpy as np

def generate_latex_table(csv_path, output_path, title):
    df = pd.read_csv(csv_path)
    
    # Rename animals to Rat 1, Rat 2, ... and Aggregate to All
    n_animals = len(df) - 1  # exclude Aggregate row
    new_names = [f'Rat {i+1}' for i in range(n_animals)] + ['All']
    df['Animal'] = new_names
    
    # Get parameter columns (exclude Animal)
    param_cols = [c for c in df.columns if c != 'Animal']
    
    # LaTeX formatting for parameter names
    latex_names = {
        'V_A': r'$V_A$',
        'theta_A': r'$\theta_A$',
        't_A_aff': r'$t_{A,\mathrm{aff}}$',
        'rate_lambda': r'$\lambda$',
        'T_0': r'$T_0$',
        'theta_E': r'$\theta_E$',
        'w': r'$w$',
        't_E_aff': r'$t_{E,\mathrm{aff}}$',
        'del_go': r'$\delta_{\mathrm{go}}$',
        'rate_norm_l': r'$\lambda_{\mathrm{norm}}$',
    }
    
    # Group mean/std pairs
    param_pairs = []
    processed = set()
    for col in param_cols:
        if col in processed:
            continue
        if '_ms_std' in col:
            continue
        if '_std' in col:
            continue
        # This is a mean column
        base = col.replace('_mean', '').replace('_ms', '')
        if '_ms' in col:
            mean_col = col
            std_col = col.replace('_ms', '_ms_std')
            label = latex_names.get(base, f'${base}$') + ' (ms)'
        else:
            mean_col = col
            std_col = col.replace('_mean', '_std')
            label = latex_names.get(base, f'${base}$')
        param_pairs.append((label, mean_col, std_col))
        processed.add(mean_col)
        processed.add(std_col)
    
    # Build LaTeX
    n_cols = len(param_pairs)
    col_spec = 'l' + 'c' * n_cols
    
    lines = []
    # Document preamble
    lines.append(r'\documentclass[11pt]{article}')
    lines.append(r'\usepackage{booktabs}')
    lines.append(r'\usepackage{amsmath}')
    lines.append(r'\usepackage[margin=0.5in,landscape]{geometry}')
    lines.append(r'\begin{document}')
    lines.append(r'\pagestyle{empty}')
    lines.append('')
    
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{' + title + '}')
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')
    
    # Header row
    header = ' & '.join([p[0] for p in param_pairs])
    lines.append(r' & ' + header + r' \\')
    lines.append(r'\midrule')
    
    # Data rows
    for idx, row in df.iterrows():
        animal = row['Animal']
        values = []
        for label, mean_col, std_col in param_pairs:
            mean_val = row[mean_col]
            std_val = row[std_col]
            # Format based on magnitude
            if abs(mean_val) >= 10:
                values.append(f'${mean_val:.1f} \\pm {std_val:.1f}$')
            elif abs(mean_val) >= 1:
                values.append(f'${mean_val:.2f} \\pm {std_val:.2f}$')
            else:
                values.append(f'${mean_val:.3f} \\pm {std_val:.3f}$')
        line = animal + ' & ' + ' & '.join(values) + r' \\'
        lines.append(line)
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')
    lines.append(r'\end{document}')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Saved: {output_path}')

# Generate for all 3 CSVs
generate_latex_table(
    'compare_animals_vbmc_aborts_results.csv',
    'compare_animals_vbmc_aborts_results.tex',
    'Aborts Model Parameters'
)

generate_latex_table(
    'compare_animals_vbmc_vanilla_tied_results.csv',
    'compare_animals_vbmc_vanilla_tied_results.tex',
    'Vanilla Tied Model Parameters'
)

generate_latex_table(
    'compare_animals_vbmc_norm_tied_results.csv',
    'compare_animals_vbmc_norm_tied_results.tex',
    'Norm Tied Model Parameters'
)
