# Fit Each Condition

This folder contains scripts for fitting condition-by-condition gamma and omega parameters for each animal.

## Condition-by-Condition Fitting

### Fitting Scripts
- `fit_single_rat_condn_by_condn_fix_t_E_w_del_go_all_animals_loop_for_paper.py` - Fit condition-by-condition gamma/omega for each animal with t_E_aff, w, del_go fixed
- `diagnostics_cond_by_cond_fit_fix_t_E_aff_w_del_go_all_animals_for_paper.py` - Diagnostics for condition-by-condition fits

### Posterior Visualization
- `plot_gamma_omega_posteriors_all_animals.py` - Creates PDF files with gamma and omega posterior distributions for each animal (10 ILDs × 3 ABLs grid)
- `plot_gamma_omega_posteriors_all_animals_by_stimulus.py` - Plots gamma and omega posterior distributions aggregated across all animals for each stimulus condition

### Utility Scripts
- `led_off_gamma_omega_pdf_utils.py` - PDF/CDF functions for gamma/omega models, including vectorized implementations.

### Data Files
- `each_animal_cond_fit_gama_omega_pkl_files/` - VBMC fit results for each condition (ABL, ILD) per animal
- `each_animal_cond_fit_gama_omega_pkl_files_LAPSES/` - Condition fits with lapse parameters
