# Fit Each Condition

This folder contains scripts for fitting condition-by-condition gamma and omega parameters for each animal.

## Condition-by-Condition Fitting

### Fitting Scripts
- `fit_single_rat_condn_by_condn_fix_t_E_w_del_go_all_animals_loop_for_paper.py` - Fit condition-by-condition gamma/omega for each animal with t_E_aff, w, del_go fixed
- `fit_mean_omega_alpha_model.py` - Loads animal condition-fit gamma/omega posteriors, averages across animals, fits the shared alpha interaction model to mean Gamma/Omega, and saves fit diagnostic figures.
- `diagnostics_cond_by_cond_fit_fix_t_E_aff_w_del_go_all_animals_for_paper.py` - Diagnostics for condition-by-condition fits

### Posterior Visualization
- `plot_gamma_omega_posteriors_all_animals.py` - Creates PDF files with gamma and omega posterior distributions for each animal (10 ILDs × 3 ABLs grid)
- `plot_gamma_omega_posteriors_all_animals_by_stimulus.py` - Plots gamma and omega posterior distributions aggregated across all animals for each stimulus condition

### Utility Scripts
- `led_off_gamma_omega_pdf_utils.py` - PDF/CDF functions for gamma/omega models, including vectorized implementations.
- `gamma_omega_alpha_utils.py` - Shared helpers for loading condition-fit posterior means, aggregating Gamma/Omega across animals, and evaluating the alpha interaction model.

### Data Files
- `each_animal_cond_fit_gama_omega_pkl_files/` - VBMC fit results for each condition (ABL, ILD) per animal
- `each_animal_cond_fit_gama_omega_pkl_files_LAPSES/` - Condition fits with lapse parameters
