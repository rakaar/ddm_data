# Lapses Supplementary Figure (2 × 4)

## Template script
- **`lapses_figure_using_template.py`** — Assembles all 8 panels into a single 2×4 figure using `figure_template.py`
- Outputs: `lapses_supp_figure_2x4.png`, `lapses_supp_figure_2x4.pdf`

## Layout

### Top row
| Panel | Description | Data source (pkl) | Producer script |
|-------|-------------|-------------------|-----------------|
| (0,0) | Lapse rate distribution (sorted, with median) | `supp_lapses_distr_plot.pkl` | `lapses_supp_figure_save_data.py` |
| (0,1) | ΔLL: NPL − (IPL + lapses) vs lapse rate | `lapse_rate_loglike_diff_data.pkl` | `lapses_likelihood_plot_for_paper.py` |
| (0,2) | ΔLL: (NPL + lapses) − (IPL + lapses) vs lapse rate | `lapse_rate_loglike_diff_data.pkl` | `lapses_likelihood_plot_for_paper.py` |
| (0,3) | Γ: condition-fit data vs NPL theory (all ABLs combined) | `gamma_sep_by_median_lapse_rate_data.pkl` + `fit_each_condn/norm_gamma_fig2_data.pkl` | `lapses_supp_figure_save_data.py` + `plot_gamma_for_norm_lapse_for_paper.py` |

### Bottom row
| Panel | Description | Data source (pkl) | Producer script |
|-------|-------------|-------------------|-----------------|
| (1,0) | ℓ (rate_norm_l): NPL vs NPL+lapse | `params_npl_npl_plus_lapse_plot_data.pkl` | `params_npl_npl_plus_lapse_ordered_for_paper.py` |
| (1,1) | λ' (rate_lambda): NPL vs NPL+lapse | `params_npl_npl_plus_lapse_plot_data.pkl` | `params_npl_npl_plus_lapse_ordered_for_paper.py` |
| (1,2) | θ_E: NPL vs NPL+lapse | `params_npl_npl_plus_lapse_plot_data.pkl` | `params_npl_npl_plus_lapse_ordered_for_paper.py` |
| (1,3) | T₀: NPL vs NPL+lapse | `params_npl_npl_plus_lapse_plot_data.pkl` | `params_npl_npl_plus_lapse_ordered_for_paper.py` |

## PKL files summary
- `supp_lapses_distr_plot.pkl` — lapse rate per animal + median
- `lapse_rate_loglike_diff_data.pkl` — log-likelihood differences (2 comparisons)
- `gamma_sep_by_median_lapse_rate_data.pkl` — condition-fit gamma split by median lapse rate
- `fit_each_condn/norm_gamma_fig2_data.pkl` — NPL theoretical gamma
- `params_npl_npl_plus_lapse_plot_data.pkl` — 4 param comparisons (NPL vs NPL+lapse, ordered by lapse prob)

## Additional input data
- `vanilla_norm_lapse_loglike_comparison_v2.csv` — log-likelihood comparison table
- `lapse_parameters_all_animals.pkl` — lapse params from vanilla+lapse and norm+lapse fits
- `oct_9_10_norm_lapse_model_fit_files/*.pkl` — norm+lapse VBMC fit results per animal
- `results_{batch}_animal_{id}.pkl` — original norm tied results per animal