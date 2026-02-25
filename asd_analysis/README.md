# ASD Analysis Scripts

- `explore_asd_dataset.py` - Filters ASD dataset by genotype/quality criteria and generates exploratory RTD/CDF/quantile plots.
- `explore_valid_trials.py` - Utility exploration script for valid-trial subsets in the ASD dataset.
- `asd_wt_all_animals_rtd_cdf.py` - Computes ASD-WT RTD histograms and CDF summaries across animals by ABL and |ILD|.
- `asd_wt_scaling_quantiles_not_RTD_for_paper_fig1.py` - Quantile-based ASD-WT scaling analysis (per-animal slopes, scaled/unscaled quantile figures).
- `asd_wt_rtd_scaling_for_paper.py` - ASD-WT adaptation of RTD scaling pipeline (aggregate RTD, QQ-derived scaling, rescaled RTDs).
- `asd_wt_rtd_scaling_for_paper_Q_scale_match.py` - ASD-WT RTD scaling variant aligned to quantile-style slope estimation, with debug comparisons.
- `asd_wt_rtd_scaling_for_paper_quantile_avg_match.py` - ASD-WT RTD scaling variant using per-animal quantile averaging to match quantile-analysis assumptions.
