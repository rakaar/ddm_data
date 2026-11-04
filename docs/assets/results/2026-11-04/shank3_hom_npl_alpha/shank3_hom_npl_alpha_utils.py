# %% HOM output/reference settings; reuse the exact verified pure WT model
"""Only paths differ. The model reads fixed proactive/lapse values from its data."""
from shank3_wt_npl_alpha_utils import (
    ROOT, ABLS, ILDS, K_DEFAULT, GLOBALS, BLOCKS, PRIORS, INITIAL,
    expand_parameters, log_likelihood, model, sample_matrix, parameter_names,
    loss_window_decision, sha256,
)

OUT = ROOT/'asd_analysis/shank3_cohort1_hom_npl_alpha_condition_delay'
REFERENCE = ROOT/'asd_analysis/shank3_cohort1_hom_gamma_omega'
