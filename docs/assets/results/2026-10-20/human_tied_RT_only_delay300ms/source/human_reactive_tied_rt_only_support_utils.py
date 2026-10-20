# %%
"""Use the requested prior with an SVI guide confined to causal delay values.

The fitted model is unchanged: its delay prior is Uniform(0, delay_upper).
For a pure reactive model the posterior is zero when delay >= minimum RT.
Only the guide's prototype uses that smaller interval for its transform.
SVI still evaluates the actual model's Uniform(0, delay_upper) log density.
"""
from human_reactive_tied_rt_only_utils import (
    ROOT, PARAM_NAMES, INITIAL, sha256, load_trials, load_all_trials, group_data,
    bound_cdf, bound_density, retained_mass, log_likelihood, reactive_model,
)

RT_LOWER, RT_UPPER = 0., 1.8


def guide_support_model(data, delay_upper, K_max=40):
    """Build AutoMultivariateNormal's transform, without changing the fitted prior."""
    return reactive_model(data, data['guide_delay_upper_s'], K_max)
