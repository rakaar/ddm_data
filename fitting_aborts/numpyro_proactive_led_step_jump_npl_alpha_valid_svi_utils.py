# %%
"""SVI helpers for valid-trial LED step-jump + NPL+alpha pilot fits."""

# %%
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ANIMAL_FIT_DIR = REPO_DIR / "fit_animal_by_animal"
for import_dir in [SCRIPT_DIR, ANIMAL_FIT_DIR]:
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import numpyro_npl_alpha_svi_utils as npl_alpha
import numpyro_proactive_led_step_jump_npl_alpha_utils as combined_likelihood


# %%
# =============================================================================
# Re-export the established NPL+alpha parameter metadata and guide helpers
# =============================================================================
GLOBAL_PARAM_NAMES = npl_alpha.GLOBAL_PARAM_NAMES
GLOBAL_BOUNDS = npl_alpha.GLOBAL_BOUNDS
DELAY_BOUNDS = npl_alpha.DELAY_BOUNDS

sample_trapezoid = npl_alpha.sample_trapezoid
sample_trapezoid_vector = npl_alpha.sample_trapezoid_vector
make_guide = npl_alpha.make_guide
clip_init_to_hard_bounds = npl_alpha.clip_init_to_hard_bounds
posterior_samples_to_frame = npl_alpha.posterior_samples_to_frame
tree_all_finite = npl_alpha.tree_all_finite

FIXED_PROACTIVE_PARAM_NAMES = [
    "V_A_base",
    "V_A_post_LED",
    "theta_A",
    "del_a_minus_del_LED",
    "del_m_plus_del_LED",
    "lapse_prob",
    "beta_lapse",
]

VALID_CATEGORIES = {"off", "on_bilateral"}


# %%
# =============================================================================
# Fixed-proactive valid-trial model
# =============================================================================
def proactive_led_npl_alpha_valid_model(
    data,
    n_conditions,
    fixed_proactive_params,
    category,
    K_max=10,
    n_quad=64,
):
    """Fit NPL+alpha parameters with proactive/lapse parameters held fixed."""
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"category must be one of {sorted(VALID_CATEGORIES)}, got {category!r}."
        )

    params = {
        name: sample_trapezoid(
            name,
            GLOBAL_BOUNDS[name]["hard"],
            GLOBAL_BOUNDS[name]["plausible"],
        )
        for name in GLOBAL_PARAM_NAMES
    }
    params["t_E_aff"] = sample_trapezoid_vector(
        "t_E_aff",
        n_conditions,
        DELAY_BOUNDS["hard"],
        DELAY_BOUNDS["plausible"],
    )

    for name in FIXED_PROACTIVE_PARAM_NAMES:
        params[name] = jnp.asarray(fixed_proactive_params[name], dtype=jnp.float64)

    likelihood_data = dict(data)
    likelihood_data["t_E_aff"] = params["t_E_aff"][data["condition_id"]]

    if category == "off":
        loglike = combined_likelihood.led_off_npl_alpha_valid_loglike_jax(
            params,
            likelihood_data,
            K_max=K_max,
            include_lapse=True,
        )
    else:
        loglike = combined_likelihood.led_on_npl_alpha_valid_loglike_jax(
            params,
            likelihood_data,
            K_max=K_max,
            n_quad=n_quad,
            include_lapse=True,
        )

    numpyro.factor("valid_rt_choice_loglike", loglike)


# %%
# =============================================================================
# Small shared helpers for the driver and diagnostics
# =============================================================================
def bounded_logit(values, hard_low, hard_high, eps=1e-6):
    values = np.asarray(values, dtype=float)
    width = hard_high - hard_low
    values = np.clip(values, hard_low + eps * width, hard_high - eps * width)
    unit_values = (values - hard_low) / width
    return np.log(unit_values / (1.0 - unit_values))


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)),
        tree,
    )
