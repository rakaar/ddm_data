# %%
"""Check uniform support, censoring, mixture mass, and likelihood gradients."""

# %%
from pathlib import Path
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.stats import uniform

import numpyro_proactive_led_step_jump_no_trunc_uniform_lapse_svi_utils as model

OUTPUT_DIR = Path(__file__).resolve().parent / "numpyro_svi_led7_s8_s9_aggregate_no_trunc_uniform_lapse_0_2p2s_outputs"
T = model.UNIFORM_END_S
params = dict(model.DEFAULT_INIT_VALUES)
reference = uniform(loc=0.0, scale=T)
grid = np.array([-0.1, 0.0, 0.1, 0.8, T - 1e-8, T + 1e-8, 3.0])
np.testing.assert_allclose(np.exp(model.uniform_lapse_logpdf_jax(grid)), reference.pdf(grid), atol=1e-14)
np.testing.assert_allclose(np.exp(model.uniform_lapse_logsurvival_jax(grid)), reference.sf(grid), atol=1e-14)
assert np.isneginf(float(model.uniform_lapse_logsurvival_jax(T)))
assert float(model.uniform_lapse_logsurvival_jax(0.0)) == 0.0
assert np.isclose(quad(lambda t: float(np.exp(model.uniform_lapse_logpdf_jax(t))), 0.0, T)[0], 1.0)

# %%
# Independent probability-space reference for all four likelihood contributions.
data = {
    "on_abort_t": jnp.array([0.05, 0.3, 0.8, 2.3]),
    "on_abort_t_led": jnp.array([0.1, 0.2, 0.4, 0.6]),
    "on_censored_t_stim": jnp.array([0.2, 0.6, 1.2, 2.3]),
    "on_censored_t_led": jnp.array([0.1, 0.2, 0.4, 0.6]),
    "off_abort_t": jnp.array([0.05, 0.3, 0.8, 2.3]),
    "off_censored_t_stim": jnp.array([0.2, 0.6, 1.2, 2.3]),
}
common = [params[k] for k in model.PARAM_NAMES[:5]]
off_common = [common[i] for i in (0, 2, 3, 4)]
p = params["lapse_prob"]
proactive = (
    model.led_on_pdf_jax(data["on_abort_t"], data["on_abort_t_led"], *common),
    1.0 - model.led_on_cdf_jax(data["on_censored_t_stim"], data["on_censored_t_led"], *common),
    model.led_off_pdf_jax(data["off_abort_t"], *off_common),
    1.0 - model.led_off_cdf_jax(data["off_censored_t_stim"], *off_common),
)
lapses = (
    reference.pdf(data["on_abort_t"]), reference.sf(data["on_censored_t_stim"]),
    reference.pdf(data["off_abort_t"]), reference.sf(data["off_censored_t_stim"]),
)
reference_loglike = sum(np.log((1.0 - p) * np.asarray(a) + p * b).sum() for a, b in zip(proactive, lapses))
actual_loglike = float(model.proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, data))
np.testing.assert_allclose(actual_loglike, reference_loglike, atol=1e-10, rtol=0.0)
gradient = jax.grad(model.proactive_led_step_jump_no_trunc_lapse_loglike_jax)(params, data)
assert model.tree_all_finite(gradient)

# %%
# Density up to censoring plus survival must preserve one trial's probability.
mass_rows = []
for led in (0, 1):
    for t_stim in (0.2, 0.7, 1.5, 2.3):
        t_led = 0.1
        def density(t):
            a = (model.led_on_pdf_jax(t, t_led, *common) if led else model.led_off_pdf_jax(t, *off_common))
            return (1.0 - p) * float(a) + p * reference.pdf(t)
        cdf = (model.led_on_cdf_jax(t_stim, t_led, *common, n_quad=256)
               if led else model.led_off_cdf_jax(t_stim, *off_common))
        survival = (1.0 - p) * (1.0 - float(cdf)) + p * reference.sf(t_stim)
        points = [v for v in (0.09, 0.14, T) if 0.0 < v < t_stim]
        mass = quad(density, 0.0, t_stim, points=points, epsabs=1e-8)[0] + survival
        np.testing.assert_allclose(mass, 1.0, atol=2e-6, rtol=0.0)
        mass_rows.append({"LED_trial": led, "t_stim": t_stim, "density_mass_plus_survival": mass})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
audit = {
    "status": "passed", "uniform_interval_s": [0.0, T],
    "log_likelihood": actual_loglike, "independent_log_likelihood": float(reference_loglike),
    "gradients_finite": True, "mass_checks": mass_rows,
    "checks": ["SciPy density and survival", "support boundaries", "unit lapse mass", "four-term likelihood", "finite gradients", "censoring mass"],
}
(OUTPUT_DIR / "likelihood_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
