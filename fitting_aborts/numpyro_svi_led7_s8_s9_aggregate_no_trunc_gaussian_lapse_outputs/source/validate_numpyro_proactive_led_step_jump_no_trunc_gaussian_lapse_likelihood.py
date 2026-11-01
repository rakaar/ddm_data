# %%
"""Validate a Gaussian lapse clock truncated at fixation start, including negative mu."""

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
from scipy.stats import truncnorm

import numpyro_proactive_led_step_jump_no_trunc_gaussian_lapse_svi_utils as model

OUTPUT_DIR = Path(__file__).resolve().parent / "numpyro_svi_led7_s8_s9_aggregate_no_trunc_gaussian_lapse_outputs"
grid = np.array([-0.1, 0.0, 0.001, 0.05, 0.2, 0.8, 2.2, 3.0])
lapse_checks = []
for mu, sigma in ((-0.4, 0.05), (0.0, 0.2), (0.2, 0.08), (-5.0, 0.9), (0.05, 0.001)):
    reference = truncnorm(-mu / sigma, np.inf, loc=mu, scale=sigma)
    actual_pdf = np.asarray(model.gaussian_lapse_logpdf_jax(grid, mu, sigma))
    actual_sf = np.asarray(model.gaussian_lapse_logsurvival_jax(grid, mu, sigma))
    np.testing.assert_allclose(actual_pdf, reference.logpdf(grid), atol=1e-8, rtol=1e-10)
    np.testing.assert_allclose(actual_sf, reference.logsf(grid), atol=1e-8, rtol=1e-10)
    assert float(model.gaussian_lapse_logsurvival_jax(0.0, mu, sigma)) == 0.0
    mean, var = reference.stats(moments="mv")
    upper = max(2.2, float(mean) + 15.0 * float(np.sqrt(var)))
    points = [v for v in (max(mu, 0.0), float(mean), float(mean + np.sqrt(var))) if 0.0 < v < upper]
    mass = quad(lambda t: float(np.exp(model.gaussian_lapse_logpdf_jax(t, mu, sigma))),
                0.0, upper, points=points, epsabs=1e-9)[0]
    np.testing.assert_allclose(mass, 1.0, atol=1e-8, rtol=0.0)
    grad = jax.grad(lambda values: model.gaussian_lapse_logpdf_jax(0.1, *values)
                    + model.gaussian_lapse_logsurvival_jax(0.6, *values))(jnp.array([mu, sigma]))
    assert np.isfinite(np.asarray(grad)).all()
    lapse_checks.append({"mu": mu, "sigma": sigma, "integrated_mass": mass,
        "retained_mean": float(mean), "retained_sd": float(np.sqrt(var)), "gradients_finite": True})

# %%
params = dict(model.DEFAULT_INIT_VALUES)
data = {"on_abort_t": jnp.array([0.05, 0.3, 0.8, 2.3]),
        "on_abort_t_led": jnp.array([0.1, 0.2, 0.4, 0.6]),
        "on_censored_t_stim": jnp.array([0.2, 0.6, 1.2, 2.3]),
        "on_censored_t_led": jnp.array([0.1, 0.2, 0.4, 0.6]),
        "off_abort_t": jnp.array([0.05, 0.3, 0.8, 2.3]),
        "off_censored_t_stim": jnp.array([0.2, 0.6, 1.2, 2.3])}
common = [params[k] for k in model.PARAM_NAMES[:5]]
off_common = [common[i] for i in (0, 2, 3, 4)]
p = params["lapse_prob"]
reference = truncnorm(-params["mu_lapse"] / params["sigma_lapse"], np.inf,
                      loc=params["mu_lapse"], scale=params["sigma_lapse"])
proactive = (model.led_on_pdf_jax(data["on_abort_t"], data["on_abort_t_led"], *common),
    1.0 - model.led_on_cdf_jax(data["on_censored_t_stim"], data["on_censored_t_led"], *common),
    model.led_off_pdf_jax(data["off_abort_t"], *off_common),
    1.0 - model.led_off_cdf_jax(data["off_censored_t_stim"], *off_common))
lapses = (reference.pdf(data["on_abort_t"]), reference.sf(data["on_censored_t_stim"]),
          reference.pdf(data["off_abort_t"]), reference.sf(data["off_censored_t_stim"]))
reference_loglike = sum(np.log((1.0-p)*np.asarray(a)+p*b).sum() for a,b in zip(proactive,lapses))
actual_loglike = float(model.proactive_led_step_jump_no_trunc_lapse_loglike_jax(params,data))
np.testing.assert_allclose(actual_loglike, reference_loglike, atol=1e-10,rtol=0.0)
assert model.tree_all_finite(jax.grad(model.proactive_led_step_jump_no_trunc_lapse_loglike_jax)(params,data))

mass_checks = []
for led in (0,1):
    for t_stim in (0.2,0.7,1.5,2.3):
        def density(t):
            a = model.led_on_pdf_jax(t,0.1,*common) if led else model.led_off_pdf_jax(t,*off_common)
            return (1.0-p)*float(a)+p*reference.pdf(t)
        cdf = (model.led_on_cdf_jax(t_stim,0.1,*common,n_quad=256) if led
               else model.led_off_cdf_jax(t_stim,*off_common))
        survival = (1.0-p)*(1.0-float(cdf))+p*reference.sf(t_stim)
        mass = quad(density,0.0,t_stim,points=[v for v in (0.09,0.14) if v<t_stim],epsabs=1e-8)[0]+survival
        np.testing.assert_allclose(mass,1.0,atol=2e-6,rtol=0.0)
        mass_checks.append({"LED_trial":led,"t_stim":t_stim,"density_mass_plus_survival":mass})

OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
audit = {"status":"passed","lapse_support_s":[0.0,None],"mu_may_be_negative":True,
         "lapse_checks":lapse_checks,"reference_log_likelihood":float(reference_loglike),
         "actual_log_likelihood":actual_loglike,"mass_checks":mass_checks,
         "checks":["SciPy log density and log survival","zero support boundary","unit mass",
                   "negative mu and narrow sigma","finite gradients","four-term likelihood","censoring mass"]}
(OUTPUT_DIR/'likelihood_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
