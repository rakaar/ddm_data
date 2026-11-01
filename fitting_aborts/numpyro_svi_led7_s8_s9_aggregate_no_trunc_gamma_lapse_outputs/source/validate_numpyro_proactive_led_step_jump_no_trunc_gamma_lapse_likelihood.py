# %%
"""Validate Gamma(shape, scale), location=0: density, censoring, and shape gradients."""

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
from scipy.stats import gamma

import numpyro_proactive_led_step_jump_no_trunc_gamma_lapse_svi_utils as model

OUTPUT_DIR = Path(__file__).resolve().parent / "numpyro_svi_led7_s8_s9_aggregate_no_trunc_gamma_lapse_outputs"
grid = np.array([-0.1, 0.0, 1e-8, 0.001, 0.05, 0.2, 0.8, 2.2, 3.0])
pdf = jax.jit(model.gamma_lapse_logpdf_jax)
survival = jax.jit(model.gamma_lapse_logsurvival_jax)
lapse_checks = []
for shape, scale in ((0.05,0.001), (0.3,0.08), (1.0,0.2), (2.5,0.04), (10.0,0.02), (49.5,0.0011), (1.2,4.9)):
    reference = gamma(shape, loc=0.0, scale=scale)
    actual_pdf = np.asarray(pdf(grid, shape, scale))
    actual_sf = np.asarray(survival(grid, shape, scale))
    expected_sf = np.maximum(reference.logsf(grid), np.log(model.SURVIVAL_FLOOR))
    np.testing.assert_allclose(actual_pdf, reference.logpdf(grid), atol=1e-8, rtol=1e-10)
    np.testing.assert_allclose(actual_sf, expected_sf, atol=1e-8, rtol=1e-10)
    np.testing.assert_allclose(model.gamma_lapse_cdf_jax(grid,shape,scale), reference.cdf(grid), atol=1e-12, rtol=1e-10)
    upper = float(reference.ppf(1.0-1e-12))
    points = [t for t in ((shape-1.0)*scale,shape*scale,(shape+np.sqrt(shape))*scale) if 0.0<t<upper]
    mass = quad(lambda t: float(np.exp(pdf(t,shape,scale))), 0.0, upper,
                points=points, epsabs=1e-8, limit=300)[0]
    np.testing.assert_allclose(mass, 1.0, atol=1e-7, rtol=0.0)
    grad = jax.grad(lambda values: model.gamma_lapse_logpdf_jax(0.1,*values)
                    + model.gamma_lapse_logsurvival_jax(0.6,*values))(jnp.array([shape,scale]))
    assert np.isfinite(np.asarray(grad)).all()
    # Check the incomplete-gamma shape derivative against independent SciPy differences.
    h = 1e-5*shape
    time = shape*scale
    difference = (gamma.logsf(time,shape+h,scale=scale)-gamma.logsf(time,shape-h,scale=scale))/(2*h)
    actual_grad = float(jax.grad(lambda a: model.gamma_lapse_logsurvival_jax(time,a,scale))(shape))
    np.testing.assert_allclose(actual_grad,difference,atol=1e-7,rtol=2e-6)
    lapse_checks.append({"shape":shape,"scale_s":scale,"integrated_mass":mass,
        "shape_gradient":actual_grad,"scipy_shape_finite_difference":float(difference),"gradients_finite":True})

# %%
params = dict(model.DEFAULT_INIT_VALUES)
data = {"on_abort_t":jnp.array([0.001,0.05,0.3,0.8]),
        "on_abort_t_led":jnp.array([0.1,0.2,0.4,0.6]),
        "on_censored_t_stim":jnp.array([0.2,0.6,1.2,2.3]),
        "on_censored_t_led":jnp.array([0.1,0.2,0.4,0.6]),
        "off_abort_t":jnp.array([0.001,0.05,0.3,0.8]),
        "off_censored_t_stim":jnp.array([0.2,0.6,1.2,2.3])}
common = [params[k] for k in model.PARAM_NAMES[:5]]
off_common = [common[i] for i in (0,2,3,4)]
checks = []
for shape,scale in ((0.3,0.08),(1.0,0.2),(2.5,0.04)):
    params.update(shape_lapse=shape,scale_lapse=scale)
    reference = gamma(shape,scale=scale)
    p = params["lapse_prob"]
    proactive = (model.led_on_pdf_jax(data["on_abort_t"],data["on_abort_t_led"],*common),
        1.0-model.led_on_cdf_jax(data["on_censored_t_stim"],data["on_censored_t_led"],*common),
        model.led_off_pdf_jax(data["off_abort_t"],*off_common),
        1.0-model.led_off_cdf_jax(data["off_censored_t_stim"],*off_common))
    lapses = (reference.pdf(data["on_abort_t"]),reference.sf(data["on_censored_t_stim"]),
              reference.pdf(data["off_abort_t"]),reference.sf(data["off_censored_t_stim"]))
    expected = sum(np.log((1.0-p)*np.asarray(a)+p*b).sum() for a,b in zip(proactive,lapses))
    actual = float(model.proactive_led_step_jump_no_trunc_lapse_loglike_jax(params,data))
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0.0)
    assert model.tree_all_finite(jax.grad(model.proactive_led_step_jump_no_trunc_lapse_loglike_jax)(params,data))
    if shape == 1.0:
        exponential_params = {name:params[name] for name in model.exponential.PARAM_NAMES if name!='beta_lapse'}
        exponential_params['beta_lapse'] = 1.0/scale
        exponential_ll = float(model.exponential.proactive_led_step_jump_no_trunc_lapse_loglike_jax(exponential_params,data))
        np.testing.assert_allclose(actual,exponential_ll,atol=1e-10,rtol=0.0)
    checks.append({"shape":shape,"scale_s":scale,"scipy_log_likelihood":float(expected),"jax_log_likelihood":actual})

mass_checks = []
for led in (0,1):
    for t_stim in (0.2,0.7,1.5,2.3):
        def density(t):
            a = model.led_on_pdf_jax(t,0.1,*common) if led else model.led_off_pdf_jax(t,*off_common)
            return (1.0-p)*float(a)+p*reference.pdf(t)
        cdf = (model.led_on_cdf_jax(t_stim,0.1,*common,n_quad=256) if led
               else model.led_off_cdf_jax(t_stim,*off_common))
        mass = quad(density,0.0,t_stim,points=[t for t in (0.09,0.14) if t<t_stim],epsabs=1e-8)[0]
        mass += (1.0-p)*(1.0-float(cdf))+p*reference.sf(t_stim)
        np.testing.assert_allclose(mass,1.0,atol=2e-6,rtol=0.0)
        mass_checks.append({"LED_trial":led,"t_stim":t_stim,"density_mass_plus_survival":mass})

OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
audit = {"status":"passed","lapse_support_s":[0.0,None],"location_s":0.0,
         "survival_floor":model.SURVIVAL_FLOOR,"lapse_checks":lapse_checks,
         "likelihood_checks":checks,"mass_checks":mass_checks,
         "checks":["SciPy density/CDF/survival","zero support boundary","unit mass",
                   "shape below and above one","incomplete-gamma shape derivative","finite gradients",
                   "four-term likelihood","shape one equals exponential","censoring mass"]}
(OUTPUT_DIR/'likelihood_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
