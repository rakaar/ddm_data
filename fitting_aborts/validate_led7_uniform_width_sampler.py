# %%
"""Check the log-width Metropolis kernel against one-dimensional quadrature."""

# %%
from pathlib import Path
import json
import os
import runpy

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("LED7_UNIFORM_WIDTH_OUTPUT_ROOT", str(
    SCRIPT_DIR / "numpyro_mcmc_led7_s8_s9_aggregate_no_trunc_uniform_lapse_fitted_width_outputs"))
os.environ["LED7_UNIFORM_WIDTH_VALIDATE_ONLY"] = "1"
os.environ["LED7_UNIFORM_WIDTH_SESSION_TYPES"] = "8"
namespace = runpy.run_path(str(SCRIPT_DIR / "fit_led7_s8_s9_uniform_fitted_width.py"))
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy import stats

# Tiny synthetic times exercise both abort and censor branches, including
# observations on either side of a proposed width. The conditional target is
# independently integrated across each density jump and survival kink.
data = {"on_abort_t": jnp.array([.03, .12, .4]), "on_abort_t_led": jnp.array([.1, .15, .2]),
        "off_abort_t": jnp.array([.08, .25, .6]), "on_censored_t_stim": jnp.array([.2, .5, 1.3]),
        "on_censored_t_led": jnp.array([.1, .3, .7]), "off_censored_t_stim": jnp.array([.15, .7, 1.8])}
params = {"V_A_base": 1.6, "V_A_post_LED": 3.0, "theta_A": 2.5,
          "del_a_minus_del_LED": .04, "del_m_plus_del_LED": .05, "lapse_prob": .3}
logs = namespace["proactive_logterms"](params, data)
likelihood_checks = []
for width in (.08, .2, .75, 2.2):
    reference = stats.uniform(scale=width)
    # Preserve the model's explicit open upper endpoint; SciPy's PDF includes
    # the endpoint. This zero-measure convention matters in a boundary test.
    lapse_logs = (np.where(np.asarray(data["on_abort_t"]) < width, reference.logpdf(data["on_abort_t"]), -np.inf),
                  reference.logsf(data["on_censored_t_stim"]),
                  np.where(np.asarray(data["off_abort_t"]) < width, reference.logpdf(data["off_abort_t"]), -np.inf),
                  reference.logsf(data["off_censored_t_stim"]))
    independent = sum(np.logaddexp(np.log1p(-params["lapse_prob"]) + np.asarray(proactive),
                                  np.log(params["lapse_prob"]) + lapse).sum()
                      for proactive, lapse in zip(logs, lapse_logs))
    actual = float(namespace["width_loglike"](width, params["lapse_prob"], logs, data))
    np.testing.assert_allclose(actual, independent, atol=1e-9, rtol=0)
    likelihood_checks.append({"width_s": width, "independent_log_likelihood": float(independent), "log_likelihood": actual})
target = jax.jit(lambda w: namespace["width_loglike"](w, params["lapse_prob"], logs, data) + namespace["width_logprior"](w))
offset = max(float(target(w)) for w in np.linspace(.01, 4.99, 500))
breaks = sorted(set([0, .02, 2.2, 5] + [.03, .12, .4, .08, .25, .6, .2, .5, 1.3, .15, .7, 1.8]))

def integral(power, low, high):
    points = [low] + [value for value in breaks if low < value < high] + [high]
    return sum(quad(lambda w: w ** power * np.exp(float(target(w)) - offset), a, b,
                    epsabs=1e-10, epsrel=1e-9)[0] for a, b in zip(points[:-1], points[1:]))

mass = integral(0, 0, 5)
expected_mean = integral(1, 0, 5) / mass
expected_sd = np.sqrt(integral(2, 0, 5) / mass - expected_mean ** 2)

def transition(width, key):
    result = namespace["width_metropolis"](key, {"width_lapse": width}, params, data)
    return result["width_lapse"], result["width_lapse"]

_, draws = jax.jit(lambda: jax.lax.scan(transition, jnp.asarray(.7), random.split(random.PRNGKey(614), 30000)))()
draws = np.asarray(draws)[1000:]
from numpyro.diagnostics import effective_sample_size
ess = float(effective_sample_size(draws[None, :]))
mean_se = expected_sd / np.sqrt(ess)
assert abs(draws.mean() - expected_mean) < max(6 * mean_se, .01)
cdf_checks = []
for threshold in (.2, .5, 1, 2):
    expected_cdf = integral(0, 0, threshold) / mass
    sample_cdf = float(np.mean(draws <= threshold))
    assert abs(expected_cdf - sample_cdf) < .025
    cdf_checks.append({"width_s": threshold, "quadrature_cdf": expected_cdf, "sample_cdf": sample_cdf})
audit = {"status": "passed", "method": "conditional posterior quadrature split at every data discontinuity",
         "n_retained": len(draws), "effective_sample_size": ess, "quadrature_mean_s": expected_mean,
         "sample_mean_s": float(draws.mean()), "mean_mcse_s": mean_se,
         "cdf_checks": cdf_checks, "four_term_likelihood_vs_scipy": likelihood_checks,
         "log_width_jacobian": "included and validated against physical-width quadrature"}
out = namespace["OUTPUT_ROOT"] / "width_sampler_validation.json"
out.write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
