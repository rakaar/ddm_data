# %%
"""Fit the sharp Uniform(0, width) lapse model with NUTS-within-Metropolis.

Width has a discontinuous likelihood and must not receive a pathwise gradient.
The six smooth parameters use NUTS; log-width uses exact Metropolis updates.
"""

# %%
# Editable settings. The earlier fixed-width and other-family fits are read only.
from pathlib import Path
import hashlib
import json
import os
import pickle
import shutil
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR / "numpyro_svi_led7_s8_s9_aggregate_no_trunc_exp_lapse_outputs"
INITIAL_ROOT = SCRIPT_DIR / "numpyro_svi_led7_s8_s9_aggregate_no_trunc_gamma_lapse_outputs"
OUTPUT_ROOT = Path(os.environ.get("LED7_UNIFORM_WIDTH_OUTPUT_ROOT", str(
    SCRIPT_DIR / "numpyro_mcmc_led7_s8_s9_aggregate_no_trunc_uniform_lapse_fitted_width_outputs")))
SESSION_TYPES = tuple(map(int, os.environ.get("LED7_UNIFORM_WIDTH_SESSION_TYPES", "8,9").split(",")))
NUM_WARMUP = int(os.environ.get("LED7_UNIFORM_WIDTH_WARMUP", "1000"))
NUM_SAMPLES = int(os.environ.get("LED7_UNIFORM_WIDTH_SAMPLES", "2000"))
NUM_CHAINS = int(os.environ.get("LED7_UNIFORM_WIDTH_CHAINS", "4"))
SAMPLE_BATCH = int(os.environ.get("LED7_UNIFORM_WIDTH_SAMPLE_BATCH", "500"))
WIDTH_MH_STEPS = 32
WIDTH_BOUNDS = {"hard": (0.0, 5.0), "plausible": (0.02, 2.2)}
SEED = 71
VALIDATE_ONLY = os.environ.get("LED7_UNIFORM_WIDTH_VALIDATE_ONLY", "0") == "1"

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import numpyro
from numpyro.diagnostics import summary as mcmc_summary
from numpyro.infer import HMCGibbs, MCMC, NUTS, init_to_value
import numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils as exponential
import numpyro_proactive_led_step_jump_no_trunc_uniform_lapse_svi_utils as uniform

PARAM_BOUNDS = {**uniform.PARAM_BOUNDS, "width_lapse": WIDTH_BOUNDS}
PARAM_NAMES = list(PARAM_BOUNDS)
# The baseline quadrature helper caches arrays; populate it outside any JIT.
exponential.step_jump._legendre_nodes_weights(64)


# %%
def likelihood_data(frame):
    data = {}
    for led, prefix in ((0, "off"), (1, "on")):
        group = frame.loc[frame.LED_trial.eq(led)]
        abort = group.loc[group.likelihood_role.eq("abort")]
        censored = group.loc[group.likelihood_role.eq("censored")]
        data[f"{prefix}_abort_t"] = jnp.asarray(abort.timed_fix.to_numpy(float))
        data[f"{prefix}_censored_t_stim"] = jnp.asarray(censored.intended_fix.to_numpy(float))
        if led:
            data["on_abort_t_led"] = jnp.asarray(abort.effective_scheduled_onset.to_numpy(float))
            data["on_censored_t_led"] = jnp.asarray(censored.effective_scheduled_onset.to_numpy(float))
    return data


def proactive_logterms(params, data):
    common = tuple(params[name] for name in PARAM_NAMES[:5])
    off_common = (common[0], common[2], common[3], common[4])
    return (
        exponential._log_positive(exponential.led_on_pdf_jax(data["on_abort_t"], data["on_abort_t_led"], *common)),
        exponential._log_positive(1.0 - exponential.led_on_cdf_jax(
            data["on_censored_t_stim"], data["on_censored_t_led"], *common, n_quad=64)),
        exponential._log_positive(exponential.led_off_pdf_jax(data["off_abort_t"], *off_common)),
        exponential._log_positive(1.0 - exponential.led_off_cdf_jax(data["off_censored_t_stim"], *off_common)),
    )


def width_loglike(width, p, logs, data):
    lapse_terms = (
        uniform.uniform_lapse_logpdf_jax(data["on_abort_t"], width),
        uniform.uniform_lapse_logsurvival_jax(data["on_censored_t_stim"], width),
        uniform.uniform_lapse_logpdf_jax(data["off_abort_t"], width),
        uniform.uniform_lapse_logsurvival_jax(data["off_censored_t_stim"], width),
    )
    return sum(jnp.sum(exponential.log_probability_mixture_jax(lp, ll, p))
               for lp, ll in zip(logs, lapse_terms))


def width_logprior(width):
    return exponential.step_jump.trapezoidal_logpdf_jax(
        width, WIDTH_BOUNDS["hard"][0], *WIDTH_BOUNDS["plausible"], WIDTH_BOUNDS["hard"][1])


def model(data):
    params = {name: exponential.step_jump.sample_trapezoid(name, bounds)
              for name, bounds in PARAM_BOUNDS.items()}
    numpyro.factor("abort_and_censor_loglike", uniform.proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, data))


def width_metropolis(rng_key, gibbs_sites, hmc_sites, data):
    # Cache the expensive proactive terms once for all width-only proposals.
    logs = proactive_logterms(hmc_sites, data)
    p = hmc_sites["lapse_prob"]

    def log_target(z):
        width = jnp.exp(z)
        # +z is the Jacobian for proposing symmetrically in log(width).
        return width_loglike(width, p, logs, data) + width_logprior(width) + z

    z = jnp.log(gibbs_sites["width_lapse"])
    initial = (z, log_target(z))

    def update(carry, key):
        z, logp = carry
        key_scale, key_move, key_accept = random.split(key, 3)
        # Symmetric mixture: fine local moves and occasional wider jumps.
        u = random.uniform(key_scale)
        scale = jnp.where(u < 0.7, 0.005, jnp.where(u < 0.95, 0.05, 0.5))
        proposed = z + scale * random.normal(key_move)
        proposed_logp = log_target(proposed)
        accept = jnp.log(random.uniform(key_accept)) < proposed_logp - logp
        return (jnp.where(accept, proposed, z), jnp.where(accept, proposed_logp, logp)), accept

    (z, _), _ = jax.lax.scan(update, initial, random.split(rng_key, WIDTH_MH_STEPS))
    return {"width_lapse": jnp.exp(z)}


# %%
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
for session_type in SESSION_TYPES:
    if session_type not in (8, 9):
        raise ValueError("Only session types 8 and 9 are authorized.")
    target_dir = OUTPUT_ROOT / f"session_type_{session_type}"
    if (target_dir / "posterior_samples.npz").exists():
        print(f"s{session_type}: saved posterior already exists; preserving it.", flush=True)
        continue
    target_dir.mkdir(exist_ok=True)
    source_dir = BASELINE_ROOT / f"session_type_{session_type}"
    manifest_path = source_dir / "source_row_manifest.csv"
    frame = pd.read_csv(manifest_path)
    frame = frame.loc[frame.likelihood_role.isin(("abort", "censored"))].copy()
    expected = 139547 if session_type == 8 else 93381
    assert len(frame) == expected
    assert sorted(frame.animal.unique()) == [90, 92, 93, 98, 99, 100, 102, 103]
    data = likelihood_data(frame)
    shutil.copy2(manifest_path, target_dir / manifest_path.name)
    initial_table = pd.read_csv(INITIAL_ROOT / f"session_type_{session_type}" / "main_fullrank_posterior_summary.csv").set_index("parameter")
    initial = {name: float(initial_table.loc[name, "mean"]) for name in uniform.PARAM_NAMES}
    initial["width_lapse"] = 0.2
    print(f"s{session_type}: {len(frame)} exact baseline rows; {jax.devices()}", flush=True)

    # Independent density/SF and full-likelihood checks, including censor times
    # beyond the fitted cutoff. No requirement that width exceed stimulus onset.
    test_t = np.array([-0.01, 0, 0.05, 0.2, 0.3, 2.2, 3.0])
    for width in (0.08, 0.2, 0.75, 2.2, 4.0):
        got_pdf = np.exp(np.asarray(uniform.uniform_lapse_logpdf_jax(test_t, width)))
        # scipy includes the endpoint, which has zero measure; this model uses <.
        want_pdf = stats.uniform.pdf(test_t, scale=width)
        want_pdf[test_t == width] = 0.0
        np.testing.assert_allclose(got_pdf, want_pdf, atol=1e-12)
        np.testing.assert_allclose(np.exp(np.asarray(uniform.uniform_lapse_logsurvival_jax(test_t, width))),
                                   stats.uniform.sf(test_t, scale=width), atol=1e-12)
    fixed = {name: initial[name] for name in uniform.PARAM_NAMES}
    loglike = jax.jit(lambda p: uniform.proactive_led_step_jump_no_trunc_lapse_loglike_jax(p, data))
    np.testing.assert_allclose(float(loglike(fixed)), float(loglike({**fixed, "width_lapse": 2.2})), atol=1e-9)
    cached_logs = jax.jit(proactive_logterms)(initial, data)
    conditional = jax.jit(lambda width: width_loglike(width, initial["lapse_prob"], cached_logs, data))
    for width in (0.08, 0.2, 0.75, 2.2):
        np.testing.assert_allclose(float(conditional(width)), float(loglike({**initial, "width_lapse": width})), atol=1e-8)
    validation = {"status": "passed", "pdf_survival_vs_scipy": True,
                  "fixed_2p2_likelihood_unchanged": True, "cached_likelihood_matches_direct": True,
                  "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "n_trials": len(frame)}
    (target_dir / "likelihood_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if VALIDATE_ONLY:
        continue

    # Find plausible, separated starts by profiling width, optimizing the six
    # smooth parameters conditionally. Width itself never receives a gradient.
    names = uniform.PARAM_NAMES
    bounds = [uniform.PARAM_BOUNDS[name]["hard"] for name in names]
    lower, upper = np.asarray(bounds).T
    center = (lower + upper) / 2
    scale = (upper - lower) / 2

    @jax.jit
    def profile_objective(y, width):
        params = dict(zip(names, center + scale * y))
        params["width_lapse"] = width
        prior = sum(exponential.step_jump.trapezoidal_logpdf_jax(params[name], low, *uniform.PARAM_BOUNDS[name]["plausible"], high)
                    for name, (low, high) in zip(names, bounds))
        return -uniform.proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, data) - prior - width_logprior(width)

    value_grad = jax.jit(jax.value_and_grad(profile_objective))
    width_grid = np.unique(np.r_[np.geomspace(0.002, 0.02, 5), np.geomspace(0.02, 0.8, 26), 1.0, 1.4, 2.2, 3.5])
    profile_rows, profile_starts = [], []
    y_start = (np.array([initial[name] for name in names]) - center) / scale
    started = perf_counter()
    for width in width_grid:
        fit = minimize(lambda y: tuple(np.asarray(v) for v in value_grad(y, width)), y_start,
                       jac=True, method="L-BFGS-B", bounds=[(-0.999999, 0.999999)] * 6,
                       options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-5})
        profile_rows.append({"width_s": width, "log_posterior": -float(fit.fun),
                             "optimizer_success": bool(fit.success), "n_eval": fit.nfev})
        profile_starts.append(dict(zip(names, center + scale * fit.x), width_lapse=float(width)))
        y_start = fit.x
        print(f"s{session_type}: width profile {width:.4f} s, log joint {-fit.fun:.3f}", flush=True)
    pd.DataFrame(profile_rows).to_csv(target_dir / "width_initialization_profile.csv", index=False)
    order = np.argsort([-row["log_posterior"] for row in profile_rows])
    best = profile_starts[order[0]]
    print(f"s{session_type}: initialize posterior sampling near width {best['width_lapse']:.4f} s", flush=True)

    # Sequential independent chains avoid multiplying GPU quadrature memory.
    chain_results = {name: [] for name in PARAM_NAMES}
    divergence_rows = []
    for chain in range(NUM_CHAINS):
        init = dict(best)
        init["width_lapse"] *= (0.9, 1.1, 0.95, 1.05)[chain % 4]
        kernel = HMCGibbs(NUTS(model, dense_mass=True, target_accept_prob=0.9,
                              init_strategy=init_to_value(values=init)),
                          gibbs_fn=lambda rng_key, gibbs_sites, hmc_sites: width_metropolis(
                              rng_key, gibbs_sites, hmc_sites, data), gibbs_sites=["width_lapse"])
        batch_size = min(SAMPLE_BATCH, NUM_SAMPLES)
        if NUM_SAMPLES % batch_size:
            raise ValueError("The requested sample count must be a multiple of the batch size.")
        sampler = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=batch_size, progress_bar=False)
        init_params = {name: jax.scipy.special.logit(
            (init[name] - bounds[0]) / (bounds[1] - bounds[0]))
            for name, bounds in zip(names, (uniform.PARAM_BOUNDS[name]["hard"] for name in names))}
        init_params["width_lapse"] = jnp.asarray(init["width_lapse"])
        print(f"s{session_type}: starting chain {chain + 1}/{NUM_CHAINS}", flush=True)
        chain_start = perf_counter()
        key = random.PRNGKey(SEED + session_type * 100 + chain)
        sampler.warmup(key, data, init_params=init_params)
        jax.block_until_ready(sampler.post_warmup_state)
        print(f"s{session_type}: chain {chain + 1} warmup complete in {perf_counter() - chain_start:.1f} s", flush=True)
        sample_batches, extra_batches = [], []
        for batch in range(NUM_SAMPLES // batch_size):
            key, key_run = random.split(key)
            sampler.run(key_run, data, extra_fields=("hmc_state.diverging", "hmc_state.accept_prob", "hmc_state.num_steps"))
            sample_batches.append({name: np.asarray(value) for name, value in sampler.get_samples().items()})
            extra_batches.append({name: np.asarray(value) for name, value in sampler.get_extra_fields().items()})
            sampler.post_warmup_state = sampler.last_state
            with (target_dir / f"chain_{chain + 1}_last_state.pkl").open("wb") as handle:
                pickle.dump(sampler.last_state, handle)
            print(f"s{session_type}: chain {chain + 1}, {batch_size * (batch + 1)}/{NUM_SAMPLES} draws; "
                  f"mean NUTS steps {extra_batches[-1]['hmc_state.num_steps'].mean():.1f}", flush=True)
        samples = {name: np.concatenate([values[name] for values in sample_batches]) for name in PARAM_NAMES}
        np.savez_compressed(target_dir / f"chain_{chain + 1}_samples.npz", **samples)
        extra = {name: np.concatenate([values[name] for values in extra_batches]) for name in extra_batches[0]}
        np.savez_compressed(target_dir / f"chain_{chain + 1}_diagnostics.npz", **extra)
        for name in PARAM_NAMES:
            chain_results[name].append(samples[name])
        divergence_rows.append({"chain": chain + 1, "divergences": int(extra["hmc_state.diverging"].sum()),
                               "mean_accept_prob": float(extra["hmc_state.accept_prob"].mean()),
                               "mean_num_steps": float(extra["hmc_state.num_steps"].mean()),
                               "runtime_s": perf_counter() - chain_start})
        print(f"s{session_type}: chain {chain + 1} complete: {divergence_rows[-1]}", flush=True)
        (target_dir / "run_status.json").write_text(json.dumps({"status": "running", "completed_chains": chain + 1,
            "elapsed_s": perf_counter() - started}, indent=2) + "\n")
    combined = {name: np.stack(values) for name, values in chain_results.items()}
    np.savez_compressed(target_dir / "posterior_samples.npz", **combined)
    diagnostics = mcmc_summary(combined, prob=0.95)
    rows = []
    for name in PARAM_NAMES:
        values = combined[name].ravel()
        rows.append({"parameter": name, "mean": float(values.mean()), "sd": float(values.std()),
                     "q025": float(np.quantile(values, 0.025)), "q975": float(np.quantile(values, 0.975)),
                     "r_hat": float(diagnostics[name]["r_hat"]), "n_eff": float(diagnostics[name]["n_eff"])})
    table = pd.DataFrame(rows)
    table.to_csv(target_dir / "posterior_summary.csv", index=False)
    pd.DataFrame(divergence_rows).to_csv(target_dir / "chain_diagnostics.csv", index=False)
    means = dict(zip(table.parameter, table["mean"]))
    max_rhat, min_ess = float(table.r_hat.max()), float(table.n_eff.min())
    divergences = sum(row["divergences"] for row in divergence_rows)
    converged = max_rhat < 1.01 and min_ess >= 400 and divergences == 0
    result = {"session_type": session_type, "status": "converged" if converged else "needs_convergence_review",
              "inference": "NUTS for six smooth parameters; 32 Metropolis proposals in log-width per transition",
              "width_proposal_jacobian": "+log(width)", "width_prior": WIDTH_BOUNDS,
              "parameter_bounds": PARAM_BOUNDS, "posterior_point_estimate": "mean of MCMC posterior samples",
              "n_chains": NUM_CHAINS, "warmup_per_chain": NUM_WARMUP, "draws_per_chain": NUM_SAMPLES,
              "max_rhat": max_rhat, "min_effective_sample_size": min_ess, "divergences": divergences,
              "posterior_mean_parameters": means, "posterior_mean_log_likelihood": float(loglike(means)),
              "n_trials": len(frame), "source_manifest": str(manifest_path), "source_manifest_sha256": validation["manifest_sha256"],
              "pooling": "all eight animals, trial pooled, OFF plus all ON, identical baseline abort and censor rows",
              "truncation": "none", "lapse_clock": "fixation start", "location_s": 0.0,
              "runtime_s": perf_counter() - started, "source_script": str(Path(__file__).resolve())}
    (target_dir / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (target_dir / "run_status.json").write_text(json.dumps({"status": result["status"]}, indent=2) + "\n")
    source_out = target_dir / "source"
    source_out.mkdir(exist_ok=True)
    for source in (Path(__file__), Path(uniform.__file__), Path(exponential.__file__), Path(exponential.step_jump.__file__)):
        shutil.copy2(source, source_out / source.name)
    print(table.to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)
