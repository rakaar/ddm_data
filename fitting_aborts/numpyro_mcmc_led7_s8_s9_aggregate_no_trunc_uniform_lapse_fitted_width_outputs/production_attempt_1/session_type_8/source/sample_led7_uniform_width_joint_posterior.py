# %%
"""Joint independence-Metropolis fits of the exact sharp uniform lapse model.

Pilot NUTS-within-Metropolis draws supply a fixed, heavy-tailed joint proposal.
All production proposals receive the exact target/proposal-density correction.
Because proposals are independent of the current state, their expensive
likelihoods can be evaluated in GPU batches before the sequential MH decisions.
"""

# %%
from pathlib import Path
import hashlib
import json
import os
import pickle
import runpy
import shutil
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ.get("LED7_UNIFORM_WIDTH_OUTPUT_ROOT", str(
    SCRIPT_DIR / "numpyro_mcmc_led7_s8_s9_aggregate_no_trunc_uniform_lapse_fitted_width_outputs")))
PILOT_ROOT = Path(os.environ.get("LED7_UNIFORM_WIDTH_PILOT_ROOT", str(OUTPUT_ROOT / "pilot_runs")))
SESSION_TYPES = tuple(map(int, os.environ.get("LED7_UNIFORM_WIDTH_SESSION_TYPES", "8,9").split(",")))
NUM_CHAINS = 4
WARMUP = 2000
NUM_DRAWS = int(os.environ.get("LED7_UNIFORM_WIDTH_JOINT_DRAWS", "30000"))
GPU_BATCH = 8
PROPOSAL_DF = 5
PROPOSAL_COV_INFLATION = 2.0
SEED = 8071

# Reuse the exact likelihood, priors, data loader and validation without running
# the pilot optimizer/sampler. Cell-style scripts deliberately have no main().
os.environ["LED7_UNIFORM_WIDTH_VALIDATE_ONLY"] = "1"
os.environ["LED7_UNIFORM_WIDTH_OUTPUT_ROOT"] = str(OUTPUT_ROOT)
namespace = runpy.run_path(str(SCRIPT_DIR / "fit_led7_s8_s9_uniform_fitted_width.py"))
OUTPUT_ROOT = namespace["OUTPUT_ROOT"]
PARAM_NAMES = namespace["PARAM_NAMES"]
BOUNDS = namespace["PARAM_BOUNDS"]
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from numpyro.diagnostics import summary as diagnostic_summary
uniform = namespace["uniform"]
exponential = namespace["exponential"]


# %%
def independence_scan(log_weights, log_uniforms, start_log_weight):
    # Same routine is independently tested below on a known Normal posterior.
    def accept_step(carry, incoming):
        current_index, current_weight = carry
        candidate_index, candidate_weight, log_u = incoming
        accept = log_u < candidate_weight - current_weight
        state = (jnp.where(accept, candidate_index, current_index),
                 jnp.where(accept, candidate_weight, current_weight))
        return state, (state[0], accept)
    _, result = jax.lax.scan(accept_step, (jnp.full(NUM_CHAINS, -1, dtype=jnp.int64), start_log_weight),
                            (jnp.broadcast_to(jnp.arange(log_weights.shape[0])[:, None], log_weights.shape), log_weights, log_uniforms))
    return result


# %%
rng = np.random.default_rng(9102)
toy_proposals = stats.t.rvs(df=5, size=(40000, NUM_CHAINS), random_state=rng) * 2
toy_weights = stats.norm.logpdf(toy_proposals) - stats.t.logpdf(toy_proposals / 2, df=5) + np.log(2)
toy_starts = np.zeros(NUM_CHAINS)
toy_start_weight = stats.norm.logpdf(toy_starts) - stats.t.logpdf(toy_starts / 2, df=5) + np.log(2)
toy_indices, _ = jax.jit(independence_scan)(jnp.asarray(toy_weights), jnp.asarray(np.log(rng.uniform(size=toy_weights.shape))), jnp.asarray(toy_start_weight))
toy_indices = np.asarray(toy_indices)
toy_samples = np.where(toy_indices >= 0, toy_proposals[np.maximum(toy_indices, 0), np.arange(NUM_CHAINS)], toy_starts)
assert abs(toy_samples[2000:].mean()) < .02
assert abs(toy_samples[2000:].std() - 1) < .02
(OUTPUT_ROOT / "joint_metropolis_validation.json").write_text(json.dumps({"status": "passed",
    "target": "independent standard Normal", "proposal": "Student-t(5), scale 2", "n_chains": NUM_CHAINS,
    "mean": float(toy_samples[2000:].mean()), "sd": float(toy_samples[2000:].std()),
    "acceptance_ratio": "log target(candidate)-log q(candidate)-log target(current)+log q(current)"}, indent=2) + "\n")

for session_type in SESSION_TYPES:
    out = OUTPUT_ROOT / f"session_type_{session_type}"
    if (out / "posterior_samples.npz").exists():
        print(f"s{session_type}: preserving completed production posterior.", flush=True)
        continue
    pilot_dir = PILOT_ROOT / f"session_type_{session_type}"
    pilot_files = sorted(pilot_dir.glob("chain_*_samples.npz"))
    if not pilot_files:
        raise FileNotFoundError(f"No pilot draws in {pilot_dir}")
    pilot_arrays = [np.load(path) for path in pilot_files]
    pilot = np.column_stack([np.concatenate([values[name] for values in pilot_arrays]) for name in PARAM_NAMES])
    center = pilot.mean(axis=0)
    covariance = np.cov(pilot, rowvar=False)
    proposal = stats.multivariate_t(loc=center, shape=PROPOSAL_COV_INFLATION * covariance, df=PROPOSAL_DF)
    frame = pd.read_csv(out / "source_row_manifest.csv")
    frame = frame.loc[frame.likelihood_role.isin(("abort", "censored"))]
    data = namespace["likelihood_data"](frame)

    def log_target(vector):
        params = dict(zip(PARAM_NAMES, vector))
        prior = sum(exponential.step_jump.trapezoidal_logpdf_jax(params[name], bounds["hard"][0],
                    *bounds["plausible"], bounds["hard"][1]) for name, bounds in BOUNDS.items())
        ll = uniform.proactive_led_step_jump_no_trunc_lapse_loglike_jax(params, data)
        return prior + ll

    batch_target = jax.jit(jax.vmap(log_target))
    rng = np.random.default_rng(SEED + session_type)
    n_steps = WARMUP + NUM_DRAWS
    proposed = proposal.rvs(size=(n_steps, NUM_CHAINS), random_state=rng)
    flat = proposed.reshape(-1, len(PARAM_NAMES))
    logq = proposal.logpdf(flat).reshape(n_steps, NUM_CHAINS)
    logp_flat = np.empty(len(flat))
    started = perf_counter()
    print(f"s{session_type}: production joint sampler; {len(pilot)} pilot draws, {len(flat)} likelihood evaluations", flush=True)
    for start in range(0, len(flat), GPU_BATCH):
        chunk = flat[start:start + GPU_BATCH]
        logp_flat[start:start + len(chunk)] = np.asarray(batch_target(jnp.asarray(chunk)))
        if start % (GPU_BATCH * 1000) == 0:
            print(f"s{session_type}: {start}/{len(flat)} exact likelihoods; elapsed {perf_counter() - started:.1f} s", flush=True)
    assert not np.isnan(logp_flat).any()
    logp = logp_flat.reshape(n_steps, NUM_CHAINS)
    starts = pilot[rng.integers(len(pilot), size=NUM_CHAINS)]
    start_logp = np.asarray(batch_target(jnp.asarray(starts)))
    start_weight = start_logp - proposal.logpdf(starts)
    log_weights = logp - logq
    indices, accepted = jax.jit(independence_scan)(jnp.asarray(log_weights),
        jnp.asarray(np.log(rng.uniform(size=(n_steps, NUM_CHAINS)))), jnp.asarray(start_weight))
    indices, accepted = np.asarray(indices), np.asarray(accepted)
    samples = np.where((indices >= 0)[:, :, None], proposed[np.maximum(indices, 0), np.arange(NUM_CHAINS)[None, :]], starts[None, :, :])
    retained = samples[WARMUP:]
    combined = {name: retained[:, :, i].T for i, name in enumerate(PARAM_NAMES)}
    diagnostics = diagnostic_summary(combined, prob=.95)
    rows = []
    for name, values in combined.items():
        rows.append({"parameter": name, "mean": float(values.mean()), "sd": float(values.std()),
            "q025": float(np.quantile(values, .025)), "q975": float(np.quantile(values, .975)),
            "r_hat": float(diagnostics[name]["r_hat"]), "n_eff": float(diagnostics[name]["n_eff"])})
    table = pd.DataFrame(rows)
    # Importance weights from the same independent proposals provide an
    # additional integration check; they are not used as MCMC posterior draws.
    weights = np.exp(log_weights.ravel() - logsumexp(log_weights))
    importance_ess = float(1 / np.sum(weights ** 2))
    importance_mean = weights @ flat
    proposal_diagnostics = {"pilot_root": str(pilot_dir), "pilot_files": [str(p) for p in pilot_files],
        "n_pilot_draws": len(pilot), "proposal_df": PROPOSAL_DF, "proposal_shape_inflation": PROPOSAL_COV_INFLATION,
        "proposal_center": center.tolist(), "proposal_covariance_before_inflation": covariance.tolist(),
        "acceptance_per_chain": accepted[WARMUP:].mean(axis=0).tolist(),
        "importance_weight_ess": importance_ess, "largest_normalized_importance_weight": float(weights.max()),
        "importance_means": dict(zip(PARAM_NAMES, map(float, importance_mean)))}
    max_rhat, min_ess = float(table.r_hat.max()), float(table.n_eff.min())
    converged = max_rhat < 1.01 and min_ess >= 400 and importance_ess >= 1000
    means = dict(zip(table.parameter, table["mean"]))
    result = {"session_type": session_type, "status": "converged" if converged else "needs_convergence_review",
        "inference": "joint independence Metropolis with fixed heavy-tailed Student-t proposal initialized from pilot NUTS-within-Metropolis",
        "exact_correction": "target/proposal-density Hastings ratio in physical parameter coordinates",
        "width_prior": namespace["WIDTH_BOUNDS"], "parameter_bounds": BOUNDS, "n_chains": NUM_CHAINS,
        "warmup_per_chain": WARMUP, "draws_per_chain": NUM_DRAWS, "max_rhat": max_rhat,
        "min_effective_sample_size": min_ess, "divergences": 0, "divergence_note": "not applicable to production Metropolis",
        "posterior_point_estimate": "mean of MCMC posterior samples", "posterior_mean_parameters": means,
        "posterior_mean_log_likelihood": float(uniform.proactive_led_step_jump_no_trunc_lapse_loglike_jax(means, data)),
        "n_trials": len(frame), "source_manifest_sha256": hashlib.sha256((out / "source_row_manifest.csv").read_bytes()).hexdigest(),
        "location_s": 0.0, "lapse_clock": "fixation start", "truncation": "none",
        "pooling": "eight animals, exact baseline OFF plus all ON abort/censored trials, trial pooled",
        "runtime_s": perf_counter() - started, "source_script": str(Path(__file__).resolve()), "proposal_diagnostics": proposal_diagnostics}
    np.savez_compressed(out / "posterior_samples.npz", **combined)
    np.savez_compressed(out / "joint_proposals_and_weights.npz", proposed=proposed, log_target=logp, log_proposal=logq,
                        accepted=accepted, indices=indices, starts=starts)
    table.to_csv(out / "posterior_summary.csv", index=False)
    (out / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "run_status.json").write_text(json.dumps({"status": result["status"]}, indent=2) + "\n")
    source = out / "source"
    source.mkdir(exist_ok=True)
    for path in (Path(__file__), SCRIPT_DIR / "fit_led7_s8_s9_uniform_fitted_width.py", Path(uniform.__file__),
                 Path(exponential.__file__), Path(exponential.step_jump.__file__)):
        shutil.copy2(path, source / path.name)
    pilot_copy = out / "pilot"
    pilot_copy.mkdir(exist_ok=True)
    for path in pilot_dir.glob("*"):
        if path.is_file() and path.name != "source_row_manifest.csv":
            shutil.copy2(path, pilot_copy / path.name)
    print(table.to_string(index=False), flush=True)
    print(f"s{session_type}: {result['status']}; min ESS {min_ess:.1f}, max Rhat {max_rhat:.5f}, "
          f"importance ESS {importance_ess:.1f}, acceptance {proposal_diagnostics['acceptance_per_chain']}", flush=True)
