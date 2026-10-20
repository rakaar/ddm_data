# %%
"""Sequential CPU SVI, pooled human RT groups; see README and run_config.json."""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/human_tied_mpl")
from pathlib import Path
import argparse
import json
import pickle
import time
from functools import partial

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import ClippedAdam
from human_reactive_tied_svi_utils import (
    ROOT, PARAM_NAMES, INITIAL, load_trials, group_data, reactive_model, sha256,
)

# %% Configuration: CLI overrides preserve one specified initialization per group.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--group", choices=["Control", "ASD", "all"], default="all")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--min-steps", type=int, default=50000)
parser.add_argument("--max-steps", type=int, default=150000)
parser.add_argument("--output-dir", type=Path, default=ROOT/"ASD_Human_Data/reactive_tied_rt_svi")
args = parser.parse_args()
assert 0 < args.min_steps <= args.max_steps
args.output_dir.mkdir(parents=True, exist_ok=True)
frame, provenance = load_trials()
config = dict(seed=args.seed, min_steps=args.min_steps, max_steps=args.max_steps,
              check_every=1000, patience=12, minimum_relative_improvement=.001,
              parameter_stability_sd=.1, stability_checkpoints=3,
              learning_rate=.0002, gradient_clip=1., guide="AutoMultivariateNormal",
              guide_init_scale=.1, elbo_particles=1, posterior_draws=10000,
              checkpoint_diagnostic_draws=4096, K_max=40, float64=True, platform="cpu",
              likelihood="joint RT and choice conditional on either bound in [0,3] s",
              initial=INITIAL, parameter_names=PARAM_NAMES,
              priors={"rate_lambda": "LogNormal(log(0.2),1.5)",
                      "T_0": "LogNormal(log(0.001),2)", "theta_E": "LogNormal(log(30),1.5)",
                      "w": "Uniform(0.05,0.95)", "t_E_aff": "Uniform(0,min(group RT)-1e-6)"},
              source=provenance,
              source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__),
                   Path(__file__).with_name("human_reactive_tied_svi_utils.py"),
                   ROOT/"fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py"]})
(args.output_dir/"run_config.json").write_text(json.dumps(config, indent=2))
print("CPU devices:", jax.devices(), "output:", args.output_dir, flush=True)

# %% One full-batch fit for each requested group; restore the best finite window.
for group in (["Control", "ASD"] if args.group == "all" else [args.group]):
    folder = args.output_dir/group.lower()
    folder.mkdir(exist_ok=True)
    if (folder/"posterior_samples.npz").exists():
        raise FileExistsError(f"Completed posterior already exists: {folder}; choose a new output directory")
    sub, conditions, data, delay_upper = group_data(frame, group)
    sub.to_csv(folder/"selected_trials.csv", index=False, float_format="%.17g")
    conditions.to_csv(folder/"conditions.csv", index=False, float_format="%.17g")
    model = partial(reactive_model, K_max=40)
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values=INITIAL), init_scale=.1)
    svi = SVI(model, guide, ClippedAdam(.0002, clip_norm=1.), Trace_ELBO(num_particles=1))
    key = random.PRNGKey(args.seed)
    diagnostics_key = random.PRNGKey(1729)  # common draws eliminate MC changes between checkpoints
    sample_diagnostics = jax.jit(lambda p: guide.sample_posterior(
        diagnostics_key, p, sample_shape=(4096,)))
    state = None
    best_state = None
    best_loss = np.inf
    best_step = 0
    patience = 0
    rows, losses, mean_history, sd_history = [], [], [], []
    started = time.monotonic()
    metadata = dict(group=group, n_trials=len(sub), participants=sorted(sub.Participant.unique().tolist()),
                    min_rt_s=float(sub.RT_s.min()), max_rt_s=float(sub.RT_s.max()),
                    delay_support_lower_s=0., delay_support_upper_s=delay_upper,
                    source_rows_sha256=sha256(folder/"selected_trials.csv"))
    (folder/"data_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n{group}: {len(sub)} trials, delay support (0, {delay_upper:.15g}) s", flush=True)
    stop_reason = "maximum_steps"
    for end in range(1000, args.max_steps+1000, 1000):
        step_count = min(1000, args.max_steps - sum(len(x) for x in losses))
        if step_count <= 0:
            break
        result = svi.run(random.fold_in(key, end//1000), step_count, data, delay_upper,
                         progress_bar=False, init_state=state, stable_update=True)
        state = result.state
        chunk = np.asarray(result.losses)
        losses.append(chunk)
        completed = sum(len(x) for x in losses)
        if not np.isfinite(chunk).all():
            raise FloatingPointError(f"{group} has nonfinite losses in window ending {completed}")
        loss = float(chunk.mean())
        improvement = (best_loss-loss)/max(1., abs(best_loss)) if np.isfinite(best_loss) else np.inf
        improved = loss < best_loss
        if improved:
            best_state, best_params, best_loss, best_step = state, result.params, loss, completed
            with (folder/"best_checkpoint.pkl").open("wb") as handle:
                pickle.dump(jax.device_get(dict(state=state, params=result.params, step=completed)), handle)
        patience = 0 if improved and improvement >= .001 else patience+1
        draws = sample_diagnostics(result.params)
        means = np.array([np.asarray(draws[n]).mean() for n in PARAM_NAMES])
        sds = np.array([np.asarray(draws[n]).std(ddof=1) for n in PARAM_NAMES])
        mean_history.append(means)
        sd_history.append(sds)
        if len(mean_history) >= 3:
            # Largest pairwise mean displacement over the last three checkpoints.
            changes = np.ptp(mean_history[-3:], axis=0)/np.minimum.reduce(sd_history[-3:])
            stable = bool(np.all(changes < .1))
        else:
            changes = np.full(5, np.nan)
            stable = False
        row = dict(step=completed, mean_negative_elbo=loss, best_loss=best_loss, best_step=best_step,
                   patience=patience, updated_best=improved, parameter_stable=stable,
                   maximum_mean_change_sd=float(np.max(changes)), elapsed_s=time.monotonic()-started)
        for i, name in enumerate(PARAM_NAMES):
            row[name+"_mean"] = means[i]
            row[name+"_sd"] = sds[i]
            row[name+"_change_sd"] = changes[i]
        rows.append(row)
        pd.DataFrame(rows).to_csv(folder/"convergence.csv", index=False)
        np.save(folder/"losses.npy", np.concatenate(losses))
        with (folder/"latest_checkpoint.pkl").open("wb") as handle:
            pickle.dump(jax.device_get(dict(state=state, params=result.params, step=completed)), handle)
        print(f"{group} {completed:6d}: loss {loss:.3f}, best {best_step}, patience {patience}/12, "
              f"max mean change {row['maximum_mean_change_sd']:.3f} SD, "
              f"elapsed {row['elapsed_s']:.1f}s", flush=True)
        if completed >= args.min_steps and patience >= 12 and stable:
            stop_reason = "patience12_and_parameter_stability"
            break
    samples = guide.sample_posterior(random.PRNGKey(args.seed+100000), best_params, sample_shape=(10000,))
    sample_matrix = np.column_stack([np.asarray(samples[n]) for n in PARAM_NAMES])
    np.savez_compressed(folder/"posterior_samples.npz", **{n: np.asarray(samples[n]) for n in PARAM_NAMES})
    np.savez_compressed(folder/"guide_parameters.npz", **{n: np.asarray(v) for n,v in best_params.items()})
    summary = pd.DataFrame({"parameter": PARAM_NAMES, "mean": sample_matrix.mean(axis=0),
                            "sd": sample_matrix.std(axis=0, ddof=1),
                            "q025": np.quantile(sample_matrix, .025, axis=0),
                            "median": np.median(sample_matrix, axis=0),
                            "q975": np.quantile(sample_matrix, .975, axis=0)})
    summary.to_csv(folder/"posterior_summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(np.cov(sample_matrix, rowvar=False), index=PARAM_NAMES, columns=PARAM_NAMES).to_csv(folder/"posterior_covariance.csv")
    pd.DataFrame(np.corrcoef(sample_matrix, rowvar=False), index=PARAM_NAMES, columns=PARAM_NAMES).to_csv(folder/"posterior_correlation.csv")
    best_index = best_step//1000-1
    restored_stable = bool(rows[best_index]["parameter_stable"])
    report = dict(stop_reason=stop_reason, completed_steps=completed, restored_best_step=best_step,
                  best_window_mean_negative_elbo=best_loss, terminal_parameter_stable=stable,
                  restored_checkpoint_parameter_stable=restored_stable,
                  flagged=not (stable and restored_stable), wall_seconds=time.monotonic()-started,
                  posterior_kind="full-rank Gaussian variational approximation in unconstrained coordinates",
                  warning="Pooled-trial uncertainty does not represent participant-to-participant variability")
    (folder/"convergence_summary.json").write_text(json.dumps(report, indent=2))
    print(group, "FINISHED", report, flush=True)
