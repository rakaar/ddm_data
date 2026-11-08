# %%
"""Export an explicitly user-stopped shared-Omega fit from its saved best window."""
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
from functools import partial
from datetime import datetime, timezone
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import jax
from jax import random
from numpyro import handlers
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from human_shared_omega_gaussian_delay_svi_utils import (
    ROOT, OUTPUT, load_data, prepare_group, parameter_names, parameter_blocks,
    guide_support_model, sample_matrix, vector_params, numerical_check, sha256,
)

# %% Finalization performs no optimizer updates and leaves the fit configuration intact.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=OUTPUT)
parser.add_argument('--group', choices=['Control', 'ASD'], default='Control')
args = parser.parse_args()
out, group = args.output_dir, args.group
folder = out / group.lower()
config = json.loads((out/'run_config.json').read_text())
request = json.loads((out/'user_requested_finalize.json').read_text())
assert request['group'] == group
assert sha256(out/'run_config.json') == request['original_run_config_sha256']
assert not (folder/'convergence_summary.json').exists(), 'Already finalized'
for path, digest in config['source_hashes'].items():
    assert sha256(ROOT/path) == digest, path
for name, digest in request['checkpoint_hashes'].items():
    assert sha256(folder/name) == digest, name
mode = config['mode']
names, blocks = parameter_names(mode), parameter_blocks(mode)
trace = pd.read_csv(folder/'convergence.csv')
losses = np.load(folder/'losses.npy')
np.testing.assert_allclose(trace.mean_negative_elbo, losses.reshape(-1,1000).mean(1), atol=1e-10)
best_index = int(np.argmin(trace.mean_negative_elbo))
best_step = int(trace.step.iloc[best_index])
with (folder/'best_checkpoint.pkl').open('rb') as stream:
    best = pickle.load(stream)
assert best_step == best['step'] == request['restored_best_step']
assert len(losses) == request['completed_steps'] == int(trace.step.iloc[-1])
_, selected, _, provenance = load_data(mode)
assert json.loads(json.dumps(provenance)) == config['data']
_, _, data, initial, _ = prepare_group(selected, group, mode, Path(config['initial_root']))
guide = AutoMultivariateNormal(partial(guide_support_model, mode=mode),
    init_loc_fn=init_to_value(values=initial), init_scale=config['guide_init_scale'])
handlers.seed(guide, rng_seed=config['seed'])(data)
samples = guide.sample_posterior(random.PRNGKey(config['seed']+100000),
    best['params'], sample_shape=(config['posterior_draws'],))
matrix = sample_matrix(samples, mode)
assert matrix.shape == (10000,64) and np.isfinite(matrix).all()
np.savez_compressed(folder/'posterior_samples.npz', **{n: np.asarray(samples[n]) for n in blocks})
np.savez_compressed(folder/'guide_parameters.npz', **{n: np.asarray(v) for n,v in best['params'].items()})
summary = pd.DataFrame(dict(parameter=names, mean=matrix.mean(0), sd=matrix.std(0,ddof=1),
    q025=np.quantile(matrix,.025,axis=0), median=np.median(matrix,axis=0), q975=np.quantile(matrix,.975,axis=0)))
summary.to_csv(folder/'posterior_summary.csv', index=False, float_format='%.17g')
for name, values in [('covariance',np.cov(matrix,rowvar=False)),('correlation',np.corrcoef(matrix,rowvar=False))]:
    pd.DataFrame(values,index=names,columns=names).to_csv(folder/f'posterior_{name}.csv',float_format='%.17g')
numerical = numerical_check(vector_params(matrix.mean(0),mode), data, mode)
last = trace.iloc[-1]
report = dict(mode=mode, free_parameters=64, stop_reason='user_requested_loss_plateau_restore_best',
    completed_steps=len(losses), restored_best_step=best_step,
    best_window_mean_negative_elbo=float(trace.mean_negative_elbo.iloc[best_index]),
    terminal_parameter_stable=bool(last.parameter_stable),
    restored_checkpoint_parameter_stable=bool(trace.parameter_stable.iloc[best_index]),
    confirmation_windows=int(last.patience), required_confirmation_windows=config['confirmation_windows'],
    updates_after_restored_best=len(losses)-best_step,
    flagged=not (last.patience>=config['confirmation_windows'] and last.parameter_stable),
    maximum_terminal_mean_change_sd=float(last.maximum_mean_change_sd), wall_seconds=float(last.elapsed_s),
    numerical_check_at_posterior_means=numerical,
    posterior_kind='full-rank Gaussian variational approximation in unconstrained coordinates',
    warning='User requested loss-based stopping; strict parameter stability was not satisfied. No global optimum claim; pooled-trial uncertainty excludes participant variability',
    finalization_source=str(Path(__file__).relative_to(ROOT)), finalization_source_sha256=sha256(__file__),
    finalization_backend=jax.default_backend(), finalized_utc=datetime.now(timezone.utc).isoformat(),
    user_request_record_sha256=sha256(out/'user_requested_finalize.json'))
(folder/'convergence_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
