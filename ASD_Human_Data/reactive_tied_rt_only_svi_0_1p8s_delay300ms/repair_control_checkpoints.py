# %%
"""Reproduce the accepted Control trajectory after stopping a leftover worker.

This is a deterministic audit replay from the saved original initialization,
not an additional statistical initialization or a new optimization search.
"""
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
from pathlib import Path
from datetime import datetime, timezone
import sys
import json
import pickle
import shutil
import hashlib
import numpy as np
import pandas as pd
import jax
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import ClippedAdam
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'ASD_Human_Data'))
from human_reactive_tied_rt_only_support_utils import (
    PARAM_NAMES, INITIAL, load_trials, group_data, reactive_model, guide_support_model,
)

# %% Archive every overwritten trace/checkpoint before restoring the accepted stop.
out = Path(__file__).resolve().parent
folder = out/'control'
archive = out/'leftover_worker_archive'
archive.mkdir(exist_ok=True)
names = ['convergence.csv','losses.npy','best_checkpoint.pkl','latest_checkpoint.pkl',
         'convergence_summary.json','guide_parameters.npz','posterior_samples.npz']
for name in names:
    shutil.copy2(folder/name, archive/name)
shutil.copy2(out/'run.log', archive/'interleaved_run.log')
report = json.loads((folder/'convergence_summary.json').read_text())
final_step, selected_step = report['completed_steps'], report['restored_best_step']
assert final_step == 81000 and selected_step == 40000
observed_losses = np.load(archive/'losses.npy')
observed_history = pd.read_csv(archive/'convergence.csv').set_index('step')
assert len(observed_losses) >= final_step and observed_history.index.max() >= final_step
posterior_hash = hashlib.sha256((folder/'posterior_samples.npz').read_bytes()).hexdigest()
guide_hash = hashlib.sha256((folder/'guide_parameters.npz').read_bytes()).hexdigest()
frame, _ = load_trials(0., 1.8)
sub, conditions, data, _ = group_data(frame, 'Control')
data.pop('choice')
data.update(rt_lower=0., rt_upper=1.8, guide_delay_upper_s=float(sub.RT_s.min()))
guide = AutoMultivariateNormal(guide_support_model, init_loc_fn=init_to_value(values=INITIAL), init_scale=.1)
svi = SVI(reactive_model, guide, ClippedAdam(.0002, clip_norm=1.), Trace_ELBO(num_particles=1))
initial = {name:jax.device_put(value) for name,value in np.load(folder/'initial_guide_parameters.npz').items()}
state = svi.init(random.PRNGKey(0), data, .3, init_params=initial)
diagnostics = jax.jit(lambda p:guide.sample_posterior(random.PRNGKey(1729),p,sample_shape=(4096,)))
maximum_loss_error, maximum_mean_error = 0., 0.
window_losses = []

# %% Verify every stored loss and checkpoint mean on the accepted trajectory.
for step in range(1000, final_step+1, 1000):
    result = svi.run(random.PRNGKey(0),1000,data,.3,progress_bar=False,init_state=state,stable_update=True)
    state = result.state
    chunk = np.asarray(result.losses)
    expected = observed_losses[step-1000:step]
    np.testing.assert_allclose(chunk,expected,rtol=1e-12,atol=1e-8)
    maximum_loss_error = max(maximum_loss_error,float(np.max(abs(chunk-expected))))
    draws = diagnostics(result.params)
    means = np.array([np.asarray(draws[name]).mean() for name in PARAM_NAMES])
    sds = np.array([np.asarray(draws[name]).std(ddof=1) for name in PARAM_NAMES])
    old_means = observed_history.loc[step,[name+'_mean' for name in PARAM_NAMES]].to_numpy(dtype=float)
    old_sds = observed_history.loc[step,[name+'_sd' for name in PARAM_NAMES]].to_numpy(dtype=float)
    np.testing.assert_allclose(means,old_means,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(sds,old_sds,rtol=1e-11,atol=1e-12)
    maximum_mean_error = max(maximum_mean_error,float(np.max(abs(means-old_means))))
    window_losses.append(float(chunk.mean()))
    if step == selected_step:
        selected = jax.device_get(dict(state=state,params=result.params,step=step))
    if step % 10000 == 0 or step == final_step:
        print('Verified Control replay',step,'of',final_step,'max loss error',maximum_loss_error,flush=True)
assert (np.argmin(window_losses)+1)*1000 == selected_step
np.testing.assert_allclose(min(window_losses),report['best_window_mean_negative_elbo'],atol=1e-10,rtol=0)
saved_guide = np.load(folder/'guide_parameters.npz')
for name,value in selected['params'].items():
    np.testing.assert_allclose(value,saved_guide[name],rtol=1e-12,atol=1e-14)
samples = guide.sample_posterior(random.PRNGKey(100000),selected['params'],sample_shape=(10000,))
saved_samples = np.load(folder/'posterior_samples.npz')
for name in PARAM_NAMES:
    np.testing.assert_allclose(samples[name],saved_samples[name],rtol=1e-12,atol=1e-14)

# %% Restore the accepted best/latest states and trim only the leftover iterations.
with (folder/'best_checkpoint.pkl').open('wb') as stream:
    pickle.dump(selected,stream)
with (folder/'latest_checkpoint.pkl').open('wb') as stream:
    pickle.dump(jax.device_get(dict(state=state,params=result.params,step=final_step)),stream)
np.save(folder/'losses.npy',observed_losses[:final_step])
observed_history.loc[:final_step].reset_index().to_csv(folder/'convergence.csv',index=False)
assert hashlib.sha256((folder/'posterior_samples.npz').read_bytes()).hexdigest() == posterior_hash
assert hashlib.sha256((folder/'guide_parameters.npz').read_bytes()).hexdigest() == guide_hash
repair = dict(status='passed',completed_at_utc=datetime.now(timezone.utc).isoformat(),
    stopped_leftover_worker_pid=1609457,accepted_final_step=final_step,accepted_best_step=selected_step,
    discarded_iterations_archived=len(observed_losses)-final_step,
    replay='Exact original initialization, optimizer, seed and RNG trajectory; checked every loss and checkpoint mean through accepted stop',
    maximum_loss_absolute_error=maximum_loss_error,maximum_checkpoint_mean_absolute_error=maximum_mean_error,
    all_10000_posterior_draws_reproduced=True,posterior_samples_byte_unchanged=True,guide_parameters_byte_unchanged=True,
    posterior_samples_sha256=posterior_hash,guide_parameters_sha256=guide_hash,
    actual_prior_upper_s=.3,rt_window_s=[0.,1.8],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'checkpoint_repair.json').write_text(json.dumps(repair,indent=2))
print('PASSED: accepted checkpoints and trajectory restored; posterior and guide files unchanged.',flush=True)
