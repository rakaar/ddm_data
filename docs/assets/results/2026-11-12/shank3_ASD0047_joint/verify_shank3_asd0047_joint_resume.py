# %% Disposable full-data resume check; never advances production checkpoints
import os
os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.35')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
if os.environ.get('ASDFIT_CPUS'):
    os.sched_setaffinity(0, {int(x) for x in os.environ['ASDFIT_CPUS'].split(',')})
import json
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal
from shank3_asd0047_joint_utils import ROOT, OUT, make_model, sha256, loss_window_decision, PARAMETER_COUNT

# %% Reconstruct selected fit exactly, then use only disposable optimizer states
selected = json.loads((OUT / 'selected_fit.json').read_text())
run = ROOT / selected['selected_run']
paths = [run / name for name in ['best_checkpoint.pkl', 'latest_checkpoint.pkl', 'posterior_samples.npz', 'losses.npy']]
before = {str(p): sha256(p) for p in paths}
config = json.loads((run / 'configuration.json').read_text())
model = make_model(config['priors'], config['grid_end'])
guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values={k: np.asarray(v) for k, v in config['initial'].items()}), init_scale=config['guide_scale'])
svi = SVI(model, guide, numpyro.optim.ClippedAdam(config['learning_rate'], clip_norm=config['clip_norm']), Trace_ELBO())
data = {k: jnp.asarray(v) for k, v in np.load(OUT / 'observations.npz').items()}
svi.init(jax.random.PRNGKey(config['seed']), data)
initial_state = pickle.loads((run / 'best_checkpoint.pkl').read_bytes())['state']
block = jax.jit(lambda state: jax.lax.scan(lambda s, _: svi.stable_update(s, data), state, None, length=5))
first, loss1 = block(initial_state)
whole, loss2 = block(first)
serialized = pickle.dumps(jax.device_get(first), pickle.HIGHEST_PROTOCOL)
restored = pickle.loads(serialized)
split, split_loss2 = block(restored)
diagnostics = OUT / 'resume_diagnostics'
diagnostics.mkdir(exist_ok=True)
for name, state in [('uninterrupted', whole), ('serialized_resume', split)]:
    (diagnostics / (name+'.pkl')).write_bytes(pickle.dumps(jax.device_get(state), pickle.HIGHEST_PROTOCOL))
np.savez_compressed(diagnostics / 'losses.npz', uninterrupted=np.asarray(loss2), resumed=np.asarray(split_loss2))
np.testing.assert_array_equal(np.asarray(loss2), np.asarray(split_loss2))
differences = []
for index, (a, b) in enumerate(zip(jax.tree_util.tree_leaves(whole), jax.tree_util.tree_leaves(split))):
    a, b = np.asarray(a), np.asarray(b)
    exact = np.array_equal(a, b)
    if a.dtype.kind in 'biu':
        np.testing.assert_array_equal(a, b)
    else:
        # Restoring host arrays can change CUDA fusion at floating roundoff.
        # Record that limitation rather than claim bitwise-identical states.
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
    differences.append(dict(leaf=index, exact=exact,
        max_absolute_difference=float(np.max(np.abs(a-b))) if a.size else 0.))
assert all(sha256(p) == before[str(p)] for p in paths)

# %% Confirm reset/minimum/tail behavior independently of optimizer progress
cases = []
for loss, best, step, prior_tail, expected in [
    (9., 10., 49000, 24, (True, 0, False)),
    (10., 10., 49000, 24, (False, 25, False)),
    (10., 10., 50000, 24, (False, 25, True)),
    (9., 10., 50000, 24, (True, 0, False)),
    (10., 10., 50000, 23, (False, 24, False)),
]:
    actual = loss_window_decision(loss, best, step, prior_tail, 50000, 25)
    assert tuple(actual) == expected
    cases.append(dict(loss=loss, best=best, step=step, prior_tail=prior_tail, result=actual))
report = dict(status='passed', full_data=True, observations=21786, parameters=PARAMETER_COUNT,
    uninterrupted_updates=10, serialized_after=5, exact_losses=True,
    exact_optimizer_guide_rng_states=all(x['exact'] for x in differences),
    optimizer_guide_agreement_tolerance=dict(atol=1e-12, rtol=1e-12),
    state_leaf_differences=differences, production_states_unchanged=True,
    stop_reset_minimum_confirmation_tests=cases)
(OUT / 'exact_resume_validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
