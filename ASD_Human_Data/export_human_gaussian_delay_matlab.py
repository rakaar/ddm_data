# %% Export the accepted wider-bound RT fit for MATLAB.
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'gamma_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms'
parser = argparse.ArgumentParser(description='Export physical posterior means and joint variational draws to MATLAB.')
parser.add_argument('--fit-dir', type=Path, default=DEFAULT)
parser.add_argument('--output-dir', type=Path)
args = parser.parse_args()
fit = args.fit_dir.resolve()
out = args.output_dir or fit / 'matlab_exports'
out.mkdir(parents=True, exist_ok=True)
config = json.loads((fit / 'run_config.json').read_text())
blocks = config['parameter_blocks']
means, posterior = {}, {}
hashes = {}

# %% Rows of samples are joint posterior draws; columns identify stimuli.
for label, folder in [('Control', 'control'), ('ASD', 'asd')]:
    source = fit / folder / 'posterior_samples.npz'
    with np.load(source) as saved:
        samples = {key: np.asarray(saved[key], dtype=np.float64).reshape(10000, -1) for key in blocks}
    ct = pd.read_csv(fit / folder / 'conditions.csv', float_precision='round_trip').sort_values('condition_index')
    assert ct.condition_index.tolist() == list(range(20))
    mapping = {
        'index': np.arange(1, 21, dtype=np.int32).reshape(-1, 1),
        'ASL_cd': ct.ASL_cd.to_numpy().reshape(-1, 1),
        'Delta_dB': ct.signed_delta.to_numpy().reshape(-1, 1),
        'Mean_dB': ct.Mean_dB.to_numpy().reshape(-1, 1),
        'n_trials': ct.n.to_numpy().reshape(-1, 1),
    }
    means[label] = {key: value.mean(axis=0, keepdims=True) for key, value in samples.items()}
    means[label]['conditions'] = mapping
    matrix = np.column_stack([samples[key] for key in blocks])
    assert matrix.shape == (10000, 83)
    posterior[label] = dict(samples, conditions=mapping,
        covariance=np.cov(matrix, rowvar=False), correlation=np.corrcoef(matrix, rowvar=False))
    for path in [source, fit / folder / 'conditions.csv']:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

# %% Common metadata explains units and the conditional-lapse convention.
metadata = dict(
    source_fit=str(fit), condition='RT', rt_window_s=np.array([[0., 1.8]]),
    model=config['model'], parameters_per_group=83, posterior_draws=10000,
    parameter_names=np.array(config['parameter_names'], dtype=object).reshape(1, -1),
    parameter_order='gamma(1:20), omega(1:20), delay_mu(1:20), delay_sigma(1:20), w, lapse_prob, lapse_prob_right',
    parameter_name_indices='Names retain Python suffixes 00..19; MATLAB conditions.index uses 1..20.',
    units='gamma: dimensionless; omega: 1/s; delay_mu and delay_sigma: seconds; w, lapse_prob, lapse_prob_right: probabilities',
    delay_definition='Underlying Gaussian mu and sigma; trial delay is truncated below zero, with no upper cutoff.',
    lapse_definition='lapse_prob is the lapse fraction among retained RT trials; lapse RT is uniform on [0,1.8] seconds.',
    scalar_sharing='w and both lapse parameters are shared across stimuli within each group; Control and ASD are fitted independently.',
    draw_structure='Each sample row is one joint draw; retain row alignment across all parameters to preserve dependence.',
    uncertainty='Variational approximation for pooled trials, not participant-to-participant variability.',
    source_sha256_json=json.dumps(hashes, sort_keys=True),
)
means['metadata'] = dict(metadata, representation='Arithmetic means of physical constrained posterior samples')
posterior['metadata'] = dict(metadata, representation='10000 saved joint physical constrained variational posterior draws; not an executable NumPyro guide')

files = [('human_rt_fit_parameter_means.mat', means), ('human_rt_fit_variational_posterior.mat', posterior)]
for name, contents in files:
    path = out / name
    savemat(path, contents, do_compression=True, long_field_names=True, oned_as='column')
    loaded = loadmat(path, struct_as_record=False, squeeze_me=False)
    for label in ['Control', 'ASD']:
        group = loaded[label][0, 0]
        for key in blocks:
            np.testing.assert_array_equal(getattr(group, key), contents[label][key])
        np.testing.assert_array_equal(group.conditions[0, 0].index, np.arange(1, 21).reshape(-1, 1))
    print(f'{path} ({path.stat().st_size:,} bytes): round-trip verified')

report = dict(passed=True, format='MATLAB v5, compressed; readable using MATLAB load',
    source_hashes=hashes, files={name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name, _ in files},
    mean_shape=[1,20], posterior_shape=[10000,20], scalar_posterior_shape=[10000,1])
(out / 'export_verification.json').write_text(json.dumps(report, indent=2) + '\n')
