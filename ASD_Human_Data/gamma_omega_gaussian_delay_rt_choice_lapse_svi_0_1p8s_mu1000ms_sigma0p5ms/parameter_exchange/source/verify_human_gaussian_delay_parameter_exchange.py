# %% Independently reconstruct parameter exchanges and recipient-weighted predictions.
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=ROOT / 'ASD_Human_Data/gamma_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms/parameter_exchange')
args = parser.parse_args()
out = args.output_dir.resolve()
manifest = json.loads((out / 'manifest.json').read_text())
fit = Path(manifest['fit_dir'])
blocks = ['gamma', 'omega', 'delay_mu', 'delay_sigma', 'w', 'lapse_prob', 'lapse_prob_right']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


for path, digest in manifest['source_hashes'].items():
    assert sha256(path) == digest, path
original = {}
for group in ['Control', 'ASD']:
    with np.load(fit / group.lower() / 'posterior_samples.npz') as draws:
        original[group] = {k: draws[k].mean(axis=0) for k in blocks}
trials_path = ROOT / 'ASD_Human_Data/simultaneous_rt_trials.csv'
trials = pd.read_csv(trials_path, float_precision='round_trip')
trials = trials.query("Condition == 'RT' and Valid == 1 and 0 <= RT_s <= 1.8").copy()
trials['signed_delta'] = trials.Delta_dB.round(2)
saved = pd.read_csv(fit / 'selected_trials.csv', float_precision='round_trip')
pd.testing.assert_frame_equal(trials[saved.columns].sort_values('source_row').reset_index(drop=True),
                              saved.sort_values('source_row').reset_index(drop=True))
assert len(trials) == 11344 and trials.source_row.is_unique
counts = trials.groupby(['Group', 'ASL_cd', 'signed_delta']).size()
parameters = pd.read_csv(out / 'model_parameters.csv', float_precision='round_trip')
condition = pd.read_csv(out / 'condition_predictions.csv', float_precision='round_trip')
pooled = pd.read_csv(out / 'pooled_rtd_curves.csv', float_precision='round_trip')
psych = pd.read_csv(out / 'psychometric_predictions.csv', float_precision='round_trip')
chrono = pd.read_csv(out / 'chronometric_predictions.csv', float_precision='round_trip')
weights = pd.read_csv(out / 'mixture_weights.csv', float_precision='round_trip')
means = {}; max_mean_gap = 0.; max_mass_gap = 0.
for model, rows in parameters.groupby('model'):
    recipient = rows.recipient.iloc[0]
    donor = 'ASD' if recipient == 'Control' else 'Control'
    swapped = ['omega'] if model.endswith('_omega') else ['delay_mu', 'delay_sigma'] if model.endswith('_delay') else []
    assert len(rows) == 83
    for block, params in rows.groupby('parameter'):
        origin = donor if block in swapped else recipient
        assert params.source_group.eq(origin).all()
        np.testing.assert_allclose(params.sort_values('condition_index').value, np.atleast_1d(original[origin][block]), atol=1e-14, rtol=1e-13)
    ct = condition.loc[condition.model.eq(model)].sort_values('condition_index').copy()
    ct['raw_n'] = [counts.loc[recipient, c.ASL_cd, c.signed_delta] for c in ct.itertuples()]
    np.testing.assert_array_equal(ct.n, ct.raw_n)
    with np.load(out / f'{model}_predictions.npz') as pred:
        t = pred['time_s']; density = pred['density']; cdf = pred['cdf']; probability = pred['p_right']
        np.testing.assert_allclose(np.trapz(pred['right_density'], t, axis=0), probability, atol=1e-5, rtol=0)
        max_mass_gap = max(max_mass_gap, float(abs(np.trapz(density, t, axis=0)-1).max()))
        for delta, c in ct.groupby('signed_delta'):
            ids = c.condition_index.to_numpy(int); w = c.raw_n.to_numpy()/c.raw_n.sum()
            expected = float(probability[ids] @ w)
            actual = psych.loc[psych.model.eq(model) & psych.signed_delta.eq(delta)].iloc[0]
            assert actual.n == c.raw_n.sum()
            np.testing.assert_allclose(actual.p_right, expected, atol=1e-13)
        ct['abs_delta'] = ct.signed_delta.abs()
        for delta, c in ct.groupby('abs_delta'):
            ids = c.condition_index.to_numpy(int); w = c.raw_n.to_numpy()/c.raw_n.sum()
            expected_pdf = np.average(density[:, ids], axis=1, weights=c.raw_n)
            expected_cdf = np.average(cdf[:, ids], axis=1, weights=c.raw_n)
            actual = pooled.loc[pooled.model.eq(model) & pooled.abs_delta.eq(delta)].sort_values('time_s')
            np.testing.assert_allclose(actual.density, expected_pdf, atol=1e-12)
            np.testing.assert_allclose(actual.cdf, expected_cdf, atol=1e-12)
            expected_mean = float(np.average(np.trapz(1-cdf[:, ids], t, axis=0), weights=c.raw_n))
            actual_mean = chrono.loc[chrono.model.eq(model) & chrono.abs_delta.eq(delta)].iloc[0]
            assert actual_mean.n == c.raw_n.sum()
            gap = abs(actual_mean.mean_RT_s-expected_mean)
            assert gap < 1e-6
            max_mean_gap = max(max_mean_gap, gap)
        for c in weights.loc[weights.model.eq(model)].itertuples():
            subset = ct.loc[ct.signed_delta.eq(c.delta)] if c.panel == 'psychometric' else ct.loc[ct.abs_delta.eq(c.delta)]
            n = ct.loc[ct.condition_index.eq(c.condition_index), 'raw_n'].iloc[0]
            assert c.n == n and c.total_n == subset.raw_n.sum()
            np.testing.assert_allclose(c.weight, n/subset.raw_n.sum(), atol=1e-14)

baseline = json.loads((out / 'preservation_baseline.json').read_text())
for path, digest in baseline['files'].items():
    assert sha256(path) == digest, path
assert hashlib.sha256(Path(baseline['page']).read_bytes()[:baseline['page_bytes']]).hexdigest() == baseline['page_prefix_sha256']
report = dict(passed=True, selected_trial_identities=len(trials), parameter_values_checked=len(parameters),
    models=parameters.model.nunique(), condition_predictions=len(condition), pooled_psychometric_points=len(psych),
    chronometric_points=len(chrono), weights_checked=len(weights), max_mean_error_s=max_mean_gap,
    max_unit_mass_error=max_mass_gap, previous_files_preserved=len(baseline['files']),
    previous_page_prefix_preserved=True, trial_source_sha256=sha256(trials_path), verifier_sha256=sha256(__file__))
(out / 'independent_validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
