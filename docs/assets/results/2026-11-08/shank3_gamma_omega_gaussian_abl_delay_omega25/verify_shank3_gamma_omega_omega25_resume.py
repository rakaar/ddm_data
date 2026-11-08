# %% Exact checkpoint/resume check in disposable 30-trial smoke runs
"""Use the actual GPU fitter, preserving the production WT/HOM runs."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl_omega25'
FITTER = ROOT/'asd_analysis/fit_shank3_gamma_omega_gaussian_abl_delay_omega25.py'
ENV = {**os.environ, 'JAX_PLATFORMS': 'cuda', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}

for name, pause in [('validation_smoke_resume', True), ('validation_smoke_resume', False),
                    ('validation_smoke_straight', False)]:
    folder = OUT/'wt'/name
    if (folder/'run_summary.json').exists():
        continue
    command = [str(ROOT/'.venv/bin/python'), str(FITTER), '--genotype', 'wt',
               '--smoke-test', '--smoke-folder', name]
    if pause:
        command += ['--checkpoint-only-after', '5']
    with (OUT/'resume_validation.log').open('a') as log:
        subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)

resumed, straight = [OUT/'wt'/name for name in ['validation_smoke_resume', 'validation_smoke_straight']]
np.testing.assert_array_equal(np.load(resumed/'losses.npy'), np.load(straight/'losses.npy'))
with np.load(resumed/'posterior_samples.npz') as a, np.load(straight/'posterior_samples.npz') as b:
    assert a.files == b.files
    for key in a.files:
        np.testing.assert_array_equal(a[key], b[key])
runs = [json.loads((p/'run_summary.json').read_text()) for p in [resumed, straight]]
assert runs[0]['restored_best_step'] == runs[1]['restored_best_step']
assert runs[0]['final_checked_step'] == runs[1]['final_checked_step'] == 15
for run in runs:
    assert run['parameter_stability_required'] is False
    confirmed = run['confirmation_windows'] >= 2
    assert (run['status'] == 'converged') == confirmed
    if not confirmed:
        assert run['stop_reason'] == 'maximum_steps_without_confirmation'
report = dict(status='passed', updates=15, interruption_after=5,
    restored_best_step=runs[0]['restored_best_step'], exact_losses=True,
    exact_posterior_draws=True, parameter_count=68, omega_hard_upper=25,
    maximum_limit_classification='inconclusive without confirmation',
    folders=[str(p.relative_to(OUT)) for p in [resumed, straight]])
(OUT/'resume_validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
