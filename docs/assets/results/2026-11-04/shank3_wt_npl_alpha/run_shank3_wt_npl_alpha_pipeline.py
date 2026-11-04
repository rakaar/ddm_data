# %% One-off resumable fit, numerical diagnostics, and independent verification
"""Run in tmux. Final publication follows actual visual inspection of both PNGs."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'asd_analysis/shank3_cohort1_wt_npl_alpha_condition_delay'
PYTHON = ROOT/'.venv/bin/python'
lock = (OUT/'pipeline.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
environment = {**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
cores = sorted(os.sched_getaffinity(0))[:4]
environment['ASDFIT_CPUS'] = ','.join(map(str,cores))
status = dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),parameter_count=37,
              fitted_choices=68184,animals=5,visual_inspection='pending')
try:
    for label,script,backend in [
        ('fit','fit_shank3_wt_npl_alpha.py','cuda'),
        ('predictions_and_figures','plot_shank3_wt_npl_alpha.py','cpu'),
        ('independent_validation','verify_shank3_wt_npl_alpha.py','cpu')]:
        if label=='fit' and (OUT/'seed_0/run_summary.json').exists():
            continue
        status.update(stage=label)
        (OUT/'completion.json').write_text(json.dumps(status,indent=2)+'\n')
        with (OUT/f'{label}.log').open('a') as log:
            subprocess.run([str(PYTHON),'asd_analysis/'+script],cwd=ROOT,
                env={**environment,'JAX_PLATFORMS':backend},stdout=log,stderr=subprocess.STDOUT,check=True)
    status.update(status='figures_ready_for_visual_inspection',stage='numerically_verified',
                  fit=json.loads((OUT/'seed_0/run_summary.json').read_text()))
except Exception as error:
    status.update(status='failed',error=repr(error),checkpoints_preserved=True)
    raise
finally:
    (OUT/'completion.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2),flush=True)
