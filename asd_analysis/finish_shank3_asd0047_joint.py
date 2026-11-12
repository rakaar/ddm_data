# %% Continue from accepted joint fits to independently checked local deliverables
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import fitz

# %% Editable paths; waiting observes a verified live fitting controller
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'asd_analysis/shank3_cohort1_ASD0047_joint_npl_alpha_gaussian_abl_contaminants'
PYTHON = str(ROOT / '.venv/bin/python')
status = dict(status='waiting_for_fits', started_utc=datetime.now(timezone.utc).isoformat())


def update(stage, **values):
    status.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
    (BASE / 'postprocessing_completion.json').write_text(json.dumps(status, indent=2) + '\n')


def command(script, arguments=(), gpu=False):
    env = {**os.environ, 'ASD0047_JOINT_VARIANT': variant,
           'JAX_PLATFORMS': 'cuda' if gpu else 'cpu',
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
           'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           'ASDFIT_CPUS': ','.join(map(str, sorted(os.sched_getaffinity(0))[:8]))}
    update(script)
    with (BASE / (Path(script).stem + '.log')).open('a') as stream:
        subprocess.run([PYTHON, 'asd_analysis/' + script, *arguments], cwd=ROOT, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


try:
    while True:
        fitting = json.loads((BASE / 'completion.json').read_text())
        if fitting['status'] == 'fit_accepted_reports_pending':
            break
        pid = fitting['pid']
        proc = Path(f'/proc/{pid}/cmdline')
        if not proc.exists() or b'run_shank3_asd0047_joint.py' not in proc.read_bytes():
            raise RuntimeError('The fitting controller has terminated before acceptance; inspect its log and resume its saved checkpoints.')
        update('waiting_for_live_fitting_controller', observed_pid=pid, fit_stage=fitting['stage'])
        time.sleep(30)
    selected = json.loads((BASE / 'selected_fit.json').read_text())
    assert selected['status'] == 'passed'
    variant = selected['variant']
    command('plot_shank3_asd0047_joint.py')
    command('verify_shank3_asd0047_joint.py', ['--round', str(selected['bounds_round']), '--seed', str(selected['selected_seed']), '--report-checks'])
    command('verify_shank3_asd0047_joint_resume.py', gpu=True)
    command('plot_shank3_asd0047_joint_comparison.py')
    command('verify_shank3_asd0047_joint_comparison.py')
    command('export_shank3_asd0047_joint_corners_and_report.py')
    command('verify_shank3_asd0047_joint_exports.py')
    previews = BASE / 'posterior_and_diagnostics/pdf_previews'
    previews.mkdir(exist_ok=True)
    with fitz.open(BASE / 'posterior_and_diagnostics/SHANK3_cohort1_ASD0047_joint_diagnostics.pdf') as document:
        for index, page in enumerate(document):
            page.get_pixmap(matrix=fitz.Matrix(1., 1.)).save(previews / f'diagnostics_page_{index+1:02d}.png')
    update('ready_for_visual_inspection', status='deliverables_generated_need_visual_inspection',
           selected_run=selected['selected_run'], variant=variant,
           independent_deliverable_validation='deliverable_validation.json')
except Exception as error:
    update('execution_error', status='failed', error=repr(error))
    raise
print(json.dumps(status, indent=2), flush=True)
