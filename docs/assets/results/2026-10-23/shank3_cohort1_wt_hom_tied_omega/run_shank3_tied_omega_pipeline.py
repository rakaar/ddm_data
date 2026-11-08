# %% GPU-only launcher; driver installation/reboot is outside this script
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'asd_analysis/shank3_cohort1_wt_hom_tied_omega'
PYTHON=ROOT/'.venv/bin/python'
RUNS=[(g,v) for g in ['wt','hom'] for v in ['omega_by_abl','omega_global']]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preflight-only',action='store_true')
parser.add_argument('--wait-for-gpu',action='store_true',help='One-off wait, not a recurring automation; a reboot terminates this process.')
args=parser.parse_args()
OUT.mkdir(exist_ok=True)
lock=(OUT/'pipeline.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
environment=os.environ.copy()
environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
available=sorted(os.sched_getaffinity(0))
status=dict(status='checking_gpu',checked_utc=datetime.now(timezone.utc).isoformat(),fits_started=False)
last_error=None
while True:
    check=subprocess.run(['nvidia-smi','--query-gpu=name,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,capture_output=True)
    status.update(nvidia_smi_returncode=check.returncode,nvidia_smi_output=check.stdout.strip(),
                  error=check.stderr.strip() or (check.stdout.strip() if check.returncode else ''))
    for key,command in [('loaded_module',['cat','/sys/module/nvidia/version']),('installed_module',['modinfo','-F','version','nvidia'])]:
        p=subprocess.run(command,text=True,capture_output=True)
        status[key]=p.stdout.strip()
    if check.returncode==0:
        probe_env={**environment,'JAX_PLATFORMS':'cuda'}
        probe=subprocess.run([str(PYTHON),'-c','import jax; jax.config.update("jax_enable_x64",True); assert jax.default_backend()=="gpu"; print(jax.devices()[0].device_kind)'],
                             env=probe_env,text=True,capture_output=True,cwd=ROOT)
        status['jax_probe_returncode']=probe.returncode
        status['jax_probe_output']=probe.stdout.strip()
        if probe.returncode==0:break
        status['error']=probe.stderr.strip()
    status['status']='waiting_for_gpu'
    (OUT/'gpu_status.json').write_text(json.dumps(status,indent=2)+'\n')
    (OUT/'completion.json').write_text(json.dumps(status,indent=2)+'\n')
    if status['error']!=last_error:
        print('GPU unavailable; full fits have not started. '+status['error'],flush=True)
        last_error=status['error']
    if args.preflight_only or not args.wait_for_gpu:raise SystemExit(0)
    time.sleep(30)
status['status']='gpu_ready'
(OUT/'gpu_status.json').write_text(json.dumps(status,indent=2)+'\n')
if args.preflight_only:
    print(json.dumps(status,indent=2),flush=True)
    raise SystemExit(0)


def stage(label,script,*arguments):
    with (OUT/f'{label}.log').open('a') as log:
        subprocess.run([str(PYTHON),'asd_analysis/'+script,*arguments],cwd=ROOT,
                       env={**environment,'JAX_PLATFORMS':'cpu'},stdout=log,stderr=subprocess.STDOUT,check=True)


# %% Verify first, benchmark fresh disposable states, then launch 4 or 2 fits
stage('preparation','prepare_shank3_tied_omega.py')
stage('prefit_validation','verify_shank3_tied_omega.py','--pre-fit')
benchmarks=[]
for group,variant in RUNS:
    stage(f'benchmark_{group}_{variant}','fit_shank3_tied_omega.py','--group',group,'--variant',variant,'--benchmark-only')
    benchmarks.append(json.loads((OUT/group/variant/'gpu_benchmark.json').read_text()))
free_mib=float(check.stdout.splitlines()[0].split(',')[1])
budget=free_mib*1024**2*.90
peaks=[b['peak_bytes'] for b in benchmarks]
concurrency=4 if all(peaks) and sum(peaks)<budget else 2
assert not all(peaks) or sum(sorted(peaks,reverse=True)[:2])<budget,'Not enough GPU memory for two safe concurrent runs.'
launch=dict(status='running',concurrent_fits=concurrency,benchmarks=benchmarks,
            launch_utc=datetime.now(timezone.utc).isoformat(),preallocation=False,
            GPU_memory_headroom_fraction=.10,initialization='fresh seed 0; benchmark states discarded')
(OUT/'launch_manifest.json').write_text(json.dumps(launch,indent=2)+'\n')
(OUT/'completion.json').write_text(json.dumps(launch,indent=2)+'\n')


def fit(item):
    index,(group,variant)=item
    folder=OUT/group/variant
    if (folder/'run_summary.json').exists():
        return json.loads((folder/'run_summary.json').read_text())
    env=environment.copy()
    cores=available[(index*2)%len(available):(index*2)%len(available)+2]
    env['ASDFIT_CPUS']=','.join(map(str,cores))
    with (OUT/group/f'{variant}.log').open('a') as log:
        subprocess.run([str(PYTHON),'asd_analysis/fit_shank3_tied_omega.py','--group',group,'--variant',variant],
                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((folder/'run_summary.json').read_text())


try:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        summaries=list(pool.map(fit,enumerate(RUNS)))
    stage('postfit_validation','verify_shank3_tied_omega.py')
    stage('predictions','predict_shank3_tied_omega.py')
    stage('figures','plot_shank3_tied_omega.py')
    stage('diagnostic_validation','verify_shank3_tied_omega_diagnostics.py')
    state=dict(status='figures_ready_for_visual_inspection',runs=summaries,
               figures_generated=4,numerical_verification='passed',final_visual_inspection='pending')
    (OUT/'completion.json').write_text(json.dumps(state,indent=2)+'\n')
    inspection=OUT/'figures/visual_inspection.json'
    if inspection.exists() and json.loads(inspection.read_text())['status']=='passed':
        stage('publication','publish_shank3_tied_omega.py')
        with (OUT/'strict_mkdocs_build.log').open('w') as log:
            subprocess.run([str(PYTHON),'-m','mkdocs','build','--strict'],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(status='complete',final_visual_inspection='passed',strict_mkdocs_build='passed')
        (OUT/'completion.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state,indent=2),flush=True)
except Exception as error:
    (OUT/'completion.json').write_text(json.dumps(dict(status='failed',error=repr(error),checkpoints_preserved=True),indent=2)+'\n')
    raise
