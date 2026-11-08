# %% Resumable GPU fits followed by automatic numerical diagnostics
"""Launch in tmux; publish only after the five actual PNGs are visually checked."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl'
PYTHON=ROOT/'.venv/bin/python'
FITTER={'wt':'fit_shank3_gamma_omega_gaussian_abl_delay.py',
        'hom':'fit_shank3_gamma_omega_gaussian_abl_delay.py'}
lock=(OUT/'pipeline.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false','JAX_PLATFORMS':'cuda'}
status=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),parameter_count_per_genotype=68)


def run(script,group=None,extra=(),label=None):
    command=[str(PYTHON),'asd_analysis/'+script]+(['--genotype',group] if group else [])+list(extra)
    folder=OUT/group if group else OUT
    with (folder/(label or Path(script).stem)).with_suffix('.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


try:
    for group in ['wt','hom']:
        if not (OUT/group/'prefit_validation.json').exists():
            run('verify_shank3_gamma_omega_gaussian_abl_delay.py',group,['--pre-fit'])
        assert json.loads((OUT/group/'prefit_validation.json').read_text())['status']=='passed'
        benchmark_file=OUT/group/'gpu_benchmark.json'
        if not benchmark_file.exists() or 'memory_limit_bytes' not in json.loads(benchmark_file.read_text()):
            run(FITTER[group],group,['--benchmark-only','--benchmark-steps','20'],label='benchmark')
    benchmarks=[json.loads((OUT/g/'gpu_benchmark.json').read_text()) for g in ['wt','hom']]
    parallel=sum(x['peak_bytes'] for x in benchmarks)<.8*min(x['memory_limit_bytes'] for x in benchmarks)
    status.update(stage='fitting',parallel=parallel,benchmarks=benchmarks)
    (OUT/'completion.json').write_text(json.dumps(status,indent=2)+'\n')
    processes=[]
    for group in ['wt','hom']:
        if (OUT/group/'seed_0/run_summary.json').exists(): continue
        log=(OUT/group/'fit.log').open('a')
        process=subprocess.Popen([str(PYTHON),'asd_analysis/'+FITTER[group],'--genotype',group],
                                 cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((group,process,log))
        if not parallel:
            assert process.wait()==0,group+' fit failed'
    while any(p.poll() is None for _,p,_ in processes): time.sleep(10)
    for group,p,log in processes:
        log.close()
        assert p.returncode==0,group+' fit failed'
    for group in ['wt','hom']:
        status.update(stage='predictions_and_verification',genotype=group)
        (OUT/'completion.json').write_text(json.dumps(status,indent=2)+'\n')
        checked=OUT/group/'independent_validation.json'
        if checked.exists() and (OUT/group/'plot_manifest.json').exists():
            record=json.loads(checked.read_text())
            from shank3_gamma_omega_gaussian_abl_delay_utils import sha256
            if record['status']=='passed' and record['verifier_sha256']==sha256(ROOT/'asd_analysis/verify_shank3_gamma_omega_gaussian_abl_delay.py'):
                continue
        run('plot_shank3_gamma_omega_gaussian_abl_delay.py',group,label='plot')
        run('verify_shank3_gamma_omega_gaussian_abl_delay.py',group,label='verification')
    run('plot_shank3_gamma_omega_gaussian_abl_delay_parameters.py',label='parameters')
    run('verify_shank3_gamma_omega_gaussian_abl_delay_figures.py',label='figure_verification')
    status.update(status='figures_ready_for_visual_inspection',stage='numerically_verified')
except Exception as error:
    status.update(status='failed',error=repr(error),checkpoints_preserved=True)
    raise
finally:
    (OUT/'completion.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2),flush=True)
