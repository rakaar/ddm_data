# %% Finish the authorized analysis when the HOM fit have saved posteriors
"""Wait for the HOM fit, then plot, independently verify, publish and build.

This is a one-off continuation of the running fits, not a recurring scheduler.
Failures stop publication and are recorded in completion.json and stage logs.
"""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'asd_analysis/shank3_cohort1_hom_gamma_omega'
POLL_SECONDS=30
VARIANTS=['condition_delay']
PYTHON=ROOT/'.venv/bin/python'
lock=(OUT/'completion.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
environment=os.environ.copy()
environment['OPENBLAS_NUM_THREADS']='1'
environment['OMP_NUM_THREADS']='1'
environment['ASDFIT_CPUS']='8,9,10,11,12,13,14,15'
os.sched_setaffinity(0,set(range(8,16)))
environment['MPLCONFIGDIR']='/tmp/asd-hom-matplotlib'
state={'status':'waiting_for_fits','started_utc':datetime.now(timezone.utc).isoformat(),
       'final_visual_inspection':'pending','completed_stages':[]}

try:
    (OUT/'completion.json').write_text(json.dumps(state,indent=2)+'\n')
    while not all((OUT/variant/'run_summary.json').exists() for variant in VARIANTS):
        for variant in VARIANTS:
            logfile=OUT/f'{variant}.log'
            if logfile.exists() and 'Traceback (most recent call last)' in logfile.read_text():
                raise RuntimeError(f'{variant} failed; see {logfile}')
        time.sleep(POLL_SECONDS)
    steps=[('plot',[str(PYTHON),'asd_analysis/plot_shank3_hom_gamma_omega.py']),
           ('independent_verification',[str(PYTHON),'asd_analysis/verify_shank3_hom_gamma_omega.py']),
           ('publication',[str(PYTHON),'asd_analysis/publish_shank3_hom_gamma_omega.py']),
           ('strict_mkdocs_build',[str(PYTHON),'-m','mkdocs','build','--strict'])]
    for name,command in steps:
        if name=='publication':
            state['status']='waiting_for_visual_inspection'
            (OUT/'completion.json').write_text(json.dumps(state,indent=2)+'\n')
            while not (OUT/'visual_inspection.json').exists():
                time.sleep(5)
            assert json.loads((OUT/'visual_inspection.json').read_text())['status']=='passed'
            state['final_visual_inspection']='passed'
        state['status']='running_'+name
        (OUT/'completion.json').write_text(json.dumps(state,indent=2)+'\n')
        print(f'{datetime.now(timezone.utc).isoformat()}: {name}',flush=True)
        with (OUT/f'{name}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['completed_stages'].append(name)
    publication=json.loads((OUT/'publication_manifest.json').read_text())
    for filename,digest in publication['copied_asset_hashes'].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest
    page=(ROOT/publication['page']).read_text()
    own=page.split('<!-- shank3-hom-gamma-omega:start -->')[1].split('<!-- shank3-hom-gamma-omega:end -->')[0]
    assert own.count('![')==2
    for variant in VARIANTS:
        for kind in ['RTDs','psychometric']:
            stem=f'SHANK3_cohort1_HOM_{variant}_{kind}'
            assert f'{stem}.png)' in own and f'{stem}.pdf)' in own
    prep=json.loads((OUT/'preparation.json').read_text())
    for filename,digest in prep['preservation_hashes'].items():
        with (ROOT/filename).open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==digest
    state['status']='numerically_verified_published_and_built'
    state['finished_utc']=datetime.now(timezone.utc).isoformat()
    state['direct_images_and_downloads_verified']=True
except Exception as error:
    state['status']='failed'
    state['error']=repr(error)
    raise
finally:
    (OUT/'completion.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state,indent=2),flush=True)
