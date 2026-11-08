# %% Editable backup scope: current fits, exact references, sources and publication
"""Create an immutable snapshot, copy to Drive, and verify every path/size/MD5."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STAMP = os.environ.get('SHANK3_BACKUP_STAMP', datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
MACHINE = socket.gethostname().split('.')[0]
SNAPSHOT = ROOT/'backup_staging'/f'shank3_gaussian_omega25_{STAMP}'
REMOTE = f'raga:ddm_fit_backups_20261108_shank3_wt_hom_gaussian_omega25_{MACHINE}/{STAMP}'
LOG = ROOT/'logs'/f'gdrive_shank3_gaussian_omega25_{STAMP}'
DIRS = [
    'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl_omega25',
    'asd_analysis/shank3_cohort1_wt_hom_gaussian_delay_model_comparison_omega25',
    'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_model_comparison_omega25',
    'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl',
    'asd_analysis/shank3_cohort1_wt_hom_gaussian_delay_model_comparison',
    'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_model_comparison',
    'asd_analysis/shank3_cohort1_wt_hom_npl_alpha_gaussian_delay_by_abl',
    'asd_analysis/shank3_cohort1_proactive_lapse/wt',
    'asd_analysis/shank3_cohort1_proactive_lapse/hom',
    'docs/assets/results/2026-11-08/shank3_gamma_omega_gaussian_abl_delay_omega25']
FILES = [
    'asd_analysis/README.md', 'docs/results/2026-11-08.md', 'docs/results/index.md', 'mkdocs.yml',
    'paper_notes/model_taxonomy_and_paper_notes.md', str(Path(__file__).relative_to(ROOT))]
for pattern in ['*shank3*gamma_omega*gaussian*abl*.py', '*shank3*gamma_omega*model_comparison*.py',
                '*shank3*gaussian_abl_delay_model_comparison*.py', '*shank3*gamma_omega*omega25*.py',
                '*shank3*npl_alpha*gaussian*abl*.py']:
    FILES += [str(p.relative_to(ROOT)) for p in (ROOT/'asd_analysis').glob(pattern)]
for relative in DIRS[:1] + DIRS[3:4] + DIRS[6:7]:
    for config in (ROOT/relative).glob('*/seed_0/configuration.json'):
        FILES += [name for name in json.loads(config.read_text())['source_hashes']
                  if (ROOT/name).suffix == '.py']
FILES += ['asd_analysis/shank3_gaussian_batched_refinement.py',
          'asd_analysis/shank3_npl_alpha_gaussian_abl_delay_batched_utils.py']
FILES = sorted(set(FILES))
LOG.parent.mkdir(exist_ok=True)
assert 'raga:' in subprocess.check_output(['rclone', 'listremotes'], text=True).splitlines()


def digest(path, algorithm):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


# %% Snapshot each original file once; keep originals untouched
if not (SNAPSHOT/'config/backup_manifest.json').exists():
    assert not SNAPSHOT.exists(), 'A partial snapshot exists; inspect it before resuming.'
    records = []
    for relative in DIRS:
        assert (ROOT/relative).is_dir(), relative
        for source in sorted((ROOT/relative).rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts:
                continue
            name = str(source.relative_to(ROOT))
            destination = SNAPSHOT/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            sha = digest(source, 'sha256')
            assert digest(destination, 'sha256') == sha
            records.append(dict(path=name, bytes=source.stat().st_size, sha256=sha,
                                md5=digest(destination, 'md5')))
    copied = {row['path'] for row in records}
    for relative in FILES:
        if relative in copied:
            continue
        source, destination = ROOT/relative, SNAPSHOT/relative
        assert source.is_file(), relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sha = digest(source, 'sha256')
        assert digest(destination, 'sha256') == sha
        records.append(dict(path=relative, bytes=source.stat().st_size, sha256=sha,
                            md5=digest(destination, 'md5')))
    config = SNAPSHOT/'config'
    config.mkdir()
    (config/'README.md').write_text(
        '# SHANK3 cohort 1 WT/HOM Gaussian-delay fits\n\n'
        'Current reference: direct condition-specific Gamma/Omega, Omega hard maximum 25 s^-1; '
        'flat prior 0.1–12. Gaussian delay mu/sigma shared across signed ILDs within ABL. '
        '68 parameters per genotype; 68,184 WT and 67,715 HOM choices in [0,1] s. '
        'Both checked 135,000 updates and restored update 110,000.\n\n'
        'Includes full posterior draws, best/latest resumable states, inputs, memberships, '
        'losses, convergence and numerical/visual audits, plots and sources. '
        'Unchanged 13-parameter NPL+alpha reference fits and WT/HOM proactive abort fits '
        'are included. The 15-bound direct fits and their comparisons are historical only; '
        'they are retired from the result-book display. Raw whole-cohort CSVs are not copied; '
        'the exact selected trial tables and likelihood arrays are included.\n\n'
        f'Source machine: {MACHINE}. Source checkout: {ROOT}.\n'
        f'Local snapshot: {SNAPSHOT}.\nDrive destination: `{REMOTE}`.\n\n'
        'Paths mirror the checkout. No original fits were changed or deleted. '
        'Transfer uses rclone copy, with no remote deletions.\n')
    manifest = dict(machine=MACHINE, source_checkout=str(ROOT), snapshot=str(SNAPSHOT), remote=REMOTE,
        created_utc=datetime.now(timezone.utc).isoformat(), source_directories=DIRS,
        source_files=len(records), source_bytes=sum(row['bytes'] for row in records), files=records)
    (config/'backup_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
else:
    manifest = json.loads((SNAPSHOT/'config/backup_manifest.json').read_text())
    assert manifest['remote'] == REMOTE

local = []
for path in sorted(SNAPSHOT.rglob('*')):
    if path.is_file():
        local.append(dict(path=str(path.relative_to(SNAPSHOT)), bytes=path.stat().st_size,
                          md5=digest(path, 'md5'), sha256=digest(path, 'sha256')))
(LOG.with_suffix('.local_files.json')).write_text(json.dumps(local, indent=2)+'\n')
print('Snapshot ready:', len(local), 'files;', sum(row['bytes'] for row in local), 'bytes;', REMOTE, flush=True)

# %% Resumable Drive copy followed by independent checksums and exact inventory
subprocess.run(['rclone', 'copy', str(SNAPSHOT), REMOTE, '--transfers', '4', '--checkers', '8',
    '--drive-chunk-size', '64M', '--stats', '30s', '--log-level', 'INFO',
    '--log-file', str(LOG.with_suffix('.transfer.log'))], check=True)
with LOG.with_suffix('.check.txt').open('w') as stream:
    subprocess.run(['rclone', 'check', str(SNAPSHOT), REMOTE, '--checkers', '8'],
                   stdout=stream, stderr=subprocess.STDOUT, check=True)
remote = json.loads(subprocess.check_output(['rclone', 'lsjson', REMOTE, '--recursive', '--files-only', '--hash'], text=True))
LOG.with_suffix('.remote_files.json').write_text(json.dumps(remote, indent=2)+'\n')
expected = {row['path']: row for row in local}
actual = {row['Path']: row for row in remote}
assert len(actual) == len(remote) and set(actual) == set(expected)
for name, row in expected.items():
    assert actual[name]['Size'] == row['bytes'], name
    assert actual[name]['Hashes']['md5'].lower() == row['md5'], name
size = json.loads(subprocess.check_output(['rclone', 'size', REMOTE, '--json'], text=True))
assert size['count'] == len(local) and size['bytes'] == sum(row['bytes'] for row in local)
for row in manifest['files']:
    assert digest(ROOT/row['path'], 'sha256') == row['sha256'], row['path']
parent, leaf = REMOTE.rsplit('/', 1)
folders = json.loads(subprocess.check_output(['rclone', 'lsjson', parent, '--dirs-only'], text=True))
folder = next(row for row in folders if row['Path'].rstrip('/') == leaf)
report = dict(status='verified', machine=MACHINE, remote=REMOTE, local_snapshot=str(SNAPSHOT),
    drive_url='https://drive.google.com/drive/folders/'+folder['ID'], files=size['count'], bytes=size['bytes'],
    source_files_unchanged=len(manifest['files']), exact_paths_sizes_md5=True, rclone_check='passed',
    source_sha256_check='passed', finished_utc=datetime.now(timezone.utc).isoformat(),
    per_directory={base: dict(files=sum(row['path'].startswith(base+'/') for row in local),
        bytes=sum(row['bytes'] for row in local if row['path'].startswith(base+'/'))) for base in DIRS})
LOG.with_suffix('.verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
