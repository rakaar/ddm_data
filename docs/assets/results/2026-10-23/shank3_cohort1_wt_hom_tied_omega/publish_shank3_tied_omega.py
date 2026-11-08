# %% Publish prepared status or independently verified and visually inspected fits
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from shank3_tied_omega_utils import ROOT,OUT,VARIANTS,LABELS,sha256

DATE='2026-10-23'
TITLE='SHANK3 cohort 1 WT vs HOM: tied-Omega and shared-delay refits'
PAGE=ROOT/'docs/results'/f'{DATE}.md'
ASSETS=ROOT/'docs/assets/results'/DATE/'shank3_cohort1_wt_hom_tied_omega'
START,END='<!-- shank3-tied-omega:start -->','<!-- shank3-tied-omega:end -->'
SOURCES=[ROOT/'asd_analysis'/name for name in [
    'prepare_shank3_tied_omega.py','shank3_tied_omega_utils.py','fit_shank3_tied_omega.py',
    'verify_shank3_tied_omega.py','verify_shank3_tied_omega_smoke.py','shank3_tied_omega_prediction_utils.py',
    'predict_shank3_tied_omega.py','plot_shank3_tied_omega.py','verify_shank3_tied_omega_diagnostics.py',
    'run_shank3_tied_omega_pipeline.py','publish_shank3_tied_omega.py']]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pending',action='store_true')
args=parser.parse_args()
assert json.loads((OUT/'prefit_validation.json').read_text())['status']=='passed'
assert json.loads((OUT/'smoke_validation.json').read_text())['status']=='passed'
base=f'../assets/results/{DATE}/shank3_cohort1_wt_hom_tied_omega/'
ASSETS.mkdir(exist_ok=True)
if args.pending:
    gpu=json.loads((OUT/'gpu_status.json').read_text())
    launch_path=OUT/'launch_manifest.json'
    if gpu['status']=='gpu_ready':
        if launch_path.exists():
            launch=json.loads(launch_path.read_text())
            progress=f"**{launch['concurrent_fits']} concurrent GPU fits are running**, launched {launch['launch_utc']}."
            pending_status='fits_running'
        else:
            progress='GPU speed and memory benchmarks are in progress; the four full fits start after these checks.'
            pending_status='benchmarking_gpu'
        status=f'''**Status: GPU restored; preparation and independent likelihood checks passed.**

The restart loaded the matching NVIDIA driver **{gpu['loaded_module']}**. JAX confirms CUDA access on the **{gpu['jax_probe_output']}**. {progress} Fits use the loss-confirmation stopping rule; parameter stability is a diagnostic only.

The final images will be added after fitting, numerical verification and visual inspection. Initial-value previews are isolated from publication.
'''
    else:
        pending_status='prepared_waiting_for_gpu'
        status=f'''**Status: prepared and independently verified; full GPU fits have not started.**

GPU access currently fails with `{gpu.get('error','unavailable')}`. The loaded NVIDIA module is **{gpu.get('loaded_module','unknown')}**, while the installed module is **{gpu.get('installed_module','unknown')}**. A reboot is expected to load the matching installed driver. As selected, the fits wait for working GPU access.

All four fresh initializations, exact input copies, prior multiplicities, SciPy likelihood agreement, finite gradients, complete-window normalization and loss-confirmation rules are checked. Tiny isolated 30-trial/15-update tests verify interruption/resume and restored-best checkpoints; they are **software checks, not scientific fits**. Initial-value plot previews are isolated from publication. The four final images will be added after fitting, numerical verification and visual inspection.
'''
    figures=''
    report_names=['preparation.json','prefit_validation.json','smoke_validation.json','gpu_status.json','preservation_baseline.json']
    if launch_path.exists():report_names.append('launch_manifest.json')
else:
    post=json.loads((OUT/'postfit_validation.json').read_text())
    diagnostics=json.loads((OUT/'diagnostic_validation.json').read_text())
    inspection=json.loads((OUT/'figures/visual_inspection.json').read_text())
    manifest=json.loads((OUT/'figures/plot_manifest.json').read_text())
    assert post['status']==diagnostics['status']==inspection['status']=='passed'
    assert not diagnostics['initial_preview'] and not manifest['initial_preview']
    assert diagnostics['figure_artifact_hashes']==manifest['artifacts']
    for name,expected in manifest['artifacts'].items():
        assert sha256(OUT/'figures'/name)==expected
        if name.endswith('.png'):assert inspection['figure_hashes'][name]==expected
        shutil.copy2(OUT/'figures'/name,ASSETS/name)
    rows=[]
    for group in ['wt','hom']:
        for variant in VARIANTS:
            run=json.loads((OUT/group/variant/'run_summary.json').read_text())
            rows.append(f'| {group.upper()} | {variant} | {run["status"]} | {run["restored_best_step"]:,} | {run["final_checked_step"]:,} |')
    status='''The four independent GPU fits are complete. The stopping classifications and restored-best checkpoints are:

| Genotype | Model | Status | Restored best | Final checked |
|---|---|---|---:|---:|
'''+ '\n'.join(rows)+ '\n'
    chunks=[]
    for variant in VARIANTS:
        param=f'SHANK3_cohort1_WT_HOM_{variant}_parameters'
        behavior=f'SHANK3_cohort1_WT_HOM_{variant}_behavior'
        chunks.append(f'''### {LABELS[variant]}

![WT/HOM reactive parameter comparison: {variant}]({base}{param}.png)

*Seven panels: three Gamma panels by ABL, then Omega, reactive delay, w and del_go. WT is blue and HOM red. Shared values are shown once. Means and central 95% credible intervals come from 10,000 draws. These are conditional variational intervals with proactive/lapse uncertainty held fixed, not across-animal error bars. Signed-ILD Gammas are independent; delays are in milliseconds.*

[Parameter PNG]({base}{param}.png) · [Parameter PDF]({base}{param}.pdf)

![WT/HOM RTD, chronometric and psychometric comparison: {variant}]({base}{behavior}.png)

*Four rows: WT's five |ILD| RTDs, HOM's five RTDs, three compact chronometrics, and separate WT/HOM psychometrics. ABL 20/40/60 are blue/orange/green. RTD data steps are 1.4 pt; matching dashed model curves are 0.7 pt. All RTDs display **−0.1 to 0.6 s**, retaining 20 ms empirical bins and separate full **[−1,1] s** data/model normalization. The visible segment is not renormalized. Chronometric dots/SEMs and psychometric data are unchanged; new model predictions use each exact valid onset's conditional-window normalization before averaging. No additional sigmoid is fitted.*

[Behavior PNG]({base}{behavior}.png) · [Behavior PDF]({base}{behavior}.pdf)
''')
    figures='\n'.join(chunks)
    report_names=['preparation.json','prefit_validation.json','smoke_validation.json','postfit_validation.json',
        'diagnostic_validation.json','gpu_status.json','launch_manifest.json','preservation_baseline.json',
        'parameter_comparisons.csv','signed_condition_means.csv','chronometric_points.csv','in_sample_comparison.csv']
    for name in ['visual_inspection.json','plot_manifest.json','plotted_series.json.gz']:
        shutil.copy2(OUT/'figures'/name,ASSETS/name)
for name in report_names:
    shutil.copy2(OUT/name,ASSETS/name)
for path in SOURCES:
    shutil.copy2(path,ASSETS/path.name)
archive_name='shank3_tied_omega_prepared_audits.zip' if args.pending else 'shank3_tied_omega_fit_audits.zip'
with zipfile.ZipFile(OUT/archive_name,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in SOURCES:
        archive.write(path,'source/'+str(path.relative_to(ROOT)))
    for name in report_names:
        archive.write(OUT/name,name)
    for group in ['wt','hom']:
        for path in (OUT/group).rglob('*'):
            if path.is_file() and path.suffix in ['.json','.csv','.npz','.npy','.gz']:
                archive.write(path,str(path.relative_to(OUT)))
shutil.copy2(OUT/archive_name,ASSETS/archive_name)
entry=f'''{START}
## {TITLE}

{status}
### Models and populations

| Model | Gamma | Omega | Reactive delay | w and del_go | Free parameters per genotype |
|---|---:|---:|---:|---:|---:|
| Omega by ABL | 30 signed conditions | 3 (one per ABL) | 1 | 2 | **36** |
| Global Omega | 30 signed conditions | 1 | 1 | 2 | **34** |

Fit WT and HOM independently in each model: **68,184 WT valid choices / five animals** (37,482 right, 30,702 left) and **67,715 HOM choices / six animals** (35,320 right, 32,395 left). Use the cleaned session-type-1 CSV, training level 16, `success ∈ {{1,−1}}`, finite recorded `timed_rt` in inclusive **[0,1] s**, ABL 59→60 then 20/40/60, and signed ILDs ±1/2/4/8/16. Preserve prepared exclusions and original trial identities without reapplying the repeat rule. **−0.1 to 0.6 s is the RTD display range, not a new fitting window.**

Freeze V_A, theta_A, t_nd, lapse_prob and beta_lapse at each genotype's unrounded selected seed-0 abort posterior means. Keep the previous complete proactive/reactive/exponential-lapse mixture and exact planned onsets; divide the joint RT/choice mixture by its complete [0,1] s window probability per trial. No abort observations enter the new fitting objective. Existing priors are applied once per free parameter. WT and HOM do not share fitted values.

Use one fresh seed-0 full-rank float64 SVI run per genotype/model, clipped Adam at 0.0002, clip norm 1 and initial guide scale 0.1. Gamma starts from the previous data-derived initialization; tied Omega uses trial-count-weighted condition-level initial values, not previous posterior means. Delay starts at 0.05 s, w at 0.5 and del_go at 0.08 s. **Stopping: minimum 50,000 updates, 1,000-update windows, 25 subsequent windows without a strictly lower loss, restore the absolute-best checkpoint; maximum 1,000,000 updates.** Parameter stability remains a diagnostic only. One initialization does not test start sensitivity.

For RTD diagnostics only, include the same 13,750 WT and 13,274 HOM pre-onset aborts in [−1,0), pooling actual ±ILD trials and exact eligible-parent onset/sign frequencies. Full data RTD populations remain 81,934 WT and 80,989 HOM observations. Chronometric and psychometric model weights instead use the fitted valid trials. Chronometric dots pool trials; SEM is sample SD across animal means / √animal count, centered on the pooled dot. For the ABL-pooled panel, combine ABLs within each animal before calculating SEM. Corresponding axes are shared across the two model variants.

{figures}
### Sources and validation downloads

Outputs: `asd_analysis/shank3_cohort1_wt_hom_tied_omega/`, organized by genotype and variant. Sources: [fitting]({base}fit_shank3_tied_omega.py), [tied-parameter model]({base}shank3_tied_omega_utils.py), [independent likelihood/checkpoint verifier]({base}verify_shank3_tied_omega.py), [predictions]({base}predict_shank3_tied_omega.py), [plotting]({base}plot_shank3_tied_omega.py), [independent diagnostic verifier]({base}verify_shank3_tied_omega_diagnostics.py), [GPU-only launcher]({base}run_shank3_tied_omega_pipeline.py). Exact source paths are `asd_analysis/` followed by these filenames.

[Prefit verification]({base}prefit_validation.json) · [Checkpoint/resume tests]({base}smoke_validation.json) · [GPU status]({base}gpu_status.json) · [Preserved source hashes]({base}preservation_baseline.json) · [Source/input/audit ZIP]({base}{archive_name})

See the [previous WT/HOM condition-delay comparison](#shank3-cohort-1-wt-vs-hom-parameters-and-chronometrics). Original fits, CSVs and figures are preserved. Model fit comparisons use the same trials within genotype and are in-sample.
{END}
'''
page=PAGE.read_text()
if START in page:
    before,rest=page.split(START,1);_,after=rest.split(END,1)
    PAGE.write_text(before+entry+after)
else:
    PAGE.write_text(page.rstrip()+'\n\n'+entry)
index_path=ROOT/'docs/results/index.md'
index=index_path.read_text()
link=f'- [{TITLE}]({DATE}.md#shank3-cohort-1-wt-vs-hom-tied-omega-and-shared-delay-refits)'
if link not in index:index_path.write_text(index.rstrip()+'\n'+link+'\n')
assert f'results/{DATE}.md' in (ROOT/'mkdocs.yml').read_text()
for name,expected in json.loads((OUT/'preservation_baseline.json').read_text()).items():
    assert sha256(ROOT/name)==expected,name
publication=dict(status=pending_status if args.pending else 'published',page=str(PAGE.relative_to(ROOT)),
    section=TITLE,initial_previews_published=False,final_PNG_embeds=0 if args.pending else 4,
    asset_hashes={str(p.relative_to(ROOT)):sha256(p) for p in ASSETS.iterdir() if p.is_file()})
(OUT/'publication_manifest.json').write_text(json.dumps(publication,indent=2)+'\n')
print(publication['status']+': '+TITLE,flush=True)
