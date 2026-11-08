# %% Editable result-book settings and source files
"""Publish HOM figures only after numerical verification and visual inspection.

--pending records an honest running status with verified preparation downloads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'asd_analysis/shank3_cohort1_hom_gamma_omega'
DATE='2026-10-23'
ASSETS=ROOT/'docs/assets/results'/DATE/'shank3_cohort1_hom_gamma_omega'
PAGE=ROOT/'docs/results'/f'{DATE}.md'
START,END='<!-- shank3-hom-gamma-omega:start -->','<!-- shank3-hom-gamma-omega:end -->'
TITLE='SHANK3 cohort 1 HOM: condition-specific reactive delays'
SOURCES=['fit_shank3_hom_gamma_omega.py','plot_shank3_hom_gamma_omega.py',
    'verify_shank3_hom_gamma_omega.py','publish_shank3_hom_gamma_omega.py',
    'finish_shank3_hom_gamma_omega.py','shank3_wt_gamma_omega_utils.py','proactive_lapse_no_led_utils.py']
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pending',action='store_true')
args=parser.parse_args()
ASSETS.mkdir(parents=True,exist_ok=True)
base=f'../assets/results/{DATE}/shank3_cohort1_hom_gamma_omega/'
prep=json.loads((OUT/'preparation.json').read_text())
assert json.loads((OUT/'prefit_validation.json').read_text())['status']=='passed'
for name in ['preparation.json','fixed_proactive.json','conditions.csv','membership.csv','prefit_validation.json']:
    shutil.copy2(OUT/name,ASSETS/name)
for name in SOURCES:
    shutil.copy2(ROOT/'asd_analysis'/name,ASSETS/name)

# %% Selection, model and precise stopping/normalization descriptions
intro=f'''## {TITLE}

Pool all **six SHANK3 cohort 1 HOM animals** from the cleaned session-type-1 CSV: training level 16, `success ∈ {{1,−1}}`, finite recorded `timed_rt` in inclusive **[0,1] s**, ABL 20/40/60 after 59→60, and signed ILDs ±1/2/4/8/16. The fit contains **67,715 choices: 35,320 right and 32,395 left**; exclude 357 valid choices above 1 s. Animals and sessions contribute their actual trials. Preparation exclusions remain in force; repeats are not filtered again.

Fit **92 parameters**: 30 Gamma, 30 Omega, 30 reactive delays (`t_E_aff` / `del_E`), shared `w`, and shared `del_go`. Each condition is a signed ABL–ILD pair, without imposed sign symmetry. Freeze all five unrounded HOM seed-0 proactive/exponential-lapse posterior means from `asd_analysis/shank3_cohort1_proactive_lapse/hom/seed_0/posterior_summary.csv`. Reuse the unchanged WT race likelihood, priors and initialization procedure. No abort or censored observations enter this reactive-fit objective; no additional lapse parameters or uniform lapse are added.

For each fitted valid trial, planned onset is `c=intended_fix_time/1000`; absolute response time is `c+timed_rt`. Normalize the **complete joint RT/choice mixture together** by its either-choice probability over `[c,c+1]`. Valid RT is not reconstructed from `fix_time`; reactive delay is not capped at the fastest observed RT. Posterior uncertainty is conditional on the frozen proactive/lapse parameters.

Run one seed-0 initialization on the RTX 4090, float64 full-batch full-rank NumPyro SVI, clipped Adam **0.0002**, clip norm **1**, and guide initial scale **0.1**. Check mean losses in 1,000-update windows, with at least 50,000 updates. Every strictly lower window resets confirmation; stop after **25 subsequent windows without improvement**, then restore the absolute-best checkpoint. **Parameter stability is reported but is not a stopping requirement**, at either the terminal or restored checkpoint. The 1,000,000-update limit is inconclusive if confirmation is unmet. Save 10,000 posterior draws and central 95% intervals. One start does not test initialization sensitivity.

Gamma starts at `0.5 × logit(measured P(right))`; Omega uses the WT symmetric-DDM mean-time relation and condition mean RT minus 0.05 s, clipped inside the same prior bounds. Initialize every reactive delay at 0.05 s, `w=0.5`, and `del_go=0.08 s`. Full priors are recorded in [preparation]({base}preparation.json).

Compare with [the preceding WT analysis](2026-10-22.md#shank3-cohort-1-wt-shared-versus-condition-specific-reactive-delays). [HOM membership]({base}membership.csv) · [Condition counts]({base}conditions.csv) · [Frozen parameters]({base}fixed_proactive.json) · [Prefit verification]({base}prefit_validation.json)
'''
if args.pending:
    body='**Fit running; final figures are not yet published.**\n\n'+intro
else:
    validation=json.loads((OUT/'independent_validation.json').read_text())
    inspection=json.loads((OUT/'visual_inspection.json').read_text())
    manifest=json.loads((OUT/'plot_manifest.json').read_text())
    assert validation['status']==inspection['status']=='passed'
    assert len(validation['plot_checks'])==1 and validation['figure_artifact_hashes']==manifest['artifacts']
    for name,digest in manifest['artifacts'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
        if name.endswith('.png'):
            assert inspection['figure_hashes'][name]==digest
        shutil.copy2(OUT/name,ASSETS/name)
    manifest['status']='independently_verified'
    (OUT/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ['plot_manifest.json','independent_validation.json','visual_inspection.json',
                 'in_sample_comparison.csv','resume_validation.json']:
        shutil.copy2(OUT/name,ASSETS/name)
    (ASSETS/'condition_delay').mkdir(exist_ok=True)
    for name in ['configuration.json','run_summary.json','posterior_summary.csv',
                 'condition_parameter_means.csv','psychometric_predictions.csv','rtd_panel_audit.csv']:
        shutil.copy2(OUT/'condition_delay'/name,ASSETS/'condition_delay'/name)
    run=json.loads((OUT/'condition_delay/run_summary.json').read_text())
    quality=pd.read_csv(OUT/'in_sample_comparison.csv').iloc[0]
    body=intro+f'''
### Fit completion

Status: **{run['status']}** under the loss-confirmation rule. Final checked update: **{run['final_checked_step']:,}**; restored best: **{run['restored_best_step']:,}**; confirmation windows: **{run['confirmation_windows']}**. Best mean negative ELBO: **{run['best_window_negative_elbo']:.6f}**. Terminal parameter stability: **{run['terminal_parameter_stable']}**; restored-checkpoint stability: **{run['restored_checkpoint_parameter_stable']}**. These are diagnostic values, not gates or proof of a global optimum.

### HOM RT distributions: model and data

![SHANK3 cohort 1 HOM RT distributions, three ABL rows and five absolute ILD columns]({base}SHANK3_cohort1_HOM_condition_delay_RTDs.png)

*Rows are ABL 20 blue, 40 orange and 60 green; columns are |ILD| 1/2/4/8/16 dB. Colored 1.4 pt empirical steps use 20 ms bins and are drawn above the thinner 0.8 pt black model curve, evaluated every 1 ms. Display −1 to 1 s. The 80,989 displayed observations are 67,715 fitted valid choices plus 13,274 strict pre-onset fixation aborts in [−1,0), timed as `fix_time/1000 − intended_fix_time/1000`. Late fixation-labelled rows are not observed aborts; abort observations are shown only and never refitted here.*

*Pool actual ±ILD trials before normalization. Model predictions average exact onset times and signed-condition frequencies across all 81,721 eligible parent trials. Normalize each pooled data/model panel separately to area one over [−1,1] s; pre-normalization window masses are retained in the audit.*

[Full-size PNG]({base}SHANK3_cohort1_HOM_condition_delay_RTDs.png) · [PDF]({base}SHANK3_cohort1_HOM_condition_delay_RTDs.pdf) · [Panel audit]({base}condition_delay/rtd_panel_audit.csv)

### HOM psychometric: model and data

![SHANK3 cohort 1 HOM measured and predicted right-choice probabilities]({base}SHANK3_cohort1_HOM_condition_delay_psychometric.png)

*ABL-colored dots show observed right-choice fractions among the fitted valid trials. Thin 1.0 pt lines connect predictions only at the ten measured signed ILDs. Integrate the complete right-choice density over [0,1] s, divide by each trial's either-choice window probability, then average over that signed condition's exact fitted-trial onsets. Predictions use arithmetic physical posterior means. No additional sigmoid is fitted.*

[Full-size PNG]({base}SHANK3_cohort1_HOM_condition_delay_psychometric.png) · [PDF]({base}SHANK3_cohort1_HOM_condition_delay_psychometric.pdf) · [Predictions and counts]({base}condition_delay/psychometric_predictions.csv)

### Diagnostics and reproducibility

[Convergence PNG]({base}SHANK3_cohort1_HOM_gamma_omega_convergence.png) · [Convergence PDF]({base}SHANK3_cohort1_HOM_gamma_omega_convergence.pdf) · [Posterior intervals]({base}condition_delay/posterior_summary.csv) · [Fit configuration]({base}condition_delay/configuration.json) · [Run summary]({base}condition_delay/run_summary.json)

In-sample joint log likelihood at posterior means: **{quality.joint_log_likelihood:.3f}**; psychometric RMSE across 30 signed conditions: **{quality.psychometric_rmse:.5f}**; mean/worst condition maximum RT-CDF gaps: **{quality.mean_rt_cdf_max_gap_1ms:.5f} / {quality.worst_rt_cdf_max_gap_1ms:.5f}**. These are descriptive in-sample checks, without held-out evaluation.

Independent verification reconstructs all trial identities, frozen parameters, probabilities, complete-mixture normalization, model densities/CDFs, signed-condition weighting, histogram areas, checkpoint restoration and the loss-confirmation rule. CPU/GPU posterior reproduction permits only float64 rounding differences. Source CSVs, prior abort fits, and WT source/results retain their recorded hashes.

[Independent verification]({base}independent_validation.json) · [Visual inspection]({base}visual_inspection.json) · [Checkpoint resume check]({base}resume_validation.json) · [Full numerical audits]({base}shank3_cohort1_hom_gamma_omega_audits.zip)
'''
    archive_name='shank3_cohort1_hom_gamma_omega_audits.zip'
    with zipfile.ZipFile(OUT/archive_name,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob('*')):
            if 'initial_preview' in path.parts or path.name in ['completion.json','publication_manifest.json']:
                continue
            if path.is_file() and path.suffix in ['.csv','.gz','.json','.npz','.npy','.pkl']:
                archive.write(path,path.relative_to(OUT))
        for name in SOURCES:
            archive.write(ROOT/'asd_analysis'/name,'source/asd_analysis/'+name)
        reference=ROOT/'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py'
        archive.write(reference,'source/ASD_Human_Data/'+reference.name)
    shutil.copy2(OUT/archive_name,ASSETS/archive_name)

# %% Replace only this analysis block; preserve all existing book content
body+='\nSources: '+', '.join(f'[{name}]({base}{name})' for name in SOURCES)+'.\n'
body+='\nExact source paths: `asd_analysis/` followed by those filenames. Outputs: `asd_analysis/shank3_cohort1_hom_gamma_omega/`.\n'
text=START+'\n'+body+'\n'+END
page=PAGE.read_text() if PAGE.exists() else f'# Results — {DATE}\n'
if START in page:
    before,rest=page.split(START,1);_,after=rest.split(END,1)
    PAGE.write_text(before+text+after)
else:
    PAGE.write_text(page.rstrip()+'\n\n'+text+'\n')
index_path=ROOT/'docs/results/index.md'
index=index_path.read_text()
label=f'- [{TITLE}]({DATE}.md#shank3-cohort-1-hom-condition-specific-reactive-delays)'
if label not in index:
    index_path.write_text(index.rstrip()+'\n'+label+'\n')
assert f'results/{DATE}.md' in (ROOT/'mkdocs.yml').read_text()
if not args.pending:
    copied={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ASSETS.rglob('*') if p.is_file()}
    (OUT/'publication_manifest.json').write_text(json.dumps({'page':str(PAGE.relative_to(ROOT)),
        'direct_png_embeds':2,'primary_pdf_downloads':2,'copied_asset_hashes':copied},indent=2)+'\n')
print('Recorded pending status' if args.pending else 'Published two direct PNGs, matching PDFs, convergence and audits',PAGE)
