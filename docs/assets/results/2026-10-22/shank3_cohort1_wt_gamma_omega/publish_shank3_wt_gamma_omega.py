# %% Result-book settings and verified inputs
"""Publish the four final figures only after independent verification passes."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'asd_analysis/shank3_cohort1_wt_gamma_omega'
DATE='2026-10-22'
ASSETS=ROOT/'docs/assets/results'/DATE/'shank3_cohort1_wt_gamma_omega'
PAGE=ROOT/'docs/results'/f'{DATE}.md'
START='<!-- shank3-wt-gamma-omega:start -->'
END='<!-- shank3-wt-gamma-omega:end -->'
VARIANTS=['shared_delay','condition_delay']
LABELS={'shared_delay':'Shared reactive delay','condition_delay':'Condition-specific reactive delays'}
SOURCES=['fit_shank3_wt_gamma_omega.py','shank3_wt_gamma_omega_utils.py','plot_shank3_wt_gamma_omega.py',
    'verify_shank3_wt_gamma_omega.py','publish_shank3_wt_gamma_omega.py','proactive_lapse_no_led_utils.py',
    'finish_shank3_wt_gamma_omega.py']
validation=json.loads((OUT/'independent_validation.json').read_text())
manifest=json.loads((OUT/'plot_manifest.json').read_text())
assert validation['status']=='passed' and len(validation['plot_checks'])==2
assert validation['figure_artifact_hashes']==manifest['artifacts']
manifest['status']='independently_verified'
(OUT/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
ASSETS.mkdir(parents=True,exist_ok=True)
for name,digest in manifest['artifacts'].items():
    assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
    shutil.copy2(OUT/name,ASSETS/name)
for name in ['preparation.json','fixed_proactive.json','conditions.csv','membership.csv','prefit_validation.json',
             'independent_validation.json','plot_manifest.json','in_sample_comparison.csv',
             'gpu_migration.json','gpu_benchmark.json','stopping_rule_change.json','visual_inspection.json']:
    shutil.copy2(OUT/name,ASSETS/name)
for name in SOURCES:
    shutil.copy2(ROOT/'asd_analysis'/name,ASSETS/name)
for variant in VARIANTS:
    (ASSETS/variant).mkdir(exist_ok=True)
    for name in ['run_summary.json','configuration.json','posterior_summary.csv','psychometric_predictions.csv',
                 'rtd_panel_audit.csv','condition_parameter_means.csv']:
        shutil.copy2(OUT/variant/name,ASSETS/variant/name)
archive_name='shank3_cohort1_wt_gamma_omega_audits.zip'
with zipfile.ZipFile(OUT/archive_name,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob('*')):
        if 'initial_preview' in path.parts:
            continue
        if path.name in ['completion.json','publication_manifest.json']:
            continue  # These describe the later publication stage, not fitting.
        if path.is_file() and path.suffix in ['.csv','.gz','.json','.npz','.npy']:
            archive.write(path,path.relative_to(OUT))
    for name in SOURCES:
        archive.write(ROOT/'asd_analysis'/name,'source/asd_analysis/'+name)
    reference=ROOT/'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py'
    archive.write(reference,'source/ASD_Human_Data/'+reference.name)
shutil.copy2(OUT/archive_name,ASSETS/archive_name)
base=f'../assets/results/{DATE}/shank3_cohort1_wt_gamma_omega/'
fixed=json.loads((OUT/'fixed_proactive.json').read_text())
fixed_rows='\n'.join(f'| `{name}` | {value:.17g} |' for name,value in zip(fixed['parameter_order'],fixed['values']))
comparison=pd.read_csv(OUT/'in_sample_comparison.csv')
comparison_rows='\n'.join(f'| {LABELS[r.variant]} | {r.parameters} | {r.joint_log_likelihood:,.3f} | '
    f'{r.psychometric_rmse:.5f} | {r.mean_rt_cdf_max_gap_1ms:.5f} | {r.worst_rt_cdf_max_gap_1ms:.5f} |'
    for r in comparison.itertuples())
run_rows=[]
for variant in VARIANTS:
    run=json.loads((OUT/variant/'run_summary.json').read_text())
    run_rows.append(f'| {LABELS[variant]} | {run["restored_best_step"]:,} | {run["final_checked_step"]:,} | '
        f'{run["best_window_negative_elbo"]:,.3f} | **{run["status"]}** |')
run_table='\n'.join(run_rows)
figures=[]
for variant in VARIANTS:
    stem=f'SHANK3_cohort1_WT_{variant}'
    figures.append(f'''### {LABELS[variant]}: RT distributions

![{LABELS[variant]} WT RT distributions, three ABL rows and five absolute ILD columns]({base}{stem}_RTDs.png)

*Colored steps are 20 ms empirical bins (ABL 20 blue, 40 orange, 60 green); black curves evaluate the full model at 1 ms. Each panel pools actual positive/negative ILD trials and then normalizes **data and model separately to area one over [−1,1] s**. Model predictions average the exact planned-onset times and signed-condition frequencies of all 83,921 eligible parent trials. The 81,934 displayed observations comprise 68,184 fitted choices and 13,750 pre-onset fixation aborts. Negative abort RT is `fix_time/1000 − intended_fix_time/1000`; valid RT is the recorded `timed_rt`. Late fixation-labelled rows are not displayed as aborts. Abort observations do not enter this new fit. Pre-normalization data/model window masses are saved in the panel audit. Both models share the same display axes.*

[Full-size PNG]({base}{stem}_RTDs.png) · [PDF]({base}{stem}_RTDs.pdf) · [Panel counts and window masses]({base}{variant}/rtd_panel_audit.csv)

### {LABELS[variant]}: psychometric comparison

![{LABELS[variant]} WT measured and predicted choice probabilities by ABL]({base}{stem}_psychometric.png)

*Dots show measured right-choice fractions among the 68,184 fitted valid trials. The model integrates the complete right-choice mixture density over [0,1] s, divides by that trial's complete either-choice window probability, and then averages over the exact onset times of valid trials in that signed condition. ABL colors match the RTD grid. Lines connect predictions at the ten measured ILDs; they are not a fitted sigmoid or a claim about intermediate conditions. All predictions use physical posterior means, with proactive/lapse means frozen.*

[Full-size PNG]({base}{stem}_psychometric.png) · [PDF]({base}{stem}_psychometric.pdf) · [Probabilities and counts]({base}{variant}/psychometric_predictions.csv) · [Condition parameters]({base}{variant}/condition_parameter_means.csv)
''')
figure_text='\n'.join(figures)
text=f'''{START}
## SHANK3 cohort 1 WT: shared versus condition-specific reactive delays

Fit **68,184 valid choices from five WT animals** in the cleaned session-type-1 CSV: SHANK3 cohort 1, training level 16, `success ∈ {{1,−1}}`, finite recorded `timed_rt` in inclusive **[0,1] seconds**, ABL 20/40/60 after 59→60, and signed ILDs ±1/2/4/8/16. There are **37,482 right and 30,702 left choices** (`response_poke=1` and `−1`). The 373 valid choices above one second are excluded. Pool actual trials across animals and sessions; preserve the prepared exclusions without reapplying repeats. No abort or censored observations enter these objectives.

| Model | Gamma | Omega | Reactive delay | Other shared parameters | Total |
| --- | ---: | ---: | ---: | --- | ---: |
| Shared delay | 30 | 30 | 1 | w, del_go | **63** |
| Condition delay | 30 | 30 | 30 | w, del_go | **92** |

Each condition is a signed ABL–ILD pair, with no imposed ±ILD symmetry. `del_E` and `t_E_aff` denote the same reactive delay. The evidence-readout interval `del_go` is fitted and shared in both models. Diffusion is fixed at one; dimensionless bounds are −1/+1, start is `2w−1`, and evidence time is `Omega × (RT−t_E_aff)`.

### Frozen abort fit and joint-window likelihood

Reuse the unrounded WT seed-0 physical posterior means from [the preceding proactive + exponential-lapse fit](#shank3-cohort-1-proactive-exponential-lapse-fits-by-genotype). Those five parameters are held fixed; no LED, uniform lapse, new lapse parameters or choice-biased lapse is added.

| Frozen parameter | Value |
| --- | ---: |
{fixed_rows}

For each valid trial, planned onset is `c=intended_fix_time/1000` and absolute response time is `t=c+timed_rt`. Valid RT is **not reconstructed from fix_time**. The existing animal proactive/reactive race uses direct condition-specific Gamma/Omega evidence kernels and the shared readout interval after a proactive trigger. The exponential lapse clock starts at fixation; half its density contributes to each choice.

The fitted log density is `log q_choice(c+RT) − log[F_total(c+1)−F_total(c)]`, where `q` and `F_total` include both the ordinary race and the fixed lapse. **The complete mixture is normalized together**, not each component separately. Reactive delay is not capped at a condition's minimum RT: proactive responses and lapses allow earlier observations.

### Priors and optimization

| Parameter | Hard bounds | Flat part of trapezoidal prior |
| --- | --- | --- |
| Gamma, positive ILD | −1–20 | 0–12 |
| Gamma, negative ILD | −20–1 | −12–0 |
| Omega (s⁻¹) | 0.01–15 | 0.1–12 |
| t_E_aff (s) | 0–0.50 | 0.01–0.20 |
| w | 0–1 | 0.4–0.6 |
| del_go (s) | 0–0.20 | 0.02–0.199 |

The two models run concurrently, with **one seed-0 initialization each**. Float64 full-batch NumPyro uses a full-rank guide, clipped Adam (learning rate 0.0002, clip norm 1), and initial guide scale 0.1. Both start at w=0.5, del_go=0.08 s and reactive delay=0.05 s; condition delays start identical. Gamma starts at `0.5 × logit(measured P(right))`, and Omega uses the symmetric-DDM mean-time relation and condition mean RT minus 0.05 s, clipped inside the prior bounds.

The fits began on CPU and continued on an NVIDIA RTX 4090 from saved steps **28,000 (shared delay)** and **19,000 (condition delays)**. Optimizer, guide and random-number states transferred exactly; the source code, data, float64 precision, priors and stopping rules were unchanged. Original CPU checkpoints, configurations, histories and logs are preserved under `cpu_before_gpu/`. [Migration provenance]({base}gpu_migration.json) · [CPU/GPU numerical and timing checks]({base}gpu_benchmark.json). Posterior draws are independently reproduced from the saved guide; cross-device floating-point comparisons use a tight float64 tolerance rather than requiring bitwise identity.

Check 1,000-step windows after at least 50,000 updates. The user removed the requirement that every parameter's last-three-checkpoint posterior-mean range be below 0.1 posterior SD. **The final stopping rule requires the minimum 50,000 updates and 25 windows since the absolute best loss.** Both runs had already met these conditions when the rule changed, so no additional updates were taken. Restore the absolute-best checkpoints at **49,000 shared-delay / 141,000 condition-delay steps**, retaining the complete later histories. The 50,000-step minimum applies to the checked run, not the step of its restored best checkpoint. Status **converged** means loss-patience stopping under this revised rule; neither run passed the removed parameter-stability test. That diagnostic remains in the audits and convergence figure. Earlier configurations, source versions and complete states are preserved under `before_loss_only_stop/`. [Authorized stopping-rule change and archive hashes]({base}stopping_rule_change.json).

Each restored guide supplies 10,000 posterior draws and central 95% intervals. A 1,000,000-step limit hit would be **inconclusive**. Uncertainty is conditional on the frozen abort parameters; initialization sensitivity remains untested.

| Model | Restored best step | Final checked step | Best window negative ELBO | Status |
| --- | ---: | ---: | ---: | --- |
{run_table}

[Convergence PNG]({base}SHANK3_cohort1_WT_gamma_omega_convergence.png) · [Convergence PDF]({base}SHANK3_cohort1_WT_gamma_omega_convergence.pdf) · [Shared-delay posterior summary]({base}shared_delay/posterior_summary.csv) · [Condition-delay posterior summary]({base}condition_delay/posterior_summary.csv)

{figure_text}

### In-sample comparison and reproducibility

These comparisons use the **same fitted trials**. Log likelihood is joint RT+choice at posterior means; larger is better. Psychometric RMSE gives equal weight to the 30 signed conditions. CDF gaps are measured on a 1 ms grid within [0,1] s. The additional parameters can improve in-sample fit; these metrics are not held-out prediction or evidence of generalization.

| Model | Parameters | Joint log likelihood | Psychometric RMSE | Mean condition max CDF gap | Worst condition max CDF gap |
| --- | ---: | ---: | ---: | ---: | ---: |
{comparison_rows}

[Full comparison, including trial-weighted metrics]({base}in_sample_comparison.csv) · [Independent verification]({base}independent_validation.json) · [Prefit checks]({base}prefit_validation.json) · [Memberships]({base}membership.csv) · [Condition counts]({base}conditions.csv) · [Source hashes and preparation]({base}preparation.json) · [Audit archive]({base}{archive_name})

Sources: [fit driver]({base}fit_shank3_wt_gamma_omega.py), [Gamma/Omega race likelihood]({base}shank3_wt_gamma_omega_utils.py), [plotter]({base}plot_shank3_wt_gamma_omega.py), [independent verifier]({base}verify_shank3_wt_gamma_omega.py), [publisher]({base}publish_shank3_wt_gamma_omega.py). Exact script paths are `asd_analysis/` followed by those filenames. Outputs: `asd_analysis/shank3_cohort1_wt_gamma_omega/`, with separate `shared_delay/` and `condition_delay/` folders.

The independent verifier reconstructs trial identities, fixed parameters, counts, signed pooling and timing coordinates; compares Wiener series and the complete likelihood with SciPy calculations; checks finite gradients and shared-delay broadcasting; reproduces posterior draws, best-checkpoint restoration and convergence classification; and independently verifies every model-grid value, choice probability, histogram area and PDF page count. Input CSVs and prior abort-fit/SHANK3 numerical exports retain their recorded hashes. Full optimizer/guide states remain in the output folders; the archive includes trials, posterior draws, complete losses, convergence records and numerical plot audits.
{END}
'''
page=PAGE.read_text()
if START in page:
    before,rest=page.split(START,1)
    _,after=rest.split(END,1)
    PAGE.write_text(before+text+after)
else:
    PAGE.write_text(page.rstrip()+'\n\n'+text)
index_path=ROOT/'docs/results/index.md'
index=index_path.read_text()
label=f'- [SHANK3 cohort 1 WT: shared versus condition-specific reactive delays]({DATE}.md#shank3-cohort-1-wt-shared-versus-condition-specific-reactive-delays)'
if label not in index:
    index_path.write_text(index.rstrip()+'\n'+label+'\n')
assert f'results/{DATE}.md' in (ROOT/'mkdocs.yml').read_text()
copied={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ASSETS.rglob('*') if p.is_file()}
(OUT/'publication_manifest.json').write_text(json.dumps({'page':str(PAGE.relative_to(ROOT)),
    'direct_png_embeds':4,'primary_pdf_downloads':4,'copied_asset_hashes':copied},indent=2)+'\n')
print('Published four direct PNG embeds, matching PDFs, convergence and verified numerical audits:',PAGE)
