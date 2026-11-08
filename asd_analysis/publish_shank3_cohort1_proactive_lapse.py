# %% Publish the verified figures and portable audit/source downloads
"""Update only this analysis's marked block in the October 22 result book."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'asd_analysis/shank3_cohort1_proactive_lapse'
DATE = '2026-10-22'
ASSETS = REPO/'docs/assets/results'/DATE/'shank3_cohort1_proactive_lapse'
PAGE = REPO/'docs/results'/f'{DATE}.md'
START = '<!-- shank3-cohort1-proactive-lapse:start -->'
END = '<!-- shank3-cohort1-proactive-lapse:end -->'
STEMS = ['SHANK3_cohort1_proactive_lapse_abort_distributions',
         'SHANK3_cohort1_proactive_lapse_convergence', 'SHANK3_cohort1_proactive_lapse_parameters']
SCRIPTS = ['fit_shank3_cohort1_proactive_lapse.py', 'proactive_lapse_no_led_utils.py',
           'plot_shank3_cohort1_proactive_lapse.py', 'verify_shank3_cohort1_proactive_lapse.py',
           'publish_shank3_cohort1_proactive_lapse.py']
validation = json.loads((OUT/'independent_validation.json').read_text())
assert validation['status'] == 'passed'
ASSETS.mkdir(parents=True, exist_ok=True)
downloads = ['preparation.json', 'selection_counts.csv', 'animal_membership.csv', 'timing_discrepancies.csv',
    'selected_parameter_summary.csv', 'start_comparison.csv', 'selected_starts.json',
    'abort_probability_comparison.csv', 'prefit_validation.json', 'independent_validation.json', 'plot_manifest.json']
for name in downloads + [stem+'.'+ext for stem in STEMS for ext in ['png','pdf']]:
    shutil.copy2(OUT/name, ASSETS/name)
for name in SCRIPTS:
    shutil.copy2(REPO/'asd_analysis'/name, ASSETS/name)
archive_name = 'shank3_cohort1_proactive_lapse_audits.zip'
with zipfile.ZipFile(OUT/archive_name, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.suffix in ['.csv','.gz','.json','.npz']:
            # Full trial identities, posterior draws and losses; pickle states stay in fit root.
            archive.write(path, path.relative_to(OUT))
    for name in SCRIPTS:
        archive.write(REPO/'asd_analysis'/name, 'source/'+name)
shutil.copy2(OUT/archive_name, ASSETS/archive_name)
base = f'../assets/results/{DATE}/shank3_cohort1_proactive_lapse/'
counts = pd.read_csv(OUT/'selection_counts.csv')
probs = pd.read_csv(OUT/'abort_probability_comparison.csv')
selected = json.loads((OUT/'selected_starts.json').read_text())
count_rows = '\n'.join(f'| {r.genotype.upper()} | {r.animals} | {r.total_trials:,} | {r.pre_onset_aborts:,} | {r.censored_trials:,} |'
                      for r in counts.itertuples())
prob_rows = '\n'.join(f'| {r.genotype.upper()} | {r.observed_abort_probability:.3%} | {r.predicted_abort_probability:.3%} |'
                     for r in probs.itertuples())
fit_rows = '\n'.join(f'| {r["genotype"].upper()} | {r["seed"]} | {r["restored_best_step"]:,} | {r["final_checked_step"]:,} | {r["status"]} |'
                    for r in selected)
text = f'''{START}
## SHANK3 cohort 1: proactive + exponential-lapse fits by genotype

**Session type 1 is the standard RT condition**, as defined in `animal_structure_columns.txt:16`. The original `SHANK3_cohort1.csv` has types 1, 2 and 23; the cleaned `ASD_session_type_1.csv` used here contains only type 1.

Retain SHANK3 cohort 1, training level 16, ABL 20/40/60 after mapping 59→60, and signed ILDs ±1/2/4/8/16. Pool actual trials within genotype across animals, sessions and stimulus conditions. Preserve prepared repeat/short-sound exclusions; do not reapply the repeat filter. **No measured-RT window or negative-RT restriction applies to this fit.** This differs from the preceding negative-RT fixation-duration histogram.

| Genotype | Animals | Eligible trials | Pre-onset aborts | Censored |
| --- | ---: | ---: | ---: | ---: |
{count_rows}
| **Total** | **14** | **225,245** | **38,224** | **187,021** |

Use fixation coordinates: observed abort duration `fix_time/1000` seconds and planned onset `intended_fix_time/1000` seconds. Only fixation aborts **strictly before** planned onset contribute density. Valid choices contribute survival at onset, and their post-stimulus RTs are unused. The **1,381 fixation-labelled trials at/after onset** are also censored. Outcome labels remain authoritative for the **26 valid trials** with logged fixation durations slightly below onset. Early aborts are retained; there is no 300 ms truncation.

### Model and priors

The five parameters are proactive drift `V_A`, bound `theta_A`, combined delay `t_nd`, lapse fraction `p`, and exponential rate `beta_lapse`. Diffusion is fixed at 1. There are no LED or reactive-process parameters. The lapse clock starts at fixation onset; only the proactive process is delayed.

`f(t) = (1-p) f_A(t-t_nd) + p beta exp(-beta t)`

`S(c) = (1-p) [1-F_A(c-t_nd)] + p exp(-beta c)`

| Parameter | Hard bounds | Flat part of trapezoidal prior |
| --- | --- | --- |
| V_A, theta_A | 0.1–8 | 0.5–3 |
| t_nd (s) | 0–0.30 | 0.02–0.14 |
| p | 0–1 | 0.01–0.30 |
| beta_lapse (s⁻¹) | 0.001–20 | 0.5–5 |

### Abort distributions with exact censoring exposure

![SHANK3 cohort 1 genotype abort data and proactive plus lapse models]({base}{STEMS[0]}.png)

*WT/HOM/HET use blue/green/orange empirical steps. Each 20 ms bin is divided by **all eligible genotype trials × 0.020 s**; its full histogram area equals the observed pre-onset abort fraction. The fitted mixture and weighted proactive/lapse components use selected-start posterior means and multiply their densities by the fraction of trials whose planned onset is later than that time. Exact model bin masses are `mean[F(min(b,c))-F(min(a,c))]`. These curves are not normalized to area one.*

[Full-size PNG]({base}{STEMS[0]}.png) · [PDF]({base}{STEMS[0]}.pdf) · [Observed/model abort probabilities]({base}abort_probability_comparison.csv)

| Genotype | Observed abort probability | Model abort probability |
| --- | ---: | ---: |
{prob_rows}

### Two-start convergence and restored checkpoints

![Both SVI starts per genotype with restored-best and final-checked steps]({base}{STEMS[1]}.png)

*Float64 full-rank NumPyro SVI uses clipped Adam (learning rate 0.0002, clip norm 1), guide initial scale 0.1 and the exact full-data likelihood. Seeds 0/1 start at [1.6, 2.5, 0.09, 0.05, 5] and [3, 1, 0.04, 0.15, 2]. Check 1,000-step windows after a minimum 50,000 steps; patience is 12, with an absolute-best finite checkpoint restored. The initial 150,000-step budget may extend to 250,000 for the patience tail. A step-limit run is labelled inconclusive. Insets show steps ≥20,000 relative to each genotype's best restored window loss, making the plateau visible. Select the lowest restored window negative ELBO; within a 0.1% tie, use posterior-mean log likelihood on identical trials.*

[Full-size PNG]({base}{STEMS[1]}.png) · [PDF]({base}{STEMS[1]}.pdf) · [Both starts and selection]({base}start_comparison.csv)

| Genotype | Selected seed | Restored step | Final checked step | Status |
| --- | ---: | ---: | ---: | --- |
{fit_rows}

### Posterior parameter comparison

![Five proactive and exponential-lapse parameters by genotype]({base}{STEMS[2]}.png)

*Dots show physical-parameter posterior means; bars show central 95% **variational-posterior intervals**, using 10,000 draws from each selected restored guide. These are pooled-trial fit uncertainties, not variation or SEM across animals. Afferent and motor delays are represented by their combined delay because this no-LED model cannot identify them separately.*

[Full-size PNG]({base}{STEMS[2]}.png) · [PDF]({base}{STEMS[2]}.pdf) · [Parameter means and intervals]({base}selected_parameter_summary.csv)

### Reproduction and audits

Sources: [fitting driver]({base}fit_shank3_cohort1_proactive_lapse.py), [no-LED likelihood]({base}proactive_lapse_no_led_utils.py), [plotting script]({base}plot_shank3_cohort1_proactive_lapse.py), [independent verifier]({base}verify_shank3_cohort1_proactive_lapse.py), [publisher]({base}publish_shank3_cohort1_proactive_lapse.py).

Exact fit root: `asd_analysis/shank3_cohort1_proactive_lapse/`. Sources are under `asd_analysis/` with the filenames linked above. Full guide and optimizer states remain in the fit root; the downloadable archive includes trial classifications, memberships, timing discrepancies, source hashes, complete losses, convergence windows, posterior draws, parameter summaries and model-bin audits.

[Audit archive]({base}{archive_name}) · [Selection counts]({base}selection_counts.csv) · [Animal memberships]({base}animal_membership.csv) · [Timing discrepancies]({base}timing_discrepancies.csv) · [Pre-fit checks]({base}prefit_validation.json) · [Independent post-fit verification]({base}independent_validation.json) · [Source hashes and selection]({base}preparation.json)

The independent verifier reconstructs every trial identity and event/censor classification, checks the SciPy inverse-Gaussian/exponential likelihood and finite-difference gradients, delay boundaries, pure-component limits, normalized densities, exact censor-adjusted bin masses, posterior draws, restored guide/optimizer states and start selection. Input CSV hashes and earlier SHANK3 figures/audits are unchanged.
{END}
'''
page = PAGE.read_text()
if START in page:
    before, rest = page.split(START, 1)
    _, after = rest.split(END, 1)
    PAGE.write_text(before+text+after)
else:
    PAGE.write_text(page.rstrip()+'\n\n'+text)
index_path = REPO/'docs/results/index.md'
index = index_path.read_text()
index_label = f'- [SHANK3 cohort 1 proactive + exponential-lapse fits by genotype]({DATE}.md#shank3-cohort-1-proactive-exponential-lapse-fits-by-genotype)'
if index_label not in index:
    index_path.write_text(index.rstrip()+'\n'+index_label+'\n')
assert f'results/{DATE}.md' in (REPO/'mkdocs.yml').read_text()
copied = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ASSETS.iterdir() if p.is_file()}
(OUT/'publication_manifest.json').write_text(json.dumps({'page':str(PAGE.relative_to(REPO)),
    'direct_png_embeds':3, 'pdf_downloads':3, 'copied_asset_hashes':copied},indent=2)+'\n')
print('Published three direct PNG embeds, PDFs, source downloads and audits:', PAGE)
