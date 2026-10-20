# %%
"""Publish the verified human TIED continuation alongside the original results."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import zipfile

import pandas as pd
from human_reactive_tied_svi_utils import ROOT, PARAM_NAMES, sha256

OLD=ROOT/"ASD_Human_Data/reactive_tied_rt_svi"
NEW=ROOT/"ASD_Human_Data/reactive_tied_rt_svi_confirm25"
ASSETS=ROOT/"docs/assets/results/2026-10-20/human_tied_confirm25"
PAGE=ROOT/"docs/results/2026-10-20.md"
ASSETS.mkdir(parents=True,exist_ok=True)
numeric=json.loads((NEW/"independent_validation.json").read_text())
continuation=json.loads((NEW/"continuation_validation.json").read_text())
visual=json.loads((NEW/"visual_inspection.json").read_text())
assert numeric["status"]==continuation["status"]==visual["status"]=="passed"
q=pd.read_csv(NEW/"comparison/fit_quality_changes.csv")
elbo=pd.read_csv(NEW/"comparison/common_draw_elbo_summary.csv").set_index("Group")
parts=["## Human RT: continuation with strict minimum confirmation", "",
       "**Outcome:** the corrected continuation found lower variational losses and substantially "
       "different parameters in both groups. Improvements in the data fit are modest and mixed: "
       "the largest RT CDF gaps decrease (Control 0.471 → 0.437; ASD 0.347 → 0.310), "
       "but average signed-condition CDF gaps increase slightly. The major RTD shape mismatch remains.", "",
       "This continues the original Control and ASD fits from their exact saved optimizer, guide, "
       "and RNG states. **Data, priors, likelihood and five shared parameters per group are unchanged.** "
       "Condition = RT only: 6,600 Control trials from 11 participants and 4,800 ASD trials from 8; "
       "all retained choices and RTs, both ASLs and all signed differences, with no additional filter.", "",
       "Every lower **1,000-step mean negative ELBO** resets patience. Require **25 subsequent windows** "
       "without a new best, plus terminal and restored-best mean stability (<0.1 posterior SD over "
       "three checkpoints). Restore the lowest finite window checkpoint. The cap is 1,000,000 total "
       "updates per group; a capped unfinished confirmation remains flagged. This confirms the "
       "specified observed-window criterion, not a global optimum.", "",
       "| Group | Original endpoint | Continued best | Final checked | Post-best windows | Status |",
       "|---|---:|---:|---:|---:|---|"]
for group in ["Control","ASD"]:
    report=json.loads((NEW/group.lower()/"convergence_summary.json").read_text())
    status="confirmed; means stable" if not report["flagged"] else "**unconfirmed / flagged**"
    parts.append(f"| {group} | {report['source_step']:,} | {report['restored_best_step']:,} | "
                 f"{report['completed_steps']:,} | {report['confirmation_windows']} | {status} |")
parts += ["", "### Original versus continued psychometrics and RTDs", "",
    "![Original and continued human reactive-only TIED fits](../assets/results/2026-10-20/human_tied_confirm25/comparison/human_tied_original_vs_continued_zoom_0_1p5s.png)", "",
    "*Control above ASD. Solid model curves: continued fit; dotted model curves: original fit. "
    "Blue/orange identify ASL 20/40 cd/m²; points and light steps show the same empirical data. "
    "Both versions use arithmetic physical posterior means and the identical joint RT/choice "
    "likelihood conditional on 0–3 s. The 0–1.5 s view is a display zoom. Model brightness is "
    "`Mean_dB`; nominal `ASL_cd` remains the legend label.*", "",
    "[Full-size comparison PNG](../assets/results/2026-10-20/human_tied_confirm25/comparison/human_tied_original_vs_continued_zoom_0_1p5s.png) · "
    "[Comparison PDF](../assets/results/2026-10-20/human_tied_confirm25/comparison/human_tied_original_vs_continued_zoom_0_1p5s.pdf) · "
    "[Full 0–3 s PNG](../assets/results/2026-10-20/human_tied_confirm25/comparison/human_tied_original_vs_continued_full_0_3s.png) · "
    "[Full 0–3 s PDF](../assets/results/2026-10-20/human_tied_confirm25/comparison/human_tied_original_vs_continued_full_0_3s.pdf)", "",
    "### Did fit quality improve?", "",
    "| Group | Metric | Original | Continued | Change |", "|---|---|---:|---:|---:|"]
labels={"mean_parameter_joint_log_likelihood":"Joint log likelihood at means (higher better)",
        "psychometric_rmse":"Psychometric RMSE (lower better)",
        "psychometric_max_abs_error":"Largest choice-probability error",
        "mean_condition_cdf_error":"Mean signed-condition RT CDF gap",
        "max_condition_cdf_error":"Largest signed-condition RT CDF gap"}
for row in q.itertuples():
    parts.append(f"| {row.Group} | {labels[row.metric]} | {row.original:.5g} | {row.continued:.5g} | {row.continued_minus_original:+.5g} |")
parts += ["", "The likelihood and curve discrepancies assess the same trial population. "
          "A lower variational loss need not improve every descriptive curve-error metric.", "",
          "A separate evaluation uses **256 common random draws** per original/continued guide "
          "to compare expected negative ELBO without further training:", "",
          "| Group | Original negative ELBO | Continued negative ELBO | Paired change ± MC SEM |",
          "|---|---:|---:|---:|"]
for group,row in elbo.iterrows():
    parts.append(f"| {group} | {row.original_mean:.3f} | {row.continued_mean:.3f} | {row.paired_mean:+.3f} ± {row.paired_mc_sem:.3f} |")
parts += ["", "### Parameter changes", "",
          "| Group | Parameter | Original mean | Continued mean | Continued central 95% |",
          "|---|---|---:|---:|---|"]
for group in ["Control","ASD"]:
    before=pd.read_csv(OLD/group.lower()/"posterior_summary.csv").set_index("parameter")
    after=pd.read_csv(NEW/group.lower()/"posterior_summary.csv").set_index("parameter")
    for name in PARAM_NAMES:
        factor=1000 if name in ["T_0","t_E_aff"] else 1
        label=name+" (ms)" if factor==1000 else name
        r=after.loc[name]
        parts.append(f"| {group} | {label} | {before.loc[name,'mean']*factor:.5g} | {r['mean']*factor:.5g} | "
                     f"[{r.q025*factor:.5g}, {r.q975*factor:.5g}] |")
parts += ["", "*Intervals are variational approximations from 10,000 constrained draws per group. "
          "Pooled-trial uncertainty does not measure participant-to-participant variability. "
          "The continuation adds no new statistical initialization.*", "",
    "### RTDs at each absolute difference", "",
    "![Continued human TIED RTDs by absolute difference](../assets/results/2026-10-20/human_tied_confirm25/human_reactive_tied_rtd_by_delta_zoom_0_1p5s.png)", "",
    "*Control above ASD; columns |ΔdB| = 0.1, 0.3, 0.65, 1.3, 2. Pool the two signs with "
    "observed trial frequencies, sum both choices, and retain 0–3 s normalization. "
    "Empirical 20 ms steps and model 1 ms curves use the same ASL colors and matching axes.*", "",
    "[Full-size PNG](../assets/results/2026-10-20/human_tied_confirm25/human_reactive_tied_rtd_by_delta_zoom_0_1p5s.png) · "
    "[PDF](../assets/results/2026-10-20/human_tied_confirm25/human_reactive_tied_rtd_by_delta_zoom_0_1p5s.pdf) · "
    "[Full 0–3 s PNG](../assets/results/2026-10-20/human_tied_confirm25/human_reactive_tied_rtd_by_delta_full_0_3s.png) · "
    "[Full 0–3 s PDF](../assets/results/2026-10-20/human_tied_confirm25/human_reactive_tied_rtd_by_delta_full_0_3s.pdf)", "",
    "### Minimum-confirmation evidence", "",
    "![Strict post-best loss confirmation](../assets/results/2026-10-20/human_tied_confirm25/comparison/human_tied_continuation_confirmation.png)", "",
    "*Gray dashed lines mark the original endpoints; black dotted lines mark selected minima "
    "and the required 25-window counter. The saved convergence CSVs retain the entire original "
    "history and explicitly replay strict patience for every window.*", "",
    "[Confirmation PDF](../assets/results/2026-10-20/human_tied_confirm25/comparison/human_tied_continuation_confirmation.pdf) · "
    "[Control corner](../assets/results/2026-10-20/human_tied_confirm25/control/parameter_corner.png) · "
    "[ASD corner](../assets/results/2026-10-20/human_tied_confirm25/asd/parameter_corner.png) · "
    "[Control parameter traces](../assets/results/2026-10-20/human_tied_confirm25/control/convergence.png) · "
    "[ASD parameter traces](../assets/results/2026-10-20/human_tied_confirm25/asd/convergence.png)", "",
    "### Reproduction and audits", "",
    "Source fit: `ASD_Human_Data/reactive_tied_rt_svi/`; continued fit: "
    "`ASD_Human_Data/reactive_tied_rt_svi_confirm25/`.", "",
    "[Fit-quality changes](../assets/results/2026-10-20/human_tied_confirm25/comparison/fit_quality_changes.csv) · "
    "[Parameter changes](../assets/results/2026-10-20/human_tied_confirm25/comparison/parameter_changes.csv) · "
    "[Numerical/source-row verification](../assets/results/2026-10-20/human_tied_confirm25/independent_validation.json) · "
    "[Continuation verification](../assets/results/2026-10-20/human_tied_confirm25/continuation_validation.json) · "
    "[Full posterior/calculation/source audit ZIP](../assets/results/2026-10-20/human_tied_confirm25/human_tied_confirm25_audits.zip)", ""]
sources=["continue_human_reactive_tied_svi.py","compare_human_reactive_tied_continuation.py",
         "fit_human_reactive_tied_svi.py",
         "verify_human_reactive_tied_continuation.py","plot_human_reactive_tied_svi.py",
         "verify_human_reactive_tied_svi.py","human_reactive_tied_svi_utils.py",
         "publish_human_reactive_tied_continuation.py"]
for name in sources:
    parts.append(f"- `ASD_Human_Data/{name}` ([source](../assets/results/2026-10-20/human_tied_confirm25/source/{name})).")
parts += ["", f"Verification matched all 11,400 source trial identities, retained the exact original "
          f"raw-loss prefixes, replayed every patience reset, checked posterior summaries and "
          f"covariances, and verified K=40/80 series, density/CDF agreement, conditional mass "
          f"and finite gradients at initial/posterior values. All {continuation['original_fit_artifacts_preserved']} "
          f"original fit artifacts remain byte-identical. PNGs were visually inspected; PDFs passed text-bound checks.", ""]
section="\n".join(parts)
(NEW/"continuation_result.md").write_text(section)
for path in NEW.rglob("*"):
    if path.is_file() and path.suffix in [".png",".pdf",".csv",".json"]:
        destination=ASSETS/path.relative_to(NEW)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
(ASSETS/"source").mkdir(exist_ok=True)
for name in sources:shutil.copy2(ROOT/"ASD_Human_Data"/name,ASSETS/"source"/name)
kernel=ROOT/"fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py"
shutil.copy2(kernel,ASSETS/"source"/kernel.name)
archive=ASSETS/"human_tied_confirm25_audits.zip"
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for path in NEW.rglob("*"):
        if path.is_file() and path.suffix not in [".png",".pdf"]:z.write(path,"fit/"+str(path.relative_to(NEW)))
    for name in sources:z.write(ROOT/"ASD_Human_Data"/name,"source/"+name)
    z.write(kernel,"source/"+kernel.name)
text=PAGE.read_text()
heading="# Results — 2026-10-20\n"
assert parts[0] not in text and heading in text
PAGE.write_text(text.replace(heading,heading+"\n"+section+"\n\n---\n",1))
(NEW/"publication.json").write_text(json.dumps(dict(page=str(PAGE.relative_to(ROOT)),
    published_at_utc=datetime.now(timezone.utc).isoformat(),archive_sha256=sha256(archive),
    source_sha256=sha256(__file__)),indent=2))
print("Published continuation comparison:",PAGE)
