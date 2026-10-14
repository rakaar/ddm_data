# %% Imports
"""Compare psychometrics for animals with the smallest JND/k changes across ABL.

Run with .venv/bin/python asd_analysis/plot_session_type_1_low_delta_psychometrics.py.
Reuse the verified October 13 fits and choice counts; do not refit or filter RT.
Both rankings use the same animals with three available paper-defined JNDs.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.special import expit


# %% Editable parameters
REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "asd_analysis/session_type_1_slope_jnd"
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO / "asd_analysis/session_type_1_low_delta_psychometrics"
ABLS = [20, 40, 60]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
ABL_MARKERS = {20: "o", 40: "s", 60: "^"}
GENOTYPE_LABELS = {"wt": "WT", "het": "HET", "hom": "HOM"}
TOP_N = 5
FIGSIZE = (20, 8)
PNG_DPI = 180
XLIM = (-18, 18)
YLIM = (0, 1)
SMOOTH_POINTS = 501
NAME = "ASD_session_type_1_low_delta_jnd_slope_psychometrics"
EXPECTED_TOP = {
    "jnd": ["ASD0061", "ASD0055", "ASD0042", "ASD0063", "ASD0027"],
    "k": ["ASD0050", "ASD0055", "ASD0051", "ASD0064", "ASD0039"],
}
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Protect the prepared/original CSVs, saved fits, and previous analysis exports
reference_manifest = json.loads((REFERENCE / "slope_jnd_validation.json").read_text())
reference_verification = json.loads((REFERENCE / "slope_jnd_independent_validation.json").read_text())
assert reference_manifest["status"] == reference_verification["status"] == "passed"
protected = set(REPO.glob("SHANK3_cohort*.csv")) | set(REPO.glob("CNTNAP2_cohort*.csv"))
protected.update(INPUT.parent.glob("*.csv"))
protected.update(REFERENCE.iterdir())
for folder in (REPO / "asd_analysis").glob("session_type_1_*"):
    if folder.is_dir() and folder != OUTPUT:
        protected.update(p for p in folder.rglob("*") if p.is_file() and p.suffix in {".csv", ".json", ".pdf", ".png"})
for date in ["2026-10-12", "2026-10-13", "2026-10-14"]:
    protected.add(REPO / f"docs/results/{date}.md")
    protected.update((REPO / f"docs/assets/results/{date}").iterdir())
protected_hashes = {}
for path in sorted(p for p in protected if p.is_file()):
    with path.open("rb") as stream:
        protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_sha256 = protected_hashes[str(INPUT.relative_to(REPO))]
assert input_sha256 == reference_manifest["input_sha256"]
print(f"Verified unchanged input and {len(protected_hashes)} protected files", flush=True)


# %% Read the saved animal probabilities and fits, retaining the existing selection
fits = pd.read_csv(REFERENCE / "slope_jnd_fits.csv")
fits = fits.loc[fits.view.eq("animal")].copy()
points = pd.read_csv(REFERENCE / "slope_jnd_condition_counts.csv")
points = points.loc[points.view.eq("animal")].copy()
assert len(fits) == 159 and len(points) == 1590
assert not fits.duplicated(["group", "genotype", "plot_ABL"]).any()
assert not points.duplicated(["group", "genotype", "plot_ABL", "ILD"]).any()
assert fits.fit_status.eq("ok").all() and np.isfinite(fits.k).all() and fits.k.gt(0).all()
np.testing.assert_array_equal(fits.k, fits.slope_k)
assert points.trials.sum() == 471416 and points.right_trials.sum() == 241539
assert points.left_trials.sum() == 229877
np.testing.assert_allclose(points.p_right, points.right_trials / points.trials, rtol=0, atol=2e-16)
for key, condition in points.groupby(["group", "genotype", "plot_ABL"]):
    assert condition.sort_values("ILD").ILD.to_list() == SIGNED_ILDS


# %% Audit all 53 animals, with common eligibility for both rankings
keys = ["group", "genotype"]
animals = points.groupby(keys, as_index=False).agg(choice_trials=("trials", "sum"),
    right_trials=("right_trials", "sum"), left_trials=("left_trials", "sum"))
animals[["filename", "animal"]] = animals.group.str.split("/", expand=True)
for metric in ["jnd", "k"]:
    values = fits.pivot(index=keys, columns="plot_ABL", values=metric)
    values.columns = [f"{metric}_ABL_{int(abl)}" for abl in values.columns]
    animals = animals.merge(values.reset_index(), on=keys, validate="one_to_one")
statuses = fits.pivot(index=keys, columns="plot_ABL", values="jnd_status")
statuses.columns = [f"jnd_status_ABL_{int(abl)}" for abl in statuses.columns]
animals = animals.merge(statuses.reset_index(), on=keys, validate="one_to_one")
animals["jnd_eligible"] = np.isfinite(animals[[f"jnd_ABL_{a}" for a in ABLS]]).all(axis=1)
animals["slope_eligible"] = np.isfinite(animals[[f"k_ABL_{a}" for a in ABLS]]).all(axis=1)
animals["common_eligible"] = animals.jnd_eligible & animals.slope_eligible
for metric, unit in [("jnd", "dB"), ("k", "per_dB")]:
    pair_columns = []
    for a, b in [(20, 40), (40, 60), (20, 60)]:
        column = f"abs_{metric}_{a}_minus_{b}_{unit}"
        animals[column] = (animals[f"{metric}_ABL_{a}"] - animals[f"{metric}_ABL_{b}"]).abs()
        pair_columns.append(column)
    # Require all three differences: missing JNDs never become a partial average.
    animals[f"delta_{metric}_{unit}"] = animals[pair_columns].sum(axis=1, min_count=3) / 3
    complete = animals[[f"{metric}_ABL_{a}" for a in ABLS]].dropna()
    np.testing.assert_allclose(animals.loc[complete.index, f"delta_{metric}_{unit}"],
        2 / 3 * (complete.max(axis=1) - complete.min(axis=1)), rtol=0, atol=2e-15)
animals["unavailable_jnd_ABLs"] = animals.apply(
    lambda row: ";".join(str(a) for a in ABLS if not np.isfinite(row[f"jnd_ABL_{a}"])), axis=1)
animals["exclusion_reason"] = np.where(animals.common_eligible, "", "One or more paper-defined JNDs unavailable")
assert len(animals) == 53 and animals.common_eligible.sum() == 50
assert set(animals.loc[~animals.common_eligible, "animal"]) == {"ASD0035", "ASD0044", "ASD0048"}
assert animals.loc[animals.common_eligible, "choice_trials"].sum() == 444718

rankings = {}
for metric, delta_column in [("jnd", "delta_jnd_dB"), ("k", "delta_k_per_dB")]:
    ranking = animals.loc[animals.common_eligible].sort_values([delta_column, "animal"], kind="stable").copy()
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    rankings[metric] = ranking
    ranking.to_csv(OUTPUT / f"{metric}_ranking.csv", index=False)
    assert ranking.head(TOP_N).animal.to_list() == EXPECTED_TOP[metric]
    rank_map = ranking.set_index("animal")["rank"]
    animals[f"{metric}_rank"] = animals.animal.map(rank_map).astype("Int64")
animals.sort_values("animal").to_csv(OUTPUT / "animal_eligibility_metrics.csv", index=False)
print("Top JND:", ", ".join(EXPECTED_TOP["jnd"]), flush=True)
print("Top k:", ", ".join(EXPECTED_TOP["k"]), flush=True)


# %% Plot measured points and the saved sigmoid parameters in two rows
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10})
fig, axes = plt.subplots(2, TOP_N, figsize=FIGSIZE, sharex=True, sharey=True)
fig.suptitle("Session type 1 | Animals with the smallest changes across ABL", fontsize=17, y=.985)
fig.legend(handles=[Line2D([], [], color=ABL_COLORS[a], marker=ABL_MARKERS[a], linestyle="-",
                          label=f"ABL {a} dB") for a in ABLS],
           loc="upper center", bbox_to_anchor=(.5, .948), ncol=3, frameon=False, fontsize=11)
fig.text(.05, .895, "Smallest ΔJND | five animals from the 50-animal common eligible set", fontsize=12, weight="bold")
fig.text(.05, .455, "Smallest Δk | same 50-animal eligible set | slope = fitted k", fontsize=12, weight="bold")
smooth_ild = np.linspace(*XLIM, SMOOTH_POINTS)
panel_records, selected_points, selected_fits, curve_records = [], [], [], []
for row_index, (metric, delta_column, delta_label, unit) in enumerate([
    ("jnd", "delta_jnd_dB", "ΔJND", "dB"), ("k", "delta_k_per_dB", "Δk", "dB⁻¹")]):
    for col_index, member in enumerate(rankings[metric].head(TOP_N).itertuples(index=False)):
        ax = axes[row_index, col_index]
        score = getattr(member, delta_column)
        ax.set_title(f"{member.animal} | {GENOTYPE_LABELS[member.genotype]}\n"
                     f"{member.filename.removesuffix('.csv')}\n"
                     f"Rank {member.rank} | {delta_label} = {score:.6f} {unit}", pad=9)
        ax.set(xlim=XLIM, ylim=YLIM, xlabel="Signed ILD (dB)")
        if col_index == 0:
            ax.set_ylabel("P(right)")
        ax.set_xticks([-16, -8, 0, 8, 16])
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.tick_params(labelbottom=True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
        for abl in ABLS:
            condition = points.loc[points.group.eq(member.group) & points.genotype.eq(member.genotype)
                                   & points.plot_ABL.eq(abl)].sort_values("ILD").copy()
            estimate = fits.loc[fits.group.eq(member.group) & fits.genotype.eq(member.genotype)
                                & fits.plot_ABL.eq(abl)].copy()
            assert len(estimate) == 1 and len(condition) == 10
            fit = estimate.iloc[0]
            measured = ax.scatter(condition.ILD, condition.p_right, color=ABL_COLORS[abl], marker=ABL_MARKERS[abl],
                                  s=28, zorder=3, label=f"ABL {abl} | n={int(condition.trials.sum()):,}")
            predicted = fit.lower + (fit.upper - fit.lower) * expit(fit.k * (smooth_ild - fit.x0))
            ax.plot(smooth_ild, predicted, color=ABL_COLORS[abl], linewidth=1.7)
            np.testing.assert_allclose(condition.fitted_p_right,
                fit.lower + (fit.upper - fit.lower) * expit(fit.k * (condition.ILD - fit.x0)), rtol=0, atol=2e-14)
            np.testing.assert_allclose(measured.get_offsets(), condition[["ILD", "p_right"]], rtol=0, atol=0)
            for table in [condition, estimate]:
                table.insert(0, "selection_metric", metric)
                table.insert(1, "selection_rank", int(member.rank))
                table.insert(2, "panel_row", row_index + 1)
                table.insert(3, "panel_column", col_index + 1)
                table.insert(4, "animal", member.animal)
            selected_points.append(condition)
            selected_fits.append(estimate)
            panel_records.append({"selection_metric": metric, "selection_rank": int(member.rank),
                "panel_row": row_index + 1, "panel_column": col_index + 1, "animal": member.animal,
                "filename": member.filename, "genotype": member.genotype, "plot_ABL": abl,
                "trials": int(condition.trials.sum()), "right_trials": int(condition.right_trials.sum()),
                "left_trials": int(condition.left_trials.sum()), "ILD_points": len(condition),
                "selection_delta": score, "delta_units": unit, "color": to_hex(ABL_COLORS[abl]), "marker": ABL_MARKERS[abl]})
            curve_records.append(pd.DataFrame({"selection_metric": metric, "selection_rank": member.rank,
                "animal": member.animal, "plot_ABL": abl, "ILD": smooth_ild, "fitted_p_right": predicted}))
        ax.legend(loc="upper left", frameon=False, fontsize=7.5, handletextpad=.35, borderaxespad=.2)
fig.text(.5, .058, "ΔM = (|M20 − M40| + |M40 − M60| + |M20 − M60|) / 3 | Rank using unrounded values | ASD0055 appears in both rows", ha="center", fontsize=10)
fig.text(.5, .034, "Training 16 | success in {1, −1} | no RT restriction | prepared repeat rule already applied | ABL 59 → 60 | signed ILDs ±1/2/4/8/16", ha="center", fontsize=9)
fig.text(.5, .010, "Common set: 50 animals, 444,718 choices | All 53 remain in the CSV | ΔJND and Δk have different units; these rankings describe stability across ABL", ha="center", fontsize=9)
fig.subplots_adjust(left=.05, right=.99, bottom=.13, top=.79, wspace=.28, hspace=.9)
pdf_path, png_path = OUTPUT / f"{NAME}.pdf", OUTPUT / f"{NAME}.png"
fig.savefig(pdf_path, metadata={"Title": "ASD psychometrics: lowest changes in JND and slope across ABL"})
fig.savefig(png_path, dpi=PNG_DPI, facecolor="white")
axis_bounds = [{"panel_row": r + 1, "panel_column": c + 1,
    "bounds": list(axes[r, c].get_position().bounds)} for r in range(2) for c in range(TOP_N)]
plt.close(fig)
panel_counts = pd.DataFrame(panel_records)
plotted_points = pd.concat(selected_points, ignore_index=True)
plotted_fits = pd.concat(selected_fits, ignore_index=True)
smooth_curves = pd.concat(curve_records, ignore_index=True)
assert len(panel_counts) == len(plotted_fits) == 30 and len(plotted_points) == 300
row_trial_counts = panel_counts.groupby("selection_metric").trials.sum().to_dict()
assert row_trial_counts == {"jnd": 43490, "k": 41474}
panel_counts.to_csv(OUTPUT / "selected_panel_counts.csv", index=False)
plotted_points.to_csv(OUTPUT / "selected_psychometric_points.csv", index=False)
plotted_fits.to_csv(OUTPUT / "selected_psychometric_fits.csv", index=False)
smooth_curves.to_csv(OUTPUT / "selected_sigmoid_curves.csv", index=False)


# %% Preserve earlier artifacts and save provenance, selection, and figure checks
for relative, digest in protected_hashes.items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, relative
manifest = {"status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "input_csv": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256,
    "reference_folder": str(REFERENCE.relative_to(REPO)), "selection": reference_manifest["selection"],
    "repeat_filter_reapplied": False, "rt_restriction": None, "refitted": False,
    "ABLs": ABLS, "ABL_mapping": {"59": 60}, "signed_ILDS": SIGNED_ILDS,
    "animals_audited": len(animals), "common_eligible_animals": int(animals.common_eligible.sum()),
    "ineligible_animals": sorted(animals.loc[~animals.common_eligible, "animal"].to_list()),
    "eligible_choices": int(animals.loc[animals.common_eligible, "choice_trials"].sum()),
    "full_selection": reference_manifest["observed_counts_per_view"],
    "metric_formula": "(|M20-M40| + |M40-M60| + |M20-M60|) / 3",
    "equivalent_formula": "2/3 * (max(M20,M40,M60) - min(M20,M40,M60))",
    "slope_definition": reference_manifest["slope_definition"], "jnd_definition": reference_manifest["jnd_definition"],
    "ranking": "Ascending unrounded mean absolute pairwise differences; exact ties by animal ID; common eligibility for both metrics",
    "top_animals": EXPECTED_TOP, "selected_row_choices": row_trial_counts, "panels": 10,
    "scatter_points": len(plotted_points), "sigmoid_curves": len(plotted_fits), "unique_selected_animals": plotted_points.animal.nunique(),
    "figure_inches": list(FIGSIZE), "png_dpi": PNG_DPI, "xlim": list(XLIM), "ylim": list(YLIM),
    "colors": {str(a): to_hex(c) for a, c in ABL_COLORS.items()}, "markers": ABL_MARKERS,
    "axis_bounds": axis_bounds, "pdf": str(pdf_path.relative_to(REPO)), "png": str(png_path.relative_to(REPO)),
    "protected_file_sha256": protected_hashes,
    "checks": ["53 animals audited; common eligible set of 50", "all three differences required; no partial missing-value averages",
        "pairwise means equal two-thirds of the range", "expected five animals per row", "444718 eligible choices",
        "43490/41474 choices across displayed rows", "existing fits and measured probabilities reused",
        "30 fitted curves and 300 measured points", "all protected input/previous-analysis hashes unchanged"]}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Saved 2 × 5 psychometrics: {pdf_path}", flush=True)
