# %% Imports
"""Empirical P(right) versus signed ILD, pooling recorded choice trials.

Run with .venv/bin/python asd_analysis/plot_session_type_1_psychometrics.py.
The genotype-only view combines SHANK3 and CNTNAP2, including all cohorts.
No reaction-time restriction, fitted curves, or animal-density averaging.
"""
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %% Parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO / "asd_analysis/session_type_1_psychometrics"
SOURCE_FILES = ["SHANK3_cohort1.csv", "SHANK3_cohort2.csv", "CNTNAP2_cohort3.csv", "CNTNAP2_cohort4.csv"]
GENOTYPES = ["wt", "het", "hom"]
GENOTYPE_LABELS = {"wt": "WT", "het": "Heterozygous", "hom": "Homozygous"}
ABLS = [20, 40, 60]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
ABL_MARKERS = {20: "o", 40: "s", 60: "^"}
XLIM = (-18, 18)
YLIM = (0, 1)
XTICKS = [-16, -8, -4, 0, 4, 8, 16]
PNG_DPI = 180
EXPECTED = {"animals": 53, "trials": 471416, "right": 241539, "left": 229877}
VIEWS = {
    "filename_genotype": {"groups": SOURCE_FILES, "rows": 4, "figsize": (17, 13), "points": 360},
    "genotype": {"groups": ["all_cohorts"], "rows": 1, "figsize": (17, 4.8), "points": 90},
}
OUTPUT.mkdir(parents=True, exist_ok=True)
input_stat = [INPUT.stat().st_size, INPUT.stat().st_mtime_ns]
with INPUT.open("rb") as stream:
    input_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
cleanup = json.loads((INPUT.parent / "short_sound_cleanup.json").read_text())
verification = json.loads((INPUT.parent / "verification.json").read_text())
assert verification["status"] == "passed"


# %% Keep all correct/error choices in the confirmed stimulus domain
data = pd.read_csv(INPUT, usecols=[
    "filename", "animal", "genotype", "session_type", "training_level", "trial_is_repeat",
    "is_short_sound", "short_duration", "ABL", "ILD", "success", "response_poke", "timed_rt",
], low_memory=False)
assert len(data) == 642370
assert data["session_type"].eq(1).all() and data["training_level"].eq(16).all()
assert data["trial_is_repeat"].eq(False).all()
assert data["is_short_sound"].eq(0).all() and data["short_duration"].eq(0).all()
data["plot_ABL"] = data["ABL"].replace({59: 60})
selected = data.loc[
    data["success"].isin([1, -1]) & data["plot_ABL"].isin(ABLS) & data["ILD"].isin(SIGNED_ILDS)
].copy()
assert selected["response_poke"].isin([-1, 1]).all()
assert set(selected["filename"]) == set(SOURCE_FILES)
assert selected["genotype"].isin(GENOTYPES).all()
assert selected.groupby("animal")["genotype"].nunique().eq(1).all()
assert {"ASD0026", "ASD0045"} <= set(selected["animal"])
selected["right"] = selected["response_poke"].eq(1)
selected["correct"] = selected["success"].eq(1)
observed = {
    "animals": selected["animal"].nunique(), "trials": len(selected),
    "right": int(selected["right"].sum()), "left": int((~selected["right"]).sum()),
}
assert observed == EXPECTED, observed
# Independent choice cross-check: the chosen side agrees with signed ILD/outcome.
assert selected["response_poke"].eq(np.sign(selected["ILD"]) * selected["success"]).all()
assert selected.groupby("genotype")["animal"].nunique().to_dict() == {"wt": 16, "het": 18, "hom": 19}
print("Psychometric selection:", json.dumps(observed), flush=True)


# %% Pool trial counts by condition; retain group and animal membership audits
points_parts, group_parts, member_parts = [], [], []
for view, config in VIEWS.items():
    rows = selected.copy()
    rows["group"] = rows["filename"] if view == "filename_genotype" else "all_cohorts"
    points = rows.groupby(["group", "genotype", "plot_ABL", "ILD"], as_index=False).agg(
        trials=("response_poke", "size"), right_trials=("right", "sum"),
        correct_trials=("correct", "sum"), animals=("animal", "nunique"),
    )
    points["left_trials"] = points["trials"] - points["right_trials"]
    points["error_trials"] = points["trials"] - points["correct_trials"]
    points["p_right"] = points["right_trials"] / points["trials"]
    points["color"] = points["plot_ABL"].map({a: to_hex(c) for a, c in ABL_COLORS.items()})
    points["marker"] = points["plot_ABL"].map(ABL_MARKERS)
    points.insert(0, "view", view)
    assert len(points) == config["points"] and points["trials"].gt(0).all()
    assert points["p_right"].between(0, 1).all()
    assert points[["trials", "right_trials", "left_trials"]].sum().tolist() == [471416, 241539, 229877]
    counts = rows.groupby(["group", "genotype"], as_index=False).agg(
        trials=("response_poke", "size"), right_trials=("right", "sum"),
        correct_trials=("correct", "sum"), animals=("animal", "nunique"),
    )
    counts["left_trials"] = counts["trials"] - counts["right_trials"]
    counts["error_trials"] = counts["trials"] - counts["correct_trials"]
    counts.insert(0, "view", view)
    assert len(counts) == len(config["groups"]) * 3 and counts["animals"].sum() == 53
    members = rows.groupby(["group", "genotype", "filename", "animal"], as_index=False).agg(
        trials=("response_poke", "size"), right_trials=("right", "sum"),
    )
    members.insert(0, "view", view)
    assert len(members) == 53
    points_parts.append(points)
    group_parts.append(counts)
    member_parts.append(members)
points = pd.concat(points_parts, ignore_index=True)
groups = pd.concat(group_parts, ignore_index=True)
membership = pd.concat(member_parts, ignore_index=True)
keys = ["genotype", "plot_ABL", "ILD"]
pd.testing.assert_frame_equal(
    points.loc[points["view"].eq("filename_genotype")].groupby(keys)[["trials", "right_trials", "left_trials", "correct_trials", "error_trials"]].sum().sort_index(),
    points.loc[points["view"].eq("genotype")].set_index(keys)[["trials", "right_trials", "left_trials", "correct_trials", "error_trials"]].sort_index(),
)
points.to_csv(OUTPUT / "psychometric_points.csv", index=False)
groups.to_csv(OUTPUT / "psychometric_group_counts.csv", index=False)
membership.to_csv(OUTPUT / "psychometric_group_membership.csv", index=False)


# %% Scatter figures: each panel contains all three ABL series
pages = []
for view, config in VIEWS.items():
    fig, axes = plt.subplots(config["rows"], 3, figsize=config["figsize"], sharex=True, sharey=True, squeeze=False)
    title = "Filename + genotype" if view == "filename_genotype" else "Genotype only | SHANK3 and CNTNAP2 cohorts combined"
    fig.suptitle(f"Session type 1 | Psychometric P(right) | {title}", fontsize=17, y=.985)
    legend = [Line2D([], [], color=ABL_COLORS[a], marker=ABL_MARKERS[a], linestyle="None", label=f"ABL {a} dB", markersize=7) for a in ABLS]
    legend_y = .95 if config["rows"] == 4 else .90
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, legend_y), ncol=3, frameon=False)
    for row_index, group in enumerate(config["groups"]):
        for column, genotype in enumerate(GENOTYPES):
            ax = axes[row_index, column]
            info = groups.loc[groups["view"].eq(view) & groups["group"].eq(group) & groups["genotype"].eq(genotype)].iloc[0]
            label = group.removesuffix(".csv") if view == "filename_genotype" else "All four cohorts"
            animal_word = "animal" if int(info["animals"]) == 1 else "animals"
            ax.set_title(f"{label} | {GENOTYPE_LABELS[genotype]}\n{int(info['animals'])} {animal_word} | n={int(info['trials']):,}", fontsize=11, pad=10)
            for abl in ABLS:
                curve = points.loc[
                    points["view"].eq(view) & points["group"].eq(group)
                    & points["genotype"].eq(genotype) & points["plot_ABL"].eq(abl)
                ].sort_values("ILD")
                assert len(curve) == 10
                scatter = ax.scatter(curve["ILD"], curve["p_right"], color=ABL_COLORS[abl], marker=ABL_MARKERS[abl], s=34, alpha=.85, clip_on=False, zorder=3)
                np.testing.assert_allclose(scatter.get_offsets(), curve[["ILD", "p_right"]].to_numpy())
                assert to_hex(scatter.get_facecolors()[0]) == to_hex(ABL_COLORS[abl])
            ax.set_xlim(*XLIM)
            ax.set_ylim(*YLIM)
            ax.set_xticks(XTICKS)
            ax.set_xticks([-2, -1, 1, 2], minor=True)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            if column == 0:
                ax.set_ylabel("P(right)", fontsize=12)
            if row_index == config["rows"] - 1:
                ax.set_xlabel("ILD (dB)", fontsize=12)
            assert tuple(ax.get_xlim()) == XLIM and tuple(ax.get_ylim()) == YLIM
            assert len(ax.collections) == 3 and len(ax.lines) == 0
    fig.subplots_adjust(left=.065, right=.985, top=.875 if config["rows"] == 4 else .69, bottom=.085 if config["rows"] == 4 else .22, wspace=.13, hspace=.45)
    fig.text(.5, .018 if config["rows"] == 4 else .025,
             "Correct/error choices; no RT restriction. Training 16; confirmed repeat rule; short-sound rows removed from prepared CSV.\n"
             "ABL 59 → 60; signed ILD ±1/2/4/8/16. P(right) = right / all choices at each condition. Trials pooled directly; all 53 animals retained.",
             ha="center", va="bottom", fontsize=9)
    pdf = OUTPUT / f"ASD_session_type_1_{view}_psychometrics.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    with fitz.open(pdf) as document:
        assert document.page_count == 1
        page_text = document[0].get_text()
        assert page_text.count("n=") == config["rows"] * 3
        assert "no RT restriction" in page_text and "P(right)" in page_text
    pages.append({"view": view, "panels": config["rows"] * 3, "points": config["points"],
                  "pdf": str(pdf.relative_to(REPO)), "png": str(png.relative_to(REPO)), "pdf_pages": 1})
    print(f"Saved {view}: {config['rows'] * 3} panels, {config['points']} scatter points", flush=True)


# %% Selection and validation manifest
assert [INPUT.stat().st_size, INPUT.stat().st_mtime_ns] == input_stat
manifest = {
    "status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_csv": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256,
    "input_file_size_and_mtime_ns": input_stat, "prepared_rows": len(data),
    "selection": "session_type=1; training_level=16; prepared next-trial repeat rule with CNP/Fixation exceptions; success in {1,-1}; no RT restriction",
    "short_sound_cleanup": {"rule": cleanup["rule"], "removed_prepared_rows": 807, "additional_plot_exclusions": 0},
    "ABL_mapping": {"59": 60}, "ABLs": ABLS, "signed_ILDS": SIGNED_ILDS,
    "right_choice": "response_poke == 1", "left_choice": "response_poke == -1",
    "denominator": "All recorded left/right correct/error choices at the group's signed ILD and mapped ABL",
    "weighting": "Pool raw trials; animals with more eligible trials contribute more weight. No equal-animal averages or deduplication.",
    "genotype_only_pooling": "Both gene families, SHANK3 and CNTNAP2, across all four cohorts",
    "genotype_order": GENOTYPES, "source_order": SOURCE_FILES,
    "observed_counts_per_view": observed,
    "genotype_animal_counts": {"wt": 16, "het": 18, "hom": 19},
    "included_trials_missing_or_nonfinite_timed_rt": int((~np.isfinite(selected["timed_rt"])).sum()),
    "ABL_colors": {str(a): to_hex(c) for a, c in ABL_COLORS.items()},
    "ABL_markers": ABL_MARKERS, "xlim": XLIM, "ylim": YLIM,
    "scatter_only": True, "points_checked": len(points), "pages": pages,
    "checks": ["prepared filters and zero remaining short-sound flags", "exact animal/trial/right/left totals",
               "recorded choices agree with signed ILD and success", "all condition denominators positive; P(right) in [0,1]",
               "genotype-only condition counts equal summed filename/genotype counts", "plotted point coordinates equal audits",
               "colors, limits, scatter-only series and sample labels", "two single-page PDFs", "input CSV unchanged during analysis"],
}
(OUTPUT / "psychometric_validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("All psychometric checks passed.", flush=True)
