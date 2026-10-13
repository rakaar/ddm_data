# %% Imports
"""P(right) by gene family and genotype, combining cohorts within each family.

Run with .venv/bin/python asd_analysis/plot_session_type_1_family_psychometrics.py.
The existing filename/genotype and combined-genotype images remain unchanged.
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
FAMILIES = ["SHANK3", "CNTNAP2"]
GENOTYPES = ["wt", "het", "hom"]
GENOTYPE_LABELS = {"wt": "WT", "het": "Heterozygous", "hom": "Homozygous"}
ABLS = [20, 40, 60]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
ABL_MARKERS = {20: "o", 40: "s", 60: "^"}
FIGSIZE = (17, 8)
PNG_DPI = 180
XLIM = (-18, 18)
YLIM = (0, 1)
EXPECTED = {"animals": 53, "trials": 471416, "right": 241539, "left": 229877}
EXPECTED_ANIMALS = {("SHANK3", "wt"): 6, ("SHANK3", "het"): 7, ("SHANK3", "hom"): 7,
                    ("CNTNAP2", "wt"): 10, ("CNTNAP2", "het"): 11, ("CNTNAP2", "hom"): 12}
PDF = OUTPUT / "ASD_session_type_1_family_genotype_psychometrics.pdf"
PNG = PDF.with_suffix(".png")
OUTPUT.mkdir(parents=True, exist_ok=True)
input_stat = [INPUT.stat().st_size, INPUT.stat().st_mtime_ns]
with INPUT.open("rb") as stream:
    input_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
reference = json.loads((OUTPUT / "psychometric_validation.json").read_text())
assert reference["status"] == "passed" and reference["input_sha256"] == input_sha256


# %% Same choice trials as the two existing views, without an RT restriction
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
selected["family"] = selected["filename"].str.extract(r"^(SHANK3|CNTNAP2)_cohort")[0]
assert selected["family"].isin(FAMILIES).all()
assert selected["genotype"].isin(GENOTYPES).all()
assert selected["response_poke"].isin([-1, 1]).all()
assert selected["response_poke"].eq(np.sign(selected["ILD"]) * selected["success"]).all()
assert {"ASD0026", "ASD0045"} <= set(selected["animal"])
selected["right"] = selected["response_poke"].eq(1)
selected["correct"] = selected["success"].eq(1)
observed = {"animals": selected["animal"].nunique(), "trials": len(selected),
            "right": int(selected["right"].sum()), "left": int((~selected["right"]).sum())}
assert observed == EXPECTED


# %% Pool raw trial counts within family/genotype; retain condition denominators
points = selected.groupby(["family", "genotype", "plot_ABL", "ILD"], as_index=False).agg(
    trials=("response_poke", "size"), right_trials=("right", "sum"),
    correct_trials=("correct", "sum"), animals=("animal", "nunique"),
)
points["left_trials"] = points["trials"] - points["right_trials"]
points["error_trials"] = points["trials"] - points["correct_trials"]
points["p_right"] = points["right_trials"] / points["trials"]
points["color"] = points["plot_ABL"].map({a: to_hex(c) for a, c in ABL_COLORS.items()})
points["marker"] = points["plot_ABL"].map(ABL_MARKERS)
assert len(points) == 180 and points["trials"].gt(0).all()
assert points["p_right"].between(0, 1).all()
groups = selected.groupby(["family", "genotype"], as_index=False).agg(
    trials=("response_poke", "size"), right_trials=("right", "sum"),
    correct_trials=("correct", "sum"), animals=("animal", "nunique"),
)
groups["left_trials"] = groups["trials"] - groups["right_trials"]
groups["error_trials"] = groups["trials"] - groups["correct_trials"]
assert len(groups) == 6
assert groups.set_index(["family", "genotype"])["animals"].to_dict() == EXPECTED_ANIMALS
assert groups[["trials", "right_trials", "left_trials"]].sum().tolist() == [471416, 241539, 229877]
membership = selected.groupby(["family", "genotype", "filename", "animal"], as_index=False).agg(
    trials=("response_poke", "size"), right_trials=("right", "sum"),
)
assert len(membership) == 53 and membership["animal"].nunique() == 53

# Independently verified existing cohort points must add to every new family point.
old_points = pd.read_csv(OUTPUT / "psychometric_points.csv")
cohort_points = old_points.loc[old_points["view"].eq("filename_genotype")].copy()
cohort_points["family"] = cohort_points["group"].str.extract(r"^(SHANK3|CNTNAP2)_cohort")[0]
keys = ["family", "genotype", "plot_ABL", "ILD"]
count_columns = ["trials", "right_trials", "left_trials", "correct_trials", "error_trials"]
pd.testing.assert_frame_equal(
    cohort_points.groupby(keys)[count_columns].sum().sort_index(),
    points.set_index(keys)[count_columns].sort_index(),
)
pd.testing.assert_frame_equal(
    points.groupby(["genotype", "plot_ABL", "ILD"])[count_columns].sum().sort_index(),
    old_points.loc[old_points["view"].eq("genotype")].set_index(["genotype", "plot_ABL", "ILD"])[count_columns].sort_index(),
)
points.to_csv(OUTPUT / "family_psychometric_points.csv", index=False)
groups.to_csv(OUTPUT / "family_psychometric_group_counts.csv", index=False)
membership.to_csv(OUTPUT / "family_psychometric_group_membership.csv", index=False)


# %% Two family rows, three genotype columns; all ABLs in each scatter panel
fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex=True, sharey=True)
fig.suptitle("Session type 1 | Psychometric P(right) | Gene family + genotype", fontsize=17, y=.985)
legend = [Line2D([], [], color=ABL_COLORS[a], marker=ABL_MARKERS[a], linestyle="None",
                 label=f"ABL {a} dB", markersize=7) for a in ABLS]
fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .945), ncol=3, frameon=False)
for row, family in enumerate(FAMILIES):
    for column, genotype in enumerate(GENOTYPES):
        ax = axes[row, column]
        info = groups.loc[groups["family"].eq(family) & groups["genotype"].eq(genotype)].iloc[0]
        ax.set_title(f"{family} | {GENOTYPE_LABELS[genotype]}\n{int(info['animals'])} animals | n={int(info['trials']):,}", fontsize=11, pad=10)
        for abl in ABLS:
            curve = points.loc[
                points["family"].eq(family) & points["genotype"].eq(genotype) & points["plot_ABL"].eq(abl)
            ].sort_values("ILD")
            assert len(curve) == 10
            scatter = ax.scatter(curve["ILD"], curve["p_right"], color=ABL_COLORS[abl],
                                 marker=ABL_MARKERS[abl], s=34, alpha=.85, clip_on=False, zorder=3)
            np.testing.assert_allclose(scatter.get_offsets(), curve[["ILD", "p_right"]].to_numpy())
            assert to_hex(scatter.get_facecolors()[0]) == to_hex(ABL_COLORS[abl])
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_xticks([-16, -8, -4, 0, 4, 8, 16])
        ax.set_xticks([-2, -1, 1, 2], minor=True)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        if column == 0:
            ax.set_ylabel("P(right)", fontsize=12)
        if row == 1:
            ax.set_xlabel("ILD (dB)", fontsize=12)
        assert len(ax.collections) == 3 and len(ax.lines) == 0
        assert tuple(ax.get_xlim()) == XLIM and tuple(ax.get_ylim()) == YLIM
fig.subplots_adjust(left=.065, right=.985, top=.82, bottom=.14, wspace=.13, hspace=.45)
fig.text(.5, .022,
         "Correct/error choices; no RT restriction. Training 16; confirmed repeat rule; short-sound rows removed from prepared CSV.\n"
         "Cohorts pooled within SHANK3 and within CNTNAP2 separately. ABL 59 → 60; signed ILD ±1/2/4/8/16. P(right) = right / all choices; trials pooled directly.",
         ha="center", va="bottom", fontsize=9)
fig.savefig(PDF)
fig.savefig(PNG, dpi=PNG_DPI)
plt.close(fig)
with fitz.open(PDF) as document:
    assert document.page_count == 1
    text = document[0].get_text()
    assert text.count("n=") == 6 and "SHANK3" in text and "CNTNAP2" in text
assert [INPUT.stat().st_size, INPUT.stat().st_mtime_ns] == input_stat


# %% Validation and provenance
manifest = {
    "status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_csv": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256,
    "input_file_size_and_mtime_ns": input_stat, "prepared_rows": len(data),
    "selection": reference["selection"], "short_sound_cleanup": reference["short_sound_cleanup"],
    "pooling": "Raw trials pooled across cohorts within each gene family and genotype; SHANK3 and CNTNAP2 remain separate",
    "weighting": reference["weighting"], "denominator": reference["denominator"],
    "right_choice": "response_poke == 1", "left_choice": "response_poke == -1",
    "family_order": FAMILIES, "genotype_order": GENOTYPES,
    "ABL_mapping": {"59": 60}, "ABLs": ABLS, "signed_ILDS": SIGNED_ILDS,
    "ABL_colors": {str(a): to_hex(c) for a, c in ABL_COLORS.items()},
    "ABL_markers": ABL_MARKERS, "xlim": XLIM, "ylim": YLIM, "scatter_only": True,
    "observed_counts": observed, "groups": 6, "points_checked": 180,
    "group_counts": groups.to_dict(orient="records"),
    "included_trials_missing_or_nonfinite_timed_rt": int((~np.isfinite(selected["timed_rt"])).sum()),
    "all_condition_counts_match_summed_existing_cohort_counts": True,
    "family_condition_counts_sum_to_existing_combined_genotype_counts": True,
    "independent_reference": "asd_analysis/session_type_1_psychometrics/psychometric_points.csv",
    "pdf": str(PDF.relative_to(REPO)), "png": str(PNG.relative_to(REPO)), "pdf_pages": 1,
    "checks": ["same verified input and choice trials", "six family/genotype groups, 53 animals, 471416 trials",
               "all 180 points equal summed independently verified cohort counts", "family counts sum to preserved combined view",
               "plotted coordinates, colors, scatter-only series and display limits", "one-page PDF with six sample labels",
               "input CSV unchanged during analysis"],
}
(OUTPUT / "family_psychometric_validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Saved separate-family psychometrics:", json.dumps(observed), flush=True)
print(groups.to_string(index=False), flush=True)
print("All 180 condition counts and probabilities verified against the existing cohort audits.", flush=True)
