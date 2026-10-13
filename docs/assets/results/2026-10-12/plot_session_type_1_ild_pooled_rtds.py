# %% Imports
"""Three RTD images matching the psychometric panel layouts, pooling all ILDs.

Run .venv/bin/python asd_analysis/plot_session_type_1_ild_pooled_rtds.py.
Combine correct/error trials before histogramming within each ABL; normalize
over 0–1 s. Existing CSVs, psychometrics, and condition-specific RTDs are read only.
"""
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd


# %% Parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO / "asd_analysis/session_type_1_ild_pooled_rtds"
ANIMAL_OUTPUT = REPO / "asd_analysis/session_type_1_rtds"
PSYCHOMETRIC_OUTPUT = REPO / "asd_analysis/session_type_1_psychometrics"
SOURCE_FILES = ["SHANK3_cohort1.csv", "SHANK3_cohort2.csv", "CNTNAP2_cohort3.csv", "CNTNAP2_cohort4.csv"]
FAMILIES = ["SHANK3", "CNTNAP2"]
GENOTYPES = ["wt", "het", "hom"]
GENOTYPE_LABELS = {"wt": "WT", "het": "Heterozygous", "hom": "Homozygous"}
ABLS = [20, 40, 60]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
BIN_WIDTH = .020
RT_WINDOW = (0.0, 1.0)
# Display zoom only; histogram edges, normalization, and sample counts use RT_WINDOW.
DISPLAY_XLIM = (0.0, 0.6)
PNG_DPI = 180
EXPECTED = {"animals": 53, "eligible_choice_trials": 471416, "window_trials": 466978}
VIEWS = {
    "filename_genotype": {"groups": SOURCE_FILES, "column": "filename", "rows": 4, "figsize": (17, 13), "title": "Filename + genotype"},
    "genotype": {"groups": ["all_cohorts"], "column": None, "rows": 1, "figsize": (17, 4.8), "title": "Genotype only | SHANK3 and CNTNAP2 combined"},
    "family_genotype": {"groups": FAMILIES, "column": "family", "rows": 2, "figsize": (17, 8), "title": "Gene family + genotype | SHANK3 and CNTNAP2 separate"},
}
OUTPUT.mkdir(parents=True, exist_ok=True)
input_stat = [INPUT.stat().st_size, INPUT.stat().st_mtime_ns]
with INPUT.open("rb") as stream:
    input_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
reference = json.loads((PSYCHOMETRIC_OUTPUT / "psychometric_validation.json").read_text())
assert reference["status"] == "passed" and reference["input_sha256"] == input_sha256
animal_manifest = json.loads((ANIMAL_OUTPUT / "validation.json").read_text())
assert animal_manifest["input_sha256"] == input_sha256
previous_outputs = [ANIMAL_OUTPUT, PSYCHOMETRIC_OUTPUT, REPO / "asd_analysis/session_type_1_genotype_rtds"]
previous_hashes = {
    str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
    for folder in previous_outputs for path in folder.iterdir() if path.is_file()
}


# %% Retain the confirmed choice selection, then apply the RT window
data = pd.read_csv(INPUT, usecols=[
    "filename", "animal", "genotype", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "short_duration", "ABL", "ILD", "success", "timed_rt",
], low_memory=False)
assert len(data) == 642370 and data["animal"].nunique() == 53
assert data["session_type"].eq(1).all() and data["training_level"].eq(16).all()
assert data["trial_is_repeat"].eq(False).all()
assert data["is_short_sound"].eq(0).all() and data["short_duration"].eq(0).all()
data["plot_ABL"] = data["ABL"].replace({59: 60})
selected = data.loc[
    data["success"].isin([1, -1]) & data["plot_ABL"].isin(ABLS) & data["ILD"].isin(SIGNED_ILDS)
].copy()
selected["plot_ABL"] = selected["plot_ABL"].astype(int)
selected["family"] = selected["filename"].str.extract(r"^(SHANK3|CNTNAP2)_cohort")[0]
assert selected["family"].isin(FAMILIES).all() and selected["genotype"].isin(GENOTYPES).all()
assert set(selected["filename"]) == set(SOURCE_FILES)
selected["finite_rt"] = np.isfinite(selected["timed_rt"])
selected["in_window"] = selected["finite_rt"] & selected["timed_rt"].between(*RT_WINDOW, inclusive="both")
observed = {"animals": selected.loc[selected["in_window"], "animal"].nunique(),
            "eligible_choice_trials": len(selected), "window_trials": int(selected["in_window"].sum())}
assert observed == EXPECTED, observed
assert {"ASD0026", "ASD0045"} <= set(selected.loc[selected["in_window"], "animal"])
print("Choice and RT-window counts:", json.dumps(observed), flush=True)
edges = np.linspace(*RT_WINDOW, round((RT_WINDOW[1] - RT_WINDOW[0]) / BIN_WIDTH) + 1)
np.testing.assert_allclose(np.diff(edges), BIN_WIDTH, rtol=0, atol=1e-15)


# %% Reference bin counts: sum existing animal histograms across all five |ILD|s
animal_bins = pd.read_csv(ANIMAL_OUTPUT / "histogram_bins.csv")
animal_bins = animal_bins.loc[animal_bins["panel"].eq("correct_error")].copy()
animal_bins["plot_ABL"] = animal_bins["plot_ABL"].astype(float).astype(int)
animal_groups = pd.read_csv(ANIMAL_OUTPUT / "animal_counts.csv")
animal_bins = animal_bins.merge(animal_groups[["filename", "animal", "genotype"]],
                                on=["filename", "animal"], validate="many_to_one")
animal_bins["family"] = animal_bins["filename"].str.extract(r"^(SHANK3|CNTNAP2)_cohort")[0]
assert int(animal_bins["bin_count"].sum()) == EXPECTED["window_trials"]


# %% Raw-trial pooling, per-ABL histograms, group membership, and count audits
histograms, curve_counts, group_counts, membership, curves = [], [], [], [], {}
for view, config in VIEWS.items():
    rows = selected.copy()
    rows["group"] = rows[config["column"]] if config["column"] else "all_cohorts"
    comparison = animal_bins.copy()
    comparison["group"] = comparison[config["column"]] if config["column"] else "all_cohorts"
    reference_bins = comparison.groupby(["group", "genotype", "plot_ABL", "bin_index"])["bin_count"].sum()
    assert rows.groupby(["group", "genotype"]).ngroups == config["rows"] * 3
    for group in config["groups"]:
        for genotype in GENOTYPES:
            panel_rows = rows.loc[rows["group"].eq(group) & rows["genotype"].eq(genotype)]
            window = panel_rows["in_window"]
            group_counts.append({
                "view": view, "group": group, "genotype": genotype,
                "eligible_choice_trials": len(panel_rows), "window_trials": int(window.sum()),
                "eligible_animals": panel_rows["animal"].nunique(),
                "window_animals": panel_rows.loc[window, "animal"].nunique(),
            })
            for (filename, animal), animal_rows in panel_rows.groupby(["filename", "animal"]):
                membership.append({
                    "view": view, "group": group, "genotype": genotype, "filename": filename,
                    "animal": animal, "eligible_choice_trials": len(animal_rows),
                    "window_trials": int(animal_rows["in_window"].sum()),
                })
            for abl in ABLS:
                condition = panel_rows.loc[panel_rows["plot_ABL"].eq(abl)]
                rt = condition.loc[condition["in_window"], "timed_rt"].to_numpy()
                counts, _ = np.histogram(rt, bins=edges)
                n = len(rt)
                assert n > 0 and int(counts.sum()) == n
                np.testing.assert_array_equal(counts, reference_bins.loc[(group, genotype, abl)].sort_index().to_numpy())
                density = counts / (n * BIN_WIDTH)
                area = float(np.dot(density, np.diff(edges)))
                assert np.isclose(area, 1.0, rtol=0, atol=1e-12)
                key = (view, group, genotype, abl)
                curves[key] = (density, n)
                identity = {"view": view, "group": group, "genotype": genotype, "plot_ABL": abl}
                curve_counts.append({
                    **identity, "color": to_hex(ABL_COLORS[abl]), "eligible_choice_trials": len(condition),
                    "window_trials": n, "window_animals": condition.loc[condition["in_window"], "animal"].nunique(),
                    "nonfinite_rt_trials": int((~condition["finite_rt"]).sum()),
                    "below_window_trials": int((condition["finite_rt"] & condition["timed_rt"].lt(RT_WINDOW[0])).sum()),
                    "above_window_trials": int((condition["finite_rt"] & condition["timed_rt"].gt(RT_WINDOW[1])).sum()),
                    "correct_window_trials": int(condition.loc[condition["in_window"], "success"].eq(1).sum()),
                    "error_window_trials": int(condition.loc[condition["in_window"], "success"].eq(-1).sum()),
                    "histogram_count": int(counts.sum()), "density_area": area,
                })
                for index, (left, right, count, value) in enumerate(zip(edges[:-1], edges[1:], counts, density)):
                    histograms.append({
                        **identity, "bin_index": index, "bin_left_s": left, "bin_right_s": right,
                        "bin_count": int(count), "window_trials": n, "density_per_s": value,
                        "color": to_hex(ABL_COLORS[abl]),
                    })

histograms = pd.DataFrame(histograms)
curve_counts = pd.DataFrame(curve_counts)
group_counts = pd.DataFrame(group_counts)
membership = pd.DataFrame(membership)
assert len(histograms) == 3150 and len(curve_counts) == 63 and len(group_counts) == 21
for view, config in VIEWS.items():
    assert group_counts.loc[group_counts["view"].eq(view), "window_trials"].sum() == 466978
    assert curve_counts.loc[curve_counts["view"].eq(view), "eligible_choice_trials"].sum() == 471416
    assert membership.loc[membership["view"].eq(view), "animal"].nunique() == 53
    assert membership.loc[membership["view"].eq(view)].shape[0] == 53
for view in ["genotype", "family_genotype"]:
    coarse = ["genotype", "plot_ABL", "bin_index"]
    if view == "family_genotype":
        coarse.insert(0, "family")
    detailed = histograms.loc[histograms["view"].eq("filename_genotype")].copy()
    collapsed = histograms.loc[histograms["view"].eq(view)].copy()
    detailed["family"] = detailed["group"].str.extract(r"^(SHANK3|CNTNAP2)_cohort")[0]
    collapsed["family"] = collapsed["group"]
    pd.testing.assert_series_equal(detailed.groupby(coarse)["bin_count"].sum().sort_index(),
                                   collapsed.groupby(coarse)["bin_count"].sum().sort_index())
histograms.to_csv(OUTPUT / "ild_pooled_rtd_histogram_bins.csv", index=False)
curve_counts.to_csv(OUTPUT / "ild_pooled_rtd_curve_counts.csv", index=False)
group_counts.to_csv(OUTPUT / "ild_pooled_rtd_group_counts.csv", index=False)
membership.to_csv(OUTPUT / "ild_pooled_rtd_group_membership.csv", index=False)


# %% Three images: psychometric panel layouts, 0–0.6 s display zoom
ylim = (0, float(np.ceil(histograms["density_per_s"].max() * 1.12)))
pages = []
for view, config in VIEWS.items():
    fig, axes = plt.subplots(config["rows"], 3, figsize=config["figsize"], sharex=True, sharey=True, squeeze=False)
    fig.suptitle(f"Session type 1 | RTD, all signed ILDs pooled | {config['title']}", fontsize=16, y=.985)
    top = .88 if config["rows"] > 1 else .79
    bottom = .11 if config["rows"] == 4 else .16 if config["rows"] == 2 else .23
    for row_index, group in enumerate(config["groups"]):
        for column, genotype in enumerate(GENOTYPES):
            ax = axes[row_index, column]
            info = group_counts.loc[group_counts["view"].eq(view) & group_counts["group"].eq(group)
                                    & group_counts["genotype"].eq(genotype)].iloc[0]
            label = group.removesuffix(".csv") if view != "genotype" else "All four cohorts"
            animal_word = "animal" if int(info["window_animals"]) == 1 else "animals"
            ax.set_title(f"{label} | {GENOTYPE_LABELS[genotype]}\n{int(info['window_animals'])} {animal_word} | n={int(info['window_trials']):,}", fontsize=11, pad=9)
            for abl in ABLS:
                density, n = curves[(view, group, genotype, abl)]
                curve = ax.stairs(density, edges, color=ABL_COLORS[abl], linewidth=1.5, fill=False,
                                  label=f"ABL {abl}: n={n:,}")
                np.testing.assert_allclose(curve.get_data().values, density)
                np.testing.assert_allclose(curve.get_data().edges, edges)
                assert to_hex(curve.get_edgecolor()) == to_hex(ABL_COLORS[abl]) and not curve.get_fill()
            ax.set_xlim(*DISPLAY_XLIM)
            ax.set_ylim(*ylim)
            ax.set_xticks(np.linspace(*DISPLAY_XLIM, 7))
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="upper right", fontsize=9, frameon=False, handlelength=2)
            if column == 0:
                ax.set_ylabel("Probability density (s⁻¹)", fontsize=11)
            if row_index == config["rows"] - 1:
                ax.set_xlabel("RT relative to stimulus onset (s)", fontsize=11)
            assert len(ax.patches) == 3 and tuple(ax.get_xlim()) == DISPLAY_XLIM
    fig.subplots_adjust(left=.065, right=.985, top=top, bottom=bottom, wspace=.13, hspace=.44)
    fig.text(.5, .02,
             "Correct/error trials; finite timed_rt in [0, 1] s. Training 16; confirmed repeat rule; short-sound rows removed from prepared CSV.\n"
             "ABL 59 → 60; signed ILDs ±1/2/4/8/16 pooled within each ABL. Solid 20 ms steps; density = count / (curve n × 0.020 s); area 1. Raw trials pooled.\n"
             "Display 0–0.6 s only; normalization and printed n still use 0–1 s.",
             ha="center", va="bottom", fontsize=9)
    prefix = f"ASD_session_type_1_{view}_ild_pooled_rtds"
    pdf = OUTPUT / f"{prefix}.pdf"
    png = OUTPUT / f"{prefix}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    with fitz.open(pdf) as document:
        assert document.page_count == 1
        assert document[0].get_text().count("n=") == config["rows"] * 3 * 4
    pages.append({"view": view, "panels": config["rows"] * 3, "pdf_pages": 1,
                  "pdf": str(pdf.relative_to(REPO)), "png": str(png.relative_to(REPO))})


# %% Provenance, exclusions, normalization, and validation
assert [INPUT.stat().st_size, INPUT.stat().st_mtime_ns] == input_stat
with INPUT.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == input_sha256
for name, digest in previous_hashes.items():
    assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest, name
manifest = {
    "status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_csv": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256,
    "input_file_size_and_mtime_ns": input_stat, "prepared_rows": len(data),
    "selection": "session_type=1, training_level=16, prepared chronological next-trial repeat rule with CNP/Fixation exceptions; success in {1,-1}; finite timed_rt in inclusive [0,1] s",
    "short_sound_cleanup": reference["short_sound_cleanup"],
    "all_animals_included": True, "observed_counts_per_view": observed,
    "nonfinite_choice_rt_trials": int((~selected["finite_rt"]).sum()),
    "finite_choice_rt_outside_window": int((selected["finite_rt"] & ~selected["in_window"]).sum()),
    "ABL_mapping": {"59": 60}, "ABLs": ABLS, "signed_ILDS": SIGNED_ILDS,
    "pooling": "Pool raw trials across all ten signed ILDs and sessions within each ABL and group; animals with more window trials contribute more weight",
    "groupings": {view: {"groups": config["groups"], "panels": config["rows"] * 3} for view, config in VIEWS.items()},
    "genotype_order": GENOTYPES, "timing": "Recorded timed_rt in seconds relative to stimulus onset; no timing recomputation",
    "normalization": "bin_count / (curve window_trials × 0.020 s); each curve area 1 over [0,1] s",
    "bin_width_s": BIN_WIDTH, "bin_edges_s": edges.tolist(),
    "bin_boundary_rule": "Left-inclusive/right-exclusive; final bin includes RT=1",
    "display_xlim_s": DISPLAY_XLIM, "shared_ylim_all_panels_and_views": ylim,
    "ABL_colors": {str(abl): to_hex(color) for abl, color in ABL_COLORS.items()}, "unfilled_solid_steps": True,
    "curve_count": len(curve_counts), "histogram_bins_checked": len(histograms),
    "all_bins_match_summed_existing_animal_ILD_counts": True,
    "all_coarser_views_match_summed_filename_genotype_counts": True,
    "max_unit_area_error": float((curve_counts["density_area"] - 1).abs().max()),
    "existing_artifact_hashes_unchanged": previous_hashes, "pages": pages,
}
(OUTPUT / "ild_pooled_rtd_validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Saved three PNG images and matching PDFs; all 3,150 histogram bins match existing animal counts.", flush=True)
