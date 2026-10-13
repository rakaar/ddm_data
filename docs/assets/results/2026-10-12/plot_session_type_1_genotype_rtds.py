# %% Imports
"""Pool session-type-1 trials by filename/genotype and family/genotype.

Run with .venv/bin/python asd_analysis/plot_session_type_1_genotype_rtds.py.
Trials are combined before histogramming. The animal plots and all input CSVs
remain unchanged. Their saved bin counts independently verify both pooled views.
"""

import hashlib
import json
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %% Parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
ANIMAL_OUTPUT = REPO / "asd_analysis/session_type_1_rtds"
OUTPUT = REPO / "asd_analysis/session_type_1_genotype_rtds"
SOURCE_FILES = [
    "SHANK3_cohort1.csv", "SHANK3_cohort2.csv",
    "CNTNAP2_cohort3.csv", "CNTNAP2_cohort4.csv",
]
FAMILIES = ["SHANK3", "CNTNAP2"]
GENOTYPES = ["wt", "het", "hom"]
GENOTYPE_LABELS = {"wt": "WT", "het": "Heterozygous", "hom": "Homozygous"}
RESTORED_ANIMALS = ["ASD0026", "ASD0045"]
ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
POOLED_COLOR = "black"
BIN_WIDTH = 0.020
VALID_WINDOW = (0.0, 1.0)
POOLED_WINDOW = (-1.0, 1.0)
VALID_XLIM = (0.0, 0.6)
POOLED_XLIM = (-0.6, 0.6)
PNG_DPI = 160
EXPECTED_TOTALS = {
    "animals": 53, "selected_rows": 610234,
    "finite_selected_rows": 607207, "nonfinite_selected_rows": 3027,
    "valid_window_trials": 466978, "pooled_window_trials": 598864,
}
VIEWS = {
    "filename_genotype": {
        "group_column": "filename", "groups": SOURCE_FILES,
        "pages": [SOURCE_FILES[:2], SOURCE_FILES[2:]],
        "page_labels": ["SHANK3 cohorts", "CNTNAP2 cohorts"],
        "title": "Filename + genotype", "expected_groups": 12,
    },
    "family_genotype": {
        "group_column": "family", "groups": FAMILIES,
        "pages": [FAMILIES], "page_labels": ["Cohorts combined within each family"],
        "title": "Gene family + genotype", "expected_groups": 6,
    },
}

OUTPUT.mkdir(parents=True, exist_ok=True)
with INPUT.open("rb") as handle:
    input_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
input_paths = [INPUT] + [REPO / name for name in SOURCE_FILES]
input_stats = {
    str(path.relative_to(REPO)): [path.stat().st_size, path.stat().st_mtime_ns]
    for path in input_paths
}
animal_artifact_hashes = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in ANIMAL_OUTPUT.iterdir() if path.is_file()
}
animal_manifest = json.loads((ANIMAL_OUTPUT / "validation.json").read_text())
assert input_sha256 == animal_manifest["input_sha256"]


# %% Select the same trials as the existing animal analysis
data = pd.read_csv(INPUT, usecols=[
    "filename", "animal", "genotype", "source_file", "session", "training_level",
    "session_type", "trial_is_repeat", "ABL", "ILD", "success", "abort_type",
    "timed_rt", "is_short_sound", "short_duration",
], low_memory=False)
assert data["training_level"].eq(16).all() and data["session_type"].eq(1).all()
assert data["trial_is_repeat"].eq(False).all()
assert (data["success"].isin([1, -1]) | data["abort_type"].eq("Fixation")).all()
assert data["genotype"].isin(GENOTYPES).all()
assert data.groupby(["filename", "animal"])["genotype"].nunique().eq(1).all()
data["plot_ABL"] = data["ABL"].replace({59: 60})
data["abs_ILD"] = data["ILD"].abs()
short_rows = data["session_type"].eq(1) & data["is_short_sound"].eq(1)
assert len(data) == 642370 and int(short_rows.sum()) == 0
assert short_rows.equals(data["session_type"].eq(1) & data["short_duration"].gt(0))
selected = data.loc[
    ~short_rows
    & data["plot_ABL"].isin(ABLS) & data["abs_ILD"].isin(ABS_ILDS)
].copy()
selected["family"] = selected["filename"].str.extract(r"^(SHANK3|CNTNAP2)")[0]
assert selected["family"].notna().all()
assert set(selected["filename"]) == set(SOURCE_FILES)
assert set(RESTORED_ANIMALS).issubset(set(selected["animal"]))
assert selected["is_short_sound"].fillna(0).eq(0).all()
selected["finite_rt"] = np.isfinite(selected["timed_rt"])
selected["valid_window"] = (
    selected["success"].isin([1, -1]) & selected["finite_rt"]
    & selected["timed_rt"].between(*VALID_WINDOW, inclusive="both")
)
selected["pooled_window"] = (
    selected["finite_rt"]
    & selected["timed_rt"].between(*POOLED_WINDOW, inclusive="both")
)
totals = {
    "animals": selected["animal"].nunique(), "selected_rows": len(selected),
    "finite_selected_rows": int(selected["finite_rt"].sum()),
    "nonfinite_selected_rows": int((~selected["finite_rt"]).sum()),
    "valid_window_trials": int(selected["valid_window"].sum()),
    "pooled_window_trials": int(selected["pooled_window"].sum()),
}
assert totals == EXPECTED_TOTALS, totals
print("Confirmed selected trials:", json.dumps(totals), flush=True)

animal_rows = []
for (filename, animal), rows in selected.groupby(["filename", "animal"], sort=False):
    animal_rows.append({
        "filename": filename, "animal": animal, "family": rows["family"].iloc[0],
        "genotype": rows["genotype"].iloc[0], "selected_rows": len(rows),
        "finite_selected_rows": int(rows["finite_rt"].sum()),
        "valid_window_trials": int(rows["valid_window"].sum()),
        "pooled_window_trials": int(rows["pooled_window"].sum()),
        "mapped_ABL59_rows": int(rows["ABL"].eq(59).sum()),
        "daily_source_files": rows["source_file"].nunique(),
        "sessions": rows.groupby(["source_file", "session"]).ngroups,
    })
animal_counts = pd.DataFrame(animal_rows)
previous_animals = pd.read_csv(ANIMAL_OUTPUT / "animal_counts.csv")
compare_columns = [column for column in previous_animals if column != "pdf_page"]
pd.testing.assert_frame_equal(
    animal_counts[compare_columns].sort_values(["filename", "animal"]).reset_index(drop=True),
    previous_animals[compare_columns].sort_values(["filename", "animal"]).reset_index(drop=True),
)


# %% Pool raw trials before histogramming; audit each curve and group membership
valid_edges = np.linspace(*VALID_WINDOW, 51)
pooled_edges = np.linspace(*POOLED_WINDOW, 101)
curve_rows, bin_rows, group_rows, membership_parts = [], [], [], []
curves = {}


def record_histogram(rows, view, group, genotype, panel, abs_ild, abl, edges, color):
    """Histogram combined trials using the original window as the denominator."""
    rt = rows["timed_rt"].to_numpy()
    finite = np.isfinite(rt)
    in_window = finite & (rt >= edges[0]) & (rt <= edges[-1])
    counts, _ = np.histogram(rt[in_window], bins=edges)
    n = int(in_window.sum())
    density = counts / (n * BIN_WIDTH) if n else np.zeros(len(counts))
    area = float(np.dot(density, np.diff(edges)))
    assert counts.sum() == n
    assert np.isclose(area, 1.0 if n else 0.0, rtol=0, atol=1e-12)
    curve_id = f"{view}/{group}/{genotype}/{panel}/{abs_ild}/{abl}"
    identity = {
        "view": view, "group": group, "genotype": genotype, "curve_id": curve_id,
        "panel": panel, "abs_ILD": str(abs_ild), "plot_ABL": str(abl),
    }
    curve_rows.append({
        **identity, "color": to_hex(color), "eligible_rows": len(rows),
        "eligible_animals": rows["animal"].nunique(),
        "window_animals": rows.loc[in_window, "animal"].nunique(),
        "finite_rt_rows": int(finite.sum()), "nonfinite_rt_rows": int((~finite).sum()),
        "below_window_rows": int((finite & (rt < edges[0])).sum()),
        "above_window_rows": int((finite & (rt > edges[-1])).sum()),
        "window_trials": n, "histogram_count": int(counts.sum()), "density_area": area,
        "window_start_s": float(edges[0]), "window_end_s": float(edges[-1]),
        "bin_width_s": BIN_WIDTH,
        "display_xlim_start_s": VALID_XLIM[0] if panel == "correct_error" else POOLED_XLIM[0],
        "display_xlim_end_s": VALID_XLIM[1] if panel == "correct_error" else POOLED_XLIM[1],
        "window_correct_trials": int(rows.loc[in_window, "success"].eq(1).sum()),
        "window_error_trials": int(rows.loc[in_window, "success"].eq(-1).sum()),
        "window_fixation_abort_trials": int(rows.loc[in_window, "abort_type"].eq("Fixation").sum()),
    })
    for i, (left, right, count, value) in enumerate(zip(edges[:-1], edges[1:], counts, density)):
        bin_rows.append({
            **identity, "bin_index": i, "bin_left_s": left, "bin_right_s": right,
            "bin_count": int(count), "window_trials": n, "density_per_s": value,
            "color": to_hex(color),
        })
    curves[(view, group, genotype, panel, str(abs_ild), str(abl))] = (density, edges, n)


for view, config in VIEWS.items():
    column = config["group_column"]
    assert selected.groupby([column, "genotype"]).ngroups == config["expected_groups"]
    for group in config["groups"]:
        for genotype in GENOTYPES:
            rows = selected.loc[selected[column].eq(group) & selected["genotype"].eq(genotype)]
            assert len(rows) > 0, (view, group, genotype)
            valid_rows = rows.loc[rows["success"].isin([1, -1])]
            for abs_ild in ABS_ILDS:
                for abl in ABLS:
                    condition = valid_rows.loc[
                        valid_rows["abs_ILD"].eq(abs_ild) & valid_rows["plot_ABL"].eq(abl)
                    ]
                    record_histogram(
                        condition, view, group, genotype, "correct_error", abs_ild,
                        abl, valid_edges, ABL_COLORS[abl],
                    )
            record_histogram(
                rows, view, group, genotype, "pooled_with_fixation", "all", "all",
                pooled_edges, POOLED_COLOR,
            )
            members = animal_counts.loc[
                animal_counts[column].eq(group) & animal_counts["genotype"].eq(genotype)
            ].copy()
            members.insert(0, "group", group)
            members.insert(0, "view", view)
            membership_parts.append(members)
            group_rows.append({
                "view": view, "group": group, "genotype": genotype,
                "animal_count": rows["animal"].nunique(), "selected_rows": len(rows),
                "finite_selected_rows": int(rows["finite_rt"].sum()),
                "nonfinite_selected_rows": int((~rows["finite_rt"]).sum()),
                "valid_window_trials": int(rows["valid_window"].sum()),
                "pooled_window_trials": int(rows["pooled_window"].sum()),
                "mapped_ABL59_rows": int(rows["ABL"].eq(59).sum()),
                "source_filenames": ";".join(name for name in SOURCE_FILES if name in set(rows["filename"])),
                "animals": ";".join(sorted(rows["animal"].unique())),
            })

curve_counts = pd.DataFrame(curve_rows)
histograms = pd.DataFrame(bin_rows)
group_counts = pd.DataFrame(group_rows)
membership = pd.concat(membership_parts, ignore_index=True)


# %% Independent check: compare every pooled bin to summed animal histogram counts
reference_bins = pd.read_csv(ANIMAL_OUTPUT / "histogram_bins.csv", float_precision="round_trip")
reference_counts = pd.read_csv(ANIMAL_OUTPUT / "curve_counts.csv")
animal_metadata = previous_animals[["filename", "animal", "genotype"]].copy()
animal_metadata["family"] = animal_metadata["filename"].str.extract(r"^(SHANK3|CNTNAP2)")[0]
reference_bins = reference_bins.merge(animal_metadata, on=["filename", "animal"], validate="many_to_one")
reference_counts = reference_counts.merge(animal_metadata, on=["filename", "animal"], validate="many_to_one")
sum_columns = [
    "eligible_rows", "finite_rt_rows", "nonfinite_rt_rows", "below_window_rows",
    "above_window_rows", "window_trials", "histogram_count", "window_correct_trials",
    "window_error_trials", "window_fixation_abort_trials",
]
view_validation = {}
for view, config in VIEWS.items():
    column = config["group_column"]
    keys = [column, "genotype", "panel", "abs_ILD", "plot_ABL"]
    summed_bins = reference_bins.groupby(keys + ["bin_index"], sort=False)["bin_count"].sum().reset_index()
    summed_bins = summed_bins.rename(columns={column: "group", "bin_count": "animal_bin_count"})
    checked_bins = histograms.loc[histograms["view"].eq(view)].merge(
        summed_bins, on=["group", "genotype", "panel", "abs_ILD", "plot_ABL", "bin_index"],
        how="outer", validate="one_to_one", indicator=True,
    )
    assert checked_bins["_merge"].eq("both").all()
    assert checked_bins["bin_count"].eq(checked_bins["animal_bin_count"]).all()
    summed_counts = reference_counts.groupby(keys, sort=False)[sum_columns].sum().reset_index()
    summed_counts = summed_counts.rename(columns={column: "group"})
    actual_counts = curve_counts.loc[curve_counts["view"].eq(view)]
    checked_counts = actual_counts.merge(
        summed_counts, on=["group", "genotype", "panel", "abs_ILD", "plot_ABL"],
        how="outer", validate="one_to_one", suffixes=("", "_animal_sum"), indicator=True,
    )
    assert checked_counts["_merge"].eq("both").all()
    for count_column in sum_columns:
        assert checked_counts[count_column].eq(checked_counts[count_column + "_animal_sum"]).all()
    groups = group_counts.loc[group_counts["view"].eq(view)]
    for key in EXPECTED_TOTALS:
        observed = membership.loc[membership["view"].eq(view), "animal"].nunique() if key == "animals" else int(groups[key].sum())
        assert observed == EXPECTED_TOTALS[key], (view, key, observed)
    assert int(actual_counts.loc[actual_counts["panel"].eq("correct_error"), "window_trials"].sum()) == EXPECTED_TOTALS["valid_window_trials"]
    assert int(actual_counts.loc[actual_counts["panel"].eq("pooled_with_fixation"), "window_trials"].sum()) == EXPECTED_TOTALS["pooled_window_trials"]
    assert len(actual_counts) == config["expected_groups"] * 16
    assert len(checked_bins) == config["expected_groups"] * (15 * 50 + 100)
    assert membership.loc[membership["view"].eq(view)].groupby(["filename", "animal"]).size().eq(1).all()
    view_validation[view] = {
        **totals, "groups": len(groups), "curves_checked": len(actual_counts),
        "histogram_bins_checked": len(checked_bins),
        "all_bins_match_summed_animal_counts": True,
        "all_curve_counts_match_summed_animal_counts": True,
        "max_nonempty_area_error": float((actual_counts.loc[actual_counts["window_trials"].gt(0), "density_area"] - 1).abs().max()),
    }
    print(f"Verified {view}: {len(groups)} groups, {len(checked_bins)} pooled bins", flush=True)
assert np.allclose(histograms["bin_right_s"] - histograms["bin_left_s"], BIN_WIDTH, rtol=0, atol=1e-14)
assert np.allclose(
    histograms["density_per_s"],
    histograms["bin_count"] / (histograms["window_trials"] * BIN_WIDTH),
    rtol=0, atol=1e-12,
)
membership.to_csv(OUTPUT / "genotype_rtd_group_membership.csv", index=False)
curve_counts.to_csv(OUTPUT / "genotype_rtd_curve_counts.csv", index=False)
histograms.to_csv(OUTPUT / "genotype_rtd_histogram_bins.csv", index=False)


# %% Figures: two filename/genotype pages and one family/genotype page
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
})
legend_handles = [
    Line2D([0], [0], color=ABL_COLORS[abl], lw=1.4, label=f"ABL {abl} dB")
    for abl in ABLS
] + [Line2D([0], [0], color=POOLED_COLOR, lw=1.4, label="Pooled + fixation")]
page_records = []
for view, config in VIEWS.items():
    pdf_path = OUTPUT / f"ASD_session_type_1_{view}_rtds.pdf"
    with PdfPages(pdf_path, metadata={
        "Title": f"ASD session-type-1 RTDs: {config['title']}",
        "Subject": "Trials pooled directly; 20 ms bins; original-window normalization",
        "Creator": "asd_analysis/plot_session_type_1_genotype_rtds.py",
    }) as pdf:
        for page_index, page_groups in enumerate(config["pages"], start=1):
            row_keys = [(group, genotype) for group in page_groups for genotype in GENOTYPES]
            assert len(row_keys) == 6
            height = 1.75 * len(row_keys) + 2.0
            fig, axes = plt.subplots(
                len(row_keys), 6, figsize=(18, height), squeeze=False,
                gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.2]},
            )
            fig.subplots_adjust(
                left=0.16, right=0.99, bottom=0.90 / height,
                top=1 - 1.05 / height, hspace=0.38, wspace=0.23,
            )
            fig.suptitle(
                f"Session type 1  |  {config['title']}  |  "
                f"{config['page_labels'][page_index - 1]}  |  Page {page_index}/{len(config['pages'])}",
                y=1 - 0.10 / height, fontsize=14,
            )
            fig.legend(
                handles=legend_handles, loc="upper center", ncol=4,
                bbox_to_anchor=(0.57, 1 - 0.38 / height), frameon=False, fontsize=10,
            )
            for row_index, (group, genotype) in enumerate(row_keys):
                group_mask = group_counts["view"].eq(view) & group_counts["group"].eq(group) & group_counts["genotype"].eq(genotype)
                animal_count = int(group_counts.loc[group_mask, "animal_count"].iloc[0])
                row_max = max(
                    curves[(view, group, genotype, "correct_error", str(ild), str(abl))][0].max()
                    for ild in ABS_ILDS for abl in ABLS
                )
                for col_index, abs_ild in enumerate(ABS_ILDS):
                    ax = axes[row_index, col_index]
                    for color_index, abl in enumerate(ABLS):
                        density, edges, n = curves[(view, group, genotype, "correct_error", str(abs_ild), str(abl))]
                        if n:
                            patch = ax.stairs(density, edges, baseline=None, fill=False, color=ABL_COLORS[abl], linewidth=1.1, linestyle="-")
                            assert to_hex(patch.get_edgecolor()) == to_hex(ABL_COLORS[abl])
                            assert not patch.get_fill() and patch.get_linestyle() == "-"
                        ax.text(
                            0.97, 0.96 - color_index * 0.12, f"n={n:,}",
                            transform=ax.transAxes, ha="right", va="top",
                            color=ABL_COLORS[abl], fontsize=7.5,
                        )
                    ax.set_xlim(*VALID_XLIM)
                    ax.set_ylim(0, max(1.0, row_max * 1.28))
                    ax.set_xticks([0, 0.2, 0.4, 0.6])
                    if col_index:
                        ax.tick_params(labelleft=False)
                    else:
                        ax.set_ylabel("Density (s⁻¹)")
                        label = Path(group).stem if view == "filename_genotype" else group
                        ax.text(
                            -0.40, 0.52, f"{label}\n{GENOTYPE_LABELS[genotype]} ({genotype})\n{animal_count} animal{'s' if animal_count != 1 else ''}",
                            transform=ax.transAxes, ha="right", va="center",
                            fontsize=9, fontweight="bold",
                        )
                    if row_index == 0:
                        ax.set_title(f"|ILD| = {abs_ild} dB")
                ax = axes[row_index, 5]
                density, edges, n = curves[(view, group, genotype, "pooled_with_fixation", "all", "all")]
                patch = ax.stairs(density, edges, baseline=None, fill=False, color=POOLED_COLOR, linewidth=1.1, linestyle="-")
                assert to_hex(patch.get_edgecolor()) == "#000000"
                assert not patch.get_fill() and patch.get_linestyle() == "-"
                ax.text(0.97, 0.96, f"n={n:,}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
                ax.axvline(0, color="0.75", linewidth=0.6, zorder=0)
                ax.set_xlim(*POOLED_XLIM)
                ax.set_ylim(0, max(1.0, density.max() * 1.20))
                ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
                if row_index == 0:
                    ax.set_title("All displayed stimuli\nCorrect/error + fixation")
                assert len({tuple(axis.get_ylim()) for axis in axes[row_index, :5]}) == 1
                assert all(tuple(axis.get_xlim()) == VALID_XLIM for axis in axes[row_index, :5])
                assert tuple(ax.get_xlim()) == POOLED_XLIM
                for ax in axes[row_index]:
                    ax.spines[["top", "right"]].set_visible(False)
                    if row_index == len(row_keys) - 1:
                        ax.set_xlabel("timed_rt (s)")
                group_counts.loc[group_mask, "pdf_page"] = page_index
            footer = fig.text(
                0.56, 0.10 / height,
                "Trials pooled directly across animals and sessions; animals with more trials contribute more weight. "
                "ABL 59 → 60. Exclude session_type=1 & is_short_sound=1 rows; all 53 animals included.\n"
                "20 ms bins. Display: 0–0.6 s / −0.6–0.6 s; printed n and unit-area normalization use 0–1 s / −1–1 s. "
                "Panels 1–5: correct/error; panel 6 adds fixation, using displayed stimuli only.",
                ha="center", va="bottom", fontsize=8,
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for axis in axes.flat:
                for label in axis.texts:
                    bounds = label.get_window_extent(renderer)
                    assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.x1
                    assert bounds.y0 >= 0 and bounds.y1 <= fig.bbox.y1
            footer_bounds = footer.get_window_extent(renderer)
            assert footer_bounds.x0 >= 0 and footer_bounds.x1 <= fig.bbox.x1
            assert all(axis.xaxis.label.get_window_extent(renderer).y0 > footer_bounds.y1 for axis in axes[-1])
            png_path = OUTPUT / f"ASD_session_type_1_{view}_rtds_page_{page_index:02d}.png"
            pdf.savefig(fig)
            fig.savefig(png_path, dpi=PNG_DPI)
            plt.close(fig)
            page_records.append({
                "view": view, "pdf_page": page_index, "row_count": len(row_keys),
                "groups_and_genotypes": [f"{group}/{genotype}" for group, genotype in row_keys],
                "pdf": str(pdf_path.relative_to(REPO)), "png": str(png_path.relative_to(REPO)),
            })
            print(f"Saved {view}, page {page_index}: six pooled groups", flush=True)
    with fitz.open(pdf_path) as document:
        assert document.page_count == len(config["pages"])
        for page in document:
            words = [entry[4].replace("−", "-") for entry in page.get_text("words")]
            assert words.count("0.6") == 6 * 6 and words.count("-0.6") == 6
            assert page.get_text().count("n=") == 6 * 16
    view_validation[view]["pdf_pages"] = len(config["pages"])
    view_validation[view]["png_pages"] = len(config["pages"])
group_counts["pdf_page"] = group_counts["pdf_page"].astype(int)
group_counts.to_csv(OUTPUT / "genotype_rtd_group_counts.csv", index=False)
pd.DataFrame(page_records).to_csv(OUTPUT / "genotype_rtd_pages.csv", index=False)


# %% Save provenance, normalization, and per-view validation manifests
assert input_stats == {
    str(path.relative_to(REPO)): [path.stat().st_size, path.stat().st_mtime_ns]
    for path in input_paths
}
assert animal_artifact_hashes == {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in ANIMAL_OUTPUT.iterdir() if path.is_file()
}
selection = dict(animal_manifest["selection"])
selection["weighting"] = (
    "Raw trials combined across animals, signed ILDs, and sessions within filename+genotype "
    "or family+genotype. Animals with more eligible trials contribute more weight. "
    "The sixth panel combines all displayed conditions directly by trial. "
    "No equal-animal or equal-condition density averaging; no deduplication."
)
histogram_settings = dict(animal_manifest["histogram"])
histogram_settings["first_five_y_scale"] = "shared within each pooled group row"
histogram_settings["display_zoom"] = (
    "Display [0,0.6] / [-0.6,0.6] s; printed n and density normalization retain "
    "[0,1] / [-1,1] s windows. The visible area is not renormalized."
)
manifest = {
    "status": "passed", "source_script": "asd_analysis/plot_session_type_1_genotype_rtds.py",
    "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_csv": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256,
    "input_file_size_and_mtime_ns": input_stats, "prepared_input_rows": len(data),
    "selection": selection, "histogram": histogram_settings,
    "genotype_order": GENOTYPES, "source_order": SOURCE_FILES, "family_order": FAMILIES,
    "row_animal_count": "Distinct selected animals in the group; curve audits also report animals in each normalization window",
    "expected_totals_per_view": EXPECTED_TOTALS, "views": view_validation,
    "pages": page_records, "existing_animal_artifacts_sha256": animal_artifact_hashes,
    "independent_reference": str(ANIMAL_OUTPUT.relative_to(REPO)),
    "checks": [
        "prepared input SHA256 matches existing verified animal analysis",
        "confirmed prepared filters and short-sound row exclusions; both animals restored",
        "same 53 animal memberships and per-animal counts as updated animal analysis",
        "12 filename/genotype groups and 6 family/genotype groups",
        "exact selected, finite, and normalization-window counts in each view",
        "every pooled bin equals the sum of existing animal bin counts",
        "all outcome and eligibility counts equal sums of animal curve audits",
        "20 ms bins; density divides pooled counts by pooled window n times 0.020 s",
        "unit areas over original windows; display zoom is not renormalized",
        "solid unfilled step curves and requested colors",
        "shared first-five-panel y scales and requested display limits",
        "PDF page counts 2 and 1; endpoint ticks and 16 sample labels per row",
        "group/sample labels fit inside page; footer below axis labels",
        "input CSV size/mtime and all existing animal artifact hashes unchanged",
    ],
}
(OUTPUT / "genotype_rtd_validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
for view in VIEWS:
    view_manifest = {
        key: value for key, value in manifest.items()
        if key not in ["views", "pages"]
    }
    view_manifest.update({
        "view": view, "observed_counts": view_validation[view],
        "pages": [page for page in page_records if page["view"] == view],
    })
    (OUTPUT / f"ASD_session_type_1_{view}_rtds_validation.json").write_text(json.dumps(view_manifest, indent=2) + "\n")
print("All pooled RTD checks passed:", json.dumps(view_validation), flush=True)
