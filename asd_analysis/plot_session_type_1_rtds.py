# %% Imports
"""Session-type-1 RTDs, with per-curve window normalization and source audits.

Run from the repository root with:
    .venv/bin/python asd_analysis/plot_session_type_1_rtds.py

The prepared CSV is read only. Its training, outcome, and chronological
next-trial repeat filters are retained; no repeat flags are reinterpreted here.
"""

import csv
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


# %% Parameters: edit these together with the expected counts for a new analysis
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
ARCHIVED_INPUT = INPUT.parent / "archive/pre_short_sound_cleanup/ASD_session_type_1.csv"
CLEANUP_MANIFEST = INPUT.parent / "short_sound_cleanup.json"
OUTPUT = REPO / "asd_analysis/session_type_1_rtds"
SOURCE_FILES = [
    "SHANK3_cohort1.csv",
    "SHANK3_cohort2.csv",
    "CNTNAP2_cohort3.csv",
    "CNTNAP2_cohort4.csv",
]
RESTORED_ANIMALS = ["ASD0026", "ASD0045"]
ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
POOLED_COLOR = "black"
BIN_WIDTH = 0.020
VALID_WINDOW = (0.0, 1.0)
POOLED_WINDOW = (-1.0, 1.0)
# Display zoom only: histogram edges, normalization, and printed n stay unchanged.
VALID_XLIM = (0.0, 0.6)
POOLED_XLIM = (-0.6, 0.6)
ANIMALS_PER_PAGE = 8
PNG_DPI = 160
EXPECTED = {
    "groups": 53,
    "selected_rows": 610234,
    "finite_selected_rows": 607207,
    "nonfinite_selected_rows": 3027,
    "valid_window_trials": 466978,
    "pooled_window_trials": 598864,
    "pdf_pages": 8,
}
EXPECTED_COHORT_GROUPS = dict(zip(SOURCE_FILES, [14, 6, 16, 17]))

OUTPUT.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUTPUT / "ASD_session_type_1_rtds.pdf"
INPUT_PATHS = [REPO / name for name in SOURCE_FILES] + [INPUT, ARCHIVED_INPUT]
cleanup = json.loads(CLEANUP_MANIFEST.read_text())
input_stats = {
    str(path.relative_to(REPO)): [path.stat().st_size, path.stat().st_mtime_ns]
    for path in INPUT_PATHS
}
with INPUT.open("rb") as handle:
    input_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()


# %% Read the prepared input and preserve its confirmed filters
plot_columns = [
    "animal", "filename", "genotype", "source_file", "source_date", "session",
    "trial", "trial_start", "training_level", "session_type", "is_short_sound",
    "short_duration", "ABL", "ILD", "success", "abort_type", "timed_rt",
    "repeated_trial", "trial_is_repeat",
]
data = pd.read_csv(INPUT, usecols=plot_columns, low_memory=False)
data.insert(0, "csv_row_number", np.arange(len(data)) + 2)
assert data["training_level"].eq(16).all()
assert data["session_type"].eq(1).all()
assert data["trial_is_repeat"].eq(False).all()
assert (data["success"].isin([1, -1]) | data["abort_type"].eq("Fixation")).all()
assert data["genotype"].isin(["wt", "hom", "het"]).all()

# Exclude individual short-sound records, not the animals containing them.
short_rows = data["session_type"].eq(1) & data["is_short_sound"].eq(1)
assert len(data) == 642370 and int(short_rows.sum()) == 0
assert short_rows.equals(data["session_type"].eq(1) & data["short_duration"].gt(0))
excluded_counts = data.loc[short_rows].groupby(
    ["filename", "animal"]
).size()
data["plot_ABL"] = data["ABL"].replace({59: 60})
data["abs_ILD"] = data["ILD"].abs()
selected = data.loc[
    ~short_rows
    & data["plot_ABL"].isin(ABLS)
    & data["abs_ILD"].isin(ABS_ILDS)
].copy()
finite = np.isfinite(selected["timed_rt"])
valid_window = (
    selected["success"].isin([1, -1])
    & finite
    & selected["timed_rt"].between(*VALID_WINDOW, inclusive="both")
)
pooled_window = finite & selected["timed_rt"].between(
    *POOLED_WINDOW, inclusive="both"
)
cohort_groups = selected.groupby("filename")["animal"].nunique().to_dict()
observed = {
    "groups": selected.groupby(["filename", "animal"]).ngroups,
    "selected_rows": len(selected),
    "finite_selected_rows": int(finite.sum()),
    "nonfinite_selected_rows": int((~finite).sum()),
    "valid_window_trials": int(valid_window.sum()),
    "pooled_window_trials": int(pooled_window.sum()),
}
for key, count in observed.items():
    assert count == EXPECTED[key], (key, count, EXPECTED[key])
assert cohort_groups == EXPECTED_COHORT_GROUPS, cohort_groups
assert set(RESTORED_ANIMALS).issubset(set(selected["animal"]))
assert selected["is_short_sound"].fillna(0).eq(0).all()
print("Selected data:", json.dumps(observed), flush=True)


# %% Complete flagged-row locations, including conditions outside the plots
# CSV row 1 is the header; data record 1 is CSV row 2. Rows are interleaved,
# so the span from the first to last flagged row is not itself an exclusion list.
location_columns = [
    "animal", "source_file", "source_date", "session", "trial", "trial_start",
    "training_level", "session_type", "is_short_sound", "short_duration",
    "ABL", "ILD", "success", "abort_type", "timed_rt", "repeated_trial",
]
flagged_parts = []
for path in INPUT_PATHS:
    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    use_columns = [column for column in location_columns if column in header]
    if "filename" in header:
        use_columns.append("filename")
    row_offset = 0
    for chunk in pd.read_csv(
        path, usecols=use_columns, chunksize=200000, low_memory=False
    ):
        short_flag = chunk["is_short_sound"].eq(1)
        duration_flag = chunk["short_duration"].gt(0)
        flagged = chunk["session_type"].eq(1) & (short_flag | duration_flag)
        if flagged.any():
            part = chunk.loc[flagged].copy()
            part.insert(0, "csv_row_number", np.flatnonzero(flagged) + row_offset + 2)
            part.insert(0, "csv_path", str(path.relative_to(REPO)))
            part.insert(0, "csv_filename", path.name)
            kind = "archived_prepared" if path == ARCHIVED_INPUT else "prepared" if path == INPUT else "original"
            part.insert(0, "csv_kind", kind)
            if "filename" not in part:
                part["filename"] = path.name
            part["is_short_sound_flag"] = short_flag.loc[flagged]
            part["positive_short_duration_flag"] = duration_flag.loc[flagged]
            plot_abl = part["ABL"].replace({59: 60})
            part["in_displayed_stimulus_domain"] = (
                plot_abl.isin(ABLS) & part["ILD"].abs().isin(ABS_ILDS)
            )
            part["whole_animal_excluded_from_plots"] = False
            part["excluded_by_short_sound_row_rule"] = part["session_type"].eq(1) & short_flag.loc[flagged]
            flagged_parts.append(part)
        row_offset += len(chunk)

flagged_locations = pd.concat(flagged_parts, ignore_index=True)
assert set(flagged_locations["animal"]) == set(RESTORED_ANIMALS)
assert flagged_locations["training_level"].eq(16).all()
assert flagged_locations["is_short_sound_flag"].all()
assert flagged_locations["positive_short_duration_flag"].all()
flagged_summary = []
for (kind, filename, animal), rows in flagged_locations.groupby(
    ["csv_kind", "csv_filename", "animal"], sort=False
):
    flagged_summary.append({
        "csv_kind": kind, "csv_filename": filename, "animal": animal,
        "csv_path": rows["csv_path"].iloc[0],
        "flagged_rows": len(rows),
        "first_csv_row": int(rows["csv_row_number"].min()),
        "last_csv_row": int(rows["csv_row_number"].max()),
        "displayed_condition_rows": int(rows["in_displayed_stimulus_domain"].sum()),
        "source_dates": sorted(rows["source_date"].unique().tolist()),
    })
assert len(flagged_locations) == 1643
assert flagged_locations.groupby("csv_kind").size().to_dict() == {
    "original": 836, "archived_prepared": 807,
}
assert int(flagged_locations.loc[
    flagged_locations["csv_kind"].eq("archived_prepared"), "in_displayed_stimulus_domain"
].sum()) == 763
flagged_locations.to_csv(OUTPUT / "flagged_row_locations.csv", index=False)
flagged_locations.loc[flagged_locations["csv_kind"].eq("archived_prepared")].to_csv(
    OUTPUT / "excluded_prepared_short_sound_rows.csv", index=False
)
pd.DataFrame(flagged_summary).to_csv(OUTPUT / "flagged_row_summary.csv", index=False)
print("Flagged-row locations saved:", len(flagged_locations), flush=True)


# %% Histograms: one reusable calculation for all conditional and pooled curves
valid_edges = np.linspace(*VALID_WINDOW, 51)
pooled_edges = np.linspace(*POOLED_WINDOW, 101)
assert np.allclose(np.diff(valid_edges), BIN_WIDTH, rtol=0, atol=1e-14)
assert np.allclose(np.diff(pooled_edges), BIN_WIDTH, rtol=0, atol=1e-14)
histogram_rows = []
count_rows = []
curves = {}
group_rows = []


def record_histogram(rows, filename, animal, panel, abs_ild, abl, edges, color):
    """Record the displayed-window denominator, bins, and density for a curve."""
    rt = rows["timed_rt"].to_numpy()
    finite_rt = np.isfinite(rt)
    in_window = finite_rt & (rt >= edges[0]) & (rt <= edges[-1])
    counts, _ = np.histogram(rt[in_window], bins=edges)
    n = int(in_window.sum())
    assert counts.sum() == n
    density = counts / (n * BIN_WIDTH) if n else np.zeros(len(counts))
    area = float(np.sum(density * np.diff(edges)))
    assert np.isclose(area, 1.0 if n else 0.0, rtol=0, atol=1e-12)
    curve_id = f"{filename}/{animal}/{panel}/{abs_ild}/{abl}"
    audit = {
        "curve_id": curve_id, "filename": filename, "animal": animal,
        "panel": panel, "abs_ILD": abs_ild, "plot_ABL": abl,
        "color": to_hex(color), "eligible_rows": len(rows),
        "finite_rt_rows": int(finite_rt.sum()),
        "nonfinite_rt_rows": int((~finite_rt).sum()),
        "below_window_rows": int((finite_rt & (rt < edges[0])).sum()),
        "above_window_rows": int((finite_rt & (rt > edges[-1])).sum()),
        "window_trials": n, "window_start_s": float(edges[0]),
        "window_end_s": float(edges[-1]), "bin_width_s": BIN_WIDTH,
        "histogram_count": int(counts.sum()), "density_area": area,
        "window_correct_trials": int(rows.loc[in_window, "success"].eq(1).sum()),
        "window_error_trials": int(rows.loc[in_window, "success"].eq(-1).sum()),
        "window_fixation_abort_trials": int(
            rows.loc[in_window, "abort_type"].eq("Fixation").sum()
        ),
    }
    count_rows.append(audit)
    for bin_index, (left, right, count, value) in enumerate(
        zip(edges[:-1], edges[1:], counts, density)
    ):
        histogram_rows.append({
            "curve_id": curve_id, "filename": filename, "animal": animal,
            "panel": panel, "abs_ILD": abs_ild, "plot_ABL": abl,
            "bin_index": bin_index, "bin_left_s": left, "bin_right_s": right,
            "bin_count": int(count), "window_trials": n,
            "density_per_s": value, "color": to_hex(color),
        })
    curves[(filename, animal, panel, abs_ild, abl)] = (counts, density, edges, n)


for filename in SOURCE_FILES:
    source_rows = selected.loc[selected["filename"].eq(filename)]
    for animal, rows in source_rows.groupby("animal", sort=True):
        genotypes = rows["genotype"].unique()
        assert len(genotypes) == 1
        valid_rows = rows.loc[rows["success"].isin([1, -1])]
        for abs_ild in ABS_ILDS:
            for abl in ABLS:
                condition_rows = valid_rows.loc[
                    valid_rows["abs_ILD"].eq(abs_ild) & valid_rows["plot_ABL"].eq(abl)
                ]
                record_histogram(
                    condition_rows, filename, animal, "correct_error", abs_ild,
                    abl, valid_edges, ABL_COLORS[abl],
                )
        record_histogram(
            rows, filename, animal, "pooled_with_fixation", "all", "all",
            pooled_edges, POOLED_COLOR,
        )
        group_rows.append({
            "filename": filename, "animal": animal, "genotype": genotypes[0],
            "selected_rows": len(rows),
            "finite_selected_rows": int(np.isfinite(rows["timed_rt"]).sum()),
            "valid_window_trials": int((
                np.isfinite(valid_rows["timed_rt"])
                & valid_rows["timed_rt"].between(*VALID_WINDOW, inclusive="both")
            ).sum()),
            "pooled_window_trials": curves[
                (filename, animal, "pooled_with_fixation", "all", "all")
            ][3],
            "mapped_ABL59_rows": int(rows["ABL"].eq(59).sum()),
            "daily_source_files": rows["source_file"].nunique(),
            "sessions": rows.groupby(["source_file", "session"]).ngroups,
        })

counts_audit = pd.DataFrame(count_rows)
histograms = pd.DataFrame(histogram_rows)
groups_audit = pd.DataFrame(group_rows)
assert len(counts_audit) == EXPECTED["groups"] * 16
assert int(counts_audit.loc[
    counts_audit["panel"].eq("correct_error"), "window_trials"
].sum()) == EXPECTED["valid_window_trials"]
assert int(counts_audit.loc[
    counts_audit["panel"].eq("pooled_with_fixation"), "window_trials"
].sum()) == EXPECTED["pooled_window_trials"]
counts_audit.to_csv(OUTPUT / "curve_counts.csv", index=False)
histograms.to_csv(OUTPUT / "histogram_bins.csv", index=False)


# %% Draw cohort pages: six columns and at most eight animal rows per page
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
genotype_labels = {"wt": "WT", "hom": "Homozygous", "het": "Heterozygous"}
legend_handles = [
    Line2D([0], [0], color=ABL_COLORS[abl], lw=1.4, label=f"ABL {abl} dB")
    for abl in ABLS
] + [Line2D([0], [0], color=POOLED_COLOR, lw=1.4, label="Pooled + fixation")]
page_records = []
with PdfPages(PDF_PATH, metadata={
    "Title": "ASD session-type-1 RTDs by animal and stimulus",
    "Subject": "20 ms, per-curve window normalization; short-sound rows excluded, all 53 animals included",
    "Creator": "asd_analysis/plot_session_type_1_rtds.py",
}) as pdf:
    for filename in SOURCE_FILES:
        cohort_animals = groups_audit.loc[
            groups_audit["filename"].eq(filename), "animal"
        ].tolist()
        cohort_pages = int(np.ceil(len(cohort_animals) / ANIMALS_PER_PAGE))
        for start in range(0, len(cohort_animals), ANIMALS_PER_PAGE):
            animals = cohort_animals[start:start + ANIMALS_PER_PAGE]
            nrows = len(animals)
            cohort_page = start // ANIMALS_PER_PAGE + 1
            page_number = len(page_records) + 1
            # Row height stays constant even on the final one-animal page.
            height = 1.75 * nrows + 2.00
            fig, axes = plt.subplots(
                nrows, 6, figsize=(18, height), squeeze=False,
                gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.2]},
            )
            fig.subplots_adjust(
                left=0.15, right=0.99, bottom=0.90 / height,
                top=1 - 1.05 / height, hspace=0.38, wspace=0.23,
            )
            fig.suptitle(
                f"{Path(filename).stem}  |  Session type 1  |  "
                f"Cohort page {cohort_page}/{cohort_pages}  |  PDF page {page_number}/8",
                y=1 - 0.10 / height, fontsize=14,
            )
            fig.legend(
                handles=legend_handles, loc="upper center", ncol=4,
                bbox_to_anchor=(0.55, 1 - 0.38 / height), frameon=False,
                fontsize=10,
            )
            for row_index, animal in enumerate(animals):
                genotype = groups_audit.loc[
                    groups_audit["filename"].eq(filename)
                    & groups_audit["animal"].eq(animal), "genotype"
                ].iloc[0]
                row_max = max(
                    curves[(filename, animal, "correct_error", ild, abl)][1].max()
                    for ild in ABS_ILDS for abl in ABLS
                )
                row_ymax = max(1.0, row_max * 1.28)
                for col_index, abs_ild in enumerate(ABS_ILDS):
                    ax = axes[row_index, col_index]
                    for color_index, abl in enumerate(ABLS):
                        counts, density, edges, n = curves[
                            (filename, animal, "correct_error", abs_ild, abl)
                        ]
                        if n:
                            patch = ax.stairs(
                                density, edges, baseline=None, fill=False,
                                color=ABL_COLORS[abl], linewidth=1.1, linestyle="-",
                            )
                            assert to_hex(patch.get_edgecolor()) == to_hex(ABL_COLORS[abl])
                            assert not patch.get_fill()
                            assert patch.get_linestyle() == "-"
                        ax.text(
                            0.97, 0.96 - color_index * 0.12, f"n={n:,}",
                            transform=ax.transAxes, ha="right", va="top",
                            color=ABL_COLORS[abl], fontsize=7.5,
                        )
                    ax.set_xlim(*VALID_XLIM)
                    ax.set_ylim(0, row_ymax)
                    ax.set_xticks([0, 0.2, 0.4, 0.6])
                    if col_index:
                        ax.tick_params(labelleft=False)
                    else:
                        ax.set_ylabel("Density (s⁻¹)")
                        ax.text(
                            -0.40, 0.52,
                            f"{animal}\n{genotype_labels[genotype]} ({genotype})",
                            transform=ax.transAxes, ha="right", va="center",
                            fontsize=9, fontweight="bold",
                        )
                    if row_index == 0:
                        ax.set_title(f"|ILD| = {abs_ild} dB")
                ax = axes[row_index, 5]
                counts, density, edges, n = curves[
                    (filename, animal, "pooled_with_fixation", "all", "all")
                ]
                patch = ax.stairs(
                    density, edges, baseline=None, fill=False, color=POOLED_COLOR,
                    linewidth=1.1, linestyle="-",
                )
                assert to_hex(patch.get_edgecolor()) == "#000000"
                assert not patch.get_fill() and patch.get_linestyle() == "-"
                ax.text(
                    0.97, 0.96, f"n={n:,}", transform=ax.transAxes,
                    ha="right", va="top", color=POOLED_COLOR, fontsize=8,
                )
                ax.axvline(0, color="0.75", linewidth=0.6, zorder=0)
                ax.set_xlim(*POOLED_XLIM)
                ax.set_ylim(0, max(1.0, density.max() * 1.20))
                ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
                if row_index == 0:
                    ax.set_title("All displayed stimuli\nCorrect/error + fixation")
                assert len({tuple(axis.get_ylim()) for axis in axes[row_index, :5]}) == 1
                assert all(tuple(axis.get_xlim()) == VALID_XLIM for axis in axes[row_index, :5])
                assert tuple(axes[row_index, 5].get_xlim()) == POOLED_XLIM
                for ax in axes[row_index]:
                    ax.spines[["top", "right"]].set_visible(False)
                    if row_index == nrows - 1:
                        ax.set_xlabel("timed_rt (s)")
            footer = fig.text(
                0.55, 0.10 / height,
                "20 ms bins; stimulus-relative timed_rt. Display: panels 1–5, 0–0.6 s; "
                "panel 6, −0.6–0.6 s. Printed n and unit-area normalization still use 0–1 s / −1–1 s.\n"
                "Panels 1–5: correct/error; panel 6: correct/error + fixation. "
                "ABL 20/40/60 × |ILD| 1/2/4/8/16; ABL 59 → 60. "
                "Signed ILDs and sessions pooled by trial. Exclude session_type=1 & is_short_sound=1 rows; all 53 animals included.",
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
            assert all(
                axis.xaxis.label.get_window_extent(renderer).y0 > footer_bounds.y1
                for axis in axes[-1]
            )
            png_path = OUTPUT / f"{Path(filename).stem}_rtds_page_{cohort_page:02d}.png"
            pdf.savefig(fig)
            fig.savefig(png_path, dpi=PNG_DPI)
            plt.close(fig)
            groups_audit.loc[
                groups_audit["filename"].eq(filename)
                & groups_audit["animal"].isin(animals), "pdf_page"
            ] = page_number
            page_records.append({
                "pdf_page": page_number, "filename": filename,
                "cohort_page": cohort_page, "cohort_pages": cohort_pages,
                "animal_count": nrows, "animals": animals,
                "png": str(png_path.relative_to(REPO)),
            })
            print(f"Saved PDF page {page_number}: {filename}, {nrows} animals", flush=True)

groups_audit["pdf_page"] = groups_audit["pdf_page"].astype(int)
groups_audit.to_csv(OUTPUT / "animal_counts.csv", index=False)
pd.DataFrame(page_records).to_csv(OUTPUT / "pages.csv", index=False)


# %% Final validation and reproducible selection/normalization manifest
with fitz.open(PDF_PATH) as document:
    observed["pdf_pages"] = document.page_count
assert observed["pdf_pages"] == EXPECTED["pdf_pages"]
assert len(page_records) == EXPECTED["pdf_pages"]
assert sum(page["animal_count"] for page in page_records) == EXPECTED["groups"]
assert all(page["animal_count"] <= ANIMALS_PER_PAGE for page in page_records)
assert groups_audit.groupby("filename")["animal"].nunique().to_dict() == cohort_groups
assert len(histograms) == EXPECTED["groups"] * (15 * 50 + 100)
assert np.allclose(
    histograms["bin_right_s"] - histograms["bin_left_s"], BIN_WIDTH,
    rtol=0, atol=1e-14,
)
assert input_stats == {
    str(path.relative_to(REPO)): [path.stat().st_size, path.stat().st_mtime_ns]
    for path in INPUT_PATHS
}
nonempty = counts_audit["window_trials"].gt(0)
manifest = {
    "status": "passed",
    "source_script": "asd_analysis/plot_session_type_1_rtds.py",
    "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_csv": str(INPUT.relative_to(REPO)),
    "input_sha256": input_sha256,
    "input_file_size_and_mtime_ns": input_stats,
    "prepared_input_rows": len(data),
    "selection": {
        "prepared_rules": "session_type=1; training_level=16; not derived trial_is_repeat; "
        "success in {1,-1} OR abort_type='Fixation'",
        "repeat_rule": "Exclude the next distinct chronological trial within "
        "animal/source_file/session after repeated_trial=True, except after CNP/Fixation. "
        "Use the already prepared CSV without reapplying this rule.",
        "excluded_animals_entirely": [],
        "excluded_row_rule": "session_type == 1 AND is_short_sound == 1",
        "requested_flag_name": "is_short_duration; actual source field is is_short_sound",
        "excluded_prepared_rows": int(short_rows.sum()),
        "rows_removed_during_preparation": cleanup["removed_rows"],
        "removed_during_preparation_by_animal": cleanup["removed_by_animal"],
        "additional_plot_short_sound_exclusions": int(short_rows.sum()),
        "cleanup_manifest": str(CLEANUP_MANIFEST.relative_to(REPO)),
        "archived_prepared_csv": str(ARCHIVED_INPUT.relative_to(REPO)),
        "excluded_rows_audit": "asd_analysis/session_type_1_rtds/excluded_prepared_short_sound_rows.csv",
        "excluded_displayed_condition_rows": int((short_rows & data["plot_ABL"].isin(ABLS) & data["abs_ILD"].isin(ABS_ILDS)).sum()),
        "removed_during_preparation_displayed_condition_rows": 763,
        "animals_retained_after_row_exclusion": RESTORED_ANIMALS,
        "rows_after_short_sound_exclusion_before_stimulus_selection": int((~short_rows).sum()),
        "excluded_rows_by_animal": [
            {"filename": filename, "animal": animal, "rows": int(n)}
            for (filename, animal), n in excluded_counts.items()
        ],
        "plot_ABL_mapping": {"59": 60}, "plot_ABLs": ABLS, "abs_ILDs": ABS_ILDS,
        "mapped_ABL59_selected_rows": int(selected["ABL"].eq(59).sum()),
        "first_five": "success in {1,-1}; finite recorded timed_rt; inclusive [0,1] s",
        "sixth": "success in {1,-1} OR abort_type='Fixation'; finite recorded timed_rt; "
        "inclusive [-1,1] s; displayed stimulus domain only",
        "genotypes": "all wt, hom, het; no genotype exclusion",
        "timing": "Recorded timed_rt in seconds relative to stimulus onset; "
        "negative fixation aborts refer to scheduled stimulus onset. No recomputation.",
        "weighting": "Observed trials pooled over signed ILDs and sessions within "
        "filename+animal; sixth panel pools trials directly over displayed conditions; "
        "no deduplication or equal-condition/animal averaging",
    },
    "histogram": {
        "bin_width_s": BIN_WIDTH, "valid_window_s": VALID_WINDOW,
        "pooled_window_s": POOLED_WINDOW,
        "valid_display_xlim_s": VALID_XLIM,
        "pooled_display_xlim_s": POOLED_XLIM,
        "display_zoom": "Only x limits changed: printed n and density normalization "
        "retain [0,1] / [-1,1] s windows. The visible area is not renormalized.",
        "edge_rule": "left-inclusive/right-exclusive bins; final bin includes right endpoint",
        "normalization": "bin_count / (window_trials * 0.020 s)",
        "nonempty_curve_area": 1, "empty_curve_behavior": "n=0; zero density; no curve",
        "ABL_colors": {str(abl): to_hex(color) for abl, color in ABL_COLORS.items()},
        "pooled_color": to_hex(POOLED_COLOR), "line_style": "solid, unfilled steps",
        "first_five_y_scale": "shared within each animal row",
    },
    "observed_counts": observed, "expected_counts": EXPECTED,
    "cohort_group_counts": cohort_groups,
    "curve_count": len(counts_audit), "nonempty_curves": int(nonempty.sum()),
    "histogram_bin_rows": len(histograms),
    "max_nonempty_area_error": float(
        (counts_audit.loc[nonempty, "density_area"] - 1).abs().max()
    ),
    "flagged_row_locations": flagged_summary,
    "flagged_original_rows": 836, "flagged_prepared_rows": 0,
    "flagged_archived_prepared_rows": 807,
    "flagged_prepared_displayed_condition_rows": 0,
    "flagged_archived_prepared_displayed_condition_rows": 763,
    "pdf": str(PDF_PATH.relative_to(REPO)), "pages": page_records,
    "checks": [
        "confirmed prepared filters", "exact selected and window trial counts",
        "cohort and filename+animal group counts", "short-sound row exclusions; both animals restored",
        "complete duration-flag locations in original and archived prepared CSVs; cleaned input has zero flags",
        "histogram sums equal window counts", "20 ms bin widths",
        "unit area for every nonempty curve", "step colors and solid/unfilled styles",
        "shared first-five-panel row y limits", "PDF has eight pages",
        "display x limits are [0,0.6] / [-0.6,0.6] s; original normalization retained",
        "sample/animal labels inside figure; footer below axis labels",
        "all input CSV sizes and modification times unchanged",
    ],
}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Validation passed:", json.dumps(observed), flush=True)
