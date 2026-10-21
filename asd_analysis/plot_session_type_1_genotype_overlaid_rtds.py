# %% Imports
"""Two RTD panels: separate families, three genotype colors, ABL/ILD pooled.

Run with .venv/bin/python from the repository root. Use the current prepared
CSV; repeat eligibility and the calibration exclusion are already applied.
Histograms are independently checked with csv/bisect and earlier CDF counts.
"""
from bisect import bisect_right
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd


# %% Editable parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
REFERENCE = REPO / "asd_analysis/session_type_1_family_genotype_psychometric_rtd_chronometric_scaling"
OUTPUT = REPO / "asd_analysis/session_type_1_genotype_overlaid_rtds"
FAMILIES = ["SHANK3", "CNTNAP2"]
GENOTYPES = ["wt", "hom", "het"]
COLORS = {"wt": "tab:blue", "hom": "tab:green", "het": "tab:orange"}
LABELS = {"wt": "WT", "hom": "HOM", "het": "HET"}
ABLS = [20, 40, 60]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
RT_WINDOW = (0., 1.)
BIN_WIDTH = .020
FIGSIZE = (14, 5.4)
PNG_DPI = 200
NAME = "ASD_session_type_1_family_genotype_ABL_ILD_pooled_rtds"
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Input/reference provenance
protected = {INPUT} | {p for p in REFERENCE.iterdir() if p.is_file()}
protected_hashes = {}
for path in sorted(protected):
    with path.open("rb") as stream:
        protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_sha256 = protected_hashes[str(INPUT.relative_to(REPO))]
reference_manifest = json.loads((REFERENCE / "validation.json").read_text())
assert reference_manifest["status"] == "passed" and reference_manifest["input_sha256"] == input_sha256


# %% Same stimulus population as the preceding family/genotype PDF; apply RT window
data = pd.read_csv(INPUT, usecols=["filename", "animal", "genotype", "session", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "success", "ABL", "ILD", "timed_rt"], low_memory=False)
assert len(data) == 610603 and data.animal.nunique() == 53
assert data.session_type.eq(1).all() and data.training_level.eq(16).all()
assert data.trial_is_repeat.eq(False).all() and data.is_short_sound.eq(0).all()
assert not (data.filename.eq("CNTNAP2_cohort4.csv") & data.session.lt(13)).any()
data["plot_ABL"] = data.ABL.replace({59: 60})
selected = data.loc[data.success.isin([1, -1]) & data.plot_ABL.isin(ABLS) & data.ILD.isin(SIGNED_ILDS)].copy()
selected["family"] = selected.filename.str.extract(r"^(SHANK3|CNTNAP2)_cohort")[0]
assert len(selected) == 451381
window = selected.loc[np.isfinite(selected.timed_rt) & selected.timed_rt.between(*RT_WINDOW, inclusive="both")].copy()
assert len(window) == 446949 and window.animal.nunique() == 53
assert window.family.isin(FAMILIES).all() and window.genotype.isin(GENOTYPES).all()
edges = np.linspace(*RT_WINDOW, 51)
np.testing.assert_allclose(np.diff(edges), BIN_WIDTH, rtol=0, atol=2e-16)
membership = window.groupby(["family", "genotype", "filename", "animal"], as_index=False).agg(window_trials=("timed_rt", "size"))
membership.to_csv(OUTPUT / "group_membership.csv", index=False)


# %% Pool all observed ABL/ILD trials before histogramming each genotype
curves, bin_tables, curve_records = {}, [], []
for family in FAMILIES:
    for genotype in GENOTYPES:
        rows = window.loc[window.family.eq(family) & window.genotype.eq(genotype)]
        histogram = np.histogram(rows.timed_rt, bins=edges)[0]
        n = len(rows)
        assert histogram.sum() == n
        density = histogram / (n * BIN_WIDTH)
        assert np.isclose(np.sum(density * np.diff(edges)), 1)
        curves[(family, genotype)] = density
        curve_records.append({"family": family, "genotype": genotype, "animals": rows.animal.nunique(),
            "window_trials": n, "color": to_hex(COLORS[genotype]), "density_area": float(sum(density * BIN_WIDTH))})
        bin_tables.append(pd.DataFrame({"family": family, "genotype": genotype, "bin_index": np.arange(50),
            "bin_left_s": edges[:-1], "bin_right_s": edges[1:], "bin_count": histogram,
            "normalization_trials": n, "density_per_s": density}))
counts = pd.DataFrame(curve_records)
bins = pd.concat(bin_tables, ignore_index=True)
counts.to_csv(OUTPUT / "curve_counts.csv", index=False)
bins.to_csv(OUTPUT / "histogram_bins.csv", index=False)


# %% Independently parse source trials and count bins with standard-library bisect
independent = Counter()
independent_animals = {}
with INPUT.open(newline="") as stream:
    for row in csv.DictReader(stream):
        if row["success"] not in {"1", "-1"}:
            continue
        abl = float(row["ABL"] or "nan")
        abl = 60 if abl == 59 else abl
        ild = float(row["ILD"] or "nan")
        rt = float(row["timed_rt"] or "nan")
        if abl not in ABLS or ild not in SIGNED_ILDS or not np.isfinite(rt) or not 0 <= rt <= 1:
            continue
        family, genotype = row["filename"].split("_")[0], row["genotype"]
        index = min(49, bisect_right(edges, rt) - 1)
        independent[(family, genotype, index)] += 1
        independent_animals.setdefault((family, genotype), set()).add(row["animal"])
for row in bins.itertuples(index=False):
    assert row.bin_count == independent[(row.family, row.genotype, row.bin_index)]
for row in counts.itertuples(index=False):
    assert row.animals == len(independent_animals[(row.family, row.genotype)])
reference_counts = pd.read_csv(REFERENCE / "group_counts.csv").set_index(["family", "genotype"])
assert counts.set_index(["family", "genotype"]).window_trials.sort_index().equals(reference_counts.window_trials.sort_index())
# Each set of twenty 1 ms CDF bins must reproduce one 20 ms RTD bin.
fine = pd.read_csv(REFERENCE / "histogram_cdf_bins.csv.gz")
fine[["family", "genotype"]] = fine.group_id.str.split("/", expand=True)
fine["coarse_bin"] = fine.bin_index // 20
expected_bins = fine.groupby(["family", "genotype", "coarse_bin"]).hist_count.sum().sort_index()
actual_bins = bins.set_index(["family", "genotype", "bin_index"]).bin_count.sort_index()
np.testing.assert_array_equal(actual_bins.to_numpy(), expected_bins.to_numpy())


# %% One 1×2 PNG; common scales, genotype colors, full [0,1] view and normalization
plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12})
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True, sharey=True)
global_max = max(float(d.max()) for d in curves.values())
for ax, family in zip(axes, FAMILIES):
    for genotype in GENOTYPES:
        row = counts.loc[counts.family.eq(family) & counts.genotype.eq(genotype)].iloc[0]
        ax.stairs(curves[(family, genotype)], edges, fill=False, baseline=None,
            color=COLORS[genotype], linewidth=2,
            label=f"{LABELS[genotype]} | {row.animals} animals | n={row.window_trials:,}")
    ax.set(title=family, xlabel="Recorded timed_rt (s)", xlim=RT_WINDOW, ylim=(0, global_max * 1.12))
    ax.set_xticks(np.arange(0, 1.01, .1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
axes[0].set_ylabel("Density (s⁻¹)")
fig.suptitle("Session type 1 | Valid RTDs pooled across ABLs and ILDs", fontsize=16, y=.98)
fig.text(.5, .035, "success ∈ {1,−1}; RT in [0,1] s; 20 ms bins; each curve has area 1 | Raw trials pooled across animals and cohorts within family/genotype",
    ha="center", fontsize=9)
fig.text(.5, .006, "ABL 20/40/60 (59→60); signed ILDs ±1/2/4/8/16 | Prepared training/repeat/short-sound/calibration rules retained",
    ha="center", fontsize=9)
fig.subplots_adjust(left=.065, right=.985, top=.85, bottom=.18, wspace=.08)
png = OUTPUT / f"{NAME}.png"
fig.savefig(png, dpi=PNG_DPI, facecolor="white")
plt.close(fig)


# %% Verification/provenance manifest
for relative, digest in protected_hashes.items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, relative
manifest = {"status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "input": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256,
    "selection": "Prepared session1/training16; success in {1,-1}; ABL59->60 then20/40/60; ILDs±1/2/4/8/16; finite timed_rt in inclusive[0,1] seconds",
    "repeat_filter_reapplied": False, "calibration_rule": "CNTNAP2 cohort4 session>=13 already applied",
    "pooling": "Raw trials pooled across ABLs, signed ILDs, sessions, animals, and cohorts within family/genotype",
    "bin_width_s": BIN_WIDTH, "normalization_window_s": list(RT_WINDOW), "display_xlim_s": list(RT_WINDOW),
    "normalization": "Bin count/(curve window trials * .020); area 1 independently per curve",
    "selected_choices": len(selected), "window_trials": len(window), "animals": window.animal.nunique(),
    "curves": 6, "bins_per_curve": 50, "colors": {g: to_hex(c) for g, c in COLORS.items()},
    "legend_order": GENOTYPES, "family_order": FAMILIES, "curve_counts": curve_records,
    "figure_inches": list(FIGSIZE), "png_dpi": PNG_DPI, "png": str(png.relative_to(REPO)),
    "checks": ["All 300 bins independently counted from source CSV", "All bins agree with pooled earlier CDF histograms",
        "Six sample totals match current family/genotype audits", "Unit areas and 20 ms bins", "Protected inputs/previous artifacts unchanged"],
    "protected_file_sha256": protected_hashes}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(counts.to_string(index=False), flush=True)
print(f"Saved {png}; all 300 bins independently verified", flush=True)
