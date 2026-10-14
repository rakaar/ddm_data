# %% Imports
"""One animal: three-ABL psychometrics and ILD-pooled reaction-time densities.

Run .venv/bin/python asd_analysis/plot_animal_psychometric_and_rtd.py.
Both panels use correct/error trials. The RT axis is a display zoom only:
every finite measured timed_rt contributes to its ABL density normalization.
"""
import hashlib
import json
import warnings
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.special import expit


# %% Parameters
REPO = Path(__file__).resolve().parents[1]
ANIMAL = "ASD0054"
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO / "asd_analysis/session_type_1_animal_psychometric_rtd" / ANIMAL
ABLS = [20, 40, 60]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
MARKERS = {20: "o", 40: "s", 60: "^"}
GENOTYPE_LABELS = {"wt": "WT", "het": "Heterozygous", "hom": "Homozygous"}
BIN_WIDTH = 0.020  # Seconds; 20 ms bins, with no RT selection window.
RT_DISPLAY_XLIM = (0.0, 0.6)
PSYCHOMETRIC_XLIM = (-18, 18)
INITIAL_K = 0.1
FIT_BOUNDS = ([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf])
MAXFEV = 10000
PNG_DPI = 200
OUTPUT.mkdir(parents=True, exist_ok=True)
input_stat = (INPUT.stat().st_size, INPUT.stat().st_mtime_ns)
with INPUT.open("rb") as stream:
    input_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()


# %% Read only this animal from the already repeat-filtered, cleaned CSV
columns = ["filename", "animal", "genotype", "session_type", "training_level",
           "trial_is_repeat", "is_short_sound", "ABL", "ILD", "success",
           "response_poke", "timed_rt", "source_file", "session"]
parts = []
for chunk in pd.read_csv(INPUT, usecols=columns, chunksize=100000):
    parts.append(chunk.loc[chunk.animal.eq(ANIMAL)].copy())
animal_rows = pd.concat(parts, ignore_index=True)
assert len(animal_rows) > 0, f"{ANIMAL} is absent from {INPUT}"
assert animal_rows.session_type.eq(1).all() and animal_rows.training_level.eq(16).all()
assert animal_rows.trial_is_repeat.eq(False).all()
assert animal_rows.is_short_sound.eq(0).all()
animal_rows["plot_ABL"] = animal_rows.ABL.replace({59: 60}).astype(int)
selected = animal_rows.loc[
    animal_rows.success.isin([1, -1])
    & animal_rows.plot_ABL.isin(ABLS)
    & animal_rows.ILD.isin(SIGNED_ILDS)
].copy()
assert selected.response_poke.isin([1, -1]).all()
assert selected.filename.nunique() == 1 and selected.genotype.nunique() == 1
filename = selected.filename.iloc[0]
genotype = selected.genotype.iloc[0]
selected["right"] = selected.response_poke.eq(1)
selected["finite_rt"] = np.isfinite(selected.timed_rt)
print(f"{ANIMAL} | {filename} | {GENOTYPE_LABELS[genotype]} | {len(selected):,} choices", flush=True)

# The existing animal-membership audit uses this same unrestricted-RT selection.
reference = pd.read_csv(REPO / "asd_analysis/session_type_1_psychometrics/psychometric_group_membership.csv")
reference = reference.loc[reference.view.eq("filename_genotype") & reference.animal.eq(ANIMAL)]
assert len(reference) == 1
assert len(selected) == reference.trials.iloc[0]
assert int(selected.right.sum()) == reference.right_trials.iloc[0]


# %% Measured P(right), with no reaction-time restriction
points = selected.groupby(["plot_ABL", "ILD"], as_index=False).agg(
    trials=("response_poke", "size"), right_trials=("right", "sum"),
)
points["left_trials"] = points.trials - points.right_trials
points["p_right"] = points.right_trials / points.trials
assert len(points) == len(ABLS) * len(SIGNED_ILDS)
assert int(points.trials.sum()) == len(selected)
points.to_csv(OUTPUT / "psychometric_points.csv", index=False)


# %% Fit the same four-parameter sigmoid to each ABL's ten probability points
def sigmoid(ild, upper, lower, x0, k):
    return lower + (upper - lower) * expit(k * (ild - x0))


fits = []
for abl in ABLS:
    condition = points.loc[points.plot_ABL.eq(abl)].sort_values("ILD")
    x = condition.ILD.to_numpy()
    probability = condition.p_right.to_numpy()
    initial = [float(probability.max()), float(probability.min()), float(np.median(x)), INITIAL_K]
    fit = {"plot_ABL": abl, "upper": np.nan, "lower": np.nan,
           "x0": np.nan, "k": np.nan, "fit_status": "optimizer_failed",
           "initial_parameters": json.dumps(initial), "optimizer_message": "", "warnings": ""}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", OptimizeWarning)
            parameters, _ = curve_fit(sigmoid, x, probability, p0=initial,
                                      bounds=FIT_BOUNDS, maxfev=MAXFEV)
        fit.update(dict(zip(["upper", "lower", "x0", "k"], parameters.tolist())))
        fit["warnings"] = "; ".join(str(item.message) for item in caught)
        if not np.isfinite(parameters).all():
            fit["fit_status"] = "nonfinite_parameters"
        else:
            fit["fit_status"] = "ok" if parameters[0] > parameters[1] and parameters[3] > 0 else "non_increasing"
            fitted = sigmoid(x, *parameters)
            fit["sum_squared_residuals"] = float(np.square(fitted - probability).sum())
    except (RuntimeError, ValueError) as error:
        fit["optimizer_message"] = str(error)
    fits.append(fit)
fits = pd.DataFrame(fits)
fits.to_csv(OUTPUT / "psychometric_fits.csv", index=False)


# %% ILD-pooled 20 ms histograms: include every finite RT, including values above 1 s
finite = selected.loc[selected.finite_rt, "timed_rt"].to_numpy()
assert len(finite) > 0
# Extend a common grid far enough to contain all finite RTs and the display range.
first_bin = int(min(0, np.floor(finite.min() / BIN_WIDTH)))
last_bin = int(max(np.ceil(finite.max() / BIN_WIDTH), np.ceil(RT_DISPLAY_XLIM[1] / BIN_WIDTH)))
edges = np.arange(first_bin, last_bin + 1) * BIN_WIDTH
np.testing.assert_allclose(np.diff(edges), BIN_WIDTH, rtol=0, atol=1e-12)
histogram_parts, counts = [], []
for abl in ABLS:
    rows = selected.loc[selected.plot_ABL.eq(abl)]
    rt = rows.loc[rows.finite_rt, "timed_rt"].to_numpy()
    bin_counts, _ = np.histogram(rt, bins=edges)
    n = len(rt)
    assert n > 0 and int(bin_counts.sum()) == n
    density = bin_counts / (n * BIN_WIDTH)
    area = float(np.dot(density, np.diff(edges)))
    assert np.isclose(area, 1, rtol=0, atol=1e-12)
    histogram_parts.append(pd.DataFrame({
        "plot_ABL": abl, "bin_index": np.arange(len(bin_counts)),
        "bin_left_s": edges[:-1], "bin_right_s": edges[1:],
        "bin_count": bin_counts, "normalization_trials": n, "density_per_s": density,
    }))
    counts.append({
        "animal": ANIMAL, "filename": filename, "genotype": genotype, "plot_ABL": abl,
        "choice_trials": len(rows), "right_trials": int(rows.right.sum()),
        "left_trials": int((~rows.right).sum()), "finite_rt_trials": n,
        "nonfinite_rt_trials": int((~rows.finite_rt).sum()),
        "rt_above_1_s": int((rt > 1).sum()),
        "rt_in_display": int(((rt >= RT_DISPLAY_XLIM[0]) & (rt <= RT_DISPLAY_XLIM[1])).sum()),
        "rt_min_s": float(rt.min()), "rt_max_s": float(rt.max()),
        "histogram_trials": int(bin_counts.sum()), "full_density_area": area,
        "color": to_hex(COLORS[abl]),
    })
histograms = pd.concat(histogram_parts, ignore_index=True)
counts = pd.DataFrame(counts)
assert int(histograms.bin_count.sum()) == int(selected.finite_rt.sum())
histograms.to_csv(OUTPUT / "rtd_histogram_bins.csv", index=False)
counts.to_csv(OUTPUT / "abl_counts.csv", index=False)
print(counts[["plot_ABL", "choice_trials", "finite_rt_trials", "rt_above_1_s", "full_density_area"]].to_string(index=False), flush=True)


# %% One row, two panels: observed probabilities and full-RT densities
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
fig.suptitle(f"{ANIMAL} | {filename.removesuffix('.csv')} | {GENOTYPE_LABELS[genotype]}\n"
             f"Session type 1 | Correct/error trials | n={len(selected):,}", fontsize=16, y=.975)
smooth_ild = np.linspace(*PSYCHOMETRIC_XLIM, 600)
for abl in ABLS:
    condition = points.loc[points.plot_ABL.eq(abl)].sort_values("ILD")
    count = counts.loc[counts.plot_ABL.eq(abl)].iloc[0]
    fit = fits.loc[fits.plot_ABL.eq(abl)].iloc[0]
    axes[0].scatter(condition.ILD, condition.p_right, s=42,
                    marker=MARKERS[abl], color=COLORS[abl], zorder=3,
                    label=f"ABL {abl} | n={int(count.choice_trials):,}")
    if fit.fit_status == "ok":
        axes[0].plot(smooth_ild, sigmoid(smooth_ild, *fit[["upper", "lower", "x0", "k"]].to_numpy(dtype=float)),
                     color=COLORS[abl], linewidth=1.7)
    bins = histograms.loc[histograms.plot_ABL.eq(abl)].sort_values("bin_index")
    axes[1].stairs(bins.density_per_s, edges, color=COLORS[abl], linewidth=1.6,
                   fill=False, label=f"ABL {abl} | n={int(count.finite_rt_trials):,}")
axes[0].set(title="Psychometric: P(right) versus signed ILD",
            xlabel="Signed ILD (dB)", ylabel="P(right)",
            xlim=PSYCHOMETRIC_XLIM, ylim=(0, 1))
axes[0].set_xticks([-16, -8, -4, 0, 4, 8, 16])
axes[1].set(title="RTDs: all signed ILDs pooled",
            xlabel="Measured timed_rt (s)", ylabel="Probability density (s⁻¹)",
            xlim=RT_DISPLAY_XLIM, ylim=(0, None))
axes[1].set_xticks(np.arange(0, RT_DISPLAY_XLIM[1] + .001, .1))
visible_bins = histograms.loc[histograms.bin_left_s.lt(RT_DISPLAY_XLIM[1])
                               & histograms.bin_right_s.gt(RT_DISPLAY_XLIM[0])]
axes[1].set_ylim(0, float(visible_bins.density_per_s.max()) * 1.12)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=10, loc="upper left" if ax is axes[0] else "upper right")
fig.text(.5, .045, "ABL 59 mapped to 60 | signed ILDs: ±1, ±2, ±4, ±8, ±16 dB | Psychometric: no RT restriction",
         ha="center", fontsize=10)
fig.text(.5, .012, "RTD: 20 ms bins; every finite RT used; full-range area = 1 per ABL; 0–0.6 s is a display zoom only",
         ha="center", fontsize=10)
fig.subplots_adjust(left=.07, right=.975, bottom=.18, top=.77, wspace=.29)
stem = f"{ANIMAL}_session_type_1_psychometric_and_full_rt_rtd"
png_path = OUTPUT / f"{stem}.png"
pdf_path = OUTPUT / f"{stem}.pdf"
fig.savefig(png_path, dpi=PNG_DPI, facecolor="white")
fig.savefig(pdf_path, facecolor="white")
plt.close(fig)


# %% Validation manifest and preservation checks
with fitz.open(pdf_path) as document:
    assert document.page_count == 1
    assert ANIMAL in document[0].get_text()
assert input_stat == (INPUT.stat().st_size, INPUT.stat().st_mtime_ns)
manifest = {
    "status": "passed", "animal": ANIMAL, "filename": filename, "genotype": genotype,
    "input": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256,
    "source_script": str(Path(__file__).relative_to(REPO)),
    "selection": "Cleaned session-type-1 CSV; training_level 16 and repeat eligibility already applied; success in {1,-1}; no RT restriction",
    "repeat_filter_reapplied": False, "abl_mapping": "59 to 60", "abls": ABLS, "signed_ilds": SIGNED_ILDS,
    "prepared_animal_rows": len(animal_rows), "choice_trials": len(selected),
    "right_trials": int(selected.right.sum()), "left_trials": int((~selected.right).sum()),
    "finite_rt_trials": int(selected.finite_rt.sum()),
    "psychometric_points": len(points), "fit_status_counts": fits.fit_status.value_counts().to_dict(),
    "sigmoid": "lower + (upper - lower) / (1 + exp(-k * (ILD - x0)))",
    "fit_parameter_order": ["upper", "lower", "x0", "k"],
    "fit_initial_guess": "[max(P), min(P), median(ILD), 0.1]",
    "fit_bounds": {"upper": [0, 1], "lower": [0, 1], "x0": "unrestricted", "k": [0, "infinity"]},
    "fit_weighting": "Unweighted curve_fit on ten measured ILD probability points per ABL",
    "fit_maxfev": MAXFEV, "rtd_bin_width_s": BIN_WIDTH,
    "rtd_normalization": "bin_count / (all finite correct/error RTs at that ABL * 0.020 s)",
    "rtd_upper_rt_cutoff": None, "rtd_display_xlim_s": list(RT_DISPLAY_XLIM),
    "rtd_histogram_range_s": [float(edges[0]), float(edges[-1])],
    "pooling": "Pool raw observed trials across source sessions and all selected signed ILDs within each ABL",
    "counts": counts.to_dict(orient="records"), "pdf_pages": 1,
    "png": str(png_path.relative_to(REPO)), "pdf": str(pdf_path.relative_to(REPO)),
    "checks": ["existing psychometric membership counts", "30 measured probability points",
               "all finite RTs histogrammed", "20 ms bins", "unit areas over full RT range",
               "single-page PDF", "input size and modification time unchanged"],
}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Saved {png_path}\nSaved {pdf_path}", flush=True)
