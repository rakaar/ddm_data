# %% Imports
"""Two rows per animal: existing psychometric/RTD/chronometric plus CDF/scaling.

Run with .venv/bin/python from the repository root. Existing three-panel audits
are reused byte-for-byte. Only the new row restricts measured timed_rt to [0,1].
Cutoffs are calculated separately for each animal and |ILD|, not copied from
the original paper's hard-coded across-animal cutoff dictionary.
"""
import hashlib
import json
from pathlib import Path
import shutil

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.special import expit


# %% Editable parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
REFERENCE = REPO / "asd_analysis/session_type_1_all_animals_psychometric_rtd_chronometric"
OUTPUT = REPO / "asd_analysis/session_type_1_all_animals_psychometric_rtd_chronometric_scaling"
ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
ABL_MARKERS = {20: "o", 40: "s", 60: "^"}
ILD_STYLES = {1: "-", 2: "--", 4: ":", 8: "-.", 16: (0, (5, 1, 1, 1, 1, 1))}
ILD_COLORS = dict(zip(ABS_ILDS, [to_hex(c) for c in plt.cm.viridis(np.linspace(.08, .86, 5))]))
PLOT_QUANTILES = np.array([.1, .3, .5, .7, .9])
FIT_QUANTILES = np.arange(.01, 1, .01)
QUANTILE_COLORS = [to_hex(c) for c in plt.cm.plasma(np.linspace(.08, .86, 5))]
QUANTILE_MARKERS = ["o", "s", "^", "D", "v"]
GENOTYPE_LABELS = {"wt": "WT", "het": "HET", "hom": "HOM"}
CDF_BIN_WIDTH = .001
NEW_RT_WINDOW = (0, 1)
CDF_DISPLAY_XLIM = (0, .15)
ONSET_WINDOW_BINS = 10
ONSET_RISE = .005
FIT_MAX_RT = .3
MIN_CONDITION_TRIALS = 5
ANIMALS_PER_PAGE = 5
PAGE_WIDTH = 18.24
ANIMAL_BLOCK_HEIGHT = 6.6
HEADER_HEIGHT = 1.6
FOOTER_HEIGHT = .6
UPPER_PANEL_HEIGHT = 2.2014248704663224
NEW_PANEL_SIZE = 2.2
PNG_DPI = 160
NAME = "ASD_session_type_1_all_animals_psychometric_rtd_chronometric_scaling"
COPIED_AUDITS = ["animal_pages.csv", "psychometric_points.csv", "psychometric_fits.csv",
                 "rtd_histogram_bins.csv", "abl_counts.csv", "chronometric_points.csv"]
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Reusable Figure 1 calculations, also exercised on synthetic failure cases
def detect_onset(curve, window_bins=ONSET_WINDOW_BINS, rise=ONSET_RISE):
    """First forward-minus-backward window rise; the current bin is excluded."""
    previous = np.full(len(curve), np.nan)
    following = np.full(len(curve), np.nan)
    if len(curve) >= 2 * window_bins + 1:
        windows = np.lib.stride_tricks.sliding_window_view(curve, window_bins).mean(axis=1)
        previous[window_bins:-window_bins] = windows[:len(curve) - 2 * window_bins]
        following[window_bins:-window_bins] = windows[window_bins + 1:]
    hits = np.flatnonzero(following - previous >= rise)
    return (int(hits[0]) if len(hits) else None), previous, following


def fit_quantile_scale(reference_q, other_q, cutoff, max_rt=FIT_MAX_RT):
    """Match the original Figure 1 shifted-difference, through-origin fit."""
    result = {"fit_status": "missing_cutoff", "fit_points": 0, "slope_m": np.nan,
              "scale_factor": np.nan, "first_difference_s": np.nan, "sum_x_squared": np.nan}
    mask = np.zeros(len(reference_q), dtype=bool)
    if not np.isfinite(cutoff):
        return result, mask
    if not np.isfinite(reference_q).all() or not np.isfinite(other_q).all():
        result["fit_status"] = "nonfinite_quantiles"
        return result, mask
    mask = (reference_q >= cutoff) & (reference_q <= max_rt)
    result["fit_points"] = int(mask.sum())
    if mask.sum() < 2:
        result["fit_status"] = "insufficient_fit_points"
        return result, mask
    x = reference_q[mask] - cutoff
    y = (other_q - reference_q)[mask]
    denominator = float(x @ x)
    result.update(first_difference_s=float(y[0]), sum_x_squared=denominator)
    if denominator <= 0:
        result["fit_status"] = "zero_reference_range"
        return result, mask
    slope = float(x @ (y - y[0]) / denominator)
    factor = 1 + slope
    result.update(slope_m=slope, scale_factor=factor,
                  fit_status="ok" if np.isfinite(factor) and factor > 0 else "invalid_scale_factor")
    return result, mask


def rescale_quantiles(values, cutoff, factor):
    if not np.isfinite(cutoff) or not np.isfinite(factor) or factor <= 0:
        return np.full_like(values, np.nan, dtype=float)
    return np.where(values > cutoff, cutoff + (values - cutoff) / factor, values)


# %% Snapshot protected inputs and prior outputs; reuse existing three-panel audits
old_manifest = json.loads((REFERENCE / "validation.json").read_text())
assert old_manifest["status"] == "passed"
assert json.loads((REFERENCE / "independent_validation.json").read_text())["status"] == "passed"
protected = set(REPO.glob("SHANK3_cohort*.csv")) | set(REPO.glob("CNTNAP2_cohort*.csv"))
protected.update(INPUT.parent.glob("*.csv"))
for folder in (REPO / "asd_analysis").glob("session_type_1_*"):
    if folder.is_dir() and folder != OUTPUT:
        protected.update(p for p in folder.rglob("*") if p.is_file() and p.suffix in {".csv", ".json", ".png", ".pdf"})
for date in ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]:
    protected.update(p for p in (REPO / f"docs/assets/results/{date}").iterdir() if p.is_file())
protected_hashes = {}
for path in sorted(protected):
    with path.open("rb") as stream:
        protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_sha256 = protected_hashes[str(INPUT.relative_to(REPO))]
assert input_sha256 == old_manifest["input_sha256"]
for name in COPIED_AUDITS:
    shutil.copy2(REFERENCE / name, OUTPUT / f"reference_{name}")
members = pd.read_csv(REFERENCE / "animal_pages.csv")
psy_points = pd.read_csv(REFERENCE / "psychometric_points.csv")
psy_fits = pd.read_csv(REFERENCE / "psychometric_fits.csv").set_index(["animal", "plot_ABL"])
rtd_bins = pd.read_csv(REFERENCE / "rtd_histogram_bins.csv")
chrono = pd.read_csv(REFERENCE / "chronometric_points.csv")
assert len(members) == 53 and members.groupby("pdf_page").size().to_list() == [5] * 10 + [3]
identity = members.set_index("animal")[["filename", "genotype"]].to_dict("index")


# %% Select [0,1]-second RTs for the NEW row only; pool raw signed trials/sessions
data = pd.read_csv(INPUT, usecols=["filename", "genotype", "animal", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "success", "ABL", "ILD", "timed_rt"], low_memory=False)
assert len(data) == 642370 and data.animal.nunique() == 53
assert data.session_type.eq(1).all() and data.training_level.eq(16).all()
assert data.trial_is_repeat.eq(False).all() and data.is_short_sound.eq(0).all()
data["plot_ABL"] = data.ABL.replace({59: 60})
selected = data.loc[data.success.isin([1, -1]) & data.plot_ABL.isin(ABLS) & data.ILD.abs().isin(ABS_ILDS)].copy()
selected["plot_ABL"] = selected.plot_ABL.astype(int)
selected["abs_ILD"] = selected.ILD.abs().astype(int)
selected["window_rt"] = np.isfinite(selected.timed_rt) & selected.timed_rt.between(*NEW_RT_WINDOW)
window = selected.loc[selected.window_rt].copy()
window["positive_ild"] = window.ILD.gt(0)
assert len(selected) == 471416 and np.isfinite(selected.timed_rt).sum() == 469228
assert len(window) == 466978 and window.animal.nunique() == 53
assert selected.loc[np.isfinite(selected.timed_rt), "timed_rt"].ge(0).all()
counts = window.groupby(["filename", "genotype", "animal", "plot_ABL", "abs_ILD"], as_index=False).agg(
    window_trials=("timed_rt", "size"), positive_ild_trials=("positive_ild", "sum"))
counts["negative_ild_trials"] = counts.window_trials - counts.positive_ild_trials
counts = counts.merge(chrono[["filename", "genotype", "animal", "plot_ABL", "abs_ILD",
    "valid_choice_trials", "finite_rt_trials", "nonfinite_rt_trials", "rt_above_1_s"]],
    on=["filename", "genotype", "animal", "plot_ABL", "abs_ILD"], validate="one_to_one")
assert len(counts) == 795 and counts.window_trials.min() >= MIN_CONDITION_TRIALS
assert (counts.window_trials + counts.rt_above_1_s).eq(counts.finite_rt_trials).all()
counts.to_csv(OUTPUT / "condition_counts.csv", index=False)
print(f"New row: {len(window):,} trials in [0,1] s, 53 animals, 795 conditions", flush=True)


# %% Per-condition histograms, CDFs, five displayed quantiles, and 99 fitting quantiles
bin_edges = np.arange(0, 1 + CDF_BIN_WIDTH, CDF_BIN_WIDTH)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
cdfs, quantiles = {}, {}
bin_tables, fine_tables, raw_quantile_records = [], [], []
for (animal, abl, ild), group in window.groupby(["animal", "plot_ABL", "abs_ILD"]):
    values = group.timed_rt.to_numpy()
    hist = np.histogram(values, bins=bin_edges)[0]
    assert hist.sum() == len(values)
    cdf = np.cumsum(hist) / len(values)
    coarse = np.quantile(values, PLOT_QUANTILES, method="linear")
    fine = np.quantile(values, FIT_QUANTILES, method="linear")
    key = (animal, int(abl), int(ild))
    cdfs[key] = cdf
    quantiles[key] = {"coarse": coarse, "fine": fine}
    bin_tables.append(pd.DataFrame({"animal": animal, "plot_ABL": int(abl), "abs_ILD": int(ild),
        "bin_index": np.arange(len(hist)), "bin_left_s": bin_edges[:-1], "bin_right_s": bin_edges[1:],
        "bin_center_s": bin_centers, "hist_count": hist, "cdf": cdf}))
    fine_tables.append(pd.DataFrame({"animal": animal, "plot_ABL": int(abl), "abs_ILD": int(ild),
                                    "quantile": FIT_QUANTILES, "rt_s": fine}))
    for q, value in zip(PLOT_QUANTILES, coarse):
        raw_quantile_records.append({"animal": animal, "plot_ABL": int(abl), "abs_ILD": int(ild),
                                    "quantile": float(q), "raw_rt_s": float(value), "window_trials": len(values)})
pd.concat(bin_tables, ignore_index=True).to_csv(OUTPUT / "histogram_cdf_bins.csv.gz", index=False, compression="gzip")
pd.concat(fine_tables, ignore_index=True).to_csv(OUTPUT / "fine_quantiles.csv.gz", index=False, compression="gzip")
raw_quantiles = pd.DataFrame(raw_quantile_records)
assert len(raw_quantiles) == 3975 and len(cdfs) == 795
raw_quantiles.to_csv(OUTPUT / "raw_quantiles.csv", index=False)
del bin_tables, fine_tables


# %% Fresh mean absolute pairwise ΔCDFs and cutoffs, one per animal/|ILD|
delta_curves, threshold_records, delta_tables = {}, [], []
for animal in members.animal:
    for ild in ABS_ILDS:
        c20, c40, c60 = [cdfs[(animal, abl, ild)] for abl in ABLS]
        d2040, d4060, d2060 = abs(c20 - c40), abs(c40 - c60), abs(c20 - c60)
        delta = (d2040 + d4060 + d2060) / 3
        onset, previous, following = detect_onset(delta)
        cutoff = float(bin_centers[onset]) if onset is not None else np.nan
        threshold_records.append({"animal": animal, **identity[animal], "abs_ILD": ild,
            "threshold_status": "ok" if onset is not None else "no_onset", "onset_bin_index": onset,
            "cutoff_s": cutoff, "cutoff_ms": cutoff * 1000, "window_bins": ONSET_WINDOW_BINS,
            "required_rise": ONSET_RISE, "previous_mean": previous[onset] if onset is not None else np.nan,
            "following_mean": following[onset] if onset is not None else np.nan,
            "observed_rise": following[onset] - previous[onset] if onset is not None else np.nan})
        delta_curves[(animal, ild)] = delta
        delta_tables.append(pd.DataFrame({"animal": animal, "abs_ILD": ild,
            "bin_index": np.arange(len(delta)), "bin_center_s": bin_centers,
            "abs_20_minus_40": d2040, "abs_40_minus_60": d4060, "abs_20_minus_60": d2060,
            "mean_abs_delta_cdf": delta, "previous_mean": previous, "following_mean": following,
            "window_rise": following - previous}))
thresholds = pd.DataFrame(threshold_records)
thresholds.to_csv(OUTPUT / "thresholds.csv", index=False)
pd.concat(delta_tables, ignore_index=True).to_csv(OUTPUT / "delta_cdf_curves.csv.gz", index=False, compression="gzip")
del delta_tables
threshold_lookup = thresholds.set_index(["animal", "abs_ILD"])
assert len(thresholds) == 265
print(f"Thresholds: {thresholds.threshold_status.eq('ok').sum()}/265; "
      f"range {thresholds.cutoff_ms.min():.1f}–{thresholds.cutoff_ms.max():.1f} ms", flush=True)


# %% Figure 1 scaling fits, their complete fitting inputs, and transformed quantiles
scale_records, fit_input_tables, scaled_records = [], [], []
for animal in members.animal:
    for ild in ABS_ILDS:
        cutoff = threshold_lookup.loc[(animal, ild), "cutoff_s"]
        q60 = quantiles[(animal, 60, ild)]["fine"]
        for abl in ABLS:
            raw = quantiles[(animal, abl, ild)]["coarse"]
            if abl == 60:
                transformed, status = raw.copy(), "reference_unchanged"
            else:
                other = quantiles[(animal, abl, ild)]["fine"]
                result, mask = fit_quantile_scale(q60, other, cutoff)
                scale_records.append({"animal": animal, **identity[animal], "abs_ILD": ild, "plot_ABL": abl,
                                      "cutoff_s": cutoff, "fit_max_rt_s": FIT_MAX_RT, **result})
                fit_input_tables.append(pd.DataFrame({"animal": animal, "plot_ABL": abl, "abs_ILD": ild,
                    "quantile": FIT_QUANTILES, "reference_rt_s": q60, "other_rt_s": other,
                    "difference_s": other - q60, "x_s": q60 - cutoff,
                    "shifted_difference_s": other - q60 - result["first_difference_s"], "used_for_fit": mask}))
                status = result["fit_status"]
                transformed = rescale_quantiles(raw, cutoff, result["scale_factor"]) if status == "ok" else np.full_like(raw, np.nan)
            for q, before, after in zip(PLOT_QUANTILES, raw, transformed):
                scaled_records.append({"animal": animal, "plot_ABL": abl, "abs_ILD": ild,
                    "quantile": float(q), "raw_rt_s": float(before), "scaled_rt_s": float(after),
                    "cutoff_s": cutoff, "scaling_status": status})
scales = pd.DataFrame(scale_records)
scaled = pd.DataFrame(scaled_records)
assert len(scales) == 530 and len(scaled) == 3975
scales.to_csv(OUTPUT / "scaling_fits.csv", index=False)
scaled.to_csv(OUTPUT / "scaled_quantiles.csv", index=False)
pd.concat(fit_input_tables, ignore_index=True).to_csv(OUTPUT / "scaling_fit_inputs.csv.gz", index=False, compression="gzip")
del fit_input_tables
print(f"Rescaling fits: {scales.fit_status.eq('ok').sum()}/530 valid", flush=True)


# %% Reconstruct existing RTDs exactly from their saved full-range histograms
rtd_curves = {}
for (animal, abl), bins in rtd_bins.groupby(["animal", "plot_ABL"]):
    bins = bins.sort_values("bin_index")
    rtd_curves[(animal, abl)] = (bins.density_per_s.to_numpy(),
        np.r_[bins.bin_left_s.to_numpy(), bins.bin_right_s.iloc[-1]], int(bins.normalization_trials.iloc[0]))


# %% Two rows per animal, with physical panel sizes kept readable on taller pages
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})
pdf_path = OUTPUT / f"{NAME}.pdf"
page_records, layout_records, plotted_records = [], [], []
old_widths = old_manifest["pages"][0]["panel_widths_in"]
left, right = .043 * PAGE_WIDTH, .98 * PAGE_WIDTH
upper_gap = (right - left - sum(old_widths)) / 2
upper_x = [left, left + old_widths[0] + upper_gap, right - old_widths[2]]
lower_gap = (right - left - 6 * NEW_PANEL_SIZE) / 5
lower_x = [left + j * (NEW_PANEL_SIZE + lower_gap) for j in range(6)]
with PdfPages(pdf_path, metadata={"Title": "ASD: per-animal psychometrics, RTDs, chronometrics, CDFs and quantile scaling"}) as pdf:
    for page_number, batch in members.groupby("pdf_page", sort=True):
        height = HEADER_HEIGHT + ANIMAL_BLOCK_HEIGHT * len(batch) + FOOTER_HEIGHT
        fig = plt.figure(figsize=(PAGE_WIDTH, height))
        fig.text(.5, 1 - .31 / height, f"Session type 1 | Psychometric, RTD, chronometric, CDF, and quantile scaling | Page {page_number}/11",
                 ha="center", fontsize=14)
        fig.text(.5, 1 - .59 / height, "Upper row: original valid-choice / all-finite-RT analyses | Lower row: measured RT in [0,1] s; fresh per-animal cutoffs",
                 ha="center", fontsize=9)
        abl_legend = fig.legend(handles=[Line2D([], [], color=COLORS[a], label=f"ABL {a} dB") for a in ABLS],
            loc="upper center", bbox_to_anchor=(.5, 1 - .73 / height), ncol=3, frameon=False, fontsize=9)
        ild_legend = fig.legend(handles=[Line2D([], [], color=ILD_COLORS[i], linestyle=ILD_STYLES[i], label=f"|ILD| {i}") for i in ABS_ILDS],
            loc="upper center", bbox_to_anchor=(.5, 1 - .98 / height), ncol=5, frameon=False, fontsize=8,
            title="CDF: use these line styles | ΔCDF: use these colors")
        ild_legend.get_title().set_fontsize(8)
        fig.legend(handles=[Line2D([], [], color=QUANTILE_COLORS[j], marker=QUANTILE_MARKERS[j], label=f"{int(q*100)}%")
            for j, q in enumerate(PLOT_QUANTILES)], loc="upper center", bbox_to_anchor=(.5, 1 - 1.31 / height),
            ncol=5, frameon=False, fontsize=8)
        for row, member in enumerate(batch.itertuples(index=False)):
            animal = member.animal
            block_top = height - HEADER_HEIGHT - row * ANIMAL_BLOCK_HEIGHT
            upper_y = block_top - .45 - UPPER_PANEL_HEIGHT
            upper_axes = []
            for j in range(3):
                h = old_widths[2] if j == 2 else UPPER_PANEL_HEIGHT
                y = upper_y + (UPPER_PANEL_HEIGHT - h) / 2
                upper_axes.append(fig.add_axes([upper_x[j]/PAGE_WIDTH, y/height, old_widths[j]/PAGE_WIDTH, h/height]))
            psy_ax, rt_ax, chrono_ax = upper_axes
            for abl in ABLS:
                p = psy_points.loc[psy_points.animal.eq(animal) & psy_points.plot_ABL.eq(abl)].sort_values("ILD")
                estimate = psy_fits.loc[(animal, abl)]
                psy_ax.scatter(p.ILD, p.p_right, color=COLORS[abl], marker=ABL_MARKERS[abl], s=26,
                               label=f"ABL {abl} | n={int(p.trials.sum()):,}", zorder=3)
                smooth = np.linspace(-18, 18, 600)
                psy_ax.plot(smooth, estimate.lower + (estimate.upper - estimate.lower) * expit(estimate.k * (smooth - estimate.x0)),
                            color=COLORS[abl], linewidth=1.7)
                density, edges, n = rtd_curves[(animal, abl)]
                rt_ax.stairs(density, edges, color=COLORS[abl], linewidth=1.6, fill=False, label=f"ABL {abl} | n={n:,}")
                c = chrono.loc[chrono.animal.eq(animal) & chrono.plot_ABL.eq(abl)].sort_values("abs_ILD")
                chrono_ax.errorbar(c.plot_x, c.mean_rt_s, yerr=c.sem_rt_s, fmt="o", linestyle="None", color=COLORS[abl],
                                  markersize=4.5, elinewidth=.9, capsize=2, capthick=.9, zorder=3)
            psy_ax.set(title=f"{animal} | {member.filename.removesuffix('.csv')} | {GENOTYPE_LABELS[member.genotype]}\n"
                             f"Psychometric | {int(member.choice_trials):,} valid choices",
                       xlabel="Signed ILD (dB)", ylabel="P(right)", xlim=(-18, 18), ylim=(0, 1))
            psy_ax.set_xticks([-16, -8, -4, 0, 4, 8, 16])
            rt_ax.set(title=f"{animal} | RTDs, all ILDs\n{int(member.finite_rt_trials):,} finite RTs; {int(member.nonfinite_rt_trials):,} missing",
                      xlabel="Measured timed_rt (s)", ylabel="Density (s⁻¹)", xlim=(0, .6))
            rt_ax.set_xticks(np.arange(0, .601, .1))
            visible_max = max(d[(e[:-1] < .6) & (e[1:] > 0)].max() for d, e, _ in [rtd_curves[(animal, a)] for a in ABLS])
            rt_ax.set_ylim(0, visible_max * 1.12)
            chrono_ax.set(title=f"{animal} | Chronometric\nPooled ±ILD", xlabel="|ILD| (dB)", ylabel="Mean RT ± SEM (s)",
                          xlim=(-.45, 4.45), ylim=(0, .45))
            chrono_ax.set_xticks(np.arange(5), [str(i) for i in ABS_ILDS])
            chrono_ax.set_yticks(np.arange(0, .451, .1))
            psy_ax.legend(loc="upper left", frameon=False, fontsize=7)
            rt_ax.legend(loc="upper right", frameon=False, fontsize=7)
            new_axes = [fig.add_axes([x/PAGE_WIDTH, (block_top - 6)/height, NEW_PANEL_SIZE/PAGE_WIDTH, NEW_PANEL_SIZE/height]) for x in lower_x]
            cdf_ax, delta_ax, *q_axes = new_axes
            for ild in ABS_ILDS:
                for abl in ABLS:
                    cdf_ax.plot(bin_centers, cdfs[(animal, abl, ild)], color=COLORS[abl], linestyle=ILD_STYLES[ild], linewidth=1)
                delta_ax.plot(bin_centers, delta_curves[(animal, ild)], color=ILD_COLORS[ild], linewidth=1.2)
                t = threshold_lookup.loc[(animal, ild), "cutoff_s"]
                if np.isfinite(t):
                    delta_ax.axvline(t, color=ILD_COLORS[ild], linestyle="--", linewidth=.9)
            cdf_ax.set(title="CDFs | all five |ILD|\n1 ms bins, normalized over [0,1] s", xlabel="RT (s)", ylabel="CDF", xlim=CDF_DISPLAY_XLIM, ylim=(0, 1))
            cutoff_labels = [f"{i}: {threshold_lookup.loc[(animal,i), 'cutoff_ms']:.1f}" if np.isfinite(threshold_lookup.loc[(animal,i), 'cutoff_s']) else f"{i}: N/A" for i in ABS_ILDS]
            delta_ax.set(title="ΔCDF | cutoffs (ms)\n" + "; ".join(cutoff_labels[:3]) + "\n" + "; ".join(cutoff_labels[3:]),
                         xlabel="RT (s)", ylabel="Mean |ΔCDF|", xlim=CDF_DISPLAY_XLIM)
            delta_ax.title.set_fontsize(7.8)
            for ax in [cdf_ax, delta_ax]:
                ax.set_xticks([0, .075, .15], ["0", ".075", ".15"])
            delta_visible = max(float(delta_curves[(animal,i)][bin_centers <= .15].max()) for i in ABS_ILDS)
            delta_ax.set_ylim(0, max(.025, delta_visible * 1.12))
            animal_scaled = scaled.loc[scaled.animal.eq(animal)]
            y_max = float(np.nanmax(animal_scaled[["raw_rt_s", "scaled_rt_s"]]))
            q_ymax = max(.1, float(np.ceil(y_max * 1.05 / .05) * .05))
            for j, abl in enumerate(ABLS):
                ax = q_axes[j]
                abl_n = int(counts.loc[counts.animal.eq(animal) & counts.plot_ABL.eq(abl), "window_trials"].sum())
                for qi, q in enumerate(PLOT_QUANTILES):
                    values = [quantiles[(animal, abl, i)]["coarse"][qi] for i in ABS_ILDS]
                    ax.plot(ABS_ILDS, values, color=QUANTILE_COLORS[qi], marker=QUANTILE_MARKERS[qi], markersize=3, linewidth=1)
                ax.set_title(f"RT quantiles | ABL {abl}\nn={abl_n:,} RTs")
                ax.set_ylabel("RT quantile (s)")
            overlay_ax = q_axes[3]
            for abl in ABLS:
                for qi, q in enumerate(PLOT_QUANTILES):
                    curve = animal_scaled.loc[animal_scaled.plot_ABL.eq(abl) & animal_scaled['quantile'].eq(q)].sort_values("abs_ILD")
                    overlay_ax.plot(curve.abs_ILD, curve.scaled_rt_s, color=COLORS[abl], marker=QUANTILE_MARKERS[qi], markersize=3, linewidth=1)
            overlay_ax.set(title="Rescaled quantiles\nABL 60 is the reference", ylabel="Scaled RT quantile (s)")
            for ax in q_axes:
                ax.set(xscale="log", xlabel="|ILD| (dB)", xlim=(.8, 20), ylim=(0, q_ymax))
                ax.set_xticks(ABS_ILDS, [str(i) for i in ABS_ILDS])
                ax.xaxis.set_minor_locator(plt.NullLocator())
            for ax in upper_axes + new_axes:
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(alpha=.16)
                ax.tick_params(labelsize=8)
            new_n = int(counts.loc[counts.animal.eq(animal), "window_trials"].sum())
            fig.text(.043, (block_top - 6.54)/height,
                f"{animal}: new row uses {new_n:,} RTs in [0,1] s | cutoff = first 10-bin forward-minus-backward ΔCDF rise ≥ 0.005 | scaling fitted up to 0.3 s",
                fontsize=8)
            for kind, axes in [("original", upper_axes), ("new", new_axes)]:
                for col, ax in enumerate(axes):
                    b = ax.get_position()
                    layout_records.append({"animal": animal, "pdf_page": int(page_number), "animal_row": row + 1,
                        "row_kind": kind, "column": col + 1, "x_in": b.x0*PAGE_WIDTH, "y_in": b.y0*height,
                        "width_in": b.width*PAGE_WIDTH, "height_in": b.height*height,
                        "xlim_low": ax.get_xlim()[0], "xlim_high": ax.get_xlim()[1],
                        "ylim_low": ax.get_ylim()[0], "ylim_high": ax.get_ylim()[1]})
            # Physical coordinates allow the verifier to find every cutoff line in the PDF.
            b = delta_ax.get_position()
            for ild in ABS_ILDS:
                t = threshold_lookup.loc[(animal, ild), "cutoff_s"]
                plotted_records.append({"animal": animal, "abs_ILD": ild, "pdf_page": int(page_number),
                    "cutoff_s": t, "color": ILD_COLORS[ild],
                    "pdf_x_pt": (b.x0 + t/.15*b.width)*PAGE_WIDTH*72 if np.isfinite(t) else np.nan,
                    "pdf_y_top_pt": (1-b.y1)*height*72, "pdf_y_bottom_pt": (1-b.y0)*height*72})
        fig.text(.5, .26/height, "Session 1, training 16, success in {1,−1}; prepared repeat rule already applied | Upper RTD: 20 ms bins, all finite RTs, 0–0.6 s display | Lower CDF/quantiles: [0,1] s",
                 ha="center", fontsize=8)
        pdf.savefig(fig)
        png = OUTPUT / f"ASD_all_animals_cdf_quantile_scaling_page_{int(page_number):02d}.png"
        fig.savefig(png, dpi=PNG_DPI, facecolor="white")
        plt.close(fig)
        page_records.append({"page": int(page_number), "animals": batch.animal.to_list(), "rows": len(batch),
            "page_width_in": PAGE_WIDTH, "page_height_in": height, "png": str(png.relative_to(REPO))})
        print(f"Page {page_number}/11: {', '.join(batch.animal)}", flush=True)
pd.DataFrame(layout_records).to_csv(OUTPUT / "panel_layout.csv", index=False)
pd.DataFrame(plotted_records).to_csv(OUTPUT / "plotted_thresholds.csv", index=False)


# %% Keep original bookmarks and verify earlier files were not modified
with fitz.open(REPO / old_manifest["pdf"]) as old_pdf:
    toc = old_pdf.get_toc()
with fitz.open(pdf_path) as document:
    document.set_toc(toc)
    temp = pdf_path.with_suffix(".bookmarks.pdf")
    document.save(temp, garbage=4, deflate=True)
temp.replace(pdf_path)
for relative, digest in protected_hashes.items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, relative
manifest = {"status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "input": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256,
    "reference_output": str(REFERENCE.relative_to(REPO)), "copied_reference_audits": COPIED_AUDITS,
    "observed_original": old_manifest["observed"], "new_window_trials": len(window), "animals": 53,
    "new_rt_field": "timed_rt", "new_rt_window_s": list(NEW_RT_WINDOW), "repeat_filter_reapplied": False,
    "cdf_conditions": len(cdfs), "threshold_conditions": len(thresholds), "available_thresholds": int(thresholds.threshold_status.eq('ok').sum()),
    "cutoff_range_ms": [float(thresholds.cutoff_ms.min()), float(thresholds.cutoff_ms.max())],
    "scaling_fits": len(scales), "valid_scaling_fits": int(scales.fit_status.eq('ok').sum()),
    "scale_factor_range": [float(scales.scale_factor.min()), float(scales.scale_factor.max())],
    "cdf_bin_width_s": CDF_BIN_WIDTH, "cdf_display_xlim_s": list(CDF_DISPLAY_XLIM),
    "onset_window_bins": ONSET_WINDOW_BINS, "onset_rise": ONSET_RISE,
    "onset_rule": "First i with mean(D[i+1:i+11]) - mean(D[i-10:i]) >= 0.005; time is bin center",
    "delta_cdf": "(|F20-F40| + |F40-F60| + |F20-F60|) / 3",
    "pooling": "Actual signed-ILD raw trials pooled across sessions within animal/ABL/abs_ILD",
    "display_quantiles": PLOT_QUANTILES.tolist(), "fine_quantiles": FIT_QUANTILES.tolist(), "quantile_method": "linear",
    "fit_max_rt_s": FIT_MAX_RT, "scaling_rule": "x=Q60-t0; y=Qabl-Q60; m=sum(x*(y-yfirst))/sum(x*x); factor=1+m; transform Q>t0 to t0+(Q-t0)/factor",
    "unavailable_policy": "No fallback cutoff; invalid/nonpositive factor or missing cutoff omits affected scaled values; all raw plots remain",
    "raw_quantile_points": len(raw_quantiles), "scaled_quantile_points": len(scaled),
    "ABL_colors": {str(a): to_hex(c) for a, c in COLORS.items()}, "ILD_colors": {str(i): c for i, c in ILD_COLORS.items()},
    "quantile_colors": QUANTILE_COLORS, "quantile_markers": QUANTILE_MARKERS,
    "animals_per_page": ANIMALS_PER_PAGE, "pdf_pages": 11, "page_width_in": PAGE_WIDTH, "png_dpi": PNG_DPI,
    "animal_order": members.animal.to_list(), "pages": page_records, "pdf": str(pdf_path.relative_to(REPO)),
    "protected_file_sha256": protected_hashes,
    "checks": ["source CSV and previous exports unchanged", "existing six reference audits copied byte-for-byte",
        "466978 new-window RTs; raw positive/negative ILD trials combined", "795 histograms/CDFs; 265 fresh per-animal thresholds",
        "530 fits with original Figure 1 shifted-difference calculation", "five quantiles per condition and unchanged ABL60 reference",
        "existing all-finite RTD/chronometric and psychometric calculations reused", "53 animals, same 11 pages and bookmarks"]}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Saved extended PDF: {pdf_path}", flush=True)
