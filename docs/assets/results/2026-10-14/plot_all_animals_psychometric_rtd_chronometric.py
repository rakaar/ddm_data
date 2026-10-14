# %% Imports
"""Five animals per page, with psychometric, RTD, and chronometric columns.

Run .venv/bin/python asd_analysis/plot_all_animals_psychometric_rtd_chronometric.py.
Reuse the verified two-column psychometric/RTD audits without modifying them.
Calculate chronometrics from combined signed-ILD trials: mean +/- sample SEM,
using all finite valid-trial timed_rt values and no measured-RT cutoff.
"""
import hashlib
import json
import shutil
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_hex
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.special import expit


# %% Parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
REFERENCE = REPO / "asd_analysis/session_type_1_all_animals_psychometric_rtd"
RT_MEAN_REFERENCE = REPO / "asd_analysis/session_type_1_animal_rt_summary/rt_statistics.csv"
OUTPUT = REPO / "asd_analysis/session_type_1_all_animals_psychometric_rtd_chronometric"
ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
PSYCHOMETRIC_MARKERS = {20: "o", 40: "s", 60: "^"}
CHRONOMETRIC_OFFSETS = {20: -.10, 40: 0, 60: .10}
GENOTYPE_LABELS = {"wt": "WT", "het": "Heterozygous", "hom": "Homozygous"}
SAMPLE_SD_DDOF = 1
ANIMALS_PER_PAGE = 5
PAGE_WIDTH = 18.24
COLUMN_WIDTH_RATIOS = [1, 1, .28]  # Compact square chronometrics; retain other panel widths.
CHRONOMETRIC_BOX_ASPECT = 1  # Physical height / width; does not change data limits.
ROW_HEIGHT = 3.8
PNG_DPI = 160
PSYCHOMETRIC_XLIM = (-18, 18)
RT_DISPLAY_XLIM = (0, .6)
CHRONOMETRIC_YLIM = (0, .45)
BIN_WIDTH = .020
PDF_NAME = "ASD_session_type_1_all_animals_psychometric_rtd_chronometric.pdf"
COPIED_AUDITS = ["animal_pages.csv", "psychometric_points.csv", "psychometric_fits.csv",
                 "rtd_histogram_bins.csv", "abl_counts.csv"]
OUTPUT.mkdir(parents=True, exist_ok=True)
old_manifest = json.loads((REFERENCE / "validation.json").read_text())
assert old_manifest["status"] == "passed"
protected = [INPUT, RT_MEAN_REFERENCE]
protected.extend(sorted(p for p in REFERENCE.iterdir() if p.is_file()))
# Preserve the previously published two-column PDF and PNG pages as well.
old_published = REPO / "docs/assets/results/2026-10-14"
protected.extend(old_published / Path(p["png"]).name for p in old_manifest["pages"])
protected.append(old_published / Path(old_manifest["pdf"]).name)
protected_hashes = {}
for path in protected:
    with path.open("rb") as stream:
        protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_sha256 = protected_hashes[str(INPUT.relative_to(REPO))]
assert input_sha256 == old_manifest["input_sha256"]
input_stat = (INPUT.stat().st_size, INPUT.stat().st_mtime_ns)
for name in COPIED_AUDITS:
    shutil.copy2(REFERENCE / name, OUTPUT / name)
    assert hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() == protected_hashes[str((REFERENCE / name).relative_to(REPO))]
members = pd.read_csv(OUTPUT / "animal_pages.csv")
psy_points = pd.read_csv(OUTPUT / "psychometric_points.csv")
fits = pd.read_csv(OUTPUT / "psychometric_fits.csv")
histograms = pd.read_csv(OUTPUT / "rtd_histogram_bins.csv")
abl_counts = pd.read_csv(OUTPUT / "abl_counts.csv")
assert len(members) == 53 and len(psy_points) == 1590 and len(fits) == len(abl_counts) == 159
assert members.groupby("pdf_page").size().to_list() == [5] * 10 + [3]
fit_lookup = fits.set_index(["animal", "plot_ABL"])


# %% Read the same valid-choice population; repeat eligibility is already applied
data = pd.read_csv(INPUT, usecols=["filename", "animal", "genotype", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "ABL", "ILD", "success", "response_poke", "timed_rt"], low_memory=False)
assert len(data) == 642370 and data.animal.nunique() == 53
assert data.session_type.eq(1).all() and data.training_level.eq(16).all()
assert data.trial_is_repeat.eq(False).all() and data.is_short_sound.eq(0).all()
data["plot_ABL"] = data.ABL.replace({59: 60})
selected = data.loc[data.success.isin([1, -1]) & data.plot_ABL.isin(ABLS) & data.ILD.isin(SIGNED_ILDS)].copy()
selected["plot_ABL"] = selected.plot_ABL.astype(int)
selected["abs_ILD"] = selected.ILD.abs().astype(int)
selected["finite_rt"] = np.isfinite(selected.timed_rt)
selected["positive_ild"] = selected.ILD.gt(0)
selected["positive_finite"] = selected.positive_ild & selected.finite_rt
selected["negative_finite"] = ~selected.positive_ild & selected.finite_rt
finite = selected.loc[selected.finite_rt].copy()
finite["above_1_s"] = finite.timed_rt.gt(1)
observed = {"animals": selected.animal.nunique(), "choices": len(selected),
    "right": int(selected.response_poke.eq(1).sum()), "left": int(selected.response_poke.eq(-1).sum()),
    "finite_rt": len(finite), "nonfinite_rt": len(selected) - len(finite),
    "rt_above_1": int(finite.above_1_s.sum())}
assert observed == old_manifest["observed"]
assert set(finite.animal) == set(members.animal)
print("Confirmed population:", json.dumps(observed), flush=True)


# %% Pool +ILD and -ILD raw trials before calculating each mean and SEM
keys = ["filename", "genotype", "animal", "plot_ABL", "abs_ILD"]
chronometric = selected.groupby(keys, as_index=False).agg(
    valid_choice_trials=("success", "size"), finite_rt_trials=("finite_rt", "sum"),
    positive_ild_choice_trials=("positive_ild", "sum"),
    positive_ild_rt_trials=("positive_finite", "sum"), negative_ild_rt_trials=("negative_finite", "sum"))
chronometric["negative_ild_choice_trials"] = chronometric.valid_choice_trials - chronometric.positive_ild_choice_trials
chronometric["nonfinite_rt_trials"] = chronometric.valid_choice_trials - chronometric.finite_rt_trials
rt_stats = finite.groupby(keys, as_index=False).agg(
    mean_rt_s=("timed_rt", "mean"), sample_sd_rt_s=("timed_rt", lambda values: values.std(ddof=SAMPLE_SD_DDOF)),
    rt_above_1_s=("above_1_s", "sum"))
chronometric = chronometric.merge(rt_stats, on=keys, validate="one_to_one")
chronometric["sem_rt_s"] = chronometric.sample_sd_rt_s / np.sqrt(chronometric.finite_rt_trials)
chronometric["mean_minus_sem_s"] = chronometric.mean_rt_s - chronometric.sem_rt_s
chronometric["mean_plus_sem_s"] = chronometric.mean_rt_s + chronometric.sem_rt_s
chronometric["category_position"] = chronometric.abs_ILD.map({ild: i for i, ild in enumerate(ABS_ILDS)})
chronometric["plot_x"] = chronometric.category_position + chronometric.plot_ABL.map(CHRONOMETRIC_OFFSETS)
chronometric["color"] = chronometric.plot_ABL.map({a: to_hex(c) for a, c in COLORS.items()})
assert len(chronometric) == 795 and chronometric.finite_rt_trials.min() >= 2
assert chronometric.groupby("animal").size().eq(15).all()
assert chronometric.finite_rt_trials.sum() == 469228 and chronometric.valid_choice_trials.sum() == 471416
assert chronometric.nonfinite_rt_trials.sum() == 2188 and chronometric.rt_above_1_s.sum() == 2250
assert (chronometric.positive_ild_rt_trials + chronometric.negative_ild_rt_trials).eq(chronometric.finite_rt_trials).all()
assert chronometric.mean_minus_sem_s.ge(CHRONOMETRIC_YLIM[0]).all()
assert chronometric.mean_plus_sem_s.le(CHRONOMETRIC_YLIM[1]).all()
chronometric.to_csv(OUTPUT / "chronometric_points.csv", index=False)


# %% Condition counts and weighted means reproduce the existing animal/ABL means
chronometric["weighted_rt_sum"] = chronometric.mean_rt_s * chronometric.finite_rt_trials
pooled = chronometric.groupby(["animal", "plot_ABL"])[["weighted_rt_sum", "finite_rt_trials", "valid_choice_trials"]].sum().sort_index()
pooled["mean_rt_s"] = pooled.weighted_rt_sum / pooled.finite_rt_trials
reference_means = pd.read_csv(RT_MEAN_REFERENCE).set_index(["animal", "plot_ABL"]).sort_index()
np.testing.assert_allclose(pooled.mean_rt_s, reference_means.mean_rt_s, rtol=0, atol=1e-12)
np.testing.assert_array_equal(pooled.finite_rt_trials, reference_means.finite_rt_trials)
np.testing.assert_array_equal(pooled.valid_choice_trials, reference_means.choice_trials)
print(f"Chronometrics: {len(chronometric)} points; minimum condition n={chronometric.finite_rt_trials.min()}; "
      f"maximum mean+SEM={chronometric.mean_plus_sem_s.max():.6f} s", flush=True)


# %% Reconstruct each RTD directly from its preserved histogram audit
curves = {}
for (animal, abl), bins in histograms.groupby(["animal", "plot_ABL"]):
    bins = bins.sort_values("bin_index")
    edges = np.r_[bins.bin_left_s.to_numpy(), bins.bin_right_s.iloc[-1]]
    np.testing.assert_allclose(np.diff(edges), BIN_WIDTH, rtol=0, atol=1e-12)
    curves[(animal, abl)] = (bins.density_per_s.to_numpy(), edges, int(bins.normalization_trials.iloc[0]))
assert int(histograms.bin_count.sum()) == 469228


# %% Five three-panel animal rows per page, retaining the original ordering
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
pdf_path = OUTPUT / PDF_NAME
page_records, plotted_records = [], []
smooth_ild = np.linspace(*PSYCHOMETRIC_XLIM, 600)
page_count = int(members.pdf_page.max())
with PdfPages(pdf_path, metadata={"Title": "ASD: psychometrics, full-RT densities, and chronometrics",
                                 "Subject": "53 animals; five rows per page; valid choices only"}) as pdf:
    for page_number, batch in members.groupby("pdf_page", sort=True):
        fig, axes = plt.subplots(len(batch), 3, figsize=(PAGE_WIDTH, ROW_HEIGHT * len(batch) + 1.6),
                                 gridspec_kw={"width_ratios": COLUMN_WIDTH_RATIOS}, squeeze=False)
        fig.suptitle(f"Session type 1 | Psychometric, RTD, and chronometric | Page {page_number}/{page_count}", fontsize=17, y=.985)
        fig.legend(handles=[Patch(facecolor=COLORS[a], label=f"ABL {a} dB") for a in ABLS],
                   loc="upper center", bbox_to_anchor=(.5, .965), ncol=3, frameon=False, fontsize=12)
        for row_index, member in enumerate(batch.itertuples(index=False)):
            animal = member.animal
            psy_ax, rt_ax, chrono_ax = axes[row_index]
            for abl in ABLS:
                p = psy_points.loc[psy_points.animal.eq(animal) & psy_points.plot_ABL.eq(abl)].sort_values("ILD")
                estimate = fit_lookup.loc[(animal, abl)]
                psy_ax.scatter(p.ILD, p.p_right, s=30, marker=PSYCHOMETRIC_MARKERS[abl], color=COLORS[abl], zorder=3,
                               label=f"ABL {abl} | n={int(p.trials.sum()):,}")
                if estimate.fit_status == "ok":
                    probability = estimate.lower + (estimate.upper - estimate.lower) * expit(estimate.k * (smooth_ild - estimate.x0))
                    psy_ax.plot(smooth_ild, probability, color=COLORS[abl], linewidth=1.7)
                density, edges, n = curves[(animal, abl)]
                rt_ax.stairs(density, edges, fill=False, color=COLORS[abl], linewidth=1.6,
                             label=f"ABL {abl} | n={n:,}")
                condition = chronometric.loc[chronometric.animal.eq(animal) & chronometric.plot_ABL.eq(abl)].sort_values("abs_ILD")
                x = condition.plot_x.to_numpy()
                mean = condition.mean_rt_s.to_numpy()
                sem = condition.sem_rt_s.to_numpy()
                errorbar = chrono_ax.errorbar(x, mean, yerr=sem, fmt="o", linestyle="None",
                    color=COLORS[abl], markersize=4.5, elinewidth=.9, capsize=2, capthick=.9,
                    label=f"ABL {abl} | n={n:,}", zorder=3)
                data_line, _, bar_collections = errorbar.lines
                np.testing.assert_allclose(data_line.get_xdata(), x)
                np.testing.assert_allclose(data_line.get_ydata(), mean)
                assert to_hex(data_line.get_color()) == to_hex(COLORS[abl])
                assert data_line.get_linestyle() == "None"
                expected_segments = np.array([[[xx, lo], [xx, hi]] for xx, lo, hi in
                    zip(x, condition.mean_minus_sem_s, condition.mean_plus_sem_s)])
                np.testing.assert_allclose(bar_collections[0].get_segments(), expected_segments)
            psy_ax.set(title=f"{animal} | {member.filename.removesuffix('.csv')} | {GENOTYPE_LABELS[member.genotype]}\n"
                             f"Psychometric | {int(member.choice_trials):,} correct/error choices",
                       xlabel="Signed ILD (dB)", ylabel="P(right)", xlim=PSYCHOMETRIC_XLIM, ylim=(0, 1))
            psy_ax.set_xticks([-16, -8, -4, 0, 4, 8, 16])
            rt_ax.set(title=f"{animal} | RTDs: all selected ILDs pooled\n{int(member.finite_rt_trials):,} finite RTs; {int(member.nonfinite_rt_trials):,} missing",
                      xlabel="Measured timed_rt (s)", ylabel="Probability density (s⁻¹)", xlim=RT_DISPLAY_XLIM)
            rt_ax.set_xticks(np.arange(0, .601, .1))
            visible_max = max(float(density[(edges[:-1] < RT_DISPLAY_XLIM[1]) & (edges[1:] > RT_DISPLAY_XLIM[0])].max())
                              for density, edges, _ in [curves[(animal, abl)] for abl in ABLS])
            rt_ax.set_ylim(0, visible_max * 1.12)
            chrono_ax.set(title=f"{animal} | Chronometric\nPooled ±ILD",
                          xlabel="|ILD| (dB)", ylabel="Mean RT ± SEM (s)",
                          xlim=(-.45, 4.45), ylim=CHRONOMETRIC_YLIM)
            chrono_ax.set_box_aspect(CHRONOMETRIC_BOX_ASPECT)
            chrono_ax.title.set_fontsize(9)
            chrono_ax.set_xticks(np.arange(len(ABS_ILDS)), [str(ild) for ild in ABS_ILDS])
            chrono_ax.set_yticks(np.arange(0, .451, .1))
            for ax in (psy_ax, rt_ax, chrono_ax):
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(alpha=.18)
            psy_ax.legend(loc="upper left", frameon=False, fontsize=8)
            rt_ax.legend(loc="upper right", frameon=False, fontsize=8)
            # The page legend gives ABL colors; adjacent RTDs give the same finite-RT totals.
        fig.text(.5, .031, "Training level 16 | success in {1, -1} | repeat rule already applied | ABL 59 mapped to 60 | signed ILDs: ±1/2/4/8/16 dB",
                 ha="center", fontsize=10)
        fig.text(.5, .015, "All finite RTs used | RTD: 20 ms bins; full-range area = 1; 0-0.6 s display zoom | Chronometric: pool +/-ILD raw trials; SEM = sample SD / sqrt(n); legends show ABL totals",
                 ha="center", fontsize=10)
        fig.subplots_adjust(left=.043, right=.98, bottom=.075, top=.9, wspace=.29, hspace=.68)
        panel_widths = [ax.get_position().width * PAGE_WIDTH for ax in axes[0]]
        np.testing.assert_allclose(np.array(panel_widths) / panel_widths[0], COLUMN_WIDTH_RATIOS)
        panel_heights = [ax.get_position().height * fig.get_size_inches()[1] for ax in axes[0]]
        np.testing.assert_allclose(panel_heights[2] / panel_widths[2], CHRONOMETRIC_BOX_ASPECT)
        # Record physical PDF coordinates so the independent verifier can check
        # every drawn chronometric dot and both ends of its SEM error bar.
        width_pt, height_pt = fig.get_size_inches() * 72
        for row_index, member in enumerate(batch.itertuples(index=False)):
            chrono_ax = axes[row_index, 2]
            bounds = chrono_ax.get_position()
            xlo, xhi = chrono_ax.get_xlim()
            ylo, yhi = chrono_ax.get_ylim()
            for point in chronometric.loc[chronometric.animal.eq(member.animal)].itertuples(index=False):
                x_pt = (bounds.x0 + (point.plot_x - xlo) / (xhi - xlo) * bounds.width) * width_pt
                y_values = [point.mean_rt_s, point.mean_minus_sem_s, point.mean_plus_sem_s]
                y_pt = [(1 - bounds.y0 - (y - ylo) / (yhi - ylo) * bounds.height) * height_pt for y in y_values]
                plotted_records.append({"animal": point.animal, "filename": point.filename, "genotype": point.genotype,
                    "plot_ABL": point.plot_ABL, "abs_ILD": point.abs_ILD, "pdf_page": int(page_number),
                    "row_on_page": row_index + 1, "plot_x": point.plot_x, "mean_rt_s": point.mean_rt_s,
                    "sem_rt_s": point.sem_rt_s, "lower_s": point.mean_minus_sem_s, "upper_s": point.mean_plus_sem_s,
                    "finite_rt_trials": point.finite_rt_trials, "color": point.color, "marker": "o",
                    "pdf_x_pt": float(x_pt), "pdf_mean_y_pt": float(y_pt[0]),
                    "pdf_lower_y_pt": float(y_pt[1]), "pdf_upper_y_pt": float(y_pt[2])})
        assert all(np.allclose(ax.get_xlim(), RT_DISPLAY_XLIM) for ax in axes[:, 1])
        assert all(np.allclose(ax.get_ylim(), CHRONOMETRIC_YLIM) for ax in axes[:, 2])
        pdf.savefig(fig)
        png_path = OUTPUT / f"ASD_all_animals_psychometric_rtd_chronometric_page_{int(page_number):02d}.png"
        fig.savefig(png_path, dpi=PNG_DPI, facecolor="white")
        plt.close(fig)
        page_records.append({"page": int(page_number), "animals": batch.animal.to_list(), "rows": len(batch),
                             "png": str(png_path.relative_to(REPO)), "page_width_in": PAGE_WIDTH,
                             "panel_widths_in": panel_widths,
                             "panel_heights_in": panel_heights,
                             "page_height_in": float(height_pt / 72)})
        print(f"Page {page_number}/{page_count}: {', '.join(batch.animal)}", flush=True)
plotted = pd.DataFrame(plotted_records)
assert len(plotted) == 795
plotted.to_csv(OUTPUT / "chronometric_plotted_points.csv", index=False)


# %% Copy the original animal bookmarks into the new PDF
with fitz.open(REPO / old_manifest["pdf"]) as document:
    old_toc = document.get_toc()
with fitz.open(pdf_path) as document:
    document.set_toc(old_toc)
    temporary_pdf = pdf_path.with_suffix(".bookmarks.pdf")
    document.save(temporary_pdf, garbage=4, deflate=True)
temporary_pdf.replace(pdf_path)
with fitz.open(pdf_path) as document:
    assert document.page_count == page_count == 11
    assert document.get_toc() == old_toc
    assert all(np.isclose(page.rect.width, PAGE_WIDTH * 72) for page in document)
assert input_stat == (INPUT.stat().st_size, INPUT.stat().st_mtime_ns)


# %% Save counts, pooling/normalization rules, and reference preservation checks
manifest = {"status": "passed", "input": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256,
    "source_script": str(Path(__file__).relative_to(REPO)), "reference_output": str(REFERENCE.relative_to(REPO)),
    "reference_rt_means": str(RT_MEAN_REFERENCE.relative_to(REPO)), "observed": observed,
    "selection": old_manifest["selection"], "repeat_filter_reapplied": False,
    "psychometric_points": len(psy_points), "reused_psychometric_fits": len(fits), "rtd_curves": len(abl_counts),
    "chronometric_points": len(chronometric), "chronometric_minimum_condition_n": int(chronometric.finite_rt_trials.min()),
    "chronometric_formula": "Arithmetic mean of all finite valid-trial RTs after pooling +ILD and -ILD within animal/ABL/abs_ILD; SEM = sample SD(ddof=1) / sqrt(n)",
    "chronometric_pooling": "Observed raw trial counts across signed ILDs and sessions; no average of sign-specific or session means",
    "chronometric_rt_cutoff": None, "chronometric_marker": "o", "chronometric_connecting_lines": False,
    "chronometric_x_spacing": "Equal-spaced categories at 0/1/2/3/4, labeled 1/2/4/8/16 dB",
    "abs_ilds": ABS_ILDS, "chronometric_offsets": {str(a): x for a, x in CHRONOMETRIC_OFFSETS.items()},
    "chronometric_ylim_s": list(CHRONOMETRIC_YLIM), "rtd_bin_width_s": BIN_WIDTH,
    "rtd_normalization": old_manifest["rtd_normalization"], "rtd_display_xlim_s": list(RT_DISPLAY_XLIM),
    "colors": {str(a): to_hex(c) for a, c in COLORS.items()}, "copied_reference_audits": COPIED_AUDITS,
    "protected_file_sha256": protected_hashes, "animal_order": members.animal.to_list(),
    "animals_per_page": ANIMALS_PER_PAGE, "columns": ["psychometric", "RTD", "chronometric"],
    "column_width_ratios": COLUMN_WIDTH_RATIOS,
    "chronometric_box_aspect": CHRONOMETRIC_BOX_ASPECT,
    "chronometric_legend": "Shared page ABL colors; finite-RT totals in adjacent RTD legend",
    "pdf": str(pdf_path.relative_to(REPO)), "pdf_pages": page_count, "pages": page_records,
    "checks": ["unchanged psychometric/RTD reference audits copied byte-for-byte", "all 53 animals; original page order/bookmarks",
        "795 mean/SEM points use all 469228 finite RTs", "signed trial counts sum to condition totals",
        "weighted condition means equal existing animal/ABL means", "all plotted offsets, dots and SEM segments",
        "0-0.45 s chronometric bounds cover all error bars", "11 18.24-inch-wide PDF pages; square chronometric panels at 28% of other column widths; matching PNGs",
        "input size and modification time unchanged"]}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Saved {pdf_path}", flush=True)
