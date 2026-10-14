# %% Imports
"""Five animals per PDF page: psychometrics and ILD-pooled full-RT densities.

Run .venv/bin/python asd_analysis/plot_all_animals_psychometric_and_rtd.py.
The cleaned session-type-1 CSV already applies the repeat rule. Both panels
use success in {1,-1}; psychometrics keep choices with missing RT, while RTDs
use all finite timed_rt values. A 0-0.6 s axis zoom does not change density
normalization. Reuse the verified animal fits from the slope/JND analysis.
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
from scipy.special import expit


# %% Parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO / "asd_analysis/session_type_1_all_animals_psychometric_rtd"
FIT_REFERENCE = REPO / "asd_analysis/session_type_1_slope_jnd"
PSYCHOMETRIC_REFERENCE = REPO / "asd_analysis/session_type_1_psychometrics"
SOURCE_FILES = ["SHANK3_cohort1.csv", "SHANK3_cohort2.csv", "CNTNAP2_cohort3.csv", "CNTNAP2_cohort4.csv"]
GENOTYPES = ["wt", "het", "hom"]
GENOTYPE_LABELS = {"wt": "WT", "het": "Heterozygous", "hom": "Homozygous"}
ABLS = [20, 40, 60]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
MARKERS = {20: "o", 40: "s", 60: "^"}
ANIMALS_PER_PAGE = 5
BIN_WIDTH = .020
RT_DISPLAY_XLIM = (0.0, 0.6)
PSYCHOMETRIC_XLIM = (-18, 18)
PAGE_WIDTH = 16
ROW_HEIGHT = 3.8
PNG_DPI = 160
EXPECTED = {"animals": 53, "choices": 471416, "right": 241539, "left": 229877,
            "finite_rt": 469228, "nonfinite_rt": 2188, "rt_above_1": 2250}
PDF_NAME = "ASD_session_type_1_all_animals_psychometric_and_full_rt_rtd.pdf"
OUTPUT.mkdir(parents=True, exist_ok=True)
protected = [INPUT, FIT_REFERENCE / "slope_jnd_fits.csv",
             FIT_REFERENCE / "slope_jnd_condition_counts.csv",
             PSYCHOMETRIC_REFERENCE / "psychometric_group_membership.csv"]
protected.extend(sorted((REPO / "asd_analysis/session_type_1_animal_psychometric_rtd/ASD0054").glob("*")))
protected_hashes = {}
for path in protected:
    if path.is_file():
        with path.open("rb") as stream:
            protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_sha256 = protected_hashes[str(INPUT.relative_to(REPO))]
reference_manifest = json.loads((PSYCHOMETRIC_REFERENCE / "psychometric_validation.json").read_text())
assert reference_manifest["input_sha256"] == input_sha256
input_stat = (INPUT.stat().st_size, INPUT.stat().st_mtime_ns)


# %% Same prepared choice selection, with no RT restriction
data = pd.read_csv(INPUT, usecols=["filename", "animal", "genotype", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "ABL", "ILD", "success", "response_poke", "timed_rt"], low_memory=False)
assert len(data) == 642370 and data.animal.nunique() == 53
assert data.session_type.eq(1).all() and data.training_level.eq(16).all()
assert data.trial_is_repeat.eq(False).all() and data.is_short_sound.eq(0).all()
data["plot_ABL"] = data.ABL.replace({59: 60})
selected = data.loc[data.success.isin([1, -1]) & data.plot_ABL.isin(ABLS) & data.ILD.isin(SIGNED_ILDS)].copy()
selected["plot_ABL"] = selected.plot_ABL.astype(int)
selected["right"] = selected.response_poke.eq(1)
selected["finite_rt"] = np.isfinite(selected.timed_rt)
assert selected.response_poke.isin([1, -1]).all()
observed = {"animals": selected.animal.nunique(), "choices": len(selected),
            "right": int(selected.right.sum()), "left": int((~selected.right).sum()),
            "finite_rt": int(selected.finite_rt.sum()), "nonfinite_rt": int((~selected.finite_rt).sum()),
            "rt_above_1": int((selected.finite_rt & selected.timed_rt.gt(1)).sum())}
assert observed == EXPECTED, observed
print("Selected trials:", json.dumps(observed), flush=True)


# %% Animal membership and page order: cohort, genotype, then animal ID
identities = ["filename", "genotype", "animal"]
members = selected.groupby(identities, as_index=False).agg(
    choice_trials=("response_poke", "size"), right_trials=("right", "sum"),
    finite_rt_trials=("finite_rt", "sum"))
members["left_trials"] = members.choice_trials - members.right_trials
members["nonfinite_rt_trials"] = members.choice_trials - members.finite_rt_trials
members["cohort_order"] = members.filename.map({name: i for i, name in enumerate(SOURCE_FILES)})
members["genotype_order"] = members.genotype.map({name: i for i, name in enumerate(GENOTYPES)})
assert members[["cohort_order", "genotype_order"]].notna().all().all()
members = members.sort_values(["cohort_order", "genotype_order", "animal"]).reset_index(drop=True)
members["pdf_page"] = np.arange(len(members)) // ANIMALS_PER_PAGE + 1
members["row_on_page"] = np.arange(len(members)) % ANIMALS_PER_PAGE + 1
members["prepared_animal_rows"] = members.animal.map(data.groupby("animal").size())
members = members.drop(columns=["cohort_order", "genotype_order"])
assert len(members) == 53 and members.animal.nunique() == 53
old_members = pd.read_csv(PSYCHOMETRIC_REFERENCE / "psychometric_group_membership.csv")
old_members = old_members.loc[old_members.view.eq("filename_genotype")].rename(columns={"trials": "choice_trials"})
pd.testing.assert_frame_equal(members.set_index(identities)[["choice_trials", "right_trials"]].sort_index(),
                              old_members.set_index(identities)[["choice_trials", "right_trials"]].sort_index())
members.to_csv(OUTPUT / "animal_pages.csv", index=False)


# %% Measured probability points and matching, previously verified sigmoid fits
points = selected.groupby(identities + ["plot_ABL", "ILD"], as_index=False).agg(
    trials=("response_poke", "size"), right_trials=("right", "sum"))
points["left_trials"] = points.trials - points.right_trials
points["p_right"] = points.right_trials / points.trials
points["group"] = points.filename + "/" + points.animal
reference_points = pd.read_csv(FIT_REFERENCE / "slope_jnd_condition_counts.csv")
reference_points = reference_points.loc[reference_points.view.eq("animal")].copy()
reference_points["plot_ABL"] = reference_points.plot_ABL.astype(int)
point_keys = ["group", "genotype", "plot_ABL", "ILD"]
pd.testing.assert_frame_equal(points.set_index(point_keys)[["trials", "right_trials", "left_trials"]].sort_index(),
                              reference_points.set_index(point_keys)[["trials", "right_trials", "left_trials"]].sort_index())
fits = pd.read_csv(FIT_REFERENCE / "slope_jnd_fits.csv")
fits = fits.loc[fits.view.eq("animal")].copy()
fits[["filename", "animal"]] = fits.group.str.rsplit("/", n=1, expand=True)
fits["plot_ABL"] = fits.plot_ABL.astype(int)
assert len(points) == 1590 and len(fits) == 159
assert set(fits.animal) == set(members.animal)
fit_lookup = fits.set_index(["animal", "plot_ABL"])
points["fitted_p_right"] = np.nan
for (animal, abl), indices in points.groupby(["animal", "plot_ABL"]).groups.items():
    estimate = fit_lookup.loc[(animal, abl)]
    assert points.loc[indices, "trials"].sum() == estimate.trials
    assert points.loc[indices, "right_trials"].sum() == estimate.right_trials
    if estimate.fit_status == "ok":
        x = points.loc[indices, "ILD"].to_numpy()
        points.loc[indices, "fitted_p_right"] = estimate.lower + (estimate.upper - estimate.lower) * expit(estimate.k * (x - estimate.x0))
np.testing.assert_allclose(points.set_index(point_keys).fitted_p_right.sort_index(),
                           reference_points.set_index(point_keys).fitted_p_right.sort_index(), atol=1e-12, rtol=0, equal_nan=True)
points.to_csv(OUTPUT / "psychometric_points.csv", index=False)
fits.to_csv(OUTPUT / "psychometric_fits.csv", index=False)


# %% Every finite RT contributes, pooling raw trials across signed ILDs and sessions
histogram_parts, count_records, curves = [], [], {}
for animal in members.animal:
    rows = selected.loc[selected.animal.eq(animal)]
    finite = rows.loc[rows.finite_rt, "timed_rt"].to_numpy()
    first_bin = int(min(0, np.floor(finite.min() / BIN_WIDTH)))
    last_bin = int(max(np.ceil(finite.max() / BIN_WIDTH), np.ceil(RT_DISPLAY_XLIM[1] / BIN_WIDTH)))
    edges = np.arange(first_bin, last_bin + 1) * BIN_WIDTH
    np.testing.assert_allclose(np.diff(edges), BIN_WIDTH, rtol=0, atol=1e-12)
    for abl in ABLS:
        condition = rows.loc[rows.plot_ABL.eq(abl)]
        rt = condition.loc[condition.finite_rt, "timed_rt"].to_numpy()
        counts, _ = np.histogram(rt, bins=edges)
        n = len(rt)
        assert n > 0 and counts.sum() == n
        density = counts / (n * BIN_WIDTH)
        area = float(np.dot(density, np.diff(edges)))
        assert np.isclose(area, 1, rtol=0, atol=1e-12)
        curves[(animal, abl)] = (density, edges, n)
        identity = {"animal": animal, "filename": condition.filename.iloc[0],
                    "genotype": condition.genotype.iloc[0], "plot_ABL": abl}
        count_records.append({**identity, "choice_trials": len(condition), "finite_rt_trials": n,
            "nonfinite_rt_trials": int((~condition.finite_rt).sum()),
            "rt_above_1_s": int((rt > 1).sum()),
            "rt_in_display": int(((rt >= RT_DISPLAY_XLIM[0]) & (rt <= RT_DISPLAY_XLIM[1])).sum()),
            "rt_min_s": float(rt.min()), "rt_max_s": float(rt.max()),
            "histogram_trials": int(counts.sum()), "full_density_area": area, "color": to_hex(COLORS[abl])})
        histogram_parts.append(pd.DataFrame({**identity, "bin_index": np.arange(len(counts)),
            "bin_left_s": edges[:-1], "bin_right_s": edges[1:], "bin_count": counts,
            "normalization_trials": n, "density_per_s": density}))
histograms = pd.concat(histogram_parts, ignore_index=True)
counts = pd.DataFrame(count_records)
assert len(counts) == 159 and counts.finite_rt_trials.sum() == 469228
assert counts.nonfinite_rt_trials.sum() == 2188 and counts.rt_above_1_s.sum() == 2250
assert histograms.bin_count.sum() == 469228
histograms.to_csv(OUTPUT / "rtd_histogram_bins.csv", index=False)
counts.to_csv(OUTPUT / "abl_counts.csv", index=False)


# %% Five two-panel animal rows per page, with per-ABL sample counts
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
pdf_path = OUTPUT / PDF_NAME
page_records = []
smooth_ild = np.linspace(*PSYCHOMETRIC_XLIM, 600)
page_count = int(members.pdf_page.max())
with PdfPages(pdf_path, metadata={"Title": "ASD: individual-animal psychometrics and full-RT densities",
                                 "Subject": "Session type 1, valid choices, 53 animals, five animals per page"}) as pdf:
    for page_number, batch in members.groupby("pdf_page", sort=True):
        fig, axes = plt.subplots(len(batch), 2, figsize=(PAGE_WIDTH, ROW_HEIGHT * len(batch) + 1.6), squeeze=False)
        fig.suptitle(f"Session type 1 | Individual-animal psychometrics and RTDs | Page {page_number}/{page_count}", fontsize=16, y=.985)
        fig.legend(handles=[Line2D([], [], marker=MARKERS[a], color=COLORS[a], label=f"ABL {a} dB") for a in ABLS],
                   loc="upper center", bbox_to_anchor=(.5, .965), ncol=3, frameon=False, fontsize=12)
        for row_index, member in enumerate(batch.itertuples(index=False)):
            animal = member.animal
            left, right = axes[row_index]
            for abl in ABLS:
                p = points.loc[points.animal.eq(animal) & points.plot_ABL.eq(abl)].sort_values("ILD")
                estimate = fit_lookup.loc[(animal, abl)]
                left.scatter(p.ILD, p.p_right, s=30, marker=MARKERS[abl], color=COLORS[abl], zorder=3,
                             label=f"ABL {abl} | n={int(p.trials.sum()):,}")
                if estimate.fit_status == "ok":
                    probability = estimate.lower + (estimate.upper - estimate.lower) * expit(estimate.k * (smooth_ild - estimate.x0))
                    left.plot(smooth_ild, probability, color=COLORS[abl], linewidth=1.7)
                density, edges, n = curves[(animal, abl)]
                right.stairs(density, edges, fill=False, color=COLORS[abl], linewidth=1.6,
                             label=f"ABL {abl} | n={n:,}")
            left.set(title=f"{animal} | {member.filename.removesuffix('.csv')} | {GENOTYPE_LABELS[member.genotype]}\n"
                           f"Psychometric | {int(member.choice_trials):,} correct/error choices",
                     xlabel="Signed ILD (dB)", ylabel="P(right)", xlim=PSYCHOMETRIC_XLIM, ylim=(0, 1))
            left.set_xticks([-16, -8, -4, 0, 4, 8, 16])
            right.set(title=f"{animal} | RTDs: all selected ILDs pooled\n{int(member.finite_rt_trials):,} finite RTs; {int(member.nonfinite_rt_trials):,} missing",
                      xlabel="Measured timed_rt (s)", ylabel="Probability density (s⁻¹)", xlim=RT_DISPLAY_XLIM)
            right.set_xticks(np.arange(0, .601, .1))
            visible_max = max(float(density[(edges[:-1] < RT_DISPLAY_XLIM[1]) & (edges[1:] > RT_DISPLAY_XLIM[0])].max())
                              for density, edges, _ in [curves[(animal, abl)] for abl in ABLS])
            right.set_ylim(0, visible_max * 1.12)
            for ax in (left, right):
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(alpha=.18)
            left.legend(loc="upper left", frameon=False, fontsize=8)
            right.legend(loc="upper right", frameon=False, fontsize=8)
            unavailable = int(fits.loc[fits.animal.eq(animal), "fit_status"].ne("ok").sum())
            if unavailable:
                left.text(.98, .03, f"Unavailable fits: {unavailable}; measured points retained", transform=left.transAxes,
                          ha="right", fontsize=8)
        fig.text(.5, .031, "Training level 16 | success in {1, -1} | repeat rule already applied | ABL 59 mapped to 60 | signed ILDs: ±1/2/4/8/16 dB",
                 ha="center", fontsize=10)
        fig.text(.5, .015, "Psychometric: no RT restriction | RTD: all finite RTs; 20 ms bins; full-range area = 1 per ABL; 0-0.6 s is a display zoom only",
                 ha="center", fontsize=10)
        fig.subplots_adjust(left=.065, right=.98, bottom=.075, top=.9, wspace=.27, hspace=.68)
        assert all(np.allclose(ax.get_xlim(), RT_DISPLAY_XLIM) for ax in axes[:, 1])
        pdf.savefig(fig)
        png_path = OUTPUT / f"ASD_all_animals_psychometric_full_rt_rtd_page_{int(page_number):02d}.png"
        fig.savefig(png_path, dpi=PNG_DPI, facecolor="white")
        plt.close(fig)
        page_records.append({"page": int(page_number), "animals": batch.animal.to_list(), "rows": len(batch),
                             "png": str(png_path.relative_to(REPO))})
        print(f"Page {page_number}/{page_count}: {', '.join(batch.animal)}", flush=True)


# %% PDF bookmarks make individual animals easy to find within shared pages
with fitz.open(pdf_path) as document:
    toc = []
    previous_cohort = previous_genotype = None
    for member in members.itertuples(index=False):
        if member.filename != previous_cohort:
            toc.append([1, member.filename.removesuffix(".csv"), int(member.pdf_page)])
            previous_cohort, previous_genotype = member.filename, None
        if member.genotype != previous_genotype:
            toc.append([2, GENOTYPE_LABELS[member.genotype], int(member.pdf_page)])
            previous_genotype = member.genotype
        toc.append([3, member.animal, int(member.pdf_page)])
    document.set_toc(toc)
    temp_pdf = pdf_path.with_suffix(".bookmarks.pdf")
    document.save(temp_pdf, garbage=4, deflate=True)
temp_pdf.replace(pdf_path)
with fitz.open(pdf_path) as document:
    assert document.page_count == page_count == 11
    for record in page_records:
        text = document[record["page"] - 1].get_text()
        assert all(animal in text for animal in record["animals"])
assert input_stat == (INPUT.stat().st_size, INPUT.stat().st_mtime_ns)


# %% Save the selection, fit provenance, histograms, and page index
manifest = {"status": "passed", "observed": observed, "input": str(INPUT.relative_to(REPO)),
    "input_sha256": input_sha256, "source_script": str(Path(__file__).relative_to(REPO)),
    "selection": "Cleaned session-type-1 CSV; training level 16; success in {1,-1}; ABL 20/40/60 after 59 to 60; signed ILDs +/-1/2/4/8/16; no RT restriction",
    "repeat_filter_reapplied": False, "psychometric_missing_rt_choices_retained": True,
    "sigmoid": "lower + (upper - lower) / (1 + exp(-k * (ILD - x0)))",
    "fit_source": str((FIT_REFERENCE / "slope_jnd_fits.csv").relative_to(REPO)),
    "fit_procedure": "Previously verified unweighted curve_fit on each animal/ABL's ten measured ILD probabilities; [upper,lower,x0,k]; initial [max(P),min(P),median(ILD),0.1]; upper/lower bounded 0-1, x0 unrestricted, k nonnegative; maxfev=10000",
    "fit_counts_and_points_match_reference": True, "fit_status_counts": fits.fit_status.value_counts().to_dict(),
    "psychometric_points": len(points), "rtd_curves": len(counts), "rtd_bin_width_s": BIN_WIDTH,
    "rtd_upper_rt_cutoff": None, "rtd_normalization": "bin_count / (all finite correct/error RTs at this animal and ABL * 0.020 s)",
    "rtd_display_xlim_s": list(RT_DISPLAY_XLIM), "abls": ABLS, "signed_ilds": SIGNED_ILDS,
    "colors": {str(a): to_hex(COLORS[a]) for a in ABLS}, "animals_per_page": ANIMALS_PER_PAGE,
    "ordering": "SHANK3 cohorts 1/2, CNTNAP2 cohorts 3/4; WT, heterozygous, homozygous; animal ID",
    "pdf": str(pdf_path.relative_to(REPO)), "pdf_pages": page_count, "pages": page_records,
    "protected_file_sha256": protected_hashes,
    "checks": ["all 53 animals", "existing animal membership/choice counts", "all 1590 points match fit audit",
               "159 reused fits match observed conditions", "every finite RT histogrammed", "20 ms bins",
               "159 full-range unit density areas", "display zoom is 0-0.6 s", "five animals per page; three on final page",
               "11 PDF pages with animal bookmarks", "input size and modification time unchanged"]}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Saved {pdf_path}", flush=True)
