# %% Imports
"""Three 53-animal plots: mean +/- SD, mean +/- SEM, median with 95% CI.

Run .venv/bin/python asd_analysis/plot_session_type_1_animal_rt_summary.py.
Pool the ten signed ILDs and sessions within each animal/ABL, retaining every
finite valid-trial timed_rt. The prepared CSV already applies the repeat rule.
Median CIs use exact binomial order statistics, without interpolation.
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
from scipy.stats import binom


# %% Parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO / "asd_analysis/session_type_1_animal_rt_summary"
REFERENCE = REPO / "asd_analysis/session_type_1_all_animals_psychometric_rtd"
SOURCE_FILES = ["SHANK3_cohort1.csv", "SHANK3_cohort2.csv", "CNTNAP2_cohort3.csv", "CNTNAP2_cohort4.csv"]
GENOTYPES = ["wt", "het", "hom"]
GENOTYPE_LABELS = {"wt": "WT", "het": "Het", "hom": "Hom"}
ABLS = [20, 40, 60]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
MARKERS = {20: "o", 40: "s", 60: "^"}
DOT_OFFSETS = {20: -.22, 40: 0, 60: .22}
CI_LEVEL = .95
FIGSIZE = (26, 8)
PNG_DPI = 180
MEDIAN_CI_REFERENCE = "https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/mediancl.htm"
METRICS = {
    "mean_sd": {"center": "mean_rt_s", "lower": "mean_sd_lower_s", "upper": "mean_sd_upper_s",
                "title": "Mean RT with sample SD", "ylabel": "Mean RT ± SD (s)",
                "footer": "SD = sample standard deviation (ddof=1); error bars describe trial spread. Mean - SD can be below zero."},
    "mean_sem": {"center": "mean_rt_s", "lower": "mean_sem_lower_s", "upper": "mean_sem_upper_s",
                 "title": "Mean RT with SEM", "ylabel": "Mean RT ± SEM (s)",
                 "footer": "SEM = sample SD / sqrt(n); bars show one standard error of the mean, with n finite RTs at that animal/ABL."},
    "median_ci95": {"center": "median_rt_s", "lower": "median_ci_lower_s", "upper": "median_ci_upper_s",
                    "title": "Median RT with 95% CI", "ylabel": "Median RT with 95% CI (s)",
                    "footer": "Median CI = exact binomial/order-statistic interval, pointwise coverage at least 95%; trial-level inference assumes independent trials."},
}
OUTPUT.mkdir(parents=True, exist_ok=True)
protected = [INPUT, REFERENCE / "abl_counts.csv", REFERENCE / "animal_pages.csv"]
protected.extend(sorted(REFERENCE.glob("*.pdf")))
protected.extend(sorted(REFERENCE.glob("*.png")))
protected_hashes = {}
for path in protected:
    with path.open("rb") as stream:
        protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_sha256 = protected_hashes[str(INPUT.relative_to(REPO))]
input_stat = (INPUT.stat().st_size, INPUT.stat().st_mtime_ns)
reference_manifest = json.loads((REFERENCE / "validation.json").read_text())
assert reference_manifest["input_sha256"] == input_sha256


# %% Keep all finite valid-trial RTs, with no reaction-time cutoff
data = pd.read_csv(INPUT, usecols=["filename", "animal", "genotype", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "ABL", "ILD", "success", "timed_rt"], low_memory=False)
assert len(data) == 642370 and data.animal.nunique() == 53
assert data.session_type.eq(1).all() and data.training_level.eq(16).all()
assert data.trial_is_repeat.eq(False).all() and data.is_short_sound.eq(0).all()
data["plot_ABL"] = data.ABL.replace({59: 60})
choices = data.loc[data.success.isin([1, -1]) & data.plot_ABL.isin(ABLS) & data.ILD.isin(SIGNED_ILDS)].copy()
choices["plot_ABL"] = choices.plot_ABL.astype(int)
finite = choices.loc[np.isfinite(choices.timed_rt)].copy()
observed = {"animals": finite.animal.nunique(), "choice_trials": len(choices),
            "finite_rt_trials": len(finite), "nonfinite_rt_trials": len(choices) - len(finite),
            "rt_above_1_s": int(finite.timed_rt.gt(1).sum())}
assert observed == {"animals": 53, "choice_trials": 471416, "finite_rt_trials": 469228,
                    "nonfinite_rt_trials": 2188, "rt_above_1_s": 2250}, observed
print("RT summary selection:", json.dumps(observed), flush=True)


# %% Sample moments and exact, non-interpolated median confidence intervals
records = []
identities = ["filename", "genotype", "animal", "plot_ABL"]
choice_counts = choices.groupby(identities).size()
for identity, rows in finite.groupby(identities):
    rt = rows.timed_rt.to_numpy()
    n = len(rt)
    mean = float(np.mean(rt))
    sd = float(np.std(rt, ddof=1))
    sem = sd / np.sqrt(n)
    median = float(np.median(rt))
    ordered = np.sort(rt)
    # With W~Binomial(n, 0.5), [X_(r), X_(n-r+1)] covers the population
    # median with probability P(r <= W <= n-r) = 1 - 2*P(W <= r-1).
    # ppf(alpha/2) selects conservative central ranks. Ranks are 1-based.
    lower_rank = int(binom.ppf((1 - CI_LEVEL) / 2, n, .5))
    upper_rank = n - lower_rank + 1
    assert 1 <= lower_rank <= upper_rank <= n
    ci_lower = float(ordered[lower_rank - 1])
    ci_upper = float(ordered[upper_rank - 1])
    coverage = float(1 - 2 * binom.cdf(lower_rank - 1, n, .5))
    assert ci_lower <= median <= ci_upper and coverage >= CI_LEVEL - 1e-12
    filename, genotype, animal, abl = identity
    record = dict(zip(identities, identity))
    record.update(choice_trials=int(choice_counts.loc[identity]), finite_rt_trials=n,
        nonfinite_rt_trials=int(choice_counts.loc[identity]) - n,
        rt_above_1_s=int((rt > 1).sum()), rt_min_s=float(rt.min()), rt_max_s=float(rt.max()),
        mean_rt_s=mean, sample_sd_rt_s=sd, sem_rt_s=float(sem), median_rt_s=median,
        mean_sd_lower_s=mean - sd, mean_sd_upper_s=mean + sd,
        mean_sem_lower_s=mean - sem, mean_sem_upper_s=mean + sem,
        median_ci_lower_s=ci_lower, median_ci_upper_s=ci_upper,
        median_ci_lower_rank=lower_rank, median_ci_upper_rank=upper_rank,
        median_ci_binomial_coverage=coverage)
    records.append(record)
statistics = pd.DataFrame(records)
assert len(statistics) == 159 and statistics.finite_rt_trials.sum() == 469228
reference_counts = pd.read_csv(REFERENCE / "abl_counts.csv")
count_columns = ["choice_trials", "finite_rt_trials", "nonfinite_rt_trials", "rt_above_1_s"]
pd.testing.assert_frame_equal(statistics.set_index(identities)[count_columns].sort_index(),
                              reference_counts.set_index(identities)[count_columns].sort_index())
statistics.to_csv(OUTPUT / "rt_statistics.csv", index=False)


# %% Animal order and finite sample counts match the example figure
members = finite.groupby(["filename", "genotype", "animal"], as_index=False).agg(finite_rt_trials=("timed_rt", "size"))
members["choice_trials"] = members.animal.map(choices.groupby("animal").size())
members["nonfinite_rt_trials"] = members.choice_trials - members.finite_rt_trials
members["cohort_order"] = members.filename.map({name: i for i, name in enumerate(SOURCE_FILES)})
members["genotype_order"] = members.genotype.map({name: i for i, name in enumerate(GENOTYPES)})
members = members.sort_values(["cohort_order", "genotype_order", "animal"]).reset_index(drop=True)
members["x_position"] = np.arange(len(members))
members = members.drop(columns=["cohort_order", "genotype_order"])
reference_members = pd.read_csv(REFERENCE / "animal_pages.csv")
assert members.animal.to_list() == reference_members.animal.to_list()
pd.testing.assert_frame_equal(members.set_index("animal")[["choice_trials", "finite_rt_trials", "nonfinite_rt_trials"]].sort_index(),
                              reference_members.set_index("animal")[["choice_trials", "finite_rt_trials", "nonfinite_rt_trials"]].sort_index())
members.to_csv(OUTPUT / "animal_membership.csv", index=False)
positions = members.set_index("animal").x_position


# %% Three scatter/error-bar images: three slightly offset ABL dots per animal
plot_records, figures = [], []
legend = [Line2D([], [], color=COLORS[a], marker=MARKERS[a], linestyle="None", markersize=6,
                 label=f"ABL {a} dB") for a in ABLS]
labels = [f"{row.animal}\n{GENOTYPE_LABELS[row.genotype]}\nn={int(row.finite_rt_trials):,}" for row in members.itertuples(index=False)]
for metric, config in METRICS.items():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.suptitle(f"Session type 1 | {config['title']} | 53 individual animals", fontsize=18, y=.985)
    fig.text(.5, .935, "469,228 finite valid-trial RTs | 159 animal/ABL estimates | 2,188 missing RTs omitted | no RT cutoff",
             ha="center", fontsize=11)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .902), ncol=3, frameon=False)
    for abl in ABLS:
        condition = statistics.loc[statistics.plot_ABL.eq(abl)].copy()
        condition["x"] = condition.animal.map(positions) + DOT_OFFSETS[abl]
        condition = condition.sort_values("x")
        x = condition.x.to_numpy()
        center = condition[config["center"]].to_numpy()
        lower = condition[config["lower"]].to_numpy()
        upper = condition[config["upper"]].to_numpy()
        error = np.vstack([center - lower, upper - center])
        container = ax.errorbar(x, center, yerr=error, fmt=MARKERS[abl], linestyle="None",
                                color=COLORS[abl], ecolor=COLORS[abl], markersize=4.5,
                                elinewidth=.85, capsize=2.2, capthick=.85, zorder=3)
        data_line, _, bar_collections = container.lines
        np.testing.assert_allclose(data_line.get_xdata(), x)
        np.testing.assert_allclose(data_line.get_ydata(), center)
        assert to_hex(data_line.get_color()) == to_hex(COLORS[abl])
        expected_segments = np.array([[[xx, lo], [xx, hi]] for xx, lo, hi in zip(x, lower, upper)])
        np.testing.assert_allclose(bar_collections[0].get_segments(), expected_segments)
        for index, row in enumerate(condition.itertuples(index=False)):
            plot_records.append({"metric": metric, "animal": row.animal, "filename": row.filename,
                "genotype": row.genotype, "plot_ABL": abl, "finite_rt_trials": row.finite_rt_trials,
                "x": float(x[index]), "center_s": float(center[index]),
                "lower_s": float(lower[index]), "upper_s": float(upper[index]),
                "color": to_hex(COLORS[abl]), "marker": MARKERS[abl]})
    ax.set_xticks(np.arange(53), labels, rotation=90, fontsize=8)
    ax.set_xlim(-.7, 52.7)
    for filename in SOURCE_FILES:
        cohort = members.index[members.filename.eq(filename)].to_numpy()
        start, stop = int(cohort[0]), int(cohort[-1])
        ax.text((start + stop) / 2, 1.025, f"{filename.removesuffix('.csv')} ({len(cohort)} animals)",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=11)
        if start:
            ax.axvline(start - .5, color=".7", linestyle=":", linewidth=.8)
    lowest = min(0, float(statistics[config["lower"]].min()))
    highest = float(statistics[config["upper"]].max())
    padding = .06 * (highest - lowest)
    ax.set_ylim(lowest - padding if lowest < 0 else 0, highest + padding)
    assert ax.get_ylim()[0] <= statistics[config["lower"]].min()
    assert ax.get_ylim()[1] >= statistics[config["upper"]].max()
    if lowest < 0:
        ax.axhline(0, color=".6", linestyle="--", linewidth=.7, zorder=1)
    ax.set_ylabel(config["ylabel"], fontsize=13)
    ax.set_xlabel("Animal, genotype, and total finite valid-trial RT count (across ABLs)", fontsize=11)
    ax.grid(axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .047, "Session type 1; training level 16; success in {1,-1}; confirmed repeat rule already applied; all 53 animals retained.\n"
             "ABL 59 mapped to 60; signed ILDs ±1/2/4/8/16 dB and sessions pooled by observed trials; all finite timed_rt values used, including >1 s.",
             ha="center", fontsize=9)
    fig.text(.5, .018, config["footer"], ha="center", fontsize=9)
    fig.subplots_adjust(left=.05, right=.99, top=.8, bottom=.29)
    stem = f"ASD_session_type_1_all_animals_rt_{metric}"
    png_path, pdf_path = OUTPUT / f"{stem}.png", OUTPUT / f"{stem}.pdf"
    fig.savefig(png_path, dpi=PNG_DPI, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    ylim = list(ax.get_ylim())
    plt.close(fig)
    with fitz.open(pdf_path) as document:
        assert document.page_count == 1
        assert all(animal in document[0].get_text() for animal in members.animal)
    figures.append({"metric": metric, "png": str(png_path.relative_to(REPO)),
                    "pdf": str(pdf_path.relative_to(REPO)), "pdf_pages": 1, "points": 159,
                    "error_bars": 159, "ylim_s": ylim})
    print(f"Saved {stem}: 159 estimates/error bars", flush=True)
pd.DataFrame(plot_records).to_csv(OUTPUT / "plotted_points.csv", index=False)


# %% Validation and interval-method provenance
assert input_stat == (INPUT.stat().st_size, INPUT.stat().st_mtime_ns)
manifest = {"status": "passed", "input": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256,
    "source_script": str(Path(__file__).relative_to(REPO)), "observed": observed,
    "selection": "Cleaned session-type-1 CSV; training level 16; success in {1,-1}; ABL 20/40/60 after 59 to 60; signed ILDs +/-1/2/4/8/16; all finite timed_rt values, no RT cutoff",
    "repeat_filter_reapplied": False, "rt_units": "seconds", "pooling": "Observed raw trials across signed ILDs and sessions within each animal/ABL",
    "sample_sd_ddof": 1, "sem_formula": "sample SD / sqrt(finite_rt_trials)",
    "median_ci_level": CI_LEVEL, "median_ci_method": "Non-interpolated binomial/order-statistic interval; r=Binomial(n,0.5).ppf(0.025); [X_(r), X_(n-r+1)] with 1-based ranks",
    "median_ci_coverage_formula": "1 - 2*Binomial(n,0.5).cdf(r-1)",
    "median_ci_coverage_min": float(statistics.median_ci_binomial_coverage.min()),
    "median_ci_coverage_max": float(statistics.median_ci_binomial_coverage.max()),
    "median_ci_assumptions": "Pointwise trial-level inference under independent sampling; no session/animal bootstrap",
    "median_ci_reference": MEDIAN_CI_REFERENCE,
    "negative_mean_minus_sd_cells": int(statistics.mean_sd_lower_s.lt(0).sum()),
    "figures": figures, "animal_order": members.animal.to_list(),
    "abl_colors": {str(a): to_hex(COLORS[a]) for a in ABLS},
    "abl_markers": {str(a): MARKERS[a] for a in ABLS}, "abl_offsets": {str(a): DOT_OFFSETS[a] for a in ABLS},
    "protected_file_sha256": protected_hashes,
    "checks": ["53 animals and 159 animal/ABL conditions", "counts agree with all-RT RTD audit",
        "means, sample SDs, SEMs and medians use all finite RTs", "median CI contains median; binomial coverage >=95%",
        "477 plotted estimates and error bars match statistic records", "all error bounds displayed without clipping",
        "3 single-page PDFs", "input size and modification time unchanged"]}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Median CI coverage:", manifest["median_ci_coverage_min"], "to", manifest["median_ci_coverage_max"], flush=True)
