# %% Imports
"""Figure 1-style animal JND/k values and signed deviations from animal means.

Run with .venv/bin/python from the repository root. Reuse saved October 13
fits without refitting or restricting measured RT. Eligibility is independent:
50 complete-JND animals and 53 complete-slope animals.
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


# %% Editable parameters
REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "asd_analysis/session_type_1_slope_jnd"
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO / "asd_analysis/session_type_1_animal_metric_deviations"
ABLS = [20, 40, 60]
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
GENOTYPE_LABELS = {"wt": "WT", "het": "HET", "hom": "HOM"}
FIGSIZE = (26, 10)
PNG_DPI = 180
DOT_ALPHA = .65
POPULATION_SD_DDOF = 0
METRICS = {
    "jnd": {"source": "jnd", "title": "JND", "units": "dB", "animals": 50,
            "trials": 444718, "raw_tick": .5, "delta_tick": .2, "filename": "jnd_and_delta_jnd"},
    "k": {"source": "slope_k", "title": "Slope = k", "units": "dB⁻¹", "animals": 53,
          "trials": 471416, "raw_tick": .05, "delta_tick": .02, "filename": "slope_and_delta_slope"},
}
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Preserve earlier artifacts and verify the saved fits describe the current CSV
reference_manifest = json.loads((REFERENCE / "slope_jnd_validation.json").read_text())
assert reference_manifest["status"] == "passed"
assert json.loads((REFERENCE / "slope_jnd_independent_validation.json").read_text())["status"] == "passed"
protected = set(REPO.glob("SHANK3_cohort*.csv")) | set(REPO.glob("CNTNAP2_cohort*.csv"))
protected.update(INPUT.parent.glob("*.csv"))
for folder in (REPO / "asd_analysis").glob("session_type_1_*"):
    if folder.is_dir() and folder != OUTPUT:
        protected.update(p for p in folder.rglob("*") if p.is_file())
for date in ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]:
    protected.update(p for p in (REPO / f"docs/assets/results/{date}").iterdir() if p.is_file())
protected_hashes = {}
for path in sorted(protected):
    with path.open("rb") as stream:
        protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
assert protected_hashes[str(INPUT.relative_to(REPO))] == reference_manifest["input_sha256"]
fits = pd.read_csv(REFERENCE / "slope_jnd_fits.csv")
fits = fits.loc[fits.view.eq("animal")].copy()
assert len(fits) == 159 and fits.fit_status.eq("ok").all()
assert fits.groupby("group").size().eq(3).all()
np.testing.assert_array_equal(fits.k, fits.slope_k)
fits[["filename", "animal"]] = fits.group.str.split("/", expand=True)
assert fits.animal.nunique() == 53
print("Reusing 159 animal/ABL fits with no RT restriction", flush=True)


# %% Audit all animals; require a complete three-ABL set separately for each metric
eligibility = fits.groupby(["animal", "filename", "genotype"], as_index=False).agg(
    choice_trials=("trials", "sum"), right_trials=("right_trials", "sum"), left_trials=("left_trials", "sum"))
for metric, config in METRICS.items():
    wide = fits.pivot(index="animal", columns="plot_ABL", values=config["source"])
    wide.columns = [f"{metric}_ABL_{a}" for a in wide.columns]
    eligibility = eligibility.merge(wide.reset_index(), on="animal", validate="one_to_one")
    columns = [f"{metric}_ABL_{a}" for a in ABLS]
    eligibility[f"{metric}_eligible"] = np.isfinite(eligibility[columns]).all(axis=1) & eligibility[columns].gt(0).all(axis=1)
    eligibility[f"{metric}_unavailable_ABLs"] = eligibility.apply(
        lambda row: ";".join(str(a) for a in ABLS if not np.isfinite(row[f"{metric}_ABL_{a}"])), axis=1)
    assert eligibility[f"{metric}_eligible"].sum() == config["animals"]
assert set(eligibility.loc[~eligibility.jnd_eligible, "animal"]) == {"ASD0035", "ASD0044", "ASD0048"}
eligibility.sort_values("animal").to_csv(OUTPUT / "animal_eligibility.csv", index=False)

means_tables, point_tables, summary_records = [], [], []
for metric, config in METRICS.items():
    selected_animals = eligibility.loc[eligibility[f"{metric}_eligible"], "animal"]
    values = fits.loc[fits.animal.isin(selected_animals)].copy()
    values["value"] = values[config["source"]]
    means = values.groupby(["animal", "filename", "genotype"], as_index=False).agg(
        animal_mean=("value", "mean"), choice_trials=("trials", "sum"),
        right_trials=("right_trials", "sum"), left_trials=("left_trials", "sum"))
    means = means.sort_values(["animal_mean", "animal"], kind="stable").reset_index(drop=True)
    means.insert(0, "metric", metric)
    means["position"] = np.arange(len(means))
    means["order_rank"] = means.position + 1
    means["units"] = config["units"]
    values = values.merge(means[["animal", "animal_mean", "position", "order_rank"]], on="animal", validate="many_to_one")
    values["signed_delta"] = values.value - values.animal_mean
    values["metric"] = metric
    values["units"] = config["units"]
    values = values[["metric", "animal", "filename", "genotype", "position", "order_rank", "plot_ABL",
        "value", "animal_mean", "signed_delta", "units", "trials", "right_trials", "left_trials"]].sort_values(["position", "plot_ABL"])
    assert len(values) == 3 * config["animals"] and means.choice_trials.sum() == config["trials"]
    np.testing.assert_allclose(values.groupby("animal").signed_delta.sum(), 0, rtol=0, atol=2e-14)
    for level, data in [("animal_means", means.animal_mean), ("ABL_deviations", values.signed_delta)]:
        center, sd = float(data.mean()), float(data.std(ddof=POPULATION_SD_DDOF))
        summary_records.append({"metric": metric, "level": level, "n": len(data), "mean": center,
            "population_sd": sd, "ddof": POPULATION_SD_DDOF, "lower": center - sd, "upper": center + sd,
            "units": config["units"]})
    means_tables.append(means)
    point_tables.append(values)
means_all = pd.concat(means_tables, ignore_index=True)
points = pd.concat(point_tables, ignore_index=True)
summaries = pd.DataFrame(summary_records)
assert len(means_all) == 103 and len(points) == 309 and len(summaries) == 4
means_all.to_csv(OUTPUT / "animal_means_order.csv", index=False)
points.to_csv(OUTPUT / "animal_abl_deviations.csv", index=False)
summaries.to_csv(OUTPUT / "summary_statistics.csv", index=False)


# %% Two figures: values above, signed deviations below; SD bars at the right
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12})
figure_records, drawn_points, drawn_bars = [], [], []
for metric, config in METRICS.items():
    means = means_all.loc[means_all.metric.eq(metric)]
    values = points.loc[points.metric.eq(metric)]
    summary = summaries.loc[summaries.metric.eq(metric)].set_index("level")
    grand_mean = summary.loc["animal_means", "mean"]
    fig = plt.figure(figsize=FIGSIZE)
    grid = fig.add_gridspec(2, 2, width_ratios=[1, .05], hspace=.29, wspace=.025,
                           left=.055, right=.975, top=.85, bottom=.255)
    upper = fig.add_subplot(grid[0, 0])
    lower = fig.add_subplot(grid[1, 0], sharex=upper)
    upper_sd = fig.add_subplot(grid[0, 1], sharey=upper)
    lower_sd = fig.add_subplot(grid[1, 1], sharey=lower)
    fig.suptitle(f"Session type 1 | {config['title']} and signed deviation from each animal's mean", fontsize=18, y=.985)
    fig.text(.5, .947, f"{config['animals']} eligible animals | {config['trials']:,} valid choices | {len(values)} ABL values in each panel | sorted by animal mean",
             ha="center", fontsize=12)
    handles = [Line2D([], [], marker="o", color=COLORS[a], linestyle="None", label=f"ABL {a} dB") for a in ABLS]
    handles += [Line2D([], [], marker="_", color="black", linestyle="None", markersize=10, label="Animal mean (three ABLs)"),
                Line2D([], [], color="black", linestyle=":", label="Grand mean (equal animals)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .925), ncol=5, frameon=False)
    for abl in ABLS:
        condition = values.loc[values.plot_ABL.eq(abl)]
        upper.scatter(condition.position, condition.value, color=COLORS[abl], s=27, marker="o", alpha=DOT_ALPHA, zorder=3)
        lower.scatter(condition.position, condition.signed_delta, color=COLORS[abl], s=27, marker="o", alpha=DOT_ALPHA, zorder=3)
    upper.scatter(means.position, means.animal_mean, marker="_", color="black", s=95, linewidths=1.5, zorder=5)
    upper.axhline(grand_mean, color="black", linestyle=":", linewidth=1)
    lower.axhline(0, color="black", linewidth=1)
    upper.set_title(f"{config['title']} at each ABL", loc="left")
    lower.set_title("Signed deviation: value at this ABL − animal mean", loc="left")
    upper.set_ylabel(f"{'JND' if metric == 'jnd' else 'k'} ({config['units']})")
    lower.set_ylabel(f"{'JND' if metric == 'jnd' else 'k'} − animal mean ({config['units']})")
    high = max(float(values.value.max()), float(summary.loc["animal_means", "upper"]))
    upper.set_ylim(0, np.ceil(high * 1.04 / config["raw_tick"]) * config["raw_tick"])
    extent = max(float(values.signed_delta.abs().max()), abs(float(summary.loc["ABL_deviations", "lower"])),
                 abs(float(summary.loc["ABL_deviations", "upper"])))
    extent = np.ceil(extent * 1.06 / config["delta_tick"]) * config["delta_tick"]
    lower.set_ylim(-extent, extent)
    labels = [f"{r.animal}\n{r.filename.removesuffix('.csv')}\n{GENOTYPE_LABELS[r.genotype]}" for r in means.itertuples()]
    lower.set_xticks(means.position, labels, rotation=90, fontsize=7)
    lower.set_xlabel("Animal | cohort | genotype (same order in both panels)")
    upper.tick_params(axis="x", bottom=False, labelbottom=False)
    for ax in [upper, lower]:
        ax.set_xlim(-.8, len(means) - .2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    for panel, ax, level in [("raw", upper_sd, "animal_means"), ("delta", lower_sd, "ABL_deviations")]:
        row = summary.loc[level]
        ax.plot([.5, .5], [row.lower, row.upper], color="gray", linewidth=4, solid_capstyle="butt")
        ax.set_xlim(0, 1)
        ax.axis("off")
        ax.set_title(f"Mean ±1 SD\nσ = {row.population_sd:.4f}", fontsize=9)
        box = ax.get_position()
        ylim = ax.get_ylim()
        drawn_bars.append({"metric": metric, "panel": panel, "mean": row['mean'], "sd": row.population_sd,
            "lower": row.lower, "upper": row.upper, "pdf_x_pt": (box.x0 + .5 * box.width) * FIGSIZE[0] * 72,
            "pdf_y_top_pt": (1 - box.y0 - (row.upper - ylim[0])/(ylim[1] - ylim[0]) * box.height) * FIGSIZE[1] * 72,
            "pdf_y_bottom_pt": (1 - box.y0 - (row.lower - ylim[0])/(ylim[1] - ylim[0]) * box.height) * FIGSIZE[1] * 72})
    fig.text(.5, .075, "Animal mean = (M20 + M40 + M60) / 3 | Signed delta = M_ABL − animal mean | Gray side bars: mean ±1 population SD (ddof=0)",
             ha="center", fontsize=10)
    fig.text(.5, .044, "Valid choices: success in {1,−1}; training 16; no RT restriction; prepared repeat rule already applied; ABL 59 → 60; signed ILDs ±1/2/4/8/16",
             ha="center", fontsize=10)
    fig.text(.5, .016, "Each animal and each ABL has equal weight in these summaries. Signed deviations here differ from the earlier mean absolute pairwise ranking scores.",
             ha="center", fontsize=10)
    for panel, ax, field in [("raw", upper, "value"), ("delta", lower, "signed_delta")]:
        b = ax.get_position()
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        for r in values.itertuples(index=False):
            value = getattr(r, field)
            drawn_points.append({"metric": metric, "panel": panel, "animal": r.animal, "plot_ABL": r.plot_ABL,
                "position": r.position, "value": value, "color": to_hex(COLORS[r.plot_ABL]),
                "pdf_x_pt": (b.x0 + (r.position-xlo)/(xhi-xlo)*b.width) * FIGSIZE[0] * 72,
                "pdf_y_pt": (1-b.y0-(value-ylo)/(yhi-ylo)*b.height) * FIGSIZE[1] * 72})
    name = f"ASD_session_type_1_animal_{config['filename']}"
    pdf, png = OUTPUT / f"{name}.pdf", OUTPUT / f"{name}.png"
    fig.savefig(pdf, metadata={"Title": f"ASD: {config['title']} and signed animal-centered deviations"})
    fig.savefig(png, dpi=PNG_DPI, facecolor="white")
    figure_records.append({"metric": metric, "animals": len(means), "choices": config['trials'],
        "ABL_values_per_panel": len(values), "animal_order": means.animal.to_list(),
        "raw_ylim": list(upper.get_ylim()), "delta_ylim": list(lower.get_ylim()),
        "pdf": str(pdf.relative_to(REPO)), "png": str(png.relative_to(REPO))})
    plt.close(fig)
    print(f"Saved {metric}: {len(means)} animals, {len(values)} values per panel", flush=True)
pd.DataFrame(drawn_points).to_csv(OUTPUT / "plotted_points.csv", index=False)
pd.DataFrame(drawn_bars).to_csv(OUTPUT / "plotted_summary_bars.csv", index=False)


# %% Validation/provenance manifest and preservation of all previous analyses
for relative, digest in protected_hashes.items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, relative
manifest = {"status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "reference_folder": str(REFERENCE.relative_to(REPO)), "input_csv": str(INPUT.relative_to(REPO)),
    "input_sha256": reference_manifest["input_sha256"], "selection": reference_manifest["selection"],
    "refitted": False, "rt_restriction": None, "repeat_filter_reapplied": False,
    "animals_audited": 53, "eligible_animals": {"jnd": 50, "k": 53},
    "eligible_choices": {"jnd": 444718, "k": 471416}, "ABL_values_per_panel": {"jnd": 150, "k": 159},
    "eligibility": "Require finite positive values at all three ABLs independently for each metric; no incomplete means",
    "jnd_excluded_animals": ["ASD0035", "ASD0044", "ASD0048"],
    "animal_mean": "Equal arithmetic mean of ABL20/40/60 values", "delta": "Signed M_ABL - animal mean",
    "grand_mean": "Equal arithmetic mean of eligible animal means", "summary_SD_ddof": POPULATION_SD_DDOF,
    "upper_summary": "Grand mean +/- population SD of animal means",
    "lower_summary": "Mean +/- population SD of all displayed signed ABL deviations",
    "sort": "Ascending unrounded animal mean; ties by animal ID", "colors": {str(a): to_hex(c) for a,c in COLORS.items()},
    "figure_inches": list(FIGSIZE), "png_dpi": PNG_DPI, "figures": figure_records,
    "audit_values": len(points), "plotted_values": len(drawn_points), "summary_bars": len(drawn_bars),
    "protected_file_sha256": protected_hashes}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Saved two PNG/PDF pairs, 53-animal eligibility, means/order, values/deltas, summaries and plotting audits", flush=True)
