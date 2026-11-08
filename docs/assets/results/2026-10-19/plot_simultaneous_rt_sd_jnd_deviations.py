"""Extend the participant JND/deviation figure with two SD-condition rows.

Run .venv/bin/python ASD_Human_Data/plot_simultaneous_rt_sd_jnd_deviations.py.
Reuse the published RT fits and fit SD separately with the same procedure.
"""

# %% Imports and editable parameters
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "simultaneous_rt_sd_jnd_deviations"
RT_REFERENCE = HERE / "simultaneous_rt_metric_deviations"
CONDITIONS = ["RT", "SD"]
ASLS = [20, 40]
SIGNED_DELTAS = [-2, -1.3, -.65, -.3, -.1, .1, .3, .65, 1.3, 2]
COLORS = {20: "tab:blue", 40: "tab:orange"}
MARKERS = {"Control": "o", "ASD": "*"}
GROUP_COLORS = {"Control": "tab:blue", "ASD": "tab:red"}
SUMMARY_STYLES = {20: "-", 40: (0, (.8, 1.3))}
SUMMARY_POSITIONS = {("Control", 20): .10, ("Control", 40): .31,
                     ("ASD", 20): .69, ("ASD", 40): .90}
BOUNDS = ([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf])
FIGSIZE = (20, 18)
PNG_DPI = 180
STEM = "human_simultaneous_rt_sd_jnd_and_delta_jnd"
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Same per-participant populations and probabilities, kept separate by condition
input_hashes, all_trials, count_tables = {}, [], []
for condition in CONDITIONS:
    path = HERE / f"simultaneous_{condition.lower()}_trials.csv"
    extraction = json.loads((HERE / f"simultaneous_{condition.lower()}_extraction.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == extraction["output_sha256"]
    assert extraction["selection"] == f"Condition == '{condition}'"
    input_hashes[path.name] = digest
    data = pd.read_csv(path, float_precision="round_trip")
    assert data.Condition.eq(condition).all() and data.Valid.eq(1).all()
    assert data.Choice.isin([-1, 1]).all() and np.isfinite(data.RT_s).all()
    assert data.GroupID.eq(data.Group.map({"Control": 0, "ASD": 1})).all()
    assert not data.duplicated(["Participant", "Trial"]).any()
    assert np.allclose(data.Delta_dB, data.Stim_right_dB-data.Stim_left_dB, atol=1e-12, rtol=0)
    expected_n = 30 if condition == "RT" else 16
    assert len(data) == expected_n * 20 * 19
    if condition == "SD":
        assert data.Stim_duration_s.eq(.03).all()
    else:
        assert data.Stim_duration_s.isna().all()
    data["delta_key"] = data.Delta_dB.round(2)
    assert sorted(data.delta_key.unique()) == SIGNED_DELTAS
    table = data.groupby(["Condition", "Participant", "Group", "GroupID", "ASL_cd", "delta_key"], as_index=False).agg(
        trials=("Choice", "size"), right_trials=("Choice", lambda x: int(x.eq(1).sum())))
    assert len(table) == 380 and table.trials.eq(expected_n).all()
    table["left_trials"] = table.trials-table.right_trials
    table["p_right"] = table.right_trials/table.trials
    table["fitted_p_right"] = np.nan
    all_trials.append(data)
    count_tables.append(table)
trials = pd.concat(all_trials, ignore_index=True)
counts = pd.concat(count_tables, ignore_index=True)
participants = trials[["Participant", "Group", "GroupID"]].drop_duplicates().sort_values(["GroupID", "Participant"]).reset_index(drop=True)
participants["position"] = np.arange(len(participants))
assert len(participants) == 19 and participants.Group.tolist() == ["Control"]*11+["ASD"]*8
assert trials.groupby(["Condition", "Participant"]).size().to_dict() == {
    (c, int(p)): (600 if c == "RT" else 320) for c in CONDITIONS for p in participants.Participant}
participants.to_csv(OUTPUT / "participant_order.csv", index=False)


# %% Reuse RT parameters exactly; calculate SD fits and the established JND crossings
def sigmoid(x, upper, lower, x0, k):
    return lower + (upper-lower)/(1+np.exp(-k*(x-x0)))


rt_fits = pd.read_csv(RT_REFERENCE / "fits.csv", float_precision="round_trip")
rt_counts = pd.read_csv(RT_REFERENCE / "condition_counts.csv", float_precision="round_trip")
check = counts.loc[counts.Condition.eq("RT")].merge(rt_counts, on=["Participant", "ASL_cd", "delta_key"], suffixes=("", "_reference"))
assert len(check) == 380
for field in ["trials", "right_trials", "left_trials", "p_right"]:
    np.testing.assert_allclose(check[field], check[f"{field}_reference"], atol=1e-12, rtol=0)
rt_fits.insert(0, "Condition", "RT")
fit_rows = []
for (participant, group, group_id, asl), table in counts.loc[counts.Condition.eq("SD")].groupby(
        ["Participant", "Group", "GroupID", "ASL_cd"], sort=False):
    table = table.sort_values("delta_key")
    x, y = table.delta_key.to_numpy(dtype=float), table.p_right.to_numpy(dtype=float)
    row = {"Condition": "SD", "Participant": int(participant), "Group": group, "GroupID": int(group_id),
           "ASL_cd": int(asl), "trials": int(table.trials.sum()),
           "right_trials": int(table.right_trials.sum()), "left_trials": int(table.left_trials.sum()),
           "upper": np.nan, "lower": np.nan, "x0": np.nan, "k": np.nan,
           "jnd": np.nan, "y_high": np.nan, "y_low": np.nan, "delta_high": np.nan, "delta_low": np.nan,
           "fit_status": "optimizer_failed", "jnd_status": "fit_unavailable", "rmse": np.nan,
           "initial_parameters": json.dumps([float(y.max()), float(y.min()), float(np.median(x)), .1])}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parameters, _ = curve_fit(sigmoid, x, y, p0=[y.max(), y.min(), np.median(x), .1], bounds=BOUNDS, maxfev=10000)
        upper, lower, x0, k = parameters
        row.update(upper=float(upper), lower=float(lower), x0=float(x0), k=float(k),
                   warnings="; ".join(str(w.message) for w in caught))
        if np.isfinite(parameters).all() and upper > lower and k > 0:
            row.update(fit_status="ok", rmse=float(np.sqrt(np.mean((sigmoid(x, *parameters)-y)**2))))
            high, low = .75*upper, .25*(1-lower)
            row.update(y_high=float(high), y_low=float(low))
            if lower < low < high < upper:
                inverse = lambda target: x0-np.log((upper-lower)/(target-lower)-1)/k
                x_high, x_low = inverse(high), inverse(low)
                jnd = (x_high-x_low)/2
                if np.isfinite([x_high, x_low, jnd]).all() and jnd > 0:
                    row.update(jnd=float(jnd), delta_high=float(x_high), delta_low=float(x_low), jnd_status="ok")
            else:
                row["jnd_status"] = "thresholds_outside_asymptotes_or_reversed"
        else:
            row["fit_status"] = "invalid_parameters"
    except (RuntimeError, ValueError, FloatingPointError) as error:
        row["optimizer_message"] = str(error)
    fit_rows.append(row)
fits = pd.concat([rt_fits, pd.DataFrame(fit_rows)], ignore_index=True)
assert len(fits) == 76
for fit in fits.itertuples(index=False):
    if fit.fit_status == "ok":
        mask = counts.Condition.eq(fit.Condition)&counts.Participant.eq(fit.Participant)&counts.ASL_cd.eq(fit.ASL_cd)
        counts.loc[mask, "fitted_p_right"] = sigmoid(counts.loc[mask, "delta_key"].to_numpy(), fit.upper, fit.lower, fit.x0, fit.k)
counts.to_csv(OUTPUT / "condition_counts.csv", index=False)
fits.to_csv(OUTPUT / "fits_and_jnd_crossings.csv", index=False)


# %% Paired-ASL eligibility and condition-specific participant means and signed deltas
eligible_rows, value_rows, mean_rows, summary_rows, group_rows = [], [], [], [], []
for condition in CONDITIONS:
    for person in participants.itertuples(index=False):
        own = fits.loc[fits.Condition.eq(condition)&fits.Participant.eq(person.Participant)].set_index("ASL_cd")
        jnds = np.array([own.loc[a, "jnd"] for a in ASLS], dtype=float)
        eligible = bool(own.jnd_status.eq("ok").all() and np.isfinite(jnds).all() and (jnds > 0).all())
        eligible_rows.append({"Condition": condition, "Participant": person.Participant, "Group": person.Group,
                              "position": person.position, "jnd_ASL_20": jnds[0], "jnd_ASL_40": jnds[1],
                              "eligible": eligible, "status_ASL_20": own.loc[20, "jnd_status"], "status_ASL_40": own.loc[40, "jnd_status"]})
        if not eligible:
            continue
        center = float(jnds.mean())
        mean_rows.append({"Condition": condition, "Participant": person.Participant, "Group": person.Group,
                          "position": person.position, "participant_mean": center})
        for asl, value in zip(ASLS, jnds):
            value_rows.append({"Condition": condition, "Participant": person.Participant, "Group": person.Group,
                               "position": person.position, "ASL_cd": asl, "jnd": value,
                               "participant_mean": center, "signed_delta": value-center})
    condition_means = [r["participant_mean"] for r in mean_rows if r["Condition"] == condition]
    deviations = [r["signed_delta"] for r in value_rows if r["Condition"] == condition]
    for level, sample in [("participant_means", condition_means), ("ASL_deviations", deviations)]:
        center, sd = float(np.mean(sample)), float(np.std(sample, ddof=0))
        summary_rows.append({"Condition": condition, "level": level, "n": len(sample), "mean": center,
                             "population_sd": sd, "lower": center-sd, "upper": center+sd, "ddof": 0})
    for group in ["Control", "ASD"]:
        for level, sample in [("participant_means", [r["participant_mean"] for r in mean_rows if r["Condition"] == condition and r["Group"] == group]),
                              ("ASL_deviations", [r["signed_delta"] for r in value_rows if r["Condition"] == condition and r["Group"] == group])]:
            center, sd = float(np.mean(sample)), float(np.std(sample, ddof=0))
            group_rows.append({"Condition": condition, "Group": group, "level": level, "n": len(sample),
                               "mean": center, "population_sd": sd, "lower": center-sd, "upper": center+sd, "ddof": 0})
eligibility, values, means = pd.DataFrame(eligible_rows), pd.DataFrame(value_rows), pd.DataFrame(mean_rows)
summary, group_summary = pd.DataFrame(summary_rows), pd.DataFrame(group_rows)
assert values.groupby(["Condition", "Participant"]).signed_delta.sum().abs().max() < 1e-12
for name, table in [("eligibility", eligibility), ("values_and_deviations", values), ("participant_means", means),
                     ("summary_statistics", summary), ("group_summary_statistics", group_summary)]:
    table.to_csv(OUTPUT / f"{name}.csv", index=False)

# The right-side bars summarize each ASL separately across eligible
# participants. Delta bars therefore no longer pool the two signed ASLs.
asl_summary_rows = []
for (condition, group, asl), own in values.groupby(["Condition", "Group", "ASL_cd"]):
    for level, field in [("participant_ASL_values", "jnd"), ("ASL_deviations", "signed_delta")]:
        center, sd = float(own[field].mean()), float(own[field].std(ddof=0))
        asl_summary_rows.append({"Condition": condition, "Group": group, "ASL_cd": int(asl),
                                 "level": level, "n": len(own), "mean": center,
                                 "population_sd": sd, "lower": center-sd, "upper": center+sd, "ddof": 0})
asl_summary = pd.DataFrame(asl_summary_rows)
asl_summary.to_csv(OUTPUT / "group_asl_summary_statistics.csv", index=False)


# %% Four rows: RT JND, RT delta, SD JND, SD delta
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12})
fig = plt.figure(figsize=FIGSIZE)
grid = fig.add_gridspec(4, 2, width_ratios=[30, 8], left=.07, right=.975,
                       top=.885, bottom=.115, hspace=.46, wspace=.035)
drawn_points, drawn_means, drawn_bars, panel_rows = [], [], [], []
for condition_index, condition in enumerate(CONDITIONS):
    own = values.loc[values.Condition.eq(condition)]
    own_means = means.loc[means.Condition.eq(condition)]
    stats = summary.loc[summary.Condition.eq(condition)].set_index("level")
    bar_stats = asl_summary.loc[asl_summary.Condition.eq(condition)]
    own_eligibility = eligibility.loc[eligibility.Condition.eq(condition)]
    for offset, panel, field, level in [(0, "raw", "jnd", "participant_means"), (1, "delta", "signed_delta", "ASL_deviations")]:
        row_index = 2*condition_index+offset
        ax = fig.add_subplot(grid[row_index, 0])
        bar_ax = fig.add_subplot(grid[row_index, 1], sharey=ax)
        for asl, x_offset in [(20, -.105), (40, .105)]:
            for group in ["Control", "ASD"]:
                selected = own.loc[own.ASL_cd.eq(asl)&own.Group.eq(group)]
                ax.scatter(selected.position+x_offset, selected[field], color=COLORS[asl], marker=MARKERS[group],
                           s=44 if group == "Control" else 110, edgecolors="black", linewidths=.35, alpha=.87, zorder=3)
                drawn_points.extend({"Condition": condition, "panel": panel, "Participant": int(p.Participant),
                                     "Group": group, "ASL_cd": asl, "marker": MARKERS[group], "color": to_hex(COLORS[asl]),
                                     "x": float(p.position+x_offset), "y": float(getattr(p, field))} for p in selected.itertuples(index=False))
        if panel == "raw":
            for group in ["Control", "ASD"]:
                selected = own_means.loc[own_means.Group.eq(group)]
                ax.scatter(selected.position, selected.participant_mean, marker="_", color=GROUP_COLORS[group], s=180, linewidths=1.5, zorder=5)
                drawn_means.extend({"Condition": condition, "Participant": int(p.Participant), "Group": group,
                                    "x": float(p.position), "y": float(p.participant_mean)} for p in selected.itertuples(index=False))
            ax.axhline(stats.loc[level, "mean"], color="black", ls=":", lw=1)
            ymax = max(own.jnd.max(), stats.loc[level, "upper"], bar_stats.loc[bar_stats.level.eq("participant_ASL_values"), "upper"].max())
            ylim = (0, float(np.ceil(ymax*1.08/.1)*.1))
            title = "RT" if condition == "RT" else "SD — 30 ms"
            ax.set_title(f"{title} | JND at each ASL | {len(own_means)}/19 paired participants", loc="left", pad=25)
            for group, x in [("Control", 5), ("ASD", 14.5)]:
                n = int(own_means.Group.eq(group).sum())
                ax.text(x, 1.025, f"{group} (n={n})", transform=ax.get_xaxis_transform(), ha="center", fontsize=11)
            ax.set_ylabel("JND (dB)")
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            ax.axhline(0, color="black", lw=1)
            extent = max(own.signed_delta.abs().max(), bar_stats.loc[bar_stats.level.eq(level), ["lower", "upper"]].abs().max().max())
            extent = float(np.ceil(extent*1.12/.1)*.1)
            ylim = (-extent, extent)
            ax.set_title(f"{condition} | Signed deviation: ASL JND − participant mean", loc="left")
            ax.set_ylabel("ΔJND (dB)")
            ax.set_xticks(participants.position, [f"P{p}" for p in participants.Participant], rotation=90, fontsize=9)
        ax.set(xlim=(-.6, 18.6), ylim=ylim)
        ax.axvline(10.5, color="gray", ls="--", lw=1, alpha=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        for unavailable in own_eligibility.loc[~own_eligibility.eligible].itertuples(index=False):
            ax.text(unavailable.position, .035, "N/A", transform=ax.get_xaxis_transform(), color="gray", ha="center", fontsize=8)
        for (group, asl), x in SUMMARY_POSITIONS.items():
            bar_level = "participant_ASL_values" if panel == "raw" else "ASL_deviations"
            stat = bar_stats.loc[bar_stats.Group.eq(group)&bar_stats.ASL_cd.eq(asl)&bar_stats.level.eq(bar_level)].iloc[0]
            bar_ax.plot([x, x], [stat.lower, stat.upper], color=GROUP_COLORS[group], lw=4,
                        linestyle=SUMMARY_STYLES[asl], solid_capstyle="butt", dash_capstyle="round")
            display_mean = 0 if abs(stat["mean"]) < .0005 else stat["mean"]
            label = f"{group}\nASL {asl}\nmean {display_mean:.3f}\nSD {stat.population_sd:.3f}"
            bar_ax.annotate(label, (x, stat.upper), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", color=GROUP_COLORS[group], fontsize=8)
            drawn_bars.append({"Condition": condition, "panel": panel, "Group": group, "ASL_cd": asl,
                               "linestyle": "solid" if asl == 20 else "dotted", "x": x,
                               "n": int(stat.n), "mean": float(stat["mean"]), "sd": float(stat.population_sd),
                               "lower": float(stat.lower), "upper": float(stat.upper), "label": label})
        bar_ax.set_xlim(0, 1)
        bar_ax.axis("off")
        bar_ax.set_title("Group × ASL mean ±1 SD\nASL 20: solid | ASL 40: dotted", fontsize=9, pad=23)
        panel_rows.append({"Condition": condition, "panel": panel, "row": row_index+1, "ylim_low": ylim[0], "ylim_high": ylim[1], "values": len(own)})
        fig.canvas.draw()
        for p in drawn_points:
            if p["Condition"] == condition and p["panel"] == panel:
                x, y = ax.transData.transform((p["x"], p["y"]))
                p.update(pdf_x_pt=float(x*72/fig.dpi), pdf_y_pt=float(FIGSIZE[1]*72-y*72/fig.dpi))
        for bar in drawn_bars:
            if bar["Condition"] == condition and bar["panel"] == panel:
                x, upper = bar_ax.transData.transform((bar["x"], bar["upper"]))
                _, lower = bar_ax.transData.transform((bar["x"], bar["lower"]))
                bar.update(pdf_x_pt=float(x*72/fig.dpi), pdf_upper_y_pt=float(FIGSIZE[1]*72-upper*72/fig.dpi), pdf_lower_y_pt=float(FIGSIZE[1]*72-lower*72/fig.dpi))
fig.suptitle("Human simultaneous RT and SD | JND and signed deviation from each participant’s mean", fontsize=18, y=.985)
fig.text(.5, .958, "RT: 11,400 choices, 19 complete pairs | SD (30 ms): 6,080 choices, 18 complete pairs | conditions fitted separately", ha="center", fontsize=11)
handles = [Line2D([], [], marker="o", ls="None", color=COLORS[a], ms=8, label=f"ASL {a} cd/m²") for a in ASLS]
handles += [Line2D([], [], marker=MARKERS[g], ls="None", color="black", ms=8 if g == "Control" else 12, label=g) for g in ["Control", "ASD"]]
handles += [Line2D([], [], marker="_", ls="None", color=GROUP_COLORS[g], ms=12, label=f"{g} participant mean") for g in ["Control", "ASD"]]
handles += [Line2D([], [], color="black", ls=":", label="Overall mean (equal participants)")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94), ncol=7, frameon=False, fontsize=10)
fig.text(.5, .075, "Participant order: Control IDs first, then ASD; same order in all rows | Participant mean = (JND20 + JND40)/2 | Signed Δ = JND_ASL − participant mean", ha="center", fontsize=9)
fig.text(.5, .052, "Right bars: each group × ASL mean ±1 population SD; Control blue, ASD red; ASL 20 solid, ASL 40 dotted | RT: 30 choices per difference; SD: 16", ha="center", fontsize=9)
fig.text(.5, .029, "N/A: incomplete paired JND; SD P52 ASL 20 has no lower threshold crossing | Unweighted four-parameter fits; JND = half the distance between the established crossings", ha="center", fontsize=9)
fig.savefig(OUTPUT / f"{STEM}.png", dpi=PNG_DPI, facecolor="white")
fig.savefig(OUTPUT / f"{STEM}.pdf", facecolor="white")
plt.close(fig)
pd.DataFrame(drawn_points).to_csv(OUTPUT / "plotted_points.csv", index=False)
pd.DataFrame(drawn_means).to_csv(OUTPUT / "plotted_means.csv", index=False)
pd.DataFrame(drawn_bars).to_csv(OUTPUT / "plotted_group_bars.csv", index=False)
pd.DataFrame(panel_rows).to_csv(OUTPUT / "panel_limits.csv", index=False)
manifest = {"status": "passed", "input_sha256": input_hashes,
            "rt_fits_sha256": hashlib.sha256((RT_REFERENCE / "fits.csv").read_bytes()).hexdigest(),
            "selected_trials": {c: int(trials.Condition.eq(c).sum()) for c in CONDITIONS},
            "fits": 76, "defined_jnds": int(fits.jnd_status.eq("ok").sum()),
            "eligible_participants": {c: int(eligibility.loc[eligibility.Condition.eq(c), "eligible"].sum()) for c in CONDITIONS},
            "paired_rule": "Both ASLs need defined positive JNDs; unavailable pairs omitted from both condition panels and summaries; participant positions retained",
            "fit_method": "unweighted curve_fit; parameters upper,lower,x0,k; initial max(P),min(P),median(x),0.1; bounds [0,0,-inf,0] to [1,1,inf,inf]; maxfev=10000",
            "jnd": "(x(.75*upper) - x(.25*(1-lower)))/2, if both crossings lie strictly between the asymptotes",
            "participant_order": participants.Participant.astype(int).tolist(), "markers": MARKERS,
            "summary_sd_ddof": 0, "right_bar_definition": "four bars per panel: Control ASL20/40, ASD ASL20/40; mean +/- population SD across eligible participants at that ASL",
            "right_bar_styles": {"20": "solid", "40": "dotted"}, "right_bar_colors": {g: to_hex(c) for g, c in GROUP_COLORS.items()},
            "rows": panel_rows, "png": f"{STEM}.png", "pdf": f"{STEM}.pdf"}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({key: manifest[key] for key in ["selected_trials", "fits", "defined_jnds", "eligible_participants", "png"]}, indent=2))
