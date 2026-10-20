"""Human simultaneous RT psychometrics and participant-centered metric plots.

Run with .venv/bin/python ASD_Human_Data/plot_simultaneous_rt_metric_deviations.py.
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
INPUT = HERE / "simultaneous_rt_trials.csv"
EXTRACTION = HERE / "simultaneous_rt_extraction.json"
OUTPUT = HERE / "simultaneous_rt_metric_deviations"
ASLS = [20, 40]  # Stored ASL_cd values, in cd/m²; these are not dB labels.
SIGNED_DELTAS = [-2, -1.3, -.65, -.3, -.1, .1, .3, .65, 1.3, 2]
COLORS = {20: "tab:blue", 40: "tab:orange"}
MARKERS = {"Control": "o", "ASD": "*"}
MEAN_COLORS = {"Control": "tab:blue", "ASD": "tab:red"}
FIT_BOUNDS = ([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf])
INITIAL_K = .1
MAXFEV = 10000
FIGSIZE = (20, 10)
PNG_DPI = 180
METRICS = {
    "jnd": {"source": "jnd", "label": "JND", "units": "dB",
            "step": .1, "stem": "human_simultaneous_rt_jnd_and_delta_jnd"},
    "slope": {"source": "k", "label": "Slope k", "units": "dB⁻¹",
              "step": 1, "stem": "human_simultaneous_rt_slope_and_delta_slope"},
}


# %% Source identity and the exact RT-only denominator
extraction = json.loads(EXTRACTION.read_text())
input_sha256 = hashlib.sha256(INPUT.read_bytes()).hexdigest()
assert input_sha256 == extraction["output_sha256"]
assert extraction["selection"] == "Condition == 'RT'"
trials = pd.read_csv(INPUT, float_precision="round_trip")
assert len(trials) == 11400 and trials.Participant.nunique() == 19
assert trials.Condition.eq("RT").all() and trials.Valid.eq(1).all()
assert trials.Stim_duration_s.isna().all() and trials.RT_s.notna().all()
assert trials.Choice.isin([-1, 1]).all()
assert (trials.Choice.eq(1) == trials.ChoseRight.eq(1)).all()
assert trials.GroupID.eq(trials.Group.map({"Control": 0, "ASD": 1})).all()
assert not trials.duplicated(["Participant", "Trial"]).any()
assert set(trials.ASL_cd) == set(ASLS)
assert np.allclose(trials.Delta_dB, trials.Stim_right_dB - trials.Stim_left_dB,
                   rtol=0, atol=1e-12)
trials["delta_key"] = trials.Delta_dB.round(2)
assert np.allclose(trials.Delta_dB, trials.delta_key, rtol=0, atol=1e-12)
assert sorted(trials.delta_key.unique().tolist()) == SIGNED_DELTAS
assert trials.groupby("Group").Participant.nunique().to_dict() == {"ASD": 8, "Control": 11}
assert trials.groupby(["Participant", "ASL_cd"]).size().eq(300).all()

participants = (trials[["Participant", "Group", "GroupID"]].drop_duplicates()
                .sort_values(["GroupID", "Participant"]).reset_index(drop=True))
participants["position"] = np.arange(len(participants))
assert participants.Group.tolist() == ["Control"] * 11 + ["ASD"] * 8
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Ten measured P(right) values per participant and nominal brightness level
trials["right"] = trials.Choice.eq(1)
counts = trials.groupby(["Participant", "Group", "GroupID", "ASL_cd", "delta_key"],
                        as_index=False).agg(trials=("right", "size"),
                                             right_trials=("right", "sum"))
counts["left_trials"] = counts.trials - counts.right_trials
counts["p_right"] = counts.right_trials / counts.trials
counts["fitted_p_right"] = np.nan
assert len(counts) == 380 and counts.trials.eq(30).all()
assert int(counts.trials.sum()) == 11400
assert counts.groupby(["Participant", "ASL_cd"]).size().eq(10).all()


# %% Same four-parameter, unweighted sigmoid and JND as the linked animal plots
def sigmoid(x, upper, lower, x0, k):
    return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))


fit_rows = []
for (participant, group, group_id, asl), condition in counts.groupby(
        ["Participant", "Group", "GroupID", "ASL_cd"], sort=False):
    condition = condition.sort_values("delta_key")
    x = condition.delta_key.to_numpy(dtype=float)
    probability = condition.p_right.to_numpy(dtype=float)
    assert x.tolist() == SIGNED_DELTAS
    initial = [float(probability.max()), float(probability.min()),
               float(np.median(x)), INITIAL_K]
    row = {"Participant": int(participant), "Group": group, "GroupID": int(group_id),
           "ASL_cd": int(asl), "trials": int(condition.trials.sum()),
           "right_trials": int(condition.right_trials.sum()),
           "left_trials": int(condition.left_trials.sum()),
           "upper": np.nan, "lower": np.nan, "x0": np.nan, "k": np.nan,
           "jnd": np.nan, "y_high": np.nan, "y_low": np.nan,
           "delta_high": np.nan, "delta_low": np.nan,
           "fit_status": "optimizer_failed", "jnd_status": "fit_unavailable",
           "rmse": np.nan, "function_evaluations": 0,
           "initial_parameters": json.dumps(initial), "optimizer_message": "", "warnings": ""}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parameters, covariance, info, message, _ = curve_fit(
                sigmoid, x, probability, p0=initial, bounds=FIT_BOUNDS,
                maxfev=MAXFEV, full_output=True)
        upper, lower, x0, k = parameters
        row.update(upper=float(upper), lower=float(lower), x0=float(x0), k=float(k),
                   function_evaluations=int(info["nfev"]), optimizer_message=message,
                   warnings="; ".join(dict.fromkeys(str(w.message) for w in caught)),
                   covariance_finite=bool(np.isfinite(covariance).all()))
        if np.isfinite(parameters).all() and upper > lower and k > 0:
            row["fit_status"] = "ok"
            predicted = sigmoid(x, *parameters)
            row["rmse"] = float(np.sqrt(np.mean((predicted - probability) ** 2)))
            counts.loc[condition.index, "fitted_p_right"] = predicted
            high, low = .75 * upper, .25 * (1 - lower)
            row.update(y_high=float(high), y_low=float(low))
            if lower < low < high < upper:
                inverse = lambda target: x0 - np.log((upper - lower) / (target - lower) - 1) / k
                delta_high, delta_low = inverse(high), inverse(low)
                jnd = (delta_high - delta_low) / 2
                if np.isfinite([delta_high, delta_low, jnd]).all() and jnd > 0:
                    row.update(delta_high=float(delta_high), delta_low=float(delta_low),
                               jnd=float(jnd), jnd_status="ok")
                else:
                    row["jnd_status"] = "nonfinite_or_nonpositive"
            else:
                row["jnd_status"] = "thresholds_outside_asymptotes_or_reversed"
        else:
            row["fit_status"] = "invalid_parameters"
    except (RuntimeError, ValueError, FloatingPointError) as error:
        row["optimizer_message"] = str(error)
    fit_rows.append(row)

fits = pd.DataFrame(fit_rows).sort_values(["GroupID", "Participant", "ASL_cd"])
assert len(fits) == 38
counts.to_csv(OUTPUT / "condition_counts.csv", index=False)
fits.to_csv(OUTPUT / "fits.csv", index=False)


# %% Metric-specific eligibility, equal-ASL participant means, signed deviations
eligibility_rows, mean_rows, point_rows, summary_rows, group_summary_rows = [], [], [], [], []
for metric, config in METRICS.items():
    field = config["source"]
    for person in participants.itertuples(index=False):
        own = fits.loc[fits.Participant.eq(person.Participant)].set_index("ASL_cd")
        values = [float(own.loc[asl, field]) for asl in ASLS]
        status_field = "jnd_status" if metric == "jnd" else "fit_status"
        statuses_valid = all(own.loc[asl, status_field] == "ok" for asl in ASLS)
        eligible = bool(statuses_valid and np.isfinite(values).all() and np.all(np.array(values) > 0))
        eligibility_rows.append({"metric": metric, "Participant": person.Participant,
                                 "Group": person.Group, "position": person.position,
                                 "value_ASL_20": values[0], "value_ASL_40": values[1],
                                 "eligible": eligible})
        if not eligible:
            continue
        participant_mean = float(np.mean(values))
        mean_rows.append({"metric": metric, "Participant": person.Participant,
                          "Group": person.Group, "position": person.position,
                          "participant_mean": participant_mean,
                          "choice_trials": int(own.trials.sum())})
        for asl, value in zip(ASLS, values):
            point_rows.append({"metric": metric, "Participant": person.Participant,
                               "Group": person.Group, "position": person.position,
                               "ASL_cd": asl, "value": value,
                               "participant_mean": participant_mean,
                               "signed_delta": value - participant_mean,
                               "trials": int(own.loc[asl, "trials"])})
    metric_means = np.array([row["participant_mean"] for row in mean_rows if row["metric"] == metric])
    metric_deltas = np.array([row["signed_delta"] for row in point_rows if row["metric"] == metric])
    assert len(metric_means) > 0
    for level, values in [("participant_means", metric_means), ("ASL_deviations", metric_deltas)]:
        center, sd = float(np.mean(values)), float(np.std(values, ddof=0))
        summary_rows.append({"metric": metric, "level": level, "n": len(values),
                             "mean": center, "population_sd": sd, "ddof": 0,
                             "lower": center - sd, "upper": center + sd,
                             "units": config["units"]})
    for group in ["Control", "ASD"]:
        own_means = np.array([row["participant_mean"] for row in mean_rows
                              if row["metric"] == metric and row["Group"] == group])
        own_deltas = np.array([row["signed_delta"] for row in point_rows
                               if row["metric"] == metric and row["Group"] == group])
        for level, values in [("participant_means", own_means),
                              ("ASL_deviations", own_deltas)]:
            if not len(values):
                continue
            center, sd = float(np.mean(values)), float(np.std(values, ddof=0))
            group_summary_rows.append({"metric": metric, "Group": group,
                                       "level": level, "n": len(values),
                                       "mean": center, "population_sd": sd,
                                       "ddof": 0, "lower": center - sd,
                                       "upper": center + sd, "units": config["units"]})

eligibility = pd.DataFrame(eligibility_rows)
means = pd.DataFrame(mean_rows)
points = pd.DataFrame(point_rows)
summary = pd.DataFrame(summary_rows)
group_summary = pd.DataFrame(group_summary_rows)
assert len(eligibility) == 38
assert len(means) == int(eligibility.eligible.sum())
assert len(points) == 2 * len(means)
assert points.groupby(["metric", "Participant"]).signed_delta.sum().abs().max() < 1e-12
eligibility.to_csv(OUTPUT / "eligibility.csv", index=False)
means.to_csv(OUTPUT / "participant_means.csv", index=False)
points.to_csv(OUTPUT / "values_and_deviations.csv", index=False)
summary.to_csv(OUTPUT / "summary_statistics.csv", index=False)
group_summary.to_csv(OUTPUT / "group_summary_statistics.csv", index=False)


# %% Matched two-panel figures with common participant order
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12})
figure_rows, drawn_points, drawn_means, drawn_bars = [], [], [], []
for metric, config in METRICS.items():
    m = means.loc[means.metric.eq(metric)]
    p = points.loc[points.metric.eq(metric)]
    stats = summary.loc[summary.metric.eq(metric)].set_index("level")
    group_stats = group_summary.loc[group_summary.metric.eq(metric)]
    fig = plt.figure(figsize=FIGSIZE)
    grid = fig.add_gridspec(2, 2, width_ratios=[32, 4.2], height_ratios=[1, 1],
                            left=.07, right=.96, top=.84, bottom=.20,
                            hspace=.28, wspace=.035)
    upper = fig.add_subplot(grid[0, 0])
    lower = fig.add_subplot(grid[1, 0], sharex=upper)
    upper_sd = fig.add_subplot(grid[0, 1], sharey=upper)
    lower_sd = fig.add_subplot(grid[1, 1], sharey=lower)
    for asl, offset in [(20, -.105), (40, .105)]:
        for group in ["Control", "ASD"]:
            selected = p.loc[p.ASL_cd.eq(asl) & p.Group.eq(group)]
            marker = MARKERS[group]
            size = 44 if group == "Control" else 110
            for panel, ax, field in [("raw", upper, "value"),
                                     ("delta", lower, "signed_delta")]:
                x = selected.position.to_numpy() + offset
                y = selected[field].to_numpy()
                ax.scatter(x, y, color=COLORS[asl], marker=marker, s=size,
                           edgecolors="black", linewidths=.35, alpha=.87, zorder=3)
                drawn_points.extend({"metric": metric, "panel": panel,
                                     "Participant": int(row.Participant), "Group": group,
                                     "ASL_cd": asl, "marker": marker,
                                     "color": to_hex(COLORS[asl]), "x": float(row.position + offset),
                                     "y": float(getattr(row, field))}
                                    for row in selected.itertuples(index=False))
    for group in ["Control", "ASD"]:
        selected_means = m.loc[m.Group.eq(group)]
        upper.scatter(selected_means.position, selected_means.participant_mean,
                      marker="_", color=MEAN_COLORS[group], s=180,
                      linewidths=1.5, zorder=5)
        drawn_means.extend({"metric": metric, "Participant": int(row.Participant),
                            "Group": group, "marker": "_",
                            "color": to_hex(MEAN_COLORS[group]),
                            "x": float(row.position), "y": float(row.participant_mean)}
                           for row in selected_means.itertuples(index=False))
    upper.axhline(float(stats.loc["participant_means", "mean"]), color="black",
                  linestyle=":", linewidth=1)
    lower.axhline(0, color="black", linewidth=1)
    for ax in [upper, lower]:
        ax.axvline(10.5, color="gray", linestyle="--", linewidth=1, alpha=.7)
        ax.set_xlim(-.6, 18.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    upper.text(5, 1.025, "Control (n=11)", transform=upper.get_xaxis_transform(),
               ha="center", fontsize=11)
    upper.text(14.5, 1.025, "ASD (n=8)", transform=upper.get_xaxis_transform(),
               ha="center", fontsize=11)
    upper.set_title(f"{config['label']} at each ASL", loc="left", pad=25)
    lower.set_title("Signed deviation: ASL value − participant mean", loc="left")
    upper.set_ylabel(f"{config['label']} ({config['units']})")
    lower.set_ylabel(f"Δ{config['label']} ({config['units']})")
    ymax = max(float(p.value.max()), float(stats.loc["participant_means", "upper"]),
               float(group_stats.loc[group_stats.level.eq("participant_means"), "upper"].max()))
    step = config["step"]
    upper.set_ylim(0, np.ceil(ymax * 1.08 / step) * step)
    extent = max(float(p.signed_delta.abs().max()),
                 abs(float(stats.loc["ASL_deviations", "lower"])),
                 abs(float(stats.loc["ASL_deviations", "upper"])),
                 float(group_stats.loc[group_stats.level.eq("ASL_deviations"), ["lower", "upper"]].abs().max().max()))
    extent = np.ceil(extent * 1.12 / step) * step
    lower.set_ylim(-extent, extent)
    lower.set_xticks(participants.position,
                     [f"P{person.Participant}" for person in participants.itertuples()],
                     rotation=90, fontsize=9)
    lower.set_xlabel("Participant (Control first, then ASD; same order in both figures)")
    upper.tick_params(axis="x", bottom=False, labelbottom=False)
    for panel, ax, level in [("raw", upper_sd, "participant_means"),
                             ("delta", lower_sd, "ASL_deviations")]:
        for group, x in [("Control", .28), ("ASD", .72)]:
            selected = group_stats.loc[group_stats.level.eq(level) & group_stats.Group.eq(group)]
            if selected.empty:
                continue
            row = selected.iloc[0]
            ax.plot([x, x], [row.lower, row.upper], color=MEAN_COLORS[group],
                    linewidth=5, solid_capstyle="butt")
            display_mean = 0.0 if abs(row["mean"]) < .0005 else row["mean"]
            label = f"{group}\nmean {display_mean:.3f}\nSD {row.population_sd:.3f}"
            ax.annotate(label, xy=(x, row.upper), xytext=(0, 8),
                        textcoords="offset points", ha="center", va="bottom",
                        color=MEAN_COLORS[group], fontsize=8)
            drawn_bars.append({"metric": metric, "panel": panel, "Group": group,
                               "color": to_hex(MEAN_COLORS[group]), "x": x,
                               "n": int(row.n), "mean": float(row["mean"]),
                               "sd": float(row.population_sd), "lower": float(row.lower),
                               "upper": float(row.upper), "label": label})
        ax.set_xlim(0, 1)
        ax.axis("off")
        ax.set_title("Group mean ±1 SD", fontsize=9)
    fig.suptitle(f"Human simultaneous RT | {config['label']} and signed deviation from each participant's mean",
                 fontsize=18, y=.985)
    fig.text(.5, .947, f"{len(m)} eligible of 19 participants (11 Control, 8 ASD) | 11,400 valid RT choices | {len(p)} ASL values in each panel",
             ha="center", fontsize=12)
    handles = [Line2D([], [], marker="o", linestyle="None", color=COLORS[asl],
                      markersize=8, label=f"ASL {asl} cd/m²") for asl in ASLS]
    handles += [Line2D([], [], marker=MARKERS[group], linestyle="None", color="black",
                       markersize=8 if group == "Control" else 12, label=group)
                for group in ["Control", "ASD"]]
    handles += [Line2D([], [], marker="_", linestyle="None", color=MEAN_COLORS[group],
                       markersize=12, label=f"{group} participant mean")
                for group in ["Control", "ASD"]]
    handles += [Line2D([], [], linestyle=":", color="black", label="Overall mean (equal participants)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .925),
               ncol=7, frameon=False)
    fig.text(.5, .105, "Participant mean = (M20 + M40) / 2 | Signed Δ = M_ASL − participant mean | Right bars: Control blue, ASD red; labels give group mean and population SD (ddof=0)",
             ha="center", fontsize=10)
    fig.text(.5, .073, "RT condition only; SD (30 ms) and sequential CB excluded; both simultaneous sessions pooled; no correctness or additional RT filter",
             ha="center", fontsize=10)
    fig.text(.5, .041, "P(right) from all 30 choices at each signed ΔdB; unweighted four-parameter sigmoid; slope is fitted k; JND uses the linked animal-plot crossings",
             ha="center", fontsize=10)
    stem = config["stem"]
    pdf, png = OUTPUT / f"{stem}.pdf", OUTPUT / f"{stem}.png"
    fig.savefig(pdf, metadata={"Title": f"Human simultaneous RT {config['label']} and signed deviations"})
    fig.savefig(png, dpi=PNG_DPI, facecolor="white")
    # Audit coordinates in PDF points let the verifier inspect the rendered marks.
    fig.canvas.draw()
    for row in drawn_points:
        if row["metric"] != metric:
            continue
        ax = upper if row["panel"] == "raw" else lower
        x_px, y_px = ax.transData.transform((row["x"], row["y"]))
        row["pdf_x_pt"] = float(x_px * 72 / fig.dpi)
        row["pdf_y_pt"] = float(FIGSIZE[1] * 72 - y_px * 72 / fig.dpi)
    for row in drawn_means:
        if row["metric"] != metric:
            continue
        x_px, y_px = upper.transData.transform((row["x"], row["y"]))
        row["pdf_x_pt"] = float(x_px * 72 / fig.dpi)
        row["pdf_y_pt"] = float(FIGSIZE[1] * 72 - y_px * 72 / fig.dpi)
    for row in drawn_bars:
        if row["metric"] != metric:
            continue
        ax = upper_sd if row["panel"] == "raw" else lower_sd
        x_px, top_px = ax.transData.transform((row["x"], row["upper"]))
        _, bottom_px = ax.transData.transform((row["x"], row["lower"]))
        row["pdf_x_pt"] = float(x_px * 72 / fig.dpi)
        row["pdf_y_top_pt"] = float(FIGSIZE[1] * 72 - top_px * 72 / fig.dpi)
        row["pdf_y_bottom_pt"] = float(FIGSIZE[1] * 72 - bottom_px * 72 / fig.dpi)
    figure_rows.append({"metric": metric, "participants": int(m.Participant.nunique()),
                        "values_per_panel": len(p), "png": str(png.relative_to(HERE)),
                        "pdf": str(pdf.relative_to(HERE)),
                        "raw_ylim": list(map(float, upper.get_ylim())),
                        "delta_ylim": list(map(float, lower.get_ylim()))})
    plt.close(fig)

pd.DataFrame(drawn_points).to_csv(OUTPUT / "plotted_points.csv", index=False)
pd.DataFrame(drawn_means).to_csv(OUTPUT / "plotted_mean_marks.csv", index=False)
pd.DataFrame(drawn_bars).to_csv(OUTPUT / "plotted_summary_bars.csv", index=False)
manifest = {
    "status": "passed", "source": str(INPUT.relative_to(HERE)), "source_sha256": input_sha256,
    "mat_source": extraction["source"], "mat_source_sha256": extraction["source_sha256"],
    "selection": "Condition == RT; Valid == 1; both simultaneous sessions pooled; SD and CB excluded; no additional RT or correctness filter",
    "participants": 19, "control_participants": 11, "asd_participants": 8,
    "choices": len(trials), "right_choices": int(trials.right.sum()),
    "left_choices": int((~trials.right).sum()),
    "ASL_cd": ASLS, "signed_Delta_dB": SIGNED_DELTAS,
    "condition_points": len(counts), "trials_per_condition_point": 30,
    "fit_count": len(fits), "fit_function": "lower + (upper-lower)/(1+exp(-k*(Delta_dB-x0)))",
    "fit_parameters": ["upper", "lower", "x0", "k"],
    "fit_initial_guess": "[max(P), min(P), median(Delta_dB), 0.1]",
    "fit_weighting": "Unweighted curve_fit across ten measured P(right) values",
    "fit_bounds": "upper/lower in [0,1], x0 unrestricted, k >= 0",
    "jnd_definition": "Half distance between crossings at y_high=.75*upper and y_low=.25*(1-lower)",
    "eligible_participants": {metric: int(eligibility.loc[eligibility.metric.eq(metric), "eligible"].sum())
                              for metric in METRICS},
    "participant_mean": "(M_ASL20 + M_ASL40) / 2 for each metric independently",
    "signed_delta": "M_ASL - participant mean",
    "order": "Control ascending Participant ID, then ASD ascending Participant ID; same order for both metrics",
    "participant_order": participants.Participant.astype(int).tolist(),
    "colors": {str(asl): to_hex(color) for asl, color in COLORS.items()},
    "markers": MARKERS, "mean_mark_colors": {group: to_hex(color) for group, color in MEAN_COLORS.items()},
    "summary_sd_ddof": 0,
    "figure_inches": list(FIGSIZE), "png_dpi": PNG_DPI,
    "figures": figure_rows, "plotted_values": len(drawn_points),
    "mean_marks": len(drawn_means), "summary_bars": len(drawn_bars),
    "right_bar_definition": "Control and ASD separately: group mean +/- 1 population SD across eligible participant means or signed ASL deviations",
}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"choices": len(trials), "condition_points": len(counts),
                  "fit_count": len(fits), "eligible_participants": manifest["eligible_participants"],
                  "figure_pngs": [row["png"] for row in figure_rows]}, indent=2))
