"""Pool raw human RT trials within Control/ASD and draw one four-row figure.

Run .venv/bin/python ASD_Human_Data/plot_simultaneous_rt_group_pooled_distributions.py.
All calculations follow the individual-participant pages, recomputed after
pooling trials. Psychometric curves fit the ten pooled probabilities.
"""

# %% Imports and editable parameters
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import expit

HERE = Path(__file__).resolve().parent
INPUT = HERE / "simultaneous_rt_trials.csv"
OUTPUT = HERE / "simultaneous_rt_group_pooled_distributions"
GROUPS = ["Control", "ASD"]
ASLS = [20, 40]
ABS_DELTAS = [.1, .3, .65, 1.3, 2]
SIGNED_DELTAS = [-2, -1.3, -.65, -.3, -.1, .1, .3, .65, 1.3, 2]
COLORS = {20: "tab:blue", 40: "tab:orange"}
MARKERS = {20: "o", 40: "s"}
GROUP_COLORS = {"Control": "tab:blue", "ASD": "tab:red"}
DELTA_STYLES = dict(zip(ABS_DELTAS, ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1))]))
DELTA_COLORS = dict(zip(ABS_DELTAS, [to_hex(c) for c in plt.cm.viridis(np.linspace(.08, .86, 5))]))
PLOT_QUANTILES = np.array([.1, .3, .5, .7, .9])
FINE_QUANTILES = np.arange(.01, 1, .01)
Q_COLORS = [to_hex(c) for c in plt.cm.plasma(np.linspace(.08, .86, 5))]
Q_MARKERS = ["o", "s", "^", "D", "v"]
RT_BIN_S = .020
CDF_BIN_S = .001
RT_LIMIT_S = 2.5  # Histogram coverage, not an added selection rule.
DISPLAY_RT_S = 1.5
CUTOFF_WINDOW_BINS = 10
CUTOFF_RISE = .005
REFERENCE_ASL = 40
PNG_DPI = 180
STEM = "human_simultaneous_rt_control_asd_pooled_distributions"
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Exact RT selection; pool participants and sessions within group
extraction = json.loads((HERE / "simultaneous_rt_extraction.json").read_text())
input_sha = hashlib.sha256(INPUT.read_bytes()).hexdigest()
assert input_sha == extraction["output_sha256"]
trials = pd.read_csv(INPUT, float_precision="round_trip")
assert len(trials) == 11400 and trials.Condition.eq("RT").all() and trials.Valid.eq(1).all()
assert trials.Choice.isin([-1, 1]).all() and np.isfinite(trials.RT_s).all()
assert trials.RT_s.between(0, RT_LIMIT_S, inclusive="left").all()
assert trials.groupby("Participant").size().eq(600).all()
assert trials.GroupID.eq(trials.Group.map({"Control": 0, "ASD": 1})).all()
trials["delta_key"] = trials.Delta_dB.round(2)
trials["abs_delta"] = trials.delta_key.abs().round(2)
assert sorted(trials.delta_key.unique()) == SIGNED_DELTAS
members = trials.groupby(["GroupID", "Group", "Participant"], as_index=False).agg(trials=("Choice", "size"))
summary = trials.groupby(["GroupID", "Group"], as_index=False).agg(
    participants=("Participant", "nunique"), trials=("Choice", "size"),
    right_trials=("Choice", lambda x: int(x.eq(1).sum())), rt_above_1_s=("RT_s", lambda x: int(x.gt(1).sum())))
assert summary.Group.tolist() == GROUPS
assert summary.participants.tolist() == [11, 8] and summary.trials.tolist() == [6600, 4800]
members.to_csv(OUTPUT / "group_membership.csv", index=False)
summary.to_csv(OUTPUT / "group_counts.csv", index=False)


# %% Refit pooled psychometrics using the established unweighted four-parameter rule
def sigmoid(x, upper, lower, x0, k):
    return lower + (upper - lower) * expit(k * (x - x0))


points = trials.groupby(["Group", "ASL_cd", "delta_key"], as_index=False).agg(
    trials=("Choice", "size"), right_trials=("Choice", lambda x: int(x.eq(1).sum())))
points["p_right"] = points.right_trials / points.trials
fit_rows = []
for (group, asl), condition in points.groupby(["Group", "ASL_cd"]):
    condition = condition.sort_values("delta_key")
    x, probability = condition.delta_key.to_numpy(), condition.p_right.to_numpy()
    initial = [float(probability.max()), float(probability.min()), float(np.median(x)), .1]
    parameters, _ = curve_fit(sigmoid, x, probability, p0=initial,
                             bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]), maxfev=10000)
    upper, lower, x0, k = parameters
    assert np.isfinite(parameters).all() and upper > lower and k > 0
    fit_rows.append({"Group": group, "ASL_cd": asl, "upper": upper, "lower": lower,
                     "x0": x0, "k": k, "trials": int(condition.trials.sum()),
                     "initial_parameters": json.dumps(initial), "fit_status": "ok",
                     "rmse": np.sqrt(np.mean((sigmoid(x, *parameters) - probability) ** 2))})
fits = pd.DataFrame(fit_rows)
assert len(points) == 40 and len(fits) == 4
points.to_csv(OUTPUT / "psychometric_points.csv", index=False)
fits.to_csv(OUTPUT / "psychometric_fits.csv", index=False)
fit_lookup = fits.set_index(["Group", "ASL_cd"])


# %% Pooled RTDs, chronometrics, CDFs, and raw quantiles
rt_edges = np.arange(0, RT_LIMIT_S + RT_BIN_S / 2, RT_BIN_S)
cdf_edges = np.arange(0, RT_LIMIT_S + CDF_BIN_S / 2, CDF_BIN_S)
cdf_centers = (cdf_edges[:-1] + cdf_edges[1:]) / 2
rtd, cdfs, quantiles = {}, {}, {}
rtd_rows, chrono_rows, cdf_rows, raw_rows, fine_rows = [], [], [], [], []
for (group, asl), condition in trials.groupby(["Group", "ASL_cd"]):
    values = condition.RT_s.to_numpy()
    counts = np.histogram(values, rt_edges)[0]
    assert counts.sum() == len(values)
    density = counts / (len(values) * RT_BIN_S)
    rtd[(group, asl)] = density
    rtd_rows.append(pd.DataFrame({"Group": group, "ASL_cd": asl, "bin_left_s": rt_edges[:-1],
                                  "bin_right_s": rt_edges[1:], "count": counts,
                                  "density_per_s": density, "normalization_trials": len(values)}))
for (group, asl, abs_delta), condition in trials.groupby(["Group", "ASL_cd", "abs_delta"]):
    values = condition.RT_s.to_numpy()
    n = len(values)
    assert n == (660 if group == "Control" else 480)
    chrono_rows.append({"Group": group, "ASL_cd": asl, "abs_delta": abs_delta, "trials": n,
                        "positive_delta_trials": int(condition.delta_key.gt(0).sum()),
                        "negative_delta_trials": int(condition.delta_key.lt(0).sum()),
                        "mean_rt_s": values.mean(), "sample_sd_rt_s": values.std(ddof=1),
                        "sem_rt_s": values.std(ddof=1) / np.sqrt(n)})
    counts = np.histogram(values, cdf_edges)[0]
    assert counts.sum() == n
    cdf = np.cumsum(counts) / n
    cdfs[(group, asl, abs_delta)] = cdf
    cdf_rows.append(pd.DataFrame({"Group": group, "ASL_cd": asl, "abs_delta": abs_delta,
                                  "bin_left_s": cdf_edges[:-1], "bin_right_s": cdf_edges[1:],
                                  "count": counts, "cdf": cdf, "normalization_trials": n}))
    coarse = np.quantile(values, PLOT_QUANTILES, method="linear")
    fine = np.quantile(values, FINE_QUANTILES, method="linear")
    quantiles[(group, asl, abs_delta)] = (coarse, fine)
    for q, value in zip(PLOT_QUANTILES, coarse):
        raw_rows.append({"Group": group, "ASL_cd": asl, "abs_delta": abs_delta,
                         "quantile": q, "raw_rt_s": value, "trials": n})
    for q, value in zip(FINE_QUANTILES, fine):
        fine_rows.append({"Group": group, "ASL_cd": asl, "abs_delta": abs_delta,
                          "quantile": q, "rt_s": value, "trials": n})
rtd_table = pd.concat(rtd_rows, ignore_index=True)
chrono = pd.DataFrame(chrono_rows)
cdf_table = pd.concat(cdf_rows, ignore_index=True)
raw_q = pd.DataFrame(raw_rows)
assert rtd_table["count"].sum() == cdf_table["count"].sum() == chrono.trials.sum() == 11400
rtd_table.to_csv(OUTPUT / "rtd_bins.csv", index=False)
chrono.to_csv(OUTPUT / "chronometric_points.csv", index=False)
cdf_table.to_csv(OUTPUT / "cdf_bins.csv.gz", index=False, compression="gzip")
raw_q.to_csv(OUTPUT / "raw_quantiles.csv", index=False)
pd.DataFrame(fine_rows).to_csv(OUTPUT / "fine_quantiles.csv", index=False)


# %% Fresh pooled CDF cutoffs and ASL-40-reference quantile scale factors
delta_curves, cutoff_rows, scale_rows, scaled_rows = {}, [], [], []
for group in GROUPS:
    for abs_delta in ABS_DELTAS:
        difference = np.abs(cdfs[(group, 20, abs_delta)] - cdfs[(group, 40, abs_delta)])
        delta_curves[(group, abs_delta)] = difference
        previous = np.full(len(difference), np.nan)
        following = np.full(len(difference), np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(difference, CUTOFF_WINDOW_BINS).mean(axis=1)
        previous[CUTOFF_WINDOW_BINS:-CUTOFF_WINDOW_BINS] = windows[:len(difference) - 2 * CUTOFF_WINDOW_BINS]
        following[CUTOFF_WINDOW_BINS:-CUTOFF_WINDOW_BINS] = windows[CUTOFF_WINDOW_BINS + 1:]
        hits = np.flatnonzero(np.round(following - previous, 12) >= CUTOFF_RISE)
        cutoff = float(cdf_centers[hits[0]]) if len(hits) else np.nan
        cutoff_rows.append({"Group": group, "abs_delta": abs_delta, "cutoff_s": cutoff,
                            "cutoff_status": "ok" if len(hits) else "not_detected"})
        reference_coarse, reference_fine = quantiles[(group, REFERENCE_ASL, abs_delta)]
        other_coarse, other_fine = quantiles[(group, 20, abs_delta)]
        mask = reference_fine >= cutoff if np.isfinite(cutoff) else np.zeros(len(FINE_QUANTILES), dtype=bool)
        factor, slope, first_difference = np.nan, np.nan, np.nan
        status = "missing_cutoff" if not np.isfinite(cutoff) else "insufficient_fit_points"
        if mask.sum() >= 2:
            x = reference_fine[mask] - cutoff
            y = (other_fine - reference_fine)[mask]
            first_difference = float(y[0])
            denominator = float(x @ x)
            if denominator > 0:
                slope = float(x @ (y - y[0]) / denominator)
                factor = 1 + slope
                status = "ok" if np.isfinite(factor) and factor > 0 else "invalid_scale_factor"
            else:
                status = "zero_reference_range"
        scale_rows.append({"Group": group, "abs_delta": abs_delta, "reference_ASL_cd": 40,
                           "target_ASL_cd": 20, "cutoff_s": cutoff, "fit_points": int(mask.sum()),
                           "slope_m": slope, "scale_factor": factor, "first_difference_s": first_difference,
                           "fit_status": status})
        transformed = (np.where(other_coarse > cutoff, cutoff + (other_coarse-cutoff)/factor, other_coarse)
                       if status == "ok" else np.full_like(other_coarse, np.nan))
        for asl, before, after, fit_status in [(20, other_coarse, transformed, status),
                                               (40, reference_coarse, reference_coarse, "reference")]:
            for q, original, adjusted in zip(PLOT_QUANTILES, before, after):
                scaled_rows.append({"Group": group, "ASL_cd": asl, "abs_delta": abs_delta,
                                    "quantile": q, "raw_rt_s": original, "scaled_rt_s": adjusted,
                                    "fit_status": fit_status})
cutoffs, scales, scaled = pd.DataFrame(cutoff_rows), pd.DataFrame(scale_rows), pd.DataFrame(scaled_rows)
cutoffs.to_csv(OUTPUT / "cdf_cutoffs.csv", index=False)
scales.to_csv(OUTPUT / "quantile_scaling_fits.csv", index=False)
scaled.to_csv(OUTPUT / "scaled_quantiles.csv", index=False)


# %% Control above ASD: each group has the same two-row layout and shared scales
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
fig = plt.figure(figsize=(18.5, 17.8))
chrono_ymax = max(.5, np.ceil((chrono.mean_rt_s + chrono.sem_rt_s).max() * 1.12 / .1) * .1)
quantile_ymax = max(.5, np.ceil(np.nanmax(scaled[["raw_rt_s", "scaled_rt_s"]].to_numpy()) * 1.08 / .1) * .1)
rtd_ymax = max(float(values[:int(DISPLAY_RT_S / RT_BIN_S)].max()) for values in rtd.values()) * 1.12
delta_ymax = max(float(values[cdf_centers <= DISPLAY_RT_S].max()) for values in delta_curves.values()) * 1.12
for group_index, group in enumerate(GROUPS):
    offset = .5 if group_index == 0 else 0
    identity = summary.loc[summary.Group.eq(group)].iloc[0]
    def add_panel(box):
        return fig.add_axes([box[0], offset + box[1]/2, box[2], box[3]/2])

    fig.text(.5, offset+.985/2, f"{group} | {int(identity.participants)} participants pooled | {int(identity.trials):,} RT trials",
             ha="center", va="top", fontsize=17, color=GROUP_COLORS[group], weight="bold")
    fig.text(.5, offset+.947/2, "Condition = RT | all recorded choices and RTs | pooled raw trials across participants and sessions",
             ha="center", fontsize=10)
    fig.legend(handles=[Line2D([], [], color=COLORS[a], marker=MARKERS[a], label=f"ASL {a} cd/m²") for a in ASLS],
               loc="upper center", bbox_to_anchor=(.5, offset+.925/2), ncol=2, frameon=False)
    upper = [add_panel(box) for box in ([.05,.565,.34,.31], [.43,.565,.34,.31], [.82,.592,.13,.27])]
    psy_ax, rt_ax, chrono_ax = upper
    smooth = np.linspace(-2.2, 2.2, 500)
    for asl in ASLS:
        p = points.loc[points.Group.eq(group) & points.ASL_cd.eq(asl)].sort_values("delta_key")
        fit = fit_lookup.loc[(group, asl)]
        psy_ax.scatter(p.delta_key, p.p_right, color=COLORS[asl], marker=MARKERS[asl], s=32, zorder=3)
        psy_ax.plot(smooth, sigmoid(smooth, fit.upper, fit.lower, fit.x0, fit.k), color=COLORS[asl], lw=1.7)
        rt_ax.stairs(rtd[(group, asl)], rt_edges, color=COLORS[asl], lw=1.5)
        c = chrono.loc[chrono.Group.eq(group) & chrono.ASL_cd.eq(asl)].sort_values("abs_delta")
        chrono_ax.errorbar(np.arange(5)+(-.1 if asl==20 else .1), c.mean_rt_s, yerr=c.sem_rt_s,
                           fmt=MARKERS[asl], color=COLORS[asl], ls="None", ms=4.5, capsize=2)
    psy_ax.set(title="Psychometric | fit to pooled P(right)", xlabel="Signed ΔdB (right − left)", ylabel="P(right)",
               xlim=(-2.2,2.2), ylim=(0,1), xticks=[-2,-1,0,1,2])
    rt_ax.set(title="RTD | 20 ms bins, full-range area = 1", xlabel="Stored RT_s (s)", ylabel="Density (s⁻¹)",
              xlim=(0,DISPLAY_RT_S), ylim=(0,rtd_ymax))
    chrono_ax.set(title="Chronometric\npooled ±ΔdB", xlabel="|ΔdB|", ylabel="Mean RT ± SEM (s)",
                  xlim=(-.45,4.45), ylim=(0,chrono_ymax), xticks=np.arange(5), xticklabels=[str(x) for x in ABS_DELTAS])
    lower = [add_panel([.045+i*.198,.16,.165,.29]) for i in range(5)]
    cdf_ax, delta_ax, q20_ax, q40_ax, overlay_ax = lower
    for abs_delta in ABS_DELTAS:
        for asl in ASLS:
            cdf_ax.plot(cdf_centers, cdfs[(group,asl,abs_delta)], color=COLORS[asl], ls=DELTA_STYLES[abs_delta], lw=1.1)
        delta_ax.plot(cdf_centers, delta_curves[(group,abs_delta)], color=DELTA_COLORS[abs_delta], lw=1.2)
        cutoff = cutoffs.loc[cutoffs.Group.eq(group)&cutoffs.abs_delta.eq(abs_delta),"cutoff_s"].iloc[0]
        if np.isfinite(cutoff):
            delta_ax.axvline(cutoff, color=DELTA_COLORS[abs_delta], ls="--", lw=.8)
    cdf_ax.set(title="CDFs | 1 ms bins", xlabel="RT (s)", ylabel="P(RT ≤ t)", xlim=(0,DISPLAY_RT_S), ylim=(0,1))
    delta_ax.set(title="|ΔCDF| | pooled cutoffs", xlabel="RT (s)", ylabel="|F20 − F40|",
                 xlim=(0,DISPLAY_RT_S), ylim=(0,max(.01,delta_ymax)))
    for asl, ax in [(20,q20_ax),(40,q40_ax)]:
        for qi,q in enumerate(PLOT_QUANTILES):
            ax.plot(ABS_DELTAS,[quantiles[(group,asl,a)][0][qi] for a in ABS_DELTAS],
                    color=Q_COLORS[qi],marker=Q_MARKERS[qi],lw=1,ms=3.5)
        ax.set_title(f"Raw quantiles | ASL {asl}")
        ax.set_ylabel("RT quantile (s)")
    for asl in ASLS:
        for qi,q in enumerate(PLOT_QUANTILES):
            curve=scaled.loc[scaled.Group.eq(group)&scaled.ASL_cd.eq(asl)&scaled["quantile"].eq(q)].sort_values("abs_delta")
            overlay_ax.plot(curve.abs_delta,curve.scaled_rt_s,color=COLORS[asl],marker=Q_MARKERS[qi],lw=1,ms=3.5)
    overlay_ax.set(title="Scaled | ASL 40 reference",ylabel="Scaled RT quantile (s)")
    failed=scales.loc[scales.Group.eq(group),"fit_status"].ne("ok").sum()
    if failed:
        overlay_ax.text(.02,.02,f"{failed}/5 ASL-20 scales unavailable",transform=overlay_ax.transAxes,fontsize=8)
    for ax in [q20_ax,q40_ax,overlay_ax]:
        ax.set(xscale="log",xlabel="|ΔdB|",xlim=(.08,2.3),ylim=(0,quantile_ymax))
        ax.set_xticks(ABS_DELTAS,[str(x) for x in ABS_DELTAS])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    for ax in upper+lower:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.18)
        ax.tick_params(labelsize=8)
    fig.legend(handles=[Line2D([],[],color=DELTA_COLORS[a],ls=DELTA_STYLES[a],label=f"|Δ| {a}") for a in ABS_DELTAS],
               loc="center left",bbox_to_anchor=(.043,offset+.505/2),ncol=5,frameon=False,fontsize=8)
    fig.text(.043,offset+.477/2,"CDF line style / |ΔCDF| color",fontsize=8)
    fig.legend(handles=[Line2D([],[],color=Q_COLORS[i],marker=Q_MARKERS[i],label=f"{q:.0%}") for i,q in enumerate(PLOT_QUANTILES)],
               loc="center right",bbox_to_anchor=(.96,offset+.505/2),ncol=5,frameon=False,fontsize=8)
    fig.text(.5,offset+.08/2,"All recorded RTs included | RTD/CDF view 0–1.5 s | Chronometric bars: pooled-trial sample SEM",ha="center",fontsize=9)
    fig.text(.5,offset+.053/2,"CDF cutoff: first 10 ms forward-minus-backward rise ≥ 0.005 | Scaling: ASL 40 reference, 1–99% quantiles above cutoff",ha="center",fontsize=9)
    fig.text(.5,offset+.025/2,"RTD pools all signed differences; chronometric, CDF, and quantiles pool +Δ and −Δ trials within |ΔdB|",ha="center",fontsize=9)
fig.savefig(OUTPUT/f"{STEM}.png",dpi=PNG_DPI,facecolor="white")
fig.savefig(OUTPUT/f"{STEM}.pdf",facecolor="white")
plt.close(fig)
manifest={"status":"passed","input_sha256":input_sha,"selection":"Condition == RT; Valid == 1; both choices; no extra RT cutoff",
          "pooling":"raw trials across participants and sessions within Group; recompute all statistics and fits",
          "groups":summary.to_dict("records"),"psychometric_fits":4,"psychometric_points":40,
          "cdf_conditions":20,"raw_quantiles":100,"cutoffs_found":int(cutoffs.cutoff_status.eq("ok").sum()),
          "scale_fits_ok":int(scales.fit_status.eq("ok").sum()),"rtd_bin_s":RT_BIN_S,"cdf_bin_s":CDF_BIN_S,
          "display_range_s":[0,DISPLAY_RT_S],"reference_ASL_cd":40,"scaling_fit":"1–99% reference quantiles >= cutoff; no upper RT limit",
          "chronometric_uncertainty":"sample SD(ddof=1)/sqrt(number of pooled trials)",
          "png":f"{STEM}.png","pdf":f"{STEM}.pdf"}
(OUTPUT/"validation.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(manifest,indent=2))
