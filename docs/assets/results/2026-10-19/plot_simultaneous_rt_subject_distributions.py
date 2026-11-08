"""Per-participant psychometric, RTD, chronometric, CDF, and RT-quantile pages.

Run .venv/bin/python ASD_Human_Data/plot_simultaneous_rt_subject_distributions.py
from the repository root. The psychometric curves reuse the published human
four-parameter fits; all other calculations use the extracted RT trials.
"""

# %% Imports and editable parameters
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


HERE = Path(__file__).resolve().parent
INPUT = HERE / "simultaneous_rt_trials.csv"
EXTRACTION = HERE / "simultaneous_rt_extraction.json"
FIT_DIR = HERE / "simultaneous_rt_metric_deviations"
OUTPUT = HERE / "simultaneous_rt_subject_distributions"
ASLS = [20, 40]
ABS_DELTAS = [.1, .3, .65, 1.3, 2]
SIGNED_DELTAS = [-2, -1.3, -.65, -.3, -.1, .1, .3, .65, 1.3, 2]
COLORS = {20: "tab:blue", 40: "tab:orange"}
MARKERS = {20: "o", 40: "s"}
DELTA_STYLES = dict(zip(ABS_DELTAS, ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1))]))
DELTA_COLORS = dict(zip(ABS_DELTAS, [to_hex(c) for c in plt.cm.viridis(np.linspace(.08, .86, 5))]))
PLOT_QUANTILES = np.array([.1, .3, .5, .7, .9])
FINE_QUANTILES = np.arange(.01, 1, .01)
Q_COLORS = [to_hex(c) for c in plt.cm.plasma(np.linspace(.08, .86, 5))]
Q_MARKERS = ["o", "s", "^", "D", "v"]
RT_BIN_S = .020
CDF_BIN_S = .001
RT_LIMIT_S = 2.5  # All observed RTs are below this edge; no trial is excluded.
RTD_DISPLAY_S = 1.5
CDF_DISPLAY_S = 1.5
CUTOFF_WINDOW_BINS = 10
CUTOFF_RISE = .005
REFERENCE_ASL = 40
PNG_DPI = 150
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Input identity and the exact denominator
extraction = json.loads(EXTRACTION.read_text())
input_sha = hashlib.sha256(INPUT.read_bytes()).hexdigest()
assert input_sha == extraction["output_sha256"]
assert extraction["selection"] == "Condition == 'RT'"
trials = pd.read_csv(INPUT, float_precision="round_trip")
assert len(trials) == 11400 and trials.Participant.nunique() == 19
assert trials.Condition.eq("RT").all() and trials.Valid.eq(1).all()
assert trials.Choice.isin([-1, 1]).all() and np.isfinite(trials.RT_s).all()
assert trials.RT_s.between(0, RT_LIMIT_S, inclusive="left").all()
assert trials.groupby("Participant").size().eq(600).all()
assert trials.groupby(["Participant", "ASL_cd"]).size().eq(300).all()
assert trials.GroupID.eq(trials.Group.map({"Control": 0, "ASD": 1})).all()
trials["delta_key"] = trials.Delta_dB.round(2)
trials["abs_delta"] = trials.delta_key.abs().round(2)
assert sorted(trials.delta_key.unique()) == SIGNED_DELTAS
assert sorted(trials.abs_delta.unique()) == ABS_DELTAS
participants = trials[["Participant", "Group", "GroupID"]].drop_duplicates().sort_values(["GroupID", "Participant"])
assert participants.Group.tolist() == ["Control"] * 11 + ["ASD"] * 8

published_counts = pd.read_csv(FIT_DIR / "condition_counts.csv")
published_fits = pd.read_csv(FIT_DIR / "fits.csv").set_index(["Participant", "ASL_cd"])
measured = trials.groupby(["Participant", "Group", "GroupID", "ASL_cd", "delta_key"], as_index=False).agg(
    trials=("Choice", "size"), right_trials=("Choice", lambda x: int(x.eq(1).sum())))
measured["left_trials"] = measured.trials - measured.right_trials
measured["p_right"] = measured.right_trials / measured.trials
compare = measured.merge(published_counts, on=["Participant", "Group", "GroupID", "ASL_cd", "delta_key"],
                         suffixes=("", "_published"), validate="one_to_one")
assert len(compare) == 380 and measured.trials.eq(30).all()
for field in ["trials", "right_trials", "left_trials", "p_right"]:
    np.testing.assert_allclose(compare[field], compare[f"{field}_published"], atol=1e-12, rtol=0)
assert len(published_fits) == 38 and published_fits.fit_status.eq("ok").all()
measured.to_csv(OUTPUT / "psychometric_points.csv", index=False)


# %% Full-range RTDs, pooled-sign chronometrics, and 1 ms CDFs
rt_edges = np.arange(0, RT_LIMIT_S + RT_BIN_S / 2, RT_BIN_S)
cdf_edges = np.arange(0, RT_LIMIT_S + CDF_BIN_S / 2, CDF_BIN_S)
cdf_centers = (cdf_edges[:-1] + cdf_edges[1:]) / 2
rtd_rows, chrono_rows, cdf_rows, quantile_rows = [], [], [], []
rtd = {}
cdfs = {}
quantiles = {}
for (participant, asl), group in trials.groupby(["Participant", "ASL_cd"]):
    values = group.RT_s.to_numpy()
    bins, _ = np.histogram(values, rt_edges)
    assert int(bins.sum()) == len(values)
    density = bins / (len(values) * RT_BIN_S)
    rtd[(participant, asl)] = density
    rtd_rows.append(pd.DataFrame({"Participant": participant, "ASL_cd": asl,
                                  "bin_left_s": rt_edges[:-1], "bin_right_s": rt_edges[1:],
                                  "count": bins, "density_per_s": density,
                                  "normalization_trials": len(values)}))
for (participant, asl, abs_delta), group in trials.groupby(["Participant", "ASL_cd", "abs_delta"]):
    values = group.RT_s.to_numpy()
    n = len(values)
    assert n == 60 and group.delta_key.gt(0).sum() == group.delta_key.lt(0).sum() == 30
    sd = values.std(ddof=1)
    chrono_rows.append({"Participant": participant, "ASL_cd": asl, "abs_delta": abs_delta,
                        "trials": n, "positive_delta_trials": 30, "negative_delta_trials": 30,
                        "mean_rt_s": values.mean(), "sample_sd_rt_s": sd, "sem_rt_s": sd / np.sqrt(n)})
    bins, _ = np.histogram(values, cdf_edges)
    assert int(bins.sum()) == n
    cdf = np.cumsum(bins) / n
    cdfs[(participant, asl, abs_delta)] = cdf
    cdf_rows.append(pd.DataFrame({"Participant": participant, "ASL_cd": asl, "abs_delta": abs_delta,
                                  "bin_left_s": cdf_edges[:-1], "bin_right_s": cdf_edges[1:],
                                  "count": bins, "cdf": cdf, "normalization_trials": n}))
    coarse = np.quantile(values, PLOT_QUANTILES, method="linear")
    fine = np.quantile(values, FINE_QUANTILES, method="linear")
    quantiles[(participant, asl, abs_delta)] = (coarse, fine)
    for q, value in zip(PLOT_QUANTILES, coarse):
        quantile_rows.append({"Participant": participant, "ASL_cd": asl, "abs_delta": abs_delta,
                              "quantile": q, "raw_rt_s": value, "trials": n})

rtd_table = pd.concat(rtd_rows, ignore_index=True)
chrono = pd.DataFrame(chrono_rows)
cdf_table = pd.concat(cdf_rows, ignore_index=True)
raw_q = pd.DataFrame(quantile_rows)
assert len(chrono) == 190 and len(raw_q) == 950 and len(cdfs) == 190
assert rtd_table["count"].sum() == cdf_table["count"].sum() == chrono.trials.sum() == 11400
rtd_table.to_csv(OUTPUT / "rtd_bins.csv", index=False)
chrono.to_csv(OUTPUT / "chronometric_points.csv", index=False)
cdf_table.to_csv(OUTPUT / "cdf_bins.csv.gz", index=False, compression="gzip")
raw_q.to_csv(OUTPUT / "raw_quantiles.csv", index=False)


# %% Pairwise |Delta CDF| cutoff and higher-ASL-reference quantile scaling
delta_curves, cutoff_rows, scale_rows, scaled_rows = {}, [], [], []
for participant in participants.Participant:
    for abs_delta in ABS_DELTAS:
        difference = np.abs(cdfs[(participant, 20, abs_delta)] - cdfs[(participant, 40, abs_delta)])
        delta_curves[(participant, abs_delta)] = difference
        previous = np.full(len(difference), np.nan)
        following = np.full(len(difference), np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(difference, CUTOFF_WINDOW_BINS).mean(axis=1)
        previous[CUTOFF_WINDOW_BINS:-CUTOFF_WINDOW_BINS] = windows[:len(difference) - 2 * CUTOFF_WINDOW_BINS]
        following[CUTOFF_WINDOW_BINS:-CUTOFF_WINDOW_BINS] = windows[CUTOFF_WINDOW_BINS + 1:]
        # Fractions of 60 trials can make an exact 0.005 rise differ by a few
        # floating-point ulps across equivalent summation orders.
        rises = np.round(following - previous, 12)
        hits = np.flatnonzero(rises >= CUTOFF_RISE)
        cutoff = float(cdf_centers[hits[0]]) if len(hits) else np.nan
        cutoff_rows.append({"Participant": participant, "abs_delta": abs_delta, "cutoff_s": cutoff,
                            "cutoff_status": "ok" if np.isfinite(cutoff) else "not_detected",
                            "rise_at_cutoff": float(rises[hits[0]]) if len(hits) else np.nan})
        reference_coarse, reference_fine = quantiles[(participant, REFERENCE_ASL, abs_delta)]
        other_coarse, other_fine = quantiles[(participant, 20, abs_delta)]
        mask = reference_fine >= cutoff if np.isfinite(cutoff) else np.zeros(len(FINE_QUANTILES), dtype=bool)
        status = "ok"
        factor = np.nan
        slope = np.nan
        first_difference = np.nan
        if not np.isfinite(cutoff):
            status = "missing_cutoff"
        elif mask.sum() < 2:
            status = "insufficient_fit_points"
        else:
            x = reference_fine[mask] - cutoff
            y = (other_fine - reference_fine)[mask]
            first_difference = float(y[0])
            denominator = float(x @ x)
            if denominator <= 0:
                status = "zero_reference_range"
            else:
                slope = float(x @ (y - first_difference) / denominator)
                factor = 1 + slope
                if not np.isfinite(factor) or factor <= 0:
                    status = "invalid_scale_factor"
        scale_rows.append({"Participant": participant, "abs_delta": abs_delta,
                           "reference_ASL_cd": REFERENCE_ASL, "target_ASL_cd": 20,
                           "cutoff_s": cutoff, "fit_points": int(mask.sum()),
                           "first_difference_s": first_difference, "slope_m": slope,
                           "scale_factor": factor, "fit_status": status})
        transformed = (np.where(other_coarse > cutoff,
                        cutoff + (other_coarse - cutoff) / factor, other_coarse)
                       if status == "ok" else np.full_like(other_coarse, np.nan))
        for q, before, after in zip(PLOT_QUANTILES, other_coarse, transformed):
            scaled_rows.append({"Participant": participant, "ASL_cd": 20, "abs_delta": abs_delta,
                                "quantile": q, "raw_rt_s": before, "scaled_rt_s": after, "fit_status": status})
        for q, before in zip(PLOT_QUANTILES, reference_coarse):
            scaled_rows.append({"Participant": participant, "ASL_cd": REFERENCE_ASL, "abs_delta": abs_delta,
                                "quantile": q, "raw_rt_s": before, "scaled_rt_s": before,
                                "fit_status": "reference"})

cutoffs = pd.DataFrame(cutoff_rows)
scales = pd.DataFrame(scale_rows)
scaled = pd.DataFrame(scaled_rows)
assert len(cutoffs) == len(scales) == 95 and len(scaled) == 950
cutoffs.to_csv(OUTPUT / "cdf_cutoffs.csv", index=False)
scales.to_csv(OUTPUT / "quantile_scaling_fits.csv", index=False)
scaled.to_csv(OUTPUT / "scaled_quantiles.csv", index=False)


# %% One two-row page per participant, ordered Control then ASD
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
pdf_path = OUTPUT / "human_simultaneous_rt_subject_distributions.pdf"
page_rows = []
with PdfPages(pdf_path, metadata={"Title": "Human simultaneous RT: individual psychometric and RT distributions"}) as pdf:
    for page, member in enumerate(participants.itertuples(index=False), 1):
        participant = int(member.Participant)
        fig = plt.figure(figsize=(18.5, 8.9))
        fig.suptitle(f"Participant {participant} | {member.Group} | simultaneous RT | page {page}/19", fontsize=16, y=.985)
        fig.text(.5, .947, "600 valid choices and RTs | ASL 20/40 cd/m² | signed visual difference is right minus left dB",
                 ha="center", fontsize=10)
        fig.legend(handles=[Line2D([], [], color=COLORS[a], marker=MARKERS[a], label=f"ASL {a} cd/m²") for a in ASLS],
                   loc="upper center", bbox_to_anchor=(.5, .925), ncol=2, frameon=False)
        upper = [fig.add_axes(box) for box in ([.05, .565, .34, .31], [.43, .565, .34, .31], [.82, .592, .13, .27])]
        psy_ax, rt_ax, chrono_ax = upper
        smooth = np.linspace(-2.2, 2.2, 500)
        for asl in ASLS:
            points = measured.loc[measured.Participant.eq(participant) & measured.ASL_cd.eq(asl)].sort_values("delta_key")
            psy_ax.scatter(points.delta_key, points.p_right, color=COLORS[asl], marker=MARKERS[asl], s=32, zorder=3)
            fit = published_fits.loc[(participant, asl)]
            psy_ax.plot(smooth, fit.lower + (fit.upper-fit.lower)*expit(fit.k*(smooth-fit.x0)), color=COLORS[asl], lw=1.7)
            rt_ax.stairs(rtd[(participant, asl)], rt_edges, color=COLORS[asl], lw=1.5, fill=False)
            part_chrono = chrono.loc[chrono.Participant.eq(participant) & chrono.ASL_cd.eq(asl)].sort_values("abs_delta")
            x = np.arange(5) + (-.1 if asl == 20 else .1)
            chrono_ax.errorbar(x, part_chrono.mean_rt_s, yerr=part_chrono.sem_rt_s,
                               fmt=MARKERS[asl], color=COLORS[asl], ls="None", ms=4.5, capsize=2)
        psy_ax.set(title="Psychometric | published four-parameter fits", xlabel="Signed ΔdB (right − left)", ylabel="P(right)",
                   xlim=(-2.2, 2.2), ylim=(0, 1), xticks=[-2,-1,0,1,2])
        rt_ax.set(title="RTD | 20 ms bins, full-range area = 1", xlabel="RT from stimulus (s)", ylabel="Density (s⁻¹)", xlim=(0, RTD_DISPLAY_S))
        rt_ax.set_ylim(0, max(float(rtd[(participant, a)][:int(RTD_DISPLAY_S/RT_BIN_S)].max()) for a in ASLS)*1.12)
        chrono_ax.set(title="Chronometric\npooled ±ΔdB", xlabel="|ΔdB|", ylabel="Mean RT ± SEM (s)",
                      xlim=(-.45, 4.45), xticks=np.arange(5), xticklabels=[str(x) for x in ABS_DELTAS])
        chrono_ax.set_ylim(0, max(float((chrono.loc[chrono.Participant.eq(participant), "mean_rt_s"] +
                                        chrono.loc[chrono.Participant.eq(participant), "sem_rt_s"]).max())*1.15, .5))
        lower = [fig.add_axes([.045 + i*.198, .16, .165, .29]) for i in range(5)]
        cdf_ax, delta_ax, q20_ax, q40_ax, overlay_ax = lower
        for abs_delta in ABS_DELTAS:
            for asl in ASLS:
                cdf_ax.plot(cdf_centers, cdfs[(participant, asl, abs_delta)], color=COLORS[asl],
                            ls=DELTA_STYLES[abs_delta], lw=1.1)
            delta_ax.plot(cdf_centers, delta_curves[(participant, abs_delta)], color=DELTA_COLORS[abs_delta], lw=1.2)
            cutoff = cutoffs.loc[cutoffs.Participant.eq(participant) & cutoffs.abs_delta.eq(abs_delta), "cutoff_s"].iloc[0]
            if np.isfinite(cutoff):
                delta_ax.axvline(cutoff, color=DELTA_COLORS[abs_delta], ls="--", lw=.8)
        cdf_ax.set(title="CDFs | 1 ms bins", xlabel="RT (s)", ylabel="P(RT ≤ t)", xlim=(0, CDF_DISPLAY_S), ylim=(0, 1))
        delta_ax.set(title="|ΔCDF| | first-rise cutoffs", xlabel="RT (s)", ylabel="|F20 − F40|", xlim=(0, CDF_DISPLAY_S))
        qmax = np.nanmax(scaled.loc[scaled.Participant.eq(participant), ["raw_rt_s", "scaled_rt_s"]].to_numpy())
        for asl, ax in [(20, q20_ax), (40, q40_ax)]:
            for qi, q in enumerate(PLOT_QUANTILES):
                values = [quantiles[(participant, asl, a)][0][qi] for a in ABS_DELTAS]
                ax.plot(ABS_DELTAS, values, color=Q_COLORS[qi], marker=Q_MARKERS[qi], lw=1, ms=3.5)
            ax.set_title(f"Raw quantiles | ASL {asl}")
            ax.set_ylabel("RT quantile (s)")
        part_scaled = scaled.loc[scaled.Participant.eq(participant)]
        for asl in ASLS:
            for qi, q in enumerate(PLOT_QUANTILES):
                curve = part_scaled.loc[part_scaled.ASL_cd.eq(asl) & part_scaled["quantile"].eq(q)].sort_values("abs_delta")
                overlay_ax.plot(curve.abs_delta, curve.scaled_rt_s, color=COLORS[asl],
                                marker=Q_MARKERS[qi], lw=1, ms=3.5)
        overlay_ax.set(title="Scaled | ASL 40 reference", ylabel="Scaled RT quantile (s)")
        for ax in [q20_ax, q40_ax, overlay_ax]:
            ax.set(xscale="log", xlabel="|ΔdB|", xlim=(.08, 2.3), ylim=(0, max(.5, np.ceil(qmax*1.08/.1)*.1)))
            ax.set_xticks(ABS_DELTAS, [str(x) for x in ABS_DELTAS])
            ax.xaxis.set_minor_locator(plt.NullLocator())
        for ax in upper + lower:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=.18)
            ax.tick_params(labelsize=8)
        fig.legend(handles=[Line2D([], [], color=DELTA_COLORS[a], linestyle=DELTA_STYLES[a], label=f"|Δ| {a}") for a in ABS_DELTAS],
                   loc="center left", bbox_to_anchor=(.043,.505), ncol=5, frameon=False, fontsize=8)
        fig.text(.043, .477, "CDF line style / |ΔCDF| color", fontsize=8)
        fig.legend(handles=[Line2D([], [], color=Q_COLORS[i], marker=Q_MARKERS[i], label=f"{q:.0%}")
                            for i, q in enumerate(PLOT_QUANTILES)],
                   loc="center right", bbox_to_anchor=(.96,.505), ncol=5, frameon=False, fontsize=8)
        fig.text(.5, .08, "CDF and quantiles: all RTs; CDF view 0–1.5 s | cutoff: first 10 ms forward-minus-backward |ΔCDF| rise ≥ 0.005",
                 ha="center", fontsize=9)
        fig.text(.5, .053, "Scaling: ASL 40 reference; fit 1–99% reference quantiles above cutoff, without the animal 0.3 s limit",
                 ha="center", fontsize=9)
        fig.text(.5, .025, "RTD uses all signed ΔdB trials per ASL; chronometric, CDF, and quantiles pool raw +Δ and −Δ trials within |ΔdB|",
                 ha="center", fontsize=9)
        png = OUTPUT / f"human_rt_subject_{participant:02d}.png"
        pdf.savefig(fig)
        fig.savefig(png, dpi=PNG_DPI, facecolor="white")
        plt.close(fig)
        page_rows.append({"page": page, "Participant": participant, "Group": member.Group,
                          "valid_choices": 600, "finite_rt_trials": 600,
                          "png": png.name, "cutoffs_found": int(cutoffs.loc[cutoffs.Participant.eq(participant), "cutoff_status"].eq("ok").sum()),
                          "scaling_fits_ok": int(scales.loc[scales.Participant.eq(participant), "fit_status"].eq("ok").sum())})
        print(f"Page {page}/19: Participant {participant} ({member.Group})", flush=True)

pages = pd.DataFrame(page_rows)
pages.to_csv(OUTPUT / "participant_pages.csv", index=False)
bookmarked_pdf = OUTPUT / "subject_distributions_bookmarked.pdf"
with fitz.open(pdf_path) as pdf:
    pdf.set_toc([[1, f"{row.Group} | Participant {row.Participant}", int(row.page)]
                 for row in pages.itertuples(index=False)])
    pdf.save(bookmarked_pdf)
bookmarked_pdf.replace(pdf_path)
manifest = {"status": "passed", "source_csv": str(INPUT.relative_to(HERE.parent)), "source_sha256": input_sha,
            "selection": "Condition == 'RT'; Valid == 1; both choices; no extra RT cutoff",
            "participants": 19, "groups": {"Control": 11, "ASD": 8}, "trials": 11400,
            "asls_cd_m2": ASLS, "signed_delta_dB": SIGNED_DELTAS,
            "rt_field": "RT_s", "rt_max_observed_s": float(trials.RT_s.max()),
            "rtd_bin_s": RT_BIN_S, "cdf_bin_s": CDF_BIN_S,
            "rtd_display_s": [0, RTD_DISPLAY_S], "cdf_display_s": [0, CDF_DISPLAY_S],
            "cdf_cutoff_rule": "first forward 10-bin mean minus backward 10-bin mean >= 0.005 after rounding rise to 12 decimals, excluding current bin",
            "scaling_reference_asl_cd_m2": REFERENCE_ASL,
            "scaling_fit_quantiles": "1-99%, linear interpolation; reference quantile >= cutoff; no upper RT fit limit",
            "scaling_rule": "x=Q40-t0; y=Q20-Q40; m=sum(x*(y-y_first))/sum(x*x); factor=1+m; Q20_scaled=t0+(Q20-t0)/factor above t0",
            "psychometric_fit_source": str((FIT_DIR / "fits.csv").relative_to(HERE.parent)),
            "psychometric_fits_sha256": hashlib.sha256((FIT_DIR / "fits.csv").read_bytes()).hexdigest(),
            "psychometric_points": len(measured), "chronometric_points": len(chrono),
            "cdf_conditions": len(cdfs), "cutoffs_found": int(cutoffs.cutoff_status.eq("ok").sum()),
            "scaling_fits_ok": int(scales.fit_status.eq("ok").sum()), "pdf": pdf_path.name,
            "pages": pages.to_dict("records")}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({k: manifest[k] for k in ["trials", "cdf_conditions", "cutoffs_found", "scaling_fits_ok", "pdf"]}, indent=2))
