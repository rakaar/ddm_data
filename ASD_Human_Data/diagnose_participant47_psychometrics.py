"""Compare Participant 47's measured psychometrics under three fit rules.

Run with .venv/bin/python ASD_Human_Data/diagnose_participant47_psychometrics.py.
This diagnostic does not replace the published JND/slope analysis.
"""

# %% Inputs and editable parameters
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.special import expit

HERE = Path(__file__).resolve().parent
INPUT = HERE / "simultaneous_rt_trials.csv"
OUTPUT = HERE / "simultaneous_rt_metric_deviations"
PARTICIPANT = 47
ASLS = [20, 40]
COLORS = {20: "tab:blue", 40: "tab:green"}
SIGNED_DELTAS = [-2, -1.3, -.65, -.3, -.1, .1, .3, .65, 1.3, 2]
EXTRACTION = json.loads((HERE / "simultaneous_rt_extraction.json").read_text())
INPUT_SHA256 = hashlib.sha256(INPUT.read_bytes()).hexdigest()
assert INPUT_SHA256 == EXTRACTION["output_sha256"]


# %% Reconstruct the measured points from individual RT choices
trials = pd.read_csv(INPUT, float_precision="round_trip")
trials = trials.loc[trials.Participant.eq(PARTICIPANT) & trials.Condition.eq("RT") & trials.Valid.eq(1)].copy()
assert len(trials) == 600 and trials.Choice.isin([-1, 1]).all()
trials["difference"] = trials.Delta_dB.round(2)
trials["right"] = trials.Choice.eq(1)
counts = trials.groupby(["ASL_cd", "difference"], as_index=False).agg(
    trials=("right", "size"), right_trials=("right", "sum"))
counts["p_right"] = counts.right_trials / counts.trials
counts["binomial_sem"] = np.sqrt(counts.p_right * (1 - counts.p_right) / counts.trials)
assert len(counts) == 20 and counts.trials.eq(30).all()
assert set(counts.ASL_cd) == set(ASLS)
assert all(counts.loc[counts.ASL_cd.eq(asl), "difference"].tolist() == SIGNED_DELTAS for asl in ASLS)


# %% Current four-parameter least squares, alternative binomial likelihood, and two-parameter least squares
def four_parameter(x, upper, lower, x0, k):
    return lower + (upper - lower) * expit(k * (x - x0))


def two_parameter(x, pse, k):
    return expit(k * (x - pse))


fit_rows = []
curves = {}
grid = np.linspace(-2.2, 2.2, 441)
for asl in ASLS:
    observed = counts.loc[counts.ASL_cd.eq(asl)].sort_values("difference")
    x = observed.difference.to_numpy(dtype=float)
    probability = observed.p_right.to_numpy(dtype=float)
    right = observed.right_trials.to_numpy(dtype=float)
    n = observed.trials.to_numpy(dtype=float)
    initial = [float(probability.max()), float(probability.min()), float(np.median(x)), .1]
    least_squares, _ = curve_fit(
        four_parameter, x, probability, p0=initial,
        bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]), maxfev=10000)

    def binomial_nll(parameters):
        upper, lower, x0, k = parameters
        if upper <= lower:
            return 1e6 + 1e6 * (lower - upper)
        fitted = np.clip(four_parameter(x, *parameters), 1e-12, 1 - 1e-12)
        return -float(np.sum(right * np.log(fitted) + (n - right) * np.log1p(-fitted)))

    starts = [least_squares, [.95, .05, 0, 4]]
    optimizations = [minimize(binomial_nll, start, method="L-BFGS-B",
                              bounds=[(0, 1), (0, 1), (None, None), (1e-6, None)])
                     for start in starts]
    best = min(optimizations, key=lambda result: result.fun)
    assert best.success and best.x[0] > best.x[1]
    two_least_squares, _ = curve_fit(
        two_parameter, x, probability, p0=[0, 3],
        bounds=([-np.inf, 0], [np.inf, np.inf]), maxfev=10000)

    for model, parameters in [("four_parameter_unweighted_LS", least_squares),
                              ("four_parameter_binomial_MLE", best.x),
                              ("two_parameter_unweighted_LS", two_least_squares)]:
        if model.startswith("four_parameter"):
            upper, lower, x0, k = parameters
            inverse = lambda target: x0 - np.log((upper - lower) / (target - lower) - 1) / k
            pse = inverse(.5)
            high, low = .75 * upper, .25 * (1 - lower)
            jnd = (inverse(high) - inverse(low)) / 2 if lower < low < high < upper else np.nan
            fitted = four_parameter(x, *parameters)
            curves[(model, asl)] = four_parameter(grid, *parameters)
        else:
            pse, k = parameters
            upper, lower, x0 = 1.0, 0.0, pse
            jnd = np.log(3) / k
            fitted = two_parameter(x, *parameters)
            curves[(model, asl)] = two_parameter(grid, *parameters)
        clipped = np.clip(fitted, 1e-12, 1 - 1e-12)
        fit_rows.append({"Participant": PARTICIPANT, "ASL_cd": asl, "model": model,
                         "input_sha256": INPUT_SHA256,
                         "trials": int(n.sum()), "upper": upper, "lower": lower,
                         "x0": x0, "k": k, "PSE_at_P_right_0_5": pse,
                         "JND_half_crossing_distance": jnd,
                         "unweighted_probability_SSE": float(np.sum((fitted - probability) ** 2)),
                         "binomial_negative_log_likelihood": float(-np.sum(
                             right * np.log(clipped) + (n - right) * np.log1p(-clipped)))})

fits = pd.DataFrame(fit_rows)
published_fits = pd.read_csv(OUTPUT / "fits.csv")
for asl in ASLS:
    diagnostic = fits.loc[fits.ASL_cd.eq(asl) & fits.model.eq("four_parameter_unweighted_LS")].iloc[0]
    published = published_fits.loc[published_fits.Participant.eq(PARTICIPANT) &
                                   published_fits.ASL_cd.eq(asl)].iloc[0]
    np.testing.assert_allclose([diagnostic.upper, diagnostic.lower, diagnostic.x0, diagnostic.k,
                                diagnostic.JND_half_crossing_distance],
                               [published.upper, published.lower, published.x0, published.k,
                                published.jnd], rtol=0, atol=1e-6)
counts.to_csv(OUTPUT / "participant47_condition_counts.csv", index=False)
fits.to_csv(OUTPUT / "participant47_fit_rule_comparison.csv", index=False)


# %% Same raw points and error bars in every panel: only the fit rule changes
panels = [
    ("four_parameter_unweighted_LS", "Current: four-parameter unweighted fit"),
    ("four_parameter_binomial_MLE", "Alternative: four-parameter binomial fit"),
    ("two_parameter_unweighted_LS", "Alternative: two-parameter unweighted fit"),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), sharex=True, sharey=True)
for ax, (model, title) in zip(axes, panels):
    for asl in ASLS:
        measured = counts.loc[counts.ASL_cd.eq(asl)].sort_values("difference")
        ax.errorbar(measured.difference, measured.p_right, yerr=measured.binomial_sem,
                    marker="o", markersize=5, linestyle="none", capsize=2,
                    color=COLORS[asl], alpha=.75, zorder=3)
        ax.plot(grid, curves[(model, asl)], color=COLORS[asl], linewidth=2)
        row = fits.loc[fits.model.eq(model) & fits.ASL_cd.eq(asl)].iloc[0]
        ax.text(.04, .95 if asl == 20 else .87,
                f"ASL {asl}: PSE {row.PSE_at_P_right_0_5:+.2f} | JND {row.JND_half_crossing_distance:.2f}",
                transform=ax.transAxes, color=COLORS[asl], ha="left", va="top", fontsize=10)
    ax.axvline(0, color="0.75", linestyle="--", linewidth=1)
    ax.axhline(.5, color="0.75", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-.03, 1.06)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("Signed ΔdB (right − left)")
    ax.grid(alpha=.12)
axes[0].set_ylabel("P(right)")
fig.suptitle("Participant 47 | 600 simultaneous RT choices | 30 per signed difference and ASL", fontsize=15)
fig.text(.5, .055, "Points and binomial SEM are identical in all panels. PSE is the P(right)=0.5 crossing; JND uses half the fitted crossing distance at 0.75 × upper and 0.25 × (1 − lower).",
         ha="center", fontsize=9)
fig.text(.5, .025, "Friend's screenshot labels: blue PSE +0.14 / JND 0.28; green PSE +0.05 / JND 0.28. Its fitting method and ASL color mapping are unconfirmed.",
         ha="center", fontsize=9)
fig.subplots_adjust(left=.07, right=.985, top=.84, bottom=.18, wspace=.09)
png = OUTPUT / "participant47_psychometric_fit_comparison.png"
pdf = OUTPUT / "participant47_psychometric_fit_comparison.pdf"
fig.savefig(png, dpi=180)
fig.savefig(pdf)
plt.close(fig)
print(fits[["ASL_cd", "model", "PSE_at_P_right_0_5", "JND_half_crossing_distance",
            "unweighted_probability_SSE", "binomial_negative_log_likelihood"]].to_string(index=False))
print(f"Saved {png} and {pdf}")


# %% Standalone view of the exact published fit, for direct comparison with other plots
fig, ax = plt.subplots(figsize=(7.5, 6.2))
for asl in ASLS:
    measured = counts.loc[counts.ASL_cd.eq(asl)].sort_values("difference")
    ax.errorbar(measured.difference, measured.p_right, yerr=measured.binomial_sem,
                marker="o", markersize=6, linestyle="none", capsize=3,
                color=COLORS[asl], alpha=.8, zorder=3)
    published = published_fits.loc[published_fits.Participant.eq(PARTICIPANT) &
                                   published_fits.ASL_cd.eq(asl)].iloc[0]
    parameters = [published.upper, published.lower, published.x0, published.k]
    ax.plot(grid, four_parameter(grid, *parameters), color=COLORS[asl], linewidth=2.3)
    pse = published.x0 - np.log((published.upper - published.lower) / (.5 - published.lower) - 1) / published.k
    ax.text(.04, .96 if asl == 20 else .90,
            f"ASL {asl} cd/m²: PSE {pse:+.2f} | JND {published.jnd:.2f}",
            transform=ax.transAxes, color=COLORS[asl], ha="left", va="top", fontsize=11)
ax.axvline(0, color="0.7", linestyle="--", linewidth=1)
ax.axhline(.5, color="0.7", linestyle="--", linewidth=1)
ax.set(xlim=(-2.2, 2.2), ylim=(-.03, 1.06), xlabel="Signed ΔdB (right − left)",
       ylabel="P(right)", title="Participant 47 | 600 RT trials | current analysis fit")
ax.set_xticks([-2, -1, 0, 1, 2])
ax.grid(alpha=.12)
fig.text(.5, .025, "30 choices per difference and ASL; error bars are binomial SEM.\nFour-parameter unweighted sigmoid; JND uses the saved animal-plot crossing rule.",
         ha="center", fontsize=9)
fig.subplots_adjust(left=.12, right=.97, top=.91, bottom=.18)
current_png = OUTPUT / "participant47_current_psychometric.png"
current_pdf = OUTPUT / "participant47_current_psychometric.pdf"
fig.savefig(current_png, dpi=180)
fig.savefig(current_pdf)
plt.close(fig)
print(f"Saved {current_png} and {current_pdf}")
