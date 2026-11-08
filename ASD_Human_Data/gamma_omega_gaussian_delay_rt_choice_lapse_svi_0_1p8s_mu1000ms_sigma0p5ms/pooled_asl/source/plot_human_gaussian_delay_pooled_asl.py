# %%
"""Pool both ASLs in saved Gaussian-delay + choice + lapse predictions and data.

This is a plotting operation: combine conditional predictions with retained-trial
weights, leaving each ASL's fitted parameters and all fit artifacts unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/human_gaussian_delay_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# %% Editable plot settings and input/output locations.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIT = ROOT / "ASD_Human_Data/gamma_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms"
GROUPS = ["Control", "ASD"]
COLORS = {"Control": "#1976d2", "ASD": "#d62728"}
ABS_DELTA = [.1, .3, .65, 1.3, 2.]
EDGES = np.linspace(0., 1.8, 91)
STEM = "human_gaussian_delay_pooled_asl_rtd_psychometric"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT)
parser.add_argument("--output-dir", type=Path, default=None)
args = parser.parse_args()
fit = args.fit_dir.resolve()
out = args.output_dir.resolve() if args.output_dir else fit / "pooled_asl"
assert out != fit, "Use a separate output directory to preserve the source fit."
out.mkdir(parents=True, exist_ok=True)
print(f"Source fit: {fit}", flush=True)
print(f"Pooled figure and audit: {out}", flush=True)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# %% Save a preservation baseline before creating derived outputs.
baseline_path = out / "preservation_baseline.json"
if not baseline_path.exists():
    existing = [p for p in fit.rglob("*") if p.is_file() and not p.is_relative_to(out)]
    existing += [p for p in (ROOT / "docs/assets/results/2026-10-22").rglob("*") if p.is_file()]
    page = ROOT / "docs/results/2026-10-22.md"
    baseline = {
        "unchanged_files": {str(p): sha256(p) for p in existing},
        "result_page": str(page), "result_page_bytes": page.stat().st_size,
        "result_page_prefix_sha256": sha256(page),
    }
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")

# %% Read the validated, normalized curves at arithmetic physical posterior means.
config = json.loads((fit / "run_config.json").read_text())
previous_plot = json.loads((fit / "plot_manifest.json").read_text())
publication = json.loads((fit / "publication_manifest.json").read_text())
assert config["mode"] == "gaussian_delay_lapse"
assert config["rt_window_s"] == [0., 1.8]
assert previous_plot["curve_parameters"] == "arithmetic physical posterior means"
for name in ["condition_model_curves.csv", "psychometric_comparison.csv", "plot_manifest.json", "run_config.json"]:
    assert sha256(fit / name) == publication["assets"][name], name

trials = pd.read_csv(fit / "selected_trials.csv", float_precision="round_trip")
conditions = pd.read_csv(fit / "psychometric_comparison.csv", float_precision="round_trip")
curves = pd.read_csv(fit / "condition_model_curves.csv", float_precision="round_trip")
source = pd.read_csv(ROOT / "ASD_Human_Data/simultaneous_rt_trials.csv", float_precision="round_trip")
selected = source.loc[source.Condition.eq("RT") & source.Valid.eq(1) & source.RT_s.between(0., 1.8)]
pd.testing.assert_frame_equal(
    trials[source.columns].sort_values("source_row").reset_index(drop=True),
    selected.sort_values("source_row").reset_index(drop=True),
)
assert trials.source_row.is_unique and len(trials) == 11344
assert trials.groupby("Group").size().to_dict() == {"ASD": 4770, "Control": 6574}
assert trials.groupby("Group").Participant.nunique().to_dict() == {"ASD": 8, "Control": 11}
assert set(trials.ASL_cd) == {20, 40} and set(trials.Choice) == {-1, 1}
np.testing.assert_allclose(trials.signed_delta, trials.Delta_dB, atol=1e-12)

# Confirm each source component's stimulus mapping and empirical counts.
for c in conditions.itertuples():
    obs = trials.loc[trials.Group.eq(c.Group) & trials.ASL_cd.eq(c.ASL_cd)
                     & np.isclose(trials.signed_delta, c.signed_delta)]
    assert len(obs) == c.n and obs.Choice.eq(1).sum() == c.right
assert len(conditions) == 40
assert not conditions.duplicated(["Group", "ASL_cd", "signed_delta"]).any()

# %% Pool raw data and count-weighted predictions; never average model parameters.
histograms, mixtures, probabilities, weights, checks = [], [], [], [], []
for group in GROUPS:
    group_trials = trials.loc[trials.Group.eq(group)]
    group_conditions = conditions.loc[conditions.Group.eq(group)]
    for delta in ABS_DELTA:
        obs = group_trials.loc[np.isclose(group_trials.signed_delta.abs(), delta)]
        components = group_conditions.loc[np.isclose(group_conditions.signed_delta.abs(), delta)]
        assert len(components) == 4
        components = components.sort_values(["ASL_cd", "signed_delta"])
        component_weights = components.n.to_numpy() / len(obs)
        np.testing.assert_allclose(component_weights.sum(), 1., atol=1e-14)
        model = None
        for c, weight in zip(components.itertuples(), component_weights):
            component = curves.loc[curves.Group.eq(group) & curves.condition_index.eq(c.condition_index)].sort_values("time_s")
            assert component.ASL_cd.eq(c.ASL_cd).all()
            np.testing.assert_allclose(component.signed_delta, c.signed_delta, atol=1e-12)
            times = component.time_s.to_numpy()
            values = component[["left_density", "right_density", "density", "cdf"]].to_numpy()
            if model is None:
                model = np.zeros_like(values)
                reference_times = times
            np.testing.assert_array_equal(times, reference_times)
            model += weight * values
            weights.append(dict(panel="rtd", Group=group, pooled_delta=delta,
                                condition_index=c.condition_index, ASL_cd=c.ASL_cd,
                                signed_delta=c.signed_delta, component_n=c.n,
                                pooled_n=len(obs), weight=weight))
        np.testing.assert_allclose(times[[0, -1]], [0., 1.8], atol=1e-14)
        mass = float(np.trapz(model[:, 2], times))
        assert abs(mass - 1.) < 1e-6 and model[:, 2].min() >= 0
        np.testing.assert_allclose(model[:, 0] + model[:, 1], model[:, 2], atol=1e-12)
        np.testing.assert_allclose(model[[0, -1], 3], [0., 1.], atol=1e-6)
        counts = np.histogram(obs.RT_s, bins=EDGES)[0]
        assert counts.sum() == len(obs)
        histograms.append(pd.DataFrame(dict(Group=group, abs_delta=delta,
            left_s=EDGES[:-1], right_s=EDGES[1:], count=counts, n=len(obs),
            density=counts / (len(obs) * np.diff(EDGES)))))
        mixtures.append(pd.DataFrame(dict(Group=group, abs_delta=delta, time_s=times,
            left_density=model[:, 0], right_density=model[:, 1], density=model[:, 2], cdf=model[:, 3])))
        checks.append(dict(Group=group, abs_delta=delta, n=len(obs), model_mass=mass,
                           component_count=4, weight_sum=float(component_weights.sum())))
    for delta, components in group_conditions.groupby("signed_delta", sort=True):
        obs = group_trials.loc[np.isclose(group_trials.signed_delta, delta)]
        assert len(components) == 2
        component_weights = components.n.to_numpy() / len(obs)
        prediction = float(component_weights @ components.model_p_right.to_numpy())
        right = int(obs.Choice.eq(1).sum())
        probabilities.append(dict(Group=group, signed_delta=delta, n=len(obs), right=right,
                                  empirical_p_right=right / len(obs), model_p_right=prediction))
        for c, weight in zip(components.itertuples(), component_weights):
            weights.append(dict(panel="psychometric", Group=group, pooled_delta=delta,
                                condition_index=c.condition_index, ASL_cd=c.ASL_cd,
                                signed_delta=c.signed_delta, component_n=c.n,
                                pooled_n=len(obs), weight=weight))

hist = pd.concat(histograms, ignore_index=True)
model_curves = pd.concat(mixtures, ignore_index=True)
psychometric = pd.DataFrame(probabilities)
tables = {
    "pooled_rtd_histograms.csv": hist,
    "pooled_model_curves.csv": model_curves,
    "pooled_psychometric.csv": psychometric,
    "mixture_weights.csv": pd.DataFrame(weights),
    "pooling_checks.csv": pd.DataFrame(checks),
    "trial_membership.csv": trials[["source_row", "Participant", "Group", "ASL_cd", "signed_delta", "RT_s", "Choice"]],
}
for name, table in tables.items():
    table.to_csv(out / name, index=False, float_format="%.17g")

# %% Overlay both groups in five RTD panels beside the shared psychometric panel.
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})
fig = plt.figure(figsize=(22, 5.2))
grid = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1.7],
                       left=.045, right=.985, bottom=.25, top=.70, wspace=.33)
ymax = np.ceil(max(hist.density.max(), model_curves.density.max()) * 1.09 * 2) / 2
for c, delta in enumerate(ABS_DELTA):
    ax = fig.add_subplot(grid[0, c])
    panel_counts = []
    for group in GROUPS:
        h = hist.loc[hist.Group.eq(group) & hist.abs_delta.eq(delta)]
        m = model_curves.loc[model_curves.Group.eq(group) & model_curves.abs_delta.eq(delta)]
        ax.stairs(h.density.to_numpy(), np.r_[h.left_s, h.right_s.iloc[-1]],
                  color=COLORS[group], alpha=.45, lw=1.4)
        ax.plot(m.time_s, m.density, color=COLORS[group], lw=2.2)
        panel_counts.append(int(h.n.iloc[0]))
    ax.set(xlim=(0, 1.8), ylim=(0, ymax), xticks=[0, .6, 1.2, 1.8],
           xlabel="RT (s)", title=f'|ΔdB| = {delta:g}\nn: Control {panel_counts[0]:,} · ASD {panel_counts[1]:,}')
    ax.grid(alpha=.16)
    if c == 0:
        ax.set_ylabel("Density (s⁻¹)")
    else:
        ax.tick_params(labelleft=False)

ax = fig.add_subplot(grid[0, 5])
for group in GROUPS:
    p = psychometric.loc[psychometric.Group.eq(group)].sort_values("signed_delta")
    ax.plot(p.signed_delta, p.model_p_right, color=COLORS[group], lw=2.2, label=group)
    ax.scatter(p.signed_delta, p.empirical_p_right, color=COLORS[group], s=28,
               edgecolor="white", linewidth=.4, zorder=3)
ax.set(title="Psychometric\nBoth ASLs pooled", xlabel="Signed ΔdB (right − left)",
       ylabel="P(right)", xlim=(-2.15, 2.15), ylim=(-.015, 1.015), xticks=[-2, -1, 0, 1, 2])
ax.grid(alpha=.18)
ax.legend(loc="lower right", frameon=True, fontsize=10)
fig.suptitle("Human simultaneous RT | both ASLs pooled | data and model", fontsize=19, y=.97)
fig.text(.5, .90, "Control: 6,574 trials, 11 participants   |   ASD: 4,770 trials, 8 participants   |   ASL 20 + 40 cd/m²", ha="center", fontsize=11)
fig.legend(handles=[
    Line2D([], [], color=".45", lw=1.4, alpha=.6, drawstyle="steps-mid", label="Data: 20 ms RT bins"),
    Line2D([], [], color=".45", marker="o", linestyle="none", label="Data: choice proportions"),
    Line2D([], [], color=".2", lw=2.2, label="Model at posterior means"),
    Line2D([], [], color=COLORS["Control"], lw=2.2, label="Control"),
    Line2D([], [], color=COLORS["ASD"], lw=2.2, label="ASD"),
], loc="upper center", bbox_to_anchor=(.5, .855), ncol=5, frameon=False)
fig.text(.5, .125, "Condition = RT; Valid = 1; 0–1.8 s retained | Both choices; raw trials pooled across participants, sessions and ASLs", ha="center", fontsize=10)
fig.text(.5, .078, "Gaussian delay + choice + uniform lapses | Existing ASL-specific predictions combined using retained-trial weights; no refit", ha="center", fontsize=10)
fig.text(.5, .031, "RTDs additionally pool ±Δ; each curve has unit mass on 0–1.8 s | Psychometric model points joined only at measured differences", ha="center", fontsize=10)
for suffix in ["png", "pdf"]:
    fig.savefig(out / f"{STEM}.{suffix}", dpi=180)
plt.close(fig)

# %% Provenance includes the saved curves, source trials, posterior samples and means.
inputs = [fit / name for name in ["run_config.json", "plot_manifest.json", "publication_manifest.json",
          "selected_trials.csv", "condition_model_curves.csv", "psychometric_comparison.csv", "parameter_summary.csv"]]
inputs += [fit / g.lower() / name for g in GROUPS for name in ["posterior_samples.npz", "convergence_summary.json"]]
inputs += [ROOT / "ASD_Human_Data/simultaneous_rt_trials.csv", Path(__file__).resolve()]
manifest = {
    "fit_dir": str(fit), "output_dir": str(out), "source": str(Path(__file__).resolve()),
    "source_hashes": {str(p): sha256(p) for p in inputs},
    "model": config["model"], "refitted": False,
    "curve_parameters": previous_plot["curve_parameters"],
    "selection": "Condition == RT, Valid == 1, 0 <= RT_s <= 1.8",
    "counts": {g: int(trials.Group.eq(g).sum()) for g in GROUPS},
    "participants": {g: int(trials.loc[trials.Group.eq(g), "Participant"].nunique()) for g in GROUPS},
    "rtd_pooling": "four ASL-by-sign normalized components, weighted by actual retained trial counts",
    "psychometric_pooling": "two ASL conditional right-choice probabilities, weighted by actual retained trial counts at each signed difference",
    "normalization": "each component retains its 0–1.8 s normalization, Gaussian-delay normalization and uniform lapse contribution",
    "histogram_bin_s": .02, "model_grid_s": .0005,
    "layout": {"rows": 1, "groups_overlaid": GROUPS, "columns_abs_delta": ABS_DELTA, "colors": COLORS, "rtd_ymax": float(ymax)},
    "interpretation": "A mixture of existing ASL-specific fitted predictions; does not impose or test equality of their parameters.",
    "files": {name: sha256(out / name) for name in list(tables) + [f"{STEM}.png", f"{STEM}.pdf"]},
}
(out / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(pd.DataFrame(checks).to_string(index=False))
print(f"Saved {out / (STEM + '.png')}")
