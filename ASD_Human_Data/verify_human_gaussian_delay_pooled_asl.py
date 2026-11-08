# %%
"""Independently reconstruct pooled-ASL plot data from trials and saved components."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIT = ROOT / "ASD_Human_Data/gamma_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT)
parser.add_argument("--output-dir", type=Path, default=None)
args = parser.parse_args()
fit = args.fit_dir.resolve()
out = args.output_dir.resolve() if args.output_dir else fit / "pooled_asl"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# %% Verify input/output identity, original row selection and posterior-mean convention.
manifest = json.loads((out / "plot_manifest.json").read_text())
for path, digest in manifest["source_hashes"].items():
    assert sha256(path) == digest, path
for name, digest in manifest["files"].items():
    assert sha256(out / name) == digest, name
assert manifest["refitted"] is False
assert manifest["curve_parameters"] == "arithmetic physical posterior means"
source = pd.read_csv(ROOT / "ASD_Human_Data/simultaneous_rt_trials.csv", float_precision="round_trip")
trials = source.query("Condition == 'RT' and Valid == 1 and 0 <= RT_s <= 1.8").copy()
# Stored subtraction has machine-precision residues; match the fit's ten stimuli.
trials["signed_delta"] = trials.Delta_dB.round(2)
np.testing.assert_allclose(trials.signed_delta, trials.Delta_dB, atol=1e-12)
trials["abs_delta"] = trials.signed_delta.abs()
membership = pd.read_csv(out / "trial_membership.csv", float_precision="round_trip")
pd.testing.assert_frame_equal(
    membership.sort_values("source_row").reset_index(drop=True),
    trials[membership.columns].sort_values("source_row").reset_index(drop=True),
)
assert len(trials) == 11344 and trials.source_row.is_unique
assert trials.groupby("Group").size().to_dict() == {"ASD": 4770, "Control": 6574}
assert int(trials.RT_s.lt(.2).sum()) == 4
means = pd.read_csv(fit / "parameter_summary.csv", float_precision="round_trip")
for group in ["Control", "ASD"]:
    with np.load(fit / group.lower() / "posterior_samples.npz") as posterior:
        samples = np.column_stack([posterior[key] for key in
            ["gamma", "omega", "delay_mu", "delay_sigma", "w", "lapse_prob", "lapse_prob_right"]])
        assert samples.shape == (10000, 83)
        np.testing.assert_allclose(means.loc[means.Group.eq(group), "mean"], samples.mean(axis=0), atol=1e-12)

# %% Psychometric check: assign the component prediction to each raw trial, then average.
components = pd.read_csv(fit / "psychometric_comparison.csv", float_precision="round_trip")
keys = ["Group", "ASL_cd", "signed_delta"]
assigned = trials.merge(components[keys + ["model_p_right", "condition_index"]],
                        on=keys, how="left", validate="many_to_one")
assert assigned.model_p_right.notna().all()
assigned["right"] = assigned.Choice.eq(1).astype(int)
expected = assigned.groupby(["Group", "signed_delta"]).agg(
    n=("source_row", "size"), right=("right", "sum"),
    empirical_p_right=("right", "mean"), model_p_right=("model_p_right", "mean"))
actual = pd.read_csv(out / "pooled_psychometric.csv", float_precision="round_trip").set_index(["Group", "signed_delta"]).sort_index()
pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-12, rtol=1e-12)

# %% Histogram check directly from source RTs, independently of saved weights.
histograms = pd.read_csv(out / "pooled_rtd_histograms.csv", float_precision="round_trip")
edges = np.arange(91) / 50.
for (group, delta), obs in assigned.groupby(["Group", "abs_delta"]):
    saved = histograms.loc[histograms.Group.eq(group) & np.isclose(histograms.abs_delta, delta)]
    counts = np.histogram(obs.RT_s, edges)[0]
    np.testing.assert_array_equal(saved["count"], counts)
    assert saved.n.eq(len(obs)).all()
    np.testing.assert_allclose(saved.density, counts / (len(obs) * .02), atol=1e-12)
    np.testing.assert_allclose(saved.left_s, edges[:-1], atol=1e-14)
    np.testing.assert_allclose(saved.right_s, edges[1:], atol=1e-14)

# %% Construct mixture weights from trial counts, then aggregate curves by time.
counts = assigned.groupby(["Group", "condition_index", "ASL_cd", "signed_delta", "abs_delta"]).size().rename("component_n").reset_index()
rtd_weights = counts.copy()
rtd_weights["pooled_n"] = rtd_weights.groupby(["Group", "abs_delta"]).component_n.transform("sum")
rtd_weights["weight"] = rtd_weights.component_n / rtd_weights.pooled_n
saved_weights = pd.read_csv(out / "mixture_weights.csv", float_precision="round_trip")
weight_keys = ["Group", "condition_index"]
for panel, group_key in [("rtd", "abs_delta"), ("psychometric", "signed_delta")]:
    expected_weights = counts.copy()
    expected_weights["pooled_n"] = counts.groupby(["Group", group_key]).component_n.transform("sum")
    expected_weights["weight"] = counts.component_n / expected_weights.pooled_n
    expected_weights["pooled_delta"] = expected_weights[group_key]
    columns = ["ASL_cd", "signed_delta", "pooled_delta", "component_n", "pooled_n", "weight"]
    left = saved_weights.loc[saved_weights.panel.eq(panel)].set_index(weight_keys).sort_index()[columns]
    right = expected_weights.set_index(weight_keys).sort_index()[columns]
    pd.testing.assert_frame_equal(left, right, check_exact=False, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(expected_weights.groupby(["Group", group_key]).weight.sum(), 1., atol=1e-14)

curves = pd.read_csv(fit / "condition_model_curves.csv", float_precision="round_trip")
curves = curves.merge(rtd_weights[["Group", "condition_index", "abs_delta", "weight"]],
                      on=["Group", "condition_index"], validate="many_to_one")
columns = ["left_density", "right_density", "density", "cdf"]
curves[columns] = curves[columns].multiply(curves.weight, axis=0)
expected_curves = curves.groupby(["Group", "abs_delta", "time_s"])[columns].sum()
actual_curves = pd.read_csv(out / "pooled_model_curves.csv", float_precision="round_trip").set_index(["Group", "abs_delta", "time_s"]).sort_index()
pd.testing.assert_frame_equal(actual_curves, expected_curves, check_exact=False, atol=1e-12, rtol=1e-12)
mass_errors = []
for _, curve in actual_curves.reset_index().groupby(["Group", "abs_delta"]):
    mass_errors.append(abs(float(np.trapz(curve.density, curve.time_s)) - 1.))
assert max(mass_errors) < 1e-6

# %% Preserve prior figures, fit files, and the entire pre-existing result-page prefix.
baseline = json.loads((out / "preservation_baseline.json").read_text())
for path, digest in baseline["unchanged_files"].items():
    assert sha256(path) == digest, path
prefix = Path(baseline["result_page"]).read_bytes()[:baseline["result_page_bytes"]]
assert hashlib.sha256(prefix).hexdigest() == baseline["result_page_prefix_sha256"]
report = {
    "passed": True, "selected_trial_identities": len(trials), "early_rts_below_200ms": 4,
    "group_counts": trials.groupby("Group").size().to_dict(),
    "psychometric_points": len(actual), "rtd_panels": 5, "rtd_group_conditions": 10, "histogram_bins": len(histograms),
    "mixture_weights_checked": len(saved_weights), "posterior_means_checked": 166,
    "max_unit_mass_error": max(mass_errors),
    "preserved_prior_files": len(baseline["unchanged_files"]), "prior_page_prefix_preserved": True,
    "verifier": str(Path(__file__).resolve()), "verifier_sha256": sha256(__file__),
    "plot_manifest_sha256": sha256(out / "plot_manifest.json"),
}
(out / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
