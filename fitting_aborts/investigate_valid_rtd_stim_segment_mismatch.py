# %%
SHOW_PLOT = True

DESIRED_BATCHES = ["LED7"]
PARAM_REDUCER = "median"

SEGMENT_MODE = "quantile"  # "quantile" or "fixed"
NUM_QUANTILE_BINS = 2
FIXED_SEGMENT_EDGES_S = (0.2, 0.4, 1.5)

ABL_VALUES = (20, 40, 60)
ILD_VALUES = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)

RT_MIN_S = -1.0
RT_MAX_S = 1.0
BIN_SIZE_S = 5e-3
X_LIM_S = (0.0, 0.150)
INTENDED_FIX_MIN_S = 0.2
INTENDED_FIX_MAX_S = 1.5

N_MC_T_STIM_SAMPLES = 2000
RNG_SEED = 12345

VALID_RATE_FIX_BINS_S = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5)

FIGURE_SIZE = (12.0, 8.5)
PNG_DPI = 300


# %%
from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent
FIT_DIR = REPO_ROOT / "fit_animal_by_animal"
if str(FIT_DIR) not in sys.path:
    sys.path.append(str(FIT_DIR))

from time_vary_norm_utils import (
    rho_A_t_VEC_fn,
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec,
)


# %%
batch_csv_dir = FIT_DIR / "batch_csvs"
results_dir = FIT_DIR
output_dir = SCRIPT_DIR / "stim_segment_mismatch_diagnostics"

bins_s = np.arange(RT_MIN_S, RT_MAX_S + BIN_SIZE_S, BIN_SIZE_S)
bin_centers_s = 0.5 * (bins_s[:-1] + bins_s[1:])
t_pts_s = np.arange(RT_MIN_S, RT_MAX_S + 1e-3, 1e-3)
mask_positive_t = t_pts_s >= 0
mask_positive_bins = bin_centers_s >= 0
mask_display_bins = (bin_centers_s >= X_LIM_S[0]) & (bin_centers_s <= X_LIM_S[1])

segment_colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
abl_colors = {
    20: "tab:blue",
    40: "tab:orange",
    60: "tab:green",
}
variant_titles = {
    "data": "Data",
    "raw_current": "Model Current",
    "per_t_conditional": "Model Valid-Conditioned",
}


def reduce_param_values(values):
    values = np.asarray(values, dtype=float)
    if PARAM_REDUCER == "mean":
        return float(np.mean(values))
    if PARAM_REDUCER == "median":
        return float(np.median(values))
    raise ValueError(f"Unknown PARAM_REDUCER: {PARAM_REDUCER}")


def get_candidate_animals():
    candidates = []
    for batch_name in DESIRED_BATCHES:
        batch_file = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"
        if not batch_file.exists():
            continue
        df = pd.read_csv(batch_file)
        if "success" not in df.columns or "animal" not in df.columns:
            continue
        valid_df = df[df["success"].isin([1, -1])].copy()
        animals = pd.Series(valid_df["animal"].dropna().unique()).tolist()
        for animal in sorted(animals):
            try:
                animal_id = int(animal)
            except Exception:
                continue
            pkl_path = results_dir / f"results_{batch_name}_animal_{animal_id}.pkl"
            if not pkl_path.exists():
                continue
            try:
                with open(pkl_path, "rb") as handle:
                    results = pickle.load(handle)
            except Exception:
                continue
            if "vbmc_aborts_results" in results and "vbmc_norm_tied_results" in results:
                candidates.append((str(batch_name), animal_id))
    return sorted(candidates)


def load_all_trials(candidate_pairs):
    animals_by_batch = {}
    for batch_name, animal_id in candidate_pairs:
        animals_by_batch.setdefault(str(batch_name), set()).add(int(animal_id))

    frames = []
    for batch_name in DESIRED_BATCHES:
        batch_file = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"
        if not batch_file.exists():
            continue
        df = pd.read_csv(batch_file)
        if "batch_name" not in df.columns:
            df["batch_name"] = str(batch_name)
        animal_numeric = pd.to_numeric(df["animal"], errors="coerce")
        df = df[animal_numeric.isin(animals_by_batch.get(str(batch_name), set()))].copy()
        for column in ["animal", "success", "RTwrtStim", "ABL", "ILD", "intended_fix"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df[
            (df["intended_fix"] >= INTENDED_FIX_MIN_S)
            & (df["intended_fix"] <= INTENDED_FIX_MAX_S)
            & (df["ABL"].isin(ABL_VALUES))
            & (df["ILD"].isin(ILD_VALUES))
        ].copy()
        frames.append(df)

    if not frames:
        raise ValueError("No trials found after filtering.")
    return pd.concat(frames, ignore_index=True)


def load_valid_trials(all_trials_df):
    valid_df = all_trials_df[all_trials_df["success"].isin([1, -1])].copy()
    valid_df = valid_df[
        (valid_df["RTwrtStim"] >= RT_MIN_S)
        & (valid_df["RTwrtStim"] <= RT_MAX_S)
    ].copy()
    if len(valid_df) == 0:
        raise ValueError("No valid trials found after filtering.")
    return valid_df


def load_aggregate_params(candidate_pairs):
    abort_records = []
    tied_records = []
    for batch_name, animal_id in candidate_pairs:
        pkl_path = results_dir / f"results_{batch_name}_animal_{animal_id}.pkl"
        with open(pkl_path, "rb") as handle:
            results = pickle.load(handle)
        abort_blob = results["vbmc_aborts_results"]
        tied_blob = results["vbmc_norm_tied_results"]
        abort_records.append(
            {
                "V_A": reduce_param_values(abort_blob["V_A_samples"]),
                "theta_A": reduce_param_values(abort_blob["theta_A_samples"]),
                "t_A_aff": reduce_param_values(abort_blob["t_A_aff_samp"]),
            }
        )
        tied_records.append(
            {
                "rate_lambda": reduce_param_values(tied_blob["rate_lambda_samples"]),
                "T_0": reduce_param_values(tied_blob["T_0_samples"]),
                "theta_E": reduce_param_values(tied_blob["theta_E_samples"]),
                "w": reduce_param_values(tied_blob["w_samples"]),
                "t_E_aff": reduce_param_values(tied_blob["t_E_aff_samples"]),
                "del_go": reduce_param_values(tied_blob["del_go_samples"]),
                "rate_norm_l": reduce_param_values(tied_blob["rate_norm_l_samples"]),
            }
        )
    abort_params = {key: reduce_param_values([record[key] for record in abort_records]) for key in abort_records[0]}
    tied_params = {key: reduce_param_values([record[key] for record in tied_records]) for key in tied_records[0]}
    return abort_params, tied_params


def get_segment_edges(valid_df):
    if SEGMENT_MODE == "fixed":
        edges = np.asarray(FIXED_SEGMENT_EDGES_S, dtype=float)
    elif SEGMENT_MODE == "quantile":
        _, edges = pd.qcut(
            valid_df["intended_fix"],
            q=NUM_QUANTILE_BINS,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        edges = np.asarray(edges, dtype=float)
    else:
        raise ValueError(f"Unsupported SEGMENT_MODE: {SEGMENT_MODE}")

    if len(edges) < 3:
        raise ValueError(f"Need at least 2 segments, got edges={edges}")
    return edges


def format_segment_label(segment_idx, edges):
    return f"[{edges[segment_idx]:.3f}, {edges[segment_idx + 1]:.3f}] s"


def make_segment_df_list(valid_df, edges):
    segment_dfs = []
    for segment_idx in range(len(edges) - 1):
        left = edges[segment_idx]
        right = edges[segment_idx + 1]
        if segment_idx == 0:
            mask = (valid_df["intended_fix"] >= left) & (valid_df["intended_fix"] <= right)
        else:
            mask = (valid_df["intended_fix"] > left) & (valid_df["intended_fix"] <= right)
        segment_dfs.append(valid_df[mask].copy())
    return segment_dfs


def compute_valid_rate_summary(all_trials_df):
    df = all_trials_df.copy()
    df["is_valid"] = df["success"].isin([1, -1]).astype(int)
    edges = np.asarray(VALID_RATE_FIX_BINS_S, dtype=float)
    cut_edges = edges.copy()
    cut_edges[0] = np.nextafter(cut_edges[0], -np.inf)
    cut_edges[-1] = np.nextafter(cut_edges[-1], np.inf)
    df["fix_bin"] = pd.cut(df["intended_fix"], bins=cut_edges, include_lowest=True)
    summary = (
        df.groupby("fix_bin", observed=False)
        .agg(
            n=("is_valid", "size"),
            valid_rate=("is_valid", "mean"),
            mean_fix=("intended_fix", "mean"),
        )
        .reset_index()
    )
    return summary


def compute_density_histogram(values):
    if len(values) == 0:
        return np.zeros(len(bins_s) - 1, dtype=float)
    hist, _ = np.histogram(values, bins=bins_s, density=True)
    return hist


def curve_to_binned_density(t_pts, density):
    cdf = cumulative_trapezoid(density, t_pts, initial=0)
    edge_cdf = np.interp(bins_s, t_pts, cdf, left=0.0, right=float(cdf[-1]))
    return np.diff(edge_cdf) / np.diff(bins_s)


def raw_density_from_pa_ca(t, p_a, c_a, abl_value, ild_value, tied_params):
    z_e = (tied_params["w"] - 0.5) * 2.0 * tied_params["theta_E"]
    up = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
        t,
        1,
        p_a,
        c_a,
        float(abl_value),
        float(ild_value),
        tied_params["rate_lambda"],
        tied_params["T_0"],
        tied_params["theta_E"],
        z_e,
        tied_params["t_E_aff"],
        tied_params["del_go"],
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        tied_params["rate_norm_l"],
        True,
        False,
        10,
    )
    down = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
        t,
        -1,
        p_a,
        c_a,
        float(abl_value),
        float(ild_value),
        tied_params["rate_lambda"],
        tied_params["T_0"],
        tied_params["theta_E"],
        z_e,
        tied_params["t_E_aff"],
        tied_params["del_go"],
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        tied_params["rate_norm_l"],
        True,
        False,
        10,
    )
    return up + down


def compute_model_variants_for_segment(segment_df, abort_params, tied_params, rng):
    t_stim_samples = rng.choice(segment_df["intended_fix"].to_numpy(), size=N_MC_T_STIM_SAMPLES, replace=True)
    shifted_t = t_pts_s[None, :] + t_stim_samples[:, None] - abort_params["t_A_aff"]
    p_a_samples = rho_A_t_VEC_fn(shifted_t, abort_params["V_A"], abort_params["theta_A"])
    c_a_samples = cumulative_trapezoid(p_a_samples, t_pts_s, axis=1, initial=0)
    p_a_mean = p_a_samples.mean(axis=0)
    c_a_mean = c_a_samples.mean(axis=0)

    curves_by_variant_and_abl = {
        "raw_current": {},
        "segment_renorm": {},
        "per_t_conditional": {},
    }
    positive_widths = np.diff(bins_s)[mask_positive_bins]

    for abl_value in ABL_VALUES:
        abl_df = segment_df[np.isclose(segment_df["ABL"], abl_value)].copy()
        counts = abl_df["ILD"].round().astype(int).value_counts().to_dict()
        total = int(len(abl_df))

        raw_mixture_curve = np.zeros(len(t_pts_s), dtype=float)
        per_t_conditional_curve = np.zeros(len(t_pts_s), dtype=float)

        for ild_value in ILD_VALUES:
            count = int(counts.get(int(ild_value), 0))
            if count == 0 or total == 0:
                continue
            weight = count / total

            raw_curve = raw_density_from_pa_ca(t_pts_s, p_a_mean, c_a_mean, abl_value, ild_value, tied_params)
            raw_mixture_curve += weight * raw_curve

            raw_matrix = raw_density_from_pa_ca(
                t_pts_s[None, :],
                p_a_samples,
                c_a_samples,
                abl_value,
                ild_value,
                tied_params,
            )
            positive_area_per_t_stim = np.trapz(raw_matrix[:, mask_positive_t], t_pts_s[mask_positive_t], axis=1)
            positive_area_per_t_stim = np.clip(positive_area_per_t_stim, 1e-12, None)
            conditional_matrix = raw_matrix / positive_area_per_t_stim[:, None]
            per_t_conditional_curve += weight * conditional_matrix.mean(axis=0)

        raw_binned = curve_to_binned_density(t_pts_s, raw_mixture_curve)
        raw_positive_area = np.sum(raw_binned[mask_positive_bins] * positive_widths)
        raw_positive_area = max(raw_positive_area, 1e-12)

        curves_by_variant_and_abl["raw_current"][int(abl_value)] = raw_binned
        curves_by_variant_and_abl["segment_renorm"][int(abl_value)] = raw_binned / raw_positive_area
        curves_by_variant_and_abl["per_t_conditional"][int(abl_value)] = curve_to_binned_density(
            t_pts_s,
            per_t_conditional_curve,
        )

    return curves_by_variant_and_abl


def compute_display_diff_stats(early_curve, late_curve):
    diff = early_curve[mask_display_bins] - late_curve[mask_display_bins]
    return float(np.min(diff)), float(np.max(diff))


def build_data_curves(segment_dfs):
    curves = []
    for segment_df in segment_dfs:
        segment_curves = {}
        for abl_value in ABL_VALUES:
            values = segment_df[np.isclose(segment_df["ABL"], abl_value)]["RTwrtStim"].to_numpy()
            segment_curves[int(abl_value)] = compute_density_histogram(values)
        curves.append(segment_curves)
    return curves


def save_figure(fig, output_base):
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")


def plot_comparison(segment_edges, data_curves, model_curves):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(ABL_VALUES),
        3,
        figsize=FIGURE_SIZE,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_edges_ms = bins_s * 1e3
    segment_labels = [format_segment_label(i, segment_edges) for i in range(len(segment_edges) - 1)]

    for row_idx, abl_value in enumerate(ABL_VALUES):
        for col_idx, variant_key in enumerate(["data", "raw_current", "per_t_conditional"]):
            ax = axes[row_idx, col_idx]
            for segment_idx in range(len(segment_labels)):
                if variant_key == "data":
                    y_values = data_curves[segment_idx][int(abl_value)]
                else:
                    y_values = model_curves[segment_idx][variant_key][int(abl_value)]
                ax.stairs(
                    y_values,
                    x_edges_ms,
                    color=segment_colors[segment_idx],
                    linewidth=1.8,
                    alpha=1.0 if segment_idx == 0 else 0.65,
                    label=f"{'Early' if segment_idx == 0 else 'Late'} {segment_labels[segment_idx]}",
                )
            ax.set_xlim(X_LIM_S[0] * 1e3, X_LIM_S[1] * 1e3)
            ax.grid(alpha=0.2, linewidth=0.6)
            ax.set_title(f"{variant_titles[variant_key]}\nABL = {abl_value}")
            if row_idx == len(ABL_VALUES) - 1:
                ax.set_xlabel("RT wrt stim (ms)")
            if col_idx == 0:
                ax.set_ylabel("Density")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_base = output_dir / f"investigate_valid_rtd_stim_segment_mismatch_{SEGMENT_MODE}"
    save_figure(fig, output_base)
    return fig, output_base


# %%
candidate_pairs = get_candidate_animals()
all_trials_df = load_all_trials(candidate_pairs)
valid_df = load_valid_trials(all_trials_df)
abort_params, tied_params = load_aggregate_params(candidate_pairs)
segment_edges = get_segment_edges(valid_df)
segment_dfs = make_segment_df_list(valid_df, segment_edges)
valid_rate_summary = compute_valid_rate_summary(all_trials_df)

rng = np.random.default_rng(RNG_SEED)
data_curves = build_data_curves(segment_dfs)
model_curves = []
for segment_df in segment_dfs:
    model_variants = compute_model_variants_for_segment(segment_df, abort_params, tied_params, rng)
    model_curves.append(model_variants)


# %%
print(f"Included batch-animal pairs ({len(candidate_pairs)}): {candidate_pairs}")
print(f"SEGMENT_MODE={SEGMENT_MODE}")
print(f"Segment edges (s): {[float(edge) for edge in segment_edges]}")
print(f"Valid trials used: {len(valid_df)}")
print("Aggregate abort params:")
for key, value in abort_params.items():
    print(f"  {key}: {value:.6f}")
print("Aggregate tied params:")
for key, value in tied_params.items():
    print(f"  {key}: {value:.6f}")

print("\nValid-rate vs intended_fix bins:")
print(valid_rate_summary.to_string(index=False))

print("\nEarly-minus-late density differences inside display window:")
for abl_value in ABL_VALUES:
    data_min, data_max = compute_display_diff_stats(
        data_curves[0][int(abl_value)],
        data_curves[-1][int(abl_value)],
    )
    raw_min, raw_max = compute_display_diff_stats(
        model_curves[0]["raw_current"][int(abl_value)],
        model_curves[-1]["raw_current"][int(abl_value)],
    )
    seg_min, seg_max = compute_display_diff_stats(
        model_curves[0]["segment_renorm"][int(abl_value)],
        model_curves[-1]["segment_renorm"][int(abl_value)],
    )
    per_t_min, per_t_max = compute_display_diff_stats(
        model_curves[0]["per_t_conditional"][int(abl_value)],
        model_curves[-1]["per_t_conditional"][int(abl_value)],
    )
    print(f"ABL = {abl_value}")
    print(f"  data              diff[min, max] = [{data_min:.4f}, {data_max:.4f}]")
    print(f"  model current     diff[min, max] = [{raw_min:.4f}, {raw_max:.4f}]")
    print(f"  model seg-renorm  diff[min, max] = [{seg_min:.4f}, {seg_max:.4f}]")
    print(f"  model per-t-cond  diff[min, max] = [{per_t_min:.4f}, {per_t_max:.4f}]")


# %%
fig, output_base = plot_comparison(segment_edges, data_curves, model_curves)
print(f"\nSaved: {output_base.with_suffix('.pdf')}")
print(f"Saved: {output_base.with_suffix('.png')}")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)
