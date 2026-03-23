# %%
SHOW_PLOT = True

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
TARGET_BATCH = 'LED7'
TARGET_ANIMAL = 92

MODEL_KEY = "vbmc_norm_tied_results"
ABORT_KEY = "vbmc_aborts_results"

ABL_VALUES = (20, 40, 60)
ILD_VALUES = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)

SEGMENT_SPECS = [
    {"name": "Q1/2", "left": 0.2, "right": 0.4, "t_stim": 0.30},
    {"name": "Q2/2", "left": 0.4, "right": 1.5, "t_stim": 0.9},
]

rt_min_s = -1.0
rt_max_s = 1.0
bin_size_s = 5e-3
intended_fix_max_s = 1.5
xlim_s = (-0.3, 0.2)
figure_size = (5.0, 6.6)
png_dpi = 300

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
    rho_A_t_fn,
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn,
)

# %%
batch_csv_dir = FIT_DIR / "batch_csvs"
results_dir = FIT_DIR
output_dir = SCRIPT_DIR / "model_rtd_stim_examples_single_animal"
output_base = output_dir / "model_rtd_by_abl_two_stim_examples_norm_tied"

abl_colors = {
    20: "tab:blue",
    40: "tab:orange",
    60: "tab:green",
}


def get_candidate_animals():
    candidates = []
    for batch_name in DESIRED_BATCHES:
        batch_file = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"
        if not batch_file.exists():
            continue
        df = pd.read_csv(batch_file)
        if "success" not in df.columns:
            continue
        valid_df = df[df["success"].isin([1, -1])].copy()
        if len(valid_df) == 0:
            continue
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
            if ABORT_KEY in results and MODEL_KEY in results:
                candidates.append((batch_name, animal_id))
    return candidates


def load_animal_valid_df(batch_name, animal_id):
    batch_file = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"
    df = pd.read_csv(batch_file)
    df = df[df["animal"].astype(str) == str(animal_id)].copy()
    df = df[df["success"].isin([1, -1])].copy()
    df["RTwrtStim"] = pd.to_numeric(df["RTwrtStim"], errors="coerce")
    df["intended_fix"] = pd.to_numeric(df["intended_fix"], errors="coerce")
    df["ABL"] = pd.to_numeric(df["ABL"], errors="coerce")
    df["ILD"] = pd.to_numeric(df["ILD"], errors="coerce")
    df = df[
        (df["RTwrtStim"] >= rt_min_s)
        & (df["RTwrtStim"] <= rt_max_s)
        & (df["intended_fix"] >= SEGMENT_SPECS[0]["left"])
        & (df["intended_fix"] <= intended_fix_max_s)
        & (df["ABL"].isin(ABL_VALUES))
        & (df["ILD"].isin(ILD_VALUES))
    ].copy()
    return df


def has_segment_coverage(batch_name, animal_id):
    df = load_animal_valid_df(batch_name, animal_id)
    if len(df) == 0:
        return False
    for segment_spec in SEGMENT_SPECS:
        seg_df = df[(df["intended_fix"] >= segment_spec["left"]) & (df["intended_fix"] <= segment_spec["right"])].copy()
        if len(seg_df) == 0:
            return False
        for abl_value in ABL_VALUES:
            if len(seg_df[np.isclose(seg_df["ABL"], abl_value)]) == 0:
                return False
    return True


def choose_target_animal(candidates):
    if TARGET_BATCH is not None and TARGET_ANIMAL is not None:
        return str(TARGET_BATCH), int(TARGET_ANIMAL)
    for batch_name, animal_id in candidates:
        if has_segment_coverage(batch_name, animal_id):
            return batch_name, animal_id
    if not candidates:
        raise ValueError("No animals found with both abort and norm-tied PKL results.")
    return candidates[0]


def load_params(batch_name, animal_id):
    pkl_path = results_dir / f"results_{batch_name}_animal_{animal_id}.pkl"
    with open(pkl_path, "rb") as handle:
        results = pickle.load(handle)

    abort_blob = results[ABORT_KEY]
    model_blob = results[MODEL_KEY]

    abort_params = {
        "V_A": float(np.mean(abort_blob["V_A_samples"])),
        "theta_A": float(np.mean(abort_blob["theta_A_samples"])),
        "t_A_aff": float(np.mean(abort_blob["t_A_aff_samp"])),
    }
    tied_params = {
        "rate_lambda": float(np.mean(model_blob["rate_lambda_samples"])),
        "T_0": float(np.mean(model_blob["T_0_samples"])),
        "theta_E": float(np.mean(model_blob["theta_E_samples"])),
        "w": float(np.mean(model_blob["w_samples"])),
        "t_E_aff": float(np.mean(model_blob["t_E_aff_samples"])),
        "del_go": float(np.mean(model_blob["del_go_samples"])),
        "rate_norm_l": float(np.mean(model_blob["rate_norm_l_samples"])),
    }
    return abort_params, tied_params, pkl_path


def compute_fixed_stim_rtd(abl_value, ild_value, t_stim, abort_params, tied_params):
    t_pts = np.arange(-2.0, 2.001, 0.001)
    p_a = np.array([
        rho_A_t_fn(t + t_stim - abort_params["t_A_aff"], abort_params["V_A"], abort_params["theta_A"])
        for t in t_pts
    ])
    c_a = cumulative_trapezoid(p_a, t_pts, initial=0)
    z_e = (tied_params["w"] - 0.5) * 2.0 * tied_params["theta_E"]

    up_mean = np.array([
        up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
            t,
            1,
            p_a[idx],
            c_a[idx],
            abl_value,
            ild_value,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            z_e,
            tied_params["t_E_aff"],
            tied_params["del_go"],
            np.nan,
            tied_params["rate_norm_l"],
            True,
            False,
            10,
        )
        for idx, t in enumerate(t_pts)
    ])
    down_mean = np.array([
        up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
            t,
            -1,
            p_a[idx],
            c_a[idx],
            abl_value,
            ild_value,
            tied_params["rate_lambda"],
            tied_params["T_0"],
            tied_params["theta_E"],
            z_e,
            tied_params["t_E_aff"],
            tied_params["del_go"],
            np.nan,
            tied_params["rate_norm_l"],
            True,
            False,
            10,
        )
        for idx, t in enumerate(t_pts)
    ])

    mask = (t_pts >= rt_min_s) & (t_pts <= rt_max_s)
    t_pts = t_pts[mask]
    total_rtd = up_mean[mask] + down_mean[mask]
    return t_pts, total_rtd


def curve_to_binned_density(t_pts, density, bins_s):
    cdf = cumulative_trapezoid(density, t_pts, initial=0)
    edge_cdf = np.interp(bins_s, t_pts, cdf, left=0.0, right=float(cdf[-1]))
    probs = np.diff(edge_cdf)
    widths = np.diff(bins_s)
    return probs / widths


def build_segment_mixture(segment_df, segment_spec, abl_value, abort_params, tied_params, bins_s):
    abl_df = segment_df[np.isclose(segment_df["ABL"], abl_value)].copy()
    counts = abl_df["ILD"].round().astype(int).value_counts().to_dict() if len(abl_df) else {}
    total = int(len(abl_df))
    if total == 0:
        return np.zeros(len(bins_s) - 1, dtype=float), {}

    mixture = np.zeros(len(bins_s) - 1, dtype=float)
    weight_map = {}
    for ild_value in ILD_VALUES:
        count = int(counts.get(int(ild_value), 0))
        if count == 0:
            continue
        weight = count / total
        t_pts, density = compute_fixed_stim_rtd(
            abl_value,
            float(ild_value),
            segment_spec["t_stim"],
            abort_params,
            tied_params,
        )
        mixture += weight * curve_to_binned_density(t_pts, density, bins_s)
        weight_map[int(ild_value)] = weight
    return mixture, weight_map


def load_data():
    candidates = get_candidate_animals()
    batch_name, animal_id = choose_target_animal(candidates)
    valid_df = load_animal_valid_df(batch_name, animal_id)
    if len(valid_df) == 0:
        raise ValueError(f"No valid RTwrtStim trials found for {batch_name}-{animal_id} after filtering.")
    abort_params, tied_params, pkl_path = load_params(batch_name, animal_id)

    bins_s = np.arange(rt_min_s, rt_max_s + bin_size_s, bin_size_s)
    segment_results = []

    for segment_spec in SEGMENT_SPECS:
        segment_df = valid_df[
            (valid_df["intended_fix"] >= segment_spec["left"])
            & (valid_df["intended_fix"] <= segment_spec["right"])
        ].copy()
        counts_by_abl = {
            int(abl_value): int(np.isclose(segment_df["ABL"], abl_value).sum())
            for abl_value in ABL_VALUES
        }
        densities_by_abl = {}
        weights_by_abl = {}
        for abl_value in ABL_VALUES:
            density, weight_map = build_segment_mixture(segment_df, segment_spec, abl_value, abort_params, tied_params, bins_s)
            densities_by_abl[int(abl_value)] = density
            weights_by_abl[int(abl_value)] = weight_map
        segment_results.append(
            {
                "segment_spec": segment_spec,
                "segment_df": segment_df,
                "counts_by_abl": counts_by_abl,
                "total": int(len(segment_df)),
                "densities_by_abl": densities_by_abl,
                "weights_by_abl": weights_by_abl,
            }
        )

    return {
        "batch_name": batch_name,
        "animal_id": animal_id,
        "valid_df": valid_df,
        "abort_params": abort_params,
        "tied_params": tied_params,
        "pkl_path": pkl_path,
        "bins_s": bins_s,
        "segment_results": segment_results,
    }


data = load_data()

# %%
def save_figure(fig, output_base_path):
    fig.savefig(output_base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base_path.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


def plot_data(data):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(SEGMENT_SPECS), 1, figsize=figure_size, sharex=True, sharey=True, squeeze=False)
    x_edges_s = data["bins_s"]

    visible_mask = (x_edges_s[:-1] >= xlim_s[0]) & (x_edges_s[1:] <= xlim_s[1])
    global_max = 0.0
    for segment_result in data["segment_results"]:
        for abl_value in ABL_VALUES:
            density = segment_result["densities_by_abl"][int(abl_value)]
            if np.any(np.isfinite(density)):
                global_max = max(global_max, float(np.nanmax(density[visible_mask])))
    y_max = 1.05 * global_max if global_max > 0 else 1.0

    for row_idx, segment_result in enumerate(data["segment_results"]):
        ax = axes[row_idx, 0]
        segment_spec = segment_result["segment_spec"]
        for abl_value in ABL_VALUES:
            ax.stairs(
                segment_result["densities_by_abl"][int(abl_value)],
                x_edges_s,
                label=f"ABL = {abl_value}",
                color=abl_colors[int(abl_value)],
                linewidth=1.8,
            )
        ax.set_xlim(*xlim_s)
        ax.set_ylim(0, y_max)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_ylabel("Density")
        ax.set_title(
            f"model {segment_spec['name']} [{segment_spec['left']:.3f}, {segment_spec['right']:.3f}] s, "
            f"t_stim={segment_spec['t_stim']:.3f} s"
        )
        if row_idx == len(data["segment_results"]) - 1:
            ax.set_xlabel("RT wrt stim (s)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ABL_VALUES), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    tagged_output_base = output_dir / (
        f"{output_base.name}_{data['batch_name']}_{data['animal_id']}"
    )
    save_figure(fig, tagged_output_base)

    print(f"Using PKL: {data['pkl_path']}")
    print(f"Selected animal: {data['batch_name']}-{data['animal_id']}")
    print(f"Filtered valid trials: {len(data['valid_df'])}")
    print("Norm-tied parameter means:")
    for key, value in data["tied_params"].items():
        print(f"  {key}: {value}")
    print("Abort parameter means:")
    for key, value in data["abort_params"].items():
        print(f"  {key}: {value}")
    for segment_result in data["segment_results"]:
        segment_spec = segment_result["segment_spec"]
        print(
            f"Segment {segment_spec['name']} [{segment_spec['left']:.3f}, {segment_spec['right']:.3f}] s, "
            f"t_stim={segment_spec['t_stim']:.3f} s"
        )
        for abl_value in ABL_VALUES:
            print(f"  ABL={abl_value}, ILD weights={segment_result['weights_by_abl'][int(abl_value)]}")
    print(f"Saved: {tagged_output_base.with_suffix('.pdf')}")
    print(f"Saved: {tagged_output_base.with_suffix('.png')}")

    return fig


fig = plot_data(data)

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)
