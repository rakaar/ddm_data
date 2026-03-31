# %%
SHOW_PLOT = True

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]

MODEL_KEY = "vbmc_norm_tied_results"
ABORT_KEY = "vbmc_aborts_results"
PARAM_REDUCER = "mean"
NUM_INTENDED_FIX_QUANTILE_BINS = 2
PROACTIVE_TRUNC_FIX_TIME_S = {"default": 0.3, "LED34_even": 0.15}
# Set to None to disable truncation entirely

N_MC_T_STIM_SAMPLES = 1000
RNG_SEED = 12345

intended_fix_min_s = 0.2
intended_fix_max_s = 1.5
t_fix_min_s = 0.0
t_fix_max_s = 2.0
model_dt_s = 1e-3
data_bin_size_s = 0.02
xlim_s = (0.15, 1.2)
panel_width = 5.0
panel_height = 4.0
png_dpi = 300

ABL_VALUES = (20, 40, 60)
ILD_VALUES = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)


def get_trunc_time(batch_name):
    if PROACTIVE_TRUNC_FIX_TIME_S is None:
        return None
    return PROACTIVE_TRUNC_FIX_TIME_S.get(
        str(batch_name), PROACTIVE_TRUNC_FIX_TIME_S["default"]
    )


# %%
from pathlib import Path
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent
FIT_DIR = REPO_ROOT / "fit_animal_by_animal"
if str(FIT_DIR) not in sys.path:
    sys.path.append(str(FIT_DIR))

from time_vary_norm_utils import rho_A_t_VEC_fn

# %%
batch_csv_dir = FIT_DIR / "batch_csvs"
results_dir = FIT_DIR
output_dir = SCRIPT_DIR / "abort_dist_early_vs_late_stim"


# ---------------------------------------------------------------------------
# Helper: reduce parameter values
# ---------------------------------------------------------------------------
def reduce_param_values(values):
    values = np.asarray(values, dtype=float)
    if PARAM_REDUCER == "mean":
        return float(np.mean(values))
    if PARAM_REDUCER == "median":
        return float(np.median(values))
    raise ValueError(f"Unknown PARAM_REDUCER: {PARAM_REDUCER}.")


# ---------------------------------------------------------------------------
# Discover eligible (batch, animal) pairs with both abort + model results
# ---------------------------------------------------------------------------
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
            if ABORT_KEY in results and MODEL_KEY in results:
                candidates.append((batch_name, animal_id))
    return candidates


# ---------------------------------------------------------------------------
# Load ALL trials (valid + aborts) for eligible animals
# ---------------------------------------------------------------------------
def load_all_trials_df(candidate_pairs):
    animals_by_batch = {}
    for batch_name, animal_id in candidate_pairs:
        animals_by_batch.setdefault(str(batch_name), set()).add(int(animal_id))

    frames = []
    for batch_name in DESIRED_BATCHES:
        animal_ids = animals_by_batch.get(str(batch_name), set())
        if not animal_ids:
            continue
        batch_file = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"
        if not batch_file.exists():
            continue
        df = pd.read_csv(batch_file)
        if "batch_name" not in df.columns:
            df["batch_name"] = str(batch_name)
        animal_numeric = pd.to_numeric(df["animal"], errors="coerce")
        df = df[animal_numeric.isin(animal_ids)].copy()
        for col in ("ABL", "ILD", "RTwrtStim", "intended_fix", "TotalFixTime", "abort_event", "success"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Apply left-truncation: remove abort_event==3 with TotalFixTime < trunc
        trunc_t = get_trunc_time(batch_name)
        if trunc_t is not None:
            early_abort = np.isclose(df["abort_event"], 3) & (df["TotalFixTime"] < trunc_t)
            df = df[~early_abort].copy()

        # Filter intended_fix range and ABL/ILD
        df = df[
            (df["intended_fix"] >= intended_fix_min_s)
            & (df["intended_fix"] <= intended_fix_max_s)
            & (df["ABL"].isin(ABL_VALUES))
            & (df["ILD"].isin(ILD_VALUES))
        ].copy()
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Quantile segmentation (on valid trials' intended_fix)
# ---------------------------------------------------------------------------
def compute_segment_edges(all_df):
    valid_df = all_df[all_df["success"].isin([1, -1])].copy()
    if_vals = valid_df["intended_fix"].dropna()
    if len(if_vals) <= 1:
        raise ValueError("Cannot segment: too few valid intended_fix values.")
    _, segment_edges = pd.qcut(
        if_vals,
        q=NUM_INTENDED_FIX_QUANTILE_BINS,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    return np.asarray(segment_edges, dtype=float)


def assign_segments(df, segment_edges):
    cut_edges = segment_edges.copy()
    cut_edges[0] = np.nextafter(cut_edges[0], -np.inf)
    cut_edges[-1] = np.nextafter(cut_edges[-1], np.inf)
    seg_ids = pd.cut(
        df["intended_fix"].clip(
            lower=float(segment_edges[0]), upper=float(segment_edges[-1])
        ),
        bins=cut_edges,
        include_lowest=True,
        labels=False,
    ).astype(float)
    out = df.copy()
    out["intended_fix_segment"] = seg_ids
    out = out.dropna(subset=["intended_fix_segment"]).copy()
    out["intended_fix_segment"] = out["intended_fix_segment"].astype(int)
    return out


def build_segment_specs(segment_edges):
    n_segments = len(segment_edges) - 1
    specs = []
    for idx in range(n_segments):
        specs.append({
            "index": idx,
            "name": f"Q{idx + 1}/{n_segments}",
            "left": float(segment_edges[idx]),
            "right": float(segment_edges[idx + 1]),
        })
    return specs


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------
def load_animal_abort_params(batch_name, animal_id):
    pkl_path = results_dir / f"results_{batch_name}_animal_{animal_id}.pkl"
    with open(pkl_path, "rb") as handle:
        results = pickle.load(handle)
    abort_blob = results[ABORT_KEY]
    return {
        "V_A": reduce_param_values(abort_blob["V_A_samples"]),
        "theta_A": reduce_param_values(abort_blob["theta_A_samples"]),
        "t_A_aff": reduce_param_values(abort_blob["t_A_aff_samp"]),
    }


# ---------------------------------------------------------------------------
# Model: proactive abort density in fixation-time coordinates
# ---------------------------------------------------------------------------
def compute_proactive_abort_density(t_fix, abort_params, t_stim_samples, trunc_time):
    """
    Compute the proactive hitting-time density wrt TotalFixTime,
    masked to [trunc_time, t_stim) for each t_stim sample,
    averaged across samples.  NOT renormalized — area = P(abort).

    Parameters
    ----------
    t_fix : 1-D array, fixation-time grid
    abort_params : dict with V_A, theta_A, t_A_aff
    t_stim_samples : 1-D array of intended_fix samples
    trunc_time : float, left-truncation threshold (seconds)

    Returns
    -------
    avg_density : 1-D array, same length as t_fix
    """
    t_fix = np.asarray(t_fix, dtype=float)
    t_stim_samples = np.asarray(t_stim_samples, dtype=float)

    # Proactive density in fixation-time coords (same for all t_stim samples)
    shifted = t_fix - abort_params["t_A_aff"]
    raw_density = rho_A_t_VEC_fn(shifted, abort_params["V_A"], abort_params["theta_A"])
    # raw_density shape: (len(t_fix),)

    # For each t_stim, mask to [trunc_time, t_stim)
    # Vectorized: masks shape (n_samples, n_time)
    left_ok = t_fix >= trunc_time  # (n_time,)
    right_ok = t_fix[None, :] < t_stim_samples[:, None]  # (n_samples, n_time)
    masks = left_ok[None, :] & right_ok  # (n_samples, n_time)

    masked = raw_density[None, :] * masks  # (n_samples, n_time)
    avg_density = np.mean(masked, axis=0)
    return avg_density


# ---------------------------------------------------------------------------
# Main load_data
# ---------------------------------------------------------------------------
def load_data():
    total_start = time.perf_counter()
    print("[progress] load_data() started", flush=True)

    # 1. Discover animals
    candidate_pairs = get_candidate_animals()
    print(f"[progress] Found {len(candidate_pairs)} eligible batch-animal pairs", flush=True)

    # 2. Load all trials
    all_df = load_all_trials_df(candidate_pairs)
    print(f"[progress] Loaded {len(all_df)} trials (valid+abort, after truncation)", flush=True)

    # 3. Segment by intended_fix quantiles
    segment_edges = compute_segment_edges(all_df)
    segment_specs = build_segment_specs(segment_edges)
    all_df = assign_segments(all_df, segment_edges)
    print(f"[progress] Segment edges: {[float(e) for e in segment_edges]}", flush=True)

    # 4. Identify included pairs (those present in data after filtering)
    pairs_df = all_df[["batch_name", "animal"]].dropna().drop_duplicates()
    included_pairs = sorted(
        (str(r.batch_name), int(r.animal))
        for r in pairs_df.itertuples(index=False)
    )
    print(f"[progress] {len(included_pairs)} included animals", flush=True)

    # 5. Fixation-time grid
    t_fix = np.arange(t_fix_min_s, t_fix_max_s + model_dt_s, model_dt_s)
    data_bins = np.arange(t_fix_min_s, t_fix_max_s + data_bin_size_s, data_bin_size_s)

    rng = np.random.default_rng(RNG_SEED)

    # Mark abort and valid
    all_df["is_abort3"] = np.isclose(all_df["abort_event"], 3).astype(int)
    all_df["is_valid"] = all_df["success"].isin([1, -1]).astype(int)

    # 6. Per-animal model curves + data histograms
    # Accumulators: list of per-animal curves for each segment + overall
    n_seg = len(segment_specs)
    seg_model_curves = [[] for _ in range(n_seg)]
    seg_data_hists = [[] for _ in range(n_seg)]
    seg_data_abort_fracs = [[] for _ in range(n_seg)]
    ovr_model_curves = []
    ovr_data_hists = []
    ovr_data_abort_fracs = []

    for pair_idx, (batch_name, animal_id) in enumerate(included_pairs):
        t0 = time.perf_counter()
        print(
            f"[progress] Animal {pair_idx + 1}/{len(included_pairs)}: "
            f"{batch_name}-{animal_id}",
            flush=True,
        )

        abort_params = load_animal_abort_params(batch_name, animal_id)
        trunc_time = get_trunc_time(batch_name)
        if trunc_time is None:
            trunc_time = 0.0

        # Animal's trials
        adf = all_df[
            (all_df["batch_name"].astype(str) == str(batch_name))
            & (np.isclose(all_df["animal"], int(animal_id)))
        ].copy()

        rng_a = np.random.default_rng(RNG_SEED + int(animal_id))

        # --- Per segment ---
        for seg_idx, seg_spec in enumerate(segment_specs):
            seg_df = adf[adf["intended_fix_segment"] == seg_spec["index"]].copy()
            n_total = len(seg_df)
            n_abort = int(seg_df["is_abort3"].sum())

            if n_total == 0:
                continue

            abort_frac = n_abort / n_total

            # Model: proactive density
            if_vals = seg_df["intended_fix"].dropna().values
            if len(if_vals) < 5:
                continue
            t_stim_samp = rng_a.choice(if_vals, size=N_MC_T_STIM_SAMPLES, replace=True)
            model_curve = compute_proactive_abort_density(
                t_fix, abort_params, t_stim_samp, trunc_time
            )
            seg_model_curves[seg_idx].append(model_curve)

            # Data: histogram of TotalFixTime for abort trials, scaled by abort_frac
            abort_trials = seg_df[seg_df["is_abort3"] == 1]
            abort_tft = abort_trials["TotalFixTime"].dropna().values
            if len(abort_tft) > 0:
                hist, _ = np.histogram(abort_tft, bins=data_bins, density=True)
                hist = hist * abort_frac
            else:
                hist = np.zeros(len(data_bins) - 1, dtype=float)
            seg_data_hists[seg_idx].append(hist)
            seg_data_abort_fracs[seg_idx].append(abort_frac)

        # --- Overall ---
        n_total_ovr = len(adf)
        n_abort_ovr = int(adf["is_abort3"].sum())
        if n_total_ovr == 0:
            continue
        abort_frac_ovr = n_abort_ovr / n_total_ovr

        if_vals_ovr = adf["intended_fix"].dropna().values
        if len(if_vals_ovr) < 5:
            continue
        t_stim_samp_ovr = rng_a.choice(if_vals_ovr, size=N_MC_T_STIM_SAMPLES, replace=True)
        model_curve_ovr = compute_proactive_abort_density(
            t_fix, abort_params, t_stim_samp_ovr, trunc_time
        )
        ovr_model_curves.append(model_curve_ovr)

        abort_trials_ovr = adf[adf["is_abort3"] == 1]
        abort_tft_ovr = abort_trials_ovr["TotalFixTime"].dropna().values
        if len(abort_tft_ovr) > 0:
            hist_ovr, _ = np.histogram(abort_tft_ovr, bins=data_bins, density=True)
            hist_ovr = hist_ovr * abort_frac_ovr
        else:
            hist_ovr = np.zeros(len(data_bins) - 1, dtype=float)
        ovr_data_hists.append(hist_ovr)
        ovr_data_abort_fracs.append(abort_frac_ovr)

        print(f"[progress]   done in {time.perf_counter() - t0:.2f}s", flush=True)

    # 7. Aggregate across animals
    segment_results = []
    for seg_idx, seg_spec in enumerate(segment_specs):
        model_avg = (
            np.mean(seg_model_curves[seg_idx], axis=0)
            if seg_model_curves[seg_idx]
            else np.zeros_like(t_fix)
        )
        data_avg = (
            np.mean(seg_data_hists[seg_idx], axis=0)
            if seg_data_hists[seg_idx]
            else np.zeros(len(data_bins) - 1)
        )
        avg_abort_frac = (
            float(np.mean(seg_data_abort_fracs[seg_idx]))
            if seg_data_abort_fracs[seg_idx]
            else 0.0
        )
        segment_results.append({
            "segment_spec": seg_spec,
            "model_density": model_avg,
            "data_density": data_avg,
            "avg_data_abort_frac": avg_abort_frac,
            "n_animals": len(seg_model_curves[seg_idx]),
        })

    ovr_model_avg = (
        np.mean(ovr_model_curves, axis=0)
        if ovr_model_curves
        else np.zeros_like(t_fix)
    )
    ovr_data_avg = (
        np.mean(ovr_data_hists, axis=0)
        if ovr_data_hists
        else np.zeros(len(data_bins) - 1)
    )
    ovr_avg_abort_frac = (
        float(np.mean(ovr_data_abort_fracs))
        if ovr_data_abort_fracs
        else 0.0
    )
    overall_result = {
        "model_density": ovr_model_avg,
        "data_density": ovr_data_avg,
        "avg_data_abort_frac": ovr_avg_abort_frac,
        "n_animals": len(ovr_model_curves),
    }

    print(f"[progress] load_data() finished in {time.perf_counter() - total_start:.2f}s", flush=True)

    return {
        "included_pairs": included_pairs,
        "t_fix": t_fix,
        "data_bins": data_bins,
        "segment_edges": segment_edges,
        "segment_results": segment_results,
        "overall_result": overall_result,
    }


data = load_data()

# %%
def plot_data(data):
    output_dir.mkdir(parents=True, exist_ok=True)

    t_fix = data["t_fix"]
    data_bins = data["data_bins"]
    segment_results = data["segment_results"]
    overall = data["overall_result"]
    segment_edges = data["segment_edges"]

    # 1×3: [Early stim | Late stim | Overall]
    n_panels = len(segment_results) + 1
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(panel_width * n_panels, panel_height),
        squeeze=False,
    )
    axes = axes[0]

    for panel_idx in range(n_panels):
        ax = axes[panel_idx]

        if panel_idx < len(segment_results):
            res = segment_results[panel_idx]
            seg_spec = res["segment_spec"]
            if len(segment_results) == 2:
                seg_name = "Early stim" if panel_idx == 0 else "Late stim"
            else:
                seg_name = seg_spec["name"]
            seg_range_str = f"[{seg_spec['left']:.3f}, {seg_spec['right']:.3f}]s"
        else:
            res = overall
            seg_name = "Overall"
            seg_range_str = f"[{segment_edges[0]:.3f}, {segment_edges[-1]:.3f}]s"

        model_density = res["model_density"]
        data_density = res["data_density"]

        # Compute areas
        model_area = float(np.trapz(model_density, t_fix))
        bin_widths = np.diff(data_bins)
        data_area = float(np.sum(data_density * bin_widths))

        # Data histogram
        ax.stairs(
            data_density, data_bins,
            color="tab:blue", linewidth=1.2, label="Data",
        )

        # Model curve
        ax.plot(
            t_fix, model_density,
            color="tab:red", linewidth=1.5, label="Theory",
        )

        ax.set_xlim(*xlim_s)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_xlabel("TotalFixTime (s)")
        if panel_idx == 0:
            ax.set_ylabel("Density (area = abort fraction)")

        title_lines = [
            f"{seg_name}  {seg_range_str}",
            f"data area={data_area:.4f}, theory area={model_area:.4f}",
            f"avg data abort frac={res['avg_data_abort_frac']:.4f}, "
            f"n_animals={res['n_animals']}",
        ]
        ax.set_title("\n".join(title_lines), fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    n_animals = len(data["included_pairs"])
    out_base = output_dir / f"abort_dist_early_vs_late_{n_animals}animals"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")

    print(f"Included animals ({n_animals}):")
    for batch_name, animal_id in data["included_pairs"]:
        print(f"  {batch_name}-{animal_id}")
    print(f"Segment edges: {[float(e) for e in data['segment_edges']]}")
    for seg_res in data["segment_results"]:
        ss = seg_res["segment_spec"]
        print(
            f"Segment {ss['name']} [{ss['left']:.3f}, {ss['right']:.3f}]s: "
            f"model area={float(np.trapz(seg_res['model_density'], data['t_fix'])):.4f}, "
            f"data area={float(np.sum(seg_res['data_density'] * np.diff(data['data_bins']))):.4f}, "
            f"avg abort frac={seg_res['avg_data_abort_frac']:.4f}"
        )
    print(
        f"Overall: model area={float(np.trapz(data['overall_result']['model_density'], data['t_fix'])):.4f}, "
        f"data area={float(np.sum(data['overall_result']['data_density'] * np.diff(data['data_bins']))):.4f}, "
        f"avg abort frac={data['overall_result']['avg_data_abort_frac']:.4f}"
    )
    print(f"Saved: {out_base.with_suffix('.pdf')}")
    print(f"Saved: {out_base.with_suffix('.png')}")

    return fig


fig = plot_data(data)

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)

# %%
