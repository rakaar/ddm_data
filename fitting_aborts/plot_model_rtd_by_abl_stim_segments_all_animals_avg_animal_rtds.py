# %%
SHOW_PLOT = True

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
# DESIRED_BATCHES = ["LED7"]

MODEL_KEY = "vbmc_norm_tied_results"
ABORT_KEY = "vbmc_aborts_results"
PARAM_REDUCER = "mean"
TRIAL_POOL_MODE = "valid"  # "valid" or "valid_plus_abort3"
SEGMENT_MODE = "quantile"  # "quantile" or "fixed"
NUM_INTENDED_FIX_QUANTILE_BINS = 2
FIXED_SEGMENT_EDGES_S = (0.2, 0.4, 1.5)
MODEL_DENSITY_MODE = "valid_conditioned"  # "raw" or "valid_conditioned"
N_JOBS = -1
JOBLIB_VERBOSE = 10

if TRIAL_POOL_MODE not in {"valid", "valid_plus_abort3"}:
    raise ValueError(f"Unsupported TRIAL_POOL_MODE: {TRIAL_POOL_MODE}")
if SEGMENT_MODE not in {"quantile", "fixed"}:
    raise ValueError(f"Unsupported SEGMENT_MODE: {SEGMENT_MODE}")
if MODEL_DENSITY_MODE not in {"raw", "valid_conditioned"}:
    raise ValueError(f"Unsupported MODEL_DENSITY_MODE: {MODEL_DENSITY_MODE}")

ABL_VALUES = (20, 40, 60)
ILD_VALUES = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)
N_MC_T_STIM_SAMPLES = 1000
RNG_SEED = 12345

intended_fix_min_s = 0.2
rt_min_s = -1.0
rt_max_s = 1.0
bin_size_s = 1e-3
intended_fix_max_s = 1.5
xlim_s = (0, 1)
figure_size = (5.0, 6.6)
png_dpi = 300


# %%
from pathlib import Path
import pickle
import sys
import time

from joblib import Parallel, delayed
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
output_dir = SCRIPT_DIR / "model_rtd_stim_segments_all_animals_avg_animal_rtds"
output_suffix_parts = []
if TRIAL_POOL_MODE == "valid_plus_abort3":
    output_suffix_parts.append("plus_abort3_pool")
output_suffix_parts.append(f"{SEGMENT_MODE}_segments")
if MODEL_DENSITY_MODE == "valid_conditioned":
    output_suffix_parts.append("valid_conditioned")
output_suffix_parts.append("avg_animal_rtds")
output_suffix = "".join(f"_{part}" for part in output_suffix_parts)
output_base = output_dir / f"model_rtd_by_abl_stim_segments_all_animals{output_suffix}"

abl_colors = {
    20: "tab:blue",
    40: "tab:orange",
    60: "tab:green",
}


def reduce_param_values(values):
    values = np.asarray(values, dtype=float)
    if PARAM_REDUCER == "mean":
        return float(np.mean(values))
    if PARAM_REDUCER == "median":
        return float(np.median(values))
    raise ValueError(f"Unknown PARAM_REDUCER: {PARAM_REDUCER}. Use 'mean' or 'median'.")


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
                candidates.append((str(batch_name), animal_id))
    return sorted(candidates)


def load_pooled_valid_df(candidate_pairs):
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
        if TRIAL_POOL_MODE == "valid_plus_abort3":
            if "abort_event" not in df.columns:
                raise ValueError(
                    f"abort_event column is required when TRIAL_POOL_MODE='valid_plus_abort3': {batch_file}"
                )
            df["abort_event"] = pd.to_numeric(df["abort_event"], errors="coerce")
            df = df[df["success"].isin([1, -1]) | np.isclose(df["abort_event"], 3)].copy()
        else:
            df = df[df["success"].isin([1, -1])].copy()
        df["animal"] = pd.to_numeric(df["animal"], errors="coerce")
        df["RTwrtStim"] = pd.to_numeric(df["RTwrtStim"], errors="coerce")
        df["intended_fix"] = pd.to_numeric(df["intended_fix"], errors="coerce")
        df["ABL"] = pd.to_numeric(df["ABL"], errors="coerce")
        df["ILD"] = pd.to_numeric(df["ILD"], errors="coerce")
        df = df[
            (df["RTwrtStim"] >= rt_min_s)
            & (df["RTwrtStim"] <= rt_max_s)
            & (df["intended_fix"] >= intended_fix_min_s)
            & (df["intended_fix"] <= intended_fix_max_s)
            & (df["ABL"].isin(ABL_VALUES))
            & (df["ILD"].isin(ILD_VALUES))
        ].copy()
        frames.append(df)

    if not frames:
        raise ValueError("No pooled segment-pool trials found for eligible animals after filtering.")
    return pd.concat(frames, ignore_index=True)


def get_included_pairs_from_df(df):
    pairs_df = df[["batch_name", "animal"]].dropna().copy()
    pairs_df["batch_name"] = pairs_df["batch_name"].astype(str)
    pairs_df["animal"] = pd.to_numeric(pairs_df["animal"], errors="coerce")
    pairs_df = pairs_df.dropna().copy()
    pairs_df = pairs_df.drop_duplicates(subset=["batch_name", "animal"]).copy()
    return sorted((str(row.batch_name), int(row.animal)) for row in pairs_df.itertuples(index=False))


def load_animal_params(batch_name, animal_id):
    pkl_path = results_dir / f"results_{batch_name}_animal_{animal_id}.pkl"
    with open(pkl_path, "rb") as handle:
        results = pickle.load(handle)

    abort_blob = results[ABORT_KEY]
    model_blob = results[MODEL_KEY]

    abort_params = {
        "V_A": reduce_param_values(abort_blob["V_A_samples"]),
        "theta_A": reduce_param_values(abort_blob["theta_A_samples"]),
        "t_A_aff": reduce_param_values(abort_blob["t_A_aff_samp"]),
    }
    tied_params = {
        "rate_lambda": reduce_param_values(model_blob["rate_lambda_samples"]),
        "T_0": reduce_param_values(model_blob["T_0_samples"]),
        "theta_E": reduce_param_values(model_blob["theta_E_samples"]),
        "w": reduce_param_values(model_blob["w_samples"]),
        "t_E_aff": reduce_param_values(model_blob["t_E_aff_samples"]),
        "del_go": reduce_param_values(model_blob["del_go_samples"]),
        "rate_norm_l": reduce_param_values(model_blob["rate_norm_l_samples"]),
    }
    return abort_params, tied_params, pkl_path


def add_intended_fix_segments(df):
    if SEGMENT_MODE == "quantile":
        if NUM_INTENDED_FIX_QUANTILE_BINS <= 0:
            raise ValueError("NUM_INTENDED_FIX_QUANTILE_BINS must be positive.")
        if df["intended_fix"].isna().any():
            raise ValueError("Cannot build quantile segments with NaN intended_fix values.")
        if float(df["intended_fix"].nunique()) <= 1:
            raise ValueError("Cannot segment intended_fix because all filtered values are identical.")

        segment_ids, segment_edges = pd.qcut(
            df["intended_fix"],
            q=NUM_INTENDED_FIX_QUANTILE_BINS,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        if len(segment_edges) - 1 != NUM_INTENDED_FIX_QUANTILE_BINS:
            raise ValueError(
                "Requested "
                f"{NUM_INTENDED_FIX_QUANTILE_BINS} quantile bins, but only {len(segment_edges) - 1} unique bins could be formed."
            )
        if segment_ids.isna().any():
            raise ValueError("Failed to assign intended_fix quantile segments to some trials.")

        segmented_df = df.copy()
        segmented_df["intended_fix_segment"] = segment_ids.astype(int)
        return segmented_df, np.asarray(segment_edges, dtype=float)

    segment_edges = np.asarray(FIXED_SEGMENT_EDGES_S, dtype=float)
    if len(segment_edges) < 3:
        raise ValueError("FIXED_SEGMENT_EDGES_S must define at least two segments.")

    cut_edges = segment_edges.copy()
    cut_edges[0] = np.nextafter(cut_edges[0], -np.inf)
    cut_edges[-1] = np.nextafter(cut_edges[-1], np.inf)
    segment_ids = pd.cut(
        df["intended_fix"],
        bins=cut_edges,
        labels=False,
        include_lowest=True,
        right=True,
    )
    if segment_ids.isna().any():
        raise ValueError("Failed to assign intended_fix fixed segments to some trials.")

    segmented_df = df.copy()
    segmented_df["intended_fix_segment"] = segment_ids.astype(int)
    return segmented_df, segment_edges


def build_segment_specs(segment_edges):
    n_segments = len(segment_edges) - 1
    segment_specs = []
    for segment_idx in range(n_segments):
        segment_specs.append(
            {
                "index": int(segment_idx),
                "name": f"Q{segment_idx + 1}/{n_segments}",
                "left": float(segment_edges[segment_idx]),
                "right": float(segment_edges[segment_idx + 1]),
            }
        )
    return segment_specs


def compute_segment_proactive_curves(t_stim_samples, abort_params):
    t_pts = np.arange(-1.0, 1.001, 0.001)
    shifted_t = t_pts[None, :] + np.asarray(t_stim_samples, dtype=float)[:, None] - abort_params["t_A_aff"]
    p_a_samples = rho_A_t_VEC_fn(shifted_t, abort_params["V_A"], abort_params["theta_A"])
    p_a_mean = np.mean(p_a_samples, axis=0)
    c_a_mean = cumulative_trapezoid(p_a_mean, t_pts, initial=0)
    return t_pts, p_a_mean, c_a_mean


def compute_segment_proactive_matrices(t_stim_samples, abort_params):
    t_pts = np.arange(-1.0, 1.001, 0.001)
    shifted_t = t_pts[None, :] + np.asarray(t_stim_samples, dtype=float)[:, None] - abort_params["t_A_aff"]
    p_a_samples = rho_A_t_VEC_fn(shifted_t, abort_params["V_A"], abort_params["theta_A"])
    c_a_samples = cumulative_trapezoid(p_a_samples, t_pts, axis=1, initial=0)
    return t_pts, p_a_samples, c_a_samples


def compute_raw_rtd_from_pa_ca(t_pts, p_a, c_a, abl_value, ild_value, tied_params):
    z_e = (tied_params["w"] - 0.5) * 2.0 * tied_params["theta_E"]
    up = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
        t_pts,
        1,
        p_a,
        c_a,
        abl_value,
        ild_value,
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
        t_pts,
        -1,
        p_a,
        c_a,
        abl_value,
        ild_value,
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


def compute_segment_averaged_rtd(t_pts, p_a_mean, c_a_mean, abl_value, ild_value, tied_params):
    density = compute_raw_rtd_from_pa_ca(t_pts, p_a_mean, c_a_mean, abl_value, ild_value, tied_params)
    mask = (t_pts >= rt_min_s) & (t_pts <= rt_max_s)
    return t_pts[mask], density[mask]


def curve_to_binned_density(t_pts, density, bins_s):
    cdf = cumulative_trapezoid(density, t_pts, initial=0)
    edge_cdf = np.interp(bins_s, t_pts, cdf, left=0.0, right=float(cdf[-1]))
    probs = np.diff(edge_cdf)
    widths = np.diff(bins_s)
    return probs / widths


def build_animal_segment_mixture(segment_df, t_pts, p_a_obj, c_a_obj, abl_value, tied_params, bins_s):
    abl_df = segment_df[np.isclose(segment_df["ABL"], abl_value)].copy()
    counts = abl_df["ILD"].round().astype(int).value_counts().to_dict() if len(abl_df) else {}
    total = int(len(abl_df))

    if total == 0:
        return (
            np.full(len(bins_s) - 1, np.nan, dtype=float),
            np.full(len(ILD_VALUES), np.nan, dtype=float),
            0,
        )

    mixture = np.zeros(len(bins_s) - 1, dtype=float)
    weight_vector = np.zeros(len(ILD_VALUES), dtype=float)
    valid_mask = t_pts >= 0.0

    for ild_idx, ild_value in enumerate(ILD_VALUES):
        count = int(counts.get(int(ild_value), 0))
        weight = count / total
        weight_vector[ild_idx] = weight
        if count == 0:
            continue

        if MODEL_DENSITY_MODE == "valid_conditioned":
            raw_matrix = compute_raw_rtd_from_pa_ca(
                t_pts[None, :],
                p_a_obj,
                c_a_obj,
                abl_value,
                float(ild_value),
                tied_params,
            )
            if raw_matrix.ndim != 2:
                raise ValueError("Expected a 2D RTD matrix when MODEL_DENSITY_MODE='valid_conditioned'.")
            conditioned_matrix = np.where(valid_mask[None, :], raw_matrix, 0.0)
            valid_area_per_t_stim = np.trapz(conditioned_matrix[:, valid_mask], t_pts[valid_mask], axis=1)
            valid_area_per_t_stim = np.clip(valid_area_per_t_stim, 1e-12, None)
            conditioned_matrix = conditioned_matrix / valid_area_per_t_stim[:, None]
            density = np.mean(conditioned_matrix, axis=0)
            mixture += weight * curve_to_binned_density(t_pts, density, bins_s)
        else:
            rt_axis, density = compute_segment_averaged_rtd(
                t_pts,
                p_a_obj,
                c_a_obj,
                abl_value,
                float(ild_value),
                tied_params,
            )
            mixture += weight * curve_to_binned_density(rt_axis, density, bins_s)

    return mixture, weight_vector, total


def compute_one_animal_rtd(batch_name, animal_id, animal_df, segment_specs, bins_s, seed):
    abort_params, tied_params, pkl_path = load_animal_params(batch_name, animal_id)
    rng = np.random.default_rng(seed)
    animal_segment_results = []

    for segment_spec in segment_specs:
        segment_df = animal_df[animal_df["intended_fix_segment"] == segment_spec["index"]].copy()
        segment_trial_count = int(len(segment_df))
        t_stim_summary = None

        if segment_trial_count == 0:
            densities_by_abl = {
                int(abl_value): np.full(len(bins_s) - 1, np.nan, dtype=float)
                for abl_value in ABL_VALUES
            }
            weights_by_abl = {
                int(abl_value): np.full(len(ILD_VALUES), np.nan, dtype=float)
                for abl_value in ABL_VALUES
            }
            trial_counts_by_abl = {int(abl_value): 0 for abl_value in ABL_VALUES}
        else:
            t_stim_samples = rng.choice(segment_df["intended_fix"].to_numpy(), size=N_MC_T_STIM_SAMPLES, replace=True)
            t_stim_summary = {
                "mean": float(np.mean(t_stim_samples)),
                "min": float(np.min(t_stim_samples)),
                "max": float(np.max(t_stim_samples)),
                "n_mc": int(len(t_stim_samples)),
            }

            if MODEL_DENSITY_MODE == "valid_conditioned":
                t_pts, p_a_obj, c_a_obj = compute_segment_proactive_matrices(t_stim_samples, abort_params)
            else:
                t_pts, p_a_obj, c_a_obj = compute_segment_proactive_curves(t_stim_samples, abort_params)

            densities_by_abl = {}
            weights_by_abl = {}
            trial_counts_by_abl = {}
            for abl_value in ABL_VALUES:
                density, weight_vector, trial_count = build_animal_segment_mixture(
                    segment_df,
                    t_pts,
                    p_a_obj,
                    c_a_obj,
                    float(abl_value),
                    tied_params,
                    bins_s,
                )
                densities_by_abl[int(abl_value)] = density
                weights_by_abl[int(abl_value)] = weight_vector
                trial_counts_by_abl[int(abl_value)] = int(trial_count)

        animal_segment_results.append(
            {
                "segment_spec": segment_spec,
                "total": segment_trial_count,
                "densities_by_abl": densities_by_abl,
                "weights_by_abl": weights_by_abl,
                "trial_counts_by_abl": trial_counts_by_abl,
                "t_stim_summary": t_stim_summary,
            }
        )

    return {
        "pair": (str(batch_name), int(animal_id)),
        "pkl_path": pkl_path,
        "segment_results": animal_segment_results,
    }


def aggregate_animal_results(per_animal_results, segment_specs, bins_s):
    aggregated_segment_results = []
    ild_values_array = np.asarray(ILD_VALUES, dtype=int)

    for segment_idx, segment_spec in enumerate(segment_specs):
        segment_totals = np.asarray(
            [animal_result["segment_results"][segment_idx]["total"] for animal_result in per_animal_results],
            dtype=int,
        )

        densities_by_abl = {}
        contributing_animals_by_abl = {}
        trial_counts_by_abl = {}
        mean_weights_by_abl = {}
        mean_t_stim_summary = {
            "n_mc": N_MC_T_STIM_SAMPLES,
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

        t_stim_means = []
        t_stim_mins = []
        t_stim_maxs = []

        for animal_result in per_animal_results:
            summary = animal_result["segment_results"][segment_idx]["t_stim_summary"]
            if summary is None:
                continue
            t_stim_means.append(summary["mean"])
            t_stim_mins.append(summary["min"])
            t_stim_maxs.append(summary["max"])

        if t_stim_means:
            mean_t_stim_summary = {
                "n_mc": N_MC_T_STIM_SAMPLES,
                "mean": float(np.mean(t_stim_means)),
                "min": float(np.min(t_stim_mins)),
                "max": float(np.max(t_stim_maxs)),
            }

        for abl_value in ABL_VALUES:
            density_list = []
            weight_list = []
            abl_trial_counts = []

            for animal_result in per_animal_results:
                animal_segment_result = animal_result["segment_results"][segment_idx]
                density_list.append(animal_segment_result["densities_by_abl"][int(abl_value)])
                weight_list.append(animal_segment_result["weights_by_abl"][int(abl_value)])
                abl_trial_counts.append(animal_segment_result["trial_counts_by_abl"][int(abl_value)])

            density_stack = np.stack(density_list, axis=0)
            weight_stack = np.stack(weight_list, axis=0)
            abl_trial_counts = np.asarray(abl_trial_counts, dtype=int)
            contributing_mask = abl_trial_counts > 0

            if np.any(contributing_mask):
                densities_by_abl[int(abl_value)] = np.nanmean(density_stack[contributing_mask], axis=0)
                mean_weight_vector = np.nanmean(weight_stack[contributing_mask], axis=0)
            else:
                densities_by_abl[int(abl_value)] = np.full(len(bins_s) - 1, np.nan, dtype=float)
                mean_weight_vector = np.full(len(ILD_VALUES), np.nan, dtype=float)

            contributing_animals_by_abl[int(abl_value)] = int(np.sum(contributing_mask))
            trial_counts_by_abl[int(abl_value)] = int(np.sum(abl_trial_counts))
            mean_weights_by_abl[int(abl_value)] = {
                int(ild_value): float(mean_weight_vector[ild_idx])
                for ild_idx, ild_value in enumerate(ild_values_array)
                if np.isfinite(mean_weight_vector[ild_idx])
            }

        aggregated_segment_results.append(
            {
                "segment_spec": segment_spec,
                "total": int(np.sum(segment_totals)),
                "contributing_animals_total": int(np.sum(segment_totals > 0)),
                "densities_by_abl": densities_by_abl,
                "contributing_animals_by_abl": contributing_animals_by_abl,
                "trial_counts_by_abl": trial_counts_by_abl,
                "mean_weights_by_abl": mean_weights_by_abl,
                "mean_t_stim_summary": mean_t_stim_summary,
            }
        )

    return aggregated_segment_results


def load_data():
    total_start = time.perf_counter()
    print("[progress] load_data() started", flush=True)

    stage_start = time.perf_counter()
    candidate_pairs = get_candidate_animals()
    print(
        f"[progress] Found {len(candidate_pairs)} eligible batch-animal pairs in {time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    stage_start = time.perf_counter()
    pooled_valid_df = load_pooled_valid_df(candidate_pairs)
    print(
        f"[progress] Loaded pooled segment dataframe with {len(pooled_valid_df)} rows in {time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    stage_start = time.perf_counter()
    pooled_valid_df, segment_edges = add_intended_fix_segments(pooled_valid_df)
    segment_specs = build_segment_specs(segment_edges)
    print(
        "[progress] Built intended_fix segments "
        f"({SEGMENT_MODE}) with edges {[float(edge) for edge in segment_edges]} "
        f"in {time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    stage_start = time.perf_counter()
    included_pairs = get_included_pairs_from_df(pooled_valid_df)
    if not included_pairs:
        raise ValueError("No eligible batch-animal pairs remained after filtering pooled segment-pool trials.")
    print(
        f"[progress] Retained {len(included_pairs)} included pairs in {time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    pair_to_animal_df = {}
    for batch_name, animal_id in included_pairs:
        animal_df = pooled_valid_df[
            (pooled_valid_df["batch_name"].astype(str) == str(batch_name))
            & np.isclose(pooled_valid_df["animal"], int(animal_id))
        ].copy()
        if len(animal_df) == 0:
            continue
        pair_to_animal_df[(str(batch_name), int(animal_id))] = animal_df

    if not pair_to_animal_df:
        raise ValueError("No per-animal dataframes found after filtering.")

    bins_s = np.arange(rt_min_s, rt_max_s + bin_size_s, bin_size_s)

    stage_start = time.perf_counter()
    print(
        f"[progress] Starting per-animal RTD computation for {len(included_pairs)} animals with N_JOBS={N_JOBS}",
        flush=True,
    )
    per_animal_results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(compute_one_animal_rtd)(
            batch_name,
            animal_id,
            pair_to_animal_df[(str(batch_name), int(animal_id))],
            segment_specs,
            bins_s,
            RNG_SEED + pair_idx,
        )
        for pair_idx, (batch_name, animal_id) in enumerate(included_pairs)
    )
    print(
        f"[progress] Computed per-animal RTDs in {time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    stage_start = time.perf_counter()
    segment_results = aggregate_animal_results(per_animal_results, segment_specs, bins_s)
    print(
        f"[progress] Aggregated per-animal RTDs in {time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    pkl_paths = [result["pkl_path"] for result in per_animal_results]

    print(f"[progress] load_data() finished in {time.perf_counter() - total_start:.2f}s", flush=True)

    return {
        "included_pairs": included_pairs,
        "valid_df": pooled_valid_df,
        "pkl_paths": pkl_paths,
        "bins_s": bins_s,
        "segment_edges": segment_edges,
        "segment_results": segment_results,
        "per_animal_results": per_animal_results,
    }


data = load_data()


# %%
def save_figure(fig, output_base_path):
    fig.savefig(output_base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base_path.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


def plot_data(data):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(data["segment_results"]),
        1,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_edges_s = data["bins_s"]

    visible_mask = (x_edges_s[:-1] >= xlim_s[0]) & (x_edges_s[1:] <= xlim_s[1])
    global_max = 0.0
    for segment_result in data["segment_results"]:
        for abl_value in ABL_VALUES:
            density = segment_result["densities_by_abl"][int(abl_value)]
            finite_density = density[np.isfinite(density)]
            if len(finite_density):
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
            f"avg per-animal model {segment_spec['name']} [{segment_spec['left']:.3f}, {segment_spec['right']:.3f}] s\n"
            f"animals={segment_result['contributing_animals_total']}, per-animal MC intended_fix n={N_MC_T_STIM_SAMPLES}"
        )
        if row_idx == len(data["segment_results"]) - 1:
            ax.set_xlabel("RT wrt stim (s)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ABL_VALUES), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    tagged_output_base = output_dir / f"{output_base.name}_{len(data['included_pairs'])}animals"
    save_figure(fig, tagged_output_base)

    fig_by_abl, axes_by_abl = plt.subplots(1, len(ABL_VALUES), figsize=(12.0, 3.8), sharex=True, sharey=True, squeeze=False)
    early_segment_result = data["segment_results"][0]
    late_segment_result = data["segment_results"][-1]

    global_max_by_abl = 0.0
    for abl_value in ABL_VALUES:
        early_density = early_segment_result["densities_by_abl"][int(abl_value)]
        late_density = late_segment_result["densities_by_abl"][int(abl_value)]
        if np.any(np.isfinite(early_density)):
            global_max_by_abl = max(global_max_by_abl, float(np.nanmax(early_density[visible_mask])))
        if np.any(np.isfinite(late_density)):
            global_max_by_abl = max(global_max_by_abl, float(np.nanmax(late_density[visible_mask])))
    y_max_by_abl = 1.05 * global_max_by_abl if global_max_by_abl > 0 else 1.0

    for col_idx, abl_value in enumerate(ABL_VALUES):
        ax = axes_by_abl[0, col_idx]
        ax.stairs(
            early_segment_result["densities_by_abl"][int(abl_value)],
            x_edges_s,
            label=early_segment_result["segment_spec"]["name"],
            color="tab:blue",
            linewidth=1.8,
        )
        ax.stairs(
            late_segment_result["densities_by_abl"][int(abl_value)],
            x_edges_s,
            label=late_segment_result["segment_spec"]["name"],
            color="tab:red",
            linewidth=1.8,
        )
        ax.set_xlim(*xlim_s)
        ax.set_ylim(0, y_max_by_abl)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_title(f"ABL = {abl_value}")
        ax.set_xlabel("RT wrt stim (s)")
        if col_idx == 0:
            ax.set_ylabel("Density")

    handles_by_abl, labels_by_abl = axes_by_abl[0, 0].get_legend_handles_labels()
    fig_by_abl.legend(handles_by_abl, labels_by_abl, loc="upper center", ncol=2, frameon=False)
    fig_by_abl.tight_layout(rect=(0, 0, 1, 0.90))

    tagged_output_base_by_abl = output_dir / f"{output_base.name}_{len(data['included_pairs'])}animals_by_abl"
    save_figure(fig_by_abl, tagged_output_base_by_abl)

    print(f"Included animals ({len(data['included_pairs'])}):")
    for batch_name, animal_id in data["included_pairs"]:
        print(f"  {batch_name}-{animal_id}")
    print(f"TRIAL_POOL_MODE: {TRIAL_POOL_MODE}")
    print(f"SEGMENT_MODE: {SEGMENT_MODE}")
    print(f"MODEL_DENSITY_MODE: {MODEL_DENSITY_MODE}")
    print(f"N_JOBS: {N_JOBS}")
    print(f"Filtered pooled segment-pool trials: {len(data['valid_df'])}")
    print(f"Segment edges (s): {[float(edge) for edge in data['segment_edges']]}")
    print(f"PKLs used: {len(data['pkl_paths'])}")
    for segment_result in data["segment_results"]:
        segment_spec = segment_result["segment_spec"]
        summary = segment_result["mean_t_stim_summary"]
        print(
            f"Segment {segment_spec['name']} [{segment_spec['left']:.3f}, {segment_spec['right']:.3f}] s, "
            f"animals={segment_result['contributing_animals_total']}, total_trials={segment_result['total']}, "
            f"per-animal MC intended_fix n={summary['n_mc']}, mean={summary['mean']:.3f}, min={summary['min']:.3f}, max={summary['max']:.3f}"
        )
        for abl_value in ABL_VALUES:
            print(
                f"  ABL={abl_value}, animals={segment_result['contributing_animals_by_abl'][int(abl_value)]}, "
                f"trials={segment_result['trial_counts_by_abl'][int(abl_value)]}, "
                f"mean ILD weights={segment_result['mean_weights_by_abl'][int(abl_value)]}"
            )
    print(f"Saved: {tagged_output_base.with_suffix('.pdf')}")
    print(f"Saved: {tagged_output_base.with_suffix('.png')}")
    print(f"Saved: {tagged_output_base_by_abl.with_suffix('.pdf')}")
    print(f"Saved: {tagged_output_base_by_abl.with_suffix('.png')}")

    return fig, fig_by_abl


fig, fig_by_abl = plot_data(data)

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)
    plt.close(fig_by_abl)

# %%
