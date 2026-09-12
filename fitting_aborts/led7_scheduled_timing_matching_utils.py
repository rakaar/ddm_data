# %%
"""Reusable integer-flow matching for LED7 scheduled-timing analyses."""

# %%
import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance


# %%
def bin_indices(values, bin_edges):
    values = np.asarray(values, dtype=float)
    indices = np.searchsorted(bin_edges, values, side="right") - 1
    on_last_edge = np.isclose(values, bin_edges[-1], atol=1e-12, rtol=0)
    indices[on_last_edge] = len(bin_edges) - 2
    in_range = (
        np.isfinite(values)
        & (values >= bin_edges[0])
        & (values <= bin_edges[-1])
        & (indices >= 0)
        & (indices < len(bin_edges) - 1)
    )
    return indices, in_range


def largest_remainder_counts(reference_counts, output_total):
    reference_counts = np.asarray(reference_counts, dtype=int)
    if reference_counts.sum() <= 0:
        raise ValueError("Reference histogram is empty.")

    expected = output_total * reference_counts / reference_counts.sum()
    allocated = np.floor(expected).astype(int)
    remainder = int(output_total - allocated.sum())
    if remainder:
        order = np.argsort(-(expected - allocated), kind="stable")
        allocated[order[:remainder]] += 1

    if int(allocated.sum()) != output_total:
        raise RuntimeError("Largest-remainder allocation has the wrong total.")
    return allocated


def select_rows_matching_two_margins(
    timing_frame,
    target_intended_counts,
    target_onset_counts,
    rng,
    *,
    matching_columns,
    intended_bins,
    onset_bins,
    sample_size,
    bin_width_s,
):
    """Choose feasible joint-cell quotas matching two requested marginals."""
    if list(timing_frame.columns) != list(matching_columns):
        raise ValueError(
            f"Matcher expected only {list(matching_columns)}, found "
            f"{list(timing_frame.columns)}."
        )
    if not timing_frame.index.is_unique:
        raise RuntimeError("Candidate source-row indices are not unique.")

    intended_indices, intended_in_range = bin_indices(
        timing_frame["intended_fix"], intended_bins
    )
    onset_indices, onset_in_range = bin_indices(
        timing_frame["effective_scheduled_onset"], onset_bins
    )
    in_matching_support = intended_in_range & onset_in_range

    work = timing_frame.loc[in_matching_support].copy()
    work["_intended_bin"] = intended_indices[in_matching_support]
    work["_onset_bin"] = onset_indices[in_matching_support]
    if len(work) < sample_size:
        raise RuntimeError(
            f"Only {len(work):,} candidate rows are in matching support; "
            f"need {sample_size:,}."
        )

    cell_counts = (
        work.groupby(["_intended_bin", "_onset_bin"], observed=True)
        .size()
        .astype(int)
    )
    actual_intended_bins = sorted(
        work["_intended_bin"].astype(int).unique().tolist()
    )
    actual_onset_bins = sorted(
        work["_onset_bin"].astype(int).unique().tolist()
    )

    # Route target intended-fix counts through observed two-dimensional cells
    # and into target onset counts. Integer flow through each observed cell is
    # the number of source rows retained from that cell.
    graph = nx.DiGraph()
    for target_intended_bin, target_count in enumerate(target_intended_counts):
        if target_count > 0:
            graph.add_node(
                f"target_intended_{target_intended_bin}",
                demand=-int(target_count),
            )

    for actual_intended_bin in actual_intended_bins:
        graph.add_node(f"actual_intended_{actual_intended_bin}", demand=0)
    for actual_onset_bin in actual_onset_bins:
        graph.add_node(f"actual_onset_{actual_onset_bin}", demand=0)

    for target_onset_bin, target_count in enumerate(target_onset_counts):
        if target_count > 0:
            graph.add_node(
                f"target_onset_{target_onset_bin}",
                demand=int(target_count),
            )

    for target_intended_bin, target_count in enumerate(target_intended_counts):
        if target_count == 0:
            continue
        for actual_intended_bin in actual_intended_bins:
            graph.add_edge(
                f"target_intended_{target_intended_bin}",
                f"actual_intended_{actual_intended_bin}",
                capacity=int(target_count),
                weight=abs(target_intended_bin - actual_intended_bin),
            )

    for (intended_bin, onset_bin), capacity in cell_counts.items():
        graph.add_edge(
            f"actual_intended_{int(intended_bin)}",
            f"actual_onset_{int(onset_bin)}",
            capacity=int(capacity),
            weight=0,
        )

    for actual_onset_bin in actual_onset_bins:
        for target_onset_bin, target_count in enumerate(target_onset_counts):
            if target_count == 0:
                continue
            graph.add_edge(
                f"actual_onset_{actual_onset_bin}",
                f"target_onset_{target_onset_bin}",
                capacity=int(target_count),
                weight=abs(actual_onset_bin - target_onset_bin),
            )

    flow_cost, flow = nx.network_simplex(graph)
    selected_per_cell = {}
    for (intended_bin, onset_bin), capacity in cell_counts.items():
        selected_count = int(
            flow[f"actual_intended_{int(intended_bin)}"].get(
                f"actual_onset_{int(onset_bin)}", 0
            )
        )
        if selected_count < 0 or selected_count > int(capacity):
            raise RuntimeError("Network-flow selection exceeds a cell capacity.")
        if selected_count:
            selected_per_cell[(int(intended_bin), int(onset_bin))] = selected_count

    if sum(selected_per_cell.values()) != sample_size:
        raise RuntimeError("Network-flow selection has the wrong total.")

    selected_indices = []
    for (intended_bin, onset_bin), selected_count in sorted(
        selected_per_cell.items()
    ):
        cell_source_indices = work.index[
            work["_intended_bin"].eq(intended_bin)
            & work["_onset_bin"].eq(onset_bin)
        ].to_numpy()
        chosen = rng.choice(
            cell_source_indices,
            size=selected_count,
            replace=False,
        )
        selected_indices.extend(chosen.tolist())

    selected_indices = np.asarray(selected_indices)
    rng.shuffle(selected_indices)
    if len(selected_indices) != sample_size:
        raise RuntimeError("Sampled source-index count is incorrect.")
    if len(np.unique(selected_indices)) != sample_size:
        raise RuntimeError("Matched selection contains repeated source rows.")
    if not np.isin(selected_indices, timing_frame.index.to_numpy()).all():
        raise RuntimeError("Matched selection contains a non-candidate row.")

    selected_timing = timing_frame.loc[selected_indices]
    selected_intended_counts, _ = np.histogram(
        selected_timing["intended_fix"], bins=intended_bins
    )
    selected_onset_counts, _ = np.histogram(
        selected_timing["effective_scheduled_onset"], bins=onset_bins
    )
    intended_l1 = int(
        np.abs(selected_intended_counts - target_intended_counts).sum()
    )
    onset_l1 = int(np.abs(selected_onset_counts - target_onset_counts).sum())
    intended_centers = 0.5 * (intended_bins[:-1] + intended_bins[1:])
    onset_centers = 0.5 * (onset_bins[:-1] + onset_bins[1:])
    intended_binned_wasserstein_ms = float(
        1000
        * wasserstein_distance(
            intended_centers,
            intended_centers,
            u_weights=selected_intended_counts,
            v_weights=target_intended_counts,
        )
    )
    onset_binned_wasserstein_ms = float(
        1000
        * wasserstein_distance(
            onset_centers,
            onset_centers,
            u_weights=selected_onset_counts,
            v_weights=target_onset_counts,
        )
    )
    expected_flow_cost_ms = 1000 * bin_width_s * flow_cost / sample_size
    if not np.isclose(
        intended_binned_wasserstein_ms + onset_binned_wasserstein_ms,
        expected_flow_cost_ms,
        atol=1e-10,
        rtol=0,
    ):
        raise RuntimeError(
            "Network-flow cost does not match the two binned marginal "
            "Wasserstein distances."
        )

    return selected_indices, {
        "candidate_rows": len(timing_frame),
        "candidate_rows_in_matching_support": len(work),
        "flow_bin_distance_cost": int(flow_cost),
        "intended_binned_W_ms": intended_binned_wasserstein_ms,
        "onset_binned_W_ms": onset_binned_wasserstein_ms,
        "intended_bin_l1": intended_l1,
        "onset_bin_l1": onset_l1,
    }
