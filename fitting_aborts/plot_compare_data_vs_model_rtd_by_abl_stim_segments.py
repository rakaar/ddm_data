# %%
SHOW_PLOT = True

DATA_PAYLOAD_DIR = "multi_animal_valid_rtd_cdf_stim_quantiles"
DATA_PAYLOAD_FILENAME = (
    "multi_animal_valid_rtd_by_abl_all_ild_segment_overlay_quantile_segments_plus_abort3_and_4_payload.pkl"
)

MODEL_PAYLOAD_DIR = "model_rtd_stim_segments_all_animals_avg_animal_rtds"
MODEL_PAYLOAD_FILENAME = (
    "model_rtd_by_abl_stim_segments_all_animals_plus_abort3_and_4_pool_quantile_segments_avg_animal_rtds_by_abl_overlay_payload.pkl"
)

OUTPUT_DIR_NAME = "compare_data_vs_model_rtd_by_abl_stim_segments"
OUTPUT_BASENAME = "compare_data_vs_model_rtd_by_abl_stim_segments"

X_LIM_MS = (-500, 1000)
FIGURE_SIZE = (12.0, 3.8)
PNG_DPI = 300


# %%
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


# %%
SCRIPT_DIR = Path(__file__).resolve().parent
data_payload_path = SCRIPT_DIR / DATA_PAYLOAD_DIR / DATA_PAYLOAD_FILENAME
model_payload_path = SCRIPT_DIR / MODEL_PAYLOAD_DIR / MODEL_PAYLOAD_FILENAME
output_dir = SCRIPT_DIR / OUTPUT_DIR_NAME
output_base = output_dir / OUTPUT_BASENAME

segment_colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]


def load_payload(payload_path: Path) -> dict:
    if not payload_path.exists():
        raise FileNotFoundError(
            f"Missing payload file: {payload_path}. Run the source plotting script that produces it first."
        )

    with open(payload_path, "rb") as handle:
        payload = pickle.load(handle)

    if payload.get("payload_kind") != "abl_segment_overlay_rtd":
        raise ValueError(
            f"Unsupported payload kind in {payload_path}: {payload.get('payload_kind')}"
        )
    return payload


def get_sorted_segment_specs(payload: dict) -> list[dict]:
    return sorted(payload["segment_specs"], key=lambda spec: int(spec["index"]))


def compare_payloads(data_payload: dict, model_payload: dict) -> None:
    data_abl_values = [int(value) for value in data_payload["abl_values"]]
    model_abl_values = [int(value) for value in model_payload["abl_values"]]
    if data_abl_values != model_abl_values:
        raise ValueError(
            f"ABL values differ between payloads: data={data_abl_values}, model={model_abl_values}"
        )

    data_segments = get_sorted_segment_specs(data_payload)
    model_segments = get_sorted_segment_specs(model_payload)
    if len(data_segments) != len(model_segments):
        raise ValueError(
            f"Segment count differs between payloads: data={len(data_segments)}, model={len(model_segments)}"
        )

    data_segment_edges = np.asarray(data_payload["segment_edges_s"], dtype=float)
    model_segment_edges = np.asarray(model_payload["segment_edges_s"], dtype=float)
    if data_segment_edges.shape != model_segment_edges.shape:
        raise ValueError(
            "Segment-edge arrays differ between payloads. Regenerate the synced raw payloads before comparing."
        )
    if not np.allclose(data_segment_edges, model_segment_edges, atol=1e-9, rtol=0.0):
        raise ValueError(
            "Segment edges differ between payloads. Regenerate the synced raw payloads before comparing."
        )


def compute_density_area_in_window(density: np.ndarray, x_edges_s: np.ndarray, window_s: tuple[float, float]) -> float:
    density = np.asarray(density, dtype=float)
    x_edges_s = np.asarray(x_edges_s, dtype=float)
    left_s, right_s = window_s
    overlap = np.clip(
        np.minimum(x_edges_s[1:], right_s) - np.maximum(x_edges_s[:-1], left_s),
        0.0,
        None,
    )
    return float(np.nansum(density * overlap))


def get_segment_area(payload: dict, abl_value: int, segment_idx: int) -> float:
    segment_payload = payload["curves_by_abl"][int(abl_value)]["segments"][int(segment_idx)]
    if "area_0_to_1" in segment_payload:
        return float(segment_payload["area_0_to_1"])
    area_window_s = tuple(float(value) for value in payload.get("area_window_s", (0.0, 1.0)))
    return compute_density_area_in_window(segment_payload["density"], payload["x_edges_s"], area_window_s)


def format_area_title(payload_a: dict, payload_b: dict, abl_value: int) -> str:
    segment_specs = get_sorted_segment_specs(payload_a)
    area_lines = []
    for segment_spec in segment_specs:
        segment_idx = int(segment_spec["index"])
        segment_name = str(segment_spec["name"])
        if len(segment_specs) == 2:
            suffix = "Early" if segment_idx == 0 else "Late"
        else:
            suffix = f"S{segment_idx + 1}"
        area_lines.append(
            f"Data_{suffix}={get_segment_area(payload_a, abl_value, segment_idx):.3f} "
            f"Model_{suffix}={get_segment_area(payload_b, abl_value, segment_idx):.3f}"
        )
    return f"ABL = {abl_value}\n" + "\n".join(area_lines)


def build_curve_label(source_name: str, segment_spec: dict) -> str:
    return (
        f"{source_name} {segment_spec['name']} "
        f"[{segment_spec['left']:.3f}, {segment_spec['right']:.3f}] s"
    )


def compute_payload_visible_max(payload: dict, x_lim_ms: tuple[float, float]) -> float:
    x_edges_ms = np.asarray(payload["x_edges_s"], dtype=float) * 1e3
    visible_mask = (x_edges_ms[:-1] >= x_lim_ms[0]) & (x_edges_ms[1:] <= x_lim_ms[1])
    if not np.any(visible_mask):
        return 0.0

    max_value = 0.0
    for abl_value in payload["abl_values"]:
        abl_payload = payload["curves_by_abl"][int(abl_value)]
        for segment_payload in abl_payload["segments"].values():
            density = np.asarray(segment_payload["density"], dtype=float)
            finite_mask = visible_mask & np.isfinite(density)
            if np.any(finite_mask):
                max_value = max(max_value, float(np.nanmax(density[finite_mask])))
    return max_value


def save_figure(fig: plt.Figure, output_base_path: Path) -> None:
    fig.savefig(output_base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base_path.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")


def plot_overlay(data_payload: dict, model_payload: dict) -> plt.Figure:
    abl_values = [int(value) for value in data_payload["abl_values"]]
    data_segment_specs = get_sorted_segment_specs(data_payload)
    model_segment_specs = get_sorted_segment_specs(model_payload)

    fig, axes = plt.subplots(
        1,
        len(abl_values),
        figsize=FIGURE_SIZE,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    data_x_edges_ms = np.asarray(data_payload["x_edges_s"], dtype=float) * 1e3
    model_x_edges_ms = np.asarray(model_payload["x_edges_s"], dtype=float) * 1e3
    y_max = 1.05 * max(
        compute_payload_visible_max(data_payload, X_LIM_MS),
        compute_payload_visible_max(model_payload, X_LIM_MS),
    )
    if y_max <= 0:
        y_max = 1.0

    for col_idx, abl_value in enumerate(abl_values):
        ax = axes[0, col_idx]

        for segment_idx, segment_spec in enumerate(data_segment_specs):
            density = np.asarray(
                data_payload["curves_by_abl"][int(abl_value)]["segments"][int(segment_spec["index"])]["density"],
                dtype=float,
            )
            ax.stairs(
                density,
                data_x_edges_ms,
                color=segment_colors[segment_idx % len(segment_colors)],
                linewidth=1.0,
                label=build_curve_label("Data", segment_spec) if col_idx == 0 else None,
                alpha=0.3
            )

        for segment_idx, segment_spec in enumerate(model_segment_specs):
            density = np.asarray(
                model_payload["curves_by_abl"][int(abl_value)]["segments"][int(segment_spec["index"])]["density"],
                dtype=float,
            )
            ax.stairs(
                density,
                model_x_edges_ms,
                color=segment_colors[segment_idx % len(segment_colors)],
                linewidth=1.0,
                label=build_curve_label("Model", segment_spec) if col_idx == 0 else None,
                alpha = 0.9            )

        ax.set_title(format_area_title(data_payload, model_payload, int(abl_value)))
        ax.set_xlim(*X_LIM_MS)
        ax.set_ylim(0, y_max)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_xlabel("RT wrt stim (ms)")
        if col_idx == 0:
            ax.set_ylabel("Density")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


# %%
output_dir.mkdir(parents=True, exist_ok=True)

data_payload = load_payload(data_payload_path)
model_payload = load_payload(model_payload_path)
compare_payloads(data_payload, model_payload)

print(f"Loaded data payload: {data_payload_path}")
print(f"Loaded model payload: {model_payload_path}")
print(
    "Data segment edges (s): "
    f"{[float(edge) for edge in np.asarray(data_payload['segment_edges_s'], dtype=float)]}"
)
print(
    "Model segment edges (s): "
    f"{[float(edge) for edge in np.asarray(model_payload['segment_edges_s'], dtype=float)]}"
)

fig = plot_overlay(data_payload, model_payload)
save_figure(fig, output_base)

print(f"Saved: {output_base.with_suffix('.pdf')}")
print(f"Saved: {output_base.with_suffix('.png')}")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)

# %%
