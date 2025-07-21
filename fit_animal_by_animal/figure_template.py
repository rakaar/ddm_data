'''Template to generate multi‑panel figures with fully‑parameterised GridSpecs.

You can now specify *any* number of rows/columns (and their ratios) at
instantiation time, so the same class works for Figure 1, Figure 3, or a
one‑off poster‑sized composite.

Quick start
===========
>>> import figure_template as ft
>>> builder = ft.FigureBuilder(
...     sup_title="Figure 3",
...     n_rows=3,
...     n_cols=4,
...     width_ratios=[1, 1, 1, 1],
...     height_ratios=[1, 0.8, 0.6],
... )
>>> # draw panels here using builder.gs[<row, col>]
>>> builder.fig.savefig("fig3.png")

Convenience wrapper
-------------------
If you still prefer the one‑liner that loads data + saves PNG/PDF, use
``build_figure()`` and pass ``grid_shape``/``width_ratios``/``height_ratios``.
'''
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ───────────────────────── Global style ────────────────────────────
@dataclass
class StyleConfig:
    TITLE_FONTSIZE: int = 24
    LABEL_FONTSIZE: int = 25
    TICK_FONTSIZE: int = 24
    LEGEND_FONTSIZE: int = 16
    SUPTITLE_FONTSIZE: int = 24
    font_family: str = "Helvetica"

    def apply(self) -> None:
        mpl.rcParams.update({
            "savefig.pad_inches": 0.6,
            "font.family": "sans-serif",
            "font.sans-serif": [
                self.font_family,
                "Helvetica Neue",
                "TeX Gyre Heros",
                "Arial",
                "sans-serif",
            ],
            "axes.labelpad": 12,
        })


STYLE = StyleConfig()
STYLE.apply()

# ───────────────────────── utilities ──────────────────────────────

def shift_axes(ax_list: Sequence[plt.Axes], dx: float = 0.0, dy: float = 0.0) -> None:
    """Shift *each* axes in *ax_list* by *(dx, dy)* in figure‑fraction coords."""
    for ax in ax_list:
        p = ax.get_position()
        ax.set_position([p.x0 + dx, p.y0 + dy, p.width, p.height])



# ──────────────────────── FigureBuilder ───────────────────────────
@dataclass
class FigureBuilder:
    """A flexible figure skeleton with a parameterised GridSpec."""

    sup_title: str = ""
    figsize: Tuple[int, int] = (12, 8)

    # GridSpec parameters
    n_rows: int = 1
    n_cols: int = 1
    width_ratios: List[float] | None = None
    height_ratios: List[float] | None = None
    hspace: float = 0.3
    wspace: float = 0.3

    # matplotlib artefacts (initialised later)
    fig: plt.Figure = field(init=False)
    gs: GridSpec = field(init=False)

    # ───────── layout creation ─────────
    def __post_init__(self) -> None:
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.subplots_adjust(left=0.06, right=0.97, top=0.96, bottom=0.06)

        # default equal ratios if none supplied
        wr = self.width_ratios or [1] * self.n_cols
        hr = self.height_ratios or [1] * self.n_rows

        self.gs = GridSpec(
            self.n_rows,
            self.n_cols,
            figure=self.fig,
            wspace=self.wspace,
            hspace=self.hspace,
            width_ratios=wr,
            height_ratios=hr,
        )

    # ───────── optional high‑level panel methods (stubs) ─────────
    def plot_custom(self, *_, **__) -> None:  # placeholder
        pass

    # ───────── entry‑point ─────────
    def finish(self) -> plt.Figure:
        """Call after you’re done adding subplots; handles the common footer."""
        self.fig.suptitle(self.sup_title, fontsize=STYLE.SUPTITLE_FONTSIZE)
        return self.fig


# ───────────────────── Convenience wrapper ────────────────────────

def _load_pickle(path: Path | str):
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def build_figure(
    *,
    data_paths: Dict[str, str] | None = None,
    grid_shape: Tuple[int, int] = (5, 6),
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    hspace: float = 0.3,
    wspace: float = 0.3,
    save_png: str | None = None,
    save_pdf: str | None = None,
    sup_title: str = "",
) -> plt.Figure:
    """One‑liner builder that supports an arbitrary GridSpec.

    Returns the *Figure* so callers can keep drawing or display it inline.
    """
    fb = FigureBuilder(
        sup_title=sup_title,
        n_rows=grid_shape[0],
        n_cols=grid_shape[1],
        width_ratios=list(width_ratios) if width_ratios else None,
        height_ratios=list(height_ratios) if height_ratios else None,
        hspace=hspace,
        wspace=wspace,
        figsize=(25, 30),  # keep large default unless caller overrides later
    )

    # If datasets are provided, load them so user code can draw in subplots.
    datasets = {k: _load_pickle(p) for k, p in (data_paths or {}).items()}

    # The caller can now import this function, build the figure, and then use
    #    ax = fb.fig.add_subplot(fb.gs[row, col])
    # to draw panels, or they can subclass *FigureBuilder* and override
    # panel‑specific methods.

    fig = fb.finish()

    if save_png:
        fig.savefig(save_png, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(save_pdf, format="pdf", bbox_inches="tight")

    return fig
