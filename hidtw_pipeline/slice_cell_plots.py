"""Per-slice statistics and raw 2D-cell visualisations.

Public functions:
    slice_summary_table  - table of cells per slice with area / aspect stats
    global_xy_bounds     - shared (X0, X1, Y0, Y1) for every panel that needs
                           identical axes across z slices
    plot_raw_cells_grid  - 2 x 5 panel of all 2D cells colour-coded by slice
    plot_zoom_window     - same panel restricted to a square window so the
                           reader can see what "candidate cells across
                           layers" looks like before matching
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .color_palette import finalize_notebook_figure


def slice_summary_table(cells_by_slice: Dict[int, list]) -> pd.DataFrame:
    rows = []
    for s in sorted(cells_by_slice.keys()):
        cells = cells_by_slice[s]
        areas = np.asarray([c.area for c in cells])
        asp = np.asarray([c.aspect_ratio for c in cells])
        rows.append({
            "slice": s,
            "n_cells": len(cells),
            "area_med": float(np.median(areas)),
            "area_q25": float(np.quantile(areas, 0.25)),
            "area_q75": float(np.quantile(areas, 0.75)),
            "aspect_med": float(np.median(asp)),
        })
    return pd.DataFrame(rows)


def global_xy_bounds(
    cells_by_slice: Dict[int, list], pad_frac: float = 0.04,
) -> Tuple[float, float, float, float]:
    cx = np.concatenate([np.asarray([c.centroid[0] for c in cs]) for cs in cells_by_slice.values()])
    cy = np.concatenate([np.asarray([c.centroid[1] for c in cs]) for cs in cells_by_slice.values()])
    xmin, xmax = float(cx.min()), float(cx.max())
    ymin, ymax = float(cy.min()), float(cy.max())
    half = max(xmax - xmin, ymax - ymin) / 2.0
    cx0 = (xmin + xmax) / 2.0
    cy0 = (ymin + ymax) / 2.0
    pad = half * pad_frac
    return cx0 - half - pad, cx0 + half + pad, cy0 - half - pad, cy0 + half + pad


def plot_raw_cells_grid(
    cells_by_slice: Dict[int, list],
    out_pdf: Path,
    title: str,
    contour_step: int = 30,
    cmap_name: str = "viridis",
    figsize: Tuple[float, float] = (13.5, 5.6),
) -> None:
    slice_ids = sorted(cells_by_slice.keys())
    cmap = plt.get_cmap(cmap_name)
    slice_color = {s: cmap((i + 0.5) / len(slice_ids)) for i, s in enumerate(slice_ids)}
    X0, X1, Y0, Y1 = global_xy_bounds(cells_by_slice)

    n = len(slice_ids)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    flat = axes.ravel()
    for ax, s in zip(flat, slice_ids):
        cells = cells_by_slice[s]
        cx = np.asarray([c.centroid[0] for c in cells])
        cy = np.asarray([c.centroid[1] for c in cells])
        sizes = np.sqrt(np.maximum(np.asarray([c.area for c in cells]), 1)) * 0.20
        ax.scatter(cx, cy, s=sizes, c=[slice_color[s]], alpha=0.55, edgecolors="none")
        for c in cells[::contour_step]:
            cnt = c.contour_points
            ax.plot(cnt[:, 0] + c.centroid[0], cnt[:, 1] + c.centroid[1],
                    color="black", lw=0.25, alpha=0.55)
        ax.set_xlim(X0, X1)
        ax.set_ylim(Y1, Y0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"S{s:02d}  (n={len(cells)})", fontsize=9, fontweight="bold")
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
    for ax in flat[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("X (px)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Y (px)", fontsize=8)
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)


def plot_zoom_window(
    cells_by_slice: Dict[int, list],
    out_pdf: Path,
    title: str,
    anchor_slice: Optional[int] = None,
    half_width: float = 200.0,
    cmap_name: str = "viridis",
    figsize: Tuple[float, float] = (13.5, 5.6),
) -> Tuple[float, float]:
    slice_ids = sorted(cells_by_slice.keys())
    cmap = plt.get_cmap(cmap_name)
    slice_color = {s: cmap((i + 0.5) / len(slice_ids)) for i, s in enumerate(slice_ids)}
    if anchor_slice is None or anchor_slice not in cells_by_slice:
        anchor_slice = slice_ids[len(slice_ids) // 2]
    anchor = cells_by_slice[anchor_slice]
    cx_centre = float(np.median([c.centroid[0] for c in anchor]))
    cy_centre = float(np.median([c.centroid[1] for c in anchor]))

    n = len(slice_ids)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    flat = axes.ravel()
    for ax, s in zip(flat, slice_ids):
        cells = cells_by_slice[s]
        n_drawn = 0
        for c in cells:
            if abs(c.centroid[0] - cx_centre) > half_width or abs(c.centroid[1] - cy_centre) > half_width:
                continue
            cnt = c.contour_points
            x = cnt[:, 0] + c.centroid[0]
            y = cnt[:, 1] + c.centroid[1]
            ax.fill(x, y, color=slice_color[s], alpha=0.45, lw=0.0)
            ax.plot(x, y, color="black", lw=0.4, alpha=0.7)
            ax.scatter(c.centroid[0], c.centroid[1], s=4, c="black", zorder=3)
            n_drawn += 1
        ax.set_xlim(cx_centre - half_width, cx_centre + half_width)
        ax.set_ylim(cy_centre + half_width, cy_centre - half_width)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"S{s:02d}  (n={n_drawn})", fontsize=9, fontweight="bold")
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
    for ax in flat[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("X (px)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Y (px)", fontsize=8)
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)
    return cx_centre, cy_centre
