"""Per-track visualisations after reconstruction.

Public functions:
    track_length_histogram - distribution of track length (layers per track)
    margin_distribution    - per-track average matching margin (top1 - top2)
    sample_tracks_xy       - 2D projection of N random tracks coloured by id
    sample_tracks_3d       - 3×2 ribbon panels (simulation01 six-panel style) or legacy lines
    evaluation_radar       - quality radar generated from `evaluate_all_tracks`
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .color_palette import PALETTE, finalize_notebook_figure

# Match simulation01_visualize.py viewing geometry for track ribbons
_VIEW_ELEV_TRACKS = 20
_VIEW_AZIM_TRACKS = -60
_TRACKS_Z_BOX_STRETCH = 3.0


def track_length_histogram(
    tracks: list, out_pdf: Path, title: str, color: str = PALETTE["primary"],
) -> None:
    lengths = np.asarray([t.get_layer_count() for t in tracks])
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    bins = np.arange(lengths.min() - 0.5, lengths.max() + 1.5, 1.0)
    ax.hist(lengths, bins=bins, color=color, edgecolor="black", alpha=0.85, lw=0.7)
    ax.set_xlabel("Track length (layers)", fontsize=8)
    ax.set_ylabel("# tracks", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xticks(np.arange(lengths.min(), lengths.max() + 1))
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)


def margin_distribution(
    tracks: list, out_pdf: Path, title: str, color: str = PALETTE["primary"],
) -> None:
    margins = np.asarray([t.get_quality_score() for t in tracks])
    margins = margins[~np.isnan(margins)]
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    bp = ax.boxplot(
        [margins], positions=[1], widths=0.55, patch_artist=True, showfliers=False,
        showmeans=True,
        meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=3.5),
        medianprops=dict(color="black", lw=1.0),
        whiskerprops=dict(color="black", lw=0.8),
        capprops=dict(color="black", lw=0.8),
        boxprops=dict(edgecolor="black", lw=0.8),
    )
    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_alpha(0.65)
    rng = np.random.default_rng(0)
    ax.scatter(np.full_like(margins, 1.0) + rng.standard_normal(len(margins)) * 0.04,
               margins, s=4, color=color, alpha=0.35, edgecolors="none")
    ax.set_xticks([1])
    ax.set_xticklabels(["all tracks"], fontsize=8)
    ax.set_ylabel("Avg matching margin (top1 - top2)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)


def _sample_track_objects(
    tracks: list,
    n_show: Optional[int] = 80,
    seed: int = 2,
    min_layers: int = 3,
) -> list:
    rng = np.random.default_rng(seed)
    candidates = [t for t in tracks if t.get_layer_count() >= min_layers]
    if n_show is None:
        return list(candidates)
    if len(candidates) > n_show:
        idx = rng.choice(len(candidates), size=n_show, replace=False)
        return [candidates[int(i)] for i in idx]
    return list(candidates)


def _split_into_n_panels(items: list, n_panels: int) -> List[list]:
    if n_panels <= 0:
        return [items]
    if not items:
        return [[] for _ in range(n_panels)]
    q, r = divmod(len(items), n_panels)
    out: List[list] = []
    i = 0
    for p in range(n_panels):
        take = q + (1 if p < r else 0)
        out.append(items[i : i + take])
        i += take
    return out


def _ribbon_from_track(track) -> Optional[Tuple[List[np.ndarray], List[float]]]:
    """Absolute XY contours and raw slice indices as Z (scaled later for display)."""
    slices = sorted(track.cells.keys())
    if len(slices) < 2:
        return None
    contours: List[np.ndarray] = []
    zs: List[float] = []
    for s in slices:
        cell = track.cells[s]
        pts = cell.contour_points_200
        if pts is None:
            pts = cell.contour_points
        xy = np.asarray(pts, dtype=np.float64) + np.asarray(cell.centroid[:2], dtype=np.float64)
        contours.append(xy)
        zs.append(float(s))
    return contours, zs


def _track_color_tab20(idx: int, n_palette: int):
    cmap = plt.get_cmap("tab20")
    return cmap((idx * 7) % 20 / 19.0 if n_palette > 0 else 0.0)


def _draw_track_ribbon_3d(
    ax,
    contours_abs: Sequence[np.ndarray],
    z_values: Sequence[float],
    color,
    alpha: float = 0.55,
    linewidth: float = 0.25,
) -> None:
    """Ribbon between consecutive layer contours (same construction as simulation01_visualize)."""
    clist = list(contours_abs)
    zlist = list(z_values)
    if len(clist) < 1:
        return
    N = max(len(c) for c in clist)
    contours_norm: List[np.ndarray] = []
    for c in clist:
        c = np.asarray(c, dtype=np.float64)
        if len(c) == N:
            contours_norm.append(c)
        else:
            idx = (np.arange(N) * len(c) / N).astype(int) % max(len(c), 1)
            contours_norm.append(c[idx])
    c3d = [np.column_stack([c, np.full(N, z)]) for c, z in zip(contours_norm, zlist)]
    faces = []
    for k in range(len(c3d) - 1):
        a = c3d[k]
        b = c3d[k + 1]
        for i in range(N):
            i2 = (i + 1) % N
            faces.append([a[i], a[i2], b[i2], b[i]])
    if faces:
        pc = Poly3DCollection(
            faces, facecolors=color, edgecolors="none", linewidths=0.0, alpha=alpha
        )
        ax.add_collection3d(pc)
    if len(c3d) >= 1 and N >= 3:
        for cap, z in ((c3d[0], zlist[0]), (c3d[-1], zlist[-1])):
            center = cap.mean(axis=0)
            tri_faces = [[cap[i], cap[(i + 1) % N], center] for i in range(N)]
            if tri_faces:
                pc2 = Poly3DCollection(
                    tri_faces,
                    facecolors=color,
                    edgecolors="none",
                    linewidths=0.0,
                    alpha=alpha * 0.9,
                )
                ax.add_collection3d(pc2)
    lines = [np.vstack([c, c[:1]]) for c in c3d]
    if lines:
        lc = Line3DCollection(
            lines,
            colors=color,
            linewidths=linewidth,
            alpha=min(1.0, alpha + 0.2),
        )
        ax.add_collection3d(lc)


def _set_3d_axes_equal_tile(
    ax,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    *,
    show_z_axis: bool = False,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.view_init(elev=_VIEW_ELEV_TRACKS, azim=_VIEW_AZIM_TRACKS)
    ax.set_xlabel("X (px)", fontsize=7, labelpad=-2)
    ax.set_ylabel("Y (px)", fontsize=7, labelpad=-2)
    if show_z_axis:
        ax.set_zlabel("Z (slice)", fontsize=7, labelpad=-2)
    else:
        ax.set_zlabel("")
        ax.set_zticks([])
        ax.set_zticklabels([])
    ax.tick_params(labelsize=6, pad=0)
    try:
        dx = float(xlim[1] - xlim[0])
        dy = float(ylim[1] - ylim[0])
        dz = float(zlim[1] - zlim[0]) * max(0.01, float(_TRACKS_Z_BOX_STRETCH))
        ax.set_box_aspect((dx, dy, dz))
    except Exception:
        pass


def _sample_track_traces(
    tracks: list,
    n_show: Optional[int] = 80,
    seed: int = 1,
    min_layers: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    candidates = [t for t in tracks if t.get_layer_count() >= min_layers]
    if n_show is None:
        chosen = candidates
    elif len(candidates) > n_show:
        idx = rng.choice(len(candidates), size=n_show, replace=False)
        chosen = [candidates[int(i)] for i in idx]
    else:
        chosen = candidates
    out = []
    for t in chosen:
        slices = sorted(t.cells.keys())
        x = np.asarray([t.cells[s].centroid[0] for s in slices])
        y = np.asarray([t.cells[s].centroid[1] for s in slices])
        z = np.asarray(slices, dtype=float)
        out.append((z, x, y))
    return out


def sample_tracks_xy(
    tracks: list,
    out_pdf: Path,
    title: str,
    bounds: Tuple[float, float, float, float],
    n_show: Optional[int] = 80,
    seed: int = 1,
) -> None:
    samples = _sample_track_traces(tracks, n_show=n_show, seed=seed)
    if not samples:
        print("no tracks to plot")
        return
    X0, X1, Y0, Y1 = bounds
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    cmap = plt.get_cmap("tab20")
    for k, (_z, x, y) in enumerate(samples):
        c = cmap(k % 20)
        ax.plot(x, y, "-", color=c, lw=0.5, alpha=0.55)
        ax.scatter(x, y, s=4, color=c, alpha=0.85, edgecolors="none")
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y1, Y0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (px)", fontsize=8)
    ax.set_ylabel("Y (px)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)


def sample_tracks_3d(
    tracks: list,
    out_pdf: Path,
    title: str,
    n_show: Optional[int] = 80,
    seed: int = 2,
    n_panels: int = 6,
    layout: str = "sixpanel",
) -> None:
    """3D track visualisation.

    ``layout="sixpanel"`` (default): 2×3 axes matching ``fig_tracks_3d_sixpanel``
    style in ``simulation01_visualize.py`` — contour ribbons (Poly3DCollection),
    shared camera (elev=20°, azim=-60°), stretched Z box aspect, no z tick labels.

    ``layout="lines"``: legacy centroid polyline + scatter (single subplot).

    Args:
        n_show: Max number of tracks with ``>= min_layers`` (default 80).
                Pass ``None`` to plot **all** such tracks (can be slow for large tiles).
    """
    chosen = _sample_track_objects(tracks, n_show=n_show, seed=seed)
    if not chosen:
        print("no tracks to plot")
        return

    if layout == "lines":
        samples = _sample_track_traces(tracks, n_show=n_show, seed=seed)
        if not samples:
            print("no tracks to plot")
            return
        fig = plt.figure(figsize=(5.5, 5.0))
        ax = fig.add_subplot(111, projection="3d")
        cmap = plt.get_cmap("tab20")
        for k, (z, x, y) in enumerate(samples):
            c = cmap(k % 20)
            ax.plot(x, y, z, "-", color=c, lw=0.45, alpha=0.7)
            ax.scatter(x, y, z, s=5, color=c, alpha=0.9, edgecolors="none")
        ax.set_xlabel("X (px)", fontsize=7)
        ax.set_ylabel("Y (px)", fontsize=7)
        ax.set_zlabel("Z (slice)", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.set_title(title, fontsize=9)
        ax.view_init(elev=_VIEW_ELEV_TRACKS, azim=_VIEW_AZIM_TRACKS)
        fig.tight_layout()
        fig.savefig(out_pdf)
        finalize_notebook_figure(fig)
        return

    # --- Ribbon six-panel (simulation01_visualize fig_tracks_3d_sixpanel style) ---
    xs_all: List[float] = []
    ys_all: List[float] = []
    ribbons_flat: List[Tuple[int, object, List[np.ndarray], List[float]]] = []
    for ti, tr in enumerate(chosen):
        rib = _ribbon_from_track(tr)
        if rib is None:
            continue
        contours_u, zs_u = rib
        for c in contours_u:
            xs_all.extend([float(c[:, 0].min()), float(c[:, 0].max())])
            ys_all.extend([float(c[:, 1].min()), float(c[:, 1].max())])
        ribbons_flat.append((ti, tr, contours_u, zs_u))

    if not ribbons_flat:
        print("no multi-layer tracks with contours to plot")
        return

    xmin, xmax = min(xs_all), max(xs_all)
    ymin, ymax = min(ys_all), max(ys_all)
    span_xy = max(xmax - xmin, ymax - ymin)
    z_scale = max(span_xy * 0.025, 1e-6)

    ribbons_scaled: List[Tuple[int, List[np.ndarray], List[float]]] = []
    slices_scaled: List[float] = []
    for ti, _tr, contours_u, zs_u in ribbons_flat:
        zs_s = [zi * z_scale for zi in zs_u]
        ribbons_scaled.append((ti, contours_u, zs_s))
        slices_scaled.extend(zs_s)

    xs_s: List[float] = []
    ys_s: List[float] = []
    for _ti, contours_u, _zs_s in ribbons_scaled:
        for c in contours_u:
            xs_s.extend([float(c[:, 0].min()), float(c[:, 0].max())])
            ys_s.extend([float(c[:, 1].min()), float(c[:, 1].max())])

    xmin_s, xmax_s = min(xs_s), max(xs_s)
    ymin_s, ymax_s = min(ys_s), max(ys_s)
    zmin_s, zmax_s = min(slices_scaled), max(slices_scaled)
    pad_xy_s = max(xmax_s - xmin_s, ymax_s - ymin_s) * 0.02 + 1e-6
    pad_z_s = max(zmax_s - zmin_s, z_scale) * 0.08 + 1e-9

    xlim = (xmin_s - pad_xy_s, xmax_s + pad_xy_s)
    ylim = (ymin_s - pad_xy_s, ymax_s + pad_xy_s)
    zlim = (zmin_s - pad_z_s, zmax_s + pad_z_s)

    n_panels = max(1, int(n_panels))
    cols = 3 if n_panels >= 6 else min(3, n_panels)
    rows = int(np.ceil(n_panels / cols))
    panel_chunks = _split_into_n_panels(ribbons_scaled, n_panels)

    panel_w_in = 58.0 / 25.4
    panel_h_in = 56.0 / 25.4
    fig = plt.figure(figsize=(panel_w_in * cols, panel_h_in * rows))

    n_tot = len(chosen)
    for pi in range(n_panels):
        ax = fig.add_subplot(rows, cols, pi + 1, projection="3d")
        chunk = panel_chunks[pi] if pi < len(panel_chunks) else []
        for ti, contours_u, zs_s in chunk:
            col = _track_color_tab20(ti, max(n_tot, 1))
            _draw_track_ribbon_3d(ax, contours_u, zs_s, color=col, alpha=0.55, linewidth=0.25)

        # Match simulation panels: hide Z ticks on all subplots (compose-ready).
        _set_3d_axes_equal_tile(ax, xlim, ylim, zlim, show_z_axis=False)

        n_sub = len(chunk)
        ax.set_title(f"Panel {pi + 1}  (n={n_sub})", fontsize=8)

    fig.suptitle(title, fontsize=9, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)


def evaluation_radar(
    metrics: dict, out_pdf: Path, title: str, color: str = PALETTE["primary"],
) -> None:
    cats = ["Avg margin", "Margin consistency", "Avg length", "Coverage"]
    vals = [
        min(1.0, metrics["avg_margin"] / 0.5),
        max(0.0, 1.0 - metrics["margin_std"] / 0.2),
        min(1.0, metrics["average_length"] / 10.0),
        metrics["coverage_ratio"],
    ]
    angles = [n / float(len(cats)) * 2 * np.pi for n in range(len(cats))]
    angles += angles[:1]
    vals = vals + vals[:1]
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, lw=1.2, color=color)
    ax.fill(angles, vals, alpha=0.25, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6)
    ax.set_rlabel_position(0)
    ax.set_title(title, fontsize=9, pad=12)
    fig.tight_layout()
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)
