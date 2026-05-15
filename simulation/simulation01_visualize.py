#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation 1 - Intuitive 3D + 2D visualisation.

Produces single-panel figures that show
  (A) what the simulated 3D cardiomyocytes look like and how they are sliced,
  (B) what the reconstructed 3D tracks look like (GT vs HiDTW vs baselines),
  (C) where and how methods fail.

All figures are vector PDFs with editable text; each figure also exports a
matching *.csv with the underlying source data.

Inputs:
  simulation01_results/phantoms/<cond>/seed<k>/ (phantoms.json, ground_truth.csv, features csv)
  simulation01_results/methods/<cond>/seed<k>/<method>/tracks.csv

Outputs:
  simulation01_results/visualization/<cond>/seed<k>/fig_viz_*.pdf (+ .csv)

Usage:
  python simulation01_visualize.py
  python simulation01_visualize.py --conditions L2_mid_clean L2_high_clean
  python simulation01_visualize.py --conditions L2_high_clean --seeds 0 1
  python simulation01_visualize.py --viz-condition L2_high_clean --viz-seed 0
  python simulation01_visualize.py --viz-condition L2_mid_clean L2_high_clean --viz-seed 0
  python simulation01_visualize.py --plotly  # also export interactive HTML (requires plotly)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Matplotlib: vector PDF with editable text
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.pad_inches"] = 0.05
matplotlib.rcParams["axes.linewidth"] = 0.8
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation01_phantom_generator import (  # noqa: E402
    CONDITIONS, resolve_condition_config,
    N_SLICES, SLICE_THICKNESS_UM, BOX_X_UM, BOX_Y_UM, BOX_Z_UM,
    Phantom, sample_phantom_surface,
)
from simulation01_run_methods import METHODS as METHOD_SPECS  # noqa: E402

METHOD_ORDER = [
    "Centroid_NN",
    "IoU_only",
    "DTW_only",
    "OT_only",
    "DTW_Centroid",
    "HiDTW_full_fixed",
    "HiDTW_full",
]

OI = {
    "Centroid_NN":       "#999999",
    "IoU_only":          "#E69F00",
    "DTW_only":          "#56B4E9",
    "OT_only":           "#009E73",
    "DTW_Centroid":      "#CC79A7",
    "HiDTW_full_fixed":  "#F0A070",
    "HiDTW_full":        "#D55E00",
}

VIEW_ELEV = 20
VIEW_AZIM = -60

# Extra vertical stretch for the 3D track six-panel TIFF only (mplot3d box
# aspect). xy ~500 µm vs z ~100 µm makes layers look overly flat; bump z
# slightly so inter-slice separation reads better when the figure is shrunk.
TRACKS_SIXPANEL_Z_BOX_STRETCH = 3


# --- 1. DATA LOADERS ---

def load_phantom_and_gt(phantom_dir: str) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Dict[str, Any]]]:
    with open(os.path.join(phantom_dir, "phantoms.json"), encoding="utf-8") as f:
        phantoms = json.load(f)
    gt_df = pd.read_csv(os.path.join(phantom_dir, "ground_truth.csv"))
    gt_df["slice"] = gt_df["slice"].astype(int)
    gt_df["gt_track_id"] = gt_df["gt_track_id"].astype(int)

    feat: Dict[str, Dict[str, Any]] = {}
    for fp in sorted(glob(os.path.join(phantom_dir, "*_S*_features.csv"))):
        df = pd.read_csv(fp)
        for _, row in df.iterrows():
            cid = str(row["cell_id"])
            cx, cy = float(row["centroid_x"]), float(row["centroid_y"])
            xs, ys = [], []
            for k in range(400):
                xk = row.get(f"x_{k}", np.nan)
                yk = row.get(f"y_{k}", np.nan)
                if pd.isna(xk):
                    break
                xs.append(float(xk)); ys.append(float(yk))
            rel = np.column_stack([np.array(xs), np.array(ys)])
            feat[cid] = {
                "slice": int(str(row["slice"]).replace("S", "")) if isinstance(row["slice"], str) else int(row["slice"]),
                "centroid": np.array([cx, cy]),
                "area": float(row["area"]),
                "contour_abs": rel + np.array([cx, cy]),
            }
    return phantoms, gt_df, feat


def load_tracks(tracks_csv: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(tracks_csv):
        return []
    df = pd.read_csv(tracks_csv)
    out = []
    for _, row in df.iterrows():
        cids = [c.strip() for c in str(row["cell_ids"]).split("|") if c.strip()]
        out.append({"track_id": int(row["track_id"]), "cell_ids": cids})
    return out


def tracks_from_gt(gt_df: pd.DataFrame) -> List[Dict[str, Any]]:
    tracks = []
    for gid, sub in gt_df.groupby("gt_track_id"):
        sub = sub.sort_values("slice")
        tracks.append({
            "track_id": int(gid),
            "cell_ids": list(sub["cell_id"].astype(str)),
        })
    return tracks


def _phantom_from_dict(d: Dict[str, Any]) -> Phantom:
    ph = Phantom(
        gt_id=int(d["gt_id"]),
        level=int(d["level"]),
        length_um=float(d["length_um"]),
        b0=float(d["b0_um"]),
        c0=float(d["c0_um"]),
        taper=float(d["taper"]),
        twist_total_rad=float(np.deg2rad(d["twist_total_deg"])),
        P0=np.array(d["P0"]),
        T=np.array(d["T"]),
        curv_dir=np.array(d["curv_dir"]),
        curv_amp=float(d["curv_amp_um"]),
        curv_category=d.get("curv_category", "none"),
        volume_um3=float(d["volume_um3"]),
    )
    return ph


# --- 2. LOW-LEVEL 3D DRAWING PRIMITIVES ---

def draw_phantom_3d(ax, ph: Phantom, color: str, alpha: float = 0.35,
                    edgecolor: str = "none", n_t: int = 60, n_u: int = 28):
    """Render a phantom as a tube of quad faces."""
    # Sample surface as a regular grid (n_t x n_u, 3)
    surf = sample_phantom_surface(ph, n_t=n_t, n_u=n_u).reshape(n_t, n_u, 3)
    faces = []
    for i in range(n_t - 1):
        for j in range(n_u):
            j2 = (j + 1) % n_u
            faces.append([surf[i, j], surf[i, j2], surf[i + 1, j2], surf[i + 1, j]])
    pc = Poly3DCollection(faces, facecolors=color, edgecolors=edgecolor,
                          linewidths=0.1, alpha=alpha)
    ax.add_collection3d(pc)
    # End caps (triangle fans)
    for i in (0, n_t - 1):
        cap_pts = surf[i]
        center = cap_pts.mean(axis=0)
        tri_faces = []
        for j in range(n_u):
            j2 = (j + 1) % n_u
            tri_faces.append([cap_pts[j], cap_pts[j2], center])
        pc2 = Poly3DCollection(tri_faces, facecolors=color, edgecolors=edgecolor,
                               linewidths=0.1, alpha=alpha * 0.9)
        ax.add_collection3d(pc2)


def draw_slice_plane(ax, z: float, box_xy: Tuple[float, float],
                     alpha: float = 0.08, color: str = "0.4",
                     edgecolor: str = "0.3"):
    """Translucent horizontal slicing plane."""
    xb, yb = box_xy
    plane = np.array([[0, 0, z], [xb, 0, z], [xb, yb, z], [0, yb, z]])
    pc = Poly3DCollection([plane], facecolors=color, edgecolors=edgecolor,
                          linewidths=0.3, alpha=alpha)
    ax.add_collection3d(pc)


def draw_track_ribbon_3d(ax, contours_abs: List[np.ndarray], z_values: List[float],
                         color: str, alpha: float = 0.7, linewidth: float = 0.3):
    """Draw a 3D ribbon connecting consecutive layers' contours.

    contours_abs: list of (N,2) ABSOLUTE 2D contour arrays, one per layer.
    z_values:     matching z heights (um).
    """
    if len(contours_abs) < 1:
        return
    # Resample all contours to the same point count (already 400 for ours)
    N = max(len(c) for c in contours_abs)
    contours_norm = []
    for c in contours_abs:
        if len(c) == N:
            contours_norm.append(c)
        else:
            idx = (np.arange(N) * len(c) / N).astype(int) % len(c)
            contours_norm.append(c[idx])
    c3d = [np.column_stack([c, np.full(N, z)]) for c, z in zip(contours_norm, z_values)]
    # Side faces (between adjacent layers)
    faces = []
    for k in range(len(c3d) - 1):
        a = c3d[k]; b = c3d[k + 1]
        for i in range(N):
            i2 = (i + 1) % N
            faces.append([a[i], a[i2], b[i2], b[i]])
    if faces:
        pc = Poly3DCollection(faces, facecolors=color, edgecolors="none",
                              linewidths=0.0, alpha=alpha)
        ax.add_collection3d(pc)
    # End caps - only add if we have points to form triangles.
    if len(c3d) >= 1 and N >= 3:
        for cap, z in ((c3d[0], z_values[0]), (c3d[-1], z_values[-1])):
            center = cap.mean(axis=0)
            tri_faces = [[cap[i], cap[(i + 1) % N], center] for i in range(N)]
            if tri_faces:
                pc = Poly3DCollection(tri_faces, facecolors=color, edgecolors="none",
                                      linewidths=0.0, alpha=alpha * 0.9)
                ax.add_collection3d(pc)
    # Edges of each contour for crispness
    lines = [np.vstack([c, c[:1]]) for c in c3d]
    if lines:
        lc = Line3DCollection(lines, colors=color, linewidths=linewidth,
                              alpha=min(1.0, alpha + 0.2))
        ax.add_collection3d(lc)


def set_3d_axes_equal(ax, xlim: Tuple[float, float], ylim: Tuple[float, float],
                      zlim: Tuple[float, float], z_box_stretch: float = 1.0,
                      show_z_axis: bool = True):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlabel("X (um)", fontsize=7, labelpad=-2)
    ax.set_ylabel("Y (um)", fontsize=7, labelpad=-2)
    if show_z_axis:
        ax.set_zlabel("Z (um)", fontsize=7, labelpad=-2)
    else:
        ax.set_zlabel("")
    ax.tick_params(labelsize=6, pad=0)
    if not show_z_axis:
        # No z-axis title and no z tick marks / numbers (panel layout for compositing).
        ax.set_zticks([])
        ax.set_zticklabels([])
    try:
        dx = float(xlim[1] - xlim[0])
        dy = float(ylim[1] - ylim[0])
        dz = float(zlim[1] - zlim[0]) * max(0.01, float(z_box_stretch))
        ax.set_box_aspect((dx, dy, dz))
    except Exception:
        pass


def gt_color(idx: int, total: int) -> str:
    cmap = plt.get_cmap("tab20")
    return cmap((idx * 7) % 20 / 19.0 if total > 0 else 0.0)


# --- 3. FIGURE FUNCTIONS ---

def _savefig(path: str, dpi: Optional[int] = None):
    """Save current figure, swallowing matplotlib 3D-collection rendering
    errors that occur when a sub-axis contains only empty artists.

    If ``path`` ends in ``.tif`` / ``.tiff`` the figure is rasterised at the
    given ``dpi`` (default 600 when dpi is None for TIFF) with LZW lossless
    compression. Text in TIFF is rasterised - prefer PDF when editable text
    is required. For PDF output the file stays a vector and ``dpi`` only
    affects any embedded raster artists.
    """
    kwargs: Dict[str, Any] = {"bbox_inches": "tight", "transparent": False}
    is_tiff = path.lower().endswith((".tif", ".tiff"))
    if is_tiff:
        kwargs["dpi"] = 600 if dpi is None else int(dpi)
        kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
    elif dpi is not None:
        kwargs["dpi"] = int(dpi)
    try:
        plt.savefig(path, **kwargs)
        print(f"      -> {path}")
    except ValueError as exc:
        # Known: Poly3DCollection with no segments raises during do_3d_projection.
        # Skip the figure rather than abort the whole visualisation run.
        print(f"      [skip] {path}: {exc}")
    finally:
        plt.close()


def fig_phantom_3d_overview(phantoms: List[Phantom],
                            gt_df: pd.DataFrame,
                            feat: Dict[str, Dict[str, Any]],
                            out_pdf: str,
                            max_cells: int = 60):
    """3D overview of the first ``max_cells`` GT cells, drawn from the
    **actual per-slice contours** stored in ``*_S??_features.csv`` (the same
    400-point contours fed to the matching pipeline), stacked layer-by-layer
    into 3D ribbons.

    This replaces the former parametric surface rendering so the figure
    faithfully reflects the data the algorithms actually see (post slicing,
    jitter, canonicalisation, etc.).
    """
    fig = plt.figure(figsize=(150 / 25.4, 110 / 25.4))
    ax = fig.add_subplot(111, projection="3d")

    gt_tracks_all = tracks_from_gt(gt_df)
    gt_tracks_all.sort(key=lambda t: int(t["track_id"]))
    total = len(gt_tracks_all)
    sel_tracks = gt_tracks_all[:max_cells]

    gt_ids_global = [int(t["track_id"]) for t in gt_tracks_all]
    gid_to_idx = {gid: i for i, gid in enumerate(gt_ids_global)}

    drawn_ribbon = 0
    drawn_flat = 0
    for t in sel_tracks:
        gid = int(t["track_id"])
        col = gt_color(gid_to_idx.get(gid, 0), max(total, 1))
        ribbons = tracks_to_ribbon_data([t], feat)
        if ribbons:
            for contours, zs in ribbons:
                draw_track_ribbon_3d(ax, contours, zs, color=col,
                                     alpha=0.55, linewidth=0.25)
                drawn_ribbon += 1
        else:
            # Fall back to drawing the single available slice as a flat
            # polygon so the cell is still shown.
            for cid in t["cell_ids"]:
                if cid not in feat:
                    continue
                contour = feat[cid]["contour_abs"]
                z = feat[cid]["slice"] * SLICE_THICKNESS_UM
                verts3d = np.column_stack([contour, np.full(len(contour), z)])
                pc = Poly3DCollection([verts3d], facecolors=col,
                                      edgecolors=col, linewidths=0.3,
                                      alpha=0.6)
                ax.add_collection3d(pc)
                drawn_flat += 1

    for k in range(N_SLICES):
        z = (k + 0.5) * SLICE_THICKNESS_UM
        draw_slice_plane(ax, z, (BOX_X_UM, BOX_Y_UM), alpha=0.06, color="0.6")
    set_3d_axes_equal(ax, (0, BOX_X_UM), (0, BOX_Y_UM), (0, BOX_Z_UM))
    ax.set_title(f"3D phantom overview (first {len(sel_tracks)} / {total} cells, "
                 f"per-slice contours)", fontsize=9)
    _savefig(out_pdf)

    ph_by_gid = {int(p.gt_id): p for p in phantoms}
    rows = []
    for t in sel_tracks:
        gid = int(t["track_id"])
        p = ph_by_gid.get(gid)
        n_slices_drawn = sum(1 for cid in t["cell_ids"] if cid in feat)
        if p is None:
            rows.append({"gt_id": gid, "level": None, "length_um": None,
                         "volume_um3": None, "curv_category": None,
                         "n_slices": n_slices_drawn})
            continue
        rows.append({"gt_id": p.gt_id, "level": p.level, "length_um": p.length_um,
                     "volume_um3": p.volume_um3, "curv_category": p.curv_category,
                     "n_slices": n_slices_drawn})
    pd.DataFrame(rows).to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def fig_levels_gallery(phantoms: List[Phantom], out_pdf: str):
    """Gallery of representative phantom cells."""
    # pick up to 3 phantoms per curvature category (if available)
    by_cat: Dict[str, List[Phantom]] = defaultdict(list)
    for p in phantoms:
        by_cat[p.curv_category].append(p)
    categories = [c for c in ["none", "low", "high"] if c in by_cat]
    if not categories:
        categories = list(by_cat.keys())[:3]
    n_cols = max(len(categories), 1)
    fig = plt.figure(figsize=(n_cols * 55 / 25.4, 70 / 25.4))
    for ci, cat in enumerate(categories):
        ax = fig.add_subplot(1, n_cols, ci + 1, projection="3d")
        # pick one representative (first one with largest length)
        ph = max(by_cat[cat], key=lambda p: p.length_um)
        draw_phantom_3d(ax, ph, color="#0072B2", alpha=0.45)
        # draw axis curve
        ts = np.linspace(0, 1, 50)
        gs = ph.gamma(ts)
        ax.plot(gs[:, 0], gs[:, 1], gs[:, 2], color="#D55E00", lw=0.8)
        # tight box around cell
        pad = max(ph.b0, ph.c0) * 2 + 5
        gamm_min = gs.min(axis=0) - pad; gamm_max = gs.max(axis=0) + pad
        set_3d_axes_equal(ax, (gamm_min[0], gamm_max[0]),
                          (gamm_min[1], gamm_max[1]),
                          (gamm_min[2], gamm_max[2]))
        ax.set_title(f"{cat} curvature\nL={ph.length_um:.0f} um", fontsize=8)
    plt.tight_layout()
    _savefig(out_pdf)


def fig_phantom_3d_tilt_effect(phantoms: List[Phantom], out_pdf: str,
                               tilt_deg: float = 10.0):
    """Single phantom with tilted vs horizontal slicing plane overlay."""
    if not phantoms:
        return
    ph = max(phantoms, key=lambda p: p.length_um)
    fig = plt.figure(figsize=(110 / 25.4, 75 / 25.4))
    ax = fig.add_subplot(111, projection="3d")
    draw_phantom_3d(ax, ph, color="#0072B2", alpha=0.35)

    tilt_rad = np.deg2rad(tilt_deg)
    for k in range(N_SLICES):
        z = (k + 0.5) * SLICE_THICKNESS_UM
        # horizontal plane
        draw_slice_plane(ax, z, (BOX_X_UM, BOX_Y_UM), alpha=0.05, color="0.75")
        # tilted plane (rotated about y axis by tilt_rad around (BOX_X_UM/2, -, z))
        # Plane z' = z + tan(tilt) * (x - BOX_X_UM/2)
        xs = np.linspace(0, BOX_X_UM, 2)
        ys = np.linspace(0, BOX_Y_UM, 2)
        X, Y = np.meshgrid(xs, ys)
        Z = z + np.tan(tilt_rad) * (X - BOX_X_UM / 2)
        quad = np.array([[X[0, 0], Y[0, 0], Z[0, 0]],
                         [X[0, 1], Y[0, 1], Z[0, 1]],
                         [X[1, 1], Y[1, 1], Z[1, 1]],
                         [X[1, 0], Y[1, 0], Z[1, 0]]])
        pc = Poly3DCollection([quad], facecolors="#E69F00", edgecolors="0.4",
                              linewidths=0.35, alpha=0.08)
        ax.add_collection3d(pc)
    pad = max(ph.b0, ph.c0) * 2 + 10
    gs = ph.gamma(np.linspace(0, 1, 30))
    lo = gs.min(axis=0) - pad; hi = gs.max(axis=0) + pad
    set_3d_axes_equal(ax, (lo[0], hi[0]), (lo[1], hi[1]), (0, BOX_Z_UM))
    ax.set_title(f"Slicing geometry: horizontal (grey) vs {tilt_deg:.0f} deg tilt (orange)", fontsize=9)
    _savefig(out_pdf)


def fig_slice_montage(gt_df: pd.DataFrame, feat: Dict[str, Dict[str, Any]],
                      out_pdf: str, max_rows: int = 1):
    """Horizontal montage: one 2D subplot per z-slice, contours filled by GT track id."""
    slices = sorted(gt_df["slice"].unique())
    n = len(slices)
    fig, axes = plt.subplots(1, n, figsize=(n * 26 / 25.4, 32 / 25.4),
                             sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    # Build color map per gt_track_id
    gt_ids = sorted(gt_df["gt_track_id"].unique())
    gid_to_color = {gid: gt_color(i, len(gt_ids)) for i, gid in enumerate(gt_ids)}

    for k, s in enumerate(slices):
        ax = axes[k]
        sub = gt_df[gt_df["slice"] == s]
        for _, row in sub.iterrows():
            cid = str(row["cell_id"])
            if cid not in feat:
                continue
            contour = feat[cid]["contour_abs"]
            ax.fill(contour[:, 0], contour[:, 1],
                    facecolor=gid_to_color[int(row["gt_track_id"])],
                    edgecolor="0.15", linewidth=0.25, alpha=0.75)
            c = feat[cid]["centroid"]
            ax.plot(c[0], c[1], ".", color="0.1", markersize=0.6)
        ax.set_title(f"S{s:02d}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(0, BOX_X_UM); ax.set_ylim(0, BOX_Y_UM)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    fig.suptitle("Input slice montage (colour = ground-truth track id)", fontsize=9, y=1.02)
    plt.tight_layout()
    _savefig(out_pdf)


def fig_slice_montage_3d_stack(gt_df: pd.DataFrame, feat: Dict[str, Dict[str, Any]],
                               out_pdf: str):
    """Stack 10 layers' contours in 3D at z=k*slice_thickness (no inter-layer lines)."""
    fig = plt.figure(figsize=(120 / 25.4, 95 / 25.4))
    ax = fig.add_subplot(111, projection="3d")
    gt_ids = sorted(gt_df["gt_track_id"].unique())
    gid_to_color = {gid: gt_color(i, len(gt_ids)) for i, gid in enumerate(gt_ids)}

    for _, row in gt_df.iterrows():
        cid = str(row["cell_id"])
        if cid not in feat:
            continue
        contour = feat[cid]["contour_abs"]
        z = feat[cid]["slice"] * SLICE_THICKNESS_UM
        verts3d = np.column_stack([contour, np.full(len(contour), z)])
        pc = Poly3DCollection([verts3d], facecolors=gid_to_color[int(row["gt_track_id"])],
                              edgecolors="0.2", linewidths=0.15, alpha=0.6)
        ax.add_collection3d(pc)
    set_3d_axes_equal(ax, (0, BOX_X_UM), (0, BOX_Y_UM), (0, BOX_Z_UM))
    ax.set_title("Input slices stacked in 3D (no inter-layer matching shown)", fontsize=9)
    _savefig(out_pdf)


def fig_neighbor_graph(gt_df: pd.DataFrame, feat: Dict[str, Dict[str, Any]],
                       phantom_dir: str, out_pdf: str, slice_num: int = 5,
                       neighbor_threshold_um: float = 45.0):
    """Show one slice with adjacency edges between neighbors (OT cluster source)."""
    fig, ax = plt.subplots(figsize=(105 / 25.4, 95 / 25.4))
    sub = gt_df[gt_df["slice"] == slice_num]
    cents = []
    for _, row in sub.iterrows():
        cid = str(row["cell_id"])
        if cid not in feat:
            continue
        contour = feat[cid]["contour_abs"]
        ax.fill(contour[:, 0], contour[:, 1], facecolor="#DDDDDD",
                edgecolor="0.3", linewidth=0.35, alpha=0.7)
        c = feat[cid]["centroid"]
        cents.append((cid, c))
    # edges for neighbors within threshold
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            d = float(np.linalg.norm(cents[i][1] - cents[j][1]))
            if d < neighbor_threshold_um:
                p1, p2 = cents[i][1], cents[j][1]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        "-", color="#0072B2", lw=0.35, alpha=0.6)
    for cid, c in cents:
        ax.plot(c[0], c[1], "o", color="#D55E00", markersize=1.2, mec="0.1", mew=0.15)
    ax.set_aspect("equal")
    ax.set_xlim(0, BOX_X_UM); ax.set_ylim(0, BOX_Y_UM)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Slice S{slice_num:02d} neighbor graph (within {neighbor_threshold_um:.0f} um)",
                 fontsize=9)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    _savefig(out_pdf)


def fig_condition_examples(all_seeds_info: List[Dict[str, Any]], out_pdf: str):
    """For each condition, show one representative mid-slice (2x3 grid)."""
    if not all_seeds_info:
        return
    by_cond: Dict[str, Dict[str, Any]] = {}
    for info in all_seeds_info:
        by_cond.setdefault(info["condition"], info)
    conds = list(by_cond.keys())
    n = len(conds)
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 45 / 25.4, rows * 42 / 25.4),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()
    mid_slice = N_SLICES // 2
    for idx, c in enumerate(conds):
        ax = axes[idx]
        info = by_cond[c]
        feat = info["feat"]; gt_df = info["gt_df"]
        sub = gt_df[gt_df["slice"] == mid_slice]
        gt_ids = sorted(gt_df["gt_track_id"].unique())
        gid_to_color = {gid: gt_color(i, len(gt_ids)) for i, gid in enumerate(gt_ids)}
        for _, row in sub.iterrows():
            cid = str(row["cell_id"])
            if cid not in feat:
                continue
            contour = feat[cid]["contour_abs"]
            ax.fill(contour[:, 0], contour[:, 1],
                    facecolor=gid_to_color[int(row["gt_track_id"])],
                    edgecolor="0.15", linewidth=0.2, alpha=0.75)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(0, BOX_X_UM); ax.set_ylim(0, BOX_Y_UM)
        ax.set_title(c, fontsize=7)
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    for idx in range(len(conds), len(axes)):
        axes[idx].axis("off")
    fig.suptitle(f"Condition gallery (mid slice S{mid_slice:02d})", fontsize=9, y=1.02)
    plt.tight_layout()
    _savefig(out_pdf)


# -------- reconstruction visualisation ----------------------------------

def tracks_to_ribbon_data(tracks: List[Dict[str, Any]],
                          feat: Dict[str, Dict[str, Any]]
                          ) -> List[Tuple[List[np.ndarray], List[float]]]:
    out = []
    for t in tracks:
        ordered = []
        for cid in t["cell_ids"]:
            if cid in feat:
                ordered.append((feat[cid]["slice"], feat[cid]["contour_abs"]))
        ordered.sort(key=lambda r: r[0])
        if len(ordered) < 2:
            continue
        contours = [c for _, c in ordered]
        zs = [s * SLICE_THICKNESS_UM for s, _ in ordered]
        out.append((contours, zs))
    return out


def fig_tracks_3d(tracks: List[Dict[str, Any]], feat: Dict[str, Dict[str, Any]],
                  title: str, out_pdf: str, color_by: str = "track_index"):
    fig = plt.figure(figsize=(125 / 25.4, 100 / 25.4))
    ax = fig.add_subplot(111, projection="3d")
    ribbons = tracks_to_ribbon_data(tracks, feat)
    for i, (contours, zs) in enumerate(ribbons):
        col = gt_color(i, max(len(ribbons), 1))
        draw_track_ribbon_3d(ax, contours, zs, color=col, alpha=0.55, linewidth=0.3)
    set_3d_axes_equal(ax, (0, BOX_X_UM), (0, BOX_Y_UM), (0, BOX_Z_UM))
    ax.set_title(f"{title}  ({len(ribbons)} tracks)", fontsize=9)
    _savefig(out_pdf)


def fig_tracks_3d_sixpanel(methods_tracks: Dict[str, List[Dict[str, Any]]],
                           gt_tracks: List[Dict[str, Any]],
                           feat: Dict[str, Dict[str, Any]],
                           method_f1: Dict[str, float],
                           out_pdf: str):
    """One 600 dpi TIFF per panel (GT + each method in ``METHOD_ORDER``).

    ``out_pdf`` is used only for its directory and basename stem, e.g.
    ``.../fig_viz_tracks_3d_sixpanel.tiff`` -> writes
    ``fig_viz_tracks_3d_sixpanel_ground_truth.tiff``,
    ``fig_viz_tracks_3d_sixpanel_Centroid_NN.tiff``, … next to it.
    No combined multi-panel figure is written. Z-axis label and z tick
    numbers are omitted for easier external panel assembly.
    """
    methods = [m for m in METHOD_ORDER if m in methods_tracks]
    if not methods:
        return
    panels = ["Ground truth"] + methods

    # Shared GT colouring so the same cell shows the same colour across all panels.
    gt_ids_global = sorted([int(t["track_id"]) for t in gt_tracks])
    gid_to_idx = {gid: i for i, gid in enumerate(gt_ids_global)}

    out_dir_path = os.path.dirname(out_pdf)
    stem = os.path.splitext(os.path.basename(out_pdf))[0]

    def _panel_slug(panel: str) -> str:
        if panel == "Ground truth":
            return "ground_truth"
        return panel.replace(" ", "_")

    # Single-panel figure size (mm -> inches), similar footprint to one cell of
    # the old multi-panel grid.
    fig_w_in = 58.0 / 25.4
    fig_h_in = 56.0 / 25.4

    for panel in panels:
        fig = plt.figure(figsize=(fig_w_in, fig_h_in))
        ax = fig.add_subplot(111, projection="3d")
        if panel == "Ground truth":
            for t in gt_tracks:
                ribbons = tracks_to_ribbon_data([t], feat)
                gid = int(t["track_id"])
                col = gt_color(gid_to_idx.get(gid, 0), max(len(gt_ids_global), 1))
                for contours, zs in ribbons:
                    draw_track_ribbon_3d(ax, contours, zs, color=col,
                                         alpha=0.55, linewidth=0.25)
            ax.set_title("Ground truth", fontsize=8, fontweight="bold")
        else:
            m = panel
            ribbons = tracks_to_ribbon_data(methods_tracks[m], feat)
            gt_id_map = _best_gt_id_per_pred(methods_tracks[m], gt_tracks)
            for i, ((contours, zs), t) in enumerate(zip(ribbons, methods_tracks[m])):
                gid = gt_id_map.get(t["track_id"], -1)
                col = gt_color(gid_to_idx.get(gid, i), max(len(gt_ids_global), 1))
                draw_track_ribbon_3d(ax, contours, zs, color=col,
                                     alpha=0.55, linewidth=0.25)
            f1 = method_f1.get(m, np.nan)
            title_weight = "bold" if m == "HiDTW_full" else "normal"
            ax.set_title(f"{m}  F1={f1:.3f}", fontsize=8, fontweight=title_weight)
        set_3d_axes_equal(ax, (0, BOX_X_UM), (0, BOX_Y_UM), (0, BOX_Z_UM),
                          z_box_stretch=TRACKS_SIXPANEL_Z_BOX_STRETCH,
                          show_z_axis=False)
        out_path = os.path.join(out_dir_path, f"{stem}_{_panel_slug(panel)}.tiff")
        _savefig(out_path, dpi=600)


def _best_gt_id_per_pred(pred_tracks: List[Dict[str, Any]],
                         gt_tracks: List[Dict[str, Any]]) -> Dict[int, int]:
    """Map pred track_id -> best-overlap GT track_id (-1 if no overlap)."""
    gt_groups = {int(t["track_id"]): set(t["cell_ids"]) for t in gt_tracks}
    out: Dict[int, int] = {}
    for t in pred_tracks:
        pred_set = set(t["cell_ids"])
        best_gid = -1; best_ovl = 0
        for gid, gset in gt_groups.items():
            ovl = len(pred_set & gset)
            if ovl > best_ovl:
                best_ovl = ovl; best_gid = gid
        out[int(t["track_id"])] = best_gid
    return out


def fig_tracks_topview(methods_tracks: Dict[str, List[Dict[str, Any]]],
                       gt_tracks: List[Dict[str, Any]],
                       feat: Dict[str, Dict[str, Any]],
                       out_pdf: str):
    """Top-view (xy) of centroid trajectories per method (grid auto-sizes)."""
    methods = [m for m in METHOD_ORDER if m in methods_tracks]
    if not methods:
        return
    n = len(methods)
    cols = 4 if n >= 5 else 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 48 / 25.4, rows * 45 / 25.4),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()
    for k, m in enumerate(methods):
        ax = axes[k]
        # Draw GT centroids faintly for reference
        for t in gt_tracks:
            pts = []
            for cid in t["cell_ids"]:
                if cid in feat:
                    pts.append((feat[cid]["slice"], feat[cid]["centroid"]))
            pts.sort()
            xy = np.array([p[1] for p in pts])
            if len(xy) > 1:
                ax.plot(xy[:, 0], xy[:, 1], "-", color="0.85", lw=0.4)
        # Draw predicted tracks
        gt_id_map = _best_gt_id_per_pred(methods_tracks[m], gt_tracks)
        gt_ids = sorted({gid for gid in gt_id_map.values() if gid >= 0})
        gid_to_idx = {gid: i for i, gid in enumerate(gt_ids)}
        for i, t in enumerate(methods_tracks[m]):
            pts = []
            for cid in t["cell_ids"]:
                if cid in feat:
                    pts.append((feat[cid]["slice"], feat[cid]["centroid"]))
            pts.sort()
            xy = np.array([p[1] for p in pts])
            if len(xy) < 2:
                continue
            gid = gt_id_map.get(t["track_id"], -1)
            col = gt_color(gid_to_idx.get(gid, i), max(len(gt_ids), 1))
            ax.plot(xy[:, 0], xy[:, 1], "-", color=col, lw=0.6, alpha=0.85)
            ax.plot(xy[0, 0], xy[0, 1], "o", color=col, markersize=1.2, mec="0.1", mew=0.2)
        ax.set_aspect("equal")
        title_weight = "bold" if m == "HiDTW_full" else "normal"
        ax.set_title(m, fontsize=7, fontweight=title_weight)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    # Hide unused axes
    for k in range(len(methods), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Top-view (xy) of predicted centroid trajectories", fontsize=9, y=1.02)
    plt.tight_layout()
    _savefig(out_pdf)


def fig_id_switch_3d(tracks: List[Dict[str, Any]],
                     gt_df: pd.DataFrame,
                     feat: Dict[str, Dict[str, Any]],
                     out_pdf: str, title: str = "ID switches in HiDTW tracks"):
    """3D tracks with red arrows at ID-switch points."""
    fig = plt.figure(figsize=(120 / 25.4, 100 / 25.4))
    ax = fig.add_subplot(111, projection="3d")
    cid_to_gt = dict(zip(gt_df["cell_id"].astype(str), gt_df["gt_track_id"].astype(int)))
    n_switches = 0
    for i, t in enumerate(tracks):
        ordered = sorted([(feat[cid]["slice"], cid, feat[cid]["centroid"])
                          for cid in t["cell_ids"] if cid in feat])
        if len(ordered) < 2:
            continue
        # draw polyline
        xyz = np.array([[c[0], c[1], s * SLICE_THICKNESS_UM]
                        for s, _, c in ordered])
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                "-", color="0.6", lw=0.35, alpha=0.7)
        # mark switches
        for k in range(len(ordered) - 1):
            _, c1, _ = ordered[k]; _, c2, _ = ordered[k + 1]
            g1 = cid_to_gt.get(c1, -1); g2 = cid_to_gt.get(c2, -1)
            if g1 >= 0 and g2 >= 0 and g1 != g2:
                n_switches += 1
                a = xyz[k]; b = xyz[k + 1]
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                        "-", color="#D55E00", lw=1.3, alpha=0.95)
                ax.scatter([b[0]], [b[1]], [b[2]], color="#D55E00",
                           s=12, edgecolors="0.15", linewidths=0.3)
    set_3d_axes_equal(ax, (0, BOX_X_UM), (0, BOX_Y_UM), (0, BOX_Z_UM))
    ax.set_title(f"{title}  ({n_switches} switches)", fontsize=9)
    _savefig(out_pdf)


def fig_error_map_topview(pred_tracks: List[Dict[str, Any]],
                          gt_df: pd.DataFrame,
                          feat: Dict[str, Dict[str, Any]],
                          out_pdf: str,
                          method: str = "HiDTW_full"):
    """Top-view error map: green = correct pair, red = wrong substitution, grey = no pred."""
    fig, ax = plt.subplots(figsize=(105 / 25.4, 95 / 25.4))
    # Build GT pair set
    cid_to_gt = dict(zip(gt_df["cell_id"].astype(str), gt_df["gt_track_id"].astype(int)))

    # predicted pairs per source cid
    pred_next: Dict[str, str] = {}
    for t in pred_tracks:
        ordered = sorted([(feat[cid]["slice"], cid) for cid in t["cell_ids"] if cid in feat])
        for k in range(len(ordered) - 1):
            s1, c1 = ordered[k]; s2, c2 = ordered[k + 1]
            if s2 - s1 == 1:
                pred_next[c1] = c2

    # For each gt pair (a -> b), check prediction
    gt_pairs = []
    for gid, sub in gt_df.groupby("gt_track_id"):
        sub = sub.sort_values("slice")
        rows = sub.to_dict("records")
        for k in range(len(rows) - 1):
            if int(rows[k + 1]["slice"]) - int(rows[k]["slice"]) == 1:
                gt_pairs.append((str(rows[k]["cell_id"]), str(rows[k + 1]["cell_id"])))

    rows_csv = []
    for a, b in gt_pairs:
        if a not in feat:
            continue
        pa = feat[a]["centroid"]
        pred_b = pred_next.get(a, None)
        if pred_b is None:
            color = "0.7"; kind = "no_pred"
        elif pred_b == b:
            color = "#009E73"; kind = "correct"
        else:
            color = "#D55E00"; kind = "wrong"
        ax.plot(pa[0], pa[1], ".", color=color, markersize=1.6,
                mec="0.2", mew=0.15, alpha=0.9)
        rows_csv.append({"start_cell": a, "end_cell_gt": b, "pred_next": pred_b,
                         "kind": kind, "x": float(pa[0]), "y": float(pa[1])})
    ax.set_aspect("equal"); ax.set_xlim(0, BOX_X_UM); ax.set_ylim(0, BOX_Y_UM)
    ax.set_xticks([]); ax.set_yticks([])
    # manual legend
    ax.plot([], [], ".", color="#009E73", markersize=3.5, label="correct")
    ax.plot([], [], ".", color="#D55E00", markersize=3.5, label="wrong")
    ax.plot([], [], ".", color="0.7", markersize=3.5, label="no pred")
    ax.legend(fontsize=6, loc="lower right", frameon=True, edgecolor="0.7")
    ax.set_title(f"Spatial error map (method = {method})", fontsize=9)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    _savefig(out_pdf)
    pd.DataFrame(rows_csv).to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def _pick_case_by_length(gt_tracks: List[Dict[str, Any]],
                         pred_tracks: List[Dict[str, Any]],
                         feat: Dict[str, Dict[str, Any]],
                         success: bool) -> Tuple[Optional[Dict[str, Any]],
                                                 Optional[Dict[str, Any]]]:
    """Select a GT track + its best-matching pred track. If success=True, pick the
    one with largest overlap (>=8); else pick the one with *smallest* overlap (and
    >=3 layers)."""
    gt_id_map = _best_gt_id_per_pred(pred_tracks, gt_tracks)
    pred_by_gt: Dict[int, Dict[str, Any]] = {}
    for t in pred_tracks:
        gid = gt_id_map.get(t["track_id"], -1)
        if gid >= 0 and (gid not in pred_by_gt or len(t["cell_ids"]) > len(pred_by_gt[gid]["cell_ids"])):
            pred_by_gt[gid] = t
    scored = []
    for gt in gt_tracks:
        if len(gt["cell_ids"]) < 5:
            continue
        gt_set = set(gt["cell_ids"])
        pred = pred_by_gt.get(int(gt["track_id"]))
        overlap = len(set(pred["cell_ids"]) & gt_set) if pred else 0
        scored.append((overlap, gt, pred))
    if not scored:
        return None, None
    scored.sort(key=lambda x: x[0], reverse=success)
    return scored[0][1], scored[0][2]


def fig_case_3d(gt_track: Dict[str, Any], pred_track: Optional[Dict[str, Any]],
                feat: Dict[str, Dict[str, Any]],
                out_pdf: str, title: str):
    """3-subplot case study: GT / Pred / Diff (same figure)."""
    if gt_track is None:
        return
    gt_contours, gt_zs = _contours_from_track(gt_track, feat)
    pred_contours, pred_zs = (_contours_from_track(pred_track, feat)
                              if pred_track else ([], []))
    gt_set = set(gt_track["cell_ids"])

    if not gt_contours and not pred_contours:
        print(f"      [skip] {out_pdf}: empty track")
        return

    all_contours_for_zoom = gt_contours + pred_contours
    all_zs_for_zoom = gt_zs + pred_zs
    zoom_z = (min(all_zs_for_zoom), max(all_zs_for_zoom)) if all_zs_for_zoom else (0, BOX_Z_UM)

    fig = plt.figure(figsize=(175 / 25.4, 80 / 25.4))
    for si, (title_i, contours, zs, color) in enumerate([
        ("Ground truth", gt_contours, gt_zs, "#0072B2"),
        ("Predicted", pred_contours, pred_zs, "#D55E00"),
    ]):
        ax = fig.add_subplot(1, 3, si + 1, projection="3d")
        if contours:
            draw_track_ribbon_3d(ax, contours, zs, color=color, alpha=0.55, linewidth=0.3)
        _zoom_box(ax, all_contours_for_zoom, zoom_z)
        ax.set_title(title_i, fontsize=8)
    # Diff view
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    diff_polys: List[np.ndarray] = []
    diff_colors: List[str] = []
    if pred_track is not None:
        slice_cid = [(feat[cid]["slice"], cid) for cid in pred_track["cell_ids"] if cid in feat]
        slice_cid.sort()
        for s, cid in slice_cid:
            contour = feat[cid]["contour_abs"]
            z = s * SLICE_THICKNESS_UM
            diff_polys.append(np.column_stack([contour, np.full(len(contour), z)]))
            diff_colors.append("#009E73" if cid in gt_set else "#D55E00")
    if diff_polys:
        pc = Poly3DCollection(diff_polys, facecolors=diff_colors, edgecolors="0.15",
                              linewidths=0.2, alpha=0.65)
        ax.add_collection3d(pc)
    gt_outline_lines = [np.vstack([np.column_stack([c, np.full(len(c), z)]),
                                   np.column_stack([c[:1], np.full(1, z)])])
                        for c, z in zip(gt_contours, gt_zs)]
    if gt_outline_lines:
        lc = Line3DCollection(gt_outline_lines, colors="0.25",
                              linewidths=0.35, alpha=0.7)
        ax.add_collection3d(lc)
    _zoom_box(ax, all_contours_for_zoom, zoom_z)
    ax.set_title("Difference\n(green=correct cell, red=wrong)", fontsize=8)
    fig.suptitle(title, fontsize=10, y=1.02)
    plt.tight_layout()
    _savefig(out_pdf)


def _contours_from_track(track: Dict[str, Any],
                         feat: Dict[str, Dict[str, Any]]) -> Tuple[List[np.ndarray], List[float]]:
    ordered = sorted([(feat[cid]["slice"], feat[cid]["contour_abs"])
                      for cid in track["cell_ids"] if cid in feat])
    contours = [c for _, c in ordered]
    zs = [s * SLICE_THICKNESS_UM for s, _ in ordered]
    return contours, zs


def _zoom_box(ax, contours: List[np.ndarray], z_range: Tuple[float, float]):
    if not contours:
        set_3d_axes_equal(ax, (0, BOX_X_UM), (0, BOX_Y_UM), (0, BOX_Z_UM))
        return
    all_xy = np.vstack(contours)
    xlo, xhi = float(all_xy[:, 0].min()) - 8, float(all_xy[:, 0].max()) + 8
    ylo, yhi = float(all_xy[:, 1].min()) - 8, float(all_xy[:, 1].max()) + 8
    zlo = max(0.0, z_range[0] - 5); zhi = min(BOX_Z_UM, z_range[1] + 5)
    set_3d_axes_equal(ax, (xlo, xhi), (ylo, yhi), (zlo, zhi))


def fig_method_disagreement(methods_tracks: Dict[str, List[Dict[str, Any]]],
                            gt_tracks: List[Dict[str, Any]],
                            feat: Dict[str, Dict[str, Any]],
                            out_pdf: str, target_gt_id: Optional[int] = None):
    """Top-view of one GT cell's trajectory with each method's predicted partner highlighted."""
    if not gt_tracks:
        return
    # Choose a representative GT track (~mid-length)
    if target_gt_id is None:
        lengths = [(len(t["cell_ids"]), int(t["track_id"])) for t in gt_tracks]
        lengths.sort()
        target_gt_id = lengths[len(lengths) // 2][1]
    gt_track = next((t for t in gt_tracks if int(t["track_id"]) == target_gt_id), None)
    if gt_track is None:
        return

    fig, ax = plt.subplots(figsize=(120 / 25.4, 95 / 25.4))
    # Draw GT cells grey
    gt_centroids = []
    for cid in gt_track["cell_ids"]:
        if cid in feat:
            contour = feat[cid]["contour_abs"]
            ax.fill(contour[:, 0], contour[:, 1], facecolor="0.88",
                    edgecolor="0.3", linewidth=0.3, alpha=0.75)
            gt_centroids.append((feat[cid]["slice"], feat[cid]["centroid"]))
    gt_centroids.sort()
    if len(gt_centroids) > 1:
        xy = np.array([c[1] for c in gt_centroids])
        ax.plot(xy[:, 0], xy[:, 1], "-", color="0.25", lw=0.7,
                label=f"GT track #{target_gt_id}")

    # For each method, find which cell was predicted "next" after each GT cell
    rows_csv = []
    for mi, (m, tracks) in enumerate([(m, methods_tracks.get(m, [])) for m in METHOD_ORDER]):
        gt_id_map = _best_gt_id_per_pred(tracks, gt_tracks)
        method_pred_set: List[str] = []
        for t in tracks:
            if gt_id_map.get(t["track_id"], -1) == target_gt_id:
                method_pred_set.extend(t["cell_ids"])
        method_pred_set = list(set(method_pred_set))
        # Highlight cells that are predicted but NOT in GT
        extra = [cid for cid in method_pred_set if cid not in set(gt_track["cell_ids"])]
        extra_xy = []
        for cid in extra:
            if cid in feat:
                c = feat[cid]["centroid"]; extra_xy.append(c)
                rows_csv.append({"method": m, "extra_cell": cid,
                                 "x": float(c[0]), "y": float(c[1])})
        if extra_xy:
            xy = np.array(extra_xy)
            ax.plot(xy[:, 0], xy[:, 1], "o", color=OI.get(m, "#333"), markersize=2.5,
                    mec="0.15", mew=0.2, alpha=0.9, label=m)

    # zoom to GT track extent
    all_xy = np.array([c[1] for c in gt_centroids])
    if len(all_xy) > 0:
        ax.set_xlim(all_xy[:, 0].min() - 50, all_xy[:, 0].max() + 50)
        ax.set_ylim(all_xy[:, 1].min() - 50, all_xy[:, 1].max() + 50)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    ax.legend(fontsize=6, frameon=True, edgecolor="0.7", loc="upper right")
    ax.set_title(f"Method disagreement around GT track #{target_gt_id}", fontsize=9)
    _savefig(out_pdf)
    pd.DataFrame(rows_csv).to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def fig_centroid_ambiguity_map(phantom_dir: str, cache_file: str, out_pdf: str,
                               title: str = "") -> Optional[Dict[str, Any]]:
    """Histogram of the GT-correct target's position in the centroid-sorted
    candidate shortlist (the cache already sorts candidates by centroid distance).

    A concentration of mass at "Rank 1" means Centroid_NN already wins most
    pairs -> HiDTW cannot help. A heavy tail at rank >= 2 is the regime in which
    HiDTW is expected to provide the advantage. Data comes from the precomputed
    cost cache written by simulation01_run_methods.py; no re-computation is
    performed here.
    """
    if not os.path.isfile(cache_file):
        print(f"    [skip] centroid ambiguity - no cache: {cache_file}")
        return None
    gt_csv = os.path.join(phantom_dir, "ground_truth.csv")
    if not os.path.isfile(gt_csv):
        return None

    gt_df = pd.read_csv(gt_csv)
    gt_df["slice"] = gt_df["slice"].astype(int)
    gt_df["cell_id"] = gt_df["cell_id"].astype(str)
    cell2gt: Dict[Tuple[int, str], int] = {}
    slice_gt2cell: Dict[Tuple[int, int], str] = {}
    for _, r in gt_df.iterrows():
        cell2gt[(int(r["slice"]), str(r["cell_id"]))] = int(r["gt_track_id"])
        slice_gt2cell[(int(r["slice"]), int(r["gt_track_id"]))] = str(r["cell_id"])

    try:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"    [skip] centroid ambiguity - cache read failed: {e}")
        return None

    ranks: List[int] = []          # 1-indexed; 0 = GT target not in shortlist
    shortlist_lengths: List[int] = []
    for direction_name in ("forward", "backward"):
        direction = data.get(direction_name, {})
        sign = +1 if direction_name == "forward" else -1
        for src_slice, cmap in direction.items():
            for src_cell_id, cand_list in cmap.items():
                gt_track = cell2gt.get((int(src_slice), str(src_cell_id)))
                if gt_track is None:
                    continue
                target_slice = int(src_slice) + sign
                gt_cell_id = slice_gt2cell.get((target_slice, int(gt_track)))
                if gt_cell_id is None:
                    continue
                shortlist_lengths.append(len(cand_list))
                rank = 0
                for i, c in enumerate(cand_list):
                    if str(c.get("target_id")) == str(gt_cell_id):
                        rank = i + 1
                        break
                ranks.append(rank)

    if not ranks:
        print(f"    [skip] centroid ambiguity - no GT-matched pairs")
        return None

    ranks_arr = np.array(ranks, dtype=int)
    cats = ["Rank 1", "Rank 2", "Rank 3-5", "Rank 6-10", "Rank >10", "Not in shortlist"]
    counts = [
        int(np.sum(ranks_arr == 1)),
        int(np.sum(ranks_arr == 2)),
        int(np.sum((ranks_arr >= 3) & (ranks_arr <= 5))),
        int(np.sum((ranks_arr >= 6) & (ranks_arr <= 10))),
        int(np.sum(ranks_arr > 10)),
        int(np.sum(ranks_arr == 0)),
    ]
    total = int(np.sum(counts))
    fractions = [c / total if total else 0.0 for c in counts]
    pct_not_top1 = 100.0 * (total - counts[0]) / total if total else 0.0
    pct_in_shortlist = 100.0 * np.mean(ranks_arr > 0) if total else 0.0

    palette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00"]
    fig, ax = plt.subplots(figsize=(96 / 25.4, 60 / 25.4))
    ax.bar(cats, counts, color=palette, edgecolor="0.2", linewidth=0.45)
    ax.set_ylabel("Count of (source, target) GT pairs", fontsize=8)
    ax.set_xlabel("Centroid rank of GT-correct target in shortlist", fontsize=8)
    headline = title or "Centroid-shortlist ambiguity"
    ax.set_title(f"{headline}  (n={total}, non-top-1 = {pct_not_top1:.1f}%)",
                 fontsize=9)
    ax.tick_params(labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right")
    _savefig(out_pdf)
    pd.DataFrame({"category": cats, "count": counts, "fraction": fractions}).to_csv(
        out_pdf.replace(".pdf", ".csv"), index=False
    )
    return {
        "n_pairs": total,
        "pct_not_top1": pct_not_top1,
        "pct_in_shortlist": pct_in_shortlist,
        "mean_shortlist_len": float(np.mean(shortlist_lengths)) if shortlist_lengths else 0.0,
    }


def fig_hidtw_weight_simplex(grid_csv: str, tuned_json: str, out_pdf: str):
    """Ternary scatter of HiDTW weight-simplex grid coloured by GT pair-F1.

    The ternary (barycentric) plot has the three weights (dtw, ot, centroid)
    as the corners of an equilateral triangle. The best point (auto-tuned) is
    highlighted with a star. Rendered as one figure per alpha value in the
    grid (stacked vertically).
    """
    if not os.path.isfile(grid_csv):
        return
    df = pd.read_csv(grid_csv)
    if df.empty:
        return
    tuned = {}
    if os.path.isfile(tuned_json):
        with open(tuned_json, encoding="utf-8") as f:
            tuned = json.load(f).get("best", {})

    alphas = sorted(df["alpha"].unique())
    n = len(alphas)
    fig, axes = plt.subplots(1, n, figsize=(n * 58 / 25.4, 58 / 25.4))
    if n == 1:
        axes = [axes]

    def barycentric_to_xy(w_dtw, w_ot, w_cent):
        # DTW corner top, OT bottom-left, CENT bottom-right
        A = np.array([0.5, np.sqrt(3) / 2])
        B = np.array([0.0, 0.0])
        C = np.array([1.0, 0.0])
        return w_dtw * A + w_ot * B + w_cent * C

    f1_min = float(df["f1"].min()); f1_max = float(df["f1"].max())
    for ai, a in enumerate(alphas):
        ax = axes[ai]
        sub = df[df["alpha"] == a]
        xs, ys, cs = [], [], []
        for _, row in sub.iterrows():
            x, y = barycentric_to_xy(row["dtw"], row["ot"], row["centroid"])
            xs.append(x); ys.append(y); cs.append(row["f1"])
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=40, edgecolors="0.2",
                        linewidths=0.3, vmin=f1_min, vmax=f1_max)
        # triangle outline
        A = np.array([0.5, np.sqrt(3) / 2]); B = np.array([0.0, 0.0]); C = np.array([1.0, 0.0])
        tri = np.vstack([A, B, C, A])
        ax.plot(tri[:, 0], tri[:, 1], "-", color="0.25", lw=0.6)
        ax.text(A[0], A[1] + 0.05, "DTW", ha="center", fontsize=7)
        ax.text(B[0] - 0.03, B[1] - 0.04, "OT", ha="right", va="top", fontsize=7)
        ax.text(C[0] + 0.03, C[1] - 0.04, "Cent", ha="left", va="top", fontsize=7)
        # mark best point if this alpha is the winner
        if tuned and abs(float(tuned.get("alpha", -1)) - a) < 1e-9:
            bx, by = barycentric_to_xy(tuned["dtw"], tuned["ot"], tuned["centroid"])
            ax.plot(bx, by, "*", color="#D55E00", markersize=18,
                    mec="0.1", mew=0.6, zorder=5,
                    label=f"best F1={tuned['f1']:.3f}")
            ax.legend(fontsize=6, loc="upper right", frameon=True, edgecolor="0.6")
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"alpha = {a:.3f}", fontsize=8)
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02, aspect=30)
    cb.set_label("GT pair-F1", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle("HiDTW_full weight-simplex tuning (ternary heatmap)", fontsize=9, y=1.02)
    _savefig(out_pdf)
    df.to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


# --- 4. PLOTLY EXPORT (optional) ---

def export_plotly_overview(phantoms: List[Phantom], out_html: str,
                           max_cells: int = 80):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(f"      [plotly skip] plotly not installed: {out_html}")
        return
    fig = go.Figure()
    sel = phantoms[:max_cells]
    for i, ph in enumerate(sel):
        surf = sample_phantom_surface(ph, n_t=40, n_u=16).reshape(40, 16, 3)
        xs = surf[..., 0].ravel(); ys = surf[..., 1].ravel(); zs = surf[..., 2].ravel()
        col = matplotlib.colors.to_hex(gt_color(i, len(sel)))
        fig.add_trace(go.Mesh3d(
            x=xs, y=ys, z=zs,
            alphahull=8, opacity=0.45,
            color=col, name=f"cell_{ph.gt_id}"))
    fig.update_layout(
        scene=dict(xaxis_title="X (um)", yaxis_title="Y (um)", zaxis_title="Z (um)",
                   aspectmode="data"),
        title="3D phantom overview (interactive)", showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"      -> {out_html}")


# --- 5. ONE DATASET RENDERING ---

def render_one(condition: str, seed: int, phantom_root: str,
               method_root: str, out_root: str,
               methods: List[str], do_plotly: bool = False,
               all_figures: bool = False,
               cache_root: Optional[str] = None) -> Dict[str, Any]:
    """Render the essential figures for one (condition, seed).

    By default emits only the four figures that carry the HiDTW-advantage
    argument:
      1. fig_viz_phantom_3d_overview
      2. fig_viz_tracks_3d_sixpanel_*  (one 600 dpi TIFF per panel: GT + each
         method in METHOD_ORDER; stem ``fig_viz_tracks_3d_sixpanel``; no z-axis
         label or z tick numbers)
      3. fig_viz_centroid_ambiguity_map (NEW - reads precomputed cache)
      4. fig_viz_hidtw_weight_simplex

    Pass ``all_figures=True`` (CLI ``--all-figures``) to also emit the legacy
    slice / neighbor / topview / id-switch / case / disagreement figures.
    """
    phantom_dir = os.path.join(phantom_root, condition, f"seed{seed}")
    if not os.path.isdir(phantom_dir):
        return {"condition": condition, "seed": seed, "skipped": True}

    out_dir = os.path.join(out_root, condition, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    phantoms_dump, gt_df, feat = load_phantom_and_gt(phantom_dir)
    phantoms = [_phantom_from_dict(d) for d in phantoms_dump["phantoms"]]

    if cache_root is None:
        cache_root = os.path.join(os.path.dirname(method_root), "cache")

    # ---- A. Phantom data (essential: overview only) -----------------------
    print(f"    [A] phantom visualisation …")
    fig_phantom_3d_overview(phantoms, gt_df, feat,
                            os.path.join(out_dir, "fig_viz_phantom_3d_overview.pdf"))
    if all_figures:
        fig_levels_gallery(phantoms, os.path.join(out_dir, "fig_viz_levels_gallery.pdf"))
        fig_phantom_3d_tilt_effect(
            phantoms, os.path.join(out_dir, "fig_viz_phantom_3d_tilt_effect.pdf"),
            tilt_deg=max(resolve_condition_config(condition).get("slice_tilt_deg", 0.0), 10.0))
        fig_slice_montage(gt_df, feat, os.path.join(out_dir, "fig_viz_slice_montage.pdf"))
        fig_slice_montage_3d_stack(gt_df, feat,
                                   os.path.join(out_dir, "fig_viz_slice_montage_3d_stack.pdf"))
        fig_neighbor_graph(gt_df, feat, phantom_dir,
                           os.path.join(out_dir, "fig_viz_neighbor_graph.pdf"))

    # ---- B. Reconstructed tracks (essential: multi-method panel) -----------
    print(f"    [B] reconstruction visualisation …")
    methods_tracks: Dict[str, List[Dict[str, Any]]] = {}
    method_f1: Dict[str, float] = {}
    for m in methods:
        tp = os.path.join(method_root, condition, f"seed{seed}", m, "tracks.csv")
        methods_tracks[m] = load_tracks(tp)
    eval_csv = os.path.join(os.path.dirname(method_root), "evaluation", "pair_metrics.csv")
    if os.path.isfile(eval_csv):
        try:
            edf = pd.read_csv(eval_csv)
            sub = edf[(edf["condition"] == condition) & (edf["seed"] == int(seed))]
            for m in methods:
                mv = sub[sub["method"] == m]["f1"]
                if len(mv):
                    method_f1[m] = float(mv.iloc[0])
        except Exception:
            pass

    gt_tracks = tracks_from_gt(gt_df)
    if methods_tracks:
        # TIFF @600 dpi per user request (text becomes rasterised in this file;
        # the vector PDF counterpart is intentionally not emitted).
        fig_tracks_3d_sixpanel(methods_tracks, gt_tracks, feat, method_f1,
                               os.path.join(out_dir, "fig_viz_tracks_3d_sixpanel.tiff"))
    if all_figures:
        fig_tracks_3d(gt_tracks, feat, "Ground truth",
                      os.path.join(out_dir, "fig_viz_tracks_3d_gt.pdf"))
        if "HiDTW_full" in methods_tracks and methods_tracks["HiDTW_full"]:
            fig_tracks_3d(methods_tracks["HiDTW_full"], feat,
                          "HiDTW_full reconstruction",
                          os.path.join(out_dir, "fig_viz_tracks_3d_hidtw.pdf"))
        if methods_tracks:
            fig_tracks_topview(methods_tracks, gt_tracks, feat,
                               os.path.join(out_dir, "fig_viz_tracks_topview.pdf"))
        if methods_tracks.get("HiDTW_full"):
            fig_id_switch_3d(methods_tracks["HiDTW_full"], gt_df, feat,
                             os.path.join(out_dir, "fig_viz_id_switch_3d.pdf"),
                             title="ID switches (HiDTW_full)")
            fig_error_map_topview(methods_tracks["HiDTW_full"], gt_df, feat,
                                  os.path.join(out_dir, "fig_viz_error_map_topview.pdf"),
                                  method="HiDTW_full")

    # ---- B'. HiDTW weight simplex (essential) ------------------------------
    if methods_tracks.get("HiDTW_full"):
        grid_csv = os.path.join(method_root, condition, f"seed{seed}",
                                "HiDTW_full", "grid_search.csv")
        tuned_json = os.path.join(method_root, condition, f"seed{seed}",
                                  "HiDTW_full", "tuned_weights.json")
        fig_hidtw_weight_simplex(grid_csv, tuned_json,
                                 os.path.join(out_dir, "fig_viz_hidtw_weight_simplex.pdf"))

    # ---- B''. Centroid ambiguity map (essential, NEW) ---------------------
    cache_file = os.path.join(cache_root, condition, f"seed{seed}", "precomputed_costs.pkl")
    ambiguity_info = fig_centroid_ambiguity_map(
        phantom_dir, cache_file,
        os.path.join(out_dir, "fig_viz_centroid_ambiguity_map.pdf"),
        title=f"{condition} / seed{seed}"
    )

    # ---- C. Case studies (legacy, gated) -----------------------------------
    if all_figures and methods_tracks.get("HiDTW_full"):
        print(f"    [C] case studies …")
        gt_s, pred_s = _pick_case_by_length(gt_tracks, methods_tracks["HiDTW_full"], feat,
                                            success=True)
        fig_case_3d(gt_s, pred_s, feat,
                    os.path.join(out_dir, "fig_viz_case_success.pdf"),
                    title="Success case (HiDTW_full vs GT)")
        gt_f, pred_f = _pick_case_by_length(gt_tracks, methods_tracks["HiDTW_full"], feat,
                                            success=False)
        fig_case_3d(gt_f, pred_f, feat,
                    os.path.join(out_dir, "fig_viz_case_failure.pdf"),
                    title="Failure case (HiDTW_full vs GT)")
        fig_method_disagreement(methods_tracks, gt_tracks, feat,
                                os.path.join(out_dir, "fig_viz_method_disagreement.pdf"))

    # ---- D. Plotly -----------------------------------------------------------
    if do_plotly:
        print(f"    [D] plotly HTML …")
        export_plotly_overview(phantoms,
                               os.path.join(out_dir, "fig_viz_phantom_3d_overview.html"))

    return {"condition": condition, "seed": seed,
            "n_phantoms": len(phantoms),
            "n_gt_tracks": int(gt_df["gt_track_id"].nunique()),
            "methods_found": [m for m, t in methods_tracks.items() if t],
            "out_dir": out_dir,
            "feat_n_cells": len(feat),
            "gt_df": gt_df, "feat": feat}


# --- 6. CLI ---

def main():
    parser = argparse.ArgumentParser(description="Simulation 1 visualisation.")
    parser.add_argument("--phantom-root", default="simulation01_results/phantoms")
    parser.add_argument("--method-root",  default="simulation01_results/methods")
    parser.add_argument("--out-root",     default="simulation01_results/visualization")
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="Only these simulation conditions (folder names under phantoms/). "
             "Example: --conditions L2_mid_clean L2_high_clean. "
             "Ignored if --viz-condition is set.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="If --seeds omitted, visualise only this many seeds per condition.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--viz-condition", nargs="+", default=None, metavar="COND",
        help="Shortcut: only these condition(s); uses a single seed (--viz-seed) for each. "
             "Example: --viz-condition L2_mid_clean L2_high_clean --viz-seed 0. "
             "For multiple seeds per condition, use --conditions … --seeds … instead.",
    )
    parser.add_argument("--viz-seed", type=int, default=0)
    parser.add_argument("--plotly", action="store_true",
                        help="Also export interactive HTML views (requires plotly).")
    parser.add_argument("--all-figures", action="store_true",
                        help="Emit the full (legacy) figure set. Default emits only "
                             "the four essential figures (overview / tracks panel / "
                             "centroid ambiguity / HiDTW simplex).")
    parser.add_argument("--cache-root", default=None,
                        help="Optional root of precomputed_costs.pkl cache "
                             "(default: <method-root parent>/cache).")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    if args.viz_condition is not None:
        cond_names = list(args.viz_condition)
        seeds = [int(args.viz_seed)]
    else:
        cond_names = args.conditions if args.conditions else list(CONDITIONS.keys())
        if args.seeds is not None:
            seeds = list(args.seeds)
        else:
            seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    unknown = [c for c in cond_names if c not in CONDITIONS]
    for c in unknown:
        print(f"  [WARN] condition {c!r} not in CONDITIONS registry - "
              f"phantom dir must still exist under --phantom-root.")

    methods = args.methods if args.methods else list(METHOD_SPECS.keys())

    print("Simulation 1 - visualisation")
    print(f"  conditions: {cond_names}")
    print(f"  seeds:      {seeds}")
    print(f"  methods:    {methods}")
    print(f"  out-root:   {args.out_root}")

    infos: List[Dict[str, Any]] = []
    for c in cond_names:
        for s in seeds:
            print(f"\n=== {c} / seed{s} ===")
            info = render_one(c, s, args.phantom_root, args.method_root,
                              args.out_root, methods, do_plotly=args.plotly,
                              all_figures=args.all_figures,
                              cache_root=args.cache_root)
            infos.append(info)

    # Cross-condition gallery (legacy, gated)
    if args.all_figures and len(infos) >= 2 and any("gt_df" in i for i in infos):
        out_pdf = os.path.join(args.out_root, "fig_viz_condition_examples.pdf")
        os.makedirs(args.out_root, exist_ok=True)
        fig_condition_examples([i for i in infos if "gt_df" in i], out_pdf)

    # Summary json (drop non-serialisable fields)
    os.makedirs(args.out_root, exist_ok=True)
    slim = [{k: v for k, v in i.items() if k not in ("gt_df", "feat")}
            for i in infos]
    with open(os.path.join(args.out_root, "_visualisation_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, default=str)
    print("\nDone.")


if __name__ == "__main__":
    main()
