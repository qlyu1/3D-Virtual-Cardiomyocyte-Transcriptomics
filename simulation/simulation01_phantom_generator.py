#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation 1 - 3D Digital Cardiomyocyte Phantom Generator.

Generates Level 1 (ideal capsule) and Level 2 (curved / tapered / twisted)
3D cardiomyocyte phantoms, virtually sections them into 10 z-slices of
10 um thickness, and exports per-slice feature CSVs **with the exact same
schema as 03_demo/Tile046/046_S01_features.csv** so they can be consumed
directly by ``cell_reconstruction_runGPU_3Drevised_v10.load_cells_from_csv_files``.

Outputs per (condition, seed):
  simulation01_results/phantoms/<condition>/seed<k>/
      SIM<condition>_S01_features.csv ... SIM<condition>_S10_features.csv
      ground_truth.csv
      phantoms.json

Usage:
  python simulation01_phantom_generator.py                 # full grid
  python simulation01_phantom_generator.py --quick         # 2 conditions x 1 seed
  python simulation01_phantom_generator.py --conditions L1_mid_clean L2_high_tilt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from tqdm import tqdm

# --- 1. GLOBAL CONFIGURATION ---

# All lengths are in micrometers unless explicitly noted.
BOX_X_UM = 500.0
BOX_Y_UM = 500.0
BOX_Z_UM = 100.0
SLICE_THICKNESS_UM = 10.0
N_SLICES = 10
NUM_CONTOUR_POINTS = 400          # MUST match the Tile046 schema (x_0..x_399)
NEIGHBOR_DIST_UM = 45.0           # adjacent_cells threshold inside a slice
MIN_SLICE_AREA_UM2 = 15.0         # slices smaller than this are dropped as non-observations

# Density presets: minimum centroid-xy separation (in um) during Poisson-disk sampling.
# Calibrated against Tile046 S05 reference (see `_compute_tile046_calibration`):
# real cardiac tissue has NN / r_eff = 1.44 with cell area coverage ~81%.
# With our Tile046-aspect cell sizes (b0 in [11,16], c0 in [6,10] -> r_eff
# median ~11 um), ``ultra`` = 20 um is the tightest packing that still lets
# rigid ellipse contours stay near-touching without systematic overlap -
# real tissue exceeds this because cells deform, but that is outside the
# scope of the abstract phantom. ``overlap_cell_rate`` is logged per slice
# so residual crossings can be spot-checked.
DENSITY_MIN_DIST = {"sparse": 50.0, "mid": 26.0, "dense": 20.0, "ultra": 16.0}

# Level 2 curvature amplitudes (peak perpendicular displacement at t=0.5, in um).
CURV_AMPLITUDE = {"none": 0.0, "low": 6.0, "high": 14.0}

# ---------------------------------------------------------------------------
# Condition schema. Each entry in CONDITIONS only lists keys that DIFFER from
# CONDITION_DEFAULTS; the effective per-condition config is built via
# ``resolve_condition_config``.
#
# New keys introduced for the centroid-ambiguity stress tests:
#   per_cell_xy_jitter_um
#       Per (cell, slice) independent Gaussian lateral displacement (breaks
#       centroid continuity cell-by-cell while shape and OT neighbour set
#       still resolve identity).
#   warp_amp_um, warp_sigma_um, warp_n_bumps
#       Per-slice smooth 2D displacement field built from a small number of
#       low-frequency Gaussian bumps (models local mechanical deformation
#       during tissue sectioning).
#   placement
#       "poisson" (legacy Poisson-disk sampling) or "clustered" (clusters of
#       3-6 coherently-oriented cells; multiple candidates will have
#       near-identical centroids and can only be resolved by shape / OT).
#   cluster_center_dist, intra_cluster_dist, cluster_size_range,
#   cluster_direction_wobble_deg
#       Only used when placement == "clustered".
#   distractor_mode
#       "blob" (legacy short round non-myocyte) or "myocyte_like" (longer,
#       elongated, off-axis non-myocytes without coherent OT neighbour set
#       - Centroid-NN cannot easily reject these).
# ---------------------------------------------------------------------------

CONDITION_DEFAULTS: Dict[str, Any] = {
    "level": 2,
    "density": "mid",
    "min_dist_um": None,            # Optional override of DENSITY_MIN_DIST[density].
    # Shape-along-z is deliberately kept constant by default: taper=1.0,
    # twist_deg=0.0 and curv="none" mean each cell's perpendicular cross-
    # section is invariant along its axis, so virtual sections at different z
    # give near-identical 2D contours for the same cell. This is the regime
    # where DTW / OT shape-matching can actually help; sub-conditions below
    # add centroid-adversarial perturbations (inter-slice shift, clusters,
    # jitter, distractors) WITHOUT reintroducing shape-along-z variation.
    "curv": "none",
    "taper": 1.0,
    "twist_deg": 0.0,
    "slice_tilt_deg": 0.0,
    "interlayer_shift": False,
    "thickness_jitter": False,
    "disarray_max_deg": 12.0,       # inter-cell orientation diversity (intra-cell shape still constant)
    "n_distractors": 0,
    "distractor_mode": "blob",
    # Per-slice fibroblast-like small-contour distractors. Each slice
    # independently samples ``per_slice_distractors_per_slice`` 2D contours
    # (area uniform in ``per_slice_distractor_area_um2_range``, aspect uniform
    # in ``per_slice_distractor_aspect_range``) with random centroid location
    # and random in-plane rotation. Unlike the legacy 3D ``n_distractors``
    # mechanism, these contours have NO cross-slice identity - they do not
    # appear in ``ground_truth.csv`` and are not affected by interlayer
    # rigid shift or per-cell jitter. Any method that links them across
    # slices incurs only false-positive pairs, matching the real-tissue
    # intuition that fibroblasts are small, scattered, and hard to trace.
    "per_slice_distractors_per_slice": 0,
    "per_slice_distractor_area_um2_range": (30.0, 100.0),
    "per_slice_distractor_aspect_range": (1.0, 1.5),
    "per_slice_distractor_min_dist_to_myocyte_um": 8.0,
    "per_cell_xy_jitter_um": 0.0,
    # DTW-adversarial per-slice shape perturbations. These are applied
    # INDEPENDENTLY for every (cell, slice) pair and break the "shape invariant
    # along z" assumption that the original 14-condition catalogue relied on.
    # Both stay at 0.0 by default so every pre-existing condition continues to
    # produce byte-identical phantoms.
    #   * per_slice_shape_rot_sigma_deg: sigma (deg) of Gaussian rotation
    #     applied to the contour about its own centroid, drawn independently
    #     for each (cell, slice). Centroid / area / neighbour set are
    #     preserved; DTW-only is attacked.
    #   * per_slice_contour_radial_noise_um: sigma (um) of per-contour-point
    #     radial Gaussian noise (zero-mean along the local outward direction
    #     from the cell centroid), drawn independently for each point of each
    #     (cell, slice). Approximate segmentation-edge noise. Preserves
    #     centroid / area; attacks DTW point-wise distance.
    "per_slice_shape_rot_sigma_deg": 0.0,
    "per_slice_contour_radial_noise_um": 0.0,
    "warp_amp_um": 0.0,
    "warp_sigma_um": 80.0,
    "warp_n_bumps": 3,
    "placement": "poisson",
    "cluster_center_dist": 65.0,
    "intra_cluster_dist": 20.0,
    "cluster_size_range": (3, 6),
    "cluster_direction_wobble_deg": 5.0,
    # Rigid inter-slice drift (translation + small rotation around box centre).
    # Rotation is kept at 0.0 by default and MUST NOT be overridden to a
    # non-zero value by any condition below: inter-slice rotation rotates 2D
    # cross-sections layer-to-layer and would defeat the shape-along-z
    # constancy the new design relies on.
    "interlayer_shift_mode": "off",      # "off" | "iid" | "random_walk"
    "interlayer_shift_sigma_um": 0.0,
    "interlayer_rot_sigma_deg": 0.0,
    # Z-coherent regional warp: bumps share (cx, cy, sigma, direction) across
    # z while their amplitudes form a low-pass random sequence smoothed with
    # a Gaussian kernel of width ``z_coherent_warp_tau_slices``.
    # Amplitudes are kept at 0.0 by default - warp perturbs shape per slice
    # and is deliberately excluded from the current condition set.
    "warp_mode": "per_slice",            # "per_slice" (legacy) | "z_coherent"
    "z_coherent_warp_amp_um": 0.0,
    "z_coherent_warp_sigma_um": 80.0,
    "z_coherent_warp_n_bumps": 3,
    "z_coherent_warp_tau_slices": 3.0,
}

# ---------------------------------------------------------------------------
# Condition catalogue - centroid-adversarial, shape-stable-along-z design.
#
# ALL conditions below inherit the new defaults (taper=1.0, twist_deg=0.0,
# curv="none", warp_amp=0, rot_sigma=0) which guarantee that, for a given
# cell, its perpendicular cross-section is identical at every z slice.
# Inter-cell shape diversity is provided by the disarray (per-cell tilt) and
# by sampling from the Tile046 shape library (see CLI ``--no-shape-library``
# for the fallback analytic-ellipse path).
#
# Stressors included:
#   - density / clustering              (centroid ambiguity within a slice)
#   - per-cell xy jitter                (centroid drifts; shape still constant)
#   - myocyte-like distractors          (centroid cannot reject look-alikes)
#   - rigid inter-slice shift (iid/rw)  (centroid pairs drift; DTW sees same
#                                        shape, OT sees same neighbourhood)
#
# Stressors deliberately excluded (they would break shape-along-z and
# therefore degrade DTW / OT):
#   - taper != 1.0, twist_deg != 0     (intra-cell shape drift along z)
#   - curv != "none"                   (curved axis -> differing oblique cuts)
#   - warp_amp_um / z_coherent warp    (per-slice or z-coherent non-rigid
#                                       shape deformation)
#   - interlayer_rot_sigma_deg > 0     (inter-slice rotation of 2D contours)
#   - thickness_jitter                 (per-slice thickness perturbation)
# ---------------------------------------------------------------------------

CONDITIONS: Dict[str, Dict[str, Any]] = {
    # Sanity baseline - no perturbation; every method should reach ~1.0 F1.
    "L2_clean_sanity":      {"density": "mid",
                             "disarray_max_deg": 10.0},

    # --- Density x per-cell jitter (reduced from 3 to 2 conditions) -----
    # Shared per_cell_xy_jitter_um = 2.5; sweeping Poisson-disk ``min_dist``
    # between mid (23 um, explicit) and ultra (16 um from DENSITY_MIN_DIST).
    "L2_mid_jitter":        {"density": "mid",
                             "min_dist_um": 23.0,
                             "disarray_max_deg": 15.0,
                             "per_cell_xy_jitter_um": 2.5},

    "L2_percell_jitter":    {"density": "ultra",
                             "disarray_max_deg": 15.0,
                             "per_cell_xy_jitter_um": 2.5},

    # Tight clusters of 4-6 coherently-oriented myocytes with small wobble.
    # Cluster-internal centroids are ambiguous; shape (plus OT neighbour
    # structure) is expected to resolve identity.
    "L2_clustered_dense":   {"density": "dense",
                             "disarray_max_deg": 15.0,
                             "placement": "clustered",
                             "cluster_center_dist": 45.0,
                             "intra_cluster_dist": 15.0,
                             "cluster_size_range": (4, 6),
                             "cluster_direction_wobble_deg": 3.0},

    # ----------------------------------------------------------------------
    # Centroid-adversarial inter-slice shift (IID) - amplitude gradient
    # ----------------------------------------------------------------------
    # Rigid iid drift: every slice has an independent xy shift drawn from
    # N(0, sigma). ``interlayer_rot_sigma_deg = 0`` ensures 2D contours do
    # not rotate between slices, so DTW/OT still see identical shapes for
    # a given cell while centroid pairs drift by O(sigma) um per slice.
    "L2_shift_iid_small":   {"density": "dense",
                             "disarray_max_deg": 15.0,
                             "interlayer_shift_mode": "iid",
                             "interlayer_shift_sigma_um": 5.0,
                             "interlayer_rot_sigma_deg": 0.0},

    "L2_shift_iid_mid":     {"density": "dense",
                             "disarray_max_deg": 15.0,
                             "interlayer_shift_mode": "iid",
                             "interlayer_shift_sigma_um": 10.0,
                             "interlayer_rot_sigma_deg": 0.0},

    "L2_shift_iid_high":    {"density": "dense",
                             "disarray_max_deg": 15.0,
                             "interlayer_shift_mode": "iid",
                             "interlayer_shift_sigma_um": 15.0,
                             "interlayer_rot_sigma_deg": 0.0},

    "L2_shift_iid_extreme": {"density": "dense",
                             "disarray_max_deg": 15.0,
                             "interlayer_shift_mode": "iid",
                             "interlayer_shift_sigma_um": 20.0,
                             "interlayer_rot_sigma_deg": 0.0},

    # Clusters + iid shift: within-cluster centroid ambiguity is worsened
    # by inter-slice translation; shape (via DTW) and cluster OT structure
    # are the only signals that should recover identity.
    "L2_clusters_shift":    {"density": "dense",
                             "disarray_max_deg": 15.0,
                             "placement": "clustered",
                             "cluster_center_dist": 45.0,
                             "intra_cluster_dist": 15.0,
                             "cluster_size_range": (4, 6),
                             "cluster_direction_wobble_deg": 3.0,
                             "interlayer_shift_mode": "iid",
                             "interlayer_shift_sigma_um": 10.0,
                             "interlayer_rot_sigma_deg": 0.0},

    "L2_clusters_shift_high": {"density": "dense",
                               "disarray_max_deg": 15.0,
                               "placement": "clustered",
                               "cluster_center_dist": 45.0,
                               "intra_cluster_dist": 15.0,
                               "cluster_size_range": (4, 6),
                               "cluster_direction_wobble_deg": 3.0,
                               "interlayer_shift_mode": "iid",
                               "interlayer_shift_sigma_um": 15.0,
                               "interlayer_rot_sigma_deg": 0.0},

    # Inter-slice shift combined with within-slice per-cell jitter (no
    # clustering, no distractors). Isolates the "layer translation + layer
    # jitter" interaction - both noise sources attack Centroid, neither
    # affects DTW/OT (contour shape is preserved).
    "L2_shift_plus_jitter": {"density": "mid",
                             "min_dist_um": 23.0,
                             "disarray_max_deg": 15.0,
                             "per_cell_xy_jitter_um": 2.5,
                             "interlayer_shift_mode": "iid",
                             "interlayer_shift_sigma_um": 10.0,
                             "interlayer_rot_sigma_deg": 0.0},

    # ----------------------------------------------------------------------
    # Per-slice fibroblast-like distractors (no cross-slice identity)
    # ----------------------------------------------------------------------
    # Small round-ish 2D contours (area 30-100 um^2, aspect 1.0-1.5) placed
    # independently on each slice. They are NOT in ground_truth.csv - any
    # method that links them across slices produces only false-positive
    # pairs. Good test for OT cluster / DTW shape rejecting fibroblasts.
    # Per-slice count sized to ~1/3 of the dense myocyte count
    # (measured ~333 myocytes/slice on previous runs -> 120 fibroblasts).
    "L2_fibroblasts_noshift": {"density": "dense",
                               "disarray_max_deg": 15.0,
                               "per_slice_distractors_per_slice": 120,
                               "per_slice_distractor_area_um2_range": (30.0, 100.0),
                               "per_slice_distractor_aspect_range": (1.0, 1.5)},

    # Fibroblast distractors combined with iid inter-slice shift: the
    # fibroblasts dilute OT neighbourhood coherence and Centroid_NN pairs
    # get yanked across slices; HiDTW should still resolve identity via
    # shape + (cluster) OT of the real myocytes.
    "L2_distractors_shift": {"density": "dense",
                             "disarray_max_deg": 15.0,
                             "per_slice_distractors_per_slice": 120,
                             "per_slice_distractor_area_um2_range": (30.0, 100.0),
                             "per_slice_distractor_aspect_range": (1.0, 1.5),
                             "interlayer_shift_mode": "iid",
                             "interlayer_shift_sigma_um": 10.0,
                             "interlayer_rot_sigma_deg": 0.0},

    # Headline stress - clusters + per-cell jitter + iid shift.
    # Pure centroid-adversarial design: NO distractors, NO warp, NO
    # rotation, NO taper/twist.
    "L2_stress_combo":      {"density": "dense",
                             "disarray_max_deg": 20.0,
                             "placement": "clustered",
                             "cluster_center_dist": 42.0,
                             "intra_cluster_dist": 15.0,
                             "cluster_size_range": (4, 6),
                             "cluster_direction_wobble_deg": 4.0,
                             "per_cell_xy_jitter_um": 2.0,
                             "interlayer_shift_mode": "iid",
                             "interlayer_shift_sigma_um": 10.0,
                             "interlayer_rot_sigma_deg": 0.0},

    # ----------------------------------------------------------------------
    # DTW-adversarial per-slice shape perturbations (C15-C18).
    # ----------------------------------------------------------------------
    # These re-introduce inter-slice shape variability that the 14-condition
    # catalogue deliberately avoids, so that DTW_only - which happily scores
    # F1 ≈ 0.85 on every shift/cluster/distractor condition above - now
    # actually has something to lose. Centroid_NN and OT_only remain mostly
    # intact because each mechanism preserves centroid / area / neighbour
    # structure; HiDTW_full should keep a visible margin via its OT +
    # centroid components.

    # Per-slice independent non-rigid warp field. Each slice gets its own
    # (amp, sigma, n_bumps) Gaussian-bump displacement field applied in the
    # absolute xy plane; warp amplitude 4 um against b0 ≈ 8 um is a
    # noticeable shape deformation but leaves cell centroids (which average
    # over the whole contour) nearly unchanged.
    "L2_per_slice_warp":    {"density": "mid",
                             "disarray_max_deg": 15.0,
                             "warp_mode": "per_slice",
                             "warp_amp_um": 4.0,
                             "warp_sigma_um": 60.0,
                             "warp_n_bumps": 3},

    # Per-(cell, slice) rotation of contour about its own centroid,
    # sigma = 15 deg (~95% of rotations within ±30 deg). Centroid, area and
    # OT neighbour structure are exactly preserved; only shape-DTW pays.
    "L2_per_slice_rotation":{"density": "mid",
                             "disarray_max_deg": 15.0,
                             "per_slice_shape_rot_sigma_deg": 15.0},

    # Per-contour-point radial Gaussian noise, sigma = 2 um (~25% of b0).
    # Approximates segmentation-edge jitter. Preserves centroid / area.
    "L2_per_slice_contour_noise": {"density": "mid",
                                   "disarray_max_deg": 15.0,
                                   "per_slice_contour_radial_noise_um": 2.0},

    # DTW stress combo: moderate levels of all three shape perturbations
    # AND a medium iid inter-slice shift (sigma = 10 um). Designed to
    # reproduce the compound noise profile of real histological serial
    # sections (segmentation noise + tissue distortion + slice mis-
    # alignment) and is expected to be the hardest condition in the suite.
    "L2_dtw_stress_combo":  {"density": "dense",
                             "disarray_max_deg": 15.0,
                             "warp_mode": "per_slice",
                             "warp_amp_um": 2.5,
                             "per_slice_shape_rot_sigma_deg": 10.0,
                             "per_slice_contour_radial_noise_um": 1.0,
                             "interlayer_shift_mode": "iid",
                             "interlayer_shift_sigma_um": 10.0,
                             "interlayer_rot_sigma_deg": 0.0},
}
QUICK_CONDITIONS = ["L2_clean_sanity", "L2_shift_iid_mid", "L2_distractors_shift"]


def resolve_condition_config(condition_name: str) -> Dict[str, Any]:
    """Merge ``CONDITION_DEFAULTS`` with per-condition overrides into a fresh dict."""
    if condition_name not in CONDITIONS:
        raise KeyError(f"Unknown condition: {condition_name}")
    cfg = dict(CONDITION_DEFAULTS)
    cfg.update(CONDITIONS[condition_name])
    return cfg


# --- 2. PHANTOM DATA CLASS ---

@dataclass
class Phantom:
    """Analytic 3D cardiomyocyte phantom as a swept ellipse along a curved axis.

    Axis gamma(t) = P0 + T * t + curv_dir * (curv_amp * 4 * t * (1-t)),  t in [0,1]
    Cross-section at parameter t is an ellipse perpendicular to gamma'(t) with
    semi-axes b(t) = b0 * (1 + (taper-1)*t), c(t) = c0 * (1 + (taper-1)*t),
    rotated about the tangent by twist_rad(t) = twist_total_rad * t.
    """
    gt_id: int
    level: int
    length_um: float
    b0: float
    c0: float
    taper: float
    twist_total_rad: float
    P0: np.ndarray          # shape (3,)
    T: np.ndarray           # shape (3,) - straight-line vector from P0 to end (length = length_um)
    curv_dir: np.ndarray    # shape (3,) - unit perpendicular, direction of bending
    curv_amp: float         # peak perpendicular displacement (um)
    volume_um3: float = 0.0
    curv_category: str = "none"
    # Optional (NUM_CONTOUR_POINTS, 2) unit-area canonical cross-section
    # template (from the Tile046 shape library). When set, the analytic
    # ellipse (b0, c0) is *derived* from this template's PCA semi-axes
    # (scaled to ``r_eff``) and ``sample_phantom_surface`` extrudes the
    # template along the axis instead of a smooth ellipse. When None,
    # falls back to the analytic ellipse.
    shape_template: Optional[np.ndarray] = None
    template_source_index: int = -1       # row index in the library CSV, -1 if none
    r_eff_um: float = 0.0                 # sqrt(cross-section area / pi), um

    # ---- helpers ---------------------------------------------------------
    def gamma(self, t: np.ndarray) -> np.ndarray:
        """Axis position(s) at parameter t (scalar or array)."""
        t = np.asarray(t)
        base = self.P0[None, :] + self.T[None, :] * t[..., None]
        bend = self.curv_dir[None, :] * (self.curv_amp * 4.0 * t * (1.0 - t))[..., None]
        out = base + bend
        if np.ndim(t) == 0:
            return out[0]
        return out

    def tangent(self, t: np.ndarray) -> np.ndarray:
        """d gamma / d t."""
        t = np.asarray(t)
        tang = np.broadcast_to(self.T[None, :], t.shape + (3,)).copy()
        tang = tang + self.curv_dir[None, :] * (self.curv_amp * 4.0 * (1.0 - 2.0 * t))[..., None]
        if np.ndim(t) == 0:
            return tang[0]
        return tang

    def semiaxes(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = np.asarray(t)
        bt = self.b0 * (1.0 + (self.taper - 1.0) * t)
        ct = self.c0 * (1.0 + (self.taper - 1.0) * t)
        return bt, ct

    def twist_rad(self, t: np.ndarray) -> np.ndarray:
        return self.twist_total_rad * np.asarray(t)

    def to_serialisable(self) -> Dict[str, Any]:
        return {
            "gt_id": int(self.gt_id),
            "level": int(self.level),
            "length_um": float(self.length_um),
            "b0_um": float(self.b0),
            "c0_um": float(self.c0),
            "taper": float(self.taper),
            "twist_total_deg": float(np.rad2deg(self.twist_total_rad)),
            "curv_amp_um": float(self.curv_amp),
            "curv_category": self.curv_category,
            "volume_um3": float(self.volume_um3),
            "P0": self.P0.tolist(),
            "T": self.T.tolist(),
            "curv_dir": self.curv_dir.tolist(),
            "r_eff_um": float(self.r_eff_um),
            "template_source_index": int(self.template_source_index),
            "uses_tile046_template": bool(self.shape_template is not None),
        }


# --- 3. GEOMETRY HELPERS ---

def orthonormal_basis_perp_to(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors orthogonal to v and to each other."""
    v = v / (np.linalg.norm(v) + 1e-12)
    if abs(v[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])
    n1 = np.cross(v, tmp)
    n1 /= np.linalg.norm(n1) + 1e-12
    n2 = np.cross(v, n1)
    n2 /= np.linalg.norm(n2) + 1e-12
    return n1, n2


def rotation_matrix_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation matrix."""
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    K = np.array([[0, -ax[2], ax[1]],
                  [ax[2], 0, -ax[0]],
                  [-ax[1], ax[0], 0]])
    return np.eye(3) * c + (1 - c) * np.outer(ax, ax) + s * K


def sample_phantom_surface(phantom: Phantom, n_t: int = 300, n_u: int = 80) -> np.ndarray:
    """Dense surface point cloud of a phantom. Returns (n_t*n_u, 3) array.

    If ``phantom.shape_template`` is set, the cross-section at parameter t is
    the template polygon scaled by (r_eff * taper_factor(t)) and rotated by
    ``twist_rad(t)`` around the local axis. Otherwise the analytic ellipse
    with semi-axes (b0, c0) is used.
    """
    t_grid = np.linspace(0.0, 1.0, n_t)
    gammas = phantom.gamma(t_grid)              # (n_t, 3)
    tangents = phantom.tangent(t_grid)          # (n_t, 3)
    twists = phantom.twist_rad(t_grid)          # (n_t,)

    tpl = phantom.shape_template
    if tpl is not None:
        m = tpl.shape[0]
        pts = np.zeros((n_t * m, 3))
        r_eff = float(phantom.r_eff_um)
        taper_factors = (1.0 + (phantom.taper - 1.0) * t_grid)
        for i in range(n_t):
            s = r_eff * taper_factors[i]
            cs = tpl * s                          # (m, 2) scaled template
            n1, n2 = orthonormal_basis_perp_to(tangents[i])
            ctw, stw = np.cos(twists[i]), np.sin(twists[i])
            n1r =  n1 * ctw + n2 * stw
            n2r = -n1 * stw + n2 * ctw
            ring = (gammas[i][None, :]
                    + cs[:, 0:1] * n1r[None, :]
                    + cs[:, 1:2] * n2r[None, :])
            pts[i * m:(i + 1) * m] = ring
        return pts

    u_grid = np.linspace(0.0, 2 * np.pi, n_u, endpoint=False)
    pts = np.zeros((n_t * n_u, 3))
    bts, cts = phantom.semiaxes(t_grid)         # each (n_t,)
    cos_u = np.cos(u_grid); sin_u = np.sin(u_grid)
    for i in range(n_t):
        n1, n2 = orthonormal_basis_perp_to(tangents[i])
        ctw, stw = np.cos(twists[i]), np.sin(twists[i])
        n1r =  n1 * ctw + n2 * stw
        n2r = -n1 * stw + n2 * ctw
        ring = (gammas[i][None, :]
                + bts[i] * cos_u[:, None] * n1r[None, :]
                + cts[i] * sin_u[:, None] * n2r[None, :])
        pts[i * n_u:(i + 1) * n_u] = ring
    return pts


def phantom_volume_analytic(phantom: Phantom, n_t: int = 400) -> float:
    """Approximate swept-ellipse volume.

    When a Tile046 shape template is attached, area(t) = r_eff^2 * taper(t)^2
    (template is unit-area). Otherwise use ellipse area pi*b(t)*c(t).
    """
    ts = np.linspace(0.0, 1.0, n_t)
    tangs = phantom.tangent(ts)
    speed = np.linalg.norm(tangs, axis=1)
    if phantom.shape_template is not None:
        taper_t = (1.0 + (phantom.taper - 1.0) * ts)
        r = float(phantom.r_eff_um)
        # Template area == pi (see ``_canonical_unit_area_template``), so
        # cross-section area at parameter t is pi * r_eff^2 * taper(t)^2.
        area_t = np.pi * (r * r) * (taper_t ** 2)
    else:
        bts, cts = phantom.semiaxes(ts)
        area_t = np.pi * bts * cts
    integrand = area_t * speed
    return float(np.trapz(integrand, ts))


def hull_2d_closed(points_xy: np.ndarray) -> Optional[np.ndarray]:
    """Convex hull of 2D points, returned as a CLOSED polygon (last != first convention: returned open)."""
    if len(points_xy) < 3:
        return None
    try:
        hull = ConvexHull(points_xy)
    except QhullError:
        return None
    return points_xy[hull.vertices]


def resample_polygon_arclength(polygon_xy: np.ndarray, n_points: int) -> np.ndarray:
    """Equidistant resample of a closed polygon (vertices in order) to exactly n_points."""
    pts = np.asarray(polygon_xy, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((n_points, 2))
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-9:
        return np.tile(pts[0], (n_points, 1))
    targets = np.linspace(0.0, total, n_points, endpoint=False)
    out = np.zeros((n_points, 2))
    idx = np.searchsorted(cum, targets, side="right")
    idx = np.clip(idx, 1, len(cum) - 1)
    lo = idx - 1
    seg_len = cum[idx] - cum[lo]
    u = np.where(seg_len > 1e-12, (targets - cum[lo]) / seg_len, 0.0)
    out = pts[lo] * (1 - u)[:, None] + pts[idx] * u[:, None]
    return out


def polygon_shoelace(poly_xy: np.ndarray) -> float:
    """Signed area via shoelace; returns absolute area."""
    x = poly_xy[:, 0]; y = poly_xy[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(poly_xy: np.ndarray) -> float:
    diff = np.diff(np.vstack([poly_xy, poly_xy[:1]]), axis=0)
    return float(np.sum(np.linalg.norm(diff, axis=1)))


def polygon_second_moments(poly_xy: np.ndarray, centroid: np.ndarray) -> Tuple[float, float, float]:
    """Return (major_axis_length, minor_axis_length, orientation_angle_deg)."""
    coords = poly_xy - centroid[None, :]
    if len(coords) < 3:
        return 0.0, 0.0, 0.0
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    # Semi-axes (approx): project coords onto principal axes, take max extent.
    proj = coords @ eigvecs
    major = 2.0 * float(np.max(np.abs(proj[:, 0])))
    minor = 2.0 * float(np.max(np.abs(proj[:, 1])))
    if minor > major:
        major, minor = minor, major
    theta = float(np.rad2deg(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return major, minor, theta


def _rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate_contour_about_own_centroid(abs_xy: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a closed contour (N, 2) about its own centroid (mean of its
    points, close to the geometric centroid for evenly-sampled contours) by
    ``angle_rad`` radians. Returns a new array; input is not mutated.

    Centroid and area are preserved (rotation is a rigid isometry). The
    neighbour structure in the slice is preserved because every point moves
    by the same rotation - so OT / centroid-based matchers are unaffected.
    DTW, on the other hand, pays a cost roughly linear in |angle_rad|·N
    because the point-to-point correspondence it compares is rotated.
    """
    pts = np.asarray(abs_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or angle_rad == 0.0:
        return pts
    centre = pts.mean(axis=0)
    R = _rotation_matrix_2d(angle_rad)
    return (pts - centre) @ R.T + centre


def add_radial_contour_noise(abs_xy: np.ndarray, sigma_um: float,
                             rng: np.random.Generator) -> np.ndarray:
    """Add independent Gaussian radial noise (sigma = sigma_um) to every
    point of a closed contour (N, 2), along the local outward direction from
    the cell centroid (mean of contour points).

    This approximates 2D segmentation-edge noise: each point's radial
    position jitters by eps ~ N(0, sigma^2) but the centroid / area are
    preserved up to O(1/N) fluctuations (mean of N iid Gaussians). DTW pays
    a cost ~= sigma^2 * N per matched pair; Centroid / OT are essentially
    unaffected.
    """
    pts = np.asarray(abs_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or sigma_um <= 0.0:
        return pts
    centre = pts.mean(axis=0)
    radial = pts - centre
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    # Guard against degenerate zero-length points (shouldn't happen for a
    # real cell contour but keeps the helper defensive).
    norms = np.where(norms < 1e-9, 1e-9, norms)
    unit = radial / norms
    eps = rng.normal(0.0, float(sigma_um), size=pts.shape[0])[:, None]
    return pts + unit * eps


def canonicalize_contour_start(rel_xy: np.ndarray) -> np.ndarray:
    """Roll a closed (N, 2) contour so the starting index is at a reproducible
    anatomical anchor, and force counter-clockwise traversal.

    Why: ``scipy.ConvexHull`` returns vertices in an order that depends on
    the input point set. For the same phantom sectioned at adjacent z the
    surface-band samples differ, so ``pts[0]`` lands at a random angular
    position - linear Soft-DTW is NOT cyclically invariant and pays heavy
    warp cost for such offsets, making DTW_only look like noise even on
    the zero-perturbation sanity baseline.

    The anchor ``argmax(x + eps * y)`` is stable in the lab-frame xy for
    any cell whose orientation is preserved across slices (which is the
    case for all our conditions: ``interlayer_rot_sigma_deg = 0`` and
    ``twist_deg = 0`` by default). The ``eps * y`` term is a
    deterministic tiebreaker for cells with multiple right-extreme points.
    """
    pts = np.asarray(rel_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return pts
    x, y = pts[:, 0], pts[:, 1]
    signed_area = 0.5 * float(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )
    if signed_area < 0.0:
        pts = pts[::-1].copy()
    scores = pts[:, 0] + 1e-9 * pts[:, 1]
    start = int(np.argmax(scores))
    return np.roll(pts, -start, axis=0)


# --- 3b. TILE046 CROSS-SECTION SHAPE LIBRARY (unit-area canonical templates) ---
#
# Motivation: analytic ellipses don't look like real cardiac myocyte
# cross-sections (Tile046 shows notches, concavities, asymmetric curvature).
# The shape library replaces the ellipse with a real Tile046 contour,
# scaled to an arbitrary ``r_eff`` (effective radius) target, keeping all
# perturbation mechanisms (jitter, warp, cluster orientation) intact.
# Cell *count* is still set by Poisson-disk sampling on centroids - the
# library only replaces the *shape* per cell.

# Per-run overrides set by ``simulate_one`` / CLI (see ``set_phantom_shape_library_config``).
_CFG_SHAPE_LIBRARY_CSV: Optional[str] = None
_CFG_ELLIPSE_CROSS_SECTION: bool = False
# When set to a positive integer, ``get_tile046_unit_templates`` pre-filters
# the library to its ``K`` most shape-diverse templates (farthest-point
# selection on canonicalised contour L2 distance). This concentrates sampling
# on contours that DTW / OT can actually tell apart - matching the
# centroid-adversarial design of the current CONDITIONS catalogue.
_CFG_SHAPE_DIVERSITY_K: Optional[int] = None

# ``csv_path -> list of (NUM_CONTOUR_POINTS, 2)`` templates with shoelace area == pi.
# Scaling by r_eff then yields area = pi * r_eff^2, so ``sqrt(area/pi) == r_eff``
# (matches the conventional "effective radius" used in the calibration stats).
_TILE046_UNIT_TEMPLATES_CACHE: Dict[str, List[np.ndarray]] = {}
# (csv_path, K) -> diverse-filtered subset (list of canonicalised templates).
_TILE046_DIVERSE_TEMPLATES_CACHE: Dict[Tuple[str, int], List[np.ndarray]] = {}
# Diagnostics emitted the first time the diverse subset is built for a
# given (csv_path, K) - pairwise-distance mean/median before vs after.
_TILE046_DIVERSE_STATS: Dict[Tuple[str, int], Dict[str, float]] = {}

# Minimum original (relative) contour area (um^2) to accept a Tile046 row into the library.
_TILE046_LIB_MIN_AREA_UM2 = 80.0


def set_phantom_shape_library_config(csv_path: Optional[str],
                                     ellipse_cross_section: bool,
                                     diversity_k: Optional[int] = None) -> None:
    """Called by ``simulate_one`` / CLI so all samplers share the same
    library path, cross-section mode, and diverse-subset size ``K``.

    ``diversity_k`` semantics:
      * None / <= 0          -> use the full library (legacy behaviour).
      * positive integer K   -> pre-filter to the K most shape-diverse
                                templates via farthest-point selection on
                                pairwise L2 contour distance.
    """
    global _CFG_SHAPE_LIBRARY_CSV, _CFG_ELLIPSE_CROSS_SECTION, _CFG_SHAPE_DIVERSITY_K
    _CFG_SHAPE_LIBRARY_CSV = csv_path
    _CFG_ELLIPSE_CROSS_SECTION = bool(ellipse_cross_section)
    if diversity_k is not None and int(diversity_k) > 0:
        _CFG_SHAPE_DIVERSITY_K = int(diversity_k)
    else:
        _CFG_SHAPE_DIVERSITY_K = None


def clear_phantom_shape_library_config() -> None:
    global _CFG_SHAPE_LIBRARY_CSV, _CFG_ELLIPSE_CROSS_SECTION, _CFG_SHAPE_DIVERSITY_K
    _CFG_SHAPE_LIBRARY_CSV = None
    _CFG_ELLIPSE_CROSS_SECTION = False
    _CFG_SHAPE_DIVERSITY_K = None


def default_tile046_shape_library_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Tile046", "046_S05_features.csv")


def _row_relative_contour_from_dict(row: Dict[str, Any], n_contour: int) -> np.ndarray:
    """(n_contour, 2) relative-xy contour (same convention as CSV ``x_i, y_i``)."""
    pts = np.empty((n_contour, 2), dtype=np.float64)
    for i in range(n_contour):
        pts[i, 0] = float(row[f"x_{i}"])
        pts[i, 1] = float(row[f"y_{i}"])
    return pts


def _canonical_unit_area_template(rel_xy: np.ndarray) -> Optional[np.ndarray]:
    """Re-center at centroid, rotate PCA major axis to +x, arc-length resample,
    then scale so shoelace area == pi (i.e. "unit effective radius").

    With this normalisation, scaling the template by ``r_eff`` yields area
    ``pi * r_eff^2`` so ``sqrt(area/pi) == r_eff`` - matching the convention
    used in ``_compute_tile046_calibration``.

    Returns ``(NUM_CONTOUR_POINTS, 2)`` or ``None`` if the input is degenerate.
    """
    pts = np.asarray(rel_xy, dtype=np.float64)
    if pts.shape[0] < 3:
        return None
    c = pts.mean(axis=0)
    pts = pts - c[None, :]
    a0 = polygon_shoelace(pts)
    if a0 < _TILE046_LIB_MIN_AREA_UM2:
        return None
    _, _, theta_deg = polygon_second_moments(pts, np.zeros(2, dtype=np.float64))
    ang = -np.deg2rad(float(theta_deg))
    pts = (_rotation_matrix_2d(ang) @ pts.T).T
    pts = resample_polygon_arclength(pts, NUM_CONTOUR_POINTS)
    a1 = polygon_shoelace(pts)
    if a1 <= 1e-9:
        return None
    pts *= np.sqrt(np.pi / a1)
    pts -= pts.mean(axis=0)[None, :]
    return pts.astype(np.float64)


def _build_tile046_unit_templates(csv_path: str) -> List[np.ndarray]:
    if not os.path.isfile(csv_path):
        return []
    df = pd.read_csv(csv_path)
    out: List[np.ndarray] = []
    for _, row in df.iterrows():
        rel = _row_relative_contour_from_dict(row.to_dict(), NUM_CONTOUR_POINTS)
        tpl = _canonical_unit_area_template(rel)
        if tpl is not None:
            out.append(tpl)
    return out


# ---------------------------------------------------------------------------
# Diverse-subset selection: pick the ``K`` canonicalised templates that are
# most mutually dissimilar so DTW / OT have a real discrimination signal.
# ---------------------------------------------------------------------------

def _start_align_template(tpl: np.ndarray) -> np.ndarray:
    """Canonicalise start-point and y-polarity so pairwise L2 is meaningful.

    Templates come out of ``_canonical_unit_area_template`` already centred,
    PCA-rotated to +x major axis, arc-length resampled, and scaled to
    unit area - but the start index and the sign of the minor axis are
    still arbitrary. Here we (a) cyclically shift so index 0 is the
    rightmost vertex (the PCA major tip), and (b) flip along y if the
    second point lies below the x-axis, so every template traverses
    counterclockwise from the right tip upwards.
    """
    idx = int(np.argmax(tpl[:, 0]))
    out = np.roll(tpl, -idx, axis=0)
    if out[1, 1] < 0.0:
        out = out * np.array([1.0, -1.0])
        idx = int(np.argmax(out[:, 0]))
        out = np.roll(out, -idx, axis=0)
    return out


def _pairwise_template_distances(templates: List[np.ndarray]) -> np.ndarray:
    """Return (M, M) matrix of pairwise L2 distances between start-aligned
    flattened contours. Empty ``templates`` yields a 0x0 matrix."""
    if not templates:
        return np.zeros((0, 0), dtype=np.float64)
    aligned = np.stack([_start_align_template(t) for t in templates], axis=0)
    flat = aligned.reshape(aligned.shape[0], -1).astype(np.float64)
    sq = np.sum(flat * flat, axis=1)
    dsq = sq[:, None] + sq[None, :] - 2.0 * (flat @ flat.T)
    dsq = np.maximum(dsq, 0.0)
    np.fill_diagonal(dsq, 0.0)
    return np.sqrt(dsq)


def _farthest_point_select(dist_matrix: np.ndarray, k: int) -> List[int]:
    """Greedy k-center / farthest-point selection on a pairwise distance
    matrix. Seeded with the template that has the largest sum of distances
    to the rest (a reproducible, library-level outlier)."""
    m = int(dist_matrix.shape[0])
    if m == 0 or k <= 0:
        return []
    k = min(int(k), m)
    seed = int(np.argmax(dist_matrix.sum(axis=1)))
    selected = [seed]
    min_dist = dist_matrix[seed].copy()
    for _ in range(k - 1):
        nxt = int(np.argmax(min_dist))
        if min_dist[nxt] <= 0.0:
            break
        selected.append(nxt)
        min_dist = np.minimum(min_dist, dist_matrix[nxt])
    return selected


def _build_diverse_templates(csv_path: str, full: List[np.ndarray],
                             k: int) -> List[np.ndarray]:
    """Select the K most mutually dissimilar templates and cache diagnostics."""
    if k <= 0 or k >= len(full):
        return list(full)
    D = _pairwise_template_distances(full)
    idxs = _farthest_point_select(D, k)
    sub = [full[i] for i in idxs]
    # Diagnostics: off-diagonal mean/median before vs after selection.
    iu = np.triu_indices(len(full), k=1)
    D_sub = D[np.ix_(idxs, idxs)]
    iu_sub = np.triu_indices(len(sub), k=1)
    stats = {
        "library_size":          int(len(full)),
        "selected_k":            int(len(sub)),
        "mean_dist_full":        float(D[iu].mean()) if iu[0].size else 0.0,
        "median_dist_full":      float(np.median(D[iu])) if iu[0].size else 0.0,
        "mean_dist_selected":    float(D_sub[iu_sub].mean()) if iu_sub[0].size else 0.0,
        "median_dist_selected":  float(np.median(D_sub[iu_sub])) if iu_sub[0].size else 0.0,
        "min_pairwise_selected": float(D_sub[iu_sub].min()) if iu_sub[0].size else 0.0,
    }
    _TILE046_DIVERSE_STATS[(csv_path, int(k))] = stats
    print(f"  [shape-lib] diverse subset: K={stats['selected_k']}/{stats['library_size']} "
          f"mean-dist full={stats['mean_dist_full']:.3f} -> "
          f"selected={stats['mean_dist_selected']:.3f} "
          f"(min pair={stats['min_pairwise_selected']:.3f})")
    return sub


def get_tile046_unit_templates(csv_path: Optional[str] = None,
                               force_ellipse: Optional[bool] = None,
                               diversity_k: Optional[int] = None) -> List[np.ndarray]:
    """Return cached unit-area templates (possibly filtered to the K most
    shape-diverse), or ``[]`` for ellipse-only / missing CSV.

    Resolution order for ``diversity_k``:
      explicit argument -> module ``_CFG_SHAPE_DIVERSITY_K`` -> None (full).
    ``K <= 0`` or ``K >= library size`` means "use the full library".
    """
    fe = _CFG_ELLIPSE_CROSS_SECTION if force_ellipse is None else bool(force_ellipse)
    if fe:
        return []
    path = csv_path or _CFG_SHAPE_LIBRARY_CSV or default_tile046_shape_library_path()
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return []
    if path not in _TILE046_UNIT_TEMPLATES_CACHE:
        _TILE046_UNIT_TEMPLATES_CACHE[path] = _build_tile046_unit_templates(path)
    full = _TILE046_UNIT_TEMPLATES_CACHE[path]
    k_eff = diversity_k if diversity_k is not None else _CFG_SHAPE_DIVERSITY_K
    if k_eff is None or int(k_eff) <= 0 or int(k_eff) >= len(full):
        return full
    key = (path, int(k_eff))
    if key not in _TILE046_DIVERSE_TEMPLATES_CACHE:
        _TILE046_DIVERSE_TEMPLATES_CACHE[key] = _build_diverse_templates(
            path, full, int(k_eff))
    return _TILE046_DIVERSE_TEMPLATES_CACHE[key]


# --- 4. TISSUE SAMPLING ---

def poisson_disk_2d(min_dist: float, box_x: float, box_y: float,
                    margin: float, rng: np.random.Generator,
                    max_attempts: int = 4000) -> np.ndarray:
    """Simple Poisson-like rejection sampling in 2D (not Bridson but fast enough)."""
    pts: List[np.ndarray] = []
    attempts = 0
    while attempts < max_attempts:
        p = np.array([rng.uniform(margin, box_x - margin),
                      rng.uniform(margin, box_y - margin)])
        ok = True
        for q in pts:
            if np.linalg.norm(p - q) < min_dist:
                ok = False
                break
        if ok:
            pts.append(p)
            attempts = 0
        else:
            attempts += 1
    return np.array(pts) if pts else np.zeros((0, 2))


def poisson_disk_clustered(cluster_center_dist: float,
                           intra_cluster_dist: float,
                           cluster_size_range: Tuple[int, int],
                           box_x: float, box_y: float,
                           margin: float, rng: np.random.Generator,
                           max_cluster_attempts: int = 2000,
                           max_member_attempts: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """Two-level placement: first scatter cluster centres via Poisson-disk,
    then place 3-6 cells around each centre with a shared local orientation.

    Returns
    -------
    xy : (N, 2) ndarray of cell centroids in the box
    cluster_id : (N,) int ndarray marking which cluster each cell belongs to.
    Cells in the same cluster will later share a near-identical long-axis
    direction (small random wobble), so their centroids pack tightly but
    shape / OT neighbour set remain the discriminative signal.
    """
    centers = poisson_disk_2d(min_dist=cluster_center_dist,
                              box_x=box_x, box_y=box_y,
                              margin=margin, rng=rng,
                              max_attempts=max_cluster_attempts)
    xy_list: List[np.ndarray] = []
    cluster_ids: List[int] = []
    # Cluster disk radius scales with target member count so tight clusters
    # (small ``intra_cluster_dist``) with many members (6-10) still have room.
    size_lo_g, size_hi_g = cluster_size_range
    target_n = 0.5 * (int(size_lo_g) + int(size_hi_g))
    radius = max(intra_cluster_dist * 1.8,
                 float(np.sqrt(max(target_n, 1.0))) * float(intra_cluster_dist))
    for cid, (cx, cy) in enumerate(centers):
        size_lo = max(1, int(size_lo_g))
        size_hi = max(size_lo, int(size_hi_g))
        n_members = int(rng.integers(size_lo, size_hi + 1))
        placed: List[np.ndarray] = []
        tries = 0
        while len(placed) < n_members and tries < max_member_attempts:
            # Sample within a disk of radius `radius` around (cx, cy),
            # clipped to the box, with intra-cluster min distance.
            r = float(np.sqrt(rng.uniform(0.0, 1.0)) * radius)
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            px = float(cx + r * np.cos(phi))
            py = float(cy + r * np.sin(phi))
            if px < margin or px > box_x - margin or py < margin or py > box_y - margin:
                tries += 1
                continue
            p = np.array([px, py])
            ok_intra = all(np.linalg.norm(p - q) >= intra_cluster_dist for q in placed)
            ok_cross = all(np.linalg.norm(p - q) >= intra_cluster_dist * 0.8 for q in xy_list)
            if ok_intra and ok_cross:
                placed.append(p)
                tries = 0
            else:
                tries += 1
        for p in placed:
            xy_list.append(p)
            cluster_ids.append(cid)
    if not xy_list:
        return np.zeros((0, 2)), np.zeros((0,), dtype=int)
    return np.array(xy_list), np.array(cluster_ids, dtype=int)


def _sample_tile046_template(rng: np.random.Generator,
                             aspect_range: Optional[Tuple[float, float]] = None,
                             ) -> Tuple[Optional[np.ndarray], int]:
    """Draw a random unit-area template. Optionally filter by ``max_extent / min_extent``.

    Returns ``(template, library_index)``. ``template`` is None when the
    library is empty (ellipse fallback) or ``ellipse_cross_section`` is True.
    """
    templates = get_tile046_unit_templates()
    if not templates:
        return None, -1
    if aspect_range is None:
        idx = int(rng.integers(0, len(templates)))
        return templates[idx], idx
    lo, hi = aspect_range
    # PCA-axis aligned + unit area, so max|x|/max|y| ~ true aspect ratio.
    filtered: List[Tuple[int, np.ndarray]] = []
    for i, tpl in enumerate(templates):
        ex = float(np.max(np.abs(tpl[:, 0])))
        ey = float(np.max(np.abs(tpl[:, 1]))) + 1e-9
        ar = ex / ey
        if lo <= ar <= hi:
            filtered.append((i, tpl))
    if not filtered:
        idx = int(rng.integers(0, len(templates)))
        return templates[idx], idx
    k = int(rng.integers(0, len(filtered)))
    return filtered[k][1], filtered[k][0]


def _semiaxes_from_template(tpl: np.ndarray, r_eff: float) -> Tuple[float, float]:
    """Return (b0, c0) as PCA extents of tpl * r_eff; b0 >= c0."""
    pts = tpl * float(r_eff)
    ex = float(np.max(np.abs(pts[:, 0])))
    ey = float(np.max(np.abs(pts[:, 1])))
    b0, c0 = (ex, ey) if ex >= ey else (ey, ex)
    return b0, c0


def sample_phantom_parameters(level: int, cx: float, cy: float,
                              taper: float, twist_deg: float,
                              curv_category: str,
                              rng: np.random.Generator,
                              disarray_max_deg: float = 12.0,
                              base_direction: Optional[np.ndarray] = None,
                              direction_wobble_deg: float = 0.0) -> Phantom:
    """Sample a single phantom's geometric parameters.

    If ``base_direction`` (a unit 3-vector) is provided, the cell's long axis
    is drawn from a small cone of half-angle ``direction_wobble_deg`` around it
    (used by clustered placement so cells in a cluster share orientation).
    Otherwise the axis deviates from +z by up to ``disarray_max_deg``.

    Cross-section geometry:
      * If the Tile046 shape library is loaded, sample a unit-area template
        (filtered to realistic aspect ratios) and scale it by a target
        effective radius ``r_eff`` in [7, 10] um. ``b0, c0`` are derived
        from the scaled template's PCA extents.
      * Otherwise fall back to the analytic-ellipse parameter draw.
    """
    # Length chosen conservatively so that with a small tilt the full cell still
    # fits inside the z=[0, BOX_Z_UM] slab with a small margin.
    L = float(rng.uniform(75.0, 95.0))

    # Default (ellipse fallback) semi-axes - calibrated to Tile046 single-cell stats.
    b0 = float(rng.uniform(9.0, 13.0))
    c0 = float(rng.uniform(5.5, 8.0))
    r_eff = float(np.sqrt(b0 * c0))  # geometric-mean-style r_eff for ellipse fallback
    shape_tpl, tpl_idx = _sample_tile046_template(rng, aspect_range=(1.25, 2.4))
    if shape_tpl is not None:
        # Sample r_eff to match Tile046-calibrated ellipse sizing range.
        # With b0 in [9, 13] and c0 in [5.5, 8], sqrt(b0*c0) in ~[7, 10].
        r_eff = float(rng.uniform(7.0, 10.0))
        b0, c0 = _semiaxes_from_template(shape_tpl, r_eff)

    if base_direction is not None:
        base = np.asarray(base_direction, dtype=float)
        base = base / (np.linalg.norm(base) + 1e-12)
        wobble_rad = np.deg2rad(float(rng.uniform(0.0, max(0.0, direction_wobble_deg))))
        n1, n2 = orthonormal_basis_perp_to(base)
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        direction = (base * np.cos(wobble_rad)
                     + (np.cos(phi) * n1 + np.sin(phi) * n2) * np.sin(wobble_rad))
        direction = direction / (np.linalg.norm(direction) + 1e-12)
    else:
        # Per-cell deviation of long axis from +z. Larger disarray_max_deg = more disordered tissue.
        tilt_from_z = np.deg2rad(float(rng.uniform(0.0, disarray_max_deg)))
        tilt_az = float(rng.uniform(0.0, 2.0 * np.pi))
        direction = np.array([
            np.sin(tilt_from_z) * np.cos(tilt_az),
            np.sin(tilt_from_z) * np.sin(tilt_az),
            np.cos(tilt_from_z),
        ])
    # Effective z-extent takes tilt into account (projected length along z).
    half_L = L / 2.0
    z_half_span = abs(direction[2]) * half_L
    margin = 3.0
    lo = z_half_span + margin
    hi = BOX_Z_UM - z_half_span - margin
    if hi <= lo:
        z_center = BOX_Z_UM / 2.0
    else:
        z_center = float(rng.uniform(lo, hi))
    P0 = np.array([cx, cy, z_center - half_L])
    T = direction * L
    # Perpendicular direction for bending (random in-plane rotation around T)
    n1, _ = orthonormal_basis_perp_to(direction)
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    curv_dir = np.cos(theta) * n1 + np.sin(theta) * np.cross(direction, n1)
    curv_dir /= np.linalg.norm(curv_dir) + 1e-12

    curv_amp = CURV_AMPLITUDE[curv_category] if level >= 2 else 0.0
    twist_rad = np.deg2rad(twist_deg) if level >= 2 else 0.0
    taper_eff = taper if level >= 2 else 1.0

    ph = Phantom(
        gt_id=-1,  # assigned later
        level=level,
        length_um=L,
        b0=b0, c0=c0,
        taper=taper_eff,
        twist_total_rad=twist_rad,
        P0=P0, T=T,
        curv_dir=curv_dir, curv_amp=curv_amp,
        curv_category=curv_category,
        shape_template=shape_tpl,
        template_source_index=tpl_idx,
        r_eff_um=float(r_eff),
    )
    ph.volume_um3 = phantom_volume_analytic(ph)
    return ph


def sample_myocyte_like_distractor(cx: float, cy: float, rng: np.random.Generator) -> Phantom:
    """Sample a harder, myocyte-like distractor cell.

    Length 30-60 um, elliptical cross-section (b0/c0 ratio 1.3-2.0), long axis
    intentionally off the +z main myocyte field by 30-50 degrees, without the
    coherent OT neighbour set of real myocytes. Centroid-NN cannot easily
    reject these because they share many single-cell statistics with real
    myocytes; only shape and neighbourhood context distinguish them.
    """
    L = float(rng.uniform(30.0, 60.0))
    short_r = float(rng.uniform(4.0, 7.0))
    aspect = float(rng.uniform(1.3, 2.0))
    long_r = short_r * aspect
    b0, c0 = long_r, short_r
    # Tile046 template with high aspect ratio (myocyte-like shape).
    shape_tpl, tpl_idx = _sample_tile046_template(rng, aspect_range=(1.3, 2.5))
    r_eff = float(np.sqrt(max(b0 * c0, 1e-6)))
    if shape_tpl is not None:
        b0, c0 = _semiaxes_from_template(shape_tpl, r_eff)
    off_axis_deg = float(rng.uniform(30.0, 50.0))
    off_axis_rad = np.deg2rad(off_axis_deg)
    azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = np.array([
        np.sin(off_axis_rad) * np.cos(azimuth),
        np.sin(off_axis_rad) * np.sin(azimuth),
        np.cos(off_axis_rad),
    ])
    direction /= np.linalg.norm(direction) + 1e-12
    half_L = L / 2.0
    z_half_span = abs(direction[2]) * half_L
    margin = 3.0
    lo = z_half_span + margin
    hi = BOX_Z_UM - z_half_span - margin
    z_center = float(rng.uniform(lo, hi)) if hi > lo else BOX_Z_UM / 2.0
    P0 = np.array([cx, cy, z_center - half_L])
    T = direction * L
    n1, _ = orthonormal_basis_perp_to(direction)
    curv_dir = n1
    ph = Phantom(
        gt_id=-1,
        level=0,
        length_um=L,
        b0=b0, c0=c0,
        taper=float(rng.uniform(0.85, 1.0)),
        twist_total_rad=0.0,
        P0=P0, T=T,
        curv_dir=curv_dir, curv_amp=0.0,
        curv_category="distractor_myocyte_like",
        shape_template=shape_tpl,
        template_source_index=tpl_idx,
        r_eff_um=float(r_eff),
    )
    ph.volume_um3 = phantom_volume_analytic(ph)
    return ph


def sample_distractor_parameters(cx: float, cy: float, rng: np.random.Generator) -> Phantom:
    """Sample a small, round, short non-myocyte distractor (e.g. fibroblast-like blob).

    Distractors are modelled as short ellipsoids (length 12-25 um, radius 3-6 um)
    with arbitrary orientation in 3D (not biased toward +z). They fill the role
    of clutter/noise in the tissue environment: HiDTW should reject matches
    between myocytes and distractors thanks to DTW shape + OT neighbour cluster.
    """
    L = float(rng.uniform(12.0, 25.0))
    r = float(rng.uniform(3.0, 6.0))
    b0 = r
    c0 = r * float(rng.uniform(0.7, 1.1))
    # For blob distractors, sample a low-aspect Tile046 template (near-round).
    shape_tpl, tpl_idx = _sample_tile046_template(rng, aspect_range=(1.0, 1.35))
    r_eff = float(np.sqrt(max(b0 * c0, 1e-6)))
    if shape_tpl is not None:
        b0, c0 = _semiaxes_from_template(shape_tpl, r_eff)
    # Fully random 3D direction (spherically uniform)
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    cos_theta = float(rng.uniform(-1.0, 1.0))
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    direction = np.array([sin_theta * np.cos(phi),
                          sin_theta * np.sin(phi),
                          cos_theta])
    z_half = abs(direction[2]) * L / 2.0
    z_center = float(rng.uniform(z_half + 2.0, BOX_Z_UM - z_half - 2.0)) \
        if BOX_Z_UM - 2.0 * z_half - 4.0 > 0 else BOX_Z_UM / 2.0
    P0 = np.array([cx, cy, z_center - L / 2.0])
    T = direction * L
    n1, _ = orthonormal_basis_perp_to(direction)
    curv_dir = n1
    ph = Phantom(
        gt_id=-1,
        level=0,       # 0 marks distractor
        length_um=L,
        b0=b0, c0=c0,
        taper=1.0,
        twist_total_rad=0.0,
        P0=P0, T=T,
        curv_dir=curv_dir, curv_amp=0.0,
        curv_category="distractor",
        shape_template=shape_tpl,
        template_source_index=tpl_idx,
        r_eff_um=float(r_eff),
    )
    ph.volume_um3 = phantom_volume_analytic(ph)
    return ph


def _ellipse_unit_area_contour(rng: np.random.Generator,
                               aspect_range: Tuple[float, float],
                               n_pts: int = NUM_CONTOUR_POINTS) -> np.ndarray:
    """Analytic ellipse normalised to area == pi (i.e. "unit effective radius"),
    with aspect randomly drawn from ``aspect_range``. Used as fallback when the
    Tile046 shape library is not available (ellipse-cross-section mode)."""
    lo, hi = float(aspect_range[0]), float(aspect_range[1])
    ar = float(rng.uniform(lo, hi))
    # Major semi = sqrt(ar), minor semi = 1/sqrt(ar) => a*b = 1 => area == pi.
    a = float(np.sqrt(ar))
    b = 1.0 / a
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    pts = np.stack([a * np.cos(theta), b * np.sin(theta)], axis=1)
    return pts


def sample_per_slice_distractor_contour(
        rng: np.random.Generator,
        existing_centroids_xy: List[Tuple[float, float]],
        area_range: Tuple[float, float],
        aspect_range: Tuple[float, float],
        min_dist_to_myocyte: float,
        box_x: float = BOX_X_UM,
        box_y: float = BOX_Y_UM,
        margin: float = 15.0,
        max_place_tries: int = 300,
        ) -> Optional[np.ndarray]:
    """Generate ONE 2D fibroblast-like distractor contour for a single slice.

    Returns an absolute-xy ``(NUM_CONTOUR_POINTS, 2)`` contour, already
    resampled to the library point count and placed at a random centroid
    at least ``min_dist_to_myocyte`` um away from every existing myocyte
    centroid. Returns ``None`` if no valid location found within
    ``max_place_tries`` attempts (the caller should just skip / retry).

    The contour has:
      * area uniformly drawn from ``area_range`` (um^2);
      * aspect ratio uniformly drawn from ``aspect_range``;
      * shape drawn from the low-aspect Tile046 shape library when
        available, else from a matching unit-area ellipse;
      * random in-plane rotation in ``[0, 2 pi)``.

    Crucially the output has NO cross-slice identity: each call is
    independent RNG draws. It is NOT a ``Phantom`` and bypasses the
    surface-sampling / ``extract_slice_contour`` pipeline.
    """
    lo_a, hi_a = float(area_range[0]), float(area_range[1])
    if hi_a < lo_a:
        lo_a, hi_a = hi_a, lo_a
    lo_ar, hi_ar = float(aspect_range[0]), float(aspect_range[1])
    if hi_ar < lo_ar:
        lo_ar, hi_ar = hi_ar, lo_ar

    tpl, _ = _sample_tile046_template(
        rng, aspect_range=(max(1.0, lo_ar * 0.9), max(1.0 + 1e-6, hi_ar * 1.1)))
    if tpl is None:
        tpl = _ellipse_unit_area_contour(rng, (lo_ar, hi_ar))
    target_area = float(rng.uniform(lo_a, hi_a))
    scale = float(np.sqrt(max(target_area, 1e-6) / np.pi))
    shape = tpl * scale
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    shape = shape @ _rotation_matrix_2d(theta).T
    shape = shape - shape.mean(axis=0, keepdims=True)

    placed_cx = placed_cy = None
    for _ in range(int(max_place_tries)):
        cx = float(rng.uniform(margin, box_x - margin))
        cy = float(rng.uniform(margin, box_y - margin))
        ok = True
        for mx, my in existing_centroids_xy:
            if (cx - mx) ** 2 + (cy - my) ** 2 < float(min_dist_to_myocyte) ** 2:
                ok = False
                break
        if ok:
            placed_cx, placed_cy = cx, cy
            break
    if placed_cx is None:
        return None
    return shape + np.array([[placed_cx, placed_cy]])


# --- 5. VIRTUAL SECTIONING ---

def slice_z_targets(slice_tilt_deg: float, jitter_on: bool,
                    rng: np.random.Generator) -> List[float]:
    """Return per-slice nominal z values (mid-plane). Tilt is handled separately on surface points."""
    zs = [(k + 0.5) * SLICE_THICKNESS_UM for k in range(N_SLICES)]
    if jitter_on:
        zs = [z + float(rng.normal(0.0, 0.8)) for z in zs]
    return zs


def apply_slice_plane_transform(points_xyz: np.ndarray, slice_tilt_rad: float) -> np.ndarray:
    """Rotate surface points by -tilt around y-axis so that tilted slicing planes map to horizontal planes."""
    if abs(slice_tilt_rad) < 1e-9:
        return points_xyz
    R = rotation_matrix_axis(np.array([0.0, 1.0, 0.0]), -slice_tilt_rad)
    return points_xyz @ R.T


def extract_slice_contour(surface_pts_xyz: np.ndarray, z_target: float,
                          band_um: float = 1.5, n_out: int = NUM_CONTOUR_POINTS,
                          min_area: float = MIN_SLICE_AREA_UM2) -> Optional[np.ndarray]:
    """Select surface points near z_target and build a CLOSED 2D contour with n_out equidistant points."""
    z = surface_pts_xyz[:, 2]
    mask = np.abs(z - z_target) < band_um
    band = surface_pts_xyz[mask]
    if len(band) < 8:
        # Widen band once if too few points (handles very thin slices near cell tips)
        mask = np.abs(z - z_target) < band_um * 2.5
        band = surface_pts_xyz[mask]
        if len(band) < 8:
            return None
    xy = band[:, :2]
    hull_verts = hull_2d_closed(xy)
    if hull_verts is None:
        return None
    # Arrange polygon vertices CCW (ConvexHull already does)
    contour = resample_polygon_arclength(hull_verts, n_out)
    area = polygon_shoelace(contour)
    if area < min_area:
        return None
    return contour


def add_interlayer_shift(contour_xy: np.ndarray, shift_xy: np.ndarray) -> np.ndarray:
    return contour_xy + shift_xy[None, :]


def rigid_xy_transform(contour_xy: np.ndarray, shift_xy: np.ndarray,
                       theta_rad: float,
                       centre: Tuple[float, float] = (BOX_X_UM / 2.0,
                                                      BOX_Y_UM / 2.0)
                       ) -> np.ndarray:
    """Rotate ``contour_xy`` around ``centre`` by ``theta_rad`` then translate by ``shift_xy``.

    Used to model rigid inter-slice registration error (translation + small
    rotation). Identity when both components are zero.
    """
    if abs(theta_rad) < 1e-12 and not np.any(shift_xy):
        return contour_xy + shift_xy[None, :]
    c = np.asarray(centre, dtype=float)
    ct, st = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    R = np.array([[ct, -st], [st, ct]], dtype=float)
    out = (contour_xy - c[None, :]) @ R.T + c[None, :]
    out = out + shift_xy[None, :]
    return out


def _gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-padded 1D Gaussian smoothing. Returns array of the same length.

    sigma is in the same units as the sample step (here: slices).
    """
    if sigma <= 0.0 or x.size <= 1:
        return np.asarray(x, dtype=float).copy()
    radius = max(1, int(np.ceil(4.0 * sigma)))
    k = np.arange(-radius, radius + 1, dtype=float)
    kern = np.exp(-0.5 * (k / sigma) ** 2)
    kern = kern / float(kern.sum())
    pad = np.concatenate([
        np.full(radius, float(x[0])),
        np.asarray(x, dtype=float),
        np.full(radius, float(x[-1])),
    ])
    out = np.convolve(pad, kern, mode="valid")
    return out[: x.size]


def build_z_coherent_warp_fields(amp_um: float, sigma_um: float, n_bumps: int,
                                  tau_slices: float, n_slices: int,
                                  box_x: float, box_y: float,
                                  rng: np.random.Generator
                                  ) -> List[List[Dict[str, Any]]]:
    """Build ``n_slices`` warp fields that share bump centres/directions but
    have smoothly varying amplitudes along z.

    Each bump ``i`` carries a 1D amplitude sequence ``a_i[k]`` obtained by
    Gaussian-smoothing white noise with width ``tau_slices`` and rescaling so
    ``max_k |a_i[k]| = amp_um`` (preserving sign). As a result:

    - Spatial warp pattern is identical across adjacent z-slices;
    - Only intensity drifts along z with correlation length ~ ``tau_slices``.

    Returns a list of length ``n_slices``; each element is a list of bump
    dicts compatible with :func:`apply_warp_to_xy`.
    """
    empty: List[List[Dict[str, Any]]] = [[] for _ in range(n_slices)]
    if amp_um <= 0.0 or n_bumps <= 0 or n_slices <= 0:
        return empty

    bump_templates: List[Dict[str, Any]] = []
    amp_sequences: List[np.ndarray] = []
    for _ in range(int(n_bumps)):
        cx = float(rng.uniform(0.1 * box_x, 0.9 * box_x))
        cy = float(rng.uniform(0.1 * box_y, 0.9 * box_y))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        dx = float(np.cos(theta))
        dy = float(np.sin(theta))
        bump_templates.append({"cx": cx, "cy": cy, "sigma": float(sigma_um),
                               "dx": dx, "dy": dy})
        seq_raw = rng.normal(0.0, 1.0, size=n_slices)
        seq_sm = _gaussian_smooth_1d(seq_raw, float(tau_slices))
        peak = float(np.max(np.abs(seq_sm)))
        if peak > 1e-9:
            seq_sm = seq_sm / peak
        # Random global sign per bump (~50/50) so net field is zero-mean.
        if rng.random() < 0.5:
            seq_sm = -seq_sm
        amp_sequences.append(seq_sm * float(amp_um))

    fields: List[List[Dict[str, Any]]] = []
    for k in range(n_slices):
        per_slice: List[Dict[str, Any]] = []
        for i, tpl in enumerate(bump_templates):
            per_slice.append({
                "cx": tpl["cx"], "cy": tpl["cy"], "sigma": tpl["sigma"],
                "amp": float(amp_sequences[i][k]),
                "dx": tpl["dx"], "dy": tpl["dy"],
            })
        fields.append(per_slice)
    return fields


def build_warp_field(amp_um: float, sigma_um: float, n_bumps: int,
                     box_x: float, box_y: float,
                     rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Build one warp field (a list of Gaussian bumps) for a single slice.

    The field is
        D(x, y) = sum_i A_i * exp(-||(x,y) - c_i||^2 / (2*sigma^2)) * dir_i
    where ``dir_i`` is a random 2D unit vector, ``A_i`` is a random amplitude
    in [amp/2, amp], and ``c_i`` is a random center inside the box.
    """
    field: List[Dict[str, Any]] = []
    if amp_um <= 0.0 or n_bumps <= 0:
        return field
    for _ in range(int(n_bumps)):
        cx = float(rng.uniform(0.1 * box_x, 0.9 * box_x))
        cy = float(rng.uniform(0.1 * box_y, 0.9 * box_y))
        amp = float(rng.uniform(0.5 * amp_um, amp_um))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        dx = float(np.cos(theta))
        dy = float(np.sin(theta))
        # 50/50 chance of flipping amplitude to give zero-mean field overall.
        if rng.random() < 0.5:
            amp = -amp
        field.append({"cx": cx, "cy": cy, "sigma": sigma_um,
                      "amp": amp, "dx": dx, "dy": dy})
    return field


def apply_warp_to_xy(points_xy: np.ndarray, field: List[Dict[str, Any]]) -> np.ndarray:
    """Apply a warp field (list of Gaussian bumps) to (N, 2) points in absolute xy.

    Returns a new array; input is not modified. Empty field = identity.
    """
    if not field:
        return points_xy
    pts = np.asarray(points_xy, dtype=float)
    dxy = np.zeros_like(pts)
    for bump in field:
        dx = pts[:, 0] - bump["cx"]
        dy = pts[:, 1] - bump["cy"]
        sigma = max(float(bump["sigma"]), 1e-6)
        weight = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        dxy[:, 0] += weight * bump["amp"] * bump["dx"]
        dxy[:, 1] += weight * bump["amp"] * bump["dy"]
    return pts + dxy


# --- 6. TILE046-SCHEMA CSV EXPORT ---

# Exact column order copied from Tile046/046_S01_features.csv.
FIXED_COLS_HEAD = [
    "cell_id", "cell_label", "region", "slice", "z",
    "centroid_x", "centroid_y",
    "area", "perimeter", "aspect_ratio",
    "major_axis", "minor_axis", "orientation_angle",
    "circularity",
]
COORD_COLS = [f"{prefix}_{i}" for i in range(NUM_CONTOUR_POINTS) for prefix in ("x", "y")]  # x_0, y_0, x_1, y_1, ...
FIXED_COLS_TAIL = ["adjacent_cells", "num_adjacent_cells"]


def _feature_row_from_contour(absolute_contour: np.ndarray, *,
                              cell_id: str, cell_label: int,
                              region: str, slice_str: str, z_value: int) -> Dict[str, Any]:
    """Build one row worth of features matching the Tile046 schema.

    ``absolute_contour`` is the ABSOLUTE 2D contour (N_pts, 2); the stored
    ``x_i, y_i`` are RELATIVE to the centroid (matching Tile046 convention).
    """
    centroid = np.mean(absolute_contour, axis=0)
    # Re-centre so centroid matches the polygon's centroid via shoelace for robustness.
    # (For a closed uniformly-sampled polygon the mean is typically close to the centroid.)
    area = polygon_shoelace(absolute_contour)
    perim = polygon_perimeter(absolute_contour)
    major, minor, theta_deg = polygon_second_moments(absolute_contour, centroid)
    aspect = major / minor if minor > 1e-6 else 1.0
    circularity = 4.0 * np.pi * area / (perim * perim) if perim > 1e-6 else 0.0
    relative = absolute_contour - centroid[None, :]
    # Canonicalise contour start index + CCW direction so Soft-DTW (which is
    # not rotation-invariant around a closed curve) receives consistent
    # sequences across slices. See ``canonicalize_contour_start`` docstring.
    relative = canonicalize_contour_start(relative)

    row: Dict[str, Any] = {
        "cell_id": cell_id,
        "cell_label": int(cell_label),
        "region": region,
        "slice": slice_str,
        "z": int(z_value),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "area": float(area),
        "perimeter": float(perim),
        "aspect_ratio": float(aspect),
        "major_axis": float(major),
        "minor_axis": float(minor),
        "orientation_angle": float(theta_deg),
        "circularity": float(circularity),
    }
    for i in range(NUM_CONTOUR_POINTS):
        row[f"x_{i}"] = float(relative[i, 0])
        row[f"y_{i}"] = float(relative[i, 1])
    return row


def _fill_adjacency(rows: List[Dict[str, Any]], threshold_um: float = NEIGHBOR_DIST_UM) -> None:
    """In-place fill of ``adjacent_cells`` / ``num_adjacent_cells`` within one slice."""
    n = len(rows)
    cents = np.array([[r["centroid_x"], r["centroid_y"]] for r in rows])
    ids = [r["cell_id"] for r in rows]
    for i in range(n):
        neighbors = []
        for j in range(n):
            if i == j:
                continue
            d = float(np.linalg.norm(cents[i] - cents[j]))
            if d < threshold_um:
                neighbors.append(ids[j])
        rows[i]["adjacent_cells"] = ";".join(neighbors)
        rows[i]["num_adjacent_cells"] = len(neighbors)


def _all_columns() -> List[str]:
    return FIXED_COLS_HEAD + COORD_COLS + FIXED_COLS_TAIL


# --- 6b. INSTRUMENTATION: overlap + Tile046 calibration stats ---

# Tile046 S05 measured reference (from ``03_demo/Tile046/046_S05_features.csv``).
# Used as the "real tissue" calibration target for `_compute_tile046_calibration`.
TILE046_REFERENCE: Dict[str, float] = {
    "median_nn_um": 59.0,
    "median_r_eff_um": 41.0,
    "ratio_nn_over_r_eff": 1.44,
    "coverage_fraction": 0.81,
    "median_aspect_ratio": 1.62,
}


def _row_to_absolute_contour(row: Dict[str, Any], n_contour: int) -> np.ndarray:
    """Reconstruct (n_contour, 2) absolute-xy contour from a feature row."""
    cx = float(row["centroid_x"])
    cy = float(row["centroid_y"])
    pts = np.empty((n_contour, 2), dtype=float)
    for i in range(n_contour):
        pts[i, 0] = float(row[f"x_{i}"]) + cx
        pts[i, 1] = float(row[f"y_{i}"]) + cy
    return pts


def _row_bounding_radius(row: Dict[str, Any], n_contour: int) -> float:
    """Max |relative contour point| - radius of bounding disk around centroid."""
    r2 = 0.0
    for i in range(n_contour):
        dx = float(row[f"x_{i}"])
        dy = float(row[f"y_{i}"])
        d2 = dx * dx + dy * dy
        if d2 > r2:
            r2 = d2
    return float(np.sqrt(r2))


def _compute_overlap_stats(per_slice_rows: List[List[Dict[str, Any]]],
                           n_contour: int = NUM_CONTOUR_POINTS,
                           ) -> List[Dict[str, Any]]:
    """Per-slice pair-wise contour overlap statistics.

    Strategy:
      1. Build a bounding-disk radius per cell from the relative contour points.
      2. Centroid-distance pre-filter: pairs whose centers are further apart
         than ``r_a + r_b`` cannot overlap - skip shapely test.
      3. For surviving pairs, build a shapely Polygon and test ``intersects``.

    Returns a list of dicts, one per slice.
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:  # pragma: no cover - shapely is a required dep
        return []

    out: List[Dict[str, Any]] = []
    for s_idx, rows in enumerate(per_slice_rows):
        n = len(rows)
        if n < 2:
            out.append({
                "slice": s_idx + 1,
                "n_cells": n,
                "n_candidate_pairs": 0,
                "n_overlap_pairs": 0,
                "overlap_fraction": 0.0,
            })
            continue
        cents = np.array([[float(r["centroid_x"]), float(r["centroid_y"])]
                          for r in rows])
        radii = np.array([_row_bounding_radius(r, n_contour) for r in rows])
        # Pairwise centroid distances (fast vectorised).
        diff = cents[:, None, :] - cents[None, :, :]
        dmat = np.sqrt(np.sum(diff * diff, axis=-1))
        # Candidate pairs: distance <= r_i + r_j  (upper triangle only).
        sum_r = radii[:, None] + radii[None, :]
        candidate = np.triu(dmat <= sum_r, k=1)
        ii, jj = np.where(candidate)
        n_candidates = int(len(ii))
        if n_candidates == 0:
            out.append({
                "slice": s_idx + 1,
                "n_cells": n,
                "n_candidate_pairs": 0,
                "n_overlap_pairs": 0,
                "overlap_fraction": 0.0,
            })
            continue
        # Lazy-build polygons only for cells that appear in at least one candidate pair.
        unique_idx = np.unique(np.concatenate([ii, jj]))
        polys: Dict[int, Any] = {}
        for k in unique_idx:
            try:
                p = Polygon(_row_to_absolute_contour(rows[int(k)], n_contour))
                if not p.is_valid:
                    p = p.buffer(0)
                polys[int(k)] = p
            except Exception:
                polys[int(k)] = None
        n_overlap = 0
        overlapping_cells: set = set()
        for i, j in zip(ii, jj):
            pi = polys.get(int(i))
            pj = polys.get(int(j))
            if pi is None or pj is None:
                continue
            try:
                if pi.intersects(pj) and not pi.touches(pj):
                    n_overlap += 1
                    overlapping_cells.add(int(i))
                    overlapping_cells.add(int(j))
            except Exception:
                continue
        total_pairs = n * (n - 1) // 2
        out.append({
            "slice": s_idx + 1,
            "n_cells": n,
            "n_candidate_pairs": n_candidates,
            "n_overlap_pairs": int(n_overlap),
            # Fraction of close (bbox-overlapping) pairs that actually overlap.
            "overlap_fraction": (n_overlap / n_candidates) if n_candidates else 0.0,
            # Fraction of all cell-pairs that overlap - tiny in dense sims.
            "overlap_pair_rate_all": (n_overlap / total_pairs) if total_pairs else 0.0,
            # Fraction of cells that overlap with at least one neighbour -
            # the most intuitive "how many cells look overlapping" measure.
            "overlap_cell_rate": (len(overlapping_cells) / n) if n else 0.0,
        })
    return out


def _compute_tile046_calibration(per_slice_rows: List[List[Dict[str, Any]]],
                                 box_x: float, box_y: float,
                                 ) -> List[Dict[str, Any]]:
    """Per-slice calibration stats comparing against Tile046 S05.

    For each slice we compute:
      * median nearest-neighbour centroid distance (um)
      * median effective radius r_eff = sqrt(area/pi) (um)
      * NN / r_eff ratio (Tile046 target ~ 1.44)
      * total cell-area coverage fraction (Tile046 target ~ 0.81)
      * median aspect ratio (Tile046 target ~ 1.62)
    """
    try:
        from scipy.spatial import KDTree
    except ImportError:  # pragma: no cover - scipy is a required dep
        return []

    box_area = float(box_x * box_y)
    out: List[Dict[str, Any]] = []
    for s_idx, rows in enumerate(per_slice_rows):
        n = len(rows)
        if n < 2:
            out.append({
                "slice": s_idx + 1,
                "n_cells": n,
                "median_nn_um": 0.0,
                "median_r_eff_um": 0.0,
                "ratio_nn_over_r_eff": 0.0,
                "coverage_fraction": 0.0,
                "median_aspect_ratio": 0.0,
            })
            continue
        cents = np.array([[float(r["centroid_x"]), float(r["centroid_y"])]
                          for r in rows])
        tree = KDTree(cents)
        nn_d, _ = tree.query(cents, k=2)
        nn = nn_d[:, 1]
        areas = np.array([float(r["area"]) for r in rows])
        r_eff = np.sqrt(np.maximum(areas, 1e-9) / np.pi)
        aspects = np.array([float(r["aspect_ratio"]) for r in rows])
        med_nn = float(np.median(nn))
        med_r = float(np.median(r_eff))
        ratio = float(med_nn / med_r) if med_r > 1e-6 else 0.0
        coverage = float(np.sum(areas) / box_area) if box_area > 0 else 0.0
        out.append({
            "slice": s_idx + 1,
            "n_cells": n,
            "median_nn_um": med_nn,
            "median_r_eff_um": med_r,
            "ratio_nn_over_r_eff": ratio,
            "coverage_fraction": coverage,
            "median_aspect_ratio": float(np.median(aspects)),
        })
    return out


# --- 7. ONE SIMULATION RUN ---

def simulate_one(condition_name: str, seed: int, out_dir: str,
                 verbose: bool = True) -> Dict[str, Any]:
    """Generate phantoms + virtual sections for one (condition, seed).

    Writes:
      <out_dir>/SIM<condition>_S{1..N}_features.csv
      <out_dir>/ground_truth.csv
      <out_dir>/phantoms.json
    Returns a small summary dict.
    """
    cfg = resolve_condition_config(condition_name)
    rng = np.random.default_rng(seed)

    # ---- Shape-library mode selection ------------------------------------
    # Global config (set by CLI / caller) is read here so every sampler
    # invocation in this run sees a consistent library path / fallback mode.
    shape_lib_csv = _CFG_SHAPE_LIBRARY_CSV or default_tile046_shape_library_path()
    use_ellipse = bool(_CFG_ELLIPSE_CROSS_SECTION)
    n_templates_available = 0
    if not use_ellipse:
        templates = get_tile046_unit_templates(shape_lib_csv, force_ellipse=False)
        n_templates_available = len(templates)
        if n_templates_available == 0:
            use_ellipse = True  # library missing / empty - transparent fallback

    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. Sample 2D centroids (Poisson-disk or clustered) -----------------
    # ``min_dist_um`` (if set) overrides the density preset, allowing per-condition
    # fine-grained tuning of the centroid-ambiguity regime.
    min_dist_override = cfg.get("min_dist_um", None)
    if min_dist_override is not None:
        min_dist = float(min_dist_override)
    else:
        min_dist = DENSITY_MIN_DIST[cfg["density"]]
    placement = str(cfg.get("placement", "poisson"))
    cluster_ids: Optional[np.ndarray] = None
    if placement == "clustered":
        centroids_xy, cluster_ids = poisson_disk_clustered(
            cluster_center_dist=float(cfg["cluster_center_dist"]),
            intra_cluster_dist=float(cfg["intra_cluster_dist"]),
            cluster_size_range=tuple(cfg["cluster_size_range"]),
            box_x=BOX_X_UM, box_y=BOX_Y_UM, margin=20.0, rng=rng,
        )
    else:
        centroids_xy = poisson_disk_2d(
            min_dist=min_dist, box_x=BOX_X_UM, box_y=BOX_Y_UM,
            margin=20.0, rng=rng, max_attempts=3500,
        )
    n_cells = len(centroids_xy)

    # ---- 2. Generate phantoms -----------------------------------------------
    phantoms: List[Phantom] = []
    disarray_max = float(cfg["disarray_max_deg"])

    # For clustered placement, each cluster picks a shared base long-axis
    # direction (from a cone of half-angle disarray_max_deg around +z),
    # and member cells wobble within cluster_direction_wobble_deg of that base.
    cluster_base_dirs: Dict[int, np.ndarray] = {}
    if cluster_ids is not None:
        for cid in np.unique(cluster_ids):
            t_rad = np.deg2rad(float(rng.uniform(0.0, disarray_max)))
            t_az = float(rng.uniform(0.0, 2.0 * np.pi))
            cluster_base_dirs[int(cid)] = np.array([
                np.sin(t_rad) * np.cos(t_az),
                np.sin(t_rad) * np.sin(t_az),
                np.cos(t_rad),
            ])
    wobble_deg = float(cfg["cluster_direction_wobble_deg"])

    for i, (cx, cy) in enumerate(centroids_xy):
        if cluster_ids is not None:
            base = cluster_base_dirs[int(cluster_ids[i])]
            ph = sample_phantom_parameters(
                level=cfg["level"], cx=float(cx), cy=float(cy),
                taper=cfg["taper"], twist_deg=cfg["twist_deg"],
                curv_category=cfg["curv"], rng=rng,
                disarray_max_deg=disarray_max,
                base_direction=base, direction_wobble_deg=wobble_deg,
            )
        else:
            ph = sample_phantom_parameters(
                level=cfg["level"], cx=float(cx), cy=float(cy),
                taper=cfg["taper"], twist_deg=cfg["twist_deg"],
                curv_category=cfg["curv"], rng=rng,
                disarray_max_deg=disarray_max,
            )
        ph.gt_id = i
        phantoms.append(ph)

    # ---- 2b. Generate non-myocyte distractor cells (if any) ------------------
    n_distractors = int(cfg["n_distractors"])
    distractor_mode = str(cfg.get("distractor_mode", "blob"))
    if n_distractors > 0:
        # Distractors can sit between myocytes, but use 0.75 * min_dist (not 0.4)
        # so distractor-distractor and distractor-myocyte pairs stay near-touching
        # rather than crossing contours (min_dist for distractor placement is the
        # centroid gap; distractor b0 is <= 7 um so this is non-overlap).
        dist_min_dist = max(14.0, min_dist * 0.75)
        dist_xy = poisson_disk_2d(
            min_dist=dist_min_dist, box_x=BOX_X_UM, box_y=BOX_Y_UM,
            margin=15.0, rng=rng, max_attempts=4000,
        )
        dist_xy = dist_xy[:n_distractors]
        next_gt_id = len(phantoms)
        for j, (cx, cy) in enumerate(dist_xy):
            if distractor_mode == "myocyte_like":
                ph = sample_myocyte_like_distractor(float(cx), float(cy), rng)
            else:
                ph = sample_distractor_parameters(float(cx), float(cy), rng)
            ph.gt_id = next_gt_id + j
            phantoms.append(ph)

    # ---- 3. Section each phantom at all slices ------------------------------
    slice_tilt_rad = np.deg2rad(cfg["slice_tilt_deg"])
    zs = slice_z_targets(cfg["slice_tilt_deg"], cfg["thickness_jitter"], rng)

    # Per-slice rigid xy shift + small rotation modelling inter-slice
    # registration error (thermal / mechanical tissue drift).
    # ``interlayer_shift_mode`` supersedes the legacy ``interlayer_shift`` flag
    # when set to "iid" or "random_walk".
    shift_mode = str(cfg["interlayer_shift_mode"])
    shift_sigma = float(cfg["interlayer_shift_sigma_um"])
    rot_sigma_rad = float(np.deg2rad(cfg["interlayer_rot_sigma_deg"]))
    shifts = np.zeros((N_SLICES, 2), dtype=float)
    rots = np.zeros(N_SLICES, dtype=float)
    if shift_mode == "iid" and shift_sigma > 0.0:
        shifts[1:] = rng.normal(0.0, shift_sigma, size=(N_SLICES - 1, 2))
    elif shift_mode == "random_walk" and shift_sigma > 0.0:
        steps = rng.normal(0.0, shift_sigma, size=(N_SLICES - 1, 2))
        shifts[1:] = np.cumsum(steps, axis=0)
    elif shift_mode == "off" and cfg["interlayer_shift"]:
        # Preserve legacy knob: iid ~N(0, 3 um) on all but the first slice.
        shifts[1:] = rng.normal(0.0, 3.0, size=(N_SLICES - 1, 2))
    if rot_sigma_rad > 0.0:
        if shift_mode == "random_walk":
            rot_steps = rng.normal(0.0, rot_sigma_rad, size=N_SLICES - 1)
            rots[1:] = np.cumsum(rot_steps)
        else:
            rots[1:] = rng.normal(0.0, rot_sigma_rad, size=N_SLICES - 1)

    # Per-slice non-rigid warp field (list of Gaussian bumps) - applied in
    # absolute xy to every contour point; zero-amplitude = identity.
    warp_mode = str(cfg["warp_mode"])
    warp_fields: List[List[Dict[str, Any]]] = []
    warp_amp = float(cfg["warp_amp_um"])
    warp_sigma = float(cfg["warp_sigma_um"])
    warp_n_bumps = int(cfg["warp_n_bumps"])
    if warp_mode == "z_coherent" and float(cfg["z_coherent_warp_amp_um"]) > 0.0:
        warp_fields = build_z_coherent_warp_fields(
            amp_um=float(cfg["z_coherent_warp_amp_um"]),
            sigma_um=float(cfg["z_coherent_warp_sigma_um"]),
            n_bumps=int(cfg["z_coherent_warp_n_bumps"]),
            tau_slices=float(cfg["z_coherent_warp_tau_slices"]),
            n_slices=N_SLICES,
            box_x=BOX_X_UM, box_y=BOX_Y_UM, rng=rng,
        )
    else:
        for _ in range(N_SLICES):
            warp_fields.append(
                build_warp_field(amp_um=warp_amp, sigma_um=warp_sigma,
                                 n_bumps=warp_n_bumps,
                                 box_x=BOX_X_UM, box_y=BOX_Y_UM, rng=rng)
            )

    # Per (cell, slice) independent lateral jitter - breaks centroid continuity
    # cell-by-cell while preserving shape/OT neighbour set.
    pc_jitter_sigma = float(cfg["per_cell_xy_jitter_um"])
    total_cells = len(phantoms)
    if pc_jitter_sigma > 0.0 and total_cells > 0:
        per_cell_jitter = rng.normal(0.0, pc_jitter_sigma,
                                     size=(total_cells, N_SLICES, 2))
    else:
        per_cell_jitter = np.zeros((total_cells, N_SLICES, 2))

    # DTW-adversarial per-(cell, slice) shape rotation. Pre-allocate the full
    # (total_cells, N_SLICES) Gaussian draw so the RNG stream is deterministic
    # regardless of slice iteration order, mirroring the per_cell_jitter
    # pattern above.
    shape_rot_sigma_deg = float(cfg.get("per_slice_shape_rot_sigma_deg", 0.0) or 0.0)
    if shape_rot_sigma_deg > 0.0 and total_cells > 0:
        per_slice_shape_rot_rad = rng.normal(
            0.0, np.deg2rad(shape_rot_sigma_deg),
            size=(total_cells, N_SLICES),
        )
    else:
        per_slice_shape_rot_rad = np.zeros((total_cells, N_SLICES))

    # Per-contour-point radial noise sigma. The actual Gaussian draws are
    # consumed inline below from the master ``rng`` (one draw per contour
    # point), which is cheaper than pre-allocating an (N_cells, N_SLICES,
    # NUM_CONTOUR_POINTS) buffer and keeps the stream deterministic as long
    # as the outer phantom × slice loop order is fixed (which it is).
    radial_noise_sigma = float(cfg.get("per_slice_contour_radial_noise_um", 0.0) or 0.0)

    # Pre-sample each phantom's surface once and rotate globally by -tilt
    surface_cache: Dict[int, np.ndarray] = {}
    for ph in phantoms:
        surf = sample_phantom_surface(ph, n_t=260, n_u=72)
        surf = apply_slice_plane_transform(surf, slice_tilt_rad)
        surface_cache[ph.gt_id] = surf

    per_slice_rows: List[List[Dict[str, Any]]] = [[] for _ in range(N_SLICES)]
    ground_truth_records: List[Dict[str, Any]] = []

    label_counter = 0
    iter_phantoms = tqdm(phantoms, desc=f"    {condition_name}/seed{seed}", leave=False, disable=not verbose)
    for idx_ph, ph in enumerate(iter_phantoms):
        surf = surface_cache[ph.gt_id]
        for s_idx in range(N_SLICES):
            z_target = zs[s_idx]
            contour = extract_slice_contour(surf, z_target=z_target)
            if contour is None:
                continue
            # 1) rigid per-slice registration: translation + small rotation
            contour_shifted = rigid_xy_transform(
                contour, shifts[s_idx], rots[s_idx],
            )
            # 2) per-cell independent Gaussian jitter for this (cell, slice)
            if pc_jitter_sigma > 0.0:
                contour_shifted = contour_shifted + per_cell_jitter[idx_ph, s_idx][None, :]
            # 2b) DTW-adversarial per-(cell, slice) contour rotation about
            #     its OWN centroid. Centroid / area / neighbour structure
            #     unaffected; DTW point-wise cost rises ~linear in angle.
            if shape_rot_sigma_deg > 0.0:
                contour_shifted = rotate_contour_about_own_centroid(
                    contour_shifted, float(per_slice_shape_rot_rad[idx_ph, s_idx])
                )
            # 2c) DTW-adversarial per-contour-point radial Gaussian noise.
            #     Zero-mean along the outward radial direction, so centroid
            #     and area are preserved up to O(1/N). DTW cost rises ~N·σ².
            if radial_noise_sigma > 0.0:
                contour_shifted = add_radial_contour_noise(
                    contour_shifted, radial_noise_sigma, rng,
                )
            # 3) non-rigid warp field (applied in absolute xy)
            if warp_fields[s_idx]:
                contour_shifted = apply_warp_to_xy(contour_shifted, warp_fields[s_idx])
            label_counter += 1
            cell_id = f"SIM{condition_name}_S{s_idx+1:02d}_{ph.gt_id:04d}"
            row = _feature_row_from_contour(
                contour_shifted,
                cell_id=cell_id,
                cell_label=label_counter,
                region=condition_name,
                slice_str=f"S{s_idx+1:02d}",
                z_value=s_idx + 1,
            )
            per_slice_rows[s_idx].append(row)
            ground_truth_records.append({
                "slice": s_idx + 1,
                "cell_id": cell_id,
                "gt_track_id": ph.gt_id,
                "phantom_length_um": ph.length_um,
                "phantom_volume_um3": ph.volume_um3,
                "phantom_level": ph.level,
                "curv_category": ph.curv_category,
                "twist_deg": float(np.rad2deg(ph.twist_total_rad)),
                "taper": ph.taper,
                "slice_tilt_deg": cfg["slice_tilt_deg"],
            })

    # ---- 3b. Per-slice fibroblast-like distractors (no cross-slice identity)
    # These small 2D contours are sampled independently per slice. They get
    # the Tile046 row schema (so v10 loaders treat them as normal cells) but
    # are NOT recorded in ``ground_truth.csv`` - any method that links them
    # across slices will produce only false-positive pairs.
    n_per_slice = int(cfg.get("per_slice_distractors_per_slice", 0) or 0)
    distractors_per_slice_placed: List[int] = [0] * N_SLICES
    if n_per_slice > 0:
        area_range = tuple(cfg["per_slice_distractor_area_um2_range"])
        aspect_range = tuple(cfg["per_slice_distractor_aspect_range"])
        min_dist_my = float(cfg["per_slice_distractor_min_dist_to_myocyte_um"])
        for s_idx in range(N_SLICES):
            myo_centroids = [(float(r["centroid_x"]), float(r["centroid_y"]))
                             for r in per_slice_rows[s_idx]]
            for k in range(n_per_slice):
                abs_c = sample_per_slice_distractor_contour(
                    rng, myo_centroids,
                    area_range=area_range,
                    aspect_range=aspect_range,
                    min_dist_to_myocyte=min_dist_my,
                    box_x=BOX_X_UM, box_y=BOX_Y_UM,
                )
                if abs_c is None:
                    continue
                # Optionally bake in the per-slice warp (if the condition
                # enables one). We deliberately do NOT add interlayer_shift
                # or per_cell_jitter because the contour has no cross-slice
                # identity in the first place.
                if warp_fields[s_idx]:
                    abs_c = apply_warp_to_xy(abs_c, warp_fields[s_idx])
                label_counter += 1
                d_cell_id = f"SIM{condition_name}_S{s_idx+1:02d}_D{k:04d}"
                row = _feature_row_from_contour(
                    abs_c,
                    cell_id=d_cell_id,
                    cell_label=label_counter,
                    region=f"{condition_name}_distractor",
                    slice_str=f"S{s_idx+1:02d}",
                    z_value=s_idx + 1,
                )
                per_slice_rows[s_idx].append(row)
                myo_centroids.append(
                    (float(row["centroid_x"]), float(row["centroid_y"])))
                distractors_per_slice_placed[s_idx] += 1

    # ---- 4. Fill adjacency and write per-slice CSVs -------------------------
    cols = _all_columns()
    for s_idx in range(N_SLICES):
        rows = per_slice_rows[s_idx]
        _fill_adjacency(rows, threshold_um=NEIGHBOR_DIST_UM)
        df = pd.DataFrame(rows, columns=cols)
        fname = f"SIM{condition_name}_S{s_idx+1:02d}_features.csv"
        df.to_csv(os.path.join(out_dir, fname), index=False)

    # ---- 5. Ground truth + phantoms.json ------------------------------------
    gt_df = pd.DataFrame(ground_truth_records)
    gt_df.to_csv(os.path.join(out_dir, "ground_truth.csv"), index=False)

    def _warp_summary(f: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compact JSON-friendly summary per slice (full bumps can be large)."""
        return {
            "n_bumps": len(f),
            "bumps": [
                {"cx": round(b["cx"], 3), "cy": round(b["cy"], 3),
                 "sigma": round(b["sigma"], 3), "amp": round(b["amp"], 3),
                 "dir": [round(b["dx"], 3), round(b["dy"], 3)]}
                for b in f
            ],
        }

    # ---- 5a. Overlap + Tile046 calibration statistics (per slice) ----------
    overlap_stats = _compute_overlap_stats(per_slice_rows, n_contour=NUM_CONTOUR_POINTS)
    tile046_cal = _compute_tile046_calibration(per_slice_rows, BOX_X_UM, BOX_Y_UM)

    phantoms_dump = {
        "condition": condition_name,
        "seed": int(seed),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "n_phantoms": len(phantoms),
        "n_cells_observed": int(sum(len(r) for r in per_slice_rows)),
        "phantoms": [p.to_serialisable() for p in phantoms],
        "cluster_ids": (cluster_ids.tolist() if cluster_ids is not None else None),
        "slice_z_targets_um": [float(z) for z in zs],
        "interlayer_shift_mode": shift_mode,
        "interlayer_shift_sigma_um": shift_sigma,
        "interlayer_rot_sigma_deg": float(cfg["interlayer_rot_sigma_deg"]),
        "interlayer_shifts_um": shifts.tolist(),
        "interlayer_rotations_deg": np.rad2deg(rots).tolist(),
        "per_cell_jitter_sigma_um": float(pc_jitter_sigma),
        "per_slice_shape_rot_sigma_deg": float(shape_rot_sigma_deg),
        "per_slice_contour_radial_noise_um": float(radial_noise_sigma),
        "warp_mode": warp_mode,
        "z_coherent_warp": (
            {
                "amp_um": float(cfg["z_coherent_warp_amp_um"]),
                "sigma_um": float(cfg["z_coherent_warp_sigma_um"]),
                "n_bumps": int(cfg["z_coherent_warp_n_bumps"]),
                "tau_slices": float(cfg["z_coherent_warp_tau_slices"]),
            }
            if warp_mode == "z_coherent"
            else None
        ),
        "warp_fields_per_slice": [_warp_summary(f) for f in warp_fields],
        "per_slice_distractors_per_slice": int(
            cfg.get("per_slice_distractors_per_slice", 0) or 0),
        "per_slice_distractors_placed_per_slice": list(distractors_per_slice_placed),
        "overlap_stats_per_slice": overlap_stats,
        "tile046_calibration_per_slice": tile046_cal,
        "tile046_reference": dict(TILE046_REFERENCE),
        "shape_library": {
            "mode": ("ellipse" if use_ellipse else "tile046"),
            "csv_path": (None if use_ellipse else os.path.abspath(shape_lib_csv)),
            "n_templates_available": int(n_templates_available),
            "min_library_area_um2": float(_TILE046_LIB_MIN_AREA_UM2),
            "diversity_k": (None if use_ellipse else _CFG_SHAPE_DIVERSITY_K),
            "diversity_stats": (
                _TILE046_DIVERSE_STATS.get(
                    (os.path.abspath(shape_lib_csv), int(_CFG_SHAPE_DIVERSITY_K)))
                if (not use_ellipse and _CFG_SHAPE_DIVERSITY_K) else None
            ),
        },
    }
    with open(os.path.join(out_dir, "phantoms.json"), "w", encoding="utf-8") as f:
        json.dump(phantoms_dump, f, indent=2, ensure_ascii=False)

    overlap_fracs = [s["overlap_fraction"] for s in overlap_stats] or [0.0]
    overlap_cell_rates = [s.get("overlap_cell_rate", 0.0) for s in overlap_stats] or [0.0]
    ratio_vals = [s["ratio_nn_over_r_eff"] for s in tile046_cal if s["ratio_nn_over_r_eff"] > 0]
    coverage_vals = [s["coverage_fraction"] for s in tile046_cal if s["coverage_fraction"] > 0]

    return {
        "condition": condition_name,
        "seed": int(seed),
        "n_phantoms": len(phantoms),
        "n_observations": int(sum(len(r) for r in per_slice_rows)),
        "mean_cells_per_slice": float(np.mean([len(r) for r in per_slice_rows])) if per_slice_rows else 0.0,
        "mean_overlap_fraction": float(np.mean(overlap_fracs)),
        "max_overlap_fraction": float(np.max(overlap_fracs)),
        "mean_overlap_cell_rate": float(np.mean(overlap_cell_rates)),
        "max_overlap_cell_rate": float(np.max(overlap_cell_rates)),
        "median_nn_ratio": float(np.median(ratio_vals)) if ratio_vals else 0.0,
        "median_coverage": float(np.median(coverage_vals)) if coverage_vals else 0.0,
    }


# --- 8. CLI ---

def main():
    parser = argparse.ArgumentParser(description="Simulation 1 phantom generator.")
    parser.add_argument("--out-root", default="simulation01_results/phantoms",
                        help="Root directory for phantom outputs.")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Specific conditions to generate. Default: full grid.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 2 conditions x 1 seed.")
    parser.add_argument("--shape-library-csv", default=None,
                        help="Path to a Tile046-style features CSV used as the "
                             "cross-section shape library. Default: "
                             "03_demo/Tile046/046_S05_features.csv.")
    parser.add_argument("--ellipse-cross-section", action="store_true",
                        help="Force the legacy analytic-ellipse cross-section "
                             "(ignore the shape library).")
    parser.add_argument("--diverse-shape-top-k", type=int, default=30,
                        help="Pre-filter the Tile046 library to its K most "
                             "shape-diverse templates via farthest-point "
                             "selection on canonicalised contour L2 distance. "
                             "Use 0 (or a value >= library size) to disable "
                             "the filter. Default: 30.")
    args = parser.parse_args()

    set_phantom_shape_library_config(
        csv_path=args.shape_library_csv,
        ellipse_cross_section=bool(args.ellipse_cross_section),
        diversity_k=int(args.diverse_shape_top_k),
    )

    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    if args.quick:
        cond_names = list(QUICK_CONDITIONS)
        seeds = [args.seed_start]
    else:
        cond_names = args.conditions if args.conditions else list(CONDITIONS.keys())
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    print("Simulation 1 - phantom generator")
    print(f"  conditions: {cond_names}")
    print(f"  seeds:      {seeds}")
    print(f"  out-root:   {args.out_root}")
    print(f"  box:        {BOX_X_UM}x{BOX_Y_UM}x{BOX_Z_UM} um   ({N_SLICES} slices of {SLICE_THICKNESS_UM} um)")

    summary: List[Dict[str, Any]] = []
    for cond in cond_names:
        if cond not in CONDITIONS:
            print(f"  [SKIP] unknown condition: {cond}")
            continue
        print(f"\n[{cond}]  cfg = {CONDITIONS[cond]}")
        for seed in seeds:
            out_dir = os.path.join(args.out_root, cond, f"seed{seed}")
            info = simulate_one(cond, seed, out_dir, verbose=True)
            summary.append(info)
            print(f"    -> seed{seed}: {info['n_phantoms']} phantoms, "
                  f"{info['n_observations']} cross-sections, "
                  f"mean {info['mean_cells_per_slice']:.1f} cells/slice, "
                  f"overlap cell-rate mean/max "
                  f"{info.get('mean_overlap_cell_rate', 0.0)*100:.1f}/"
                  f"{info.get('max_overlap_cell_rate', 0.0)*100:.1f}%, "
                  f"NN/r_eff={info.get('median_nn_ratio', 0.0):.2f} "
                  f"(target 1.44), coverage={info.get('median_coverage', 0.0)*100:.1f}% "
                  f"(target 81%)")

    # Global index
    os.makedirs(args.out_root, exist_ok=True)
    idx_path = os.path.join(args.out_root, "_index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "conditions_used": cond_names, "seeds": seeds,
                   "schema_columns": _all_columns()[:20] + ["...x_i, y_i..."] + _all_columns()[-2:]},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  Index written to {idx_path}")
    print("Done.")


if __name__ == "__main__":
    main()
