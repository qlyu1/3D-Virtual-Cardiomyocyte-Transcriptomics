#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation 1 - Run 4 ablation methods on all phantom datasets.

Methods (shared track-building skeleton, only the per-pair cost differs):
  * Centroid_NN   : cost = centroid distance                            (main baseline)
  * DTW_only      : cost = Soft-DTW (raw, normalised globally to [0,10])
  * OT_only       : cost = Cluster Sinkhorn OT (raw, normalised)
  * HiDTW_full    : cost = w_dtw*DTW + w_ot*OT + w_cent*Centroid        (auto-tuned
                    per (condition, seed) via simplex grid search on GT pair-F1,
                    mirrors the per-region BO used on real data)

Inputs  : simulation01_results/phantoms/<condition>/seed<k>/SIM*_S??_features.csv
Outputs : simulation01_results/methods/<condition>/seed<k>/<method>/tracks.csv

The expensive GPU step (raw DTW/OT/centroid/IoU for every source-cell -> top-K
target-cell pair, in both directions) is computed ONCE per (condition, seed)
and cached on disk; each method simply applies its own weighted cost formula
to the cached distances.

Usage:
  python simulation01_run_methods.py --gpu-id 0
  python simulation01_run_methods.py --quick
  python simulation01_run_methods.py --conditions L2_stress_combo --seeds 0 1 2
  python simulation01_run_methods.py --methods Centroid_NN HiDTW_full
"""

from __future__ import annotations

# In single-worker mode we pin CUDA_VISIBLE_DEVICES before importing torch.
# In multi-worker mode each child does this in ``_worker_init`` instead.
import argparse as _argparse_early
import os as _os_early

_early_parser = _argparse_early.ArgumentParser(add_help=False)
_early_parser.add_argument("--gpu-id", type=int, default=None)
_early_parser.add_argument("--num-workers", type=int, default=None)
_early_args, _ = _early_parser.parse_known_args()
_early_single_worker = (_early_args.num_workers is None
                        or int(_early_args.num_workers) <= 1)
if _early_args.gpu_id is not None and _early_single_worker:
    _os_early.environ["CUDA_VISIBLE_DEVICES"] = str(_early_args.gpu_id)
    print(f"[Early Init] CUDA_VISIBLE_DEVICES = {_early_args.gpu_id}")
del _early_parser, _early_args, _early_single_worker

import argparse
import copy
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SIM_DIR)
for _p in (_REPO_ROOT, _SIM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from soft_dtw_cuda import SoftDTW  # noqa: E402
from hidtw_pipeline import cell_reconstruction_runGPU_3Drevised_v10 as v10  # noqa: E402

Cell = v10.Cell
MAX_XY_DISTANCE = v10.MAX_XY_DISTANCE
OT_PENALTY_VALUE = v10.OT_PENALTY_VALUE
NUM_CONTOUR_POINTS = v10.NUM_CONTOUR_POINTS

from simulation01_phantom_generator import (  # noqa: E402
    CONDITIONS,
    QUICK_CONDITIONS,
    canonicalize_contour_start,
)


# --- 1. METHOD DEFINITIONS ---

# Static methods: fixed weights, no tuning.
STATIC_METHODS: Dict[str, Dict[str, float]] = {
    "Centroid_NN":      {"centroid": 1.0},
    "DTW_only":         {"dtw": 1.0},
    "OT_only":          {"ot": 1.0},
}

# Tuned methods: weights selected per-dataset via simplex grid search on GT
# pair-F1. The placeholder weights here are only used with --no-hidtw-tune.
TUNED_METHODS: Dict[str, Dict[str, float]] = {
    "HiDTW_full":       {"dtw": 1.0 / 3.0, "ot": 1.0 / 3.0, "centroid": 1.0 / 3.0},
}

METHODS: Dict[str, Dict[str, float]] = {**STATIC_METHODS, **TUNED_METHODS}

# Track-building parameters (shared across methods).
RM_THRESHOLD = 0.01
MIN_TRACK_LENGTH = 3
TOP_K_CANDIDATES = 40
TRACK_MARGIN_WEIGHT = 0.7
TRACK_LENGTH_WEIGHT = 0.3

# Auto-tune search space for HiDTW_full.
HIDTW_WEIGHT_GRID_STEP = 0.1                   # 66 combinations on the simplex
HIDTW_ALPHA_GRID = (0.005, 0.01, 0.05)          # 3 RM thresholds to sweep


# --- 2. GPU PRE-COMPUTATION OF ALL PAIRWISE COSTS ---

def _safe_polygon(points_xy: np.ndarray):
    """Build a shapely Polygon, self-repairing with buffer(0) if invalid."""
    from shapely.geometry import Polygon

    poly = Polygon(points_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _batch_dtw(src: Cell, cands: List[Tuple[Cell, float]],
               sdtw: SoftDTW, device: torch.device) -> np.ndarray:
    """Return (n_cands,) array of raw DTW distances.

    Contour sequences are passed through ``canonicalize_contour_start``
    before being batched into Soft-DTW. This fixes the start-index /
    direction drift inherited from the upstream segmentation (real data)
    or ``ConvexHull`` (synthetic phantoms), to which linear Soft-DTW is
    extremely sensitive. v10's main pipeline is untouched; this layer
    only covers the simulation runner.
    """
    n = len(cands)
    src_rel = canonicalize_contour_start(src.contour_points_200)
    tgt_rels = np.stack([
        canonicalize_contour_start(c[0].contour_points_200) for c in cands
    ])
    x = torch.tensor(np.tile(src_rel, (n, 1, 1)), dtype=torch.float32).to(device)
    y = torch.tensor(tgt_rels, dtype=torch.float32).to(device)
    return sdtw(x, y).cpu().numpy()


def _compute_one_direction(src_cells: List[Cell], tgt_cells: List[Cell],
                           cells_by_slice: Dict[int, List[Cell]],
                           sdtw: SoftDTW, device: torch.device,
                           top_k: int = TOP_K_CANDIDATES) -> Dict[str, List[Dict[str, Any]]]:
    """For each source cell, compute raw DTW/OT/centroid/IoU for top-K centroid-nearest targets."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    id_to_tgt_cell = {c.cell_id: c for c in tgt_cells}

    for src in tqdm(src_cells, desc=f"    slice {src_cells[0].slice_index}",
                    leave=False, disable=(len(src_cells) < 3)):
        # Centroid pre-filter (match v10's MAX_XY_DISTANCE window)
        cands: List[Tuple[Cell, float]] = []
        for tgt in tgt_cells:
            d = float(np.linalg.norm(src.centroid - tgt.centroid))
            if d < MAX_XY_DISTANCE:
                cands.append((tgt, d))
        cands.sort(key=lambda x: x[1])
        cands = cands[:top_k]
        if not cands:
            result[src.cell_id] = []
            continue

        # Batch DTW (GPU when available)
        dtws = _batch_dtw(src, cands, sdtw, device)

        cand_list: List[Dict[str, Any]] = []
        for i, (tgt, cent_d) in enumerate(cands):
            cand_list.append({
                "target_id": tgt.cell_id,
                "target_slice": int(tgt.slice_index),
                "dtw_raw": float(dtws[i]),
                "ot_raw": float(OT_PENALTY_VALUE),
                "centroid_raw": float(cent_d),
                "iou_cost": 1.0,   # will be overwritten below
            })

        # OT for ALL candidates (simulation budget permits it and is fairer for OT_only)
        for c in cand_list:
            tgt_cell = id_to_tgt_cell.get(c["target_id"])
            if tgt_cell is None:
                continue
            try:
                cl_s = v10.get_cluster_cells(src, cells_by_slice)
                cl_t = v10.get_cluster_cells(tgt_cell, cells_by_slice)
                if len(cl_s) == 1 and len(cl_t) == 1:
                    c["ot_raw"] = 0.0
                    continue
                ot_raw = v10.compute_cluster_ot_distance(
                    src, tgt_cell, cells_by_slice, use_area_weights=True)
                if np.isfinite(ot_raw):
                    c["ot_raw"] = float(ot_raw)
            except Exception:
                pass

        # IoU (shapely, CPU). Done in absolute coordinates (centroid + relative contour).
        src_abs = src.centroid[None, :] + src.contour_points_200
        src_poly = _safe_polygon(src_abs)
        src_area = max(src_poly.area, 1e-9)
        for c in cand_list:
            tgt_cell = id_to_tgt_cell.get(c["target_id"])
            if tgt_cell is None:
                c["iou_cost"] = 1.0
                continue
            tgt_abs = tgt_cell.centroid[None, :] + tgt_cell.contour_points_200
            tgt_poly = _safe_polygon(tgt_abs)
            try:
                inter = src_poly.intersection(tgt_poly).area
                union = src_poly.area + tgt_poly.area - inter
                iou = inter / union if union > 1e-9 else 0.0
            except Exception:
                iou = 0.0
            c["iou_cost"] = float(1.0 - max(0.0, min(1.0, iou)))

        result[src.cell_id] = cand_list

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def precompute_all_costs(cells_by_slice: Dict[int, List[Cell]],
                         cache_file: str, force: bool = False) -> Dict[str, Any]:
    if os.path.exists(cache_file) and not force:
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        print(f"    [cache] loaded {cache_file}")
        return cached

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sdtw = SoftDTW(use_cuda=torch.cuda.is_available(), gamma=0.1)
    slices = sorted(cells_by_slice.keys())

    forward: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    backward: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    t0 = time.time()
    for si in range(len(slices) - 1):
        s_lo, s_hi = slices[si], slices[si + 1]
        print(f"    layer pair {s_lo} -> {s_hi}  "
              f"(|src|={len(cells_by_slice[s_lo])}  |tgt|={len(cells_by_slice[s_hi])})")
        forward[s_lo] = _compute_one_direction(
            cells_by_slice[s_lo], cells_by_slice[s_hi], cells_by_slice, sdtw, device)
        print(f"    layer pair {s_hi} -> {s_lo}")
        backward[s_hi] = _compute_one_direction(
            cells_by_slice[s_hi], cells_by_slice[s_lo], cells_by_slice, sdtw, device)

    data = {"forward": forward, "backward": backward}
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    [cache] wrote {cache_file}  ({time.time()-t0:.1f}s)")
    return data


# --- 3. PER-REGION NORMALISATION DIVISORS ---

def calibrate_divisors(precomputed: Dict[str, Any]) -> Dict[str, float]:
    """Data-driven divisors so every raw cost maps roughly to [0, 10]."""
    dtw_all, ot_all, cent_all, iou_all = [], [], [], []
    for direction in ("forward", "backward"):
        for cand_map in precomputed[direction].values():
            for cand_list in cand_map.values():
                for c in cand_list:
                    dtw_all.append(c["dtw_raw"])
                    if c["ot_raw"] < OT_PENALTY_VALUE:
                        ot_all.append(c["ot_raw"])
                    cent_all.append(c["centroid_raw"])
                    iou_all.append(c["iou_cost"])
    dtw_div = float(np.percentile(dtw_all, 75) * 10.0 / 7.5) if dtw_all else 1.0
    ot_div = float(np.percentile(ot_all, 75) * 10.0 / 7.5) if ot_all else 1.0
    cent_div = float(np.percentile(cent_all, 75) * 10.0 / 7.5) if cent_all else 1.0
    # IoU cost already in [0,1]; scale to [0,10] by multiplying by 10.
    iou_div = 0.1
    return {"dtw": max(dtw_div, 1e-6),
            "ot":  max(ot_div, 1e-6),
            "centroid": max(cent_div, 1e-6),
            "iou": iou_div}


# --- 4. CUSTOM TRACK BUILDER (method-agnostic, RM + greedy conflict resolve) ---

def _score_candidates(cands: List[Dict[str, Any]], weights: Dict[str, float],
                      divisors: Dict[str, float]) -> List[Dict[str, Any]]:
    """Score candidates via method weights (GLOBAL) + adaptive LOCAL for RM decision.

    Each candidate gets three fields added:
      loss_local   - for ranking (adaptive P75 scaling inside current batch)
      loss_global  - for quality scoring (fixed divisors)
    """
    if not cands:
        return []

    # adaptive local divisors (P75 within batch)
    def _p75(vals: List[float], fallback: float) -> float:
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            return fallback
        return max(1e-6, float(np.percentile(vals, 75)) * 10.0 / 7.5)

    dtws = [c["dtw_raw"] for c in cands]
    ots = [c["ot_raw"] for c in cands if c["ot_raw"] < OT_PENALTY_VALUE]
    cents = [c["centroid_raw"] for c in cands]
    ious = [c["iou_cost"] for c in cands]

    dtw_loc = _p75(dtws, divisors["dtw"])
    ot_loc = _p75(ots, divisors["ot"])
    cent_loc = _p75(cents, divisors["centroid"])
    iou_loc = 0.1   # IoU cost already normalised to [0,1]

    scored: List[Dict[str, Any]] = []
    for c in cands:
        # Per-term normalised component (local and global)
        dtw_l = min(10.0, c["dtw_raw"] / dtw_loc * 10.0)
        ot_l = 10.0 if c["ot_raw"] >= OT_PENALTY_VALUE else min(10.0, c["ot_raw"] / ot_loc * 10.0)
        cent_l = min(10.0, c["centroid_raw"] / cent_loc * 10.0)
        iou_l = min(10.0, c["iou_cost"] / iou_loc * 10.0)

        dtw_g = min(10.0, c["dtw_raw"] / divisors["dtw"] * 10.0)
        ot_g = 10.0 if c["ot_raw"] >= OT_PENALTY_VALUE else min(10.0, c["ot_raw"] / divisors["ot"] * 10.0)
        cent_g = min(10.0, c["centroid_raw"] / divisors["centroid"] * 10.0)
        iou_g = min(10.0, c["iou_cost"] / divisors["iou"] * 10.0)

        w = weights
        loss_local = (w.get("dtw", 0) * dtw_l + w.get("ot", 0) * ot_l
                      + w.get("centroid", 0) * cent_l + w.get("iou", 0) * iou_l)
        loss_global = (w.get("dtw", 0) * dtw_g + w.get("ot", 0) * ot_g
                       + w.get("centroid", 0) * cent_g + w.get("iou", 0) * iou_g)
        scored.append({**c, "loss_local": loss_local, "loss_global": loss_global})
    scored.sort(key=lambda x: x["loss_local"])
    return scored


def _find_best(cid: str, source_slice: int, direction: str,
               precomputed: Dict[str, Any], used: set,
               weights: Dict[str, float], divisors: Dict[str, float],
               alpha: float = RM_THRESHOLD):
    lookup = precomputed["forward"] if direction == "forward" else precomputed["backward"]
    if source_slice not in lookup or cid not in lookup[source_slice]:
        return None
    cands = [c for c in lookup[source_slice][cid] if c["target_id"] not in used]
    if not cands:
        return None
    scored = _score_candidates(cands, weights, divisors)
    if not scored:
        return None
    top1 = scored[0]
    if len(scored) > 1:
        top2 = scored[1]
        # abs(top1) + eps in the denominator keeps the "gap relative to the
        # winner's scale" semantics when Soft-DTW returns a small or
        # negative top1 (degenerate softmin on near-identical contours).
        rm_local = ((top2["loss_local"] - top1["loss_local"])
                    / (abs(top1["loss_local"]) + 1e-6))
        rm_global = ((top2["loss_global"] - top1["loss_global"])
                     / (abs(top1["loss_global"]) + 1e-6))
    else:
        rm_local = 1.0
        rm_global = 1.0
    if rm_local < alpha:
        return None
    return (top1["target_id"], top1["target_slice"], rm_local, rm_global)


def _build_direction(start_id: str, start_slice: int, direction: str,
                     precomputed: Dict[str, Any], used: set,
                     cells_by_slice: Dict[int, List[Cell]],
                     weights: Dict[str, float], divisors: Dict[str, float],
                     alpha: float = RM_THRESHOLD):
    cells_list: List[Tuple[int, str]] = [(start_slice, start_id)]
    margins: List[float] = []
    cur_id, cur_s = start_id, start_slice
    slices = sorted(cells_by_slice.keys())
    min_s, max_s = slices[0], slices[-1]
    while True:
        nxt = cur_s + 1 if direction == "forward" else cur_s - 1
        if nxt < min_s or nxt > max_s or nxt not in cells_by_slice:
            break
        m = _find_best(cur_id, cur_s, direction, precomputed, used, weights, divisors, alpha)
        if m is None:
            break
        tgt_id, tgt_s, _rm_local, rm_global = m
        cells_list.append((tgt_s, tgt_id))
        margins.append(rm_global)
        cur_id, cur_s = tgt_id, tgt_s
    return cells_list, margins


def _merge_bi(fwd_cells, fwd_margins, bwd_cells, bwd_margins):
    bwd_lower = sorted([(s, c) for (s, c) in bwd_cells if s < fwd_cells[0][0]])
    merged_cells = bwd_lower + fwd_cells
    merged_margins = list(reversed(bwd_margins)) + fwd_margins
    return merged_cells, merged_margins


def _track_quality(t: Dict[str, Any], m_min: float, m_range: float, max_len: int) -> float:
    avg_m = float(np.mean(t["margins"])) if t["margins"] else 0.0
    norm_m = (avg_m - m_min) / m_range if m_range > 1e-9 else 0.0
    norm_l = min(1.0, t["length"] / max_len) if max_len > 0 else 0.0
    return TRACK_MARGIN_WEIGHT * norm_m + TRACK_LENGTH_WEIGHT * norm_l


def _resolve_conflicts(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not tracks:
        return []
    margins_list = [float(np.mean(t["margins"])) if t["margins"] else 0.0 for t in tracks]
    lengths_list = [t["length"] for t in tracks]
    m_min, m_max = min(margins_list), max(margins_list)
    m_range = max(m_max - m_min, 1e-9)
    max_len = max(lengths_list) if lengths_list else 1
    tracks.sort(key=lambda t: _track_quality(t, m_min, m_range, max_len), reverse=True)
    cell_owner: Dict[str, int] = {}
    track_map: Dict[int, Dict[str, Any]] = {}
    selected: List[Dict[str, Any]] = []
    for tidx, t in enumerate(tracks):
        cids = set(cid for _, cid in t["cells"])
        conflict = {cell_owner[c] for c in cids if c in cell_owner}
        if not conflict:
            selected.append(t)
            track_map[tidx] = t
            for c in cids:
                cell_owner[c] = tidx
        else:
            q_t = _track_quality(t, m_min, m_range, max_len)
            worst_q = min(_track_quality(track_map[ci], m_min, m_range, max_len) for ci in conflict)
            if q_t > worst_q:
                worst_id = min(conflict,
                               key=lambda ci: _track_quality(track_map[ci], m_min, m_range, max_len))
                worst = track_map[worst_id]
                selected.remove(worst)
                for _, wid in worst["cells"]:
                    if cell_owner.get(wid) == worst_id:
                        del cell_owner[wid]
                del track_map[worst_id]
                selected.append(t)
                track_map[tidx] = t
                for c in cids:
                    cell_owner[c] = tidx
    return selected


def build_all_tracks(cells_by_slice: Dict[int, List[Cell]],
                     precomputed: Dict[str, Any],
                     weights: Dict[str, float],
                     divisors: Dict[str, float],
                     alpha: float = RM_THRESHOLD) -> List[Dict[str, Any]]:
    all_tracks: List[Dict[str, Any]] = []
    used: set = set()
    slices = sorted(cells_by_slice.keys())
    min_s = slices[0]
    while True:
        round_tracks: List[Dict[str, Any]] = []
        for start_s in slices:
            for cell in cells_by_slice[start_s]:
                if cell.cell_id in used:
                    continue
                fwd_cells, fwd_m = _build_direction(
                    cell.cell_id, cell.slice_index, "forward",
                    precomputed, used, cells_by_slice, weights, divisors, alpha)
                if start_s > min_s:
                    bwd_cells, bwd_m = _build_direction(
                        cell.cell_id, cell.slice_index, "backward",
                        precomputed, used, cells_by_slice, weights, divisors, alpha)
                    merged_cells, merged_m = _merge_bi(fwd_cells, fwd_m, bwd_cells, bwd_m)
                else:
                    merged_cells, merged_m = fwd_cells, fwd_m
                if len(merged_cells) < MIN_TRACK_LENGTH:
                    continue
                ss = [s for s, _ in merged_cells]
                if any(ss[i + 1] - ss[i] != 1 for i in range(len(ss) - 1)):
                    continue
                round_tracks.append({"cells": merged_cells,
                                     "margins": merged_m,
                                     "length": len(merged_cells)})
        if not round_tracks:
            break
        sel = _resolve_conflicts(round_tracks)
        all_tracks.extend(sel)
        for t in sel:
            for _, cid in t["cells"]:
                used.add(cid)
    return all_tracks


# --- 4b. AUTO-TUNE HIDTW_FULL WEIGHTS (simplex grid search, GT pair-F1) ---

def _simplex_grid(step: float = HIDTW_WEIGHT_GRID_STEP) -> List[Dict[str, float]]:
    """All (w_dtw, w_ot, w_cent) triples on the unit simplex with given step."""
    n = int(round(1.0 / step))
    combos = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            combos.append({"dtw": round(i / n, 6),
                           "ot":  round(j / n, 6),
                           "centroid": round(k / n, 6)})
    return combos


def _gt_pairs_from_df(gt_df: pd.DataFrame) -> set:
    """Set of (slice_a, cell_a, slice_b, cell_b) GT adjacent-layer pairs."""
    pairs = set()
    for gid, sub in gt_df.groupby("gt_track_id"):
        sub = sub.sort_values("slice")
        rows = sub.to_dict("records")
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            if int(b["slice"]) - int(a["slice"]) == 1:
                pairs.add((int(a["slice"]), str(a["cell_id"]),
                           int(b["slice"]), str(b["cell_id"])))
    return pairs


def _pair_f1_from_tracks(tracks: List[Dict[str, Any]], gt_pairs_set: set) -> Tuple[float, float, float, int]:
    """Compute pair-level Precision / Recall / F1 (and #pred pairs) of a tracks list."""
    pred_pairs = set()
    for t in tracks:
        ordered = sorted(t["cells"], key=lambda x: x[0])
        for i in range(len(ordered) - 1):
            s1, c1 = ordered[i]; s2, c2 = ordered[i + 1]
            if int(s2) - int(s1) == 1:
                pred_pairs.add((int(s1), str(c1), int(s2), str(c2)))
    tp = len(pred_pairs & gt_pairs_set)
    fp = len(pred_pairs - gt_pairs_set)
    fn = len(gt_pairs_set - pred_pairs)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return f1, p, r, len(pred_pairs)


def tune_hidtw_weights(cells_by_slice: Dict[int, List[Cell]],
                       precomputed: Dict[str, Any],
                       divisors: Dict[str, float],
                       gt_pairs_set: set,
                       weight_step: float = HIDTW_WEIGHT_GRID_STEP,
                       alpha_grid: Sequence[float] = HIDTW_ALPHA_GRID,
                       verbose: bool = True) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Grid search over (alpha, dtw_w, ot_w, cent_w) maximising GT pair-F1.

    Returns (best_config, full_grid_log_rows). ``best_config`` is a dict with
    keys ``alpha, dtw, ot, centroid, f1, precision, recall``.
    """
    combos = _simplex_grid(weight_step)
    rows: List[Dict[str, Any]] = []
    best = {"f1": -1.0}
    it = combos
    total = len(combos) * len(alpha_grid)
    if verbose:
        it = tqdm(combos, desc="      HiDTW tune", leave=False)
    for w in it:
        for alpha in alpha_grid:
            tracks = build_all_tracks(cells_by_slice, precomputed,
                                      w, divisors, alpha=alpha)
            f1, p, r, n_pred = _pair_f1_from_tracks(tracks, gt_pairs_set)
            row = {"alpha": float(alpha),
                   "dtw": float(w["dtw"]), "ot": float(w["ot"]),
                   "centroid": float(w["centroid"]),
                   "f1": float(f1), "precision": float(p), "recall": float(r),
                   "n_tracks": int(len(tracks)),
                   "n_pred_pairs": int(n_pred)}
            rows.append(row)
            if f1 > best["f1"] + 1e-12:
                best = dict(row)
    return best, rows




# --- 5. OUTPUT WRITER (schema = all_tracks_summary.csv) ---

def tracks_to_dataframe(tracks: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for tid, t in enumerate(tracks, start=1):
        ids = [cid for _, cid in sorted(t["cells"])]
        margins = t["margins"]
        rows.append({
            "track_id": tid,
            "cell_ids": "|".join(ids),
            "track_length": len(ids),
            "avg_margin": float(np.mean(margins)) if margins else 0.0,
            "min_margin": float(np.min(margins)) if margins else 0.0,
            "max_margin": float(np.max(margins)) if margins else 0.0,
            "std_margin": float(np.std(margins)) if len(margins) > 1 else 0.0,
            "num_connections": len(margins),
        })
    return pd.DataFrame(rows)


# --- 6. ONE DATASET (condition, seed) ---

def run_one(condition: str, seed: int, phantom_root: str, method_root: str,
            cache_root: str, methods_to_run: List[str],
            force_precompute: bool = False,
            hidtw_tune: bool = True,
            hidtw_tune_step: float = HIDTW_WEIGHT_GRID_STEP,
            hidtw_alpha_grid: Sequence[float] = HIDTW_ALPHA_GRID) -> Dict[str, Any]:
    phantom_dir = os.path.join(phantom_root, condition, f"seed{seed}")
    if not os.path.isdir(phantom_dir):
        return {"condition": condition, "seed": seed, "skipped": True,
                "reason": f"phantom dir missing: {phantom_dir}"}

    out_base = os.path.join(method_root, condition, f"seed{seed}")
    os.makedirs(out_base, exist_ok=True)

    # ---- load cells ---------------------------------------------------------
    cells_by_slice = v10.load_cells_from_csv_files(phantom_dir)
    n_cells = sum(len(v) for v in cells_by_slice.values())
    print(f"  cells loaded: {n_cells} across slices {sorted(cells_by_slice.keys())}")

    # ---- pre-compute all pairwise costs ------------------------------------
    cache_file = os.path.join(cache_root, condition, f"seed{seed}", "precomputed_costs.pkl")
    print(f"  pre-compute costs ...")
    precomputed = precompute_all_costs(cells_by_slice, cache_file, force=force_precompute)

    # ---- calibrate divisors -----------------------------------------------
    divisors = calibrate_divisors(precomputed)
    print(f"  divisors: {divisors}")
    with open(os.path.join(out_base, "divisors.json"), "w", encoding="utf-8") as f:
        json.dump(divisors, f, indent=2)

    # ---- resolve each method's weights + alpha -----------------------------
    # For tuned methods (HiDTW_full), run a grid search against GT pair-F1;
    # for static methods, use their pre-defined weights with the global alpha.
    method_configs: Dict[str, Dict[str, Any]] = {}
    tune_best: Optional[Dict[str, float]] = None

    for mname in methods_to_run:
        if mname in STATIC_METHODS:
            method_configs[mname] = {
                "weights": STATIC_METHODS[mname],
                "alpha": RM_THRESHOLD,
                "source": "fixed",
            }
        elif mname in TUNED_METHODS:
            if hidtw_tune:
                if tune_best is None:
                    gt_df = pd.read_csv(os.path.join(phantom_dir, "ground_truth.csv"))
                    gt_df["slice"] = gt_df["slice"].astype(int)
                    gt_df["cell_id"] = gt_df["cell_id"].astype(str)
                    gt_pairs_set = _gt_pairs_from_df(gt_df)
                    print(f"  [tune] HiDTW_full weight simplex search "
                          f"(step={hidtw_tune_step}, alphas={list(hidtw_alpha_grid)}, "
                          f"|gt_pairs|={len(gt_pairs_set)}) ...")
                    t_tune0 = time.time()
                    tune_best, tune_rows = tune_hidtw_weights(
                        cells_by_slice, precomputed, divisors, gt_pairs_set,
                        weight_step=hidtw_tune_step, alpha_grid=hidtw_alpha_grid,
                    )
                    tune_elapsed = time.time() - t_tune0
                    print(f"  [tune] best  alpha={tune_best['alpha']:.4f}  "
                          f"w=({tune_best['dtw']:.2f},{tune_best['ot']:.2f},"
                          f"{tune_best['centroid']:.2f})  F1={tune_best['f1']:.4f}  "
                          f"({tune_elapsed:.1f}s, {len(tune_rows)} evals)")
                    mdir = os.path.join(out_base, mname)
                    os.makedirs(mdir, exist_ok=True)
                    with open(os.path.join(mdir, "tuned_weights.json"), "w", encoding="utf-8") as f:
                        json.dump({"best": tune_best,
                                   "search": {"weight_step": hidtw_tune_step,
                                              "alpha_grid": list(hidtw_alpha_grid)},
                                   "n_evaluations": len(tune_rows),
                                   "elapsed_sec": float(tune_elapsed)},
                                  f, indent=2)
                    pd.DataFrame(tune_rows).to_csv(
                        os.path.join(mdir, "grid_search.csv"), index=False)
                method_configs[mname] = {
                    "weights": {"dtw": tune_best["dtw"],
                                "ot": tune_best["ot"],
                                "centroid": tune_best["centroid"]},
                    "alpha": float(tune_best["alpha"]),
                    "source": "tuned_on_gt_f1",
                }
            else:
                method_configs[mname] = {
                    "weights": TUNED_METHODS[mname],
                    "alpha": RM_THRESHOLD,
                    "source": "tuned_disabled_placeholder",
                }
        else:
            print(f"  [warn] unknown method {mname}, skipping")

    # ---- run each method ---------------------------------------------------
    per_method_summary: Dict[str, Any] = {}
    for mname, cfg in method_configs.items():
        t0 = time.time()
        tracks = build_all_tracks(cells_by_slice, precomputed,
                                  cfg["weights"], divisors, alpha=cfg["alpha"])
        elapsed = time.time() - t0
        mdir = os.path.join(out_base, mname)
        os.makedirs(mdir, exist_ok=True)
        df = tracks_to_dataframe(tracks)
        df.to_csv(os.path.join(mdir, "tracks.csv"), index=False)
        with open(os.path.join(mdir, "weights_used.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        per_method_summary[mname] = {
            "n_tracks": int(len(tracks)),
            "mean_length": float(np.mean([t["length"] for t in tracks])) if tracks else 0.0,
            "max_length": int(max([t["length"] for t in tracks])) if tracks else 0,
            "cells_assigned": int(sum(t["length"] for t in tracks)),
            "elapsed_sec": float(elapsed),
            "weights": cfg["weights"],
            "alpha": cfg["alpha"],
            "source": cfg["source"],
        }
        print(f"    {mname:18s}  tracks={per_method_summary[mname]['n_tracks']:4d}  "
              f"mean_len={per_method_summary[mname]['mean_length']:5.2f}  "
              f"covered={per_method_summary[mname]['cells_assigned']:4d}  "
              f"({elapsed:.2f}s)  [{cfg['source']}]")

    summary = {
        "condition": condition, "seed": seed, "n_cells": n_cells,
        "methods": per_method_summary,
        "static_method_defaults": STATIC_METHODS,
        "tuned_best": tune_best,
        "divisors": divisors,
    }
    with open(os.path.join(out_base, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# --- 6b. MULTI-GPU WORKER HELPERS (dataset-level parallelism) ---

# Set by ``_worker_init`` in each child; used as a log-line prefix.
_WORKER_GPU_SLOT: Optional[str] = None


def _resolve_gpu_ids(num_workers: Optional[int]) -> List[str]:
    """Return GPU ids to distribute across workers (honours
    CUDA_VISIBLE_DEVICES; otherwise uses 0..N-1; falls back to "" on CPU)."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        visible = [x.strip() for x in env.split(",") if x.strip() != ""]
    else:
        try:
            n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            n_visible = 0
        visible = [str(i) for i in range(n_visible)]

    if not visible:
        print("[warn] no CUDA devices visible; workers will run on CPU.")
        n = max(1, int(num_workers) if num_workers else 1)
        return [""] * n

    if num_workers is None or num_workers <= 0:
        return visible
    n_req = int(num_workers)
    if n_req > len(visible):
        print(f"[warn] requested {n_req} workers but only {len(visible)} GPU(s) "
              f"visible ({visible}); scaling down.")
        return visible
    return visible[:n_req]


def _worker_init(gpu_queue) -> None:
    """Pin this child to one gpu and cap BLAS threads."""
    global _WORKER_GPU_SLOT
    try:
        gpu_id = gpu_queue.get(timeout=30)
    except Exception:
        gpu_id = ""
    _WORKER_GPU_SLOT = str(gpu_id)
    if gpu_id != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TQDM_DISABLE", "1")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "2")
    print(f"[worker pid={os.getpid()}] bound to GPU {gpu_id or '(cpu)'}",
          flush=True)


def _run_one_job(job_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Worker-side entry point: runs ``run_one`` for one (cond, seed)."""
    cond = job_kwargs["condition"]
    seed = job_kwargs["seed"]
    tag = f"[GPU{_WORKER_GPU_SLOT or 'cpu'}][{cond}/seed{seed}]"
    print(f"\n{tag} >>> start", flush=True)
    t0 = time.time()
    try:
        summary = run_one(
            condition=cond,
            seed=seed,
            phantom_root=job_kwargs["phantom_root"],
            method_root=job_kwargs["method_root"],
            cache_root=job_kwargs["cache_root"],
            methods_to_run=job_kwargs["methods_to_run"],
            force_precompute=job_kwargs.get("force_precompute", False),
            hidtw_tune=job_kwargs.get("hidtw_tune", True),
            hidtw_tune_step=job_kwargs.get("hidtw_tune_step", HIDTW_WEIGHT_GRID_STEP),
            hidtw_alpha_grid=job_kwargs.get("hidtw_alpha_grid", HIDTW_ALPHA_GRID),
        )
    except Exception as e:  # pragma: no cover - propagate but keep pool alive
        import traceback
        print(f"{tag} !!! FAILED: {e}\n{traceback.format_exc()}", flush=True)
        return {"condition": cond, "seed": seed, "error": str(e)}
    print(f"{tag} <<< done ({time.time()-t0:.1f}s)", flush=True)
    return summary


# --- 7. CLI ---

def main():
    parser = argparse.ArgumentParser(description="Simulation 1 method runner.")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="Single-GPU mode: pin the parent process to this "
                             "device before importing torch. Ignored when "
                             "--num-workers > 1 (workers self-pin instead).")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Dataset-level parallel workers. None / 0 "
                             "auto-detects torch.cuda.device_count(); 1 keeps "
                             "the legacy single-process loop; >1 spawns a "
                             "multiprocessing pool where each child pins "
                             "itself to a distinct GPU. Takes precedence over "
                             "--gpu-oversubscribe when set explicitly.")
    parser.add_argument("--gpu-oversubscribe", type=int, default=1,
                        help="Number of worker processes per physical GPU "
                             "(default 1). Used only when --num-workers is "
                             "unset. Each (cond, seed) job is fully self-"
                             "contained (writes to its own directory, reads "
                             "its own cache file) so oversubscribing is safe "
                             "and lets GPU DTW kernels overlap with CPU-bound "
                             "OT / shapely-IoU / HiDTW tuning from sibling "
                             "workers on the same card. 2-3 is typical.")
    parser.add_argument("--phantom-root", default="simulation01_results/phantoms")
    parser.add_argument("--method-root",  default="simulation01_results/methods")
    parser.add_argument("--cache-root",   default="simulation01_results/cache")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Subset of methods to run. Default: all {list(METHODS.keys())}")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-precompute", action="store_true")
    parser.add_argument("--no-hidtw-tune", action="store_true",
                        help="Skip per-dataset simplex search for HiDTW_full "
                             "(falls back to Tile046 BO weights).")
    parser.add_argument("--hidtw-tune-step", type=float, default=HIDTW_WEIGHT_GRID_STEP,
                        help="Simplex grid step for HiDTW_full weight tuning (default 0.1).")
    parser.add_argument("--hidtw-alpha-grid", nargs="+", type=float,
                        default=list(HIDTW_ALPHA_GRID),
                        help="Alpha (RM threshold) grid for HiDTW_full tuning.")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    if args.quick:
        cond_names = list(QUICK_CONDITIONS)
        seeds = [args.seed_start]
    else:
        cond_names = args.conditions if args.conditions else list(CONDITIONS.keys())
        if args.seeds is not None:
            seeds = list(args.seeds)
        else:
            seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    methods_to_run = args.methods if args.methods else list(METHODS.keys())

    # Decide single-process vs multi-worker layout.
    # --gpu-oversubscribe replicates the visible gpu list (default auto);
    # --num-workers overrides and round-robins workers across visible gpus.
    # Each (cond, seed) job is atomic (own phantom_dir / cache_file / out_base
    # for the duration of run_one), so oversubscription cannot race.
    base_ids = _resolve_gpu_ids(None)
    has_gpu = bool(base_ids) and base_ids != [""]
    oversub = max(1, int(args.gpu_oversubscribe))

    if args.num_workers is not None:
        num_workers = max(1, int(args.num_workers))
        if oversub != 1:
            print(f"[warn] --num-workers={num_workers} takes precedence over "
                  f"--gpu-oversubscribe={oversub}; the latter is ignored.")
        if has_gpu:
            # Round-robin: worker i -> base_ids[i % len(base_ids)]
            gpu_ids = [base_ids[i % len(base_ids)] for i in range(num_workers)]
        else:
            gpu_ids = [""] * num_workers
    else:
        if has_gpu:
            gpu_ids = list(base_ids) * oversub
        else:
            gpu_ids = [""] * oversub
        num_workers = len(gpu_ids)

    if args.gpu_id is not None and num_workers > 1:
        print(f"[warn] --gpu-id={args.gpu_id} is ignored because num_workers="
              f"{num_workers}; workers self-pin to {gpu_ids}.")

    gpu_worker_counts: Dict[str, int] = {}
    for g in gpu_ids:
        gpu_worker_counts[g] = gpu_worker_counts.get(g, 0) + 1

    print("Simulation 1 - method runner")
    print(f"  conditions:   {cond_names}")
    print(f"  seeds:        {seeds}")
    print(f"  methods:      {methods_to_run}")
    print(f"  phantom-root: {args.phantom_root}")
    print(f"  method-root:  {args.method_root}")
    print(f"  cache-root:   {args.cache_root}")
    print(f"  num-workers:  {num_workers}")
    print(f"  worker GPUs:  {gpu_ids}")
    if has_gpu:
        print(f"  workers/GPU:  "
              f"{ {g: gpu_worker_counts[g] for g in base_ids} }")

    valid_conds = [c for c in cond_names if c in CONDITIONS]
    for c in cond_names:
        if c not in CONDITIONS:
            print(f"[skip] unknown condition {c}")
    jobs: List[Dict[str, Any]] = []
    for cond in valid_conds:
        for seed in seeds:
            jobs.append({
                "condition": cond,
                "seed": int(seed),
                "phantom_root": args.phantom_root,
                "method_root": args.method_root,
                "cache_root": args.cache_root,
                "methods_to_run": methods_to_run,
                "force_precompute": args.force_precompute,
                "hidtw_tune": not args.no_hidtw_tune,
                "hidtw_tune_step": args.hidtw_tune_step,
                "hidtw_alpha_grid": list(args.hidtw_alpha_grid),
            })

    all_summaries: List[Dict[str, Any]] = []

    if num_workers <= 1:
        print(f"  CUDA available: {torch.cuda.is_available()}")
        for job in jobs:
            print(f"\n=== {job['condition']} / seed{job['seed']} ===")
            s = run_one(
                condition=job["condition"],
                seed=job["seed"],
                phantom_root=job["phantom_root"],
                method_root=job["method_root"],
                cache_root=job["cache_root"],
                methods_to_run=job["methods_to_run"],
                force_precompute=job["force_precompute"],
                hidtw_tune=job["hidtw_tune"],
                hidtw_tune_step=job["hidtw_tune_step"],
                hidtw_alpha_grid=job["hidtw_alpha_grid"],
            )
            all_summaries.append(s)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        gpu_queue = ctx.Queue()
        for g in gpu_ids:
            gpu_queue.put(g)
        print(f"  dispatching {len(jobs)} job(s) across {num_workers} worker(s) ...")
        with ctx.Pool(num_workers, initializer=_worker_init,
                      initargs=(gpu_queue,)) as pool:
            for summary in pool.imap_unordered(_run_one_job, jobs, chunksize=1):
                all_summaries.append(summary)

    os.makedirs(args.method_root, exist_ok=True)
    all_summaries.sort(key=lambda s: (str(s.get("condition", "")), int(s.get("seed", -1))))
    with open(os.path.join(args.method_root, "_runs_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll runs summary -> {os.path.join(args.method_root, '_runs_summary.json')}")


if __name__ == "__main__":
    main()
