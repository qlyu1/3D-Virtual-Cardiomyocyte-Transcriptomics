#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation 1 - Evaluation.

Consumes:
  simulation01_results/phantoms/<cond>/seed<k>/ground_truth.csv, phantoms.json, *_features.csv
  simulation01_results/methods/<cond>/seed<k>/<method>/tracks.csv

Computes (all using known 3D ground truth):

  PAIR-LEVEL:
    precision / recall / F1 of adjacent-layer cell-to-cell matches.
    id_switch_rate     : fraction of predicted pair links that switch GT identity.

  TRACK-LEVEL (Hungarian assignment on cell-overlap):
    purity             : max |pred ∩ gt| / |pred|
    completeness       : |pred ∩ gt_matched| / |gt_matched|
    fragmentation_rate : mean number of predicted tracks covering the same GT track
    coverage           : total cells assigned / total cells observed
    over_segmentation  : #pred / #gt tracks
    under_segmentation : 1 - purity (equivalent metric, reported separately)

  GEOMETRY:
    length_correlation   : Pearson r between predicted arc-length and GT length
    volume_correlation   : Pearson r between predicted stacked volume and GT volume
    length_rel_error     : mean |pred - gt| / gt
    volume_rel_error     : mean |pred - gt| / gt
    mean_layered_IoU     : mean per-layer 2D IoU between matched slices

Outputs:
  simulation01_results/evaluation/
    pair_metrics.csv
    track_metrics_per_track.csv
    track_metrics_per_dataset.csv
    geometry_metrics.csv
    summary.json
    fig_*.pdf  (+ fig_*.csv with source data)

Usage:
  python simulation01_evaluate.py
  python simulation01_evaluate.py --conditions L1_mid_clean L2_high_clean
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, wilcoxon

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
matplotlib.rcParams["savefig.pad_inches"] = 0.03
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation01_phantom_generator import (  # noqa: E402
    CONDITIONS, resolve_condition_config, SLICE_THICKNESS_UM, N_SLICES,
)
from simulation01_run_methods import METHODS as METHOD_SPECS  # noqa: E402

try:
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False

METHOD_ORDER = [
    "Centroid_NN",
    "DTW_only",
    "OT_only",
    "HiDTW_full",
]

OI = {
    "Centroid_NN":  "#999999",
    "DTW_only":     "#56B4E9",
    "OT_only":      "#009E73",
    "HiDTW_full":   "#D55E00",
}

# Reader-friendly aliases applied at aggregation time. The phantom generator
# and runner still use the raw L2_* keys; tables and figures use C##_*.
CONDITION_ALIAS: Dict[str, str] = {
    "L2_clean_sanity":         "C01_clean",
    "L2_mid_jitter":           "C02_mid_jitter",
    "L2_percell_jitter":       "C03_percell_jitter",
    "L2_clustered_dense":      "C04_clusters",
    "L2_shift_iid_small":      "C05_shift_small",
    "L2_shift_iid_mid":        "C06_shift_mid",
    "L2_shift_iid_high":       "C07_shift_high",
    "L2_shift_iid_extreme":    "C08_shift_extreme",
    "L2_clusters_shift":       "C09_clusters_shift",
    "L2_clusters_shift_high":  "C10_clusters_shift_high",
    "L2_shift_plus_jitter":    "C11_shift_mid_jitter",
    "L2_fibroblasts_noshift":  "C12_distractors_noshift",
    "L2_distractors_shift":    "C13_distractors_shift",
    "L2_stress_combo":         "C14_stress_combo",
    # DTW-adversarial per-slice shape perturbations (C15-C18).
    "L2_per_slice_warp":          "C15_slice_warp",
    "L2_per_slice_rotation":      "C16_slice_rotation",
    "L2_per_slice_contour_noise": "C17_contour_noise",
    "L2_dtw_stress_combo":        "C18_dtw_stress_combo",
}


def _alias(cond: str) -> str:
    """Return ``CONDITION_ALIAS[cond]`` if present, else ``cond``."""
    return CONDITION_ALIAS.get(cond, cond)


def _sort_conditions(values: List[str]) -> List[str]:
    """Sort aliased conditions (C01_... first), then any leftover names."""
    aliased = sorted([v for v in values if v.startswith("C") and "_" in v])
    others = sorted([v for v in values if not (v.startswith("C") and "_" in v)])
    return aliased + others


# --- 1. LOADERS ---

def load_ground_truth(phantom_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    gt_df = pd.read_csv(os.path.join(phantom_dir, "ground_truth.csv"))
    gt_df["slice"] = gt_df["slice"].astype(int)
    gt_df["gt_track_id"] = gt_df["gt_track_id"].astype(int)
    with open(os.path.join(phantom_dir, "phantoms.json"), encoding="utf-8") as f:
        phantoms = json.load(f)
    return gt_df, phantoms


def load_tracks(tracks_csv: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(tracks_csv):
        return []
    df = pd.read_csv(tracks_csv)
    out = []
    for _, row in df.iterrows():
        cids = [c.strip() for c in str(row["cell_ids"]).split("|") if c.strip()]
        out.append({
            "track_id": int(row["track_id"]),
            "cell_ids": cids,
            "length": int(row.get("track_length", len(cids))),
            "avg_margin": float(row.get("avg_margin", 0.0)),
        })
    return out


def load_features(phantom_dir: str) -> Dict[str, Dict[str, Any]]:
    """cell_id -> {slice, centroid (absolute), area, contour_abs (N,2 absolute)}."""
    out: Dict[str, Dict[str, Any]] = {}
    csvs = sorted(glob(os.path.join(phantom_dir, "*_S*_features.csv")))
    for fp in csvs:
        df = pd.read_csv(fp)
        for _, row in df.iterrows():
            cid = str(row["cell_id"])
            slice_num = int(str(row["slice"]).replace("S", "")) if isinstance(row["slice"], str) \
                else int(row["slice"])
            cx = float(row["centroid_x"]); cy = float(row["centroid_y"])
            xs: List[float] = []; ys: List[float] = []
            for k in range(400):
                x_col = f"x_{k}"; y_col = f"y_{k}"
                if x_col in row and not pd.isna(row[x_col]):
                    xs.append(float(row[x_col]))
                    ys.append(float(row[y_col]))
                else:
                    break
            rel = np.column_stack([np.asarray(xs), np.asarray(ys)])
            out[cid] = {
                "slice": slice_num,
                "centroid": np.array([cx, cy]),
                "area": float(row.get("area", 0.0)),
                "contour_abs": rel + np.array([cx, cy]),
            }
    return out


# --- 2. PAIR-LEVEL METRICS ---

def enumerate_pairs_gt(gt_df: pd.DataFrame) -> set:
    """Set of (slice_i, cell_i, slice_i+1, cell_j) GT pair tuples where same gt_track_id."""
    pairs = set()
    by_track = gt_df.groupby("gt_track_id")
    for gid, sub in by_track:
        sub = sub.sort_values("slice")
        rows = sub.to_dict("records")
        for i in range(len(rows) - 1):
            a = rows[i]; b = rows[i + 1]
            if int(b["slice"]) - int(a["slice"]) == 1:
                pairs.add((int(a["slice"]), a["cell_id"], int(b["slice"]), b["cell_id"]))
    return pairs


def enumerate_pairs_pred(tracks: List[Dict[str, Any]], cid_to_slice: Dict[str, int]) -> set:
    pairs = set()
    for t in tracks:
        cids = t["cell_ids"]
        ordered = sorted([(cid_to_slice.get(c, -1), c) for c in cids if c in cid_to_slice])
        for i in range(len(ordered) - 1):
            s1, c1 = ordered[i]; s2, c2 = ordered[i + 1]
            if s2 - s1 == 1:
                pairs.add((s1, c1, s2, c2))
    return pairs


def compute_pair_metrics(pred_pairs: set, gt_pairs: set,
                         gt_df: pd.DataFrame,
                         tracks: List[Dict[str, Any]],
                         cid_to_slice: Dict[str, int]) -> Dict[str, float]:
    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # ID-switch rate: for each predicted connection, does it switch GT track identity?
    cid_to_gt = dict(zip(gt_df["cell_id"].astype(str), gt_df["gt_track_id"].astype(int)))
    n_transitions = 0; n_switches = 0
    for t in tracks:
        ordered = sorted([(cid_to_slice.get(c, -1), c) for c in t["cell_ids"] if c in cid_to_slice])
        for i in range(len(ordered) - 1):
            _, c1 = ordered[i]; _, c2 = ordered[i + 1]
            g1 = cid_to_gt.get(c1, -1); g2 = cid_to_gt.get(c2, -1)
            if g1 < 0 or g2 < 0:
                continue
            n_transitions += 1
            if g1 != g2:
                n_switches += 1
    id_switch_rate = n_switches / n_transitions if n_transitions else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_pred_pairs": len(pred_pairs), "n_gt_pairs": len(gt_pairs),
        "id_switch_rate": id_switch_rate,
        "n_transitions_checked": n_transitions,
    }


# --- 3. TRACK-LEVEL HUNGARIAN ASSIGNMENT ---

def hungarian_match_tracks(pred_tracks: List[Dict[str, Any]],
                           gt_groups: Dict[int, List[str]]) -> Dict[int, int]:
    """Assign each predicted track to a GT track by maximum cell-id overlap.

    Returns: pred_track_idx -> gt_track_id (or -1 if unmatched)
    """
    gt_ids = sorted(gt_groups.keys())
    if not pred_tracks or not gt_ids:
        return {}
    n_pred = len(pred_tracks); n_gt = len(gt_ids)
    # Square the matrix for linear_sum_assignment
    n = max(n_pred, n_gt)
    cost = np.full((n, n), 1e6, dtype=np.float64)
    for i, t in enumerate(pred_tracks):
        pred_set = set(t["cell_ids"])
        for j, g in enumerate(gt_ids):
            gt_set = set(gt_groups[g])
            overlap = len(pred_set & gt_set)
            # negative overlap as cost (so higher overlap = lower cost)
            cost[i, j] = -overlap
    row, col = linear_sum_assignment(cost)
    mapping: Dict[int, int] = {}
    for i, j in zip(row, col):
        if i < n_pred and j < n_gt:
            if cost[i, j] < 0:   # had some overlap
                mapping[i] = gt_ids[j]
            else:
                mapping[i] = -1
    return mapping


def compute_track_metrics(pred_tracks: List[Dict[str, Any]],
                          gt_groups: Dict[int, List[str]],
                          n_total_cells: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    mapping = hungarian_match_tracks(pred_tracks, gt_groups)

    per_track_rows = []
    for i, t in enumerate(pred_tracks):
        pred_set = set(t["cell_ids"])
        gt_id = mapping.get(i, -1)
        gt_set = set(gt_groups.get(gt_id, [])) if gt_id >= 0 else set()
        inter = len(pred_set & gt_set)
        purity = inter / len(pred_set) if pred_set else 0.0
        completeness = inter / len(gt_set) if gt_set else 0.0
        per_track_rows.append({
            "pred_track_idx": i,
            "pred_length": len(pred_set),
            "matched_gt_id": int(gt_id),
            "gt_length": len(gt_set),
            "overlap": inter,
            "purity": purity,
            "completeness": completeness,
        })
    per_track_df = pd.DataFrame(per_track_rows)

    # Fragmentation: average number of predicted tracks covering a given GT track.
    gt_cover_counts = defaultdict(int)
    for i, t in enumerate(pred_tracks):
        seen_gt_ids = set()
        for cid in t["cell_ids"]:
            # find which GT does this cell belong to
            for gid, ids in gt_groups.items():
                if cid in ids:
                    seen_gt_ids.add(gid)
                    break
        for gid in seen_gt_ids:
            gt_cover_counts[gid] += 1
    fragmentation = (float(np.mean(list(gt_cover_counts.values())))
                     if gt_cover_counts else 0.0)

    # Coverage: cells assigned by any predicted track (one cell may appear in one track only
    # under our greedy conflict resolver; intersect with total observed cell count).
    assigned_cells = set()
    for t in pred_tracks:
        assigned_cells.update(t["cell_ids"])
    coverage = len(assigned_cells) / n_total_cells if n_total_cells else 0.0

    summary = {
        "n_pred_tracks": int(len(pred_tracks)),
        "n_gt_tracks": int(len(gt_groups)),
        "mean_purity": float(per_track_df["purity"].mean()) if len(per_track_df) else 0.0,
        "median_purity": float(per_track_df["purity"].median()) if len(per_track_df) else 0.0,
        "mean_completeness": float(per_track_df["completeness"].mean()) if len(per_track_df) else 0.0,
        "fragmentation_rate": float(fragmentation),
        "coverage": float(coverage),
        "over_segmentation_ratio": (len(pred_tracks) / len(gt_groups)) if gt_groups else 0.0,
    }
    return per_track_df, summary


# --- 4. GEOMETRY METRICS (length, volume, layered IoU) ---

def predicted_track_arc_length(track: Dict[str, Any],
                               feat: Dict[str, Dict[str, Any]]) -> float:
    """Arc-length of centroid sequence through z-slabs. Uses SLICE_THICKNESS_UM as Δz."""
    ordered = []
    for cid in track["cell_ids"]:
        if cid in feat:
            ordered.append((feat[cid]["slice"], feat[cid]["centroid"]))
    ordered.sort()
    if len(ordered) < 2:
        return 0.0
    tot = 0.0
    for i in range(len(ordered) - 1):
        c1 = ordered[i][1]; c2 = ordered[i + 1][1]
        dz = SLICE_THICKNESS_UM
        tot += float(np.sqrt(np.sum((c2 - c1) ** 2) + dz * dz))
    # Add end-cap contributions (first and last half-slab).
    return tot + SLICE_THICKNESS_UM


def predicted_track_volume(track: Dict[str, Any],
                           feat: Dict[str, Dict[str, Any]]) -> float:
    """Sum of slice areas times slice thickness."""
    tot = 0.0
    for cid in track["cell_ids"]:
        if cid in feat:
            tot += feat[cid]["area"] * SLICE_THICKNESS_UM
    return float(tot)


def _poly_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    if not HAS_SHAPELY:
        return 0.0
    pa = Polygon(poly_a); pb = Polygon(poly_b)
    if not pa.is_valid:
        pa = pa.buffer(0)
    if not pb.is_valid:
        pb = pb.buffer(0)
    try:
        inter = pa.intersection(pb).area
    except Exception:
        return 0.0
    union = pa.area + pb.area - inter
    return inter / union if union > 1e-9 else 0.0


def compute_geometry_metrics(pred_tracks: List[Dict[str, Any]],
                             gt_groups: Dict[int, List[str]],
                             mapping: Dict[int, int],
                             feat: Dict[str, Dict[str, Any]],
                             phantom_params: Dict[int, Dict[str, Any]]
                             ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Per-track length/volume/layered-IoU metrics, using Hungarian mapping."""
    rows = []
    for i, t in enumerate(pred_tracks):
        gid = mapping.get(i, -1)
        pred_len = predicted_track_arc_length(t, feat)
        pred_vol = predicted_track_volume(t, feat)
        if gid < 0 or gid not in phantom_params:
            rows.append({"pred_track_idx": i, "matched_gt_id": int(gid),
                         "pred_length_um": pred_len, "pred_volume_um3": pred_vol,
                         "gt_length_um": np.nan, "gt_volume_um3": np.nan,
                         "length_abs_err": np.nan, "volume_abs_err": np.nan,
                         "mean_layered_iou": np.nan, "n_layers_iou": 0})
            continue

        gt_len = float(phantom_params[gid]["length_um"])
        gt_vol = float(phantom_params[gid]["volume_um3"])

        # Layered IoU (only over slices present in BOTH pred track and GT group with 1:1 cell pairing)
        pred_slice_to_cid: Dict[int, str] = {}
        for cid in t["cell_ids"]:
            if cid in feat:
                pred_slice_to_cid[feat[cid]["slice"]] = cid
        gt_slice_to_cid: Dict[int, str] = {}
        for cid in gt_groups[gid]:
            if cid in feat:
                gt_slice_to_cid[feat[cid]["slice"]] = cid
        common = sorted(set(pred_slice_to_cid.keys()) & set(gt_slice_to_cid.keys()))
        ious = []
        for s in common:
            p_cid = pred_slice_to_cid[s]; g_cid = gt_slice_to_cid[s]
            if p_cid == g_cid:
                ious.append(1.0); continue
            pa = feat[p_cid]["contour_abs"]; pb = feat[g_cid]["contour_abs"]
            ious.append(_poly_iou(pa, pb))
        mean_iou = float(np.mean(ious)) if ious else 0.0

        rows.append({
            "pred_track_idx": i,
            "matched_gt_id": int(gid),
            "pred_length_um": pred_len,
            "pred_volume_um3": pred_vol,
            "gt_length_um": gt_len,
            "gt_volume_um3": gt_vol,
            "length_abs_err": abs(pred_len - gt_len),
            "volume_abs_err": abs(pred_vol - gt_vol),
            "mean_layered_iou": mean_iou,
            "n_layers_iou": len(ious),
        })

    df = pd.DataFrame(rows)
    df_ok = df.dropna(subset=["gt_length_um", "gt_volume_um3"])

    def _corr(a: pd.Series, b: pd.Series) -> float:
        if len(a) < 2:
            return 0.0
        try:
            r, _ = pearsonr(a, b)
            return float(r) if np.isfinite(r) else 0.0
        except Exception:
            return 0.0

    summary = {
        "length_pearson_r": _corr(df_ok["pred_length_um"], df_ok["gt_length_um"]) if len(df_ok) else 0.0,
        "volume_pearson_r": _corr(df_ok["pred_volume_um3"], df_ok["gt_volume_um3"]) if len(df_ok) else 0.0,
        "length_mean_rel_err": (float((df_ok["length_abs_err"] / df_ok["gt_length_um"]).mean())
                                if len(df_ok) else 0.0),
        "volume_mean_rel_err": (float((df_ok["volume_abs_err"] / df_ok["gt_volume_um3"]).mean())
                                if len(df_ok) else 0.0),
        "mean_layered_iou": float(df_ok["mean_layered_iou"].mean()) if len(df_ok) else 0.0,
        "n_matched": int(len(df_ok)),
    }
    return df, summary


# --- 5. PER-DATASET + PER-METHOD EVALUATION ---

def evaluate_one(condition: str, seed: int, phantom_root: str, method_root: str,
                 methods: List[str]) -> Dict[str, Any]:
    phantom_dir = os.path.join(phantom_root, condition, f"seed{seed}")
    method_dir_base = os.path.join(method_root, condition, f"seed{seed}")

    gt_df, phantoms = load_ground_truth(phantom_dir)
    feat = load_features(phantom_dir)
    phantom_params = {int(p["gt_id"]): p for p in phantoms["phantoms"]}
    gt_groups: Dict[int, List[str]] = defaultdict(list)
    for _, row in gt_df.iterrows():
        gt_groups[int(row["gt_track_id"])].append(str(row["cell_id"]))
    cid_to_slice = dict(zip(gt_df["cell_id"].astype(str), gt_df["slice"].astype(int)))
    n_total_cells = len(gt_df)
    gt_pairs = enumerate_pairs_gt(gt_df)

    result: Dict[str, Any] = {
        "condition": condition, "seed": int(seed),
        "n_gt_cells": int(n_total_cells),
        "n_gt_tracks": int(len(gt_groups)),
        "n_gt_pairs": int(len(gt_pairs)),
        "methods": {},
    }

    for mname in methods:
        tracks_csv = os.path.join(method_dir_base, mname, "tracks.csv")
        tracks = load_tracks(tracks_csv)
        pred_pairs = enumerate_pairs_pred(tracks, cid_to_slice)
        pair_m = compute_pair_metrics(pred_pairs, gt_pairs, gt_df, tracks, cid_to_slice)

        per_track_df, track_summary = compute_track_metrics(tracks, gt_groups, n_total_cells)
        mapping = hungarian_match_tracks(tracks, gt_groups)
        geom_df, geom_summary = compute_geometry_metrics(tracks, gt_groups, mapping,
                                                         feat, phantom_params)

        result["methods"][mname] = {
            "pair": pair_m,
            "track": track_summary,
            "geometry": geom_summary,
            "per_track_detail": per_track_df.to_dict("records"),
            "per_track_geometry": geom_df.to_dict("records"),
        }
    return result


# --- 6. AGGREGATION -> CSV TABLES ---

def aggregate_to_tables(per_dataset: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    pair_rows: List[Dict[str, Any]] = []
    track_rows: List[Dict[str, Any]] = []
    geom_rows: List[Dict[str, Any]] = []
    per_track_rows: List[Dict[str, Any]] = []
    per_track_geom_rows: List[Dict[str, Any]] = []

    for d in per_dataset:
        # Rename once at aggregation so all downstream CSVs + plots carry the
        # reader-friendly C##_… label while upstream code keeps using the raw
        # generator keys.
        cond = _alias(d["condition"]); seed = d["seed"]
        for mname, m in d["methods"].items():
            pm = m["pair"]
            pair_rows.append({
                "condition": cond, "seed": seed, "method": mname,
                **pm,
            })
            tr = m["track"]
            track_rows.append({"condition": cond, "seed": seed, "method": mname, **tr})
            gm = m["geometry"]
            geom_rows.append({"condition": cond, "seed": seed, "method": mname, **gm})
            for rec in m["per_track_detail"]:
                per_track_rows.append({"condition": cond, "seed": seed, "method": mname, **rec})
            for rec in m["per_track_geometry"]:
                per_track_geom_rows.append({"condition": cond, "seed": seed, "method": mname, **rec})

    return {
        "pair_metrics": pd.DataFrame(pair_rows),
        "track_metrics_per_dataset": pd.DataFrame(track_rows),
        "geometry_metrics": pd.DataFrame(geom_rows),
        "track_metrics_per_track": pd.DataFrame(per_track_rows),
        "geometry_per_track": pd.DataFrame(per_track_geom_rows),
    }


# --- 7. FIGURES ---

def _savefig(path: str):
    plt.savefig(path, bbox_inches="tight", transparent=False)
    plt.close()
    print(f"    -> {path}")


def _method_stats(df: pd.DataFrame, metric: str) -> Dict[str, Tuple[float, float]]:
    """Return method -> (mean, std) across rows, in METHOD_ORDER."""
    out: Dict[str, Tuple[float, float]] = {}
    for m in METHOD_ORDER:
        sub = df[df["method"] == m]
        if not len(sub) or metric not in sub.columns:
            continue
        vals = sub[metric].astype(float)
        out[m] = (float(vals.mean()), float(vals.std(ddof=0)))
    return out


def fig_bar_by_method(stats: Dict[str, Tuple[float, float]], ylabel: str,
                      title: str, out_pdf: str, ylim: Optional[Tuple[float, float]] = None):
    if not stats:
        return
    fig, ax = plt.subplots(figsize=(92 / 25.4, 62 / 25.4))
    labels = list(stats.keys())
    means = [stats[m][0] for m in labels]
    stds = [stats[m][1] for m in labels]
    colors = [OI.get(m, "#333333") for m in labels]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=2.5, color=colors,
           edgecolor="0.15", linewidth=0.5, alpha=0.9,
           error_kw={"linewidth": 0.6})
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _savefig(out_pdf)
    pd.DataFrame([{"method": m, "mean": stats[m][0], "std": stats[m][1]} for m in labels]
                 ).to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def _fig_metric_vs_condition(
    pair_df: pd.DataFrame,
    out_pdf: str,
    metric: str,
    ylabel: str,
    title: str,
    zoom_out_pdf: Optional[str] = None,
) -> None:
    """Generic "metric × condition" grouped-bar chart with an optional
    auto-zoomed sibling. Used for both F1 and recall via thin wrappers."""
    if pair_df.empty or metric not in pair_df.columns:
        return
    conds = _sort_conditions(list(pair_df["condition"].unique()))
    n_m = len(METHOD_ORDER)
    if not conds or n_m == 0:
        return

    means_grid = np.full((n_m, len(conds)), np.nan)
    stds_grid = np.full((n_m, len(conds)), np.nan)
    ns_grid = np.zeros((n_m, len(conds)), dtype=int)
    for k, m in enumerate(METHOD_ORDER):
        sub = pair_df[pair_df["method"] == m]
        if sub.empty:
            continue
        for j, c in enumerate(conds):
            v = sub[sub["condition"] == c][metric].astype(float)
            if len(v):
                means_grid[k, j] = float(v.mean())
                stds_grid[k, j] = float(v.std(ddof=0))
                ns_grid[k, j] = int(len(v))

    def _render(ylim: Tuple[float, float], path: str, zoom: bool) -> None:
        fig_w = max(120, 12 * len(conds) + 30)
        fig_h = 82 if zoom else 70
        fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
        x = np.arange(len(conds))
        width = 0.8 / max(n_m, 1)
        for k, m in enumerate(METHOD_ORDER):
            means = means_grid[k]
            stds = stds_grid[k]
            if np.all(np.isnan(means)):
                continue
            ax.bar(x + (k - (n_m - 1) / 2) * width, means, width, yerr=stds,
                   capsize=2.6 if zoom else 1.6,
                   color=OI.get(m, "#333333"),
                   edgecolor="0.15",
                   linewidth=0.7 if zoom else 0.4,
                   label=m,
                   error_kw={"linewidth": 0.9 if zoom else 0.5,
                             "capthick": 0.9 if zoom else 0.5})
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=35, ha="right",
                           fontsize=9 if zoom else 7)
        ax.set_ylabel(ylabel, fontsize=11 if zoom else 8)
        title_tail = (f"  -  y-axis zoomed to [{ylim[0]:.2f}, {ylim[1]:.2f}]"
                      if zoom else "")
        ax.set_title(title + title_tail,
                     fontsize=12 if zoom else 9,
                     fontweight="semibold" if zoom else "normal")
        ax.tick_params(labelsize=10 if zoom else 7,
                       length=5 if zoom else 3,
                       width=1.0 if zoom else 0.8)
        ax.legend(fontsize=10 if zoom else 6,
                  ncol=min(n_m, 4), frameon=False,
                  loc="upper center",
                  bbox_to_anchor=(0.5, -0.28 if zoom else -0.32))
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="0.85" if zoom else "0.9",
                lw=0.7 if zoom else 0.4)
        ax.set_axisbelow(True)
        if zoom:
            ax.text(0.0, 1.018,
                    f"y-axis truncated · baseline = {ylim[0]:.2f}, not 0",
                    transform=ax.transAxes,
                    fontsize=10, color="0.25",
                    ha="left", va="bottom", style="italic")
            for sp in ("left", "bottom"):
                ax.spines[sp].set_linewidth(1.1)
                ax.spines[sp].set_color("0.15")
        _savefig(path)

    _render((0.0, 1.05), out_pdf, zoom=False)

    if zoom_out_pdf is not None:
        lo_mat = means_grid - stds_grid
        hi_mat = means_grid + stds_grid
        lo_finite = lo_mat[np.isfinite(lo_mat)]
        hi_finite = hi_mat[np.isfinite(hi_mat)]
        if lo_finite.size and hi_finite.size:
            y_lo_raw = max(0.0, float(np.min(lo_finite)) - 0.02)
            y_hi_raw = min(1.03, float(np.max(hi_finite)) + 0.02)
            y_lo = float(np.floor(y_lo_raw * 20) / 20.0)
            y_hi = float(np.ceil(y_hi_raw * 20) / 20.0)
            if y_hi - y_lo < 0.05:
                y_hi = min(1.0, y_lo + 0.1)
            if y_hi - y_lo > 0.95:
                y_hi = 1.03
            _render((y_lo, y_hi), zoom_out_pdf, zoom=True)

    rows: List[Dict[str, Any]] = []
    for k, m in enumerate(METHOD_ORDER):
        for j, c in enumerate(conds):
            rows.append({
                "method": m, "condition": c,
                f"mean_{metric}": (float(means_grid[k, j])
                                   if np.isfinite(means_grid[k, j]) else np.nan),
                f"std_{metric}": (float(stds_grid[k, j])
                                  if np.isfinite(stds_grid[k, j]) else np.nan),
                "n": int(ns_grid[k, j]),
            })
    pd.DataFrame(rows).to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def fig_f1_vs_condition(pair_df: pd.DataFrame, out_pdf: str,
                        zoom_out_pdf: Optional[str] = None) -> None:
    """Pair-matching accuracy (F1) per condition, one coloured bar per method.
    See ``_fig_metric_vs_condition`` for layout details and the zoom policy."""
    _fig_metric_vs_condition(
        pair_df, out_pdf,
        metric="f1",
        ylabel="Pair accuracy (F1)",
        title="Pair-matching accuracy across conditions",
        zoom_out_pdf=zoom_out_pdf,
    )


def fig_recall_vs_condition(pair_df: pd.DataFrame, out_pdf: str,
                            zoom_out_pdf: Optional[str] = None) -> None:
    """Pair **matching accuracy in the literal sense** per condition:
    ``recall = TP / (TP + FN)`` = fraction of all ground-truth adjacent-layer
    matches that the method recovered (irrespective of extra false matches,
    which are captured by precision). Same layout as ``fig_f1_vs_condition``
    so the two PDFs can be placed side-by-side in a composite panel."""
    _fig_metric_vs_condition(
        pair_df, out_pdf,
        metric="recall",
        ylabel="Pair matching accuracy  (recall = TP / total GT pairs)",
        title="Pair matching accuracy (recall) across conditions",
        zoom_out_pdf=zoom_out_pdf,
    )


def fig_precision_recall_scatter(pair_df: pd.DataFrame, out_pdf: str):
    if pair_df.empty:
        return
    fig, ax = plt.subplots(figsize=(80 / 25.4, 72 / 25.4))
    for m in METHOD_ORDER:
        sub = pair_df[pair_df["method"] == m]
        if sub.empty:
            continue
        ax.scatter(sub["recall"], sub["precision"], s=28,
                   color=OI.get(m, "#333333"), edgecolors="0.15",
                   linewidths=0.4, alpha=0.85, label=m)
    # diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall", fontsize=8); ax.set_ylabel("Precision", fontsize=8)
    ax.set_title("Precision-Recall per dataset", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower left", frameon=True,
              edgecolor="0.7", ncol=2)
    _savefig(out_pdf)
    pair_df[["condition", "seed", "method", "precision", "recall", "f1"]].to_csv(
        out_pdf.replace(".pdf", ".csv"), index=False)


def fig_violin_per_track(per_track_df: pd.DataFrame, metric: str,
                         ylabel: str, title: str, out_pdf: str):
    if per_track_df.empty:
        return
    fig, ax = plt.subplots(figsize=(100 / 25.4, 66 / 25.4))
    data = []; labels = []; colors = []
    for m in METHOD_ORDER:
        sub = per_track_df[per_track_df["method"] == m][metric].dropna()
        if not len(sub):
            continue
        data.append(sub.astype(float).values)
        labels.append(m)
        colors.append(OI.get(m, "#333333"))
    if not data:
        plt.close()
        return
    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col); pc.set_alpha(0.7); pc.set_edgecolor("0.15"); pc.set_linewidth(0.5)
    if "cmeans" in parts:
        parts["cmeans"].set_color("black"); parts["cmeans"].set_linewidth(0.8)
    if "cmedians" in parts:
        parts["cmedians"].set_color("0.2"); parts["cmedians"].set_linewidth(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8); ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    _savefig(out_pdf)
    per_track_df[["condition", "seed", "method", metric]].dropna().to_csv(
        out_pdf.replace(".pdf", ".csv"), index=False)


def fig_length_volume_scatter(per_track_geom_df: pd.DataFrame, kind: str, out_pdf: str,
                              methods: Optional[List[str]] = None):
    """kind ∈ {'length','volume'}: scatter pred vs gt across datasets."""
    if per_track_geom_df.empty:
        return
    x_col = "gt_length_um" if kind == "length" else "gt_volume_um3"
    y_col = "pred_length_um" if kind == "length" else "pred_volume_um3"
    title = ("Predicted vs true cell length" if kind == "length"
             else "Predicted vs true cell volume")
    axis_label = "length (um)" if kind == "length" else "volume (um^3)"

    use_methods = methods if methods is not None else METHOD_ORDER
    fig, ax = plt.subplots(figsize=(84 / 25.4, 78 / 25.4))
    xmax_vals = []
    for m in use_methods:
        sub = per_track_geom_df[(per_track_geom_df["method"] == m)
                                & per_track_geom_df[x_col].notna()]
        if not len(sub):
            continue
        xs = sub[x_col].astype(float).values
        ys = sub[y_col].astype(float).values
        ax.scatter(xs, ys, s=14, color=OI.get(m, "#333333"),
                   edgecolors="0.15", linewidths=0.25, alpha=0.55, label=m)
        xmax_vals.append(max(xs.max(), ys.max()))
    if not xmax_vals:
        plt.close()
        return
    lim = max(xmax_vals) * 1.08
    ax.plot([0, lim], [0, lim], "k--", lw=0.6, alpha=0.4, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(f"True {axis_label}", fontsize=8)
    ax.set_ylabel(f"Predicted {axis_label}", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=6, loc="upper left", frameon=True, edgecolor="0.7")
    _savefig(out_pdf)
    cols = ["condition", "seed", "method", x_col, y_col]
    per_track_geom_df[cols].dropna().to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def fig_metric_heatmap(pair_df: pd.DataFrame, track_df: pd.DataFrame,
                       geom_df: pd.DataFrame, out_pdf: str,
                       title: str = "Method × metric summary (mean over conditions × seeds)"):
    if pair_df.empty:
        return
    metrics = [
        ("Pair F1",            pair_df,  "f1",                   False),
        ("Pair precision",     pair_df,  "precision",            False),
        ("Pair recall",        pair_df,  "recall",               False),
        ("ID-switch rate",     pair_df,  "id_switch_rate",       True),
        ("Track purity",       track_df, "mean_purity",          False),
        ("Track completeness", track_df, "mean_completeness",    False),
        ("Fragmentation",      track_df, "fragmentation_rate",   True),
        ("Coverage",           track_df, "coverage",             False),
        ("Length r (Pearson)", geom_df,  "length_pearson_r",     False),
        ("Volume r (Pearson)", geom_df,  "volume_pearson_r",     False),
        ("Mean layered IoU",   geom_df,  "mean_layered_iou",     False),
    ]
    M = np.full((len(metrics), len(METHOD_ORDER)), np.nan)
    raw = M.copy()
    for i, (_, df, col, inv) in enumerate(metrics):
        if col not in df.columns:
            continue
        for j, m in enumerate(METHOD_ORDER):
            sub = df[df["method"] == m][col].astype(float)
            if len(sub):
                raw[i, j] = float(sub.mean())
                M[i, j] = float(sub.mean())

    # Normalise each row to [0,1] for coloring (invert where "lower = better")
    M_norm = np.zeros_like(M)
    for i in range(M.shape[0]):
        row = M[i]
        if np.all(np.isnan(row)):
            continue
        lo = np.nanmin(row); hi = np.nanmax(row)
        span = hi - lo if (hi - lo) > 1e-9 else 1.0
        norm = (row - lo) / span
        if metrics[i][3]:
            norm = 1.0 - norm
        M_norm[i] = norm

    fig, ax = plt.subplots(figsize=(115 / 25.4, 88 / 25.4))
    im = ax.imshow(M_norm, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[0] for m in metrics], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                txt = "--"
            elif abs(M[i, j]) >= 100:
                txt = f"{M[i, j]:.0f}"
            elif abs(M[i, j]) >= 1:
                txt = f"{M[i, j]:.2f}"
            else:
                txt = f"{M[i, j]:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="white" if M_norm[i, j] > 0.6 else "0.15")
    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Normalised score (1 = best)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9)
    _savefig(out_pdf)
    # CSV
    rows = []
    for i, (name, _df, col, inv) in enumerate(metrics):
        for j, m in enumerate(METHOD_ORDER):
            rows.append({"metric": name, "method": m,
                         "mean_value": raw[i, j], "lower_is_better": inv})
    pd.DataFrame(rows).to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def fig_f1_vs_density_or_curv(pair_df: pd.DataFrame, group_by: str, out_pdf: str,
                              xlabel: str):
    """F1 grouped by density or curvature category from CONDITIONS."""
    if pair_df.empty:
        return
    # Map condition -> group key (use resolved config so override-only entries
    # work). Aliases are applied so the joined DataFrame filter matches.
    groups: Dict[str, List[str]] = defaultdict(list)
    for cname in CONDITIONS:
        cfg = resolve_condition_config(cname)
        gkey = cfg["density"] if group_by == "density" else cfg.get("curv", "none")
        groups[gkey].append(_alias(cname))
    g_order = list(groups.keys())
    if group_by == "density":
        g_order = [g for g in ["sparse", "mid", "dense"] if g in groups] + \
                  [g for g in groups if g not in ("sparse", "mid", "dense")]
    elif group_by == "curv":
        g_order = [g for g in ["none", "low", "high"] if g in groups] + \
                  [g for g in groups if g not in ("none", "low", "high")]

    fig, ax = plt.subplots(figsize=(100 / 25.4, 66 / 25.4))
    x = np.arange(len(g_order))
    n_m = len(METHOD_ORDER)
    width = 0.8 / max(n_m, 1)
    for k, m in enumerate(METHOD_ORDER):
        means = []; sems = []
        for g in g_order:
            conds = groups[g]
            sub = pair_df[(pair_df["method"] == m) & (pair_df["condition"].isin(conds))]["f1"]
            means.append(float(sub.mean()) if len(sub) else np.nan)
            sems.append(float(sub.std(ddof=0)) if len(sub) else np.nan)
        ax.bar(x + (k - (n_m - 1) / 2) * width, means, width, yerr=sems,
               capsize=1.6, color=OI.get(m, "#333"),
               edgecolor="0.15", linewidth=0.4, label=m,
               error_kw={"linewidth": 0.5})
    ax.set_xticks(x); ax.set_xticklabels(g_order, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel("Pair F1", fontsize=8)
    ax.set_title(f"Pair F1 by {xlabel}", fontsize=9)
    ax.set_ylim(0, 1.05); ax.tick_params(labelsize=7)
    ax.grid(axis="y", color="0.9", lw=0.4); ax.set_axisbelow(True)
    ax.legend(fontsize=6, ncol=min(n_m, 4), frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.2))
    _savefig(out_pdf)
    rows = []
    for m in METHOD_ORDER:
        for g in g_order:
            conds = groups[g]
            sub = pair_df[(pair_df["method"] == m) & (pair_df["condition"].isin(conds))]["f1"]
            rows.append({"method": m, group_by: g,
                         "mean_f1": float(sub.mean()) if len(sub) else np.nan,
                         "std_f1": float(sub.std(ddof=0)) if len(sub) else np.nan,
                         "n": int(len(sub))})
    pd.DataFrame(rows).to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def fig_f1_L1_vs_L2(pair_df: pd.DataFrame, out_pdf: str):
    if pair_df.empty:
        return
    # Assign level from ORIGINAL (pre-alias) condition name prefix by reversing
    # the alias map; aliased names all start with "C" and lose the L1/L2 info.
    alias_to_raw = {v: k for k, v in CONDITION_ALIAS.items()}

    def _level(c: str) -> str:
        raw = alias_to_raw.get(c, c)
        if raw.startswith("L1"):
            return "Level 1"
        if raw.startswith("L2"):
            return "Level 2"
        return "Other"

    pair_df = pair_df.copy()
    pair_df["level"] = pair_df["condition"].apply(_level)
    levels = [lv for lv in ["Level 1", "Level 2", "Other"]
              if lv in set(pair_df["level"].unique())]
    if len(levels) < 2:
        # Only a single level is represented in the evaluation; the chart adds
        # no information, skip instead of emitting a trivial / misleading PDF.
        print(f"    [skip] {out_pdf}: only one level present ({levels}).")
        return

    fig, ax = plt.subplots(figsize=(92 / 25.4, 66 / 25.4))
    x = np.arange(len(levels))
    n_m = len(METHOD_ORDER)
    width = 0.8 / max(n_m, 1)
    for k, m in enumerate(METHOD_ORDER):
        means = []; sems = []
        for lv in levels:
            sub = pair_df[(pair_df["method"] == m) & (pair_df["level"] == lv)]["f1"]
            means.append(float(sub.mean()) if len(sub) else 0.0)
            sems.append(float(sub.std(ddof=0)) if len(sub) else 0.0)
        ax.bar(x + (k - (n_m - 1) / 2) * width, means, width, yerr=sems,
               capsize=2, color=OI.get(m, "#333"),
               edgecolor="0.15", linewidth=0.4, label=m,
               error_kw={"linewidth": 0.5})
    ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=8)
    ax.set_ylabel("Pair F1", fontsize=8)
    ax.set_title("Level 1 (ideal capsule) vs Level 2 (curved / tapered / twisted)", fontsize=9)
    ax.set_ylim(0, 1.05); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, ncol=3, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _savefig(out_pdf)
    rows = []
    for m in METHOD_ORDER:
        for lv in levels:
            sub = pair_df[(pair_df["method"] == m) & (pair_df["level"] == lv)]["f1"]
            rows.append({"method": m, "level": lv,
                         "mean_f1": float(sub.mean()) if len(sub) else np.nan,
                         "std_f1": float(sub.std(ddof=0)) if len(sub) else np.nan})
    pd.DataFrame(rows).to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


# --- 7b. PAIRED STATISTICAL COMPARISON (HiDTW vs per-dataset baselines) ---

# Direction of improvement: for id_switch_rate lower is better, for all others
# higher is better. We report signed ``delta = hidtw - baseline`` and a boolean
# ``hidtw_better`` that normalises direction so ``hidtw_better=True`` means the
# delta favours HiDTW regardless of metric orientation.
METRIC_DIRECTION_HIGHER_IS_BETTER = {
    "f1": True,
    "id_switch_rate": False,
    "mean_layered_iou": True,
}


def _metric_tables_for_paired(pair_df: pd.DataFrame, geom_df: pd.DataFrame
                              ) -> Dict[str, pd.DataFrame]:
    tables = {}
    if not pair_df.empty:
        tables["f1"] = pair_df[["condition", "seed", "method", "f1"]].rename(columns={"f1": "value"})
        tables["id_switch_rate"] = pair_df[["condition", "seed", "method", "id_switch_rate"]].rename(
            columns={"id_switch_rate": "value"}
        )
    if not geom_df.empty and "mean_layered_iou" in geom_df.columns:
        tables["mean_layered_iou"] = geom_df[["condition", "seed", "method", "mean_layered_iou"]].rename(
            columns={"mean_layered_iou": "value"}
        )
    return tables


def compute_paired_deltas(pair_df: pd.DataFrame, geom_df: pd.DataFrame,
                          target: str = "HiDTW_full",
                          baselines: Optional[List[str]] = None) -> pd.DataFrame:
    """Long-format paired delta table.

    One row per (condition, seed, baseline, metric). ``delta = value[target] - value[baseline]``;
    ``hidtw_better`` normalises for metrics where lower-is-better.
    """
    tables = _metric_tables_for_paired(pair_df, geom_df)
    rows: List[Dict[str, Any]] = []
    for metric, df in tables.items():
        if target not in set(df["method"].unique()):
            continue
        pivot = df.pivot_table(index=["condition", "seed"], columns="method",
                               values="value", aggfunc="first")
        pool = [m for m in pivot.columns if m != target]
        if baselines is not None:
            pool = [m for m in pool if m in baselines]
        higher_is_better = METRIC_DIRECTION_HIGHER_IS_BETTER.get(metric, True)
        for baseline in pool:
            for (cond, seed), r in pivot.iterrows():
                t_val = r.get(target, np.nan)
                b_val = r.get(baseline, np.nan)
                if not (np.isfinite(t_val) and np.isfinite(b_val)):
                    continue
                delta = float(t_val - b_val)
                rows.append({
                    "condition": str(cond),
                    "seed": int(seed),
                    "target": target,
                    "baseline": baseline,
                    "metric": metric,
                    "value_target": float(t_val),
                    "value_baseline": float(b_val),
                    "delta": delta,
                    "hidtw_better": bool((delta > 0) if higher_is_better else (delta < 0)),
                    "direction_higher_is_better": bool(higher_is_better),
                })
    return pd.DataFrame(rows)


def _wilcoxon_pval(deltas: np.ndarray, higher_is_better: bool) -> Tuple[float, float]:
    """Return (W, p_two_sided). Falls back to NaN if all zeros or <2 samples."""
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.all(d == 0):
        return float("nan"), float("nan")
    try:
        res = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return float("nan"), float("nan")


def compute_wilcoxon_summary(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Wilcoxon signed-rank across all (condition, seed) per (baseline, metric)."""
    rows: List[Dict[str, Any]] = []
    if delta_df.empty:
        return pd.DataFrame(rows)
    for (baseline, metric), sub in delta_df.groupby(["baseline", "metric"], sort=False):
        deltas = sub["delta"].to_numpy()
        higher = bool(sub["direction_higher_is_better"].iloc[0])
        W, p = _wilcoxon_pval(deltas, higher)
        # Normalise delta so "hidtw_better" = positive
        oriented = deltas if higher else -deltas
        rows.append({
            "baseline": baseline,
            "metric": metric,
            "direction_higher_is_better": higher,
            "n": int(len(deltas)),
            "mean_delta_raw": float(np.mean(deltas)) if len(deltas) else float("nan"),
            "median_delta_raw": float(np.median(deltas)) if len(deltas) else float("nan"),
            "mean_delta_oriented": float(np.mean(oriented)) if len(oriented) else float("nan"),
            "median_delta_oriented": float(np.median(oriented)) if len(oriented) else float("nan"),
            "wilcoxon_W": W,
            "p_two_sided": p,
            "hidtw_wins_fraction": float(np.mean(oriented > 0)) if len(oriented) else float("nan"),
        })
    return pd.DataFrame(rows)


def _paired_delta_f1_table(
    delta_df: pd.DataFrame,
    baselines: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Per-(condition, baseline) summary for ΔF1 = HiDTW_full − baseline.

    Returns ``(df_rows, conditions_sorted, baselines_used)``. Empty inputs
    yield an empty frame and empty lists.
    """
    sub = delta_df[delta_df["metric"] == "f1"]
    if sub.empty:
        return pd.DataFrame(), [], []
    if baselines is None:
        present = set(sub["baseline"].unique())
        baselines = [m for m in METHOD_ORDER if m != "HiDTW_full" and m in present]
    if not baselines:
        return pd.DataFrame(), [], []

    conds = _sort_conditions(list(sub["condition"].unique()))
    if not conds:
        return pd.DataFrame(), [], []

    rows: List[Dict[str, Any]] = []
    for cond in conds:
        for baseline in baselines:
            vals = sub[(sub["condition"] == cond)
                       & (sub["baseline"] == baseline)]["delta"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            n = int(len(vals))
            if n == 0:
                rows.append({"condition": cond, "baseline": baseline, "n": 0,
                             "mean_delta_f1": np.nan, "std": np.nan,
                             "ci95": np.nan, "p_two_sided": np.nan})
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            sem = std / np.sqrt(n) if n > 1 else 0.0
            ci95 = 1.96 * sem
            _, p = _wilcoxon_pval(vals, higher_is_better=True)
            rows.append({"condition": cond, "baseline": baseline, "n": n,
                         "mean_delta_f1": mean, "std": std, "ci95": ci95,
                         "p_two_sided": p})
    return pd.DataFrame(rows), conds, baselines


def fig_paired_delta_f1(delta_df: pd.DataFrame, out_pdf: str,
                        baselines: Optional[List[str]] = None) -> None:
    """Forest plot of per-condition mean ΔF1 (HiDTW_full − baseline) with 95%
    CI, drawn **for every baseline simultaneously** (Centroid_NN / DTW_only /
    OT_only by default). Each condition occupies one y-slot; the three
    baselines are shown as vertically-offset, colour-coded points so the
    reader can compare HiDTW's margin against all three competitors at a
    glance. A ``*`` annotation marks baselines where the within-condition
    Wilcoxon two-sided p is < 0.05.
    """
    df_rows, conds, baselines = _paired_delta_f1_table(delta_df, baselines)
    if not conds or df_rows.empty:
        return

    n_b = len(baselines)
    # Vertical offsets for the 2-4 baseline markers within each condition row.
    # Slightly wider than before (0.32 vs 0.28) so CIs don't overlap even at
    # small reproduction sizes.
    if n_b == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.32, 0.32, n_b)

    # Draw at ~double-column width (180 mm) with generous per-row height so
    # text and lines stay legible when this figure is composited into a larger
    # multi-panel layout and scaled down ~2x for print.
    fig_w_mm = 180.0
    fig_h_mm = max(95.0, 16.0 * len(conds) + 40.0)
    fig, ax = plt.subplots(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    y_base = np.arange(len(conds))

    # Marker shape per baseline -> remains distinguishable in grayscale /
    # low-DPI reproductions (color is the primary but not only channel).
    MARKER_SHAPE = {
        "Centroid_NN": "o",
        "DTW_only":    "s",
        "OT_only":     "^",
        "HiDTW_full":  "D",
    }

    for bi, baseline in enumerate(baselines):
        color = OI.get(baseline, "#333333")
        marker = MARKER_SHAPE.get(baseline, "o")
        sub_b = df_rows[df_rows["baseline"] == baseline].set_index("condition")
        xs: List[float] = []; ys: List[float] = []; errs: List[float] = []
        pvals: List[float] = []
        for ci, cond in enumerate(conds):
            if cond not in sub_b.index:
                continue
            row = sub_b.loc[cond]
            if not np.isfinite(row["mean_delta_f1"]):
                continue
            xs.append(float(row["mean_delta_f1"]))
            ys.append(y_base[ci] + offsets[bi])
            errs.append(float(row["ci95"]) if np.isfinite(row["ci95"]) else 0.0)
            pvals.append(float(row["p_two_sided"]) if np.isfinite(row["p_two_sided"]) else np.nan)
        ax.errorbar(
            xs, ys, xerr=errs, fmt=marker, color=color,
            markersize=8.5, markeredgecolor="0.12", markeredgewidth=0.9,
            elinewidth=1.5, capsize=4.0, capthick=1.3,
            ecolor=color, alpha=1.0,
            label=f"HiDTW_full − {baseline}", zorder=5,
        )
        for x, y, err, p in zip(xs, ys, errs, pvals):
            if np.isfinite(p) and p < 0.05:
                ax.text(x + err + 0.004, y, "*", va="center", ha="left",
                        fontsize=13, color=color, fontweight="bold",
                        zorder=6)

    # Zero-reference line (thicker, darker so it reads after downscaling).
    ax.axvline(0.0, color="0.18", lw=1.4, linestyle="--", zorder=1)

    # Alternating bands first (behind data) to group baselines per condition.
    for ci in range(len(conds)):
        if ci % 2 == 1:
            ax.axhspan(ci - 0.5, ci + 0.5, color="0.955", zorder=-10)

    ax.set_yticks(y_base)
    ax.set_yticklabels(conds, fontsize=11)
    ax.set_xlabel("Δ Pair F1   (HiDTW_full − baseline)",
                  fontsize=13, labelpad=8)
    ax.set_title("Paired ΔF1 per condition - HiDTW_full vs each baseline",
                 fontsize=14, pad=14, fontweight="semibold")
    ax.tick_params(labelsize=11, length=5, width=1.0)
    ax.grid(axis="x", color="0.82", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Emphasise left/bottom spines; hide right/top (already hidden via rcParams
    # but also explicit here so the rcParam choice upstream is irrelevant).
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(1.2)
        ax.spines[sp].set_color("0.15")
    for sp in ("right", "top"):
        ax.spines[sp].set_visible(False)

    ax.set_ylim(-0.5, len(conds) - 0.5)

    # Force symmetric x-limits around zero with a healthy margin so CIs don't
    # hit the axis edge after composition.
    xs_all = df_rows["mean_delta_f1"].to_numpy(dtype=float)
    errs_all = df_rows["ci95"].to_numpy(dtype=float)
    finite = np.isfinite(xs_all) & np.isfinite(errs_all)
    if finite.any():
        x_extent = float(np.nanmax(np.abs(xs_all[finite]) + errs_all[finite]))
        x_lim = max(x_extent * 1.18, 0.05)
        ax.set_xlim(-x_lim, x_lim)

    # Legend: bottom, readable type, wider spacing for clarity at small sizes.
    leg = ax.legend(
        fontsize=11, frameon=False, ncol=min(n_b, 3),
        loc="upper center", bbox_to_anchor=(0.5, -0.06),
        handletextpad=0.55, columnspacing=1.8, markerscale=1.0,
        borderaxespad=0.0,
    )
    # Thicker legend handles survive shrinkage.
    for handle in leg.legend_handles if hasattr(leg, "legend_handles") else leg.legendHandles:
        try:
            handle.set_linewidth(1.6)
        except Exception:
            pass

    # Footnote for asterisk meaning - deliberately sits outside the title so
    # the title reads cleanly in the panel.
    ax.text(0.0, 1.012,
            "asterisk · Wilcoxon two-sided p < 0.05    |    error bars · 95% CI",
            transform=ax.transAxes, fontsize=10, color="0.28",
            ha="left", va="bottom", style="italic")

    _savefig(out_pdf)
    df_rows.to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def fig_paired_delta_f1_zoom_trend(
    delta_df: pd.DataFrame,
    out_pdf: str,
    x_min: float = -0.25,
    x_max: float = 1.0,
    baselines: Optional[List[str]] = None,
) -> None:
    """Same layout as ``fig_paired_delta_f1`` (forest + CI + Wilcoxon stars) but
    with a **fixed** x-axis window ``[x_min, x_max]`` (default ``[-0.25, 1]``)
    and one **polyline per baseline** linking mean ΔF1 across conditions.
    """
    df_rows, conds, baselines = _paired_delta_f1_table(delta_df, baselines)
    if not conds or df_rows.empty:
        return

    n_b = len(baselines)
    if n_b == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.32, 0.32, n_b)

    MARKER_SHAPE = {
        "Centroid_NN": "o",
        "DTW_only":    "s",
        "OT_only":     "^",
        "HiDTW_full":  "D",
    }

    fig_w_mm = 180.0
    fig_h_mm = max(95.0, 16.0 * len(conds) + 40.0)
    fig, ax = plt.subplots(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    y_base = np.arange(len(conds))

    for ci in range(len(conds)):
        if ci % 2 == 1:
            ax.axhspan(ci - 0.5, ci + 0.5, color="0.955", zorder=-10)

    ax.axvline(0.0, color="0.18", lw=1.4, linestyle="--", zorder=1)

    for bi, baseline in enumerate(baselines):
        color = OI.get(baseline, "#333333")
        marker = MARKER_SHAPE.get(baseline, "o")
        sub_b = df_rows[df_rows["baseline"] == baseline].set_index("condition")
        xs_line: List[float] = []
        ys_line: List[float] = []
        for ci, cond in enumerate(conds):
            y_pt = float(y_base[ci] + offsets[bi])
            ys_line.append(y_pt)
            if cond not in sub_b.index:
                xs_line.append(float("nan"))
                continue
            row = sub_b.loc[cond]
            m = row["mean_delta_f1"]
            xs_line.append(float(m) if np.isfinite(m) else float("nan"))
        ax.plot(xs_line, ys_line, color=color, lw=1.15, alpha=0.92, zorder=4)

        xs: List[float] = []
        ys: List[float] = []
        errs: List[float] = []
        pvals: List[float] = []
        for ci, cond in enumerate(conds):
            if cond not in sub_b.index:
                continue
            row = sub_b.loc[cond]
            if not np.isfinite(row["mean_delta_f1"]):
                continue
            xs.append(float(row["mean_delta_f1"]))
            ys.append(y_base[ci] + offsets[bi])
            errs.append(float(row["ci95"]) if np.isfinite(row["ci95"]) else 0.0)
            pvals.append(float(row["p_two_sided"]) if np.isfinite(row["p_two_sided"]) else np.nan)
        ax.errorbar(
            xs, ys, xerr=errs, fmt=marker, color=color,
            markersize=8.5, markeredgecolor="0.12", markeredgewidth=0.9,
            elinewidth=1.5, capsize=4.0, capthick=1.3,
            ecolor=color, alpha=1.0,
            label=f"HiDTW_full − {baseline}", zorder=5,
        )
        for x, y, err, p in zip(xs, ys, errs, pvals):
            if np.isfinite(p) and p < 0.05:
                x_star = min(x + err + 0.004, x_max - 1e-3)
                x_star = max(x_star, x_min + 1e-3)
                ax.text(x_star, y, "*", va="center", ha="left",
                        fontsize=13, color=color, fontweight="bold",
                        zorder=6, clip_on=True)

    ax.set_yticks(y_base)
    ax.set_yticklabels(conds, fontsize=11)
    ax.set_xlabel("Δ Pair F1   (HiDTW_full − baseline)",
                  fontsize=13, labelpad=8)
    ax.set_title(
        "Paired ΔF1 per condition - x-axis [{:.2f}, {:.2f}] + baseline trend lines".format(
            x_min, x_max),
        fontsize=14, pad=14, fontweight="semibold",
    )
    ax.tick_params(labelsize=11, length=5, width=1.0)
    ax.grid(axis="x", color="0.82", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(1.2)
        ax.spines[sp].set_color("0.15")
    for sp in ("right", "top"):
        ax.spines[sp].set_visible(False)

    ax.set_ylim(-0.5, len(conds) - 0.5)
    ax.set_xlim(float(x_min), float(x_max))

    leg = ax.legend(
        fontsize=11, frameon=False, ncol=min(n_b, 3),
        loc="upper center", bbox_to_anchor=(0.5, -0.06),
        handletextpad=0.55, columnspacing=1.8, markerscale=1.0,
        borderaxespad=0.0,
    )
    for handle in leg.legend_handles if hasattr(leg, "legend_handles") else leg.legendHandles:
        try:
            handle.set_linewidth(1.6)
        except Exception:
            pass

    ax.text(
        0.0, 1.012,
        "lines · mean ΔF1 across conditions per baseline  |  "
        "asterisk · Wilcoxon p < 0.05  |  bars · 95% CI  |  "
        "x-axis fixed to [{:.2f}, {:.2f}]".format(x_min, x_max),
        transform=ax.transAxes, fontsize=10, color="0.28",
        ha="left", va="bottom", style="italic",
    )

    _savefig(out_pdf)
    df_rows.to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def fig_paired_delta_f1_grouped_bars(
    delta_df: pd.DataFrame,
    out_pdf: str,
    baselines: Optional[List[str]] = None,
) -> None:
    """Grouped bars over conditions - same geometry as ``fig_f1_vs_condition``
    (``_fig_metric_vs_condition``), but bar height is mean **paired** ΔF1
    (HiDTW_full − baseline) per seed; error bars use the **population** std
    (``ddof=0``) across seeds, matching the F1-by-condition figure.
    """
    _, conds, baselines = _paired_delta_f1_table(delta_df, baselines)
    if not conds:
        return
    sub = delta_df[delta_df["metric"] == "f1"]
    if sub.empty:
        return

    n_b = len(baselines)
    means_grid = np.full((n_b, len(conds)), np.nan)
    stds_grid = np.full((n_b, len(conds)), np.nan)
    ns_grid = np.zeros((n_b, len(conds)), dtype=int)
    for bi, baseline in enumerate(baselines):
        for j, cond in enumerate(conds):
            v = sub[(sub["condition"] == cond) & (sub["baseline"] == baseline)]["delta"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v):
                means_grid[bi, j] = float(np.mean(v))
                stds_grid[bi, j] = float(np.std(v, ddof=0))
                ns_grid[bi, j] = int(len(v))

    fig_w = max(120, 12 * len(conds) + 30)
    fig_h = 70
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    x = np.arange(len(conds))
    width = 0.8 / max(n_b, 1)

    for bi, baseline in enumerate(baselines):
        means = means_grid[bi]
        stds = stds_grid[bi]
        if np.all(np.isnan(means)):
            continue
        ax.bar(
            x + (bi - (n_b - 1) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=1.6,
            color=OI.get(baseline, "#333333"),
            edgecolor="0.15",
            linewidth=0.4,
            label=f"HiDTW_full − {baseline}",
            error_kw={"linewidth": 0.5, "capthick": 0.5},
            zorder=2,
        )

    # Connect bar tops (mean ΔF1) across conditions for each baseline.
    for bi, baseline in enumerate(baselines):
        color = OI.get(baseline, "#333333")
        xp = x + (bi - (n_b - 1) / 2) * width
        yp = means_grid[bi].astype(float)
        ax.plot(xp, yp, color=color, lw=1.05, alpha=0.95, zorder=4, clip_on=False)

    ax.axhline(0.0, color="0.35", lw=0.9, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Δ Pair F1  (HiDTW_full − baseline)", fontsize=8)
    ax.set_title("Paired ΔF1 vs baseline - grouped bars (mean ± std over seeds)",
                 fontsize=9, fontweight="normal")
    ax.tick_params(labelsize=7)
    ax.legend(
        fontsize=6, ncol=min(n_b, 4), frameon=False,
        loc="upper center", bbox_to_anchor=(0.5, -0.32),
    )
    ax.grid(axis="y", color="0.9", lw=0.4)
    ax.set_axisbelow(True)

    lo_mat = means_grid - stds_grid
    hi_mat = means_grid + stds_grid
    lo_finite = lo_mat[np.isfinite(lo_mat)]
    hi_finite = hi_mat[np.isfinite(hi_mat)]
    if lo_finite.size and hi_finite.size:
        y_lo = float(np.min(lo_finite)) - 0.02
        y_hi = float(np.max(hi_finite)) + 0.02
        ax.set_ylim(min(0.0, y_lo), max(y_hi, 0.05))

    _savefig(out_pdf)
    rows_csv: List[Dict[str, Any]] = []
    for bi, baseline in enumerate(baselines):
        for j, c in enumerate(conds):
            rows_csv.append({
                "baseline": baseline,
                "condition": c,
                "mean_delta_f1": (float(means_grid[bi, j])
                                  if np.isfinite(means_grid[bi, j]) else np.nan),
                "std_delta_f1_pop": (float(stds_grid[bi, j])
                                     if np.isfinite(stds_grid[bi, j]) else np.nan),
                "n": int(ns_grid[bi, j]),
            })
    pd.DataFrame(rows_csv).to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


# --- 7c. PER-CONDITION FIGURE BUNDLE ---
#
# For every aliased condition C##_… we emit a dedicated sub-folder with a rich
# collection of single-panel PDFs comparing the four methods on that condition
# only. This keeps every figure compliant with the user's "single panel, no
# A/B/C/D labels" rule while still giving readers a deep dive per scenario.

def _per_cond_bar_f1_with_seed_dots(p_sub: pd.DataFrame, cond: str, out_pdf: str):
    """Bar of F1 mean + overlaid per-seed points (shows within-condition
    variance without stacking a second figure)."""
    if p_sub.empty:
        return
    fig, ax = plt.subplots(figsize=(96 / 25.4, 66 / 25.4))
    methods = [m for m in METHOD_ORDER if (p_sub["method"] == m).any()]
    x = np.arange(len(methods))
    means: List[float] = []
    stds: List[float] = []
    for m in methods:
        vals = p_sub[p_sub["method"] == m]["f1"].astype(float)
        means.append(float(vals.mean()) if len(vals) else 0.0)
        stds.append(float(vals.std(ddof=0)) if len(vals) else 0.0)
    colors = [OI.get(m, "#333") for m in methods]
    ax.bar(x, means, yerr=stds, capsize=2.5, color=colors, edgecolor="0.15",
           linewidth=0.5, alpha=0.75, error_kw={"linewidth": 0.6})
    # Per-seed dots jittered horizontally
    rng = np.random.default_rng(seed=abs(hash(cond)) % (2**31))
    for xi, m in enumerate(methods):
        vals = p_sub[p_sub["method"] == m]["f1"].astype(float).to_numpy()
        if not len(vals):
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(xi + jitter, vals, s=14, color="0.1",
                   edgecolors="white", linewidths=0.4, zorder=5, alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Pair F1", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", color="0.92", lw=0.4); ax.set_axisbelow(True)
    ax.set_title(f"{cond} - Pair F1 per seed (bar = mean ± SD; dots = seeds)",
                 fontsize=9)
    _savefig(out_pdf)
    pd.DataFrame([{"method": m, "mean_f1": mu, "std_f1": sd}
                  for m, mu, sd in zip(methods, means, stds)]
                 ).to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def _per_cond_paired_delta(delta_df: pd.DataFrame, cond: str, out_pdf: str):
    """Within-condition paired ΔF1 per baseline - useful when you want the
    full picture for one scenario without having to read the big forest plot."""
    sub = delta_df[(delta_df["condition"] == cond) & (delta_df["metric"] == "f1")]
    if sub.empty:
        return
    baselines = [m for m in METHOD_ORDER if m != "HiDTW_full" and (sub["baseline"] == m).any()]
    if not baselines:
        return
    fig, ax = plt.subplots(figsize=(92 / 25.4, 62 / 25.4))
    x = np.arange(len(baselines))
    means: List[float] = []
    cis: List[float] = []
    pvals: List[float] = []
    for b in baselines:
        vals = sub[sub["baseline"] == b]["delta"].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            means.append(np.nan); cis.append(0.0); pvals.append(np.nan); continue
        m = float(np.mean(vals))
        sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        means.append(m); cis.append(1.96 * sem)
        _, p = _wilcoxon_pval(vals, higher_is_better=True)
        pvals.append(p)
    colors = [OI.get(b, "#333") for b in baselines]
    ax.bar(x, means, yerr=cis, capsize=3, color=colors, edgecolor="0.15",
           linewidth=0.5, alpha=0.85, error_kw={"linewidth": 0.7})
    # Dots: individual seed deltas
    rng = np.random.default_rng(seed=(abs(hash(cond)) + 7) % (2**31))
    for xi, b in enumerate(baselines):
        vals = sub[sub["baseline"] == b]["delta"].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(xi + jitter, vals, s=14, color="0.1",
                   edgecolors="white", linewidths=0.4, zorder=5, alpha=0.95)
    ax.axhline(0.0, color="0.35", lw=0.7, linestyle="--")
    for xi, p in enumerate(pvals):
        if np.isfinite(p) and p < 0.05:
            ax.text(xi, means[xi] + cis[xi] + 0.01, "*",
                    ha="center", va="bottom", fontsize=9, color="0.15")
    ax.set_xticks(x)
    ax.set_xticklabels([f"vs {b}" for b in baselines], fontsize=7)
    ax.set_ylabel("Δ Pair F1 (HiDTW_full − baseline)", fontsize=8)
    ax.set_title(f"{cond} - paired ΔF1 (95% CI; * Wilcoxon p<0.05)",
                 fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", color="0.92", lw=0.4); ax.set_axisbelow(True)
    _savefig(out_pdf)
    pd.DataFrame([{"baseline": b, "mean_delta_f1": mu,
                   "ci95": ci, "p_two_sided": p}
                  for b, mu, ci, p in zip(baselines, means, cis, pvals)]
                 ).to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def _per_cond_radar(p_sub: pd.DataFrame, t_sub: pd.DataFrame, g_sub: pd.DataFrame,
                    cond: str, out_pdf: str):
    """Radar / spider plot of 6 normalised metrics per method within a
    condition. Fragmentation is inverted so "farther from centre = better"
    in every axis."""
    if p_sub.empty:
        return
    # (display_name, source_df, column, lower_is_better)
    specs = [
        ("F1",               p_sub, "f1",                 False),
        ("Precision",        p_sub, "precision",          False),
        ("Recall",           p_sub, "recall",             False),
        ("1 − ID-switch",    p_sub, "id_switch_rate",     True),
        ("Purity",           t_sub, "mean_purity",        False),
        ("Layered IoU",      g_sub, "mean_layered_iou",   False),
    ]
    methods = [m for m in METHOD_ORDER if (p_sub["method"] == m).any()]
    if not methods:
        return
    values: Dict[str, List[float]] = {m: [] for m in methods}
    for (_, df, col, lower_better) in specs:
        for m in methods:
            sub = df[df["method"] == m][col] if col in df.columns else pd.Series([], dtype=float)
            v = float(sub.astype(float).mean()) if len(sub) else 0.0
            if lower_better:
                v = 1.0 - v
            values[m].append(v)
    # Clamp to [0, 1] - radar axes are normalised "higher = better".
    for m in methods:
        values[m] = [max(0.0, min(1.0, x)) for x in values[m]]

    labels = [s[0] for s in specs]
    n_ax = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += [angles[0]]

    fig = plt.figure(figsize=(95 / 25.4, 95 / 25.4))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6, color="0.4")
    ax.tick_params(pad=-2)
    ax.grid(color="0.85", lw=0.4)

    for m in methods:
        data = values[m] + [values[m][0]]
        col = OI.get(m, "#333")
        ax.plot(angles, data, color=col, lw=1.2, label=m, alpha=0.95)
        ax.fill(angles, data, color=col, alpha=0.12)
    ax.legend(fontsize=6, frameon=False, loc="upper right",
              bbox_to_anchor=(1.28, 1.08))
    ax.set_title(f"{cond} - method profile", fontsize=9, pad=14)
    _savefig(out_pdf)
    rows: List[Dict[str, Any]] = []
    for i, lbl in enumerate(labels):
        for m in methods:
            rows.append({"metric": lbl, "method": m, "value_norm": values[m][i]})
    pd.DataFrame(rows).to_csv(out_pdf.replace(".pdf", ".csv"), index=False)


def _per_cond_scatter_prf(p_sub: pd.DataFrame, cond: str, out_pdf: str):
    """Precision vs Recall scatter for this condition (seeds as points)."""
    if p_sub.empty:
        return
    fig, ax = plt.subplots(figsize=(82 / 25.4, 72 / 25.4))
    for m in METHOD_ORDER:
        sub = p_sub[p_sub["method"] == m]
        if not len(sub):
            continue
        ax.scatter(sub["recall"], sub["precision"], s=36,
                   color=OI.get(m, "#333"), edgecolors="0.15",
                   linewidths=0.45, alpha=0.9, label=m)
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall", fontsize=8); ax.set_ylabel("Precision", fontsize=8)
    ax.set_title(f"{cond} - precision vs recall (per seed)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower left", frameon=True,
              edgecolor="0.7", ncol=2)
    _savefig(out_pdf)
    p_sub[["seed", "method", "precision", "recall", "f1"]].to_csv(
        out_pdf.replace(".pdf", ".csv"), index=False)


def render_per_condition_figures(tables: Dict[str, pd.DataFrame],
                                 delta_df: pd.DataFrame,
                                 out_dir: str) -> None:
    """For every aliased condition, emit a dedicated bundle of single-panel
    figures under ``<out_dir>/per_condition/<cond>/``."""
    p = tables["pair_metrics"]; t = tables["track_metrics_per_dataset"]
    g = tables["geometry_metrics"]; pt = tables["track_metrics_per_track"]
    pg = tables["geometry_per_track"]
    if p.empty:
        return
    root = os.path.join(out_dir, "per_condition")
    os.makedirs(root, exist_ok=True)
    conds = _sort_conditions(list(p["condition"].unique()))
    print(f"\n[per-condition figures] {len(conds)} scenarios -> {root}/")
    for cond in conds:
        cdir = os.path.join(root, cond)
        os.makedirs(cdir, exist_ok=True)
        print(f"  · {cond}")

        p_sub  = p[p["condition"]  == cond]
        t_sub  = t[t["condition"]  == cond]
        g_sub  = g[g["condition"]  == cond]
        pt_sub = pt[pt["condition"] == cond]
        pg_sub = pg[pg["condition"] == cond]

        # metric-level bars (mean ± std across seeds)
        fig_bar_by_method(_method_stats(p_sub, "f1"),             "Pair F1",
                          f"{cond} - Pair F1",
                          os.path.join(cdir, "fig_f1_bar.pdf"), ylim=(0, 1.05))
        fig_bar_by_method(_method_stats(p_sub, "precision"),      "Pair precision",
                          f"{cond} - Pair precision",
                          os.path.join(cdir, "fig_precision_bar.pdf"), ylim=(0, 1.05))
        fig_bar_by_method(_method_stats(p_sub, "recall"),         "Pair recall",
                          f"{cond} - Pair recall",
                          os.path.join(cdir, "fig_recall_bar.pdf"), ylim=(0, 1.05))
        fig_bar_by_method(_method_stats(p_sub, "id_switch_rate"), "ID-switch rate",
                          f"{cond} - ID-switch rate (lower is better)",
                          os.path.join(cdir, "fig_id_switch_bar.pdf"))
        fig_bar_by_method(_method_stats(t_sub, "coverage"),       "Coverage",
                          f"{cond} - Cell coverage",
                          os.path.join(cdir, "fig_coverage_bar.pdf"), ylim=(0, 1.05))
        fig_bar_by_method(_method_stats(t_sub, "fragmentation_rate"),
                          "Fragmentation rate",
                          f"{cond} - Predicted tracks per GT track",
                          os.path.join(cdir, "fig_fragmentation_bar.pdf"))
        fig_bar_by_method(_method_stats(t_sub, "mean_purity"),    "Track purity",
                          f"{cond} - Mean track purity",
                          os.path.join(cdir, "fig_purity_bar.pdf"), ylim=(0, 1.05))
        fig_bar_by_method(_method_stats(t_sub, "mean_completeness"),
                          "Track completeness",
                          f"{cond} - Mean track completeness",
                          os.path.join(cdir, "fig_completeness_bar.pdf"), ylim=(0, 1.05))
        fig_bar_by_method(_method_stats(g_sub, "mean_layered_iou"), "Layered IoU",
                          f"{cond} - Mean layered 2D IoU",
                          os.path.join(cdir, "fig_layered_iou_bar.pdf"), ylim=(0, 1.05))
        fig_bar_by_method(_method_stats(g_sub, "length_pearson_r"),
                          "Pearson r (length)",
                          f"{cond} - Length Pearson r",
                          os.path.join(cdir, "fig_length_r_bar.pdf"), ylim=(-0.1, 1.05))
        fig_bar_by_method(_method_stats(g_sub, "volume_pearson_r"),
                          "Pearson r (volume)",
                          f"{cond} - Volume Pearson r",
                          os.path.join(cdir, "fig_volume_r_bar.pdf"), ylim=(-0.1, 1.05))

        # distributions
        fig_violin_per_track(pt_sub, "purity", "Track purity",
                             f"{cond} - per-track purity",
                             os.path.join(cdir, "fig_track_purity_violin.pdf"))
        fig_violin_per_track(pt_sub, "completeness", "Track completeness",
                             f"{cond} - per-track completeness",
                             os.path.join(cdir, "fig_track_completeness_violin.pdf"))
        fig_violin_per_track(pg_sub, "mean_layered_iou", "Mean layered IoU",
                             f"{cond} - per-track layered IoU",
                             os.path.join(cdir, "fig_layered_iou_violin.pdf"))

        # geometry scatter
        fig_length_volume_scatter(pg_sub, "length",
                                  os.path.join(cdir, "fig_length_scatter.pdf"))
        fig_length_volume_scatter(pg_sub, "volume",
                                  os.path.join(cdir, "fig_volume_scatter.pdf"))

        # summary heatmap / radar / paired delta / seed dots
        fig_metric_heatmap(p_sub, t_sub, g_sub,
                           os.path.join(cdir, "fig_metric_heatmap.pdf"),
                           title=f"{cond} - method × metric summary")
        _per_cond_bar_f1_with_seed_dots(p_sub, cond,
                                        os.path.join(cdir, "fig_f1_per_seed.pdf"))
        _per_cond_scatter_prf(p_sub, cond,
                              os.path.join(cdir, "fig_precision_recall_scatter.pdf"))
        _per_cond_radar(p_sub, t_sub, g_sub, cond,
                        os.path.join(cdir, "fig_method_radar.pdf"))
        if delta_df is not None and not delta_df.empty:
            _per_cond_paired_delta(delta_df, cond,
                                   os.path.join(cdir, "fig_paired_delta_f1_per_baseline.pdf"))


# --- 8. CLI ---

def main():
    parser = argparse.ArgumentParser(description="Simulation 1 evaluator.")
    parser.add_argument("--phantom-root", default="simulation01_results/phantoms")
    parser.add_argument("--method-root",  default="simulation01_results/methods")
    parser.add_argument("--out-dir",      default="simulation01_results/evaluation")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=None)
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    cond_names = args.conditions if args.conditions else list(CONDITIONS.keys())
    if args.seeds is not None:
        seeds = list(args.seeds)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    methods = args.methods if args.methods else list(METHOD_SPECS.keys())

    os.makedirs(args.out_dir, exist_ok=True)

    print("Simulation 1 - evaluation")
    print(f"  conditions: {cond_names}")
    print(f"  seeds:      {seeds}")
    print(f"  methods:    {methods}")
    print(f"  out-dir:    {args.out_dir}")

    per_dataset: List[Dict[str, Any]] = []
    for c in cond_names:
        for s in seeds:
            pd_dir = os.path.join(args.phantom_root, c, f"seed{s}")
            if not os.path.isdir(pd_dir):
                print(f"[skip] no phantom dir for {c}/seed{s}")
                continue
            print(f"\n=== {c} / seed{s} ===")
            result = evaluate_one(c, s, args.phantom_root, args.method_root, methods)
            per_dataset.append(result)
            for mname, m in result["methods"].items():
                pm = m["pair"]; tm = m["track"]; gm = m["geometry"]
                print(f"    {mname:14s}  F1={pm['f1']:.3f}  purity={tm['mean_purity']:.3f}  "
                      f"cov={tm['coverage']:.3f}  lenR={gm['length_pearson_r']:.3f}  "
                      f"IoU={gm['mean_layered_iou']:.3f}")

    # Aggregate and write tables
    tables = aggregate_to_tables(per_dataset)
    for name, df in tables.items():
        df.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        print(f"  [table] {name}.csv  rows={len(df)}")

    # Summary JSON
    summary = {
        "conditions": cond_names, "seeds": seeds, "methods": methods,
        "n_datasets": len(per_dataset),
        "aggregate_means": {},
    }
    for mname in methods:
        sub_p = tables["pair_metrics"][tables["pair_metrics"]["method"] == mname]
        sub_t = tables["track_metrics_per_dataset"][tables["track_metrics_per_dataset"]["method"] == mname]
        sub_g = tables["geometry_metrics"][tables["geometry_metrics"]["method"] == mname]
        summary["aggregate_means"][mname] = {
            "pair_f1": float(sub_p["f1"].mean()) if len(sub_p) else 0.0,
            "pair_precision": float(sub_p["precision"].mean()) if len(sub_p) else 0.0,
            "pair_recall": float(sub_p["recall"].mean()) if len(sub_p) else 0.0,
            "id_switch_rate": float(sub_p["id_switch_rate"].mean()) if len(sub_p) else 0.0,
            "track_purity": float(sub_t["mean_purity"].mean()) if len(sub_t) else 0.0,
            "track_completeness": float(sub_t["mean_completeness"].mean()) if len(sub_t) else 0.0,
            "coverage": float(sub_t["coverage"].mean()) if len(sub_t) else 0.0,
            "fragmentation_rate": float(sub_t["fragmentation_rate"].mean()) if len(sub_t) else 0.0,
            "length_pearson_r": float(sub_g["length_pearson_r"].mean()) if len(sub_g) else 0.0,
            "volume_pearson_r": float(sub_g["volume_pearson_r"].mean()) if len(sub_g) else 0.0,
            "mean_layered_iou": float(sub_g["mean_layered_iou"].mean()) if len(sub_g) else 0.0,
        }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Figures
    print("\n[figures]")
    pd_p = tables["pair_metrics"]; pd_t = tables["track_metrics_per_dataset"]
    pd_g = tables["geometry_metrics"]; pd_tt = tables["track_metrics_per_track"]
    pd_gg = tables["geometry_per_track"]

    # Full-range F1 bar chart plus a zoomed variant for near-ceiling differences.
    fig_f1_vs_condition(
        pd_p,
        os.path.join(args.out_dir, "fig_f1_vs_condition.pdf"),
        zoom_out_pdf=os.path.join(args.out_dir, "fig_f1_vs_condition_zoom.pdf"),
    )
    fig_recall_vs_condition(
        pd_p,
        os.path.join(args.out_dir, "fig_recall_vs_condition.pdf"),
        zoom_out_pdf=os.path.join(args.out_dir, "fig_recall_vs_condition_zoom.pdf"),
    )
    fig_f1_vs_density_or_curv(pd_p, "density", os.path.join(args.out_dir, "fig_f1_vs_density.pdf"), "Density")
    fig_f1_vs_density_or_curv(pd_p, "curv",    os.path.join(args.out_dir, "fig_f1_vs_curvature.pdf"), "Curvature category")
    fig_f1_L1_vs_L2(pd_p,     os.path.join(args.out_dir, "fig_f1_L1_vs_L2.pdf"))
    fig_precision_recall_scatter(pd_p, os.path.join(args.out_dir, "fig_precision_recall_scatter.pdf"))

    fig_bar_by_method(_method_stats(pd_p, "f1"),              "Pair F1",
                      "Pair F1 (mean ± std)",                 os.path.join(args.out_dir, "fig_f1_bar.pdf"),
                      ylim=(0, 1.05))
    fig_bar_by_method(_method_stats(pd_p, "id_switch_rate"),  "ID-switch rate",
                      "ID-switch rate (lower is better)",     os.path.join(args.out_dir, "fig_id_switch_bar.pdf"))
    fig_bar_by_method(_method_stats(pd_t, "coverage"),        "Coverage",
                      "Cell coverage ratio",                  os.path.join(args.out_dir, "fig_coverage_bar.pdf"),
                      ylim=(0, 1.05))
    fig_bar_by_method(_method_stats(pd_t, "fragmentation_rate"), "Fragmentation rate",
                      "Mean predicted tracks per GT track",    os.path.join(args.out_dir, "fig_fragmentation_bar.pdf"))
    fig_bar_by_method(_method_stats(pd_t, "mean_purity"),     "Track purity",
                      "Mean track purity",                    os.path.join(args.out_dir, "fig_purity_bar.pdf"),
                      ylim=(0, 1.05))
    fig_bar_by_method(_method_stats(pd_g, "mean_layered_iou"), "Layered 3D IoU",
                      "Mean per-layer 2D IoU along matched tracks",
                      os.path.join(args.out_dir, "fig_layered_iou_bar.pdf"), ylim=(0, 1.05))
    fig_bar_by_method(_method_stats(pd_g, "length_pearson_r"), "Pearson r (length)",
                      "Predicted vs true cell length (Pearson r)",
                      os.path.join(args.out_dir, "fig_length_r_bar.pdf"), ylim=(-0.1, 1.05))
    fig_bar_by_method(_method_stats(pd_g, "volume_pearson_r"), "Pearson r (volume)",
                      "Predicted vs true cell volume (Pearson r)",
                      os.path.join(args.out_dir, "fig_volume_r_bar.pdf"), ylim=(-0.1, 1.05))

    fig_violin_per_track(pd_tt, "purity",        "Track purity",
                         "Per-track purity distribution",
                         os.path.join(args.out_dir, "fig_track_purity_violin.pdf"))
    fig_violin_per_track(pd_tt, "completeness",  "Track completeness",
                         "Per-track completeness distribution",
                         os.path.join(args.out_dir, "fig_track_completeness_violin.pdf"))
    fig_violin_per_track(pd_gg, "mean_layered_iou", "Mean layered IoU",
                         "Per-track mean layered IoU distribution",
                         os.path.join(args.out_dir, "fig_layered_iou_violin.pdf"))

    fig_length_volume_scatter(pd_gg, "length", os.path.join(args.out_dir, "fig_length_scatter_all_methods.pdf"))
    fig_length_volume_scatter(pd_gg, "volume", os.path.join(args.out_dir, "fig_volume_scatter_all_methods.pdf"))
    fig_length_volume_scatter(pd_gg, "length",
                              os.path.join(args.out_dir, "fig_length_scatter_hidtw.pdf"),
                              methods=["HiDTW_full"])
    fig_length_volume_scatter(pd_gg, "volume",
                              os.path.join(args.out_dir, "fig_volume_scatter_hidtw.pdf"),
                              methods=["HiDTW_full"])

    fig_metric_heatmap(pd_p, pd_t, pd_g,
                       os.path.join(args.out_dir, "fig_metric_heatmap.pdf"))

    # Paired HiDTW vs baselines (F1 / id_switch / mean_layered_IoU).
    print("  [paired] computing HiDTW_full vs baselines paired deltas ...")
    baselines_to_compare = [m for m in METHOD_ORDER if m != "HiDTW_full"]
    delta_df = compute_paired_deltas(pd_p, pd_g, target="HiDTW_full",
                                     baselines=baselines_to_compare)
    delta_df.to_csv(os.path.join(args.out_dir, "paired_delta_vs_baselines.csv"),
                    index=False)
    wilcox_df = compute_wilcoxon_summary(delta_df)
    wilcox_df.to_csv(os.path.join(args.out_dir, "paired_wilcoxon_summary.csv"),
                     index=False)
    fig_paired_delta_f1(delta_df,
                        os.path.join(args.out_dir, "fig_paired_delta_f1.pdf"))
    fig_paired_delta_f1_zoom_trend(
        delta_df,
        os.path.join(args.out_dir, "fig_paired_delta_f1_zoom025_trendlines.pdf"),
        x_min=-0.25,
        x_max=1.0,
    )
    fig_paired_delta_f1_grouped_bars(
        delta_df,
        os.path.join(args.out_dir, "fig_paired_delta_f1_bars_vs_condition.pdf"),
    )

    # Per-condition figure bundles
    render_per_condition_figures(tables, delta_df, args.out_dir)
    if not wilcox_df.empty:
        print("  [paired] Wilcoxon (HiDTW_full vs each baseline):")
        for _, r in wilcox_df.iterrows():
            print(f"      {r['baseline']:14s}  {r['metric']:18s}  "
                  f"n={int(r['n'])}  median_Δ={r['median_delta_raw']:+.4f}  "
                  f"wins={r['hidtw_wins_fraction']*100:5.1f}%  p={r['p_two_sided']:.2e}")

    print(f"\nDone. Evaluation outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
