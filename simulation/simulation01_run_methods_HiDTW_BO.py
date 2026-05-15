#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation 1 - HiDTW only, tuned by unsupervised Bayesian optimisation.

Companion to ``simulation01_run_methods.py``. The sibling script tunes the
HiDTW combiner with a 66-point simplex grid against GT pair-F1; this script
mirrors the real-data BO setup (``hidtw_pipeline/bayes_opt_margin_v10.py``)
and selects (alpha, dtw, ot, centroid) via ``skopt.gp_minimize`` on
``global_score = 0.4*confidence + 0.4*coverage + 0.2*length``. GT pair-F1
is logged offline as a diagnostic column only.

Outputs (per (condition, seed)):
    simulation01_results/methods/<cond>/seed<k>/HiDTW_full_BO/
        tracks.csv
        best_weights.json
        bo_history.json
        bo_history.csv
        convergence.png

Aggregated:
    simulation01_results/methods/hidtw_full_bo_summary_by_condition_seed.csv

Usage:
    python simulation01_run_methods_HiDTW_BO.py
    python simulation01_run_methods_HiDTW_BO.py --conditions L2_clean_sanity --seeds 0
    python simulation01_run_methods_HiDTW_BO.py --n-calls 20 --n-initial 8
    python simulation01_run_methods_HiDTW_BO.py --skip-existing
"""

from __future__ import annotations

# Pin CUDA_VISIBLE_DEVICES before importing torch.
import argparse as _argparse_early
import os as _os_early

_early_parser = _argparse_early.ArgumentParser(add_help=False)
_early_parser.add_argument("--gpu-id", type=int, default=None)
_early_args, _ = _early_parser.parse_known_args()
if _early_args.gpu_id is not None:
    _os_early.environ["CUDA_VISIBLE_DEVICES"] = str(_early_args.gpu_id)
    print(f"[Early Init] CUDA_VISIBLE_DEVICES = {_early_args.gpu_id}")
del _early_parser, _early_args

import argparse
import json
import os
import re
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Matplotlib for the per-seed convergence PDF
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

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SIM_DIR)
for _p in (_REPO_ROOT, _SIM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the heavy lifting (precompute, calibration, scoring, conflict resolve,
# GT-pair helper, output writer) from the sibling script.
from simulation01_run_methods import (  # noqa: E402
    Cell,
    MIN_TRACK_LENGTH,
    _gt_pairs_from_df,
    _merge_bi,
    _pair_f1_from_tracks,
    _resolve_conflicts,
    _score_candidates,
    calibrate_divisors,
    precompute_all_costs,
    tracks_to_dataframe,
)
from simulation01_phantom_generator import CONDITIONS  # noqa: E402
from hidtw_pipeline import cell_reconstruction_runGPU_3Drevised_v10 as v10  # noqa: E402

try:
    from skopt import gp_minimize  # type: ignore
    from skopt.space import Real  # type: ignore
    from skopt.utils import use_named_args  # type: ignore
    HAS_SKOPT = True
except Exception as _skopt_err:  # pragma: no cover
    HAS_SKOPT = False
    _SKOPT_IMPORT_ERROR = _skopt_err


# --- 1. GLOBAL-SCORE OBJECTIVE (mirrors v10 evaluate_all_tracks) ---
#
#   global_confidence = min(100, mean(all_decision_margins) / 0.5 * 100)
#   coverage          = total_cells_in_tracks / total_available_cells * 100
#   length            = min(100, average_length / 10.0 * 100)
#   global_score      = 0.4*conf + 0.4*coverage + 0.2*length
#
# ``all_decision_margins`` carries the LOCAL relative margin per probe (0.0
# when no candidate exists), matching v10's ``track_3d_cells`` convention.

GLOBAL_CONF_REFERENCE = 0.5
LENGTH_REFERENCE = 10.0


def compute_global_score(
    tracks: List[Dict[str, Any]],
    cells_by_slice: Dict[int, List[Cell]],
    decisions_log: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Return v10-compatible quality metrics for a candidate parameter set."""
    n_tracks = len(tracks)
    track_lengths = [int(t["length"]) for t in tracks] if tracks else []
    total_cells_in_tracks = int(sum(track_lengths))
    avg_length = float(np.mean(track_lengths)) if track_lengths else 0.0
    total_available = int(sum(len(c) for c in cells_by_slice.values()))
    coverage_ratio = (
        total_cells_in_tracks / total_available if total_available else 0.0
    )

    if decisions_log:
        decision_margins = np.asarray(
            [d["relative_margin_local"] for d in decisions_log], dtype=float
        )
        global_avg_margin = float(np.mean(decision_margins))
        global_margin_std = float(np.std(decision_margins))
        n_total = int(len(decision_margins))
        low_conf = int(np.sum(decision_margins < 0.2))
        no_cand = int(sum(1 for d in decisions_log if not d["has_candidate"]))
        low_confidence_ratio = low_conf / n_total if n_total else 1.0
        no_candidate_ratio = no_cand / n_total if n_total else 1.0
    else:
        global_avg_margin = 0.0
        global_margin_std = 0.0
        n_total = 0
        low_confidence_ratio = 1.0
        no_candidate_ratio = 1.0

    global_confidence_score = float(min(100.0, (global_avg_margin / GLOBAL_CONF_REFERENCE) * 100.0))
    coverage_score = float(coverage_ratio * 100.0)
    length_score = float(min(100.0, (avg_length / LENGTH_REFERENCE) * 100.0))
    global_score = float(
        0.4 * global_confidence_score + 0.4 * coverage_score + 0.2 * length_score
    )

    return {
        "global_score": global_score,
        "global_confidence_score": global_confidence_score,
        "coverage_score": coverage_score,
        "length_score": length_score,
        "global_avg_margin": global_avg_margin,
        "global_margin_std": global_margin_std,
        "coverage_ratio": coverage_ratio,
        "average_length": avg_length,
        "n_tracks": int(n_tracks),
        "total_cells_in_tracks": int(total_cells_in_tracks),
        "total_available_cells": int(total_available),
        "total_decisions": int(n_total),
        "low_confidence_ratio": float(low_confidence_ratio),
        "no_candidate_ratio": float(no_candidate_ratio),
    }


# --- 2. TRACK BUILDER WITH GLOBAL DECISION LOG ---


def _evaluate_top_match(
    cid: str,
    source_slice: int,
    direction: str,
    precomputed: Dict[str, Any],
    used: set,
    weights: Dict[str, float],
    divisors: Dict[str, float],
    alpha: float,
) -> Dict[str, Any]:
    """Like ``simulation01_run_methods._find_best`` but always returns a dict
    (exposes ``has_candidate`` and un-thresholded margins so the caller can
    log every probe, accepted or rejected)."""
    lookup = precomputed["forward"] if direction == "forward" else precomputed["backward"]
    no_cand = {
        "has_candidate": False,
        "rm_local": 0.0,
        "rm_global": 0.0,
        "accepted": False,
        "target_id": None,
        "target_slice": None,
    }
    if source_slice not in lookup or cid not in lookup[source_slice]:
        return no_cand
    cands = [c for c in lookup[source_slice][cid] if c["target_id"] not in used]
    if not cands:
        return no_cand
    scored = _score_candidates(cands, weights, divisors)
    if not scored:
        return no_cand
    top1 = scored[0]
    if len(scored) > 1:
        top2 = scored[1]
        rm_local = (top2["loss_local"] - top1["loss_local"]) / (
            abs(top1["loss_local"]) + 1e-6
        )
        rm_global = (top2["loss_global"] - top1["loss_global"]) / (
            abs(top1["loss_global"]) + 1e-6
        )
    else:
        rm_local = 1.0
        rm_global = 1.0
    accepted = bool(rm_local >= alpha)
    return {
        "has_candidate": True,
        "rm_local": float(rm_local),
        "rm_global": float(rm_global),
        "accepted": accepted,
        "target_id": top1["target_id"] if accepted else None,
        "target_slice": top1["target_slice"] if accepted else None,
    }


def _build_direction_decisions(
    start_id: str,
    start_slice: int,
    direction: str,
    precomputed: Dict[str, Any],
    used: set,
    cells_by_slice: Dict[int, List[Cell]],
    weights: Dict[str, float],
    divisors: Dict[str, float],
    alpha: float,
    decisions_log: List[Dict[str, Any]],
) -> Tuple[List[Tuple[int, str]], List[float]]:
    """Walk one direction and append every probe (accepted or not) to
    ``decisions_log`` (mutated in place; one entry per probe attempt)."""
    cells_list: List[Tuple[int, str]] = [(start_slice, start_id)]
    margins: List[float] = []
    cur_id, cur_s = start_id, start_slice
    slices = sorted(cells_by_slice.keys())
    min_s, max_s = slices[0], slices[-1]
    while True:
        nxt = cur_s + 1 if direction == "forward" else cur_s - 1
        if nxt < min_s or nxt > max_s or nxt not in cells_by_slice:
            break
        info = _evaluate_top_match(
            cur_id, cur_s, direction, precomputed, used, weights, divisors, alpha
        )
        decisions_log.append(
            {
                "slice": int(nxt),
                "direction": direction,
                "has_candidate": bool(info["has_candidate"]),
                "relative_margin_local": float(info["rm_local"]),
                "relative_margin_global": float(info["rm_global"]),
                "accepted": bool(info["accepted"]),
            }
        )
        if not info["accepted"]:
            break
        cells_list.append((info["target_slice"], info["target_id"]))
        margins.append(info["rm_global"])
        cur_id, cur_s = info["target_id"], info["target_slice"]
    return cells_list, margins


def build_tracks_with_decisions(
    cells_by_slice: Dict[int, List[Cell]],
    precomputed: Dict[str, Any],
    weights: Dict[str, float],
    divisors: Dict[str, float],
    alpha: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Same as ``simulation01_run_methods.build_all_tracks`` but also returns
    the per-decision log consumed by ``compute_global_score``. The tracks
    output is identical for any given ``(weights, alpha)``."""
    all_tracks: List[Dict[str, Any]] = []
    used: set = set()
    decisions_log: List[Dict[str, Any]] = []
    slices = sorted(cells_by_slice.keys())
    min_s = slices[0]
    while True:
        round_tracks: List[Dict[str, Any]] = []
        for start_s in slices:
            for cell in cells_by_slice[start_s]:
                if cell.cell_id in used:
                    continue
                fwd_cells, fwd_m = _build_direction_decisions(
                    cell.cell_id,
                    cell.slice_index,
                    "forward",
                    precomputed,
                    used,
                    cells_by_slice,
                    weights,
                    divisors,
                    alpha,
                    decisions_log,
                )
                if start_s > min_s:
                    bwd_cells, bwd_m = _build_direction_decisions(
                        cell.cell_id,
                        cell.slice_index,
                        "backward",
                        precomputed,
                        used,
                        cells_by_slice,
                        weights,
                        divisors,
                        alpha,
                        decisions_log,
                    )
                    merged_cells, merged_m = _merge_bi(fwd_cells, fwd_m, bwd_cells, bwd_m)
                else:
                    merged_cells, merged_m = fwd_cells, fwd_m
                if len(merged_cells) < MIN_TRACK_LENGTH:
                    continue
                ss = [s for s, _ in merged_cells]
                if any(ss[i + 1] - ss[i] != 1 for i in range(len(ss) - 1)):
                    continue
                round_tracks.append(
                    {
                        "cells": merged_cells,
                        "margins": merged_m,
                        "length": len(merged_cells),
                    }
                )
        if not round_tracks:
            break
        sel = _resolve_conflicts(round_tracks)
        all_tracks.extend(sel)
        for t in sel:
            for _, cid in t["cells"]:
                used.add(cid)
    return all_tracks, decisions_log


# --- 3. 4-D BAYESIAN OPTIMISATION (mirrors bayes_opt_margin_v10) ---

# Search space identical to the real-data BO bounds.
BO_SPACE = [
    Real(0.01, 0.25, name="RELATIVE_MARGIN_THRESHOLD"),
    Real(0.10, 0.60, name="DTW_WEIGHT"),
    Real(0.10, 0.60, name="OT_WEIGHT"),
    Real(0.10, 0.60, name="CENTROID_WEIGHT"),
] if HAS_SKOPT else []


def _normalise_weights(dtw_w: float, ot_w: float, cent_w: float) -> Tuple[float, float, float]:
    """L1-project the 3-D weight vector onto the unit simplex (matches the
    real-data BO, which passes ``w / sum(w)`` into the matcher)."""
    total = float(dtw_w + ot_w + cent_w)
    if total <= 0.0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return float(dtw_w / total), float(ot_w / total), float(cent_w / total)


def _seed_for_condition_seed(condition: str, seed: int, base_seed: int) -> int:
    """Stable random seed for ``gp_minimize``: ``base_seed + 100*seed`` plus
    a numeric tail derived from the condition name."""
    nums = [int(x) for x in re.findall(r"\d+", condition)]
    cond_offset = nums[0] if nums else (abs(hash(condition)) % 1000)
    return int(base_seed + 100 * int(seed) + (cond_offset % 1000))


def run_bo_for_one(
    condition: str,
    seed: int,
    *,
    phantom_root: str,
    method_root: str,
    cache_root: str,
    out_method_name: str = "HiDTW_full_BO",
    n_calls: int = 20,
    n_initial: int = 8,
    base_random_seed: int = 42000,
    include_gt_f1: bool = True,
    skip_existing: bool = False,
    force_precompute: bool = False,
) -> Dict[str, Any]:
    """End-to-end BO run for a single ``(condition, seed)`` simulation phantom."""
    if not HAS_SKOPT:
        raise RuntimeError(
            "scikit-optimize (skopt) is required for HiDTW BO; install with "
            f"`pip install scikit-optimize`. Original error: {_SKOPT_IMPORT_ERROR!r}"
        )

    phantom_dir = os.path.join(phantom_root, condition, f"seed{seed}")
    if not os.path.isdir(phantom_dir):
        return {
            "condition": condition,
            "seed": int(seed),
            "skipped": True,
            "reason": f"phantom dir missing: {phantom_dir}",
        }

    out_base = os.path.join(method_root, condition, f"seed{seed}")
    out_dir = os.path.join(out_base, out_method_name)
    best_path = os.path.join(out_dir, "best_weights.json")
    if skip_existing and os.path.isfile(best_path):
        try:
            with open(best_path, encoding="utf-8") as f:
                cached = json.load(f)
            print(f"  [skip] {best_path} already exists.")
            return {
                "condition": condition,
                "seed": int(seed),
                "skipped": True,
                "reason": "already exists",
                "best": cached.get("best", {}),
            }
        except Exception:
            pass
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== BO  {condition} / seed{seed}  ->  {out_dir} ===")

    # ---- load cells ---------------------------------------------------------
    cells_by_slice = v10.load_cells_from_csv_files(phantom_dir)
    n_cells = sum(len(v) for v in cells_by_slice.values())
    print(f"  cells loaded: {n_cells} across slices {sorted(cells_by_slice.keys())}")

    # ---- precompute (cached if available) -----------------------------------
    cache_file = os.path.join(cache_root, condition, f"seed{seed}", "precomputed_costs.pkl")
    print(f"  pre-compute costs ...")
    precomputed = precompute_all_costs(cells_by_slice, cache_file, force=force_precompute)

    divisors = calibrate_divisors(precomputed)
    print(f"  divisors: {divisors}")

    # ---- (optional) GT pair set for offline diagnostics ---------------------
    gt_pairs_set = None
    if include_gt_f1:
        try:
            gt_df = pd.read_csv(os.path.join(phantom_dir, "ground_truth.csv"))
            gt_df["slice"] = gt_df["slice"].astype(int)
            gt_df["cell_id"] = gt_df["cell_id"].astype(str)
            gt_pairs_set = _gt_pairs_from_df(gt_df)
            print(f"  GT adjacent pairs: {len(gt_pairs_set)} (offline diagnostic only)")
        except Exception as e:
            print(f"  [warn] could not load GT for diagnostics: {e}")
            gt_pairs_set = None

    # ---- closure-state captured by the skopt objective ----------------------
    history: List[Dict[str, Any]] = []
    iteration_counter = {"i": 0}

    @use_named_args(BO_SPACE)
    def objective(**params: float) -> float:
        rm_thresh = float(params["RELATIVE_MARGIN_THRESHOLD"])
        dtw_w = float(params["DTW_WEIGHT"])
        ot_w = float(params["OT_WEIGHT"])
        cent_w = float(params["CENTROID_WEIGHT"])
        dtw_n, ot_n, cent_n = _normalise_weights(dtw_w, ot_w, cent_w)

        weights = {"dtw": dtw_n, "ot": ot_n, "centroid": cent_n}
        t_iter = time.time()
        tracks, decisions_log = build_tracks_with_decisions(
            cells_by_slice, precomputed, weights, divisors, alpha=rm_thresh
        )
        scores = compute_global_score(tracks, cells_by_slice, decisions_log)
        elapsed = time.time() - t_iter

        # Offline GT pair-F1: diagnostic only, never enters the BO objective.
        diag = {"offline_gt_pair_f1": float("nan"),
                "offline_gt_pair_precision": float("nan"),
                "offline_gt_pair_recall": float("nan"),
                "offline_gt_pair_n_pred": -1}
        if gt_pairs_set is not None:
            f1, p, r, n_pred = _pair_f1_from_tracks(tracks, gt_pairs_set)
            diag = {
                "offline_gt_pair_f1": float(f1),
                "offline_gt_pair_precision": float(p),
                "offline_gt_pair_recall": float(r),
                "offline_gt_pair_n_pred": int(n_pred),
            }

        iteration = iteration_counter["i"]
        iteration_counter["i"] += 1
        row = {
            "iteration": int(iteration),
            "alpha_rm_threshold": rm_thresh,
            "dtw_raw": dtw_w,
            "ot_raw": ot_w,
            "centroid_raw": cent_w,
            "w_dtw": dtw_n,
            "w_ot": ot_n,
            "w_centroid": cent_n,
            **scores,
            **diag,
            "iter_elapsed_sec": float(elapsed),
        }
        history.append(row)
        if include_gt_f1:
            print(
                f"    [{iteration:02d}] RM={rm_thresh:.4f}  "
                f"w=({dtw_n:.2f},{ot_n:.2f},{cent_n:.2f})  "
                f"score={scores['global_score']:6.2f}  "
                f"(conf={scores['global_confidence_score']:5.1f}, "
                f"cov={scores['coverage_score']:5.1f}, "
                f"len={scores['length_score']:5.1f})  "
                f"f1[diag]={diag['offline_gt_pair_f1']:.3f}  ({elapsed:.1f}s)"
            )
        else:
            print(
                f"    [{iteration:02d}] RM={rm_thresh:.4f}  "
                f"w=({dtw_n:.2f},{ot_n:.2f},{cent_n:.2f})  "
                f"score={scores['global_score']:6.2f}  ({elapsed:.1f}s)"
            )
        return -float(scores["global_score"])  # gp_minimize minimises

    # ---- run gp_minimize ----------------------------------------------------
    rseed = _seed_for_condition_seed(condition, seed, base_random_seed)
    print(
        f"  [BO] gp_minimize  n_calls={n_calls}  n_initial={n_initial}  "
        f"random_state={rseed}"
    )
    t_bo = time.time()
    result = gp_minimize(
        objective,
        BO_SPACE,
        n_calls=int(n_calls),
        n_initial_points=int(n_initial),
        random_state=int(rseed),
        verbose=False,
        n_jobs=1,
    )
    bo_elapsed = time.time() - t_bo
    best_idx = int(np.argmin(np.asarray(result.func_vals, dtype=float)))
    best_row = history[best_idx] if 0 <= best_idx < len(history) else {}
    print(
        f"  [BO] best iter={best_idx}  global_score={-float(result.fun):.2f}  "
        f"({bo_elapsed:.1f}s)"
    )

    # ---- final tracks at best params (rebuild + write tracks.csv) -----------
    best_alpha = float(result.x[0])
    best_dtw_n, best_ot_n, best_cent_n = _normalise_weights(
        float(result.x[1]), float(result.x[2]), float(result.x[3])
    )
    best_weights = {"dtw": best_dtw_n, "ot": best_ot_n, "centroid": best_cent_n}
    final_tracks, final_decisions = build_tracks_with_decisions(
        cells_by_slice, precomputed, best_weights, divisors, alpha=best_alpha
    )
    final_scores = compute_global_score(final_tracks, cells_by_slice, final_decisions)
    if gt_pairs_set is not None:
        f1, p, r, n_pred = _pair_f1_from_tracks(final_tracks, gt_pairs_set)
        final_diag = {
            "offline_gt_pair_f1": float(f1),
            "offline_gt_pair_precision": float(p),
            "offline_gt_pair_recall": float(r),
            "offline_gt_pair_n_pred": int(n_pred),
        }
    else:
        final_diag = {
            "offline_gt_pair_f1": float("nan"),
            "offline_gt_pair_precision": float("nan"),
            "offline_gt_pair_recall": float("nan"),
            "offline_gt_pair_n_pred": -1,
        }

    df_tracks = tracks_to_dataframe(final_tracks)
    df_tracks.to_csv(os.path.join(out_dir, "tracks.csv"), index=False)

    best_payload = {
        "best": {
            "alpha": best_alpha,
            "dtw": best_dtw_n,
            "ot": best_ot_n,
            "centroid": best_cent_n,
            **final_scores,
            **final_diag,
        },
        "search": {
            "objective": "global_score (40% margin + 40% coverage + 20% length)",
            "space": [
                {"name": "RELATIVE_MARGIN_THRESHOLD", "low": 0.01, "high": 0.25},
                {"name": "DTW_WEIGHT", "low": 0.10, "high": 0.60},
                {"name": "OT_WEIGHT", "low": 0.10, "high": 0.60},
                {"name": "CENTROID_WEIGHT", "low": 0.10, "high": 0.60},
            ],
            "n_calls": int(n_calls),
            "n_initial_points": int(n_initial),
            "random_state": int(rseed),
            "base_random_seed": int(base_random_seed),
            "weight_normalisation": "L1 (post-hoc projection onto simplex)",
        },
        "n_evaluations": int(len(history)),
        "elapsed_sec": float(bo_elapsed),
        "include_gt_f1_diagnostic": bool(include_gt_f1),
        "phantom_dir": phantom_dir,
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "best_weights.json"), "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    # ---- BO history (csv + json) -------------------------------------------
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "bo_history.csv"), index=False)
    history_json = {
        "iterations": [r["iteration"] for r in history],
        "params_raw": [
            [r["alpha_rm_threshold"], r["dtw_raw"], r["ot_raw"], r["centroid_raw"]]
            for r in history
        ],
        "params_normalised": [
            [r["alpha_rm_threshold"], r["w_dtw"], r["w_ot"], r["w_centroid"]]
            for r in history
        ],
        "scores": [r["global_score"] for r in history],
        "best_score": float(-result.fun),
        "best_params": {
            "RELATIVE_MARGIN_THRESHOLD": float(best_alpha),
            "DTW_WEIGHT": float(best_dtw_n),
            "OT_WEIGHT": float(best_ot_n),
            "CENTROID_WEIGHT": float(best_cent_n),
        },
        "optimization_time_hours": float(bo_elapsed / 3600.0),
        "random_state": int(rseed),
    }
    with open(os.path.join(out_dir, "bo_history.json"), "w", encoding="utf-8") as f:
        json.dump(history_json, f, indent=2)

    # ---- convergence plot --------------------------------------------------
    try:
        scores_arr = np.asarray([r["global_score"] for r in history], dtype=float)
        running_best = np.maximum.accumulate(scores_arr)
        fig, ax = plt.subplots(figsize=(95 / 25.4, 60 / 25.4))
        ax.plot(scores_arr, "o-", lw=0.9, ms=4, color="#4E79A7", label="per-iter score")
        ax.plot(running_best, "-", lw=1.3, color="#D55E00", label="running best")
        ax.axhline(float(-result.fun), ls="--", color="0.4", lw=0.8,
                   label=f"best={float(-result.fun):.2f}")
        ax.set_xlabel("BO iteration", fontsize=8)
        ax.set_ylabel("global_score", fontsize=8)
        ax.set_title(f"{condition} / seed{seed}  -  HiDTW BO (unsupervised)", fontsize=9)
        ax.tick_params(labelsize=7, length=3, width=0.7)
        ax.legend(fontsize=6.5, frameon=False)
        ax.grid(axis="y", color="0.92", lw=0.4)
        ax.set_axisbelow(True)
        fig.savefig(os.path.join(out_dir, "convergence.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] convergence plot failed: {e}")

    summary_for_aggregator = {
        "condition": condition,
        "seed": int(seed),
        "out_dir": out_dir,
        "best": best_payload["best"],
        "n_evaluations": int(len(history)),
        "n_initial_points": int(n_initial),
        "n_calls": int(n_calls),
        "random_state": int(rseed),
        "bo_elapsed_sec": float(bo_elapsed),
        "best_weights_json_rel": os.path.relpath(
            os.path.join(out_dir, "best_weights.json"), method_root
        ).replace(os.sep, "/"),
    }
    return summary_for_aggregator


# --- 4. AGGREGATE SUMMARY ACROSS (condition, seed) ---

# Keep in sync with simulation01_evaluate.CONDITION_ALIAS.
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
    "L2_per_slice_warp":          "C15_slice_warp",
    "L2_per_slice_rotation":      "C16_slice_rotation",
    "L2_per_slice_contour_noise": "C17_contour_noise",
    "L2_dtw_stress_combo":        "C18_dtw_stress_combo",
}


def _aggregate_bo_summaries(method_root: str, out_method_name: str) -> str:
    """Collect every ``best_weights.json`` under ``method_root`` into one CSV."""
    rows: List[Dict[str, Any]] = []
    pattern = os.path.join(method_root, "*", "seed*", out_method_name, "best_weights.json")
    from glob import glob

    for fp in sorted(glob(pattern)):
        rel = os.path.relpath(fp, method_root).replace(os.sep, "/")
        parts = fp.split(os.sep)
        cond_raw = parts[-4]
        seed_dir = parts[-3]
        try:
            seed = int(seed_dir.replace("seed", ""))
        except Exception:
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        b = payload.get("best", {})
        sch = payload.get("search", {})
        rows.append(
            {
                "condition_raw": cond_raw,
                "condition_alias": CONDITION_ALIAS.get(cond_raw, cond_raw),
                "seed": seed,
                "w_dtw": float(b.get("dtw", float("nan"))),
                "w_ot": float(b.get("ot", float("nan"))),
                "w_centroid": float(b.get("centroid", float("nan"))),
                "alpha_rm_threshold": float(b.get("alpha", float("nan"))),
                "bo_global_score": float(b.get("global_score", float("nan"))),
                "bo_global_confidence_score": float(b.get("global_confidence_score", float("nan"))),
                "bo_coverage_score": float(b.get("coverage_score", float("nan"))),
                "bo_length_score": float(b.get("length_score", float("nan"))),
                "bo_global_avg_margin": float(b.get("global_avg_margin", float("nan"))),
                "bo_coverage_ratio": float(b.get("coverage_ratio", float("nan"))),
                "bo_average_length": float(b.get("average_length", float("nan"))),
                "bo_n_tracks": int(b.get("n_tracks", -1)),
                "bo_total_decisions": int(b.get("total_decisions", -1)),
                "bo_low_confidence_ratio": float(b.get("low_confidence_ratio", float("nan"))),
                "bo_no_candidate_ratio": float(b.get("no_candidate_ratio", float("nan"))),
                "offline_gt_pair_f1": float(b.get("offline_gt_pair_f1", float("nan"))),
                "offline_gt_pair_precision": float(b.get("offline_gt_pair_precision", float("nan"))),
                "offline_gt_pair_recall": float(b.get("offline_gt_pair_recall", float("nan"))),
                "n_bo_calls": int(sch.get("n_calls", -1)),
                "n_initial_points": int(sch.get("n_initial_points", -1)),
                "bo_random_seed": int(sch.get("random_state", -1)),
                "bo_elapsed_sec": float(payload.get("elapsed_sec", float("nan"))),
                "best_weights_json": rel,
            }
        )
    if not rows:
        print("  [aggregate] no best_weights.json files found; skip CSV.")
        return ""

    df = pd.DataFrame(rows).sort_values(["condition_raw", "seed"]).reset_index(drop=True)
    out_csv = os.path.join(method_root, "hidtw_full_bo_summary_by_condition_seed.csv")
    df.to_csv(out_csv, index=False, float_format="%.12g")
    print(f"  [aggregate] wrote {len(df)} rows -> {out_csv}")
    return out_csv


# --- 5. CLI ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulation 1 - HiDTW only, tuned via Bayesian optimisation on the "
            "unsupervised global_score (no GT in BO loop)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="Pin CUDA_VISIBLE_DEVICES before importing torch.")
    parser.add_argument("--phantom-root", default="simulation01_results/phantoms")
    parser.add_argument("--method-root", default="simulation01_results/methods")
    parser.add_argument("--cache-root", default="simulation01_results/cache")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Subset of conditions; default = all in CONDITIONS registry.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds to run; default = 0..n_seeds-1 from --seed-start.")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-calls", type=int, default=20,
                        help="Total BO evaluations per (condition, seed). Default 20.")
    parser.add_argument("--n-initial", type=int, default=8,
                        help="Random initial points before the GP kicks in. Default 8.")
    parser.add_argument("--base-random-seed", type=int, default=42000,
                        help="Seed offset for gp_minimize random_state per (cond, seed).")
    parser.add_argument("--out-method-name", default="HiDTW_full_BO",
                        help="Sub-directory name written under each seed dir.")
    parser.add_argument("--no-gt-f1", action="store_true",
                        help="Skip the offline GT pair-F1 diagnostic (faster, "
                             "but loses the column that lets you compare BO "
                             "vs oracle weights).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (cond, seed) entries whose best_weights.json "
                             "already exists under --out-method-name.")
    parser.add_argument("--force-precompute", action="store_true",
                        help="Recompute the GPU cost cache even if it exists "
                             "(only matters for cache misses; BO never re-runs DTW).")
    args = parser.parse_args()

    if not HAS_SKOPT:
        raise SystemExit(
            "ERROR: scikit-optimize not available. Install with "
            f"`pip install scikit-optimize` (raised: {_SKOPT_IMPORT_ERROR!r})"
        )

    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    cond_names = args.conditions if args.conditions else list(CONDITIONS.keys())
    if args.seeds is not None:
        seeds = list(args.seeds)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    valid_conds = []
    for c in cond_names:
        if c in CONDITIONS:
            valid_conds.append(c)
        else:
            print(f"[skip] unknown condition {c}")

    print("Simulation 1 - HiDTW BO (unsupervised global_score)")
    print(f"  conditions:  {valid_conds}")
    print(f"  seeds:       {seeds}")
    print(f"  n_calls:     {args.n_calls}")
    print(f"  n_initial:   {args.n_initial}")
    print(f"  out method:  {args.out_method_name}")
    print(f"  phantom:     {args.phantom_root}")
    print(f"  method-root: {args.method_root}")
    print(f"  cache:       {args.cache_root}")
    print(f"  gt-f1 diag:  {not args.no_gt_f1}")
    print(f"  skip-exist:  {args.skip_existing}")

    summaries: List[Dict[str, Any]] = []
    t_total = time.time()
    for cond in valid_conds:
        for seed in seeds:
            try:
                s = run_bo_for_one(
                    cond,
                    int(seed),
                    phantom_root=args.phantom_root,
                    method_root=args.method_root,
                    cache_root=args.cache_root,
                    out_method_name=args.out_method_name,
                    n_calls=int(args.n_calls),
                    n_initial=int(args.n_initial),
                    base_random_seed=int(args.base_random_seed),
                    include_gt_f1=not args.no_gt_f1,
                    skip_existing=bool(args.skip_existing),
                    force_precompute=bool(args.force_precompute),
                )
            except Exception as e:
                import traceback

                print(f"!!! FAILED {cond}/seed{seed}: {e}\n{traceback.format_exc()}")
                s = {"condition": cond, "seed": int(seed), "error": str(e)}
            summaries.append(s)

    _aggregate_bo_summaries(args.method_root, args.out_method_name)

    runs_path = os.path.join(args.method_root, "_hidtw_bo_runs_summary.json")
    os.makedirs(args.method_root, exist_ok=True)
    with open(runs_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nAll BO runs summary -> {runs_path}")
    print(f"Total elapsed: {(time.time() - t_total) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
