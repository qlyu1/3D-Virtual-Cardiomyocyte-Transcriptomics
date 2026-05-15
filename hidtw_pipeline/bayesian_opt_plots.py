"""Visualisations for Bayesian-optimisation runs.

Public functions:
    bo_convergence       - score per iteration + best-so-far envelope
    bo_param_vs_score    - one panel per parameter showing trial scatter and
                           the best point as a star
    compare_track_lengths - overlay multiple reconstruction runs (e.g. manual
                           vs BO-best) on a single histogram
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .color_palette import PALETTE, finalize_notebook_figure


def bo_convergence(
    history: pd.DataFrame, out_pdf: Path, title: str,
    score_col: str = "Global_Score",
) -> None:
    scores = history[score_col].to_numpy()
    best = np.maximum.accumulate(scores)
    iters = history["Iteration"].to_numpy() + 1
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.plot(iters, scores, "o", ms=4, color=PALETTE["secondary"], alpha=0.55,
            label="trial score")
    ax.plot(iters, best, "-", lw=1.4, color=PALETTE["primary"], label="best so far")
    ax.set_xlabel("BO iteration", fontsize=8)
    ax.set_ylabel("Global score (0..100)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)


def bo_param_vs_score(
    history: pd.DataFrame, out_pdf: Path, title: str,
    score_col: str = "Global_Score",
    param_cols: Tuple[str, ...] = (
        "RELATIVE_MARGIN_THRESHOLD", "DTW_WEIGHT", "OT_WEIGHT", "CENTROID_WEIGHT"),
) -> None:
    scores = history[score_col].to_numpy()
    score_norm = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-9)
    fig, axes = plt.subplots(
        1, len(param_cols),
        figsize=(2.6 * len(param_cols) + 1.5, 2.6),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    bidx = int(np.argmax(scores))
    for ax, col in zip(axes, param_cols):
        ax.scatter(history[col], scores, c=score_norm, cmap="viridis",
                   s=24, edgecolor="black", lw=0.4)
        ax.scatter(history[col].iloc[bidx], scores[bidx], s=70, marker="*",
                   facecolors=PALETTE["accent"], edgecolors="black",
                   linewidths=0.6, zorder=5, label="best")
        ax.set_xlabel(col, fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
    axes[0].set_ylabel("Global score (0..100)", fontsize=8)
    axes[-1].legend(fontsize=6, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)


def compare_track_lengths(
    runs: Dict[str, list], out_pdf: Path, title: str,
    palette: Tuple[str, str] = (PALETTE["secondary"], PALETTE["primary"]),
) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    for (name, tracks), color in zip(runs.items(), palette):
        lengths = np.asarray([t.get_layer_count() for t in tracks])
        if not lengths.size:
            continue
        bins = np.arange(2.5, max(lengths.max(), 10) + 1.5, 1.0)
        ax.hist(lengths, bins=bins, alpha=0.55, color=color,
                edgecolor="black", lw=0.6,
                label=f"{name}  (n={len(tracks)})")
    ax.set_xlabel("Track length (layers)", fontsize=8)
    ax.set_ylabel("# tracks", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)
