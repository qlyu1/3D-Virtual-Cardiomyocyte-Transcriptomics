"""Sanity-check plots for the per-track expression matrix.

Public functions:
    hist_total_umi_per_cell - distribution of total UMI per reconstructed
                              3D cell
    hist_n_genes_per_cell   - distribution of distinct genes detected per
                              3D cell
    gene_z_traces           - per-track expression(z) traces for a small
                              dictionary of marker genes (e.g. Glul, Gja1)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .color_palette import PALETTE, finalize_notebook_figure


def hist_total_umi_per_cell(
    meta: pd.DataFrame, out_pdf: Path, title: str,
    color: str = PALETTE["primary"],
) -> None:
    vals = meta["total_umi"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    bins = np.linspace(0, np.quantile(vals, 0.99), 60)
    ax.hist(vals, bins=bins, color=color, alpha=0.85, edgecolor="black", lw=0.5)
    ax.set_xlabel("Total UMI per track", fontsize=8)
    ax.set_ylabel("# tracks", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)


def hist_n_genes_per_cell(
    meta: pd.DataFrame, out_pdf: Path, title: str,
    color: str = PALETTE["secondary"],
) -> None:
    vals = meta["n_genes"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    bins = np.linspace(0, vals.max() if vals.size else 1, 60)
    ax.hist(vals, bins=bins, color=color, alpha=0.85, edgecolor="black", lw=0.5)
    ax.set_xlabel("# distinct genes detected per track", fontsize=8)
    ax.set_ylabel("# tracks", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)


def gene_z_traces(
    tracks: list,
    cell_lut: Dict[Tuple[int, int], Dict[int, float]],
    gene_index_map: Dict[str, int],
    out_pdf: Path,
    title: str,
    n_show: int = 30,
    seed: int = 7,
    cmap_name: str = "viridis",
) -> None:
    """For each gene in `gene_index_map`, draw the per-z expression trace
    of `n_show` random tracks. Each gene gets its own subplot row."""
    rng = np.random.default_rng(seed)
    candidates = [t for t in tracks if t.get_layer_count() >= 5]
    if not candidates:
        print("no tracks long enough for a z-trace plot")
        return
    if len(candidates) > n_show:
        idx = rng.choice(len(candidates), size=n_show, replace=False)
        chosen = [candidates[int(i)] for i in idx]
    else:
        chosen = candidates

    cmap = plt.get_cmap(cmap_name)
    n_genes = len(gene_index_map)
    fig, axes = plt.subplots(n_genes, 1, figsize=(5.0, 2.4 * n_genes), sharex=True)
    if n_genes == 1:
        axes = [axes]
    for ax, (sym, fi) in zip(axes, gene_index_map.items()):
        for k, t in enumerate(chosen):
            zs = sorted(t.cells.keys())
            xs: List[int] = []
            ys: List[float] = []
            for s in zs:
                lab = getattr(t.cells[s], "cell_label", None)
                if lab is None:
                    m = re.search(r"T(\d+)S(\d+)[_-](\d+)", str(t.cells[s].cell_id))
                    if not m:
                        continue
                    lab = int(m.group(3))
                slot = cell_lut.get((int(s), int(lab)))
                if slot is None:
                    continue
                xs.append(int(s))
                ys.append(float(slot.get(int(fi), 0.0)))
            if not xs:
                continue
            ax.plot(xs, ys, color=cmap((k + 0.5) / len(chosen)),
                    lw=0.6, alpha=0.75)
        ax.set_ylabel(f"{sym}\nUMI / member", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
    axes[-1].set_xlabel("Z slice", fontsize=8)
    axes[0].set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_pdf)
    finalize_notebook_figure(fig)
