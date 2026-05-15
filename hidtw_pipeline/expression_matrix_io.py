"""Per-track UMI aggregation + CSV / gzip / summary I/O.

Public functions:
    load_tile_expression_long   - concatenate `expression-T<tile>-S*.csv` (TSV)
    build_cell_lookup           - (z, cell_label) -> {feature_index: total_exp}
    gene_universe               - sorted feature_index list across the tile
    aggregate_tracks_to_csr     - per-track sum into a sparse cells x genes CSR
    write_matrix_csv_gz         - stream a sparse CSR to a .csv.gz wide CSV
    write_tracks_summary_csv    - write the same schema as the production
                                  pipeline's `<tile>_all_tracks_summary.csv`
"""

from __future__ import annotations

import csv as _csv
import gzip
import io
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


def load_tile_expression_long(expr_dir: Path, tile_pref: str) -> pd.DataFrame:
    """Concatenate every `expression-T<tile>-S*.csv` (TSV) under `expr_dir`."""
    frames: List[pd.DataFrame] = []
    pattern = re.compile(rf"expression-T{tile_pref}-S(\d+)\.csv$")
    for p in sorted(expr_dir.iterdir()):
        if not pattern.search(p.name):
            continue
        df = pd.read_csv(
            p, sep="\t",
            usecols=["feature_index", "cell_label", "z", "total_exp"],
            dtype={"feature_index": np.int64, "cell_label": np.int64,
                   "z": np.int64, "total_exp": np.float64},
        )
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no expression-T{tile_pref}-S*.csv under {expr_dir}")
    return pd.concat(frames, ignore_index=True)


def build_cell_lookup(
    expr_long: pd.DataFrame,
) -> Dict[Tuple[int, int], Dict[int, float]]:
    g = (expr_long.groupby(["z", "cell_label", "feature_index"], sort=False)["total_exp"]
         .sum().reset_index())
    lut: Dict[Tuple[int, int], Dict[int, float]] = {}
    for z, lab, feat, val in g[["z", "cell_label", "feature_index", "total_exp"]].itertuples(
        index=False, name=None
    ):
        slot = lut.setdefault((int(z), int(lab)), {})
        slot[int(feat)] = float(val)
    return lut


def gene_universe(expr_long: pd.DataFrame) -> List[int]:
    return sorted(int(g) for g in expr_long["feature_index"].unique())


def aggregate_tracks_to_csr(
    tracks: list,
    cell_lut: Dict[Tuple[int, int], Dict[int, float]],
    universe: List[int],
    tile_prefix: str,
) -> Tuple[List[str], csr_matrix, pd.DataFrame]:
    """Return (cell_ids, CSR cells x genes, meta DataFrame).

    Each track is identified by `T{tile}_TR{track_id:06d}` (matching the
    convention in `build_3d_cell_expression.py`).
    """
    gene_to_col = {g: i for i, g in enumerate(universe)}
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    cell_ids: List[str] = []
    meta_rows: List[Dict[str, object]] = []

    for tid, track in enumerate(tracks, start=1):
        slices = sorted(track.cells.keys())
        members = [(s, int(track.cells[s].cell_label)) for s in slices
                   if hasattr(track.cells[s], "cell_label")]
        if not members:
            # Fall back to parsing cell_id 'T046S03_0012' when cell_label is absent.
            members = []
            for s in slices:
                cid = str(track.cells[s].cell_id)
                m = re.search(r"T(\d+)S(\d+)[_-](\d+)", cid)
                if m:
                    members.append((int(m.group(2)), int(m.group(3))))
        agg: Dict[int, float] = defaultdict(float)
        n_match = 0
        for key in members:
            slot = cell_lut.get((int(key[0]), int(key[1])))
            if slot is None:
                continue
            n_match += 1
            for f, v in slot.items():
                agg[f] += v
        if not agg:
            continue
        row_idx = len(cell_ids)
        cell_id = f"T{tile_prefix}_TR{tid:06d}"
        cell_ids.append(cell_id)
        for f, v in agg.items():
            col = gene_to_col.get(int(f))
            if col is None:
                continue
            rows.append(row_idx)
            cols.append(col)
            vals.append(float(v))
        meta_rows.append({
            "cell_id": cell_id,
            "tile": tile_prefix,
            "track_id": int(tid),
            "track_length": int(track.get_layer_count()),
            "n_member_cells": int(len(members)),
            "n_member_cells_with_expr": int(n_match),
            "total_umi": float(sum(agg.values())),
            "n_genes": int(len(agg)),
            "start_slice": int(track.start_slice),
            "end_slice": int(track.end_slice),
            "avg_margin": float(track.get_quality_score()),
        })
    csr = coo_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(len(cell_ids), len(universe)),
    ).tocsr()
    meta = pd.DataFrame(meta_rows)
    return cell_ids, csr, meta


def write_matrix_csv_gz(
    out_path: Path,
    cell_ids: List[str],
    csr: csr_matrix,
    universe: List[int],
    chunk_rows: int = 256,
) -> None:
    n_rows, n_cols = csr.shape
    assert n_cols == len(universe)
    assert n_rows == len(cell_ids)
    with gzip.open(out_path, "wb", compresslevel=6) as gz:
        header = "cell_id," + ",".join(f"gene_{g}" for g in universe) + "\n"
        gz.write(header.encode("ascii"))
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            block = csr[start:end].toarray().astype(np.int64, copy=False)
            buf = io.StringIO()
            for i in range(end - start):
                buf.write(cell_ids[start + i])
                buf.write(",")
                row = block[i]
                buf.write(",".join("%d" % v for v in row))
                buf.write("\n")
            gz.write(buf.getvalue().encode("ascii"))


def write_tracks_summary_csv(
    tracks: list, out_csv: Path, tile_prefix: str,
) -> None:
    """Write a summary CSV in the same schema as the production pipeline's
    `<tile>_all_tracks_summary.csv` (so downstream scripts can re-use it)."""
    with open(out_csv, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow([
            "track_id", "start_slice", "end_slice", "layer_count",
            "avg_margin", "min_margin", "max_margin", "std_margin", "direction",
            "cells_area_mean", "cells_area_std", "cell_ids",
        ])
        for tid, track in enumerate(tracks, start=1):
            slices = sorted(track.cells.keys())
            direction = "ascending" if (len(slices) >= 2 and slices[-1] > slices[0]) else \
                        ("descending" if len(slices) >= 2 else "unknown")
            qd = track.get_quality_details()
            areas = np.asarray([track.cells[s].area for s in slices])
            cell_ids = "|".join(str(track.cells[s].cell_id) for s in slices)
            w.writerow([
                tid,
                track.start_slice, track.end_slice, track.get_layer_count(),
                f"{qd['avg_margin']:.6f}", f"{qd['min_margin']:.6f}",
                f"{qd['max_margin']:.6f}", f"{qd['std_margin']:.6f}",
                direction,
                f"{float(areas.mean()):.2f}",
                f"{float(areas.std()):.2f}" if areas.size > 1 else "0.00",
                cell_ids,
            ])
