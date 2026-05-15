"""HiDTW Tile046 demo: GPU reconstruction, Bayesian optimisation, tutorial helpers.

Core scripts (same package directory):

- ``cell_reconstruction_runGPU_3Drevised_v10.py`` — contour Soft-DTW + OT matching
- ``bayes_opt_margin_v10.py`` — margin-based hyper-parameter search

Tutorial notebooks typically do::

    import hidtw_pipeline as th

Plotting helpers are loaded **lazily** via :mod:`hidtw_pipeline.notebook_tools` so that::

    import hidtw_pipeline.cell_reconstruction_runGPU_3Drevised_v10

does not pull in matplotlib.
"""

from __future__ import annotations

import importlib
from typing import Any

_NOTEBOOK_EXPORTS = (
    "PALETTE",
    "global_xy_bounds",
    "plot_raw_cells_grid",
    "plot_zoom_window",
    "slice_summary_table",
    "evaluation_radar",
    "margin_distribution",
    "sample_tracks_3d",
    "sample_tracks_xy",
    "track_length_histogram",
    "bo_convergence",
    "bo_param_vs_score",
    "compare_track_lengths",
    "aggregate_tracks_to_csr",
    "build_cell_lookup",
    "gene_universe",
    "load_tile_expression_long",
    "write_matrix_csv_gz",
    "write_tracks_summary_csv",
    "gene_z_traces",
    "hist_n_genes_per_cell",
    "hist_total_umi_per_cell",
)

__all__ = list(_NOTEBOOK_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _NOTEBOOK_EXPORTS:
        nt = importlib.import_module(".notebook_tools", __name__)
        return getattr(nt, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
