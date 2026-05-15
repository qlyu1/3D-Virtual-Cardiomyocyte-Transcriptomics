"""Tutorial plotting / aggregation helpers for ``tutorial_HiDTW_Tile046.ipynb``.

Import via lazy-loading from ``hidtw_pipeline``::

    import hidtw_pipeline as th

Submodules:
    color_palette           - colour map shared across all figures
    slice_cell_plots        - per-slice statistics + raw-cell visualisations
    track_quality_plots     - per-track visualisations (length, margins, xy / 3D, radar)
    bayesian_opt_plots      - Bayesian-optimisation convergence and trial scatter plots
    expression_matrix_io    - per-track UMI aggregation + CSV / gzip / summary I/O
    expression_sanity_plots - sanity-check histograms and gene-vs-z trace plots
"""

from .color_palette import PALETTE
from .slice_cell_plots import (
    global_xy_bounds,
    plot_raw_cells_grid,
    plot_zoom_window,
    slice_summary_table,
)
from .track_quality_plots import (
    evaluation_radar,
    margin_distribution,
    sample_tracks_3d,
    sample_tracks_xy,
    track_length_histogram,
)
from .bayesian_opt_plots import (
    bo_convergence,
    bo_param_vs_score,
    compare_track_lengths,
)
from .expression_matrix_io import (
    aggregate_tracks_to_csr,
    build_cell_lookup,
    gene_universe,
    load_tile_expression_long,
    write_matrix_csv_gz,
    write_tracks_summary_csv,
)
from .expression_sanity_plots import (
    gene_z_traces,
    hist_n_genes_per_cell,
    hist_total_umi_per_cell,
)

__all__ = [
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
]
