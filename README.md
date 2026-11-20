# HiDTW: GPU-Accelerated 3D Cell Reconstruction

HiDTW links 2D segmented cells across consecutive sections into 3D tracks by
combining contour Soft-DTW, neighbour-cluster Optimal Transport, and centroid
distance on the GPU. A dual-normalisation scheme keeps the three terms
comparable, and Bayesian optimisation tunes the four matching weights per
dataset.

## Repository layout

```
.
├── hidtw_pipeline/                      # Reconstruction + BO package
│   ├── cell_reconstruction_runGPU_3Drevised_v10.py
│   ├── bayes_opt_margin_v10.py
│   └── ...                              # plot / I/O helpers for the notebook
├── soft_dtw_cuda.py                     # CUDA Soft-DTW kernel (Maghoumi)
├── extract_all_features_from_images.py  # 2D feature extraction from raw images
├── tutorial_HiDTW_demodata_Tile046.ipynb
├── demo_data/                           # 10-slice Tile046 example
│   ├── Tile046/                         # 046_S*_features.csv
│   └── 046/                             # expression-T046-S*.csv
├── model_3d_construction/               # mesh & GLB export
├── simulation/                          # synthetic phantoms + baselines
├── benchmark/                           # real-data baseline runner
├── paper_figures/                       # Figure 1 / 4 / 5 / 6 notebooks & R scripts
├── requirements.txt
└── LICENSE
```

`hidtw_pipeline/` holds the GPU matcher (`cell_reconstruction_runGPU_3Drevised_v10.py`)
and the Bayesian optimiser (`bayes_opt_margin_v10.py`); the rest of the package
exposes the plotting and aggregation helpers used by the tutorial notebook.

## Installation

```bash
git clone https://github.com/qlyu1/3D-Virtual-Cardiomyocyte-Transcriptomics.git
cd 3D-Virtual-Cardiomyocyte-Transcriptomics
conda create -n hidtw python=3.10 -y && conda activate hidtw
pip install -r requirements.txt jupyterlab ipykernel
```

A CUDA-capable GPU is required: Soft-DTW and Sinkhorn OT both run on the GPU
and the matcher aborts on CPU. Successfully run on Ubuntu 22.04, Python 3.10,
PyTorch 2.1.2, CUDA 11.8, one 48 GB vGPU, 25 vCPUs (Intel Xeon Platinum 8470Q),
and 92 GB RAM; other compatible configurations may also work.

Expected installation time: approximately 15 minutes, including download and
installation of additional dependencies, excluding image deployment and
repository/demo-data downloads.

## Quick start

The tutorial notebook walks through loading, calibration, reconstruction,
Bayesian optimisation, and per-track UMI aggregation on the bundled Tile046
slices:

```bash
jupyter lab tutorial_HiDTW_demodata_Tile046.ipynb
```

For the manual demo, run sections **1–8 → 10b → 11–12**. For Bayesian
optimisation, run **1–10 → 11–12** instead.

Expected demo runtime: approximately **48 minutes**.

CLI-only run (the matcher discovers data via `glob.glob("Tile*")` in cwd, so
symlink or copy `demo_data/Tile046` to `./Tile046` first):

```bash
ln -s demo_data/Tile046 Tile046
python -m hidtw_pipeline.bayes_opt_margin_v10 \
    --region Tile046 --gpu 0 --n-calls 30 --n-initial 8
```

## Outputs

Demo outputs in `tutorial_outputs/Tile046/`: `Tile046.csv.gz` (expression),
`Tile046_meta.csv`, `Tile046_all_tracks_summary.csv`, and QC plots.

| File | Contents |
|---|---|
| `{region}_tracks_round_{N}.csv` | per-round track assignments |
| `{region}_all_tracks_summary.csv` | track id, span, member 2D cells, margins |
| `divisor_cache_{region}.json` | cached Soft-DTW / OT / centroid divisors |
| `{region}_evaluation_{params}.json` | global score + per-component metrics |
| `bayes_opt_results_v9_{region}/` | BO trial history + best-parameter directory |

## Run on your own data

Create the following directories in the repository root (replace `123` with
your tile identifier):

```text
Tile123/123_S01_features.csv
Tile123/123_S02_features.csv
...
123/expression-T123-S01.csv
123/expression-T123-S02.csv
...
```

Feature files are comma-separated, with one cell per row: `cell_id`,
`centroid_x`, `centroid_y`, `area`, `aspect_ratio`, and contour pairs
`x_0`, `y_0`, …, `x_399`, `y_399`, following the demo format. Use centroids in
a common registered pixel coordinate system and contours centred on each
cell's centroid. Include semicolon-separated neighbour IDs in `adjacent_cells`
and their count in `num_adjacent_cells`. Slice numbers are read from filenames.

Expression files are **tab-separated** despite the `.csv` extension and require
`feature_index`, `cell_label`, `z`, and `total_exp`. Feature cell IDs such as
`T123S01_0001` must encode the matching slice (`z=1`) and cell label
(`cell_label=1`) used in the expression files.

In the tutorial's setup cell, replace the demo paths and identifiers with:

```python
TILE_NAME = "Tile123"
TILE_PREFIX = "123"
DEMO_DATA_DIR = WORK_DIR / TILE_NAME
DEMO_EXPR_DIR = WORK_DIR / TILE_PREFIX
```

Then follow the same tutorial workflow. For CLI reconstruction and optimisation:

```bash
python -m hidtw_pipeline.bayes_opt_margin_v10 \
    --region Tile123 --gpu 0 --n-calls 30 --n-initial 8
```

## Method summary

- **Soft-DTW** on 200-point resampled contours scores cell shape similarity.
- **Sinkhorn OT** on the one-hop neighbour cluster captures local context.
- **Centroid distance** anchors the match spatially.
- Each raw distance is divided by a dataset-specific divisor so its 75th
  percentile maps to ~7.5 on a 0–10 scale; the three scores then combine via
  `DTW_WEIGHT / OT_WEIGHT / CENTROID_WEIGHT` (sum to 1).
- A match is accepted only if the relative margin
  `(top1 − top2) / top1` exceeds `RELATIVE_MARGIN_THRESHOLD`.
- `bayes_opt_margin_v10` searches these four parameters with `gp_minimize`,
  scoring each trial by `0.4 · confidence + 0.4 · coverage + 0.2 · length`.

## Data

Raw images and precomputed contour features used in the paper:
[https://doi.org/10.6084/m9.figshare.31305214](https://doi.org/10.6084/m9.figshare.31305214)

The archive contains one main reconstruction dataset and three manually
validated gold-standard sets used for ablation, noise, and parameter-sensitivity
studies.

## Citation

```bibtex
@article{XXX,
  title   = {Three-dimensional Virtual Adult Cardiomyocyte Transcriptomics},
  author  = {Luo, C. and Lyu, Y. and Guo, X. and Cheng, L. and Liang, Q. and
             Wang, S. and Wang, Y. and Zhang, S. and Wang, S. and Liu, T. and
             Luo, Y. and Lu, F. and Ran, B. and Zhang, Y. and Liu, X. and
             Wang, Y. and Qin, G. and Wu, J. and Lyu, Q.R.},
  journal = {},
  year    = {2026},
  doi     = {...}
}
```
