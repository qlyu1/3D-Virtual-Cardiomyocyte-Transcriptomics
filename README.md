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
pip install -r requirements.txt
```

A CUDA-capable GPU is required: Soft-DTW and Sinkhorn OT both run on the GPU
and the matcher aborts on CPU. Tested with PyTorch 2.x on CUDA 11.

## Quick start

The tutorial notebook walks through loading, calibration, reconstruction,
Bayesian optimisation, and per-track UMI aggregation on the bundled Tile046
slices:

```bash
jupyter lab tutorial_HiDTW_demodata_Tile046.ipynb
```

CLI-only run (the matcher discovers data via `glob.glob("Tile*")` in cwd, so
symlink or copy `demo_data/Tile046` to `./Tile046` first):

```bash
ln -s demo_data/Tile046 Tile046
python -m hidtw_pipeline.bayes_opt_margin_v10 \
    --region Tile046 --gpu 0 --n-calls 30 --n-initial 8
```

## Outputs

| File | Contents |
|---|---|
| `{region}_tracks_round_{N}.csv` | per-round track assignments |
| `{region}_all_tracks_summary.csv` | track id, span, member 2D cells, margins |
| `divisor_cache_{region}.json` | cached Soft-DTW / OT / centroid divisors |
| `{region}_evaluation_{params}.json` | global score + per-component metrics |
| `bayes_opt_results_v9_{region}/` | BO trial history + best-parameter directory |

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
