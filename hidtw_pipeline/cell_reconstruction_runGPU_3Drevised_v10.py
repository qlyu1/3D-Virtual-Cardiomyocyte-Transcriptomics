#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell 3D Reconstruction with GPU-Accelerated Matching

PERFORMANCE OPTIMIZATIONS:
1. Pre-computed contour resampling
   
2. GPU Soft-DTW for contour similarity
   - Batch processing with unified 200-point contours
   
3. GPU Sinkhorn algorithm for Optimal Transport
   - Entropic regularization with minimal accuracy loss
   
4. Dual normalization system
   - LOCAL: Adaptive batch scaling for better discrimination
   - GLOBAL: Fixed scaling for consistent quality scoring
   
"""

# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
# This must be done before any CUDA initialization
import sys
import os
import argparse

# Parent of ``hidtw_pipeline/`` is the demo root (``soft_dtw_cuda`` and data links live there).
_demo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _demo_root not in sys.path:
    sys.path.insert(0, _demo_root)

# Early GPU selection: parse --gpu-id before torch import
# Use parse_known_args to avoid conflicts with full argparse later
# Supports both: --gpu-id 1 and --gpu-id=1
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument('--gpu-id', type=int, default=None)
_early_args, _ = _early_parser.parse_known_args()

if _early_args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_early_args.gpu_id)
    print(f"[Early Init] Setting CUDA_VISIBLE_DEVICES={_early_args.gpu_id}")

# Clean up temporary parser
del _early_parser, _early_args

import numpy as np
import pandas as pd
import glob
import time
import heapq
import multiprocessing as mp
from tqdm import tqdm
import matplotlib
# Non-interactive backend only for CLI runs; importing this module from Jupyter must
# not override an interactive backend or figures never appear in notebooks.
if __name__ == "__main__":
    matplotlib.use("Agg")
import torch
from soft_dtw_cuda import SoftDTW
import csv
import json
import warnings
from numba.core.errors import NumbaPerformanceWarning

# Suppress GPU under-utilization warning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

# --- Feature weights (safe defaults; overridden by Bayesian Optimization) ---
DTW_WEIGHT = 0.33        # [SAFE DEFAULT] Weight of contour similarity (Soft-DTW).
                         # Real run: replaced by Bayesian Optimization.
OT_WEIGHT = 0.33         # [SAFE DEFAULT] Weight of neighbor-cluster similarity (Optimal Transport).
                         # Real run: replaced by Bayesian Optimization.
CENTROID_WEIGHT = 0.34   # [SAFE DEFAULT] Weight of centroid distance (spatial proximity).
                         # Real run: replaced by Bayesian Optimization.

_CACHE_DIR = os.environ.get('DIVISOR_CACHE_DIR', os.getcwd())

# --- Global normalization divisors (safe fallbacks; overridden by sampling calibration) ---
# These constants map raw distances to a roughly comparable [0, 10] scale:
#   normalized = min(10.0, (raw_distance / DIVISOR) * 10.0)
#
# [SAFE FALLBACKS]
# Real run: DTW/OT divisors are computed via sampling on the current region/dataset
# (e.g., set so that the 75th percentile maps to ~7.5), then saved to cache.
DTW_NORM_DIVISOR = 1390000.0     # [SAFE FALLBACK] Soft-DTW raw distance divisor.
                                 # Real run: replaced by sampled calibration on this region.

OT_NORM_DIVISOR  = 1507000.0     # [SAFE FALLBACK] OT/Sinkhorn raw distance divisor.
                                 # Real run: replaced by sampled calibration on this region.

CENTROID_NORM_DIVISOR = 70.0     # [SAFE FALLBACK] Centroid distance divisor (pixels/µm).
                                 # Real run: may stay fixed or be calibrated if enabled.

# Margin-based threshold (Relative Margin approach)
# Instead of fixed LOSS_THRESHOLD, we use relative_margin = margin / top1_score
# where margin = top1_score - top2_score
# A match is accepted if relative_margin >= RELATIVE_MARGIN_THRESHOLD
RELATIVE_MARGIN_THRESHOLD = 0.01  # Accept matches where margin is at least 1% of top1_score
                                  # Real run: replaced by Bayesian Optimization.

# Contour resampling for DTW/OT computation
NUM_CONTOUR_POINTS = 200  # Uniformly resample all contours to 200 points for computation

Z_DISTANCE = 10.0       # Distance between layers
QUALITY_TOP_PERCENT = 0.1  # Select top 10% high-quality tracks

# OT computation optimization: Only compute OT for top X% candidates based on DTW similarity
# Combined with DTW pre-filtering for maximum efficiency
OT_DTW_FILTER_PERCENT = 0.5  # Compute OT only for top 50% candidates

# OT penalty value (for candidates that don't pass DTW pre-filter)
OT_PENALTY_VALUE = 20000.0  # Large raw value to indicate "not computed" status

# Sinkhorn OT parameters
SINKHORN_REG = 0.01          # Entropic regularization (smaller = more accurate but slower)
SINKHORN_MAX_ITER = 100      # Maximum Sinkhorn iterations
SINKHORN_TOL = 1e-6          # Convergence tolerance

# Track selection comprehensive quality weights (used in conflict resolution)
TRACK_MARGIN_WEIGHT = 0.7    # Weight for margin quality (70%)
TRACK_LENGTH_WEIGHT = 0.3    # Weight for track length (30%)
# Comprehensive quality = 70% × normalized_margin + 30% × normalized_length
# This favors high-confidence matches while also preferring longer tracks

# Global score weights (used in final evaluation)
GLOBAL_MARGIN_WEIGHT = 0.4    # Weight for margin quality (40%)
GLOBAL_COVERAGE_WEIGHT = 0.4  # Weight for coverage ratio (40%)
GLOBAL_LENGTH_WEIGHT = 0.2    # Weight for average track length (20%)
# Global score = 40% × margin + 40% × coverage + 20% × length

# Set the number of processes for parallel computing, None means use all available CPU cores
NUM_PROCESSES = 16  

# Global variables for margin statistics (for quality scoring only)
all_margins = []  # Collect all margins (top1 - top2) for analysis (not for threshold)

# Global decision-level margin collection 
# Record margins at all matching decision points, regardless of acceptance
all_decision_margins = []  # Global decision-level margin distribution
all_decision_info = []  # (slice, has_candidate, relative_margin, accepted)

class Cell:
    def __init__(self, cell_id, slice_index, centroid_x, centroid_y, area, aspect_ratio, contour_points, 
                 adjacent_cells=None, num_adjacent_cells=0):
        self.cell_id = cell_id
        self.slice_index = slice_index
        self.centroid = np.array([centroid_x, centroid_y])
        self.area = area
        self.aspect_ratio = aspect_ratio
        self.contour_points = contour_points  # Original contour (e.g., 400 points)
        self.used = False  # Mark whether this cell has been used for track construction
        # Pre-computed resampled contour for DTW/OT computation (PERFORMANCE OPTIMIZATION)
        # This avoids repeated resampling during matching (same cell may be candidate hundreds of times)
        self.contour_points_200 = None  # Will be set after initialization
        # Adjacent cells information for OT cluster analysis
        self.adjacent_cells = adjacent_cells if adjacent_cells else []
        self.num_adjacent_cells = num_adjacent_cells

    def __repr__(self):
        return f"Cell(id={self.cell_id}, slice={self.slice_index}, centroid={self.centroid}, area={self.area:.1f}, neighbors={self.num_adjacent_cells})"

class CellTrack:
    def __init__(self, base_cell):
        self.cells = {base_cell.slice_index: base_cell}
        self.start_slice = base_cell.slice_index
        self.end_slice = base_cell.slice_index
        self.centroids = {base_cell.slice_index: base_cell.centroid}
        self.areas = {base_cell.slice_index: base_cell.area}
        self.contours = {base_cell.slice_index: base_cell.contour_points}
        self.margins = []  # Store margin (top1 - top2) for each connection in this track
    
    def add_cell(self, cell):
        """Add a cell to the track, ensuring only one cell per layer"""
        if not isinstance(cell.slice_index, int):
            raise TypeError(f"slice_index is not an integer: {cell.slice_index} (type: {type(cell.slice_index)})")
    
        if cell.slice_index in self.cells:
            raise ValueError(
                f"Attempting to add multiple cells from slice {cell.slice_index} to track!\n"
                f"Existing cell: {self.cells[cell.slice_index].cell_id}, new cell: {cell.cell_id}"
            )
    
        self.cells[cell.slice_index] = cell
        self.centroids[cell.slice_index] = cell.centroid
        self.areas[cell.slice_index] = cell.area
        self.contours[cell.slice_index] = cell.contour_points
        self.start_slice = min(self.start_slice, cell.slice_index)
        self.end_slice = max(self.end_slice, cell.slice_index)
    
    def get_quality_score(self, cells_by_slice=None):
        """
        Calculate track quality score based on margin (confidence) of connections
        
        Quality = Average Margin = Average(top1_score - top2_score)
        
        Higher margin = lower perplexity = higher quality
        - Large margin: top1 candidate is much better than top2, confident match
        - Small margin: top1 and top2 are similar, uncertain match (high perplexity)
        
        Returns:
            Average margin (higher is better), range typically [0, 1]
        """
        if len(self.margins) == 0:
            return 0.0
        
        return np.mean(self.margins)
    
    def get_quality_details(self, cells_by_slice=None):
        """
        Get detailed quality metrics based on margin (perplexity)
        
        Returns:
            dict with keys:
                - 'avg_margin': average margin (top1 - top2) across all connections
                - 'min_margin': minimum margin (weakest connection)
                - 'max_margin': maximum margin (strongest connection)
                - 'std_margin': standard deviation of margins
                - 'num_connections': number of connections in track
        """
        if len(self.margins) == 0:
            return {
                'avg_margin': 0.0,
                'min_margin': 0.0,
                'max_margin': 0.0,
                'std_margin': 0.0,
                'num_connections': 0
            }
        
        return {
            'avg_margin': float(np.mean(self.margins)),
            'min_margin': float(np.min(self.margins)),
            'max_margin': float(np.max(self.margins)),
            'std_margin': float(np.std(self.margins)) if len(self.margins) > 1 else 0.0,
            'num_connections': len(self.margins)
        }
      
    def get_layer_count(self):
        return len(self.cells)
      
    def __repr__(self):
        return f"CellTrack(slices={self.start_slice}-{self.end_slice}, cells={len(self.cells)})"


def load_celltracks_from_matcher_exports(csv_paths, cells_by_slice):
    """Rebuild ``CellTrack`` objects from CSV rows written by ``find_high_quality_tracks_layerwise``.

    ``csv_paths`` should list exports such as ``Tile046_manual_tracks_round_1.csv`` (all rounds).
    Cell geometries come from the already-loaded ``cells_by_slice`` lookup.

    Note:
        Per-connection margins are **not** encoded in the CSV; rebuilt tracks have empty
        ``margins``. Prefer loading ``manual_evaluation.json`` / ``best_evaluation.json``
        when you need previously computed ``evaluate_all_tracks`` summaries instead of
        re-running that function on restored tracks.
    """
    id_to_cell = {}
    for cells in cells_by_slice.values():
        for cell in cells:
            id_to_cell[str(cell.cell_id)] = cell

    tracks = []
    for path in csv_paths:
        p = path if isinstance(path, str) else str(path)
        if not os.path.isfile(p):
            continue
        with open(p, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw = row.get("cell_ids") or ""
                ids = [x.strip() for x in raw.split("|") if x.strip()]
                cells_seq = []
                for cid in ids:
                    c = id_to_cell.get(cid)
                    if c is None:
                        continue
                    cells_seq.append(c)
                if len(cells_seq) < 2:
                    continue
                cells_seq.sort(key=lambda c: int(c.slice_index))
                tr = CellTrack(cells_seq[0])
                for c in cells_seq[1:]:
                    tr.add_cell(c)
                tracks.append(tr)
    return tracks


def load_cells_from_csv_files(cell_dir=None):
    """Load cell data from CSV files in a specific cell directory
    
    Args:
        cell_dir: Specific cell directory to load (e.g., 'cell-001'). 
                  If None, will load from all directories.
    """
    import re
    
    cells_by_slice = {}
    
    if cell_dir is None:
        # Load from all directories 
        cell_dirs = sorted(glob.glob("cell-*"))
        if not cell_dirs:
            cell_dirs = sorted(glob.glob("Tile*"))
        if not cell_dirs:
            print("Error: No cell-* or Tile* directories found!")
            return cells_by_slice
        print(f"Found {len(cell_dirs)} directories: {cell_dirs}")
        all_csv_files = []
        for cdir in cell_dirs:
            csv_pattern = os.path.join(cdir, "lighter_*.csv")
            csv_files = sorted(glob.glob(csv_pattern))
            # If no lighter_*.csv found, try *_S*_features.csv
            if not csv_files:
                csv_pattern = os.path.join(cdir, "*_S*_features.csv")
                csv_files = sorted(glob.glob(csv_pattern))
            all_csv_files.extend(csv_files)
            print(f"  {cdir}: found {len(csv_files)} CSV files")
    else:
        # Load from specific directory
        if not os.path.exists(cell_dir):
            print(f"Error: Directory {cell_dir} not found!")
            return cells_by_slice
        
        csv_pattern = os.path.join(cell_dir, "lighter_*.csv")
        all_csv_files = sorted(glob.glob(csv_pattern))
        # If no lighter_*.csv found, try *_S*_features.csv
        if not all_csv_files:
            csv_pattern = os.path.join(cell_dir, "*_S*_features.csv")
            all_csv_files = sorted(glob.glob(csv_pattern))
        print(f"Loading from {cell_dir}: found {len(all_csv_files)} CSV files")
    
    print(f"\nStarting to load {len(all_csv_files)} CSV files")
    
    # Regular expression for extracting slice number from filename
    # Example: lighter_01_cell-001.csv -> slice_index = 1
    # Or: 014_S01_features.csv -> slice_index = 1
    slice_pattern = re.compile(r'(?:lighter_(\d+)_cell-\d+\.csv|_S(\d+)_features\.csv)')
    
    for file in tqdm(all_csv_files, desc="Loading CSV files"):
        # Extract slice index from filename
        try:
            match = slice_pattern.search(os.path.basename(file))
            if match:
                # Try to get the first or second capture group (depending on file format)
                slice_index = int(match.group(1) or match.group(2))
            else:
                print(f"Unable to extract slice number from filename: {file}")
                continue
        except Exception as e:
            print(f"Error parsing filename {file}: {e}")
            continue

        # Load CSV data
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue
        
        # Collect all cells from this CSV
        # Determine which column name is used for cell ID (check once per file)
        if 'ID' in df.columns:
            cell_id_col = 'ID'
        elif 'cell_id' in df.columns:
            cell_id_col = 'cell_id'
        else:
            print(f"Warning: No 'ID' or 'cell_id' column found in {file}")
            continue
        
        for _, row in df.iterrows():
            try:
                # Extract cell ID using pre-determined column name
                # IMPORTANT: Convert to string for type consistency with adjacent_cells
                cell_id = str(row[cell_id_col])
                
                # Extract centroid coordinates, area and aspect ratio
                centroid_x = row['centroid_x']
                centroid_y = row['centroid_y']
                area = row['area']
                aspect_ratio = row['aspect_ratio']

                # Extract adjacent cells information
                adjacent_cells_str = row.get('adjacent_cells', '')
                num_adjacent_cells = row.get('num_adjacent_cells', 0)
                
                # Parse adjacent cells list (ensure type consistency with cell_id)
                adjacent_cells = []
                if pd.notna(adjacent_cells_str) and adjacent_cells_str != '':
                    # Split and strip whitespace, ensure all are strings
                    adjacent_cells = [s.strip() for s in str(adjacent_cells_str).split(';') if s.strip()]

                # Extract contour points (x_0, y_0, x_1, y_1, ...)
                contour_points = []
                for i in range(400):
                    x_col = f'x_{i}'
                    y_col = f'y_{i}'
                    if x_col in row and y_col in row:
                        x = float(row[x_col])
                        y = float(row[y_col])
                        contour_points.append([x, y])
                    else:
                        break

                if len(contour_points) < 5:
                    print(f"Warning: Cell {cell_id} has only {len(contour_points)} contour points, skipping")
                    continue

                contour_points = np.array(contour_points)

                # Create Cell object with adjacent cells information
                cell = Cell(cell_id, slice_index, centroid_x, centroid_y, area, aspect_ratio, contour_points,
                           adjacent_cells, num_adjacent_cells)
                
                # PRE-COMPUTE resampled contour (PERFORMANCE OPTIMIZATION)
                # Resample once at load time to avoid repeated resampling during matching
                cell.contour_points_200 = resample_contour(contour_points, NUM_CONTOUR_POINTS)
                
                # Add cell to corresponding layer's list
                if slice_index in cells_by_slice:
                    cells_by_slice[slice_index].append(cell)
                else:
                    cells_by_slice[slice_index] = [cell]
                    
            except Exception as e:
                print(f"Error processing cell {row.get('ID', 'Unknown')}: {e}")
                continue

    # Print cell count for each layer
    print("\nCell distribution by slice:")
    for slice_idx in sorted(cells_by_slice.keys()):
        print(f"  Slice {slice_idx}: {len(cells_by_slice[slice_idx])} cells")
    
    total_cells = sum(len(cells) for cells in cells_by_slice.values())
    print(f"\nTotal cells loaded: {total_cells}")
        
    return cells_by_slice


# ============================================================================
# OT (Optimal Transport) Cluster Similarity Functions
# ============================================================================

def dtw_distance_cpu(contour1, contour2):
    """
    Compute HARD DTW distance using CPU (FALLBACK ONLY - not normally used)
    
    WARNING: This is a pure CPU implementation of classic (hard) DTW,
    NOT soft-DTW. The distance scale is different from GPU soft-DTW.
    Only used as emergency fallback if GPU computation fails.
    
    Args:
        contour1: Point coordinates of the first contour (n, 2)
        contour2: Point coordinates of the second contour (m, 2)
    
    Returns:
        Hard DTW distance (different scale than soft-DTW!)
    """
    n, m = len(contour1), len(contour2)
    
    # Initialize DTW matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    # Fill DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Compute Euclidean distance between point pairs as cost
            cost = np.linalg.norm(contour1[i-1] - contour2[j-1])
            
            # DTW recurrence formula
            dtw[i, j] = cost + min(
                dtw[i-1, j],      # insertion
                dtw[i, j-1],      # deletion
                dtw[i-1, j-1]     # match
            )
    
    return dtw[n, m]


def dtw_distance_gpu_single(contour1, contour2, gamma=0.1):
    """
    Compute single pair contour distance using GPU soft-DTW
    
    Args:
        contour1: Point coordinates of the first contour (n, 2) - numpy array
        contour2: Point coordinates of the second contour (m, 2) - numpy array
        gamma: soft-DTW smoothing parameter
    
    Returns:
        DTW distance (scalar)
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert to tensor and add batch dimension
        x = torch.tensor(contour1, dtype=torch.float32).unsqueeze(0).to(device)  # [1, n, 2]
        y = torch.tensor(contour2, dtype=torch.float32).unsqueeze(0).to(device)  # [1, m, 2]
        
        # Create soft-DTW object
        sdtw = SoftDTW(use_cuda=torch.cuda.is_available(), gamma=gamma)
        
        # Compute distance
        dist = sdtw(x, y).item()
        
        return dist
    except Exception as e:
        # Fallback to CPU version if GPU computation fails
        return dtw_distance_cpu(contour1, contour2)


def get_cluster_cells(cell, cells_by_slice):
    """
    Get specified cell and all its adjacent cells (cluster)
    
    Args:
        cell: Cell object
        cells_by_slice: Dictionary of cells indexed by slice
    
    Returns:
        List containing Cell objects (including target cell itself)
    """
    cluster = [cell]
    
    # Get all cells in current layer
    if cell.slice_index not in cells_by_slice:
        return cluster
    
    slice_cells = cells_by_slice[cell.slice_index]
    
    # Create mapping from cell_id to Cell object
    cell_map = {c.cell_id: c for c in slice_cells}
    
    # Add adjacent cells
    for adj_cell_id in cell.adjacent_cells:
        if adj_cell_id in cell_map:
            cluster.append(cell_map[adj_cell_id])
    
    return cluster


def compute_ot_cost_matrix(cluster_a, cluster_b, use_gpu_batch=True, gamma=0.1):
    """
    Compute cost matrix between two clusters (accelerated using GPU batch soft-DTW)
    
    Args:
        cluster_a: List of Cell objects in cluster A
        cluster_b: List of Cell objects in cluster B
        use_gpu_batch: Whether to use GPU batch computation (default True, significant speedup)
        gamma: soft-DTW smoothing parameter
    
    Returns:
        Cost matrix C (shape: len(cluster_a) x len(cluster_b))
    """
    m = len(cluster_a)
    n = len(cluster_b)
    C = np.zeros((m, n))
    
    if use_gpu_batch and torch.cuda.is_available() and m * n > 1:
        try:
            # GPU batch computation: significantly accelerates OT cost matrix calculation
            device = torch.device("cuda")
            sdtw = SoftDTW(use_cuda=True, gamma=gamma)
            
            # Batch compute all pairings
            for i, cell_a in enumerate(cluster_a):
                # Use pre-computed resampled contours (all are NUM_CONTOUR_POINTS = 200)
                contours_a = [cell_a.contour_points_200] * n
                contours_b = [cell_b.contour_points_200 for cell_b in cluster_b]
                
                # All contours are already 200 points, no padding needed!
                # Convert directly to tensors
                x_list = []
                y_list = []
                for ca, cb in zip(contours_a, contours_b):
                    x_list.append(torch.tensor(ca, dtype=torch.float32))
                    y_list.append(torch.tensor(cb, dtype=torch.float32))
                
                # Stack into batch and transfer to GPU
                x_batch = torch.stack(x_list).to(device)  # [n, max_len, 2]
                y_batch = torch.stack(y_list).to(device)  # [n, max_len, 2]
                
                # Batch compute DTW distances
                dists = sdtw(x_batch, y_batch)  # [n]
                
                # Save results
                C[i, :] = dists.cpu().numpy()
            
            return C
            
        except Exception as e:
            # GPU batch computation failed, fallback to single GPU computation
            print(f"Warning: GPU batch OT failed, falling back to single GPU: {e}")
    
    # Fallback: use single GPU or CPU computation (using pre-computed resampled contours)
    for i, cell_a in enumerate(cluster_a):
        for j, cell_b in enumerate(cluster_b):
            # Use pre-computed resampled contours
            contour_a = cell_a.contour_points_200
            contour_b = cell_b.contour_points_200
            C[i, j] = dtw_distance_gpu_single(contour_a, contour_b, gamma=gamma)
    
    return C


def sinkhorn_ot_gpu_batch(C_batch, weights_a_batch, weights_b_batch):
    """
    GPU-accelerated BATCH Sinkhorn algorithm for multiple OT problems
    
    Solves multiple OT problems in parallel on GPU 
    
    Uses global parameters: SINKHORN_REG, SINKHORN_MAX_ITER, SINKHORN_TOL
    
    Args:
        C_batch: List of cost matrices [(m1, n1), (m2, n2), ...]
        weights_a_batch: List of source weights [(m1,), (m2,), ...]
        weights_b_batch: List of target weights [(n1,), (n2,), ...]
    
    Returns:
        ot_distances: List of OT distances
        T_batch: List of transport matrices
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ot_distances = []
    T_batch = []
    
    # Process each OT problem (can't truly batch due to different dimensions)
    # But GPU computation is still much faster than CPU
    for C, weights_a, weights_b in zip(C_batch, weights_a_batch, weights_b_batch):
        try:
            ot_dist, T = sinkhorn_ot_gpu(C, weights_a, weights_b, 
                                        reg=SINKHORN_REG, 
                                        max_iter=SINKHORN_MAX_ITER, 
                                        tol=SINKHORN_TOL, 
                                        use_gpu=True)
            ot_distances.append(ot_dist)
            T_batch.append(T)
        except:
            ot_distances.append(float('inf'))
            T_batch.append(None)
    
    return ot_distances, T_batch


def sinkhorn_ot_gpu(C, weights_a, weights_b, reg=0.05, max_iter=100, tol=1e-6, use_gpu=True):
    """
    GPU-accelerated Sinkhorn algorithm for entropic regularized optimal transport
    
    This is much faster than exact EMD, especially on GPU, with negligible accuracy loss.
    
    Algorithm:
        min_{T} <T, C> + reg * KL(T || weights_a ⊗ weights_b)
        s.t. T @ 1 = weights_a, T^T @ 1 = weights_b, T >= 0
    
    Args:
        C: Cost matrix (m x n), numpy array or torch tensor
        weights_a: Source weights (m,), sum to 1
        weights_b: Target weights (n,), sum to 1
        reg: Entropic regularization parameter (default 0.05)
             - Smaller = closer to exact OT but slower convergence
             - Larger = faster but more regularized
        max_iter: Maximum iterations (default 100)
        tol: Convergence tolerance (default 1e-6)
        use_gpu: Whether to use GPU (default True)
    
    Returns:
        ot_distance: Optimal transport distance (scalar)
        T: Transport matrix (m x n), numpy array
    """
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    
    # Convert to torch tensors
    if isinstance(C, np.ndarray):
        C_torch = torch.tensor(C, dtype=torch.float32, device=device)
    else:
        C_torch = C.to(device)
    
    a = torch.tensor(weights_a, dtype=torch.float32, device=device)
    b = torch.tensor(weights_b, dtype=torch.float32, device=device)
    
    m, n = C_torch.shape
    
    # Adaptive reg: dynamically adjust based on Cost Matrix size to prevent numerical underflow
    # reg should be 1-5% of the typical value of C, ensure exp(-C/reg) does not underflow to 0
    adaptive_reg = float(torch.median(C_torch).item()) * 0.05
    adaptive_reg = max(reg, adaptive_reg)  # Use at least the input reg value
    adaptive_reg = min(adaptive_reg, 10000.0)  # Upper limit to prevent over-regularization
    
    # Compute kernel matrix K = exp(-C / reg)
    K = torch.exp(-C_torch / adaptive_reg)
    
    # Initialize dual variables
    u = torch.ones(m, dtype=torch.float32, device=device) / m
    v = torch.ones(n, dtype=torch.float32, device=device) / n
    
    # Sinkhorn iterations
    for iteration in range(max_iter):
        u_prev = u.clone()
        
        # Update u and v
        u = a / (K @ v + 1e-16)  # Add small epsilon to avoid division by zero
        v = b / (K.T @ u + 1e-16)
        
        # Check convergence
        err = torch.max(torch.abs(u - u_prev))
        if err < tol:
            break
    
    # Compute transport matrix T = diag(u) @ K @ diag(v)
    T = u.unsqueeze(1) * K * v.unsqueeze(0)
    
    # Compute OT distance
    ot_distance = torch.sum(T * C_torch).item()
    
    # Convert back to numpy
    T_numpy = T.cpu().numpy()
    
    return ot_distance, T_numpy


def solve_optimal_transport(C, weights_a, weights_b, use_gpu=True):
    """
    Solve optimal transport problem (GPU-ACCELERATED with Sinkhorn algorithm)
    
    CHANGED: Now uses GPU-accelerated Sinkhorn algorithm by default
    - Minimal accuracy loss (entropic regularization ≈ 0.05)
    - Falls back to CPU POT/scipy if GPU fails

    """
    # Try GPU Sinkhorn first (MUCH FASTER!)
    if use_gpu and torch.cuda.is_available():
        try:
            return sinkhorn_ot_gpu(C, weights_a, weights_b, 
                                  reg=SINKHORN_REG, 
                                  max_iter=SINKHORN_MAX_ITER, 
                                  tol=SINKHORN_TOL)
        except Exception as e:
            # GPU failed, fall back to CPU
            pass
    
    # Fallback to CPU methods
    try:
        import ot as pot
        # Use POT library's exact OT solver (CPU)
        T = pot.emd(weights_a, weights_b, C)
        ot_distance = np.sum(T * C)
        return ot_distance, T
    except ImportError:
        # If POT library is not installed, use scipy's linear programming solver
        from scipy.optimize import linprog
        
        m, n = C.shape
        
        # Objective function: minimize sum(T_ij * C_ij)
        c = C.flatten()
        
        # Constraints:
        # 1. Row sum = weights_a (mass conservation for each source cell)
        # 2. Column sum = weights_b (mass conservation for each target cell)
        
        # Equality constraint matrix
        A_eq = []
        b_eq = []
        
        # Row sum constraint
        for i in range(m):
            row = np.zeros(m * n)
            row[i * n:(i + 1) * n] = 1
            A_eq.append(row)
            b_eq.append(weights_a[i])
        
        # Column sum constraint
        for j in range(n):
            col = np.zeros(m * n)
            col[j::n] = 1
            A_eq.append(col)
            b_eq.append(weights_b[j])
        
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        
        # Boundary constraint: T_ij >= 0
        bounds = [(0, None) for _ in range(m * n)]
        
        # Solve linear programming
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            T = result.x.reshape(m, n)
            ot_distance = result.fun
            return ot_distance, T
        else:
            # Solve failed, return infinity
            return float('inf'), None


def _normalize_area_weights(areas: np.ndarray) -> np.ndarray:
    """Convert cluster areas to OT probability weights; uniform fallback if invalid sum."""
    a = np.asarray(areas, dtype=np.float64).ravel()
    n = a.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    s = float(np.nansum(a))
    if not np.isfinite(s) or s <= 0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = a / s
    if not np.all(np.isfinite(w)):
        return np.full(n, 1.0 / n, dtype=np.float64)
    return w


def compute_cluster_ot_distance(cell1, cell2, cells_by_slice, use_area_weights=True):
    """
    Compute RAW OT distance between two cell's neighbor clusters
    
    IMPORTANT: Returns RAW OT distance (not normalized)
    Caller is responsible for normalization to ensure consistent scaling
    
    Args:
        cell1: First cell
        cell2: Second cell
        cells_by_slice: Cell dictionary indexed by slice
        use_area_weights: Whether to use area as weights
    
    Returns:
        RAW OT distance (caller should normalize)
    """
    try:
        # Get two cell's clusters
        cluster1 = get_cluster_cells(cell1, cells_by_slice)
        cluster2 = get_cluster_cells(cell2, cells_by_slice)
        
        # If any cluster is empty, return infinity
        if len(cluster1) == 0 or len(cluster2) == 0:
            return float('inf')
        
        # If both cells have no neighbors, return 0 (perfect match)
        if len(cluster1) == 1 and len(cluster2) == 1:
            return 0.0
        
        # Compute cost matrix
        C = compute_ot_cost_matrix(cluster1, cluster2)
        
        # Compute weights
        if use_area_weights:
            areas1 = np.array([c.area for c in cluster1])
            areas2 = np.array([c.area for c in cluster2])
            weights1 = _normalize_area_weights(areas1)
            weights2 = _normalize_area_weights(areas2)
        else:
            weights1 = np.ones(len(cluster1)) / len(cluster1)
            weights2 = np.ones(len(cluster2)) / len(cluster2)
        
        # Solve optimal transport (GPU-accelerated Sinkhorn)
        ot_distance, T = solve_optimal_transport(C, weights1, weights2, use_gpu=True)
        
        # Return RAW OT distance (not normalized)
        return ot_distance
        
    except Exception as e:
        # Any error returns infinity
        return float('inf')


# Maximum allowed XY distance, beyond which no matching is performed (units: pixels or μm, depending on your data definition)
MAX_XY_DISTANCE = 100.0

def is_within_spatial_window(cell1, cell2):
    """Determine if the centroids of two cells are close enough"""
    dx = cell1.centroid[0] - cell2.centroid[0]
    dy = cell1.centroid[1] - cell2.centroid[1]
    distance = np.sqrt(dx**2 + dy**2)
    return distance < MAX_XY_DISTANCE


def resample_contour(contour, num_points):
    """
    Uniformly resample contour to specified number of points
    
    This ensures all contours have the same number of points for DTW/OT computation,
    reducing computational cost and standardizing the comparison.
    
    Args:
        contour: Original contour points (N x 2 array)
        num_points: Target number of points (e.g., 200)
    
    Returns:
        Resampled contour (num_points x 2 array)
    """
    # Handle edge case
    if len(contour) == num_points:
        return contour
    
    # Ensure contour is closed
    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0:1]])
    
    # Calculate cumulative distance of contour
    distances = np.zeros(len(contour))
    for i in range(1, len(contour)):
        distances[i] = distances[i-1] + np.linalg.norm(contour[i] - contour[i-1])
    
    total_length = distances[-1]
    
    if total_length == 0:
        # Degenerate case: all points are the same
        return np.tile(contour[0], (num_points, 1))
    
    # Create equidistant points
    resampled = np.zeros((num_points, 2))
    for i in range(num_points):
        # Equidistant sampling
        target_distance = i * total_length / num_points
        
        # Find corresponding position
        idx = np.searchsorted(distances, target_distance)
        if idx >= len(contour):
            idx = len(contour) - 1
        
        if idx > 0:
            # Linear interpolation to find exact position
            prev_idx = idx - 1
            segment_length = distances[idx] - distances[prev_idx]
            if segment_length > 0:
                t = (target_distance - distances[prev_idx]) / segment_length
                resampled[i] = contour[prev_idx] * (1 - t) + contour[idx] * t
            else:
                resampled[i] = contour[idx]
        else:
            resampled[i] = contour[idx]
    
    return resampled


def calculate_loss(cell1, cell2, use_gpu=True, stats=None, cells_by_slice=None, return_details=False):
    """
    Calculate matching loss between two cells (SIMPLIFIED to 3 features only)
    
    Features:
        1. DTW distance (contour similarity)
        2. OT distance (neighbor cluster similarity)
        3. Centroid distance (spatial proximity)
    
    Args:
        return_details: If True, return (loss, dtw_raw, ot_raw, centroid_raw)
                       If False, return loss only (backward compatible)
    
    Returns:
        If return_details=False: Loss score (lower is better), range [0, 10]
        If return_details=True: (loss, dtw_raw, ot_raw, centroid_raw)
    """
    start_time = time.time()
    
    try:
        # Use pre-computed resampled contours (PERFORMANCE OPTIMIZATION)
        # Contours were already resampled to NUM_CONTOUR_POINTS at load time
        contour1 = cell1.contour_points_200
        contour2 = cell2.contour_points_200
        
        # Check if contour point count exceeds CUDA limits (should be OK with 200 points)
        if len(contour1) > 1024 or len(contour2) > 1024:
            if return_details:
                return float('inf'), float('inf'), float('inf'), float('inf')
            return float('inf')
            
        if use_gpu and torch.cuda.is_available():
            try:
                # 1. DTW loss (using pre-computed resampled contours)
                # Use single GPU call instead of fake batch
                dtw_raw = dtw_distance_gpu_single(contour1, contour2, gamma=0.1)
                dtw_loss = min(10.0, dtw_raw / DTW_NORM_DIVISOR * 10.0)
                
                if stats is not None:
                    stats['gpu_time'] = stats.get('gpu_time', 0) + (time.time() - start_time)
                    stats['gpu_calls'] = stats.get('gpu_calls', 0) + 1
            except Exception as e:
                if return_details:
                    return float('inf'), float('inf'), float('inf'), float('inf')
                return float('inf')
        else:
            # GPU not available, return infinite loss value
            if return_details:
                return float('inf'), float('inf'), float('inf'), float('inf')
            return float('inf')
        
        # 2. OT loss (neighbor cluster similarity)
        ot_raw = 0.0
        ot_loss = 0.0
        if cells_by_slice is not None:
            ot_raw = compute_cluster_ot_distance(cell1, cell2, cells_by_slice, use_area_weights=True)
            # Hard fail on OT error (quality-first approach)
            if ot_raw == float('inf'):
                if return_details:
                    return float('inf'), dtw_raw, float('inf'), float('inf')
                return float('inf')
            # Normalize RAW OT to [0, 10] range (consistent with batch path)
            ot_loss = min(10.0, ot_raw / OT_NORM_DIVISOR * 10.0)
        
        # 3. Centroid loss (spatial proximity)
        cx1, cy1 = cell1.centroid
        cx2, cy2 = cell2.centroid
        centroid_raw = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        centroid_loss = min(10.0, centroid_raw / CENTROID_NORM_DIVISOR * 10.0)
        
        # Total loss (weighted sum of 3 features)
        total_loss = (DTW_WEIGHT * dtw_loss + 
                     OT_WEIGHT * ot_loss + 
                     CENTROID_WEIGHT * centroid_loss)
        
        if return_details:
            return total_loss, dtw_raw, ot_raw, centroid_raw
        return total_loss
    
    except Exception as e:
        # Return infinity on error, indicating match failure
        if return_details:
            return float('inf'), float('inf'), float('inf'), float('inf')
        return float('inf')

    
def batch_calculate_dtw(cell, candidates, used_cells, batch_size=264, stats=None, direction=None, cells_by_slice=None):
    """Batch calculate DTW distances, OT cluster similarities, and centroid distances (DUAL NORMALIZATION)
    
    DUAL NORMALIZATION SYSTEM:
    - LOCAL normalization: Adaptive scaling based on current batch statistics
      → Used for candidate selection (find top1/top2)
    - GLOBAL normalization: Fixed scaling using global constants
      → Used for quality scoring and track evaluation
    
    OPTIMIZATION: Uses DTW-based pre-filtering before expensive OT computation.
    - Step 1: Compute DTW for ALL candidates (GPU-accelerated, fast)
    - Step 2: Sort by DTW and only compute OT for top X% most similar candidates
    - Step 3: Assign penalty OT loss to remaining candidates
    
    Features computed: DTW + OT + Centroid (3 features only)
    
    This reduces OT computation by (1-X)%, significantly speeding up the matching process.
    X is controlled by OT_DTW_FILTER_PERCENT parameter.
    
    Args:
        cell: Current cell object
        candidates: List of candidate cells
        used_cells: Set of used cell IDs
        batch_size: Single GPU batch processing size (increased to 264 to reduce OT computation frequency)
        stats: Optional dictionary for collecting performance statistics
        direction: Track construction direction
        cells_by_slice: Dictionary of cells by slice for OT cluster analysis
        
    Returns:
        List containing (candidate, loss_local, loss_global, dtw_raw, ot_raw, centroid_raw) tuples
    """
    start_time = time.time()
    
    # Check GPU availability
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    try:
        # Create soft-DTW object
        sdtw = SoftDTW(use_cuda=use_cuda, gamma=0.1)
        
        valid_pairs = []
        valid_indices = []
        
        # Filter valid cell pairs
        for i, candidate in enumerate(candidates):
            # Check direction constraints
            if direction == "forward" and candidate.slice_index != cell.slice_index + 1:
                continue
            if direction == "backward" and candidate.slice_index != cell.slice_index - 1:
                continue
                
            if candidate.cell_id not in used_cells and is_within_spatial_window(cell, candidate):
                # No need to check CUDA limits - we use pre-computed 200-point contours (contour_points_200)
                # which are always within CUDA's limit (1024 points)
                valid_pairs.append((cell, candidate))
                valid_indices.append(i)
                
        # If no valid pairs, return empty result
        if not valid_pairs:
            if stats:
                stats['empty_batches'] = stats.get('empty_batches', 0) + 1
            return []
        
        results = []
        successful_pairs = 0
        
        # Batch processing
        for i in range(0, len(valid_pairs), batch_size):
            batch_start_time = time.time()
            
            try:
                batch_pairs = valid_pairs[i:i+batch_size]
                batch_indices = valid_indices[i:i+batch_size]
                
                # Prepare batch data with unified length (TRUE BATCH processing)
                # Use pre-computed resampled contours (all are NUM_CONTOUR_POINTS = 200)
                x_batch_list = []
                y_batch_list = []
                
                # All contours are already resampled to NUM_CONTOUR_POINTS (200), no padding needed!
                for current_cell, candidate in batch_pairs:
                    # Use pre-computed resampled contours (PERFORMANCE OPTIMIZATION)
                    x_contour = current_cell.contour_points_200
                    y_contour = candidate.contour_points_200
                    
                    # Convert to tensors (no padding needed since all are 200 points)
                    x_batch_list.append(torch.tensor(x_contour, dtype=torch.float32))
                    y_batch_list.append(torch.tensor(y_contour, dtype=torch.float32))
                
                # TRUE BATCH: Stack into [B, 200, 2] and transfer to GPU ONCE
                x_batch_stacked = torch.stack(x_batch_list).to(device)  # [B, 200, 2]
                y_batch_stacked = torch.stack(y_batch_list).to(device)  # [B, 200, 2]
                
                # Compute ALL DTW distances in ONE GPU call (TRUE parallel batch processing)
                dtw_dists = sdtw(x_batch_stacked, y_batch_stacked)  # [B]
                dtw_dists_cpu = dtw_dists.cpu().numpy()  # Transfer back to CPU once
                
                # OPTIMIZATION: Pre-filter candidates based on DTW before computing expensive OT
                # Sort by DTW and only compute OT for top X% most similar (lowest DTW) candidates
                dtw_with_indices = [(j, float(dtw_dists_cpu[j])) for j in range(len(batch_pairs))]
                dtw_with_indices.sort(key=lambda x: x[1])  # Sort by DTW value (ascending = most similar first)
                
                # Calculate threshold: only compute OT for top X% based on DTW
                ot_compute_threshold = int(len(dtw_with_indices) * OT_DTW_FILTER_PERCENT)
                ot_compute_threshold = max(1, ot_compute_threshold)  # At least compute 1
                
                # Create set of indices that will get OT computation
                compute_ot_indices = set(idx for idx, _ in dtw_with_indices[:ot_compute_threshold])
                
                # ========== STEP 1: Collect DTW and prepare for batch OT ==========
                # First pass: collect DTW and centroid for all pairs
                pair_info = []  # (j, candidate, dtw_raw, centroid_raw, cluster1, cluster2)
                ot_batch_data = []  # For batch GPU OT computation
                ot_batch_indices = []  # Track which pair needs OT
                
                for j in range(len(batch_pairs)):
                    try:
                        current_cell, candidate = batch_pairs[j]
                        
                        # Get pre-computed DTW raw distance
                        dtw_raw = float(dtw_dists_cpu[j])
                        
                        # Calculate centroid raw distance
                        cx1, cy1 = current_cell.centroid
                        cx2, cy2 = candidate.centroid
                        centroid_raw = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                        
                        # Prepare for OT computation
                        ot_raw = 0.0
                        cluster1 = None
                        cluster2 = None
                        
                        if cells_by_slice is not None and j in compute_ot_indices:
                            # Get clusters for OT computation
                            cluster1 = get_cluster_cells(current_cell, cells_by_slice)
                            cluster2 = get_cluster_cells(candidate, cells_by_slice)
                            
                            if len(cluster1) > 0 and len(cluster2) > 0:
                                if len(cluster1) == 1 and len(cluster2) == 1:
                                    # Trivial case: no neighbors
                                    ot_raw = 0.0
                                else:
                                    # Need to compute OT - add to batch
                                    ot_batch_indices.append(len(pair_info))
                                    ot_batch_data.append((cluster1, cluster2))
                            else:
                                continue
                        elif cells_by_slice is not None and j not in compute_ot_indices:
                            # Didn't pass DTW filter, assign penalty
                            ot_raw = OT_PENALTY_VALUE
                        
                        pair_info.append((j, candidate, dtw_raw, ot_raw, centroid_raw, cluster1, cluster2))
                        
                    except Exception:
                        continue
                
                # ========== STEP 1.5: Batch GPU OT computation ==========
                if len(ot_batch_data) > 0 and torch.cuda.is_available():
                    # Prepare batch OT computation
                    C_batch = []
                    weights_a_batch = []
                    weights_b_batch = []
                    
                    for cluster1, cluster2 in ot_batch_data:
                        # Compute cost matrix using GPU
                        C = compute_ot_cost_matrix(cluster1, cluster2, use_gpu_batch=True)
                        areas1 = np.array([c.area for c in cluster1], dtype=np.float64)
                        areas2 = np.array([c.area for c in cluster2], dtype=np.float64)
                        weights1 = _normalize_area_weights(areas1)
                        weights2 = _normalize_area_weights(areas2)
                        
                        C_batch.append(C)
                        weights_a_batch.append(weights1)
                        weights_b_batch.append(weights2)
                    
                    # Batch GPU Sinkhorn OT (10-100x FASTER than CPU!)
                    try:
                        ot_distances, T_batch = sinkhorn_ot_gpu_batch(C_batch, weights_a_batch, weights_b_batch)
                        
                        # Update pair_info with computed OT values
                        for batch_idx, pair_idx in enumerate(ot_batch_indices):
                            ot_val = ot_distances[batch_idx]
                            # Unify inf handling: treat as penalty for consistency
                            if not np.isfinite(ot_val):
                                ot_val = OT_PENALTY_VALUE
                            # Update ot_raw in pair_info
                            j, candidate, dtw_raw, _, centroid_raw, cluster1, cluster2 = pair_info[pair_idx]
                            pair_info[pair_idx] = (j, candidate, dtw_raw, ot_val, centroid_raw, cluster1, cluster2)
                    except Exception as e:
                        # Batch OT failed, fall back to individual computation
                        for batch_idx, pair_idx in enumerate(ot_batch_indices):
                            j, candidate, dtw_raw, _, centroid_raw, cluster1, cluster2 = pair_info[pair_idx]
                            try:
                                C = C_batch[batch_idx]
                                weights1 = weights_a_batch[batch_idx]
                                weights2 = weights_b_batch[batch_idx]
                                ot_val, T = solve_optimal_transport(C, weights1, weights2, use_gpu=True)
                                # Unify inf handling: treat as penalty
                                if not np.isfinite(ot_val):
                                    ot_val = OT_PENALTY_VALUE
                                pair_info[pair_idx] = (j, candidate, dtw_raw, ot_val, centroid_raw, cluster1, cluster2)
                            except:
                                # Computation failed, use penalty
                                pair_info[pair_idx] = (j, candidate, dtw_raw, OT_PENALTY_VALUE, centroid_raw, cluster1, cluster2)
                
                # ========== STEP 2: Build raw_values from pair_info ==========
                raw_values = [(j, candidate, dtw_raw, ot_raw, centroid_raw) 
                             for j, candidate, dtw_raw, ot_raw, centroid_raw, _, _ in pair_info]
                
                if not raw_values:
                    continue
                
                # ========== STEP 3: Calculate LOCAL normalization ==========
                dtw_raws = [v[2] for v in raw_values]
                ot_raws = [v[3] for v in raw_values if v[3] < OT_PENALTY_VALUE]  # Exclude penalty values
                centroid_raws = [v[4] for v in raw_values]
                
                dtw_local_scale = np.percentile(dtw_raws, 75) * 10.0 / 7.5 if len(dtw_raws) > 0 else 1.0
                ot_local_scale = np.percentile(ot_raws, 75) * 10.0 / 7.5 if len(ot_raws) > 0 else 1.0
                centroid_local_scale = np.percentile(centroid_raws, 75) * 10.0 / 7.5 if len(centroid_raws) > 0 else 1.0
                
                # Add lower bound protection to prevent division by zero or extreme scaling
                # If all values are very small (e.g., trivial OT cases), scale becomes unreliable
                dtw_local_scale = max(dtw_local_scale, 1e-6)
                ot_local_scale = max(ot_local_scale, 1e-6)
                centroid_local_scale = max(centroid_local_scale, 1e-6)
                
                # ========== STEP 4: Compute dual losses ==========
                for j, candidate, dtw_raw, ot_raw, centroid_raw in raw_values:
                    try:
                        # LOCAL loss (for candidate selection)
                        dtw_loss_local = min(10.0, dtw_raw / dtw_local_scale * 10.0)
                        ot_loss_local = min(10.0, ot_raw / ot_local_scale * 10.0)
                        centroid_loss_local = min(10.0, centroid_raw / centroid_local_scale * 10.0)
                        
                        loss_local = (DTW_WEIGHT * dtw_loss_local + 
                                     OT_WEIGHT * ot_loss_local + 
                                     CENTROID_WEIGHT * centroid_loss_local)
                        
                        # GLOBAL loss (for quality scoring)
                        dtw_loss_global = min(10.0, dtw_raw / DTW_NORM_DIVISOR * 10.0)
                        
                        # Handle OT penalty values specially for GLOBAL scoring
                        if ot_raw == OT_PENALTY_VALUE:
                            # Penalty value: assign max loss (10.0) instead of normalizing
                            ot_loss_global = 10.0
                        else:
                            ot_loss_global = min(10.0, ot_raw / OT_NORM_DIVISOR * 10.0)
                        
                        centroid_loss_global = min(10.0, centroid_raw / CENTROID_NORM_DIVISOR * 10.0)
                        
                        loss_global = (DTW_WEIGHT * dtw_loss_global + 
                                      OT_WEIGHT * ot_loss_global + 
                                      CENTROID_WEIGHT * centroid_loss_global)
                        
                        # Save result with dual losses
                        results.append((candidate, loss_local, loss_global, dtw_raw, ot_raw, centroid_raw))
                        successful_pairs += 1
                        
                    except Exception:
                        continue
                
                # Clean up batch GPU tensors to free memory
                del x_batch_stacked, y_batch_stacked, dtw_dists, dtw_dists_cpu
                
                if stats:
                    batch_time = time.time() - batch_start_time
                    stats['batch_times'] = stats.get('batch_times', []) + [batch_time]
                    stats['batch_sizes'] = stats.get('batch_sizes', []) + [len(batch_pairs)]
            
            except Exception:
                continue
        
        # Collect statistics
        if stats:
            stats['gpu_batch_total_time'] = time.time() - start_time
            stats['total_pairs'] = len(valid_pairs)
            stats['successful_pairs'] = successful_pairs
        
        return results
    
    except Exception:
        if stats:
            stats['batch_errors'] = stats.get('batch_errors', 0) + 1
        return []  # Return empty result on error
    
    finally:
        # Clean up GPU memory only at the end of all batches
        if use_cuda:
            torch.cuda.empty_cache()            


def compute_adaptive_norm_divisors(cell, candidates, used_cells, direction, cells_by_slice):
    """
    Quick pass to compute adaptive normalization divisors based on candidate statistics
    
    Returns:
        (dtw_divisor, ot_divisor): Adaptive normalization divisors, or (None, None) if not enough samples
    """
    try:
        # Quick filter to get valid candidates
        valid_candidates = []
        for candidate in candidates:
            if direction == "forward" and candidate.slice_index != cell.slice_index + 1:
                continue
            if direction == "backward" and candidate.slice_index != cell.slice_index - 1:
                continue
            if candidate.cell_id in used_cells:
                continue
            if not is_within_spatial_window(cell, candidate):
                continue
            valid_candidates.append(candidate)
        
        if len(valid_candidates) < 5:
            return None, None
        
        # Sample up to 20 candidates for statistics (for speed)
        import random
        sample_size = min(20, len(valid_candidates))
        sampled = random.sample(valid_candidates, sample_size)
        
        # Collect DTW samples (using pre-computed resampled contours)
        dtw_samples = []
        for candidate in sampled:
            try:
                # Use pre-computed resampled contours
                contour1 = cell.contour_points_200
                contour2 = candidate.contour_points_200
                if len(contour1) <= 1024 and len(contour2) <= 1024:
                    dtw_val = dtw_distance_gpu_single(contour1, contour2)
                    if dtw_val != float('inf'):
                        dtw_samples.append(dtw_val)
            except:
                pass
        
        # Collect OT samples (fewer, as OT is expensive)
        ot_samples = []
        if cells_by_slice is not None and len(sampled) >= 3:
            ot_sample_size = min(5, len(sampled))
            for candidate in sampled[:ot_sample_size]:
                try:
                    cluster1 = get_cluster_cells(cell, cells_by_slice)
                    cluster2 = get_cluster_cells(candidate, cells_by_slice)
                    if len(cluster1) > 1 or len(cluster2) > 1:
                        C = compute_ot_cost_matrix(cluster1, cluster2)
                        areas1 = np.array([c.area for c in cluster1])
                        areas2 = np.array([c.area for c in cluster2])
                        weights1 = _normalize_area_weights(areas1)
                        weights2 = _normalize_area_weights(areas2)
                        ot_raw, T = solve_optimal_transport(C, weights1, weights2, use_gpu=True)
                        if ot_raw != float('inf'):
                            ot_samples.append(ot_raw)
                except:
                    pass
        
        # Compute adaptive divisors
        dtw_divisor = None
        ot_divisor = None
        
        if len(dtw_samples) >= 3:
            dtw_75p = np.percentile(dtw_samples, 75)
            dtw_divisor = dtw_75p * 10.0 / 7.5  # Map 75p to 7.5
            # Safety bounds
            dtw_divisor = np.clip(dtw_divisor, DTW_NORM_DIVISOR / 5.0, DTW_NORM_DIVISOR * 5.0)
        
        if len(ot_samples) >= 2:
            ot_75p = np.percentile(ot_samples, 75)
            ot_divisor = ot_75p * 10.0 / 7.5
            # Safety bounds
            ot_divisor = np.clip(ot_divisor, OT_NORM_DIVISOR / 5.0, OT_NORM_DIVISOR * 5.0)
        
        return dtw_divisor, ot_divisor
    
    except Exception as e:
        return None, None


def find_best_match(cell, candidates, used_cells, top_k=20, use_gpu=True, direction="forward", cells_by_slice=None, use_adaptive_norm=False):
    """
    Find the best matching cell using Relative Margin threshold (DUAL NORMALIZATION)
    
    DUAL NORMALIZATION SYSTEM:
    - LOCAL normalization: Adaptive to current batch, used for candidate selection
    - GLOBAL normalization: Fixed divisors, used for quality scoring
    
    Process:
    1. Compute dual losses (local & global) via batch_calculate_dtw
    2. Use LOCAL scores to select top1/top2 (better discrimination)
    3. Return GLOBAL margin for track quality scoring (consistent comparison)
    
    Args:
        cell: Current cell
        candidates: List of candidate cells
        used_cells: Set of used cell IDs
        top_k: Number of candidates to filter based on centroid distance
        use_gpu: Whether to use GPU acceleration
        direction: Track construction direction, "forward" or "backward"
        cells_by_slice: Dictionary of cells by slice for OT cluster analysis
        use_adaptive_norm: Whether to use adaptive normalization (deprecated, now always uses dual norm)
        
    Returns:
        best_cell: Best matching cell (top1, selected using LOCAL scores)
        relative_margin: Margin / top1_score (LOCAL, for decision threshold)
        top1_score_local: LOCAL score of best match
        top2_score_local: LOCAL score of second best match
        margin_global: GLOBAL margin (for quality scoring and track evaluation)
    """
    if not candidates:
        return None, 0.0, 0.0, 0.0, 0.0  # best_cell, relative_margin, top1_score_local, top2_score_local, margin_global

    # Check GPU availability
    gpu_available = use_gpu and torch.cuda.is_available()
    if use_gpu and not gpu_available:
        print("Warning: GPU requested but not available")
        return None, 0.0, 0.0, 0.0, 0.0  # best_cell, relative_margin, top1_score_local, top2_score_local, margin_global
    
    # === ADAPTIVE NORMALIZATION (optional) ===
    global DTW_NORM_DIVISOR, OT_NORM_DIVISOR
    original_dtw_divisor = DTW_NORM_DIVISOR
    original_ot_divisor = OT_NORM_DIVISOR
    
    try:
        if use_adaptive_norm:
            adaptive_dtw, adaptive_ot = compute_adaptive_norm_divisors(
                cell, candidates, used_cells, direction, cells_by_slice
            )
            if adaptive_dtw is not None:
                DTW_NORM_DIVISOR = adaptive_dtw
            if adaptive_ot is not None:
                OT_NORM_DIVISOR = adaptive_ot
        
        # Filter out unreasonable candidate layers
        filtered_candidates = []
        current_slice = cell.slice_index
    
        for candidate in candidates:
            # If forward matching, only accept higher layer candidates with consecutive layer numbers
            if direction == "forward" and candidate.slice_index != current_slice + 1:
                continue
            # If backward matching, only accept lower layer candidates with consecutive layer numbers
            if direction == "backward" and candidate.slice_index != current_slice - 1:
                continue
            # If direction unspecified, ensure consecutive layer numbers (difference of 1)
            if direction not in ["forward", "backward"] and abs(candidate.slice_index - current_slice) != 1:
                continue
                
            # Check if candidate cell has been used
            if candidate.cell_id in used_cells:
                continue
                
            filtered_candidates.append(candidate)
        
        if not filtered_candidates:
            return None, 0.0, 0.0, 0.0, 0.0

        # Filter Top-K candidates based on centroid Euclidean distance
        def centroid_distance(candidate):
            return np.linalg.norm(cell.centroid - candidate.centroid)

        # Use only centroid distance for candidate filtering (SIMPLIFIED)
        if len(filtered_candidates) > top_k:
            nearest_candidates = heapq.nsmallest(top_k, filtered_candidates, key=centroid_distance)
        else:
            nearest_candidates = filtered_candidates
        
        # Additional spatial window filtering
        within_window_candidates = [c for c in nearest_candidates if is_within_spatial_window(cell, c)]
        if not within_window_candidates:
            return None, 0.0, 0.0, 0.0, 0.0
        
        results = []
        
        # Use GPU for processing (with dual normalization)
        if gpu_available:
            results = batch_calculate_dtw(cell, within_window_candidates, used_cells, direction=direction, cells_by_slice=cells_by_slice)
            if not results:  # If GPU calculation fails, try individual calculation
                for candidate in within_window_candidates:
                    if candidate.cell_id not in used_cells:
                        loss, dtw_raw, ot_raw, centroid_raw = calculate_loss(
                            cell, candidate, use_gpu=True, cells_by_slice=cells_by_slice, return_details=True
                        )
                        if loss != float('inf'):
                            # Individual calculation: use same loss for both local and global (no adaptive scaling)
                            # Include real raw values to avoid polluting statistics
                            results.append((candidate, loss, loss, dtw_raw, ot_raw, centroid_raw))
        else:
            return None, 0.0, 0.0, 0.0, 0.0

        if not results:
            return None, 0.0, 0.0, 0.0, 0.0

        # Convert LOCAL loss to score for candidate selection
        # Lower loss -> higher score
        candidate_scores = []
        for result in results:
            # Handle both dual-norm format and fallback format
            if len(result) == 6:
                candidate, loss_local, loss_global, dtw_raw, ot_raw, centroid_raw = result
            else:
                # Fallback: old format
                candidate, loss_local = result[0], result[1]
                loss_global = loss_local
            
            if loss_local != float('inf'):
                score_local = 1.0 / (1.0 + loss_local)  # Use LOCAL loss for scoring
                score_global = 1.0 / (1.0 + loss_global)  # Also compute GLOBAL score
                candidate_scores.append((candidate, score_local, score_global, loss_local, loss_global))
        
        if not candidate_scores:
            return None, 0.0, 0.0, 0.0, 0.0
        
        # Sort by LOCAL score (descending) - use local for selection
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top1 and top2 based on LOCAL scores
        top1_candidate, top1_score_local, top1_score_global, top1_loss_local, top1_loss_global = candidate_scores[0]
        
        if len(candidate_scores) > 1:
            top2_candidate, top2_score_local, top2_score_global, top2_loss_local, top2_loss_global = candidate_scores[1]
            margin_local = top1_score_local - top2_score_local
            margin_global = top1_score_global - top2_score_global
        else:
            # Only one candidate
            top2_score_local = 0.0
            top2_score_global = 0.0
            margin_local = top1_score_local
            margin_global = top1_score_global
        
        # Calculate relative margin based on LOCAL scores (for decision making)
        if top1_score_local > 1e-6:  # Avoid division by zero
            relative_margin = margin_local / top1_score_local
        else:
            relative_margin = 0.0
        
        # Return: use LOCAL relative_margin for decision, but also return GLOBAL margin for quality scoring
        return top1_candidate, relative_margin, top1_score_local, top2_score_local, margin_global
    
    finally:
        # Always restore original normalization divisors
        DTW_NORM_DIVISOR = original_dtw_divisor
        OT_NORM_DIVISOR = original_ot_divisor

def build_track_from_cell(base_cell, cells_by_slice, used_cells, direction=None):
    """
    Build track from starting cell using Relative Margin threshold
    
    CHANGED: Uses relative_margin = margin / top1_score >= RELATIVE_MARGIN_THRESHOLD
    - Collects margin values for quality scoring (track perplexity)
    - No need for margin_threshold parameter, uses global RELATIVE_MARGIN_THRESHOLD
    
    Args:
        base_cell: Starting cell
        cells_by_slice: Dictionary of cells organized by layer index
        used_cells: Set of used cell IDs
        direction: Track construction direction, "forward" or "backward", None means auto-determine
        
    Returns:
        track: Built track with margin information
    """
    track = CellTrack(base_cell)
    global all_margins, all_decision_margins, all_decision_info
    
    # If no direction explicitly specified, auto-determine based on starting layer
    if direction is None:
        # Determine track direction
        is_forward = True  # Default to upward construction (increasing layer numbers)
        
        # If starting layer is too high, can change to downward construction
        if base_cell.slice_index > 8:  # This threshold can be adjusted
            is_forward = False
            
        direction = "forward" if is_forward else "backward"
    
    if direction == "forward":
        # Upward matching (increasing layer numbers)
        current_cell = base_cell
        current_slice = base_cell.slice_index
        
        # Increment layer numbers, ensure continuity
        while True:
            next_slice = current_slice + 1
            
            # If already exceeding maximum layer, stop
            if next_slice > max(cells_by_slice.keys()):
                break
                
            # If no data for this layer, stop
            if next_slice not in cells_by_slice:
                break
                
            candidates = cells_by_slice[next_slice]
            best_match, relative_margin, top1_score_local, top2_score_local, margin_global = find_best_match(
                current_cell, 
                candidates, 
                used_cells, 
                direction="forward",
                cells_by_slice=cells_by_slice
            )
            
            # Record all decision margins (regardless of acceptance)
            # NEW: Record ALL decision margins (eliminating selection bias)
            if best_match is not None:
                # If there is a candidate: record actual relative_margin
                all_decision_margins.append(relative_margin)
                accepted = (relative_margin >= RELATIVE_MARGIN_THRESHOLD)
                all_decision_info.append({
                    'slice': next_slice,
                    'has_candidate': True,
                    'relative_margin': relative_margin,
                    'accepted': accepted
                })
            else:
                # If there is no candidate: record very low value indicating high perplexity
                all_decision_margins.append(0.0)
                all_decision_info.append({
                    'slice': next_slice,
                    'has_candidate': False,
                    'relative_margin': 0.0,
                    'accepted': False
                })
            
            # Accept match if relative_margin >= threshold (using LOCAL margin for decision)
            if best_match and relative_margin >= RELATIVE_MARGIN_THRESHOLD:
                track.add_cell(best_match)
                # Store GLOBAL margin for quality scoring (ensures consistency across tracks)
                track.margins.append(margin_global)  # Store GLOBAL margin for quality scoring
                all_margins.append(margin_global)    # Collect GLOBAL margins for analysis
                current_cell = best_match
                current_slice = next_slice
            else:
                break
    else:  # direction == "backward"
        # Downward matching (decreasing layer numbers)
        current_cell = base_cell
        current_slice = base_cell.slice_index
        
        # Decrement layer numbers, ensure continuity
        while True:
            prev_slice = current_slice - 1
            
            # If already below minimum layer, stop
            if prev_slice < min(cells_by_slice.keys()):
                break
                
            # If no data for this layer, stop
            if prev_slice not in cells_by_slice:
                break
                
            candidates = cells_by_slice[prev_slice]
            best_match, relative_margin, top1_score_local, top2_score_local, margin_global = find_best_match(
                current_cell, 
                candidates, 
                used_cells, 
                direction="backward",
                cells_by_slice=cells_by_slice
            )
            
            # Record all decision margins (regardless of acceptance)
            # NEW: Record ALL decision margins (eliminating selection bias)
            if best_match is not None:
                # If there is a candidate: record actual relative_margin
                all_decision_margins.append(relative_margin)
                accepted = (relative_margin >= RELATIVE_MARGIN_THRESHOLD)
                all_decision_info.append({
                    'slice': prev_slice,
                    'has_candidate': True,
                    'relative_margin': relative_margin,
                    'accepted': accepted
                })
            else:
                # If there is no candidate: record very low value indicating high perplexity
                all_decision_margins.append(0.0)
                all_decision_info.append({
                    'slice': prev_slice,
                    'has_candidate': False,
                    'relative_margin': 0.0,
                    'accepted': False
                })
            
            # Accept match if relative_margin >= threshold (using LOCAL margin for decision)
            if best_match and relative_margin >= RELATIVE_MARGIN_THRESHOLD:
                track.add_cell(best_match)
                # Store GLOBAL margin for quality scoring (ensures consistency across tracks)
                track.margins.append(margin_global)  # Store GLOBAL margin for quality scoring
                all_margins.append(margin_global)    # Collect GLOBAL margins for analysis
                current_cell = best_match
                current_slice = prev_slice
            else:
                break

    return track


def merge_bidirectional_tracks(forward_track, backward_track):
    """
    Merge forward and backward tracks built from the same starting cell
    
    IMPORTANT: Margins must be reversed for backward track to maintain correct order!
    
    This allows building complete tracks that span both directions from a middle starting point,
    instead of forcing a choice between forward-only or backward-only.
    
    Args:
        forward_track: Track built in forward direction (increasing slices)
                      cells: {S, S+1, S+2, ...}
                      margins: [margin(S→S+1), margin(S+1→S+2), ...]
        backward_track: Track built in backward direction (decreasing slices)
                       cells: {S, S-1, S-2, ...}
                       margins: [margin(S→S-1), margin(S-1→S-2), ...]  # Built going DOWN
    
    Returns:
        Merged track: cells={..., S-2, S-1, S, S+1, S+2, ...}
                     margins: [..., margin(S-2→S-1), margin(S-1→S), margin(S→S+1), margin(S+1→S+2), ...]
    """
    # Both tracks share the same starting cell at their origins
    # We need to merge them without duplicating the starting cell
    
    # Get all slices from both tracks
    forward_slices = sorted(forward_track.cells.keys())
    backward_slices = sorted(backward_track.cells.keys())
    
    # The starting cell should be the same (lowest in forward, highest in backward)
    start_slice = forward_slices[0]  # Should equal backward_slices[-1] if built correctly
    
    # Create merged track starting from the base cell
    base_cell = forward_track.cells[start_slice]
    merged = CellTrack(base_cell)
    
    # Add cells from backward track (lower slices, excluding starting cell)
    # Must add in ascending order: S-2, S-1 (not S-1, S-2)
    backward_lower = sorted([s for s in backward_slices if s < start_slice])
    for slice_idx in backward_lower:
        merged.add_cell(backward_track.cells[slice_idx])
    
    # Add margins from backward track (REVERSED!)
    # backward margins are [S→S-1, S-1→S-2], but we need [..., S-2→S-1, S-1→S]
    # Reverse to get correct order for merged track going upward
    merged.margins.extend(reversed(backward_track.margins))
    
    # Add cells from forward track (higher slices, excluding starting cell)
    for slice_idx in forward_slices:
        if slice_idx > start_slice:  # Only add cells above starting point
            merged.add_cell(forward_track.cells[slice_idx])
    
    # Add margins from forward track (already in correct order)
    merged.margins.extend(forward_track.margins)
    
    return merged


def find_high_quality_tracks_layerwise(cells_by_slice, output_prefix="", verbose=True):
    """Build cell tracks using Relative Margin threshold (ONE-STAGE approach)
    
    RELATIVE MARGIN METHOD:
    - Accept match if relative_margin = margin / top1_score >= RELATIVE_MARGIN_THRESHOLD
    - No need for two-stage process, directly build tracks with threshold
    - More efficient: only one pass through the data
    
    Args:
        cells_by_slice: Dictionary of cells organized by slice
        output_prefix: Prefix for output filenames (e.g., 'cell-001_')
        verbose: If False, suppress explanatory logging (progress bars still show tqdm).
    """
    import matplotlib.pyplot as plt
    
    global all_margins, all_decision_margins, all_decision_info
    
    # Get minimum and maximum layer numbers
    min_slice = min(cells_by_slice.keys())
    max_slice = max(cells_by_slice.keys())
    
    # Reset global margin collection (for analysis only, not for threshold)
    all_margins = []
    
    # Reset global decision-level margin collectors
    # NEW: Reset global decision-level margin collectors
    all_decision_margins = []
    all_decision_info = []
    
    if verbose:
        print("\n" + "="*80)
        print(f"Building tracks with Relative Margin threshold = {RELATIVE_MARGIN_THRESHOLD:.2f}")
        print("="*80)
        print(
            "Note: Accept match if (top1_score - top2_score) / top1_score >= "
            f"{RELATIVE_MARGIN_THRESHOLD:.2f}"
        )

    all_tracks = []
    used_cells = set()
    iteration = 1

    while True:
        if verbose:
            print(f"\nStarting round {iteration} track construction (Relative Margin)")
        new_tracks = []

        # Process in layer order, from layer 1 to maximum layer
        for start_slice in sorted(cells_by_slice.keys()):
            slice_cells = cells_by_slice[start_slice]
            slice_tracks = []

            pending_cells = [c for c in slice_cells if c.cell_id not in used_cells]
            if verbose:
                print(
                    f"Round {iteration} - layer {start_slice}: "
                    f"{len(pending_cells)} unused cells (of {len(slice_cells)} in slice)"
                )

            # Only iterate cells not yet assigned — tqdm total matches real work per round
            for cell in tqdm(
                pending_cells,
                desc=f"Round {iteration} - slice {start_slice}",
                disable=False,
            ):
                # Build track upward from starting layer (uses global RELATIVE_MARGIN_THRESHOLD)
                forward_track = build_track_from_cell(
                    cell, 
                    cells_by_slice, 
                    used_cells,
                    direction="forward"
                )
                
                # If starting layer is not the lowest, can also build track downward
                if start_slice > min_slice:
                    backward_track = build_track_from_cell(
                        cell, 
                        cells_by_slice, 
                        used_cells,
                        direction="backward"
                    )
                    
                    # Merge both directions to get complete track (not just pick longer one)
                    # This ensures we don't lose valid connections in either direction
                    track = merge_bidirectional_tracks(forward_track, backward_track)
                else:
                    track = forward_track

                # Only keep tracks that are long enough (>=3 layers)
                if track.get_layer_count() >= 3:
                    slice_tracks.append(track)
            
            # Add all tracks built from this layer to current round track set
            new_tracks.extend(slice_tracks)

        # If no new tracks found, end iteration
        if not new_tracks:
            if verbose:
                print("No new tracks found, iteration ended")
            break
            
        # Validate track validity (continuity and direction consistency)
        valid_tracks = []
        for track in new_tracks:
            slices = sorted(track.cells.keys())
            is_valid = True
            
            # Check if track is continuous
            for i in range(1, len(slices)):
                if slices[i] - slices[i-1] != 1:
                    is_valid = False
                    break
            
            # Check if track is within valid range (1-10 layers)
            if is_valid and (slices[-1] > max_slice or slices[0] < min_slice):
                is_valid = False
                
            # Ensure track doesn't jump back (S1->S10 then back to S1)
            if is_valid and len(slices) > 0:
                first_slice = slices[0]
                last_slice = slices[-1]
                # Check for circular tracks
                if first_slice in track.cells and last_slice in track.cells:
                    # Calculate track order consistency
                    is_ascending = last_slice > first_slice
                    for i in range(1, len(slices)):
                        if (slices[i] > slices[i-1]) != is_ascending:
                            is_valid = False
                            break
            
            if is_valid:
                valid_tracks.append(track)
        
        if verbose:
            print(f"After filtering unreasonable tracks, retained {len(valid_tracks)}/{len(new_tracks)} tracks")
        new_tracks = valid_tracks

        # ========== Compute normalized quality scores (70% margin + 30% length) ==========
        if verbose:
            print(f"This round built {len(new_tracks)} tracks, computing normalized quality scores...")
        
        # Step 1: Collect all margin and length values
        track_margins = []
        track_lengths = []
        for track in new_tracks:
            q_margin = track.get_quality_score(cells_by_slice)  # Average margin
            length = track.get_layer_count()
            track_margins.append(q_margin)
            track_lengths.append(length)
        
        # Step 2: Calculate normalization parameters
        if len(track_margins) > 0:
            margin_min = min(track_margins)
            margin_max = max(track_margins)
            max_length_ref = max(track_lengths)
        else:
            margin_min = 0
            margin_max = 1
            max_length_ref = 10
        
        # Avoid division by zero
        margin_range = margin_max - margin_min
        if margin_range < 1e-9:
            margin_range = 1.0
        
        if verbose:
            print(f"  Margin range: [{margin_min:.6f}, {margin_max:.6f}]")
            print(f"  Length range: [3, {max_length_ref}]")
            print(
                "  Comprehensive quality = "
                f"{TRACK_MARGIN_WEIGHT:.1%} × margin + {TRACK_LENGTH_WEIGHT:.1%} × length"
            )
        
        # Step 3: Define comprehensive quality function
        def compute_track_quality(track):
            """
            Compute normalized comprehensive quality score
            
            Score = TRACK_MARGIN_WEIGHT × normalized_margin + TRACK_LENGTH_WEIGHT × normalized_length
            Default: 70% margin + 30% length
            
            This favors tracks with both:
            - High margin (low perplexity, confident matches)
            - Long tracks (more layers, better reconstruction)
            
            Returns:
                Comprehensive quality score in [0, 1]
            """
            q_margin = track.get_quality_score(cells_by_slice)
            length = track.get_layer_count()
            
            # Normalize margin to [0, 1]
            normalized_margin = (q_margin - margin_min) / margin_range
            
            # Normalize length to [0, 1]
            normalized_length = min(1.0, length / max_length_ref)
            
            # Weighted combination (use global weights)
            comprehensive_quality = (TRACK_MARGIN_WEIGHT * normalized_margin + 
                                    TRACK_LENGTH_WEIGHT * normalized_length)
            
            return comprehensive_quality
        
        # Sort all tracks by comprehensive quality score (higher is better)
        new_tracks.sort(key=lambda x: compute_track_quality(x), reverse=True)
        
        # Show top and bottom tracks for debugging
        if verbose and len(new_tracks) > 0:
            top_track = new_tracks[0]
            top_quality = compute_track_quality(top_track)
            top_margin = top_track.get_quality_score(cells_by_slice)
            top_length = top_track.get_layer_count()
            
            print(f"  Top track: margin={top_margin:.6f}, length={top_length}, comprehensive_quality={top_quality:.4f}")
            
            if len(new_tracks) > 1:
                bottom_track = new_tracks[-1]
                bottom_quality = compute_track_quality(bottom_track)
                bottom_margin = bottom_track.get_quality_score(cells_by_slice)
                bottom_length = bottom_track.get_layer_count()
                print(f"  Bottom track: margin={bottom_margin:.6f}, length={bottom_length}, comprehensive_quality={bottom_quality:.4f}")
        
        if verbose:
            print(
                "Now selecting based on comprehensive quality "
                f"({TRACK_MARGIN_WEIGHT:.0%} margin + {TRACK_LENGTH_WEIGHT:.0%} length) "
                "with conflict resolution..."
            )
        
        # METHOD B: Select tracks with quality-based conflict resolution
        # Instead of just checking disjoint sets, compare track qualities when conflicts occur
        cell_to_track = {}  # Maps cell_id -> track object
        selected_tracks = []  # Final list of selected tracks
        conflict_count = 0
        replacement_count = 0
        
        for track in new_tracks:
            quality = compute_track_quality(track)  # Use comprehensive quality
            track_cell_ids = set(cell.cell_id for cell in track.cells.values())
            
            # Find all tracks that share cells with current track
            conflicting_tracks = set()
            for cell_id in track_cell_ids:
                if cell_id in cell_to_track:
                    conflicting_tracks.add(cell_to_track[cell_id])
            
            if not conflicting_tracks:
                # No conflict - add track directly
                selected_tracks.append(track)
                for cell_id in track_cell_ids:
                    cell_to_track[cell_id] = track
            else:
                # Conflict detected - compare qualities
                conflict_count += 1
                
                # Find the minimum quality among conflicting tracks (using comprehensive quality)
                min_conflicting_quality = min(
                    compute_track_quality(t) for t in conflicting_tracks
                )
                
                # If current track is better than the worst conflicting track, replace it
                if quality > min_conflicting_quality:
                    # Find which track to replace (the one with minimum comprehensive quality)
                    # Use min() instead of == comparison to avoid floating-point equality issues
                    track_to_replace = min(conflicting_tracks, key=compute_track_quality)
                    
                    if track_to_replace:
                        # Remove the old track
                        selected_tracks.remove(track_to_replace)
                        replacement_count += 1
                        
                        # Remove old track's cells from mapping
                        old_cell_ids = set(c.cell_id for c in track_to_replace.cells.values())
                        for old_cell_id in old_cell_ids:
                            if old_cell_id in cell_to_track and cell_to_track[old_cell_id] == track_to_replace:
                                del cell_to_track[old_cell_id]
                        
                        # Add new track
                        selected_tracks.append(track)
                        for cell_id in track_cell_ids:
                            cell_to_track[cell_id] = track
                # else: current track is not better, skip it
        
        high_quality_tracks = selected_tracks
        
        if verbose:
            print(f"  Selected {len(high_quality_tracks)} tracks")
            print(f"  Conflicts encountered: {conflict_count}")
            print(f"  Tracks replaced due to better quality: {replacement_count}")

        # Show comprehensive quality statistics of selected tracks
        if verbose and len(high_quality_tracks) > 0:
            selected_qualities = [compute_track_quality(t) for t in high_quality_tracks]
            selected_margins = [t.get_quality_score(cells_by_slice) for t in high_quality_tracks]
            selected_lengths = [t.get_layer_count() for t in high_quality_tracks]
            
            print(f"  Selected tracks statistics:")
            print(f"    Comprehensive quality: mean={np.mean(selected_qualities):.4f}, "
                  f"min={np.min(selected_qualities):.4f}, max={np.max(selected_qualities):.4f}")
            print(f"    Margin: mean={np.mean(selected_margins):.6f}, "
                  f"min={np.min(selected_margins):.6f}, max={np.max(selected_margins):.6f}")
            print(f"    Length: mean={np.mean(selected_lengths):.2f}, "
                  f"min={np.min(selected_lengths)}, max={np.max(selected_lengths)}")

        # Output CSV with margin-based quality metrics (SIMPLIFIED)
        csv_filename = f"{output_prefix}tracks_round_{iteration}.csv"
        with open(csv_filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "track_id", "start_slice", "end_slice", "cell_count", 
                "avg_margin", "min_margin", "max_margin", "std_margin", "direction", 
                "cell_ids"
            ])
            for i, track in enumerate(high_quality_tracks):
                slices = sorted(track.cells.keys())
                if len(slices) >= 2:
                    direction = "ascending" if slices[-1] > slices[0] else "descending"
                else:
                    direction = "unknown"
                
                # Get margin-based quality metrics
                quality_details = track.get_quality_details(cells_by_slice)
                    
                cell_ids = [str(track.cells[s].cell_id) for s in sorted(track.cells.keys())]
                
                writer.writerow([
                    i + 1, 
                    track.start_slice, 
                    track.end_slice, 
                    track.get_layer_count(),
                    f"{quality_details['avg_margin']:.6f}",
                    f"{quality_details['min_margin']:.6f}",
                    f"{quality_details['max_margin']:.6f}",
                    f"{quality_details['std_margin']:.6f}",
                    direction,
                    '|'.join(cell_ids)
                ])
        if verbose:
            print(f"Saved round {iteration} track data as {csv_filename}")

        # Update global track list and used cell set
        all_tracks.extend(high_quality_tracks)
        for track in high_quality_tracks:
            for cell in track.cells.values():
                used_cells.add(cell.cell_id)

        # Draw and save margin distribution plot
        track_margins = []
        for track in high_quality_tracks:
            track_margins.extend(track.margins)
        
        if track_margins:
            plt.figure(figsize=(8, 5))
            plt.hist(track_margins, bins=50, color='skyblue', edgecolor='black')
            # Note: margin_threshold is not used anymore in Relative Margin method
            # Instead, we show the distribution of actual margins
            plt.title(f"Margin Distribution - Round {iteration}\n(Relative Margin threshold α={RELATIVE_MARGIN_THRESHOLD:.2f})")
            plt.xlabel("Margin (top1_score - top2_score)")
            plt.ylabel("Number of Connections")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_prefix}margin_distribution_round_{iteration}.png")
            plt.close()
            if verbose:
                print(f"Saved round {iteration} margin histogram")
            
        # Visualize current round track distribution
        if high_quality_tracks:
            plt.figure(figsize=(10, 6))
            for i, track in enumerate(high_quality_tracks[:min(20, len(high_quality_tracks))]):
                slices = sorted(track.cells.keys())
                centroids = [track.cells[s].centroid for s in slices]
                x = [c[0] for c in centroids]
                y = [c[1] for c in centroids]
                plt.plot(x, y, 'o-', linewidth=1, markersize=4, label=f"Track {i+1}")
            
            plt.title(f"Top Tracks Spatial Distribution - Round {iteration}")
            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")
            plt.grid(True, linestyle='--', alpha=0.7)
            if len(high_quality_tracks) <= 10:  # Show legend only when there are few tracks
                plt.legend(loc='best', fontsize='small')
            plt.tight_layout()
            plt.savefig(f"{output_prefix}tracks_distribution_round_{iteration}.png")
            plt.close()
            if verbose:
                print(f"Saved round {iteration} track distribution plot")
        
        iteration += 1

    return all_tracks
  

def plot_track_length_distribution(tracks, filename="track_lengths.png"):
    """Plot track length distribution histogram"""
    import matplotlib.pyplot as plt

    # Handle empty tracks list
    if not tracks:
        print(f"Warning: No tracks to plot, skipping {filename}")
        return
    
    lengths = [track.get_layer_count() for track in tracks]
    if not lengths:
        print(f"Warning: No track lengths to plot, skipping {filename}")
        return
    
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=range(1, max(lengths)+2), align='left', color='teal', edgecolor='black')
    plt.xlabel("Number of Layers in Track")
    plt.ylabel("Number of Tracks")
    plt.title("Track Length Distribution")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
 
    
def evaluate_all_tracks(tracks, cells_by_slice):
    """Perform global scoring for all tracks (DECISION-LEVEL MARGIN-BASED SCORING)
    
    This provides true "global confidence" instead of "confidence of easy samples only"
    """
    
    global all_decision_margins, all_decision_info
    
    # Initialize scoring metrics
    metrics = {
        "total_tracks": len(tracks),
        "average_length": 0,
        "length_distribution": {},
        "coverage_ratio": 0,
        "global_avg_margin": 0,        # Global average decision margin (all decision points)
        "global_margin_std": 0,        # Global decision margin standard deviation
        "low_confidence_ratio": 0,     # Low confidence decision ratio (margin < 0.2)
        "no_candidate_ratio": 0,       # No candidate ratio
        "total_decisions": 0,          # Total number of decisions
        "global_score": 0
    }
    
    if len(tracks) == 0:
        print("\nWarning: No tracks to evaluate!")
        return metrics
    
    # 1. Basic track statistics
    total_cells = 0
    track_lengths = []
    track_avg_margins = []
    track_min_margins = []
    
    for track in tracks:
        length = track.get_layer_count()
        track_lengths.append(length)
        total_cells += length
        
        # Update length distribution statistics
        if length not in metrics["length_distribution"]:
            metrics["length_distribution"][length] = 0
        metrics["length_distribution"][length] += 1
        
        # Get track quality score (average margin)
        quality_score = track.get_quality_score(cells_by_slice)
        track_avg_margins.append(quality_score)
        
        # Get minimum margin (weakest connection in track)
        quality_details = track.get_quality_details(cells_by_slice)
        if quality_details['num_connections'] > 0:
            track_min_margins.append(quality_details['min_margin'])
    
    # Calculate average length
    metrics["average_length"] = np.mean(track_lengths) if track_lengths else 0
    
    # 2. Calculate coverage ratio - proportion of cells used by tracks to total cells
    total_available_cells = sum(len(cells) for cells in cells_by_slice.values())
    metrics["coverage_ratio"] = total_cells / total_available_cells if total_available_cells > 0 else 0
    
    # 3. Calculate GLOBAL decision-level margin metrics (eliminate selection bias)
    # NEW: Statistics based on ALL decision points, not just accepted connections
    if len(all_decision_margins) > 0:
        metrics["global_avg_margin"] = float(np.mean(all_decision_margins))
        metrics["global_margin_std"] = float(np.std(all_decision_margins))
        metrics["total_decisions"] = len(all_decision_margins)
        
        # Low confidence decision ratio (margin < 0.2)
        low_conf_count = sum(1 for m in all_decision_margins if m < 0.2)
        metrics["low_confidence_ratio"] = low_conf_count / len(all_decision_margins)
        
        # No candidate ratio
        if len(all_decision_info) > 0:
            no_cand_count = sum(1 for d in all_decision_info if not d['has_candidate'])
            metrics["no_candidate_ratio"] = no_cand_count / len(all_decision_info)
    else:
        # If no decision data, use default values
        metrics["global_avg_margin"] = 0.0
        metrics["global_margin_std"] = 0.0
        metrics["low_confidence_ratio"] = 1.0  # Assume all are low confidence
        metrics["no_candidate_ratio"] = 1.0
        metrics["total_decisions"] = 0
    
    # 4. Calculate OLD margin-based quality metrics (only for comparison)
    # OLD: Only based on accepted connections (has selection bias)
    metrics["avg_margin"] = np.mean(track_avg_margins) if track_avg_margins else 0
    metrics["margin_std"] = np.std(track_avg_margins) if len(track_avg_margins) > 1 else 0
    metrics["avg_min_margin"] = np.mean(track_min_margins) if track_min_margins else 0
    
    # 5. Calculate global comprehensive score (use global decision perplexity)
    # NEW: Use GLOBAL decision-level margin instead of accepted-only margin
    
    # 5.1 Global confidence score (based on all decision points)
    # Global confidence score (based on ALL decision points, not just accepted)
    # Normalize global_avg_margin to [0, 100]
    # Higher global_avg_margin = lower perplexity = higher confidence
    global_confidence_score = min(100, (metrics["global_avg_margin"] / 0.5) * 100)
    
    # 5.2 Coverage score
    coverage_score = metrics["coverage_ratio"] * 100
    
    # 5.3 Average length score
    max_length_ref = 10.0
    length_score = min(100, (metrics["average_length"] / max_length_ref) * 100)
    
    # New Global Score formula (simplified, eliminate selection bias)
    # NEW Global Score = 0.4 × global_confidence + 0.4 × coverage + 0.2 × length
    # 

    metrics["global_score"] = (0.4 * global_confidence_score + 
                               0.4 * coverage_score + 
                               0.2 * length_score)
    
    # Print scoring results
    print("\n" + "="*80)
    print("Track Global Evaluation Results")
    print("="*80)
    print(f"Total tracks: {metrics['total_tracks']}")
    print(f"Average track length: {metrics['average_length']:.2f} layers")
    print(f"Cell coverage ratio: {metrics['coverage_ratio']*100:.2f}%")
    
    print(f"\nGlobal Decision-Level Statistics:")
    print(f"   Total decisions: {metrics['total_decisions']}")
    print(f"   Global avg margin: {metrics['global_avg_margin']:.6f}")
    print(f"   Global margin std dev: {metrics['global_margin_std']:.6f}")
    print(f"   Low confidence ratio (margin < 0.2): {metrics['low_confidence_ratio']*100:.2f}%")
    print(f"   No candidate ratio: {metrics['no_candidate_ratio']*100:.2f}%")
    print(f"   Global confidence score: {global_confidence_score:.2f}/100")
    
    print(f"\nCoverage Statistics:")
    print(f"   Coverage ratio: {metrics['coverage_ratio']*100:.2f}%")
    print(f"   Coverage score: {coverage_score:.2f}/100")
    
    print(f"\nLength Statistics:")
    print(f"   Average length: {metrics['average_length']:.2f} layers")
    print(f"   Length score: {length_score:.2f}/100")
    
    print(f"\nGlobal Score:")
    print(f"   {metrics['global_score']:.2f} / 100.0")
    print(f"   = 0.4 * Global_Confidence + 0.4 * Coverage + 0.2 * Length")
    print(f"   = 0.4*{global_confidence_score:.2f} + 0.4*{coverage_score:.2f} + 0.2*{length_score:.2f}")
    
    print(f"\nAccepted-Only Margin Statistics (for comparison):")
    print(f"   Avg margin (accepted only): {metrics['avg_margin']:.6f}")
    print(f"   Margin std dev: {metrics['margin_std']:.6f}")
    print(f"   Avg min margin: {metrics['avg_min_margin']:.6f}")
    
    print(f"\nRelative Margin Threshold: {RELATIVE_MARGIN_THRESHOLD:.2f}")
    print("="*80)
    
    # Length distribution information
    print("\nTrack length distribution:")
    for length, count in sorted(metrics["length_distribution"].items()):
        print(f"  {length} layer tracks: {count} tracks ({count/metrics['total_tracks']*100:.1f}%)")
    
    return metrics

def plot_evaluation_radar(metrics, filename="evaluation_radar.png"):
    """Plot evaluation radar chart (Margin-Based Quality Evaluation)"""
    import matplotlib.pyplot as plt
    
    # Prepare radar chart data - now focused on margin-based quality metrics
    categories = ['Avg Margin', 'Margin Consistency', 'Avg Length', 'Coverage']
    values = [
        min(1.0, metrics["avg_margin"] / 0.5),         # Margin normalized to 0-1 (assume max ~0.5)
        max(0, 1.0 - metrics["margin_std"] / 0.2),     # Consistency (lower std = better)
        min(1.0, metrics["average_length"] / 10.0),    # Length normalized to 0-1
        metrics["coverage_ratio"]                       # Already 0-1
    ]
    
    # Create radar chart
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Calculate angles and close the figure
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close
    values += values[:1]  # Close
    
    # Draw lines and fill
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='green')
    ax.fill(angles, values, alpha=0.25, color='green')
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add grid
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_rlabel_position(0)
    
    plt.title(f"Track Quality Evaluation (Margin-Based)\n(Global Score: {metrics['global_score']:.1f}/100 = 70% Margin + 20% Coverage + 10% Length)\nRelative Margin Threshold: α={RELATIVE_MARGIN_THRESHOLD:.2f}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    

def _get_divisor_cache_file(region_name):
    """Get cache file path for specified region (each region has independent file, avoid multi-GPU conflicts)"""
    import re
    # Sanitize region_name to prevent path injection (e.g., "data/Tile048" -> "data_Tile048")
    safe_name = os.path.basename(region_name)  # Remove directory components
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', safe_name)  # Replace invalid chars with underscore
    return os.path.join(_CACHE_DIR, f"divisor_cache_{safe_name}.json")


def _load_divisor_cache(region_name):
    """Load divisor cache from file for specified region"""
    cache_file = _get_divisor_cache_file(region_name)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def _save_divisor_cache(region_name, divisor_data):
    """Save divisor cache to independent file for specified region"""
    cache_file = _get_divisor_cache_file(region_name)
    try:
        with open(cache_file, 'w') as f:
            json.dump(divisor_data, f, indent=2)
        print(f"Cache saved: {cache_file}")
    except Exception as e:
        print(f"Failed to save cache: {e}")
        print(f"Cache file path: {cache_file}")


def calibrate_global_divisors(cells_by_slice, num_samples=50):
    """
    Sample calibration for GLOBAL normalization parameters
    
    Called once at program start, quickly sample and calculate GLOBAL divisor
    Ensure quality scoring uses data-driven normalization parameters
    
    Args:
        cells_by_slice: Cell data dictionary
        num_samples: Sampling number (default 50 pairs of cells)
    
    Returns:
        (dtw_divisor, ot_divisor, centroid_divisor): GLOBAL normalization parameters
    """
    print("\n" + "="*80)
    print("Sample calibration for GLOBAL normalization parameters (for quality scoring)")
    print("="*80)
    
    import random
    
    # 1. Randomly sample cell pairs (adjacent layers)
    all_layers = sorted(cells_by_slice.keys())
    sample_pairs = []
    
    print(f"Sampling {num_samples} cell pairs from {len(all_layers)} layers...")
    
    attempts = 0
    max_attempts = 1000
    
    while len(sample_pairs) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Randomly select two adjacent layers
        if len(all_layers) < 2:
            break
        layer_idx = random.randint(0, len(all_layers)-2)
        layer1 = all_layers[layer_idx]
        layer2 = all_layers[layer_idx + 1]
        
        # Randomly select one cell from each layer
        if len(cells_by_slice[layer1]) > 0 and len(cells_by_slice[layer2]) > 0:
            cell1 = random.choice(cells_by_slice[layer1])
            cell2 = random.choice(cells_by_slice[layer2])
            
            # Spatial distance constraint
            distance = np.linalg.norm(cell1.centroid - cell2.centroid)
            if distance < MAX_XY_DISTANCE:
                sample_pairs.append((cell1, cell2))
    
    if len(sample_pairs) < 2:
        print(f"Warning: Insufficient samples ({len(sample_pairs)} < 2), using default GLOBAL divisor")
        return None, None, CENTROID_NORM_DIVISOR
    
    print(f"Sampled {len(sample_pairs)} cell pairs (attempted {attempts} times)")
    
    # 2. Calculate DTW and OT only, use v8 default for centroid
    dtw_samples = []
    ot_samples = []
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sdtw = SoftDTW(use_cuda=use_cuda, gamma=0.1)
    
    print("Computing sample metrics (DTW and OT)...")
    for cell1, cell2 in tqdm(sample_pairs, desc="Progress"):
        # DTW
        try:
            x = torch.tensor(cell1.contour_points_200, dtype=torch.float32).unsqueeze(0).to(device)
            y = torch.tensor(cell2.contour_points_200, dtype=torch.float32).unsqueeze(0).to(device)
            dtw_val = sdtw(x, y).item()
            dtw_samples.append(dtw_val)
        except:
            pass
        
        # OT (only compute for cells with neighbors)
        try:
            cluster1 = get_cluster_cells(cell1, cells_by_slice)
            cluster2 = get_cluster_cells(cell2, cells_by_slice)
            if len(cluster1) > 1 or len(cluster2) > 1:
                C = compute_ot_cost_matrix(cluster1, cluster2, use_gpu_batch=True)
                areas1 = np.array([c.area for c in cluster1])
                areas2 = np.array([c.area for c in cluster2])
                weights1 = _normalize_area_weights(areas1)
                weights2 = _normalize_area_weights(areas2)
                ot_val, T = solve_optimal_transport(C, weights1, weights2, use_gpu=True)
                if ot_val != float('inf') and ot_val < 100000:
                    ot_samples.append(ot_val)
        except:
            pass
    
    # 3. Calculate divisor (75th percentile -> 7.5)
    dtw_divisor = None
    ot_divisor = None
    centroid_divisor = CENTROID_NORM_DIVISOR  # Not sampled; keep module default (v8: 70.0)
    
    if len(dtw_samples) > 2:
        dtw_divisor = np.percentile(dtw_samples, 75) * 10.0 / 7.5
    
    if len(ot_samples) > 2:
        ot_divisor = np.percentile(ot_samples, 75) * 10.0 / 7.5
    
    # 4. Output statistics
    print("\nSampling statistics:")
    if dtw_divisor:
        print(f"   DTW:")
        print(f"      Samples: {len(dtw_samples)}")
        print(f"      Median: {np.median(dtw_samples):.0f}")
        print(f"      75th percentile: {np.percentile(dtw_samples, 75):.0f}")
        print(f"      GLOBAL_DIVISOR = {dtw_divisor:.0f}")
    else:
        print(f"   DTW: Insufficient samples ({len(dtw_samples)}), using default")
    
    if ot_divisor:
        print(f"   OT:")
        print(f"      Samples: {len(ot_samples)}")
        print(f"      Median: {np.median(ot_samples):.0f}")
        print(f"      75th percentile: {np.percentile(ot_samples, 75):.0f}")
        print(f"      GLOBAL_DIVISOR = {ot_divisor:.0f}")
    else:
        print(f"   OT: Insufficient samples ({len(ot_samples)}), using default")
    
    print(f"   Centroid: divisor fixed at {CENTROID_NORM_DIVISOR:.1f} (not calibrated)")
    
    print("="*80 + "\n")
    
    return dtw_divisor, ot_divisor, centroid_divisor


def process_single_cell_region(cell_dir, gpu_available):
    """Process a single cell region and generate outputs
    
    Args:
        cell_dir: Cell directory name (e.g., 'cell-001')
        gpu_available: Whether GPU is available
    """
    print("\n" + "="*80)
    print(f"Processing {cell_dir}")
    print("="*80)
    
    region_start_time = time.time()
    
    print("\nPhase 1: Loading cell data")
    print("-------------------")
    # Load cell data from specific cell directory
    cells_by_slice = load_cells_from_csv_files(cell_dir)
    total_cells = sum(len(cells) for cells in cells_by_slice.values())
    print(f"Loaded a total of {total_cells} cells distributed across {len(cells_by_slice)} slices")
    
    # Phase 1.5: Calibrate GLOBAL normalization parameters
    global DTW_NORM_DIVISOR, OT_NORM_DIVISOR, CENTROID_NORM_DIVISOR
    
    cached_data = _load_divisor_cache(cell_dir)
    
    if cached_data is not None:
        dtw_div, ot_div, cent_div = cached_data.get('dtw'), cached_data.get('ot'), cached_data.get('centroid')
        cache_file = _get_divisor_cache_file(cell_dir)
        print(f"Using cached normalization parameters (region: {cell_dir}, source: {cache_file})")
    else:
        print(f"First time processing {cell_dir}, calibrating normalization parameters...")
        dtw_div, ot_div, cent_div = calibrate_global_divisors(cells_by_slice, num_samples=50)
        divisor_data = {
            'dtw': dtw_div,
            'ot': ot_div,
            'centroid': cent_div
        }
        _save_divisor_cache(cell_dir, divisor_data)
    
    if dtw_div is not None:
        DTW_NORM_DIVISOR = dtw_div
        print(f"Updated DTW_NORM_DIVISOR = {DTW_NORM_DIVISOR:.0f} (data-driven)")
    else:
        print(f"Using default DTW_NORM_DIVISOR = {DTW_NORM_DIVISOR:.0f}")
    
    if ot_div is not None:
        OT_NORM_DIVISOR = ot_div
        print(f"Updated OT_NORM_DIVISOR = {OT_NORM_DIVISOR:.0f} (data-driven)")
    else:
        print(f"Using default OT_NORM_DIVISOR = {OT_NORM_DIVISOR:.0f}")
    
    if cent_div is not None:
        CENTROID_NORM_DIVISOR = cent_div
        print(f"Updated CENTROID_NORM_DIVISOR = {CENTROID_NORM_DIVISOR:.1f} (data-driven)")
    else:
        print(f"Using default CENTROID_NORM_DIVISOR = {CENTROID_NORM_DIVISOR:.1f}")
    
    print("\nPhase 2: Building cell tracks")
    print("-------------------")
    # Build high-quality tracks
    print("Starting to build cell tracks, this process may take a long time...")
    tracks = find_high_quality_tracks_layerwise(cells_by_slice, output_prefix=f"{cell_dir}_")
    print(f"Track construction completed, generated {len(tracks)} high-quality tracks")
    
    # Display GPU memory usage
    if gpu_available:
        # Clean GPU memory
        torch.cuda.empty_cache()
    
    # Display track information
    print("\nTrack length distribution:")
    tracks_by_quality = {}
    for track in tracks:
        length = track.get_layer_count()
        if length not in tracks_by_quality:
            tracks_by_quality[length] = 0
        tracks_by_quality[length] += 1
    
    for length, count in sorted(tracks_by_quality.items()):
        print(f"  {length} layer tracks: {count} tracks")
    
    plot_track_length_distribution(tracks, filename=f"{cell_dir}_track_lengths.png")
    
    # Generate summary CSV file (MARGIN-BASED QUALITY METRICS)
    print("\nGenerating summary CSV file for all tracks")
    csv_filename = f"{cell_dir}_all_tracks_summary.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "track_id", "start_slice", "end_slice", "layer_count", 
            "avg_margin", "min_margin", "max_margin", "std_margin", "direction", 
            "cells_area_mean", "cells_area_std", 
            "cell_ids"
        ])
        
        for i, track in enumerate(tracks):
            slices = sorted(track.cells.keys())
            if len(slices) >= 2:
                direction = "ascending" if slices[-1] > slices[0] else "descending"
            else:
                direction = "unknown"
            
            # Get margin-based quality metrics
            quality_details = track.get_quality_details(cells_by_slice)
            
            # Calculate area statistics
            areas = [track.cells[s].area for s in slices]
            area_mean = np.mean(areas) if areas else 0
            area_std = np.std(areas) if len(areas) > 1 else 0
                
            cell_ids = [str(track.cells[s].cell_id) for s in sorted(track.cells.keys())]
            
            writer.writerow([
                i + 1, 
                track.start_slice, 
                track.end_slice, 
                track.get_layer_count(),
                f"{quality_details['avg_margin']:.6f}",
                f"{quality_details['min_margin']:.6f}",
                f"{quality_details['max_margin']:.6f}",
                f"{quality_details['std_margin']:.6f}",
                direction,
                f"{area_mean:.2f}",
                f"{area_std:.2f}",
                '|'.join(cell_ids)
            ])
    print(f"Saved all track summary information as {csv_filename}")
    
    print("\nPerforming global track scoring...")
    evaluation_metrics = evaluate_all_tracks(tracks, cells_by_slice)
    
    # Save scoring results for subsequent parameter comparison
    # Include all optimized parameters in filename (SIMPLIFIED: 3 features + relative margin)
    params_str = (f"DTW{DTW_WEIGHT:.4f}_OT{OT_WEIGHT:.4f}_"
                  f"CENT{CENTROID_WEIGHT:.4f}_"
                  f"DTWN{DTW_NORM_DIVISOR:.0f}_OTN{OT_NORM_DIVISOR:.0f}_"
                  f"CENTN{CENTROID_NORM_DIVISOR:.0f}_"
                  f"RM{RELATIVE_MARGIN_THRESHOLD:.2f}")
    eval_filename = f"{cell_dir}_evaluation_{params_str}.json"
    with open(eval_filename, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    print(f"Saved scoring results as {eval_filename}")
    
    # Generate scoring radar chart
    print("Generating scoring radar chart...")
    # Use simplified params_str for radar chart filename (avoid too long filename)
    radar_params = f"DTW{DTW_WEIGHT:.2f}_OT{OT_WEIGHT:.2f}_CENT{CENTROID_WEIGHT:.2f}_RM{RELATIVE_MARGIN_THRESHOLD:.2f}"
    radar_filename = f"{cell_dir}_evaluation_radar_{radar_params}.png"
    plot_evaluation_radar(evaluation_metrics, filename=radar_filename)
    print(f"Saved scoring radar chart as {radar_filename}")
    
    elapsed_time = time.time() - region_start_time
    print(f"\n{cell_dir} reconstruction completed, time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return tracks


def main(target_cells=None):
    """Main function for cell 3D reconstruction
    
    Args:
        target_cells: List of specific cell directories to process (e.g., ['Tile048', 'Tile049'])
                      If None, will process all Tile* directories
    """
    print("Starting cell 3D reconstruction process...")
    print("Processing multiple cell regions separately...")
    overall_start_time = time.time()
    
    # Detect GPU availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"GPU found: {torch.cuda.get_device_name(0)}")
        print(f"Will use GPU acceleration for DTW computation")
        
        # Clean GPU memory
        torch.cuda.empty_cache()
    else:
        print("No GPU detected! This program requires GPU computation and cannot continue.")
        print("Please run this program in an environment with NVIDIA GPU and ensure CUDA is properly installed.")
        return  # Terminate program execution
    
    # Set number of processes
    global NUM_PROCESSES
    if NUM_PROCESSES is None:
        NUM_PROCESSES = mp.cpu_count()
    print(f"Using {NUM_PROCESSES} CPU cores for parallel computation")
    
    # Get all cell directories (支持 cell-* 和 Tile* 两种格式)
    if target_cells is not None:
        # 如果指定了目标目录，直接使用（验证存在性）
        cell_dirs = []
        for target in target_cells:
            if os.path.exists(target) and os.path.isdir(target):
                cell_dirs.append(target)
            else:
                print(f"Warning: Directory {target} not found, skipping")
        
        if not cell_dirs:
            print(f"Error: None of the specified target directories exist!")
            print(f"Specified: {target_cells}")
            return
        
        print(f"\nProcessing specified directories: {cell_dirs}")
    else:
        # 自动检测：优先 cell-*，其次 Tile*
        all_cell_dirs = sorted(glob.glob("cell-*"))
        if not all_cell_dirs:
            all_cell_dirs = sorted(glob.glob("Tile*"))
        
        if not all_cell_dirs:
            print("Error: No cell-* or Tile* directories found!")
            return
        
        cell_dirs = all_cell_dirs
        print(f"\nFound {len(cell_dirs)} directories to process: {cell_dirs}")
    
    # Process each cell region separately
    all_region_tracks = {}
    for cell_dir in cell_dirs:
        try:
            tracks = process_single_cell_region(cell_dir, gpu_available)
            all_region_tracks[cell_dir] = tracks
        except Exception as e:
            print(f"\nError processing {cell_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("ALL REGIONS PROCESSING COMPLETED")
    print("="*80)
    for cell_dir, tracks in all_region_tracks.items():
        print(f"{cell_dir}: {len(tracks)} tracks generated")
    
    overall_elapsed_time = time.time() - overall_start_time
    print(f"\nTotal processing time: {overall_elapsed_time:.2f} seconds ({overall_elapsed_time/60:.2f} minutes)")
    print("\nProgram execution completed")


if __name__ == "__main__":
    # Command line arguments (argparse already imported at top for early GPU selection)
    parser = argparse.ArgumentParser(description='Cell 3D Reconstruction')
    parser.add_argument('--target-cells', type=str, nargs='+', default=None,
                       help='Target cell directories to process (e.g., Tile048 Tile049 Tile052)')
    parser.add_argument('--gpu-id', type=int, default=None,
                       help='Specific GPU ID to use (0, 1, etc.)')
    args = parser.parse_args()
    
    # Note: CUDA_VISIBLE_DEVICES is already set at file top (before torch import)
    # to ensure proper GPU initialization
    
    # Use command line argument or process all regions by default
    target_cells = args.target_cells
    if target_cells is None:
        # Auto-detect Tile* directories
        detected = glob.glob("Tile*")
        detected = sorted([d for d in detected if os.path.isdir(d)])
        if detected:
            target_cells = detected
            print(f"Auto-detected target cells: {target_cells}")
    
    main(target_cells=target_cells)
    
    
    
    
    
