#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark spatial alignment methods (PASTE2, SLAT, CAST, Spateo, 3dOT)
on cardiomyocyte tiles.

For each method:
  1. Build AnnData per slice (spatial coords + expression)
  2. Run pairwise alignment between adjacent slices
  3. Extract cell-to-cell matching from alignment output
  4. Build tracks by chaining matches across slices
  5. Output tracks as CSV (track_id, cell_ids)

Usage:
  python run_benchmark_methods.py --methods PASTE2 SLAT CAST Spateo 3dOT
  python run_benchmark_methods.py --tile-prefixes 046 034 044 076
  python run_benchmark_methods.py --device cpu

Output (per tile):
  benchmark_results/Tile<NNN>/<METHOD>/tracks.csv
  benchmark_results/Tile<NNN>/<METHOD>/pairwise_matches.csv
  benchmark_results/Tile<NNN>/comparison_summary.csv
"""

import os
import sys
import re
import time
import types
import argparse
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

_BENCH_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_ROOT = os.path.join(_BENCH_ROOT, "data")
_DEFAULT_OUTDIR = os.path.join(_BENCH_ROOT, "benchmark_results")
_OUTDIR_FOR_CAST = _DEFAULT_OUTDIR

# --- 1. DATA PREPARATION: Tile + expression dir -> per-slice AnnData ---

def _one_slice_anndata(sn, contour_sub, expr_df, common_labels, normalize_xy):
    """Build single-slice AnnData from aligned contour/expr subsets."""
    cell_ids = contour_sub["cell_id"].values.astype(str)
    labels = contour_sub["join_key"].values
    cx = contour_sub["centroid_x"].values.astype(np.float32)
    cy = contour_sub["centroid_y"].values.astype(np.float32)
    if normalize_xy is not None:
        gcx_min, gcx_max, gcy_min, gcy_max = normalize_xy
        cx = (cx - gcx_min) / (gcx_max - gcx_min + 1e-6)
        cy = (cy - gcy_min) / (gcy_max - gcy_min + 1e-6)

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    expr_sub = expr_df[expr_df["join_key"].isin(common_labels)]
    genes = sorted(expr_sub["feature_index"].unique())
    gene_to_col = {g: j for j, g in enumerate(genes)}
    n_cells = len(cell_ids)
    n_genes = len(genes)
    rows, cols, vals = [], [], []
    for _, row in expr_sub.iterrows():
        lab = int(row["join_key"])
        if lab in label_to_idx:
            r = label_to_idx[lab]
            c = gene_to_col[int(row["feature_index"])]
            rows.append(r)
            cols.append(c)
            vals.append(float(row["total_exp"]))
    X = csr_matrix((vals, (rows, cols)), shape=(n_cells, n_genes))
    obs = pd.DataFrame({"cell_id": cell_ids, "slice_num": sn}, index=cell_ids)
    var = pd.DataFrame(index=[f"gene_{g}" for g in genes])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = np.column_stack([cx, cy]).astype(np.float64)
    return adata


def build_anndata_slices(contour_dir, expr_dir, tile_prefix="046", normalize_spatial=False):
    """Build one AnnData per slice with obsm['spatial'] and X (expression).

    Filenames: {tile_prefix}_S{sn}_features.csv, expression-T{tile_prefix}-S{sn}.csv
    If normalize_spatial, apply global min-max across all slices (same as standalone 3dOT loader).
    """
    print("[DATA] Building AnnData objects per slice ...")

    slice_files = sorted([f for f in os.listdir(contour_dir) if f.endswith("_features.csv")])
    slice_nums = []
    for f in slice_files:
        m = re.search(r"_S(\d+)_features", f)
        if m:
            slice_nums.append(int(m.group(1)))
    slice_nums = sorted(set(slice_nums))

    if not normalize_spatial:
        slices = {}
        for sn in slice_nums:
            contour_file = os.path.join(contour_dir, f"{tile_prefix}_S{sn:02d}_features.csv")
            expr_file = os.path.join(expr_dir, f"expression-T{tile_prefix}-S{sn:02d}.csv")
            if not os.path.exists(contour_file) or not os.path.exists(expr_file):
                print(f"  [WARN] Missing contour or expression for S{sn:02d}, skipping")
                continue
            contour_df = pd.read_csv(contour_file)
            expr_df = pd.read_csv(expr_file, sep="\t")
            contour_df["join_key"] = contour_df["cell_label"].astype(int)
            expr_df["join_key"] = expr_df["cell_label"].astype(int)
            common_labels = set(contour_df["join_key"]) & set(expr_df["join_key"])
            contour_sub = contour_df[contour_df["join_key"].isin(common_labels)].copy()
            contour_sub = contour_sub.drop_duplicates(subset="join_key").sort_values("join_key")
            adata = _one_slice_anndata(sn, contour_sub, expr_df, common_labels, None)
            slices[sn] = adata
            print(f"  S{sn:02d}: {adata.n_obs} cells × {adata.n_vars} genes")
        print(f"[DATA] Built {len(slices)} slices\n")
        return slices

    # Two-pass global spatial normalization
    global_cx_min, global_cx_max = np.inf, -np.inf
    global_cy_min, global_cy_max = np.inf, -np.inf
    contour_data = {}
    for sn in slice_nums:
        contour_file = os.path.join(contour_dir, f"{tile_prefix}_S{sn:02d}_features.csv")
        expr_file = os.path.join(expr_dir, f"expression-T{tile_prefix}-S{sn:02d}.csv")
        if not os.path.exists(contour_file) or not os.path.exists(expr_file):
            continue
        contour_df = pd.read_csv(contour_file)
        expr_df = pd.read_csv(expr_file, sep="\t")
        contour_df["join_key"] = contour_df["cell_label"].astype(int)
        expr_df["join_key"] = expr_df["cell_label"].astype(int)
        common_labels = set(contour_df["join_key"]) & set(expr_df["join_key"])
        contour_sub = contour_df[contour_df["join_key"].isin(common_labels)].copy()
        contour_sub = contour_sub.drop_duplicates(subset="join_key").sort_values("join_key")
        cx = contour_sub["centroid_x"].values.astype(np.float32)
        cy = contour_sub["centroid_y"].values.astype(np.float32)
        global_cx_min = min(global_cx_min, cx.min())
        global_cx_max = max(global_cx_max, cx.max())
        global_cy_min = min(global_cy_min, cy.min())
        global_cy_max = max(global_cy_max, cy.max())
        contour_data[sn] = (contour_sub, expr_df, common_labels)

    norm_bounds = (global_cx_min, global_cx_max, global_cy_min, global_cy_max)
    slices = {}
    for sn in sorted(contour_data.keys()):
        contour_sub, expr_df, common_labels = contour_data[sn]
        adata = _one_slice_anndata(sn, contour_sub, expr_df, common_labels, norm_bounds)
        slices[sn] = adata
        print(f"  S{sn:02d}: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"[DATA] Built {len(slices)} slices (spatial normalized)\n")
    return slices


# --- 2. MATCHING: extract cell-to-cell assignment from alignment output ---

def match_by_ot_matrix(pi, cell_ids_src, cell_ids_tgt):
    """Extract 1-to-1 matching from an OT transport matrix π.
    For each source cell, pick the target with highest transport mass."""
    pi = np.array(pi)
    matches = {}
    assigned_tgt = set()
    src_order = np.argsort(-pi.max(axis=1))
    for i in src_order:
        best_j = np.argmax(pi[i])
        if cell_ids_tgt[best_j] not in assigned_tgt:
            matches[cell_ids_src[i]] = cell_ids_tgt[best_j]
            assigned_tgt.add(cell_ids_tgt[best_j])
    return matches


def match_by_nearest_neighbor(coords_src, coords_tgt, cell_ids_src, cell_ids_tgt):
    """After spatial alignment, match each source cell to its nearest target cell."""
    tree = cKDTree(coords_tgt)
    dists, idxs = tree.query(coords_src, k=1)
    matches = {}
    assigned_tgt = set()
    order = np.argsort(dists)
    for i in order:
        tgt_id = cell_ids_tgt[idxs[i]]
        if tgt_id not in assigned_tgt:
            matches[cell_ids_src[i]] = tgt_id
            assigned_tgt.add(tgt_id)
    return matches


# --- 3. TRACK BUILDING: chain pairwise matches into multi-layer tracks ---

def build_tracks(pairwise_matches, slice_nums):
    """Chain pairwise matches (s_i -> s_{i+1}) into tracks across all slices.

    Args:
        pairwise_matches: dict of (s_from, s_to) -> {cell_id_from: cell_id_to}
        slice_nums: sorted list of slice numbers

    Returns:
        list of tracks, each track = [(slice_num, cell_id), ...]
    """
    all_tracks = []
    used = set()

    for start_slice in slice_nums:
        adata_key = start_slice
        if (start_slice, start_slice + 1) not in pairwise_matches:
            if start_slice == slice_nums[0]:
                pass
            else:
                continue

        first_pair = None
        for sp in sorted(pairwise_matches.keys()):
            if sp[0] == start_slice:
                first_pair = sp
                break
        if first_pair is None:
            continue

        match_dict = pairwise_matches[first_pair]
        for start_cell in match_dict:
            if (start_slice, start_cell) in used:
                continue

            track = [(start_slice, start_cell)]
            cur_cell = start_cell
            cur_slice = start_slice

            while True:
                nxt_slice = cur_slice + 1
                pair_key = (cur_slice, nxt_slice)
                if pair_key not in pairwise_matches:
                    break
                m = pairwise_matches[pair_key]
                if cur_cell not in m:
                    break
                nxt_cell = m[cur_cell]
                if (nxt_slice, nxt_cell) in used:
                    break
                track.append((nxt_slice, nxt_cell))
                cur_cell = nxt_cell
                cur_slice = nxt_slice

            if len(track) >= 3:
                all_tracks.append(track)
                for s, c in track:
                    used.add((s, c))

    return all_tracks


# --- 4. METHOD ADAPTERS ---

def run_PASTE2(slices_dict, device):
    """PASTE2: partial OT alignment."""
    from paste2 import PASTE2, projection

    slice_nums = sorted(slices_dict.keys())
    pairwise_matches = {}

    for idx in range(len(slice_nums) - 1):
        s1, s2 = slice_nums[idx], slice_nums[idx + 1]
        a1 = slices_dict[s1].copy()
        a2 = slices_dict[s2].copy()
        print(f"    PASTE2: aligning S{s1:02d} -> S{s2:02d} ...")

        pi = PASTE2.partial_pairwise_align(a1, a2, s=0.8, alpha=0.1)

        ids_src = a1.obs["cell_id"].values.astype(str)
        ids_tgt = a2.obs["cell_id"].values.astype(str)
        pairwise_matches[(s1, s2)] = match_by_ot_matrix(pi, ids_src, ids_tgt)

    return pairwise_matches


def run_SLAT(slices_dict, device):
    """SLAT: graph-based matching (directly outputs cell matches)."""
    from scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT as _run_SLAT, spatial_match

    slice_nums = sorted(slices_dict.keys())
    pairwise_matches = {}

    for idx in range(len(slice_nums) - 1):
        s1, s2 = slice_nums[idx], slice_nums[idx + 1]
        a1 = slices_dict[s1].copy()
        a2 = slices_dict[s2].copy()
        print(f"    SLAT: matching S{s1:02d} -> S{s2:02d} ...")

        Cal_Spatial_Net(a1, k_cutoff=10, model='KNN')
        Cal_Spatial_Net(a2, k_cutoff=10, model='KNN')
        edges, features = load_anndatas([a1, a2], feature='DPCA', check_order=False)
        embd0, embd1, slat_time = _run_SLAT(features, edges)
        best, index, distance = spatial_match(features, adatas=[a1, a2], reorder=False)

        ids_src = a2.obs["cell_id"].values.astype(str)
        ids_tgt = a1.obs["cell_id"].values.astype(str)

        matches = {}
        assigned = set()
        for i in range(len(best)):
            src_id = ids_src[i]
            tgt_id = ids_tgt[best[i]]
            if tgt_id not in assigned:
                matches[src_id] = tgt_id
                assigned.add(tgt_id)

        # SLAT matches moving(s2) -> reference(s1), we need s1 -> s2
        reverse = {v: k for k, v in matches.items()}
        pairwise_matches[(s1, s2)] = reverse

    return pairwise_matches


def run_CAST(slices_dict, device):
    """CAST: graph embedding + affine/FFD registration, then NN matching."""
    import importlib
    import CAST as _cast_pkg
    # Ensure we can access submodules (handle editable-install path issues)
    cast_dir = os.path.dirname(_cast_pkg.__file__)
    if cast_dir not in sys.path:
        sys.path.insert(0, os.path.dirname(cast_dir))
    importlib.reload(_cast_pkg)
    from CAST.utils import extract_coords_exp
    from CAST import CAST_MARK, CAST_STACK
    from CAST.CAST_Stack import reg_params
    import torch

    work_dir = os.path.join(_OUTDIR_FOR_CAST, "CAST_work")
    os.makedirs(work_dir, exist_ok=True)
    Device = 0 if device == "cuda" else -1

    slice_nums = sorted(slices_dict.keys())
    aligned_slices = {sn: slices_dict[sn].copy() for sn in slice_nums}
    pairwise_matches = {}

    for idx in range(len(slice_nums) - 1):
        s1, s2 = slice_nums[idx], slice_nums[idx + 1]
        a1 = aligned_slices[s2].copy()
        a2 = aligned_slices[s1].copy()
        print(f"    CAST: aligning S{s2:02d} -> S{s1:02d} ...")

        a1.obs["x"] = a1.obsm["spatial"][:, 0]
        a1.obs["y"] = a1.obsm["spatial"][:, 1]
        a2.obs["x"] = a2.obsm["spatial"][:, 0]
        a2.obs["y"] = a2.obsm["spatial"][:, 1]

        sample_list = ["query", "ref"]
        sdata = a1.concatenate(a2)
        sdata.obs.replace({"batch": {"0": sample_list[0], "1": sample_list[1]}}, inplace=True)
        sdata.layers["norm1e4"] = sc.pp.normalize_total(
            sdata, target_sum=1e4, inplace=False)["X"].toarray()

        coords_raw, exps = extract_coords_exp(
            sdata, batch_key="batch", cols=["x", "y"],
            count_layer=".X", data_format="norm1e4")

        if device == "cuda":
            embed_dict = CAST_MARK(coords_raw, exps, work_dir, graph_strategy="delaunay")
        else:
            embed_dict = CAST_MARK(coords_raw, exps, work_dir, gpu_t=-1, graph_strategy="delaunay")

        params_dist = reg_params(
            dataname="query", gpu=Device,
            iterations=150, dist_penalty1=0, bleeding=500,
            d_list=[3, 2, 1, 1/2, 1/3], attention_params=[None, 3, 1, 0],
            dist_penalty2=[0], alpha_basis_bs=[0], meshsize=[8],
            iterations_bs=[1], attention_params_bs=[[None, 3, 1, 0]],
            mesh_weight=[None])
        params_dist.alpha_basis = torch.Tensor(
            [1/1000, 1/1000, 1/50, 5, 5]).reshape(5, 1).to(params_dist.device)

        coord_final = CAST_STACK(
            coords_raw, embed_dict, work_dir, sample_list, params_dist, sub_node_idxs=None)

        query_coords = coord_final[sample_list[0]]
        if hasattr(query_coords, "numpy"):
            query_coords = query_coords.numpy()
        ref_coords = coord_final[sample_list[1]]
        if hasattr(ref_coords, "numpy"):
            ref_coords = ref_coords.numpy()

        aligned_slices[s2].obsm["spatial"] = query_coords
        aligned_slices[s1].obsm["spatial"] = ref_coords

        ids_ref = aligned_slices[s1].obs["cell_id"].values.astype(str)
        ids_query = aligned_slices[s2].obs["cell_id"].values.astype(str)
        pairwise_matches[(s1, s2)] = match_by_nearest_neighbor(
            ref_coords, query_coords, ids_ref, ids_query)

    return pairwise_matches


def run_Spateo(slices_dict, device):
    """Spateo: morpho_align (rigid + nonrigid), then NN matching."""
    import spateo as st

    slice_nums = sorted(slices_dict.keys())
    aligned_slices = {sn: slices_dict[sn].copy() for sn in slice_nums}
    pairwise_matches = {}

    for idx in range(len(slice_nums) - 1):
        s1, s2 = slice_nums[idx], slice_nums[idx + 1]
        a1 = aligned_slices[s1].copy()
        a2 = aligned_slices[s2].copy()
        print(f"    Spateo: aligning S{s1:02d} <-> S{s2:02d} ...")

        a1.layers["counts"] = a1.X.copy()
        sc.pp.normalize_total(a1)
        sc.pp.log1p(a1)
        sc.pp.highly_variable_genes(a1, n_top_genes=2000)

        a2.layers["counts"] = a2.X.copy()
        sc.pp.normalize_total(a2)
        sc.pp.log1p(a2)
        sc.pp.highly_variable_genes(a2, n_top_genes=2000)

        aligned_pair, pis = st.align.morpho_align(
            models=[a1, a2], verbose=False,
            spatial_key="spatial", key_added="align_spatial",
            device=device)

        coords_1 = aligned_pair[0].obsm.get("align_spatial_nonrigid",
                    aligned_pair[0].obsm.get("align_spatial_rigid",
                    aligned_pair[0].obsm["spatial"]))
        coords_2 = aligned_pair[1].obsm.get("align_spatial_nonrigid",
                    aligned_pair[1].obsm.get("align_spatial_rigid",
                    aligned_pair[1].obsm["spatial"]))

        ids_src = aligned_slices[s1].obs["cell_id"].values.astype(str)
        ids_tgt = aligned_slices[s2].obs["cell_id"].values.astype(str)

        if pis is not None and len(pis) > 0:
            pi = pis[0]
            if hasattr(pi, "toarray"):
                pi = pi.toarray()
            pairwise_matches[(s1, s2)] = match_by_ot_matrix(
                np.array(pi), ids_src, ids_tgt)
        else:
            pairwise_matches[(s1, s2)] = match_by_nearest_neighbor(
                coords_1, coords_2, ids_src, ids_tgt)

    return pairwise_matches


# --- 4b. 3d-OT (lazy import; spatial normalized copy for alignment only) ---

_3dot_modules = None

DPCA_COMPS = 50
KNN_GRAPH = 10
ENCODER_EPOCHS = 2000
ALIGN_EPOCHS = 1000
ENCODER_LR = 1e-3
ALIGN_LR = 5e-4
ENCODER_PATIENCE = 200
SIM_K = 10
OT_K = 20
RECON_K = 5
LOSS_ARGS = {
    "backward_dist_weight": 1.0,
    "use_smooth_flow": True,
    "smooth_flow_loss_weight": 0.1,
    "nb_neigh_smooth_flow": 5,
    "use_div_flow": True,
    "div_flow_loss_weight": 0.1,
    "div_neighbor": 4,
    "lattice_steps": 10,
}


def _get_3dot_modules():
    """Load 3d-OT once: path, rpy2 mock, optional CPU chamfer patch."""
    global _3dot_modules
    if _3dot_modules is not None:
        return _3dot_modules

    import torch
    import torch.optim as optim

    dot3_dir = os.path.join(_BENCH_ROOT, "3d-OT")
    if dot3_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(dot3_dir))

    if "rpy2" not in sys.modules:
        class _FakeR:
            def __getitem__(self, key):
                return lambda *a, **kw: [""]

            def __getattr__(self, name):
                return lambda *a, **kw: None

        _mod_rpy2 = types.ModuleType("rpy2")
        _mod_robjects = types.ModuleType("rpy2.robjects")
        _mod_robjects.r = _FakeR()
        _mod_pkg = types.ModuleType("rpy2.robjects.packages")
        _mod_pkg.importr = lambda *a, **kw: None
        _mod_np2ri = types.ModuleType("rpy2.robjects.numpy2ri")
        _mod_np2ri.activate = lambda: None
        _mod_np2ri.numpy2rpy = lambda x: x
        sys.modules["rpy2"] = _mod_rpy2
        sys.modules["rpy2.robjects"] = _mod_robjects
        sys.modules["rpy2.robjects.packages"] = _mod_pkg
        sys.modules["rpy2.robjects.numpy2ri"] = _mod_np2ri

    from lib_3d_OT.utils import dpca, prepare_data
    from lib_3d_OT.single_modialty import extractMODEL, UnifiedModel, train
    from lib_3d_OT.ottools import ot as ot_module

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        import lib_3d_OT.ottools.losses as _losses_mod

        def _cpu_safe_chamfer_loss(pc1, pc2, point_weight=None,
                                   backward_dist_weight=1.0, mask=None,
                                   use_chamfer_cuda=True):
            dist1, dist2, idx1, idx2 = _losses_mod.chamfer_dist_2d_pt(
                pc1, pc2, backward_dist_weight)
            dist1_mean = torch.mean(dist1)
            if backward_dist_weight == 0.0 or dist2 is None:
                return dist1_mean
            return dist1_mean + backward_dist_weight * torch.mean(dist2)

        _losses_mod.chamfer_loss = _cpu_safe_chamfer_loss

    _3dot_modules = (dpca, prepare_data, extractMODEL, UnifiedModel, train, ot_module, optim)
    print("[3dOT] lib_3d_OT loaded")
    return _3dot_modules


def copy_slices_normalized_for_3dot(slices_dict):
    """Deep copy per slice; global min-max normalize obsm['spatial'] to [0,1]."""
    slice_nums = sorted(slices_dict.keys())
    xs, ys = [], []
    for sn in slice_nums:
        sp = slices_dict[sn].obsm["spatial"]
        xs.extend(sp[:, 0].ravel().tolist())
        ys.extend(sp[:, 1].ravel().tolist())
    gcx_min, gcx_max = min(xs), max(xs)
    gcy_min, gcy_max = min(ys), max(ys)
    out = {}
    for sn in slice_nums:
        a = slices_dict[sn].copy()
        cx = a.obsm["spatial"][:, 0].astype(np.float64)
        cy = a.obsm["spatial"][:, 1].astype(np.float64)
        cx_n = (cx - gcx_min) / (gcx_max - gcx_min + 1e-6)
        cy_n = (cy - gcy_min) / (gcy_max - gcy_min + 1e-6)
        a.obsm["spatial"] = np.column_stack([cx_n, cy_n])
        out[sn] = a
    return out


def train_encoder_early_stop(graph, input_dim, device, lr, max_epochs, patience, extractMODEL, optim):
    import torch.nn.functional as F

    model = extractMODEL(args=None, input_dim=input_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    best_state, min_loss, no_improve = None, float("inf"), 0
    for _epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        z = model.get_features(graph)
        dec = model.decode(z)
        loss = F.mse_loss(dec, graph.express)
        loss.backward()
        opt.step()
        val = loss.item()
        if val < min_loss:
            min_loss, best_state, no_improve = val, model.state_dict(), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_state, min_loss


def _ensure_3dot_truth_placeholder(adata: ad.AnnData) -> None:
    """lib_3d_OT prepare_data() reads obs['truth']; DLPFC demos use cell-type labels.

    Benchmark tiles have no such column - use a dummy label (unused in Graph forward).
    """
    if "truth" not in adata.obs.columns:
        adata.obs["truth"] = np.zeros(len(adata.obs_names), dtype=np.int64)


def run_3dOT_pair(adata_a, adata_b, device, dpca, prepare_data, extractMODEL, UnifiedModel, train, ot_module, optim):
    adata_a_feat, adata_b_feat = dpca(
        [adata_a.copy(), adata_b.copy()], join="inner", n_comps=DPCA_COMPS
    )
    input_dim = adata_a_feat.obsm["feat"].shape[1]
    adata_a_feat.obsm["spatial"] = adata_a.obsm["spatial"]
    adata_b_feat.obsm["spatial"] = adata_b.obsm["spatial"]
    _ensure_3dot_truth_placeholder(adata_a_feat)
    _ensure_3dot_truth_placeholder(adata_b_feat)

    graph_a = prepare_data(adata_a_feat, location="spatial", nb_neighbors=KNN_GRAPH).to(device)
    graph_b = prepare_data(adata_b_feat, location="spatial", nb_neighbors=KNN_GRAPH).to(device)

    best_a, _ = train_encoder_early_stop(
        graph_a, input_dim, device, ENCODER_LR, ENCODER_EPOCHS, ENCODER_PATIENCE, extractMODEL, optim)
    best_b, _ = train_encoder_early_stop(
        graph_b, input_dim, device, ENCODER_LR, ENCODER_EPOCHS, ENCODER_PATIENCE, extractMODEL, optim)

    unified = UnifiedModel(
        input_dim1=input_dim, input_dim2=input_dim,
        simk=SIM_K, otk=OT_K, reconk=RECON_K,
        best_encoder1=best_a, best_encoder2=best_b,
    ).to(device)

    opt_u = optim.Adam(unified.parameters(), lr=ALIGN_LR)
    sch_u = optim.lr_scheduler.CosineAnnealingLR(opt_u, T_max=ALIGN_EPOCHS, eta_min=1e-5)

    train(unified, [graph_a, graph_b], opt_u, sch_u, device,
          nb_epochs=ALIGN_EPOCHS, args=LOSS_ARGS)

    import torch
    unified.eval()
    with torch.no_grad():
        feat_a, feat_b, g_a, g_b = unified.get_encoder_results(
            unified.encoder1, unified.encoder2, graph_a, graph_b)
        pc_a = unified.min_max_normalize(g_a.pcloud)
        pc_b = unified.min_max_normalize(g_b.pcloud)
        T_tensor, _ = ot_module.sinkhorn(
            feat_a, feat_b, pc_a, pc_b,
            epsilon=torch.exp(unified.log_epsilon),
            gamma=torch.exp(unified.log_gamma),
            max_iter=100, sim_k=unified.simk, dist_k=unified.otk,
        )
    T_np = T_tensor[0].cpu().numpy()
    row_sums = T_np.sum(axis=1, keepdims=True)
    T_norm = T_np / (row_sums + 1e-12)

    def strip_suffix(names):
        return [n.rsplit("-", 1)[0] if n.endswith(("-0", "-1")) else n for n in names]

    ids_a = strip_suffix(adata_a_feat.obs_names.tolist())
    ids_b = strip_suffix(adata_b_feat.obs_names.tolist())
    return T_norm, ids_a, ids_b


def run_3dOT(slices_dict, device):
    import torch

    dpca, prepare_data, extractMODEL, UnifiedModel, train, ot_module, optim = _get_3dot_modules()
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    slice_nums = sorted(slices_dict.keys())
    slices_n = copy_slices_normalized_for_3dot(slices_dict)
    pairwise_matches = {}

    for idx in range(len(slice_nums) - 1):
        s1, s2 = slice_nums[idx], slice_nums[idx + 1]
        a1 = slices_n[s1]
        a2 = slices_n[s2]
        print(f"    3dOT: S{s1:02d} -> S{s2:02d} ...")
        T, ids_a, ids_b = run_3dOT_pair(
            a1, a2, dev, dpca, prepare_data, extractMODEL, UnifiedModel, train, ot_module, optim)
        pairwise_matches[(s1, s2)] = match_by_ot_matrix(T, ids_a, ids_b)

    return pairwise_matches


# --- 5. OUTPUT & COMPARISON ---

def save_results(method_name, pairwise_matches, tracks, slices_dict, outdir):
    """Save pairwise matches and tracks to CSV."""
    method_dir = os.path.join(outdir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    # Pairwise matches
    rows = []
    for (s1, s2), matches in sorted(pairwise_matches.items()):
        for src, tgt in matches.items():
            rows.append({"slice_from": s1, "slice_to": s2,
                         "cell_from": src, "cell_to": tgt})
    pd.DataFrame(rows).to_csv(
        os.path.join(method_dir, "pairwise_matches.csv"), index=False)

    # Tracks
    track_rows = []
    for tid, track in enumerate(tracks):
        cell_ids = [cid for _, cid in track]
        slice_range = f"S{track[0][0]:02d}-S{track[-1][0]:02d}"
        track_rows.append({
            "track_id": tid + 1,
            "length": len(track),
            "start_slice": track[0][0],
            "end_slice": track[-1][0],
            "slice_range": slice_range,
            "cell_ids": "|".join(cell_ids),
        })
    pd.DataFrame(track_rows).to_csv(
        os.path.join(method_dir, "tracks.csv"), index=False)

    print(f"  -> {method_dir}/pairwise_matches.csv ({len(rows)} matches)")
    print(f"  -> {method_dir}/tracks.csv ({len(tracks)} tracks)")


def generate_summary(all_results, slices_dict, outdir, tile_prefix=""):
    """Summary comparison across all methods."""
    total_cells = sum(a.n_obs for a in slices_dict.values())
    rows = []
    for method, info in all_results.items():
        tracks = info["tracks"]
        n_matches = sum(len(m) for m in info["matches"].values())
        n_tracks = len(tracks)
        lengths = [len(t) for t in tracks] if tracks else [0]
        cells_in_tracks = sum(lengths)
        row = {
            "tile_prefix": tile_prefix,
            "method": method,
            "n_pairwise_matches": n_matches,
            "n_tracks": n_tracks,
            "avg_track_length": f"{np.mean(lengths):.2f}" if tracks else "0",
            "max_track_length": max(lengths) if tracks else 0,
            "cells_in_tracks": cells_in_tracks,
            "coverage": f"{cells_in_tracks / total_cells:.3f}" if total_cells else "0",
            "runtime_sec": f"{info['runtime']:.1f}",
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    summary_path = os.path.join(outdir, "comparison_summary.csv")
    df.to_csv(summary_path, index=False)
    print("\nBenchmark summary:")
    print(df.to_string(index=False))
    print(f"\n  -> {summary_path}")


# --- 6. MAIN ---

METHOD_REGISTRY = {
    "PASTE2": run_PASTE2,
    "SLAT":   run_SLAT,
    "CAST":   run_CAST,
    "Spateo": run_Spateo,
    "3dOT":   run_3dOT,
}

ALL_METHODS = list(METHOD_REGISTRY.keys())
DEFAULT_METHODS = [m for m in ALL_METHODS if m != "3dOT"]


def main():
    global _OUTDIR_FOR_CAST

    parser = argparse.ArgumentParser(
        description="Benchmark spatial alignment methods (multi-tile; 3dOT optional)")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=ALL_METHODS, help="Methods to run (3dOT is slow; omitted from default)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument(
        "--tile-prefixes", nargs="+", default=["046"],
        help="Tile IDs, e.g. 046 034 044 076 (dirs: <data-root>/TileNNN and <data-root>/NNN)",
    )
    parser.add_argument(
        "--data-root", type=str, default=_DEFAULT_DATA_ROOT,
        help="Parent folder containing Tile<NNN>/ and <NNN>/ expression folders",
    )
    parser.add_argument(
        "--contour-dir", type=str, default=None,
        help="Override contour folder (single-tile runs only; use with --expr-dir)",
    )
    parser.add_argument(
        "--expr-dir", type=str, default=None,
        help="Override expression folder (single-tile runs only)",
    )
    parser.add_argument("--outdir", type=str, default=_DEFAULT_OUTDIR)
    args = parser.parse_args()

    if (args.contour_dir is None) ^ (args.expr_dir is None):
        parser.error("Provide both --contour-dir and --expr-dir, or neither.")
    if args.contour_dir is not None and len(args.tile_prefixes) != 1:
        parser.error("Custom --contour-dir/--expr-dir requires exactly one --tile-prefixes value.")

    os.makedirs(args.outdir, exist_ok=True)

    print("Benchmark: spatial alignment (multi-tile)")
    print(f"  Tiles:   {args.tile_prefixes}")
    print(f"  Methods: {args.methods}")
    print(f"  Device:  {args.device}")

    for tile_prefix in args.tile_prefixes:
        tile_prefix = str(tile_prefix).strip()
        if args.contour_dir is not None:
            contour_dir = args.contour_dir
            expr_dir = args.expr_dir
        else:
            contour_dir = os.path.join(args.data_root, f"Tile{tile_prefix}")
            expr_dir = os.path.join(args.data_root, tile_prefix)

        dataset_outdir = os.path.join(args.outdir, f"Tile{tile_prefix}")
        os.makedirs(dataset_outdir, exist_ok=True)
        _OUTDIR_FOR_CAST = dataset_outdir

        print(f"\n[Tile {tile_prefix}] contour    = {contour_dir}")
        print(f"          expression = {expr_dir}")
        print(f"          output     = {dataset_outdir}")

        slices_dict = build_anndata_slices(
            contour_dir, expr_dir, tile_prefix=tile_prefix, normalize_spatial=False)
        slice_nums = sorted(slices_dict.keys())
        if len(slice_nums) < 2:
            print(f"  [SKIP] Need >=2 slices for tile {tile_prefix}, found {len(slice_nums)}")
            continue

        all_results = {}
        for method_name in args.methods:
            print(f"\n[Tile {tile_prefix}] running {method_name}")

            func = METHOD_REGISTRY[method_name]
            t0 = time.time()
            try:
                pairwise_matches = func(slices_dict, args.device)
                elapsed = time.time() - t0

                tracks = build_tracks(pairwise_matches, slice_nums)
                save_results(method_name, pairwise_matches, tracks, slices_dict, dataset_outdir)

                all_results[method_name] = {
                    "matches": pairwise_matches,
                    "tracks": tracks,
                    "runtime": elapsed,
                }
                print(f"  Done in {elapsed:.1f}s: {len(tracks)} tracks")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  FAILED after {elapsed:.1f}s: {e}")
                import traceback
                traceback.print_exc()
                all_results[method_name] = {
                    "matches": {},
                    "tracks": [],
                    "runtime": elapsed,
                }

        generate_summary(all_results, slices_dict, dataset_outdir, tile_prefix=tile_prefix)

    print("\nAll done.")


if __name__ == "__main__":
    main()
