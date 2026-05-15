#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract cell contour features from all image regions
Processes all region folders in current directory (014, 015, ..., 086)
Output: feature files for 10 slices per region

"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import cv2
import warnings
import glob
warnings.filterwarnings('ignore')

from scipy.spatial.distance import cdist
from tqdm import tqdm

# output directory
output_dir = './all_features'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# slice mapping
SLICE_MAPPING = {
    "S01": "004X0001",
    "S02": "004X0002",
    "S03": "004X0003",
    "S04": "004X0004",
    "S05": "004X0005",
    "S06": "004X0006",
    "S07": "004X0007",
    "S08": "004X0008",
    "S09": "004X0009",
    "S10": "004X0010"
}

def read_cell_outlines(file_path):
    """Read cell outlines from file. Each line = one cell (x1,y1,x2,y2,...)
    Line number = cell_label. Returns list of (cell_label, coords) tuples"""
    cell_outlines = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split(',')
            if len(parts) >= 2:  # need at least one coordinate pair
                try:
                    # convert coord strings to int array and reshape to (n, 2)
                    coords = np.array([int(x) for x in parts]).reshape(-1, 2)
                    # line number is the cell_label
                    cell_outlines.append((line_num, coords))
                except Exception as e:
                    print(f"    Warning: Failed to parse line {line_num}: {e}")
                    continue
    
    return cell_outlines

def sample_contour_points(points, n_samples=400, start_at_top=True):
    """Sample points uniformly from contour, start at top, go clockwise"""
    # calculate contour center
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    
    # compute angles relative to center
    angles = np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x)
    angles = np.mod(angles, 2 * np.pi)
    angles_clockwise = 2 * np.pi - angles
    
    # sort points by clockwise angle
    sorted_indices = np.argsort(angles_clockwise)
    points_sorted = points[sorted_indices]
    
    # start from 12 o'clock position
    if start_at_top:
        top_angle = 3 * np.pi / 2
        angles_for_comparison = np.mod(angles, 2 * np.pi)
        distances_to_top = np.minimum(
            np.abs(angles_for_comparison - top_angle),
            2 * np.pi - np.abs(angles_for_comparison - top_angle)
        )
        top_index = np.argmin(distances_to_top)
        top_index_in_sorted = np.where(sorted_indices == top_index)[0][0]
        points_sorted = np.roll(points_sorted, -top_index_in_sorted, axis=0)
    
    # compute cumulative distance along contour
    dists = np.zeros(len(points_sorted))
    for i in range(1, len(points_sorted)):
        dx = points_sorted[i, 0] - points_sorted[i-1, 0]
        dy = points_sorted[i, 1] - points_sorted[i-1, 1]
        dists[i] = dists[i-1] + np.sqrt(dx*dx + dy*dy)
    
    total_dist = dists[-1]
    
    if total_dist == 0:
        return np.tile(points_sorted[0], (n_samples, 1))
    
    sample_dists = np.linspace(0, total_dist, n_samples, endpoint=False)
    
    try:
        points_loop = np.vstack([points_sorted, points_sorted[0]])
        dists_loop = np.append(dists, total_dist)
        
        f_x = np.interp(sample_dists, dists_loop, points_loop[:, 0])
        f_y = np.interp(sample_dists, dists_loop, points_loop[:, 1])
        
        sampled_points = np.column_stack([f_x, f_y])
        return sampled_points
    except:
        sampled_indices = np.searchsorted(dists, sample_dists) % len(points_sorted)
        return points_sorted[sampled_indices]

def calculate_cell_metrics(points):
    """Calculate various metrics from cell contour"""
    # centroid
    centroid_x = np.mean(points[:, 0])
    centroid_y = np.mean(points[:, 1])
    
    # area using shoelace formula
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i, 0] * points[j, 1]
        area -= points[j, 0] * points[i, 1]
    area = abs(area) / 2.0
    
    # perimeter
    perimeter = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = points[j, 0] - points[i, 0]
        dy = points[j, 1] - points[i, 1]
        perimeter += np.sqrt(dx*dx + dy*dy)
    
    # ellipse fitting
    if len(points) >= 5:
        try:
            contour = points.astype(np.float32).reshape(-1, 1, 2)
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            major_axis = max(axes)
            minor_axis = min(axes)
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
            
            orientation_angle = angle
            if orientation_angle > 90:
                orientation_angle -= 180
                
            return {
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'area': area,
                'perimeter': perimeter,
                'aspect_ratio': aspect_ratio,
                'major_axis': major_axis,
                'minor_axis': minor_axis,
                'orientation_angle': orientation_angle,
                'circularity': (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            }
        except:
            return {
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'area': area,
                'perimeter': perimeter,
                'aspect_ratio': 1.0,
                'major_axis': 0.0,
                'minor_axis': 0.0,
                'orientation_angle': 0.0,
                'circularity': (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            }
    else:
        return {
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': 1.0,
            'major_axis': 0.0,
            'minor_axis': 0.0,
            'orientation_angle': 0.0,
            'circularity': (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        }

def extract_contour_points(row, num_points=400):
    """Extract contour point coordinates from dataframe row"""
    points = []
    for i in range(num_points):
        x = row[f'x_{i}']
        y = row[f'y_{i}']
        points.append([x, y])
    return np.array(points)

def find_adjacent_cells(df, distance_threshold=5.0, num_points=400, sample_points=100):
    """Find neighboring cells for each cell"""
    adjacent_dict = {}
    cell_ids = df['cell_id'].tolist()
    n_cells = len(df)
    
    print(f"    Extracting contours (sampling {sample_points} from {num_points} points)...")
    contours = []
    sample_indices = np.linspace(0, num_points-1, sample_points, dtype=int)
    
    for idx, row in df.iterrows():
        contour_full = extract_contour_points(row, num_points)
        contour_sampled = contour_full[sample_indices]
        contours.append(contour_sampled)
    
    print(f"    Computing adjacency...")
    for i in tqdm(range(n_cells), desc="    Progress", leave=False):
        adjacent_cells = []
        contour_i = contours[i]
        
        for j in range(n_cells):
            if i == j:
                continue
            
            contour_j = contours[j]
            
            min_distance = float('inf')
            chunk_size = 50
            for start_i in range(0, len(contour_i), chunk_size):
                end_i = min(start_i + chunk_size, len(contour_i))
                chunk_distances = cdist(contour_i[start_i:end_i], contour_j, metric='euclidean')
                chunk_min = np.min(chunk_distances)
                min_distance = min(min_distance, chunk_min)
            
            if min_distance <= distance_threshold:
                adjacent_cells.append(cell_ids[j])
        
        adjacent_dict[cell_ids[i]] = adjacent_cells
    
    return adjacent_dict

def add_adjacent_cells_info(df, distance_threshold=5.0, num_points=400, sample_points=100):
    """Add adjacent cell information to dataframe"""
    adjacent_dict = find_adjacent_cells(df, distance_threshold, num_points, sample_points)
    
    df['adjacent_cells'] = df['cell_id'].apply(
        lambda x: ';'.join(adjacent_dict.get(x, []))
    )
    
    df['num_adjacent_cells'] = df['adjacent_cells'].apply(
        lambda x: len(x.split(';')) if x else 0
    )
    
    return df

def transform_contours_to_centered(df, num_points=400):
    """Transform contour points to centroid-centered coordinate system"""
    contour_x_cols = [f'x_{i}' for i in range(num_points)]
    contour_y_cols = [f'y_{i}' for i in range(num_points)]
    
    print(f"    Centering contour coordinates...")
    
    for idx in tqdm(range(len(df)), desc="    Centering", leave=False):
        centroid_x = df.at[idx, 'centroid_x']
        centroid_y = df.at[idx, 'centroid_y']
        
        for x_col in contour_x_cols:
            if pd.notna(df.at[idx, x_col]):
                df.at[idx, x_col] = df.at[idx, x_col] - centroid_x
        
        for y_col in contour_y_cols:
            if pd.notna(df.at[idx, y_col]):
                df.at[idx, y_col] = df.at[idx, y_col] - centroid_y
    
    return df

def process_and_save_slice(region_id, slice_name, outline_file_path):
    """Process single slice and save features"""
    print(f"    Processing slice: {slice_name}")
    
    # load cell outlines
    try:
        cell_outlines = read_cell_outlines(outline_file_path)
        print(f"      Loaded {len(cell_outlines)} cell outlines")
    except Exception as e:
        print(f"      Failed: {e}")
        return False
    
    if len(cell_outlines) == 0:
        print(f"      Warning: no outlines found")
        return False
    
    # process each cell
    all_cells_data = []
    
    for cell_label, cell_outline in cell_outlines:
        slice_num = int(slice_name[1:])
        # build cell_id using region ID
        cell_id = f"T{region_id}S{slice_num:02d}_{cell_label:04d}"
        
        metrics = calculate_cell_metrics(cell_outline)
        sampled_points = sample_contour_points(cell_outline, n_samples=400, start_at_top=True)
        
        cell_data = {
            'cell_id': cell_id,
            'cell_label': cell_label,
            'region': region_id,
            'slice': slice_name,
            'z': int(slice_name[1:])
        }
        
        cell_data.update(metrics)
        
        for i, point in enumerate(sampled_points):
            cell_data[f'x_{i}'] = point[0]
            cell_data[f'y_{i}'] = point[1]
        
        all_cells_data.append(cell_data)
    
    df = pd.DataFrame(all_cells_data)
    print(f"      Extracted features for {len(df)} cells")
    
    # post-processing
    print(f"      [Step 1] Finding adjacent cells...")
    df = add_adjacent_cells_info(df, distance_threshold=5.0, num_points=400, sample_points=100)
    print(f"        Adjacent cells stats: mean={df['num_adjacent_cells'].mean():.2f}, min={df['num_adjacent_cells'].min()}, max={df['num_adjacent_cells'].max()}")
    
    print(f"      [Step 2] Centering contour coordinates...")
    df = transform_contours_to_centered(df, num_points=400)
    
    # create region subfolder
    region_output_dir = os.path.join(output_dir, f"Tile{region_id}")
    if not os.path.exists(region_output_dir):
        os.makedirs(region_output_dir)
    
    # save to region subfolder
    output_file = os.path.join(region_output_dir, f"{region_id}_{slice_name}_features.csv")
    df.to_csv(output_file, index=False)
    print(f"      ✓ Saved: Tile{region_id}/{region_id}_{slice_name}_features.csv ({len(df)} cells)")
    
    return True

def process_region(region_id, region_dir):
    """Process all slices for a single region"""
    print(f"\n{'='*80}")
    print(f"Processing region: {region_id}")
    print(f"{'='*80}")
    
    txt_outlines_dir = region_dir / 'txt_outlines'
    
    if not txt_outlines_dir.exists():
        print(f"  Warning: txt_outlines directory not found")
        return 0
    
    success_count = 0
    
    # process all slices (S01-S10)
    for slice_name in sorted(SLICE_MAPPING.keys()):
        outline_prefix = SLICE_MAPPING[slice_name]
        outline_file = txt_outlines_dir / f"{outline_prefix}-cell-{region_id}_cp_outlines.txt"
        
        if not outline_file.exists():
            print(f"    Warning: file not found: {outline_file.name}")
            continue
        
        if process_and_save_slice(region_id, slice_name, outline_file):
            success_count += 1
    
    print(f"  Region {region_id} complete: processed {success_count}/{len(SLICE_MAPPING)} slices")
    return success_count

def main():
    """Extract cell features from all regions"""
    
    base_dir = Path('.')
    
    print(f"{'='*80}")
    print(f"Cell Contour Feature Extraction")
    print(f"{'='*80}")
    print(f"Working directory: {base_dir.absolute()}")
    print(f"Output directory: {output_dir}")
    print(f"\nScanning for region folders...")
    
    # auto-discover all region folders
    region_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.isdigit() and len(item.name) == 3:
            txt_outlines = item / 'txt_outlines'
            if txt_outlines.exists():
                region_dirs.append(item)
    
    region_dirs.sort(key=lambda x: x.name)
    
    if not region_dirs:
        print("Error: no region folders found")
        return
    
    print(f"Found {len(region_dirs)} region folders:")
    for region_dir in region_dirs:
        print(f"  - {region_dir.name}")
    
    # process all regions
    total_success = 0
    total_expected = len(region_dirs) * len(SLICE_MAPPING)
    region_summary = []
    
    for region_dir in region_dirs:
        region_id = region_dir.name
        success_count = process_region(region_id, region_dir)
        total_success += success_count
        region_summary.append((region_id, success_count))
    
    # summary
    print(f"\n{'='*80}")
    print(f"✓ Processing complete!")
    print(f"{'='*80}")
    print(f"Total: successfully processed {total_success}/{total_expected} slice files")
    print(f"Output directory: {output_dir}")
    
    print(f"\nProcessing status by region:")
    for region_id, success_count in region_summary:
        status = "✓" if success_count == len(SLICE_MAPPING) else "⚠"
        print(f"  {status} Region {region_id}: {success_count}/{len(SLICE_MAPPING)} slices (saved in Tile{region_id}/)")
    
    # count generated files
    total_files = len(list(Path(output_dir).glob('**/*.csv')))
    print(f"\nTotal files generated: {total_files}")
    print(f"File location: {Path(output_dir).absolute()}")
    print(f"File structure: all_features/TileXXX/XXX_S0X_features.csv")

if __name__ == "__main__":
    main()

