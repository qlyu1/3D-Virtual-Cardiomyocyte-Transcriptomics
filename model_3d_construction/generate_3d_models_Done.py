import pandas as pd
import numpy as np
import re
import os
import time
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, namedtuple

print(f"trimesh version: {trimesh.__version__}")

Z_SCALE = 8
Z_DISTANCE = 5.0
SMOOTH_STEPS = 3
NUM_POINTS = 100

MODULE_COLORS = {
    'Module 1': [0.941, 0.498, 0.306, 1.0],
    'Module 2': [0.976, 0.698, 0.227, 1.0],
    'Module 3': [0.384, 0.745, 0.482, 1.0],
    'Module 4': [0.220, 0.604, 0.851, 1.0],
    'Module 5': [0.557, 0.427, 0.718, 1.0],
    'Module 6': [0.945, 0.329, 0.498, 1.0],
    'Others':  [0.7, 0.7, 0.7, 1.0],
    'default': [0.7, 0.7, 0.7, 1.0]
}

Cell = namedtuple('Cell', ['cell_id', 'contour_points', 'slice_idx', 'area', 'centroid'])

class Track:
    def __init__(self, track_id, module=None):
        self.track_id = track_id
        self.cells = {}
        self.start_slice = None
        self.end_slice = None
        self.module = module
    
    def add_cell(self, cell):
        self.cells[cell.slice_idx] = cell
        if self.start_slice is None or cell.slice_idx < self.start_slice:
            self.start_slice = cell.slice_idx
        if self.end_slice is None or cell.slice_idx > self.end_slice:
            self.end_slice = cell.slice_idx
    
    def get_layer_count(self):
        return len(self.cells)

def extract_slice_number(cell_id):
    match = re.search(r'S(\d+)', cell_id)
    return int(match.group(1)) if match else None

def normalize_cell_id(cell_id):
    match = re.match(r'T(\d+)S(\d+)([-_])(\d+)', cell_id)
    if match:
        return f'T{match.group(1).zfill(3)}S{match.group(2).zfill(2)}_{match.group(4).zfill(4)}'
    return cell_id

def load_cell_contours(contours_file):
    df = pd.read_csv(contours_file)
    cell_contours = {}
    id_column = 'cell_id' if 'cell_id' in df.columns else 'ID'
    
    if 'point_0_x' in df.columns:
        point_count = len([col for col in df.columns if col.startswith('point_') and col.endswith('_x')])
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading contours"):
            cell_id = row[id_column]
            contour_points = []
            
            for i in range(point_count):
                x, y = row.get(f'point_{i}_x'), row.get(f'point_{i}_y')
                if pd.notna(x) and pd.notna(y):
                    contour_points.append((x, y))
            
            if len(contour_points) >= 3:
                centroid = (row['centroid_x'], row['centroid_y']) if 'centroid_x' in row and pd.notna(row['centroid_x']) else tuple(np.mean(contour_points, axis=0))
                normalized_id = normalize_cell_id(cell_id)
                cell_contours[normalized_id] = Cell(cell_id, np.array(contour_points), extract_slice_number(cell_id), row.get('area', 0), centroid)
    
    return cell_contours

def load_tracks(tracks_file, cell_contours):
    df = pd.read_csv(tracks_file)
    has_module = 'Module' in df.columns
    tracks = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading tracks"):
        track = Track(row['track_id'], row.get('Module') if has_module else None)
        cell_ids = row['cell_ids'].split('|')
        
        for cell_id in cell_ids:
            normalized_id = normalize_cell_id(cell_id.strip())
            if normalized_id in cell_contours:
                track.add_cell(cell_contours[normalized_id])
        
        if track.get_layer_count() >= 2:
            tracks.append(track)
    
    return tracks

def resample_contour(contour, num_points):
    if len(contour) < 3:
        center = np.mean(contour, axis=0) if len(contour) > 0 else np.array([0.0, 0.0])
        radii = [np.linalg.norm(p - center) for p in contour]
        radius = np.mean(radii) if radii else 10.0
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        return np.column_stack((center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)))
    
    if not np.allclose(contour[0], contour[-1]):
        contour = contour.copy()
        contour[-1] = contour[0]
    
    distances = np.zeros(len(contour))
    for i in range(1, len(contour)):
        distances[i] = distances[i-1] + np.linalg.norm(contour[i] - contour[i-1])
    
    total_length = distances[-1]
    if total_length == 0:
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        return np.column_stack((contour[0][0] + 5 * np.cos(theta), contour[0][1] + 5 * np.sin(theta)))
    
    resampled = np.zeros((num_points, 2))
    for i in range(num_points):
        target_distance = i * total_length / num_points
        idx = np.searchsorted(distances, target_distance)
        idx = min(idx, len(contour) - 1)
        
        if idx > 0:
            prev_idx = idx - 1
            segment_length = distances[idx] - distances[prev_idx]
            if segment_length > 0:
                t = (target_distance - distances[prev_idx]) / segment_length
                resampled[i] = contour[prev_idx] * (1 - t) + contour[idx] * t
            else:
                resampled[i] = contour[idx]
        else:
            resampled[i] = contour[idx]
    
    resampled[-1] = resampled[0]
    return resampled

def create_smooth_cell_tracks(tracks, num_points=NUM_POINTS, z_scale=Z_SCALE, z_distance=Z_DISTANCE, smooth_steps=SMOOTH_STEPS):
    meshes_by_material = defaultdict(list)
    
    for track in tqdm(tracks, desc="Creating 3D models"):
        slices = sorted(track.cells.keys())
        if len(slices) < 2:
            continue
        
        module_name = track.module if hasattr(track, 'module') and track.module else 'default'
        material_name = str(module_name)
        
        all_layer_points = []
        for slice_idx in slices:
            points = resample_contour(track.cells[slice_idx].contour_points, num_points)
            points[-1] = points[0]
            all_layer_points.append((slice_idx, points))
        
        vertices, triangles = [], []
        current_vertex_count = 0
        
        for i in range(len(all_layer_points) - 1):
            current_slice_idx, current_points = all_layer_points[i]
            next_slice_idx, next_points = all_layer_points[i + 1]
            
            if len(current_points) != len(next_points):
                continue
            
            intermediate_layers = smooth_steps + 2
            current_z = current_slice_idx * z_distance * z_scale
            next_z = next_slice_idx * z_distance * z_scale
            z_step = (next_z - current_z) / (intermediate_layers - 1)
            
            for j in range(intermediate_layers):
                t = j / (intermediate_layers - 1)
                smooth_t = t * t * (3 - 2 * t)
                layer_points = current_points * (1 - smooth_t) + next_points * smooth_t
                
                if 0.2 < t < 0.8:
                    centroid = np.mean(layer_points, axis=0)
                    layer_points += (centroid - layer_points) * 0.15 * (0.5 - abs(t - 0.5)) * 2
                
                z = current_z + z_step * j
                for pt in layer_points:
                    vertices.append([pt[0], pt[1], z])
                
                if j > 0:
                    prev_start = current_vertex_count + (j-1) * len(layer_points)
                    curr_start = current_vertex_count + j * len(layer_points)
                    
                    for k in range(len(layer_points) - 1):
                        triangles.extend([
                            [prev_start + k, curr_start + k, prev_start + k + 1],
                            [curr_start + k, curr_start + k + 1, prev_start + k + 1]
                        ])
                    
                    triangles.extend([
                        [prev_start + len(layer_points) - 1, curr_start + len(layer_points) - 1, prev_start],
                        [curr_start + len(layer_points) - 1, curr_start, prev_start]
                    ])
            
            current_vertex_count += intermediate_layers * len(current_points)
        
        first_slice_idx, first_points = all_layer_points[0]
        first_z = first_slice_idx * z_distance * z_scale
        center = np.mean(first_points, axis=0)
        center_index = len(vertices)
        vertices.append([center[0], center[1], first_z])
        
        for k in range(len(first_points) - 1):
            triangles.append([center_index, k, k + 1])
        triangles.append([center_index, len(first_points) - 1, 0])
        
        last_slice_idx, last_points = all_layer_points[-1]
        last_z = last_slice_idx * z_distance * z_scale
        center = np.mean(last_points, axis=0)
        center_index = len(vertices)
        vertices.append([center[0], center[1], last_z])
        last_start = len(vertices) - len(last_points) - 1
        
        for k in range(len(last_points) - 1):
            triangles.append([center_index, last_start + k + 1, last_start + k])
        triangles.append([center_index, last_start, last_start + len(last_points) - 1])
        
        if vertices and triangles:
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
            smoothed_mesh = mesh.smoothed(iterations=3)
            meshes_by_material[material_name].append(smoothed_mesh)
    
    if meshes_by_material:
        scene = trimesh.Scene()
        
        for mat_name, mesh_list in meshes_by_material.items():
            if not mesh_list:
                continue
            
            combined = trimesh.util.concatenate(mesh_list)
            rgba = MODULE_COLORS.get(mat_name, MODULE_COLORS['default'])
            rgba8 = (np.array(rgba) * 255).astype(np.uint8)
            vertex_colors = np.tile(rgba8, (len(combined.vertices), 1))
            combined.visual = trimesh.visual.ColorVisuals(mesh=combined, vertex_colors=vertex_colors)
            scene.add_geometry(combined, geom_name=mat_name)
        
        return scene
    
    return None

def plot_track_length_distribution(tracks):
    if not tracks:
        return
    
    lengths = [track.get_layer_count() for track in tracks]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2), alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Track Length Distribution')
    plt.xlabel('Length (Number of Layers)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('track_length_distribution.png', dpi=300)

def plot_module_distribution(tracks):
    module_counts = {}
    for track in tracks:
        module = track.module if hasattr(track, 'module') and track.module else 'Unknown'
        module_counts[module] = module_counts.get(module, 0) + 1
    
    if not module_counts:
        return
    
    plt.figure(figsize=(10, 8))
    modules = list(module_counts.keys())
    counts = list(module_counts.values())
    colors = [MODULE_COLORS.get(m, [0.7, 0.7, 0.7])[:3] for m in modules]
    
    plt.pie(counts, labels=modules, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True, explode=[0.05] * len(modules))
    plt.axis('equal')
    plt.title('Cell Track Module Distribution')
    plt.tight_layout()
    plt.savefig('module_distribution.png', dpi=300)

def main():
    contours_file = 'transformed_cell_contours.csv'
    tracks_file = 'all_tiles_best_merged.csv'
    
    if not os.path.exists(contours_file) or not os.path.exists(tracks_file):
        print("Error: Input files not found")
        return
    
    cell_contours = load_cell_contours(contours_file)
    tracks = load_tracks(tracks_file, cell_contours)
    
    plot_track_length_distribution(tracks)
    plot_module_distribution(tracks)
    
    scene = create_smooth_cell_tracks(tracks)
    
    if scene:
        glb_file = 'all_tracks_colored_by_module.glb'
        scene.export(glb_file)
        print(f"Exported: {glb_file}")
    else:
        print("Failed to create 3D model")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Completed in {time.time() - start_time:.2f}s")
