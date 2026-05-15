#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Optimizer for Margin-Based Cell Reconstruction Parameters (v9 - Unified Script)

Supports command-line arguments to specify region and GPU

Usage:
  python bayes_opt_margin_v10.py --region Tile046 --gpu 0 --n-calls 30 --n-initial 8

Optimization Parameters (4D):
1. RELATIVE_MARGIN_THRESHOLD (0.0-0.25) - Relative margin threshold
2. DTW_WEIGHT (0.2-0.6) - DTW contour similarity weight
3. OT_WEIGHT (0.2-0.6) - OT neighbor cluster weight
4. CENTROID_WEIGHT (0.2-0.6) - Centroid distance weight

Optimization Objective:
Maximize global_score = 40% * margin_quality + 40% * coverage + 20% * length

Temporary files are automatically cleaned up to keep working directory tidy.

"""

import os
import subprocess
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import glob
import shutil
import sys
import argparse
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

# Main reconstruction script (same directory as this optimiser)
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(_PKG_DIR, "cell_reconstruction_runGPU_3Drevised_v10.py")

# Work root directory (where Bayesian optimization is launched)
WORK_ROOT_DIR = os.getcwd()  # Save working directory at startup

# Configuration will be set dynamically via command-line arguments
# TARGET_REGIONS and GPU_IDS will be set in main() based on parameters
RESULT_DIR = None  # Will be set based on region dynamically
TARGET_REGIONS = []  # Will be set based on --region argument
GPU_IDS = []  # Will be set based on --gpu argument

# Parameter search space (4D)
space = [
    Real(0.01, 0.25, name='RELATIVE_MARGIN_THRESHOLD'),   # Margin threshold (0.01-0.4)
    Real(0.1, 0.6, name='DTW_WEIGHT'),                   # DTW weight
    Real(0.1, 0.6, name='OT_WEIGHT'),                    # OT weight
    Real(0.1, 0.6, name='CENTROID_WEIGHT'),              # Centroid weight
]


def create_modified_script(rm_threshold, dtw_w, ot_w, cent_w, output_script):
    """
    Create temporary script with specific parameters
    
    Args:
        rm_threshold: Relative margin threshold
        dtw_w: DTW weight (unnormalized)
        ot_w: OT weight (unnormalized)
        cent_w: Centroid weight (unnormalized)
        output_script: Output script path
    """
    with open(MAIN_SCRIPT, 'r') as f:
        content = f.read()
    
    # Normalize weights to sum to 1
    total = dtw_w + ot_w + cent_w
    dtw_norm = dtw_w / total
    ot_norm = ot_w / total
    cent_norm = cent_w / total
    
    # Use precise replacement (include comments to ensure unique matching)
    # Compatible with v10's multi-line comment format
    import re
    
    # Replace DTW_WEIGHT (compatible with v10's multi-line comment format)
    # Matches pattern: DTW_WEIGHT = 0.33        # [SAFE DEFAULT] ...
    content = re.sub(
        r'DTW_WEIGHT\s*=\s*0\.33\s+#\s*\[SAFE DEFAULT\].*',
        f'DTW_WEIGHT = {dtw_norm:.6f}        # [SAFE DEFAULT] Weight of contour similarity (Soft-DTW).',
        content
    )
    
    # Replace OT_WEIGHT
    content = re.sub(
        r'OT_WEIGHT\s*=\s*0\.33\s+#\s*\[SAFE DEFAULT\].*',
        f'OT_WEIGHT = {ot_norm:.6f}         # [SAFE DEFAULT] Weight of neighbor-cluster similarity (Optimal Transport).',
        content
    )
    
    # Replace CENTROID_WEIGHT
    content = re.sub(
        r'CENTROID_WEIGHT\s*=\s*0\.34\s+#\s*\[SAFE DEFAULT\].*',
        f'CENTROID_WEIGHT = {cent_norm:.6f}   # [SAFE DEFAULT] Weight of centroid distance (spatial proximity).',
        content
    )
    
    # Replace RELATIVE_MARGIN_THRESHOLD (compatible with v10 format)
    # Matches pattern: RELATIVE_MARGIN_THRESHOLD = 0.01  # Accept matches where margin...
    content = re.sub(
        r'RELATIVE_MARGIN_THRESHOLD\s*=\s*0\.01\s+#\s*Accept matches.*',
        f'RELATIVE_MARGIN_THRESHOLD = {rm_threshold:.6f}  # Accept matches where margin is at least {rm_threshold*100:.1f}% of top1_score',
        content
    )
    
    # Save modified script
    with open(output_script, 'w') as f:
        f.write(content)


def run_script_single_region(script_path, region_name, gpu_id, output_base_dir):
    """
    Run script to process single region with specified GPU
    
    Args:
        script_path: Script path
        region_name: Region name (e.g., 'Tile048', 'Tile049')
        gpu_id: GPU ID (0 or 1)
        output_base_dir: Output base directory
        
    Returns:
        bool: Whether successful
    """
    try:
        # Create region-specific output directory
        output_dir = os.path.join(output_base_dir, f"{region_name}_gpu{gpu_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current working directory
        original_dir = os.getcwd()
        
        # Check if data directory exists
        if not os.path.exists(region_name):
            print(f"Warning: Data directory {region_name} not found")
            return False
        
        
        # Create data link in output directory
        target_dir = os.path.join(output_dir, region_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy CSV files (use hard links to save space)
            # Support both lighter_*.csv and *_S*_features.csv formats
            csv_files = glob.glob(os.path.join(region_name, "lighter_*.csv"))
            if not csv_files:
                csv_files = glob.glob(os.path.join(region_name, "*_S*_features.csv"))
            for csv_file in csv_files:
                target_file = os.path.join(target_dir, os.path.basename(csv_file))
                try:
                    os.link(csv_file, target_file)
                except OSError:
                    shutil.copy(csv_file, target_file)
        
        # Use absolute path to run script, avoid relative path issues
        script_abs_path = os.path.abspath(script_path)
        
        # Switch to output directory to run (data already linked here)
        os.chdir(output_dir)
        
        # Set environment variable: pass work root directory to subprocess
        env = os.environ.copy()
        env['DIVISOR_CACHE_DIR'] = WORK_ROOT_DIR
        
        # Use command-line arguments to specify GPU and target region
        cmd = ['python', script_abs_path, '--target-cells', region_name, '--gpu-id', str(gpu_id)]
        subprocess.run(cmd, check=True, env=env)
        
        # Return to original directory
        os.chdir(original_dir)
        
        return True
        
    except subprocess.CalledProcessError as e:
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return False
    except Exception as e:
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return False


def run_script_dual_gpu(script_path, regions=TARGET_REGIONS, gpus=GPU_IDS):
    """
    Run script in parallel to process multiple regions (using multiple GPUs)
    
    Args:
        script_path: Script path
        regions: Region list
        gpus: GPU ID list
        
    Returns:
        bool: Whether all regions succeeded
    """
    import concurrent.futures
    import threading
    
    # Get parameter string for creating separate output directory
    # Support temp_ and best_ prefixes, automatically include region identifier
    params_str = os.path.basename(script_path).replace("temp_", "").replace("best_", "").replace(".py", "")
    output_base_dir = f"output_{params_str}"
    os.makedirs(output_base_dir, exist_ok=True)
    
    
    # Use thread pool for parallel execution (subprocess releases GIL)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(regions)) as executor:
        futures = []
        for i, region in enumerate(regions):
            gpu_id = gpus[i % len(gpus)]  # Round-robin GPU assignment
            future = executor.submit(
                run_script_single_region,
                script_path,
                region,
                gpu_id,
                output_base_dir
            )
            futures.append((region, gpu_id, future))
        
        # Wait for all tasks to complete
        results = []
        for region, gpu_id, future in futures:
            try:
                success = future.result()
                results.append(success)
            except Exception as e:
                print(f"[ERROR] GPU {gpu_id} failed: {e}")
                results.append(False)
    
    all_success = all(results)
    
    return all_success


def get_evaluation_score_multi_region(params_str, regions=TARGET_REGIONS):
    """
    Extract and combine global_score from evaluation files of multiple regions
    
    Strategy: Take average of all region scores (can be modified to weighted average)
    
    Args:
        params_str: Parameter string
        regions: Region list
        
    Returns:
        float: Combined score
    """
    output_base_dir = f"output_{params_str}"
    
    region_scores = []
    
    for region in regions:
        # Find evaluation file for this region
        region_output = os.path.join(output_base_dir, f"{region}_gpu*")
        eval_pattern = os.path.join(region_output, "*_evaluation_*.json")
        eval_files = glob.glob(eval_pattern)
        
        if not eval_files:
            continue
        
        latest_file = max(eval_files, key=os.path.getmtime)
        
        try:
            with open(latest_file, 'r') as f:
                metrics = json.load(f)
            score = metrics.get("global_score", -1)
            region_scores.append(score)
        except Exception as e:
            continue
    
    if not region_scores:
        print("[ERROR] No valid evaluation results")
        return -1
    
    # Combined score
    combined_score = np.mean(region_scores)
    
    return combined_score


def save_evaluation_to_result_dir(params_str, regions=TARGET_REGIONS, cleanup_temp=True):
    """
    Save multi-region evaluation files to result directory and clean up temporary files
    
    Args:
        params_str: Parameter string
        regions: Region list
        cleanup_temp: Whether to clean up temporary output directory (default True)
    """
    target_dir = f"{RESULT_DIR}/{params_str}"
    os.makedirs(target_dir, exist_ok=True)
    
    output_base_dir = f"output_{params_str}"
    
    # File patterns to save
    patterns_to_save = [
        "*_evaluation_*.json",
        "*_evaluation_radar_*.png",
        "*_track_lengths.png",
        "*_margin_distribution_*.png",
        "*_tracks_distribution_*.png",
        "*_tracks_round_*.csv",
        "*_all_tracks_summary.csv"
    ]
    
    saved_count = 0
    
    # Create subdirectory for each region and save results
    if os.path.exists(output_base_dir):
        for region in regions:
            region_dir = os.path.join(target_dir, region)
            os.makedirs(region_dir, exist_ok=True)
            
            # Find output directory for this region
            region_output = os.path.join(output_base_dir, f"{region}_gpu*")
            region_outputs = glob.glob(region_output)
            
            for region_output_dir in region_outputs:
                if os.path.isdir(region_output_dir):
                    for pattern in patterns_to_save:
                        for file in glob.glob(os.path.join(region_output_dir, pattern)):
                            basename = os.path.basename(file)
                            try:
                                shutil.copy(file, os.path.join(region_dir, basename))
                                saved_count += 1
                            except Exception as e:
                                print(f"  Warning: Failed to copy file {basename}: {e}")
    
    
    # Delayed cleanup: Don't immediately delete temp directory to avoid multi-GPU conflicts
    # Temp directories will be cleaned up after optimization completes
    # if cleanup_temp and os.path.exists(output_base_dir):
    #     try:
    #         shutil.rmtree(output_base_dir)
    #     except:
    #         pass


def cleanup_all_temp_files():
    """Clean up all residual temporary files and directories"""
    # Clean up temporary output_* directories
    for output_dir in glob.glob("output_*"):
        if os.path.isdir(output_dir):
            try:
                shutil.rmtree(output_dir)
            except:
                pass
    
    # Clean up temporary script files
    for temp_script in glob.glob("temp_*.py") + glob.glob("best_*.py"):
        try:
            os.remove(temp_script)
        except:
            pass
    
    # Clean up orphaned evaluation files (support both cell-* and Tile* formats)
    eval_patterns = [
        "cell-*_evaluation_*.json", "cell-*_evaluation_*.png",
        "Tile*_evaluation_*.json", "Tile*_evaluation_*.png"
    ]
    for pattern in eval_patterns:
        for eval_file in glob.glob(pattern):
            if os.path.isfile(eval_file):
                try:
                    os.remove(eval_file)
                except:
                    pass


@use_named_args(space)
def objective(**params):
    """
    Optimization objective function
    
    Args:
        RELATIVE_MARGIN_THRESHOLD: Margin threshold
        DTW_WEIGHT: DTW weight
        OT_WEIGHT: OT weight
        CENTROID_WEIGHT: Centroid weight
    
    Returns:
        Negative global_score (since gp_minimize minimizes)
    """
    rm_thresh = params['RELATIVE_MARGIN_THRESHOLD']
    dtw_w = params['DTW_WEIGHT']
    ot_w = params['OT_WEIGHT']
    cent_w = params['CENTROID_WEIGHT']
    
    # Normalize weights
    total = dtw_w + ot_w + cent_w
    dtw_norm = dtw_w / total
    ot_norm = ot_w / total
    cent_norm = cent_w / total
    
    # Format parameter string for filename
    params_str = (f"RM{rm_thresh:.4f}_DTW{dtw_norm:.4f}_"
                  f"OT{ot_norm:.4f}_CENT{cent_norm:.4f}")
    params_str_safe = params_str.replace('.', '_')
    
    # Add region identifier to avoid filename conflicts in multi-region concurrent runs
    region_tag = f"{TARGET_REGIONS[0]}_gpu{GPU_IDS[0]}"
    params_str_with_region = f"{params_str_safe}_{region_tag}"
    
    # Simplified parameter output
    print(f"[EVAL] RM={rm_thresh:.4f}, DTW={dtw_norm:.4f}, OT={ot_norm:.4f}, CENT={cent_norm:.4f}")
    
    # Check if this parameter set has already been evaluated (support both cell-* and Tile* formats)
    # IMPORTANT: Use params_str_with_region (not params_str_safe) to match save path
    result_subdir = f"{RESULT_DIR}/{params_str_with_region}"
    if os.path.exists(result_subdir):
        # Try to find evaluation files in both formats
        existing_files = glob.glob(f"{result_subdir}/**/cell-*_evaluation_*.json", recursive=True)
        if not existing_files:
            existing_files = glob.glob(f"{result_subdir}/**/Tile*_evaluation_*.json", recursive=True)
        if existing_files:
            try:
                with open(existing_files[0], 'r') as f:
                    metrics = json.load(f)
                score = metrics.get("global_score", -1)
                print(f"[CACHED] Score={score:.2f}")
                return -score
            except:
                pass
    
    # Temp script name includes region identifier to avoid conflicts
    temp_script = f"temp_{params_str_with_region}.py"
    create_modified_script(rm_thresh, dtw_w, ot_w, cent_w, temp_script)
    
    success = run_script_dual_gpu(temp_script, regions=TARGET_REGIONS, gpus=GPU_IDS)
    
    if os.path.exists(temp_script):
        os.remove(temp_script)
    
    if not success:
        print("[ERROR] Evaluation failed")
        return 0
    
    # Add delay to ensure all files are written (multi-GPU concurrent needs more time)
    time.sleep(2.0)
    
    # Use parameter string with region identifier to get evaluation score
    score = get_evaluation_score_multi_region(params_str_with_region, regions=TARGET_REGIONS)
    print(f"[SCORE] {score:.2f}")
    
    # Save results using parameter string with region identifier (avoid conflicts)
    save_evaluation_to_result_dir(params_str_with_region, regions=TARGET_REGIONS, cleanup_temp=False)
    
    # Return negative score (since Bayesian optimization minimizes by default)
    return -score


def run_optimization(n_calls=30, n_initial_points=8, quiet=False, round_num=1):
    """
    Run Bayesian optimization process (4D parameter space)
    
    Args:
        n_calls: Total number of evaluations
        n_initial_points: Number of initial random points
        quiet: Whether to run in quiet mode
        round_num: Optimization round number (used to generate different random seeds)
    """
    # Set different random seed based on region and round to ensure different initial points
    # Generate unique random seed for each Tile/Region
    region_name = TARGET_REGIONS[0]
    
    # Base seed changes with round: round 1=42000, round 2=43000, round 3=44000...
    base_seed = 42000 + (round_num - 1) * 1000
    
    # Try to extract number from region name as part of seed
    import re
    numbers = re.findall(r'\d+', region_name)
    if numbers:
        # Use extracted number to generate seed (e.g., Tile048 round1 -> 42048, round2 -> 43048)
        region_num = int(numbers[0])
        random_seed = base_seed + region_num
    else:
        # Use hash of region name to generate seed
        random_seed = base_seed + abs(hash(region_name)) % 1000
    
    if not quiet:
        print(f"[START] Bayesian optimization: n_calls={n_calls}, n_initial={n_initial_points}, seed={random_seed}, round={round_num}")
        print(f"[START] Target={TARGET_REGIONS[0]}, GPU={GPU_IDS[0]}")
    
    start_time = time.time()
    
    # Create result directory
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # Run Bayesian optimization (using different random seeds)
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_seed,  # Different region uses different seed
        verbose=False if quiet else True,
        n_jobs=1  # Sequential execution (each run requires GPU program)
    )
    
    # Extract best parameters
    best_rm = result.x[0]
    best_dtw = result.x[1]
    best_ot = result.x[2]
    best_cent = result.x[3]
    
    # Normalize weights
    total = best_dtw + best_ot + best_cent
    best_dtw_norm = best_dtw / total
    best_ot_norm = best_ot / total
    best_cent_norm = best_cent / total
    
    best_score = -result.fun
    elapsed_time = time.time() - start_time
    
    # Output results (simplified)
    if not quiet:
        print(f"\n[RESULT] Best score: {best_score:.2f}, Time: {elapsed_time/3600:.2f}h")
        print(f"[RESULT] Best params: RM={best_rm:.4f}, DTW={best_dtw_norm:.4f}, OT={best_ot_norm:.4f}, CENT={best_cent_norm:.4f}")
    
    # Save optimization history
    history = {
        'iterations': [int(i) for i in range(len(result.func_vals))],
        'params': [[float(p) for p in x] for x in result.x_iters],
        'scores': [float(-y) for y in result.func_vals],
        'best_score': float(best_score),
        'best_params': {
            'RELATIVE_MARGIN_THRESHOLD': float(best_rm),
            'DTW_WEIGHT': float(best_dtw_norm),
            'OT_WEIGHT': float(best_ot_norm),
            'CENTROID_WEIGHT': float(best_cent_norm)
        },
        'optimization_time_hours': elapsed_time / 3600
    }
    
    with open(f"{RESULT_DIR}/optimization_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create history table
    normalized_params = []
    for p in history['params']:
        rm, dtw, ot, cent = p
        total = dtw + ot + cent
        normalized_params.append([rm, dtw/total, ot/total, cent/total])
    
    df = pd.DataFrame({
        'Iteration': history['iterations'],
        'RELATIVE_MARGIN_THRESHOLD': [p[0] for p in normalized_params],
        'DTW_WEIGHT': [p[1] for p in normalized_params],
        'OT_WEIGHT': [p[2] for p in normalized_params],
        'CENTROID_WEIGHT': [p[3] for p in normalized_params],
        'Global_Score': history['scores']
    })
    
    df.to_csv(f"{RESULT_DIR}/optimization_history.csv", index=False)
    
    # Plot convergence curve
    if not quiet:
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(len(result.func_vals)),
            [-float(y) for y in result.func_vals],
            'o-',
            label='Score History',
            markersize=6
        )
        plt.axhline(y=float(best_score), color='r', linestyle='--', 
                    linewidth=2, label=f'Best Score: {best_score:.2f}')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Global Score', fontsize=12)
        plt.title(f'Convergence - {TARGET_REGIONS[0]}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/convergence.png", dpi=150)
        plt.close()
    
    # Plot parameter space (only in non-quiet mode)
    if not quiet:
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            ax = axes.flatten()
            
            plot_objective(result, dimensions=["RELATIVE_MARGIN_THRESHOLD", "DTW_WEIGHT"], ax=ax[0])
            ax[0].set_title("Margin Threshold vs DTW Weight")
            
            plot_objective(result, dimensions=["RELATIVE_MARGIN_THRESHOLD", "OT_WEIGHT"], ax=ax[1])
            ax[1].set_title("Margin Threshold vs OT Weight")
            
            plot_objective(result, dimensions=["DTW_WEIGHT", "OT_WEIGHT"], ax=ax[2])
            ax[2].set_title("DTW Weight vs OT Weight")
            
            plot_objective(result, dimensions=["DTW_WEIGHT", "CENTROID_WEIGHT"], ax=ax[3])
            ax[3].set_title("DTW Weight vs Centroid Weight")
            
            plot_objective(result, dimensions=["OT_WEIGHT", "CENTROID_WEIGHT"], ax=ax[4])
            ax[4].set_title("OT Weight vs Centroid Weight")
            
            plot_objective(result, dimensions=["RELATIVE_MARGIN_THRESHOLD", "CENTROID_WEIGHT"], ax=ax[5])
            ax[5].set_title("Margin Threshold vs Centroid Weight")
            
            plt.tight_layout()
            plt.savefig(f"{RESULT_DIR}/parameter_space.png", dpi=150)
            plt.close()
        except Exception as e:
            if not quiet:
                print(f"[WARNING] Failed to plot parameter space: {e}")
    
    # Run final evaluation with best parameters
    if not quiet:
        print(f"\n[FINAL] Running final evaluation with best params...")
    
    best_params_str = f"BEST_RM{best_rm:.4f}_DTW{best_dtw_norm:.4f}_OT{best_ot_norm:.4f}_CENT{best_cent_norm:.4f}"
    best_params_str_safe = best_params_str.replace('.', '_')
    
    # Add region identifier to avoid conflicts
    region_tag = f"{TARGET_REGIONS[0]}_gpu{GPU_IDS[0]}"
    best_params_str_with_region = f"{best_params_str_safe}_{region_tag}"
    best_script = f"best_{best_params_str_with_region}.py"
    
    create_modified_script(best_rm, best_dtw, best_ot, best_cent, best_script)
    run_script_dual_gpu(best_script, regions=TARGET_REGIONS, gpus=GPU_IDS)
    
    # Create best parameters directory and copy multi-region results
    best_dir_path = f"{RESULT_DIR}/BEST_{best_params_str_safe}"
    os.makedirs(best_dir_path, exist_ok=True)
    
    # Copy final model and evaluation results (using output directory with region identifier)
    output_best = f"output_{best_params_str_with_region}"
    if os.path.exists(output_best):
        # Create subdirectory for each region and copy results
        for region in TARGET_REGIONS:
            region_dir = os.path.join(best_dir_path, region)
            os.makedirs(region_dir, exist_ok=True)
            
            # Find output directory for this region
            region_outputs = glob.glob(os.path.join(output_best, f"{region}_gpu*"))
            
            for region_output_dir in region_outputs:
                if os.path.isdir(region_output_dir):
                    for pattern in ["*_evaluation_*.json", "*_evaluation_*.png", 
                                   "*_all_tracks_summary.csv", "*_track_lengths.png",
                                   "*_margin_distribution_*.png", "*_tracks_round_*.csv"]:
                        for file in glob.glob(os.path.join(region_output_dir, pattern)):
                            shutil.copy(file, os.path.join(region_dir, os.path.basename(file)))
                    
                    # Copy 3D models (if needed)
                    for obj_pattern in ["*_all_tracks.obj", "*_long_tracks.obj"]:
                        for obj_file in glob.glob(os.path.join(region_output_dir, obj_pattern)):
                            shutil.copy(obj_file, os.path.join(region_dir, os.path.basename(obj_file)))
    
    # Delete temporary script
    if os.path.exists(best_script):
        os.remove(best_script)
    
    # Don't delete temporary output directory to avoid concurrent conflicts and result loss
    # Clean up best parameters temporary output directory
    # if os.path.exists(output_best):
    #     try:
    #         shutil.rmtree(output_best)
    #         print(f"  Cleaned temp directory: {output_best}/")
    #     except Exception as e:
    #         print(f"  Failed to clean temp directory: {e}")
    
    # Don't automatically clean temporary files to avoid concurrent conflicts and result loss
    # If cleanup needed, manually run cleanup_all_temp_files() after all optimizations complete
    # Clean up temporary files
    # if not quiet:
    #     cleanup_all_temp_files()
    # else:
    #     # Clean up in quiet mode too, but no output
    #     for output_dir in glob.glob("output_*"):
    #         if os.path.isdir(output_dir):
    #             try:
    #                 shutil.rmtree(output_dir)
    #             except:
    #                 pass
    #     for temp_script in glob.glob("temp_*.py") + glob.glob("best_*.py"):
    #         try:
    #             os.remove(temp_script)
    #         except:
    #             pass
    
    # Completion message
    if not quiet:
        print(f"\n[COMPLETED] Optimization finished")
        print(f"[COMPLETED] Results saved to: {RESULT_DIR}/")
        print(f"[COMPLETED] Best params saved to: {best_dir_path}/")
    
    return result


def main():
    """Main function"""
    global RESULT_DIR, TARGET_REGIONS, GPU_IDS
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Bayesian Optimizer for Margin-Based Cell Reconstruction Parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Tile048 using GPU 0
  python bayes_opt_margin_v9.py --region Tile048 --gpu 0 --n-calls 30 --n-initial 8
  
  # Tile049 using GPU 1  
  python bayes_opt_margin_v9.py --region Tile049 --gpu 1 --n-calls 30 --n-initial 8
  
  # Quick test
  python bayes_opt_margin_v9.py --region Tile048 --gpu 0 --n-calls 20 --n-initial 5
        """
    )
    parser.add_argument('--region', type=str, required=True,
                       help='Target region: e.g., Tile048, Tile049, Tile052, etc.')
    parser.add_argument('--gpu', type=int, required=True,
                       help='GPU ID: 0, 1, 2, ...')
    parser.add_argument('--n-calls', type=int, default=30,
                       help='Total number of evaluations (default 30)')
    parser.add_argument('--n-initial', type=int, default=8,
                       help='Number of initial random points (default 8)')
    parser.add_argument('--round', type=int, default=1,
                       help='Optimization round (default 1), different rounds use different random seeds')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode, only output key information')
    
    args = parser.parse_args()
    
    # Set configuration based on arguments (support any Tile directory)
    TARGET_REGIONS = [args.region]
    # Add round identifier to result directory name
    if args.round > 1:
        RESULT_DIR = f"bayes_opt_results_v9_{args.region}_round{args.round}"
    else:
        RESULT_DIR = f"bayes_opt_results_v9_{args.region}"
    
    GPU_IDS = [args.gpu]
    
    # Simplified validation
    if not os.path.exists(MAIN_SCRIPT):
        print(f"ERROR: Main script {MAIN_SCRIPT} not found")
        return
    
    for region in TARGET_REGIONS:
        if not os.path.exists(region):
            print(f"ERROR: Region directory {region} not found")
            return
    
    # Output configuration (simplified)
    if not args.quiet:
        print(f"[CONFIG] Region={args.region}, GPU={args.gpu}, n_calls={args.n_calls}, n_initial={args.n_initial}, round={args.round}")
        print(f"[CONFIG] Target={TARGET_REGIONS[0]}, ResultDir={RESULT_DIR}")
    
    # Run optimization process
    result = run_optimization(n_calls=args.n_calls, n_initial_points=args.n_initial, quiet=args.quiet, round_num=args.round)
    
    if not args.quiet:
        print(f"[DONE] Optimization completed")


if __name__ == "__main__":
    main()

