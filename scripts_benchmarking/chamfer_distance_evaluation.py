#!/usr/bin/env python3
"""
Chamfer Distance Evaluation Script

This script calculates the Chamfer distance between ground truth and predicted point clouds
using PyTorch3D implementation.
"""

import numpy as np
import trimesh
import argparse
import json
import time
from pathlib import Path
import torch
from pytorch3d.loss import chamfer_distance


def load_point_cloud(ply_path):
    """
    Load point cloud from PLY file
    
    Args:
        ply_path: Path to PLY file
    
    Returns:
        np.ndarray: Point cloud coordinates (N, 3)
    """
    try:
        mesh = trimesh.load(ply_path)
        if hasattr(mesh, 'vertices'):
            return np.array(mesh.vertices)
        else:
            print(f"Error: Could not load vertices from {ply_path}")
            return None
    except Exception as e:
        print(f"Error loading {ply_path}: {e}")
        return None


def calculate_chamfer_distance(gt_points, pred_points):
    """
    Calculate Chamfer distance using PyTorch3D
    
    Args:
        gt_points: Ground truth points (N, 3)
        pred_points: Predicted points (M, 3)
    
    Returns:
        float: Chamfer distance
    """
    # Convert to PyTorch tensors
    gt_tensor = torch.tensor(gt_points, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
    pred_tensor = torch.tensor(pred_points, dtype=torch.float32).unsqueeze(0)  # (1, M, 3)
    
    # Calculate Chamfer distance
    chamfer_dist, _ = chamfer_distance(gt_tensor, pred_tensor)
    return chamfer_dist.item()


def main():
    parser = argparse.ArgumentParser(description='Calculate Chamfer distance between ground truth and prediction point clouds')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth PLY file')
    parser.add_argument('--pred_path', type=str, required=True, help='Path to prediction PLY file')
    parser.add_argument('--output_dir', type=str, help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load point clouds
    print(f"Loading ground truth: {args.gt_path}")
    print(f"Loading prediction: {args.pred_path}")
    
    gt_points = load_point_cloud(args.gt_path)
    pred_points = load_point_cloud(args.pred_path)
    
    if gt_points is None or pred_points is None:
        print("Failed to load point clouds")
        return
    
    print(f"Ground Truth Points: {len(gt_points):,}")
    print(f"Prediction Points: {len(pred_points):,}")
    
    # Calculate Chamfer distance
    print("Calculating Chamfer distance...")
    start_time = time.time()
    chamfer_dist = calculate_chamfer_distance(gt_points, pred_points)
    calc_time = time.time() - start_time
    
    print(f"Chamfer Distance: {chamfer_dist:.6f}")
    print(f"Calculation time: {calc_time:.2f} seconds")
    
    # Prepare results
    results = {
        "chamfer_distance": chamfer_dist,
        "calculation_time_seconds": calc_time,
        "gt_points_count": len(gt_points),
        "pred_points_count": len(pred_points),
        "gt_bounds": {
            "min": np.min(gt_points, axis=0).tolist(),
            "max": np.max(gt_points, axis=0).tolist()
        },
        "pred_bounds": {
            "min": np.min(pred_points, axis=0).tolist(),
            "max": np.max(pred_points, axis=0).tolist()
        }
    }
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.gt_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_file = output_dir / 'chamfer_distance_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()