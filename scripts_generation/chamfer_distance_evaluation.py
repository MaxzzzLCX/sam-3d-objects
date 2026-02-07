#!/usr/bin/env python3
"""
Chamfer Distance Evaluation Script

This script calculates the Chamfer distance between ground truth and predicted point clouds
using PyTorch3D implementation.
"""

import os
import numpy as np
import trimesh
import argparse
import json
import time
from pathlib import Path
import torch
from pytorch3d.loss import chamfer_distance
import open3d as o3d


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
    Calculate both bidirectional and unidirectional Chamfer distances using PyTorch3D
    
    Args:
        gt_points: Ground truth points (N, 3)
        pred_points: Predicted points (M, 3)
    
    Returns:
        dict: Dictionary containing:
            - bidirectional_chamfer: Standard bidirectional Chamfer distance
            - unidirectional_chamfer: Distance from GT to prediction (sensitive to missing regions)
    """
    # Convert to PyTorch tensors
    gt_tensor = torch.tensor(gt_points, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
    pred_tensor = torch.tensor(pred_points, dtype=torch.float32).unsqueeze(0)  # (1, M, 3)
    
    # Calculate bidirectional Chamfer distance (default)
    bidirectional_dist, _ = chamfer_distance(gt_tensor, pred_tensor, single_directional=False)
    
    # Calculate unidirectional Chamfer distance (GT -> prediction)
    # This measures how well each GT point is represented in the prediction
    # Higher values indicate missing regions in the prediction
    unidirectional_dist, _ = chamfer_distance(gt_tensor, pred_tensor, single_directional=True)
    
    return {
        'bidirectional_chamfer': bidirectional_dist.item(),
        'unidirectional_chamfer': unidirectional_dist.item()
    }

def rescale_points(points):
    """
    Rescale points to fit within unit cube centered at origin.
    """

    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    center = (min_bounds + max_bounds) / 2.0
    scale = np.max(max_bounds - min_bounds)
    points_rescaled = (points - center) / scale
    return points_rescaled


def icp_alignment(gt_points, pred_points, max_iterations=50, tolerance=1e-6):
    """
    Perform ICP alignment of predicted points to ground truth points.
    
    Args:
        gt_points: Ground truth points (N, 3)
        pred_points: Predicted points (M, 3)
        max_iterations: Maximum number of ICP iterations
        tolerance: Convergence tolerance
    
    Returns:
        np.ndarray: Aligned predicted points (M, 3)
    """
    # Convert to Open3D point clouds
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
    
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
    
    result = o3d.pipelines.registration.registration_icp(
        pred_pcd, gt_pcd, max_correspondence_distance=0.05,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    transformation = result.transformation
    pred_pcd.transform(transformation)
    aligned_pred_points = np.asarray(pred_pcd.points)
    return gt_points, aligned_pred_points


def calculate_chamfer_distance_for_files(gt_path, pred_path, output_dir=None):
    """
    Calculate Chamfer distance between ground truth and prediction point clouds
    
    Args:
        gt_path: Path to ground truth PLY file
        pred_path: Path to prediction PLY file
        output_dir: Directory to save results (defaults to same as gt_path)
    
    Returns:
        dict: Results dictionary with chamfer distance and metadata, or None if failed
    """
    # Load point clouds
    gt_points = load_point_cloud(gt_path)
    pred_points = load_point_cloud(pred_path)
    
    if gt_points is None or pred_points is None:
        print("Failed to load point clouds")
        return None
    
    # Rescale point clouds to unit cube
    gt_points = rescale_points(gt_points)
    pred_points = rescale_points(pred_points)


    # Calculate Chamfer distances (Pre ICP)
    chamfer_results = calculate_chamfer_distance(gt_points, pred_points)

    # Visualize pre-ICP alignment of two point clouds
    pc = trimesh.PointCloud(np.concatenate([gt_points, pred_points], axis=0))
    pc_path = f"{output_dir}/pre_icp_alignment.ply"
    os.makedirs(os.path.dirname(pc_path), exist_ok=True)
    pc.export(pc_path)

    # Performs ICP alignment for two point clouds now
    gt_points_aligned, pred_points_aligned = icp_alignment(gt_points, pred_points)

    # Visualize the post-ICP alignment of two point clouds
    pc_aligned = trimesh.PointCloud(np.concatenate([gt_points_aligned, pred_points_aligned], axis=0))
    pc_aligned_path = f"{output_dir}/post_icp_alignment.ply"
    os.makedirs(os.path.dirname(pc_aligned_path), exist_ok=True)
    pc_aligned.export(pc_aligned_path)

    # Calculate Chamfer distances after ICP alignment
    chamfer_results_icp = calculate_chamfer_distance(gt_points_aligned, pred_points_aligned)
    improvement = (chamfer_results['bidirectional_chamfer'] - chamfer_results_icp['bidirectional_chamfer']) / chamfer_results['bidirectional_chamfer'] * 100.0

    print(f"Chamfer Distance before ICP: {chamfer_results}")
    print(f"Chamfer Distance after ICP: {chamfer_results_icp}")
    
    # Prepare results
    results = {
        "chamfer_distance": chamfer_results_icp['bidirectional_chamfer'], # Default CD (Post-ICP, bidirectional)
        "bidirectional_chamfer_distance": chamfer_results_icp['bidirectional_chamfer'],
        "unidirectional_chamfer_distance": chamfer_results_icp['unidirectional_chamfer'],
        "chamfer_distance_pre_icp": chamfer_results['bidirectional_chamfer'],
        "chamfer_distance_post_icp": chamfer_results_icp['bidirectional_chamfer'],
        "ICP_improvement_percentage": improvement,
        "chamfer_distance_pre_icp_unidirectional": chamfer_results['unidirectional_chamfer'], # How well is GT represented in prediction
        "chamfer_distance_post_icp_unidirectional": chamfer_results_icp['unidirectional_chamfer'],
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
    if output_dir is None:
        output_dir = Path(gt_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_file = output_dir / 'chamfer_distance_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """
    Main function for standalone execution
    """
    parser = argparse.ArgumentParser(description='Calculate Chamfer distance between ground truth and prediction point clouds')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth PLY file')
    parser.add_argument('--pred_path', type=str, required=True, help='Path to prediction PLY file')
    parser.add_argument('--output_dir', type=str, help='Directory to save results')
    
    args = parser.parse_args()
    
    print(f"Loading ground truth: {args.gt_path}")
    print(f"Loading prediction: {args.pred_path}")
    
    results = calculate_chamfer_distance_for_files(args.gt_path, args.pred_path, args.output_dir)
    
    if results:
        print(f"Ground Truth Points: {results['gt_points_count']:,}")
        print(f"Prediction Points: {results['pred_points_count']:,}")
        print("Calculating Chamfer distances...")
        print(f"Bidirectional Chamfer Distance: {results['bidirectional_chamfer_distance']:.6f}")
        print(f"Unidirectional Chamfer Distance (GT→Pred): {results['unidirectional_chamfer_distance']:.6f}")
        
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.gt_path).parent
        results_file = output_dir / 'chamfer_distance_results.json'
        print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()