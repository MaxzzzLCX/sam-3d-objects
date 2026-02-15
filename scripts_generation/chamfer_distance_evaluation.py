#!/usr/bin/env python3
"""
Chamfer Distance Evaluation Script

This script calculates the Chamfer distance between ground truth and predicted point clouds
using PyTorch3D implementation.
"""

import os
import numpy as np
from sympy import rotations
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


def chamfer_distance_evaluation(gt_points, pred_points, output_dir, debug=False):
    """
    The function that does full evaluation of Chamfer distance.
    """

    # Rescale point clouds to unit cube
    gt_points = rescale_points(gt_points)
    pred_points = rescale_points(pred_points)

    ## NOTE: The generation from the model will be in global canonical coordinates. So alignment with GT is not necessary
    # Thus, we try multiple initializations, and pick the one with best initial Chamfer distance.

    best_chamfer_distance = float('inf')
    best_eval_results = {}
    best_initial_rotation = None

    # Generate all 24 possible 90° axis-aligned rotations
    # Method: For each of 6 faces (+X, -X, +Y, -Y, +Z, -Z), 
    # apply 4 rotations (0°, 90°, 180°, 270°) = 6 × 4 = 24
    
    initial_rotations = []
    
    # Define which axis points "forward" (6 possibilities)
    forward_directions = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),   # +Z forward (identity base)
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),  # -Z forward (180° around X)
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # +X forward (90° around Y)
        np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),  # -X forward (-90° around Y)
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # +Y forward (90° around X)
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),  # -Y forward (-90° around X)
    ]
    
    # For each forward direction, rotate around the Z-axis (0°, 90°, 180°, 270°)
    for base in forward_directions:
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotation_z = np.eye(3)
            elif angle == 90:
                rotation_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            elif angle == 180:
                rotation_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            else:  # 270
                rotation_z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            
            initial_rotations.append(rotation_z @ base)
    
    print(f"Testing {len(initial_rotations)} possible orientations...")

    for i, rotation in enumerate(initial_rotations):
        # Apply initial rotation to predicted points
        pred_points_rotated = pred_points @ rotation.T
        
        # Calculate Chamfer distance after ICP
        chamfer_distance = calculate_chamfer_distance(gt_points, pred_points_rotated)
        
        if chamfer_distance['bidirectional_chamfer'] < best_chamfer_distance:
            best_chamfer_distance = chamfer_distance['bidirectional_chamfer']
            best_eval_results = chamfer_distance
            best_initial_rotation = rotation
        
        print(f"Rotation {i}: Chamfer Distance = {chamfer_distance['bidirectional_chamfer']:.6f}")
    
    print(f"Best initial rotation:\n{best_initial_rotation}")
    pred_points_best_initial = pred_points @ best_initial_rotation.T

    # Calculate Chamfer distances (Pre ICP)
    chamfer_results = calculate_chamfer_distance(gt_points, pred_points_best_initial)

    # Visualize pre-ICP alignment of two point clouds
    if debug:
        pc = trimesh.PointCloud(np.concatenate([gt_points, pred_points_best_initial], axis=0))
        pc_path = f"{output_dir}/pre_icp_alignment.ply"
        os.makedirs(os.path.dirname(pc_path), exist_ok=True)
        pc.export(pc_path)


    # Performs ICP alignment for two point clouds now
    gt_points_aligned, pred_points_aligned = icp_alignment(gt_points, pred_points_best_initial)

    # Visualize the post-ICP alignment of two point clouds
    if debug:
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
        "num_points": len(pred_points),
        "bidirectional_chamfer_distance": chamfer_results_icp['bidirectional_chamfer'],
        "unidirectional_chamfer_distance": chamfer_results_icp['unidirectional_chamfer'],
        "pre_icp_bidirectional_chamfer_distance": chamfer_results['bidirectional_chamfer'],
        "post_icp_bidirectional_chamfer_distance": chamfer_results_icp['bidirectional_chamfer'],
        "CD_ICP_improvement_percentage": improvement,
        "pre_icp_unidirectional_chamfer_distance": chamfer_results['unidirectional_chamfer'], # How well is GT represented in prediction
        "post_icp_unidirectional_chamfer_distance": chamfer_results_icp['unidirectional_chamfer'],
        "gt_points_count": len(gt_points),
        "pred_points_count": len(pred_points_best_initial),
        "gt_bounds": {
            "min": np.min(gt_points, axis=0).tolist(),
            "max": np.max(gt_points, axis=0).tolist()
        },
        "pred_bounds": {
            "min": np.min(pred_points_best_initial, axis=0).tolist(),
            "max": np.max(pred_points_best_initial, axis=0).tolist()
        }
    }
    
    # Set output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results for the individual viewpoint evaluation stats
    if debug:
        results_file = os.path.join(output_dir, 'chamfer_distance_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results, best_initial_rotation


def chamfer_distance_evaluation_from_files(gt_path, pred_path, output_dir, debug=False):
    """
    A wrapper function that loads point clouds from files,
    performs Chamfer distance evaluation, and saves results.
    
    Args:
        gt_path: Path to ground truth PLY file
        pred_path: Path to prediction PLY file
        output_dir: Directory to save results
    
    Returns:
        dict: Results dictionary with chamfer distance and metadata, or None if failed
    """

    output_dir = f"{output_dir}/chamfer"

    # Load point clouds
    gt_points = load_point_cloud(gt_path)
    pred_points = load_point_cloud(pred_path)
    
    if gt_points is None or pred_points is None:
        print("Failed to load point clouds")
        return None
    
    results, _ = chamfer_distance_evaluation(gt_points, pred_points, output_dir, debug)

    return results
    
    


def main():
    """
    Main function for standalone execution
    """
    parser = argparse.ArgumentParser(description='Calculate Chamfer distance between ground truth and prediction point clouds')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth PLY file')
    parser.add_argument('--pred_path', type=str, required=True, help='Path to prediction PLY file')
    parser.add_argument('--output_dir', type=str, help='Directory to save results')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output')
    
    args = parser.parse_args()
    
    print(f"Loading ground truth: {args.gt_path}")
    print(f"Loading prediction: {args.pred_path}")
    
    results = chamfer_distance_evaluation_from_files(args.gt_path, args.pred_path, args.output_dir, args.debug)
    
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