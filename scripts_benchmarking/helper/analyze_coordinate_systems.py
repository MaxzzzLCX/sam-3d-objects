#!/usr/bin/env python3

import numpy as np
import trimesh
from pathlib import Path
import json

def analyze_coordinate_system(prediction_dir):
    """Analyze coordinate systems of ground truth mesh vs SAM3D predictions"""
    
    prediction_path = Path(prediction_dir)
    
    # Load ground truth mesh 
    gt_mesh_file = prediction_path / "ground_truth_mesh_reference.ply"
    if not gt_mesh_file.exists():
        print(f"Ground truth mesh not found: {gt_mesh_file}")
        return
        
    gt_mesh = trimesh.load(gt_mesh_file)
    print(f"Ground Truth Mesh:")
    print(f"  Vertices: {len(gt_mesh.vertices)}")
    print(f"  Bounds: {gt_mesh.bounds}")
    print(f"  Center: {gt_mesh.centroid}")
    print(f"  Extents: {gt_mesh.extents}")
    
    # Analyze principal axes
    gt_pca = trimesh.points.PointCloud(gt_mesh.vertices).principal_inertia_components
    print(f"  Principal axes:\n{gt_pca}")
    
    # Load SAM3D predictions
    sam3d_voxels = np.load(prediction_path / "sam3d_voxels_normalized.npy")
    print(f"\nSAM3D Normalized Voxels:")
    print(f"  Shape: {sam3d_voxels.shape}")
    print(f"  Bounds: [{sam3d_voxels.min(axis=0)} - {sam3d_voxels.max(axis=0)}]")
    print(f"  Center: {sam3d_voxels.mean(axis=0)}")
    print(f"  Extents: {sam3d_voxels.max(axis=0) - sam3d_voxels.min(axis=0)}")
    
    # Analyze principal axes of voxel points
    voxel_pca = trimesh.points.PointCloud(sam3d_voxels).principal_inertia_components
    print(f"  Principal axes:\n{voxel_pca}")
    
    # Compute rotation between coordinate systems
    print(f"\n=== COORDINATE SYSTEM ANALYSIS ===")
    
    # Compare orientations
    print("Principal axis comparison:")
    for i in range(3):
        gt_axis = gt_pca[i]
        
        # Find closest SAM3D axis (by dot product)
        dots = [abs(np.dot(gt_axis, voxel_pca[j])) for j in range(3)]
        closest_idx = np.argmax(dots)
        closest_alignment = dots[closest_idx]
        
        print(f"  GT axis {i}: {gt_axis}")  
        print(f"  Closest SAM3D axis {closest_idx}: {voxel_pca[closest_idx]}")
        print(f"  Alignment (abs dot product): {closest_alignment:.4f}")
        print()
    
    # Try to find the rotation matrix
    try:
        # Using Kabsch algorithm to find optimal rotation
        H = np.dot(gt_pca.T, voxel_pca)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
            
        print(f"Rotation matrix from GT to SAM3D:")
        print(R)
        
        # Convert to axis-angle representation
        from scipy.spatial.transform import Rotation as R_scipy
        rotation = R_scipy.from_matrix(R)
        axis_angle = rotation.as_rotvec()
        angle_deg = np.linalg.norm(axis_angle) * 180 / np.pi
        axis = axis_angle / np.linalg.norm(axis_angle) if np.linalg.norm(axis_angle) > 0 else [0,0,1]
        
        print(f"Rotation: {angle_deg:.1f}° around axis {axis}")
        
    except Exception as e:
        print(f"Could not compute rotation matrix: {e}")

if __name__ == "__main__":
    # Find the latest prediction directory
    base_path = Path("/scratch/cl927/nutritionverse-3d-new")
    
    for item_dir in base_path.glob("*/SAM3D_singleview_prediction/*/"):
        if item_dir.is_dir() and (item_dir / "ground_truth_mesh_reference.ply").exists():
            print(f"Analyzing: {item_dir}")
            analyze_coordinate_system(item_dir)
            print("="*60)
            break