#!/usr/bin/env python3
"""
Minimal SAM3D voxel fusion without alignment
"""

import numpy as np
import trimesh

def coords_to_grid(coords, grid_size=64):
    """Convert normalized coordinates to occupancy grid"""
    # Denormalize from [-0.5, 0.5] to [0, 63]
    voxel_coords = np.round((coords + 0.5) * 63.0).astype(int)
    
    # Create binary occupancy grid
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    # Filter valid coordinates
    valid_mask = (
        (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < grid_size) &
        (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < grid_size) &
        (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < grid_size)
    )
    
    valid_coords = voxel_coords[valid_mask]
    if len(valid_coords) > 0:
        grid[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = 1.0
    
    return grid

def arithmetic_mean_fusion(grid1, grid2):
    """Simple arithmetic mean fusion"""
    return (grid1 + grid2) / 2.0

def min_entropy_fusion(grid1, grid2):
    """Minimum entropy fusion - select more confident prediction"""
    # For binary grids, entropy is minimized when value is closer to 0 or 1
    # Use absolute distance from 0.5 as confidence measure
    confidence1 = np.abs(grid1 - 0.5)
    confidence2 = np.abs(grid2 - 0.5)
    
    # Select grid1 where it's more confident, otherwise grid2
    use_grid1 = confidence1 >= confidence2
    return np.where(use_grid1, grid1, grid2)

def grid_to_points(grid, threshold=0.5):
    """Convert occupancy grid back to point coordinates"""
    occupied_coords = np.array(np.where(grid > threshold)).T
    if len(occupied_coords) == 0:
        return np.array([]).reshape(0, 3)
    
    # Convert back to normalized coordinates [-0.5, 0.5]
    points = (occupied_coords / 63.0) - 0.5
    return points

def main():
    # File paths
    file1 = "/scratch/cl927/nutritionverse-3d-new/_test_fusion_id-11-red-apple-145g/SAM3D_singleview_prediction/000/sam3d_voxels_normalized.npy"
    file2 = "/scratch/cl927/nutritionverse-3d-new/_test_fusion_id-11-red-apple-145g/SAM3D_singleview_prediction/001/sam3d_voxels_normalized.npy"
    
    # Load coordinates
    coords1 = np.load(file1)
    coords2 = np.load(file2)
    
    print(f"Loaded view 1: {len(coords1)} voxels")
    print(f"Loaded view 2: {len(coords2)} voxels")
    
    # Convert to grids
    grid1 = coords_to_grid(coords1)
    grid2 = coords_to_grid(coords2)
    
    print(f"Grid 1 occupancy: {grid1.sum()}")
    print(f"Grid 2 occupancy: {grid2.sum()}")
    
    # Arithmetic mean fusion
    print("\n=== Arithmetic Mean Fusion ===")
    fused_mean = arithmetic_mean_fusion(grid1, grid2)
    points_mean = grid_to_points(fused_mean)
    print(f"Fused points (arithmetic): {len(points_mean)}")
    
    # Save arithmetic mean result
    if len(points_mean) > 0:
        pc_mean = trimesh.PointCloud(points_mean)
        pc_mean.export("TEST_fusion_arithmetic.ply")
        print("Saved: TEST_fusion_arithmetic.ply")
    
    # Min entropy fusion
    print("\n=== Min Entropy Fusion ===")
    fused_entropy = min_entropy_fusion(grid1, grid2)
    points_entropy = grid_to_points(fused_entropy)
    print(f"Fused points (min entropy): {len(points_entropy)}")
    
    # Save min entropy result
    if len(points_entropy) > 0:
        pc_entropy = trimesh.PointCloud(points_entropy)
        pc_entropy.export("TEST_fusion_min_entropy.ply")
        print("Saved: TEST_fusion_min_entropy.ply")
    
    # Statistics
    print(f"\n=== Statistics ===")
    print(f"Union (both methods should be similar): {len(np.unique(np.vstack([coords1, coords2]) if len(coords1) > 0 and len(coords2) > 0 else coords1, axis=0))}")

if __name__ == "__main__":
    main()
