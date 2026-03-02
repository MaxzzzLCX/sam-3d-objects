#!/usr/bin/env python3
"""
Debug Surface Completeness Evaluation

This script debugs why our surface completeness evaluation isn't working
as expected on simple geometric shapes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from surface_completeness_evaluation import evaluate_surface_completeness


def debug_surface_completeness(voxel_coords, title="Debug", grid_size=64):
    """Debug version that shows the occupancy grid"""
    
    if len(voxel_coords) == 0:
        print(f"{title}: Empty voxels")
        return
    
    print(f"\n{title}:")
    print(f"  Original voxels: {len(voxel_coords)}")
    print(f"  Voxel range: x[{voxel_coords[:, 0].min():.3f}, {voxel_coords[:, 0].max():.3f}]")
    print(f"               y[{voxel_coords[:, 1].min():.3f}, {voxel_coords[:, 1].max():.3f}]") 
    print(f"               z[{voxel_coords[:, 2].min():.3f}, {voxel_coords[:, 2].max():.3f}]")
    
    # Convert to grid coordinates
    grid_coords = np.round((voxel_coords + 0.5) * (grid_size - 1)).astype(int)
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)
    
    # Remove duplicates
    unique_coords = np.unique(grid_coords, axis=0)
    print(f"  Unique grid coords: {len(unique_coords)} (removed {len(grid_coords) - len(unique_coords)} duplicates)")
    
    # Create occupancy grid
    occupancy_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    for coord in unique_coords:
        occupancy_grid[coord[0], coord[1], coord[2]] = True
    
    # Check how many boundary voxels are occupied
    boundary_occupied = 0
    boundary_total = 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Front and back faces
            boundary_total += 2
            if occupancy_grid[i, j, 0]: boundary_occupied += 1
            if occupancy_grid[i, j, grid_size-1]: boundary_occupied += 1
            
            # Left and right faces (avoid double counting corners)
            if i != 0 and i != grid_size-1:
                boundary_total += 2
                if occupancy_grid[0, i, j]: boundary_occupied += 1
                if occupancy_grid[grid_size-1, i, j]: boundary_occupied += 1
            
            # Top and bottom faces (avoid double counting edges)
            if i != 0 and i != grid_size-1 and j != 0 and j != grid_size-1:
                boundary_total += 2
                if occupancy_grid[i, 0, j]: boundary_occupied += 1
                if occupancy_grid[i, grid_size-1, j]: boundary_occupied += 1
    
    print(f"  Boundary occupancy: {boundary_occupied}/{boundary_total} ({boundary_occupied/boundary_total*100:.1f}%)")
    
    # Calculate centroid
    centroid = np.mean(unique_coords, axis=0).astype(int)
    centroid = np.clip(centroid, 0, grid_size - 1)
    print(f"  Centroid: {centroid}")
    print(f"  Centroid occupied: {occupancy_grid[centroid[0], centroid[1], centroid[2]]}")
    
    # Run evaluation
    results = evaluate_surface_completeness(voxel_coords, grid_size)
    print(f"  Surface complete: {results['is_complete']}")
    print(f"  Flood volume ratio: {results['flood_volume_ratio']:.3f}")
    

def create_solid_cube_surface(size=0.4, thickness=2):
    """Create a solid cube surface with specified thickness"""
    voxels = []
    half_size = size / 2
    
    # Create a thick shell by using multiple layers
    for layer in range(-thickness//2, thickness//2 + 1):
        offset = layer * (size / 64)  # Small offset for thickness
        
        # Generate points on each face
        for i in range(32):
            for j in range(32):
                u = (i / 31) * size - half_size
                v = (j / 31) * size - half_size
                
                # All 6 faces
                voxels.extend([
                    [u, v, -half_size + offset],  # Front face
                    [u, v, half_size + offset],   # Back face  
                    [-half_size + offset, u, v],  # Left face
                    [half_size + offset, u, v],   # Right face
                    [u, -half_size + offset, v],  # Bottom face
                    [u, half_size + offset, v],   # Top face
                ])
    
    return np.array(voxels)


def test_improved_surfaces():
    """Test with improved surface generation"""
    
    print("Testing Improved Surface Generation")
    print("=" * 50)
    
    # Test with thicker cube surface
    thick_cube = create_solid_cube_surface(size=0.4, thickness=4)
    debug_surface_completeness(thick_cube, "Thick Cube", 64)
    
    # Test at lower resolution to ensure complete coverage
    print("\n" + "=" * 30)
    print("Testing at 32³ resolution:")
    
    from scripts_benchmarking.helper_completness.test_surface_completeness import create_cube_surface
    cube_32 = create_cube_surface(size=0.4, resolution=24)
    debug_surface_completeness(cube_32, "Cube 32³", 32)
    
    # Test very dense cube at 64³
    dense_cube = create_cube_surface(size=0.4, resolution=48)
    debug_surface_completeness(dense_cube, "Dense Cube 64³", 64)


if __name__ == "__main__":
    test_improved_surfaces()