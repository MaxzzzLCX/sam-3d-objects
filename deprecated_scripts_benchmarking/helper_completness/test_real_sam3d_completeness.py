#!/usr/bin/env python3
"""
Test Surface Completeness on Real SAM3D Voxels

This script tests surface completeness evaluation on real SAM3D voxel coordinates
to verify it works correctly on actual sparse voxel representations.
"""

import numpy as np
import os
import sys
sys.path.append('/scratch/cl927/sam-3d-objects/scripts_benchmarking')

from surface_completeness_evaluation import evaluate_surface_completeness


def load_voxel_coordinates_from_npy(npy_path):
    """Load voxel coordinates from .npy file"""
    try:
        voxels = np.load(npy_path)
        print(f"Loaded {len(voxels)} voxel coordinates from {npy_path}")
        return voxels
    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return None


def test_sam3d_voxel_completeness():
    """Test surface completeness on real SAM3D voxel outputs"""
    
    print("Testing Surface Completeness on Real SAM3D Voxels")
    print("=" * 60)
    
    # Look for some SAM3D output directories with normalized voxels
    base_paths = [
        "/scratch/cl927/nutritionverse-3d-new/id-37-costco-cucumber-sushi-roll-1-16g/SAM3D_singleview_prediction",
        "/scratch/cl927/nutritionverse-3d-new/id-11-red-apple-145g/SAM3D_singleview_prediction", 
        "/scratch/cl927/nutritionverse-3d-new/id-26-chicken-leg-133g/SAM3D_singleview_prediction"
    ]
    
    for base_path in base_paths:
        if not os.path.exists(base_path):
            continue
            
        food_name = os.path.basename(os.path.dirname(base_path))
        print(f"\n{'-' * 50}")
        print(f"Testing: {food_name}")
        print(f"{'-' * 50}")
        
        # Check first few views
        for view_idx in range(min(3, len(os.listdir(base_path)))):
            view_dir = os.path.join(base_path, f"{view_idx:03d}")
            if not os.path.exists(view_dir):
                continue
                
            # Look for normalized voxels file
            normalized_voxels_file = os.path.join(view_dir, "sam3d_voxels_normalized.npy")
            if not os.path.exists(normalized_voxels_file):
                print(f"  View {view_idx}: sam3d_voxels_normalized.npy not found")
                continue
            
            print(f"  View {view_idx}:")
            voxels = load_voxel_coordinates_from_npy(normalized_voxels_file)
            
            if voxels is None:
                continue
            
            print(f"    Voxel range: x[{voxels[:, 0].min():.3f}, {voxels[:, 0].max():.3f}]")
            print(f"                 y[{voxels[:, 1].min():.3f}, {voxels[:, 1].max():.3f}]")
            print(f"                 z[{voxels[:, 2].min():.3f}, {voxels[:, 2].max():.3f}]")
            
            # Test surface completeness at different resolutions
            for grid_size in [32, 64]:
                print(f"    Grid {grid_size}³:")
                
                results = evaluate_surface_completeness(voxels, grid_size=grid_size)
                
                print(f"      Surface complete: {results['is_complete']}")
                print(f"      Centroid reachable: {results['centroid_reachable']}") 
                print(f"      Flood volume ratio: {results['flood_volume_ratio']:.3f}")
                print(f"      Evaluation time: {results['evaluation_time']:.3f}s")
                
    print(f"\n{'=' * 60}")
    print("Test completed!")


if __name__ == "__main__":
    test_sam3d_voxel_completeness()