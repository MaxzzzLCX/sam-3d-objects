#!/usr/bin/env python3
"""
Test Discrete vs Normalized Surface Completeness Evaluation

This script compares the discrete method (using raw SAM3D coordinates) 
vs the normalized method (with coordinate transformations).
"""

import numpy as np
import sys
sys.path.append('/scratch/cl927/sam-3d-objects/scripts_benchmarking')

from surface_completeness_evaluation import (
    evaluate_surface_completeness_auto,
    evaluate_surface_completeness_discrete, 
    evaluate_surface_completeness_normalized,
    convert_raw_to_discrete
)


def compare_methods():
    """Compare discrete vs normalized surface completeness evaluation"""
    
    print("Comparing Discrete vs Normalized Surface Completeness Evaluation")
    print("=" * 70)
    
    # Test cases - using real SAM3D data
    test_cases = [
        "/scratch/cl927/nutritionverse-3d-new/id-11-red-apple-145g/SAM3D_singleview_prediction/000/sam3d_voxels_raw.npy",
        "/scratch/cl927/nutritionverse-3d-new/id-11-red-apple-145g/SAM3D_singleview_prediction/000/sam3d_voxels_normalized.npy",
        "/scratch/cl927/nutritionverse-3d-new/id-37-costco-cucumber-sushi-roll-1-16g/SAM3D_singleview_prediction/000/sam3d_voxels_raw.npy",
        "/scratch/cl927/nutritionverse-3d-new/id-37-costco-cucumber-sushi-roll-1-16g/SAM3D_singleview_prediction/000/sam3d_voxels_normalized.npy"
    ]
    
    for i, voxel_path in enumerate(test_cases):
        if i % 2 == 0:
            print(f"\n{'-' * 50}")
            food_name = voxel_path.split('/')[-3].split('-', 2)[-1]
            print(f"Testing: {food_name}")
            print(f"{'-' * 50}")
        
        try:
            voxels = np.load(voxel_path)
            file_type = "RAW" if "raw.npy" in voxel_path else "NORMALIZED"
            
            print(f"\n{file_type} voxels ({len(voxels)} points):")
            
            # Method 1: Auto-detection
            results_auto = evaluate_surface_completeness_auto(voxels)
            print(f"  Auto method: {results_auto['method']}")
            print(f"    Complete: {results_auto['is_complete']}")
            print(f"    Flood ratio: {results_auto['flood_volume_ratio']:.3f}")
            print(f"    Time: {results_auto['evaluation_time']:.3f}s")
            
            if results_auto['method'] == 'discrete':
                print(f"    Detected grid: {results_auto['detected_grid_size']}³")
                for axis, info in results_auto['grid_info'].items():
                    print(f"    {axis.upper()}-axis: {info['unique_count']} unique, range {info['range']}")
            
            # Method 2: Force normalized method at 64³
            results_norm64 = evaluate_surface_completeness_normalized(voxels, 64)
            print(f"  Normalized 64³:")
            print(f"    Complete: {results_norm64['is_complete']}")
            print(f"    Flood ratio: {results_norm64['flood_volume_ratio']:.3f}")
            print(f"    Time: {results_norm64['evaluation_time']:.3f}s")
            
            # Method 3: For raw coordinates, try manual discrete conversion
            if "raw.npy" in voxel_path:
                discrete_coords, grid_info = convert_raw_to_discrete(voxels)
                max_grid_size = max(info['size'] for info in grid_info.values())
                
                results_discrete = evaluate_surface_completeness_discrete(discrete_coords, max_grid_size)
                print(f"  Manual discrete {max_grid_size}³:")
                print(f"    Complete: {results_discrete['is_complete']}")
                print(f"    Flood ratio: {results_discrete['flood_volume_ratio']:.3f}")
                print(f"    Time: {results_discrete['evaluation_time']:.3f}s")
                
                # Compare with smaller grid for normalized
                results_norm_small = evaluate_surface_completeness_normalized(voxels, max_grid_size)
                print(f"  Normalized {max_grid_size}³:")
                print(f"    Complete: {results_norm_small['is_complete']}")
                print(f"    Flood ratio: {results_norm_small['flood_volume_ratio']:.3f}")
                print(f"    Time: {results_norm_small['evaluation_time']:.3f}s")
                
                # Analysis
                discrete_complete = results_discrete['is_complete']
                norm_complete_64 = results_norm64['is_complete']
                norm_complete_small = results_norm_small['is_complete']
                
                if discrete_complete != norm_complete_64:
                    print(f"    ⚠️  DIFFERENCE: Discrete vs Norm64: {discrete_complete} vs {norm_complete_64}")
                if discrete_complete != norm_complete_small:
                    print(f"    ⚠️  DIFFERENCE: Discrete vs Norm-same-size: {discrete_complete} vs {norm_complete_small}")
            
        except Exception as e:
            print(f"  Error loading {voxel_path}: {e}")
    
    print(f"\n{'=' * 70}")
    print("Summary:")
    print("- 'discrete' method: Uses exact voxel grid structure, no precision loss")
    print("- 'normalized' method: Re-discretizes coordinates, may introduce gaps")
    print("- Raw coordinates should use discrete method for best accuracy")
    print("- Normalized coordinates may need normalized method due to transformations")


if __name__ == "__main__":
    compare_methods()