#!/usr/bin/env python3
"""
Test batch evaluation surface completeness with raw vs normalized voxels
"""

import os
import numpy as np
import sys
sys.path.append('/scratch/cl927/sam-3d-objects/scripts_benchmarking')

from surface_completeness_evaluation import evaluate_surface_completeness


def test_batch_evaluation_logic():
    """Test the logic used in batch evaluation for surface completeness"""
    
    print("Testing Batch Evaluation Surface Completeness Logic")
    print("=" * 60)
    
    # Simulate the batch evaluation directory structure
    output_dir = "/scratch/cl927/nutritionverse-3d-new/id-11-red-apple-145g/SAM3D_singleview_prediction/000"
    
    print(f"Testing with output directory: {output_dir}")
    print()
    
    # Test the batch evaluation logic
    raw_voxels_file = os.path.join(output_dir, "sam3d_voxels_raw.npy")
    
    if os.path.exists(raw_voxels_file):
        raw_voxels = np.load(raw_voxels_file)
        completeness_results = evaluate_surface_completeness(raw_voxels)
        print(f"✓ Using raw voxels for surface completeness ({len(raw_voxels)} voxels)")
        print(f"  Method: {completeness_results['method']}")
        print(f"  Complete: {completeness_results['is_complete']}")
        print(f"  Flood ratio: {completeness_results['flood_volume_ratio']:.3f}")
        if completeness_results['method'] == 'discrete':
            print(f"  Grid size: {completeness_results['detected_grid_size']}")
    else:
        print("✗ Raw voxels file not found - would fall back to normalized")
    
    print()
    
    # Also show what would happen with normalized (fallback)
    normalized_file = os.path.join(output_dir, "sam3d_voxels_normalized.npy")
    if os.path.exists(normalized_file):
        normalized_voxels = np.load(normalized_file)
        norm_results = evaluate_surface_completeness(normalized_voxels, grid_size=64)
        print(f"Fallback - normalized voxels result ({len(normalized_voxels)} voxels):")
        print(f"  Method: {norm_results['method']}")
        print(f"  Complete: {norm_results['is_complete']}")
        print(f"  Flood ratio: {norm_results['flood_volume_ratio']:.3f}")
    
    print()
    print("Summary:")
    print("- Batch evaluation will prefer raw voxels (more accurate)")
    print("- Falls back to normalized voxels if raw not available")
    print("- Auto-detection chooses discrete vs normalized method")


if __name__ == "__main__":
    test_batch_evaluation_logic()