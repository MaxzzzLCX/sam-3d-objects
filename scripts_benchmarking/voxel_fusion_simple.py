#!/usr/bin/env python3
"""
Simple Voxel Fusion Script

This script takes two normalized voxel representations of the same object
and performs simple fusion by combining them to assess alignment quality.
"""

import numpy as np
import argparse
import json
import os
from pathlib import Path
import time


def load_normalized_voxels(voxel_path):
    """
    Load normalized voxel representation
    
    Args:
        voxel_path: Path to sam3d_voxels_normalized.npy file
        
    Returns:
        np.ndarray: Voxel coordinates (N, 3) or None if failed
    """
    try:
        voxels = np.load(voxel_path)
        print(f"Loaded voxels: {voxels.shape}")
        return voxels
    except Exception as e:
        print(f"Error loading voxels from {voxel_path}: {e}")
        return None


def simple_voxel_fusion(voxels1, voxels2):
    """
    Perform simple fusion by combining two sets of voxels
    
    Args:
        voxels1: First set of voxel coordinates (N1, 3)
        voxels2: Second set of voxel coordinates (N2, 3)
        
    Returns:
        dict: Fusion results containing combined voxels and statistics
    """
    print("Performing simple voxel fusion...")
    
    # Combine voxel sets
    all_voxels = np.vstack([voxels1, voxels2])
    
    # Remove duplicate voxels (same coordinates)
    unique_voxels = np.unique(all_voxels, axis=0)
    
    # Calculate statistics
    original_count1 = len(voxels1)
    original_count2 = len(voxels2)
    total_combined = len(all_voxels)
    unique_count = len(unique_voxels)
    overlap_count = total_combined - unique_count
    
    # Calculate overlap ratio
    overlap_ratio = overlap_count / min(original_count1, original_count2) * 100
    
    # Calculate bounds for each set and combined
    bounds1 = {
        'min': np.min(voxels1, axis=0),
        'max': np.max(voxels1, axis=0),
        'center': np.mean(voxels1, axis=0)
    }
    
    bounds2 = {
        'min': np.min(voxels2, axis=0),
        'max': np.max(voxels2, axis=0),
        'center': np.mean(voxels2, axis=0)
    }
    
    bounds_combined = {
        'min': np.min(unique_voxels, axis=0),
        'max': np.max(unique_voxels, axis=0),
        'center': np.mean(unique_voxels, axis=0)
    }
    
    # Calculate center distance (misalignment indicator)
    center_distance = np.linalg.norm(bounds1['center'] - bounds2['center'])
    
    return {
        'voxels1': voxels1,
        'voxels2': voxels2,
        'combined_voxels': all_voxels,
        'unique_voxels': unique_voxels,
        'statistics': {
            'original_count1': original_count1,
            'original_count2': original_count2,
            'total_combined': total_combined,
            'unique_count': unique_count,
            'overlap_count': overlap_count,
            'overlap_ratio': overlap_ratio,
            'center_distance': center_distance,
            'bounds1': bounds1,
            'bounds2': bounds2,
            'bounds_combined': bounds_combined
        }
    }


def save_fusion_ply(fusion_results, output_dir):
    """
    Save fusion results as a colored PLY file
    
    Args:
        fusion_results: Results from simple_voxel_fusion()
        output_dir: Directory to save PLY file
    """
    voxels1 = fusion_results['voxels1']
    voxels2 = fusion_results['voxels2']
    
    # Create sets for fast overlap detection
    voxels1_set = set(map(tuple, voxels1))
    voxels2_set = set(map(tuple, voxels2))
    
    # Find overlapping and unique voxels
    overlap_set = voxels1_set.intersection(voxels2_set)
    only_1_set = voxels1_set - overlap_set
    only_2_set = voxels2_set - overlap_set
    
    # Convert back to arrays
    overlap_voxels = np.array(list(overlap_set)) if overlap_set else np.empty((0, 3))
    only_1_voxels = np.array(list(only_1_set)) if only_1_set else np.empty((0, 3))
    only_2_voxels = np.array(list(only_2_set)) if only_2_set else np.empty((0, 3))
    
    # Combine all voxels
    all_voxels = []
    all_colors = []
    
    # Add voxels from view 1 only (red)
    if len(only_1_voxels) > 0:
        all_voxels.append(only_1_voxels)
        all_colors.extend([[255, 0, 0]] * len(only_1_voxels))  # Red
    
    # Add voxels from view 2 only (blue)
    if len(only_2_voxels) > 0:
        all_voxels.append(only_2_voxels)
        all_colors.extend([[0, 0, 255]] * len(only_2_voxels))  # Blue
    
    # Add overlapping voxels (purple)
    if len(overlap_voxels) > 0:
        all_voxels.append(overlap_voxels)
        all_colors.extend([[128, 0, 128]] * len(overlap_voxels))  # Purple
    
    if not all_voxels:
        print("Warning: No voxels to save")
        return None
    
    # Combine all voxels and colors
    combined_voxels = np.vstack(all_voxels)
    combined_colors = np.array(all_colors)
    
    # Create PLY file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_file = output_dir / 'voxel_fusion_colored.ply'
    
    # Write PLY header
    with open(ply_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(combined_voxels)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertex data
        for voxel, color in zip(combined_voxels, combined_colors):
            f.write(f"{voxel[0]:.6f} {voxel[1]:.6f} {voxel[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
    
    print(f"Colored PLY saved to: {ply_file}")
    
    # Print color legend
    print(f"\nColor Legend:")
    print(f"  Red: View 1 only ({len(only_1_voxels)} voxels)")
    print(f"  Blue: View 2 only ({len(only_2_voxels)} voxels)")
    print(f"  Purple: Overlapping ({len(overlap_voxels)} voxels)")
    
    return str(ply_file)


def print_fusion_statistics(fusion_results):
    """Print comprehensive fusion statistics"""
    stats = fusion_results['statistics']
    
    print(f"\n{'='*60}")
    print("SIMPLE VOXEL FUSION RESULTS")
    print(f"{'='*60}")
    
    print(f"Original voxel counts:")
    print(f"  View 1: {stats['original_count1']:,} voxels")
    print(f"  View 2: {stats['original_count2']:,} voxels")
    print(f"  Total combined: {stats['total_combined']:,} voxels")
    print(f"  Unique voxels: {stats['unique_count']:,} voxels")
    
    print(f"\nOverlap analysis:")
    print(f"  Overlapping voxels: {stats['overlap_count']:,}")
    print(f"  Overlap ratio: {stats['overlap_ratio']:.1f}% (relative to smaller set)")
    
    print(f"\nAlignment analysis:")
    print(f"  Center distance: {stats['center_distance']:.6f}")
    print(f"  View 1 center: [{stats['bounds1']['center'][0]:.4f}, {stats['bounds1']['center'][1]:.4f}, {stats['bounds1']['center'][2]:.4f}]")
    print(f"  View 2 center: [{stats['bounds2']['center'][0]:.4f}, {stats['bounds2']['center'][1]:.4f}, {stats['bounds2']['center'][2]:.4f}]")
    
    print(f"\nBounding boxes:")
    for i, (key, bounds) in enumerate([('bounds1', stats['bounds1']), ('bounds2', stats['bounds2'])], 1):
        print(f"  View {i} bounds:")
        print(f"    Min: [{bounds['min'][0]:.4f}, {bounds['min'][1]:.4f}, {bounds['min'][2]:.4f}]")
        print(f"    Max: [{bounds['max'][0]:.4f}, {bounds['max'][1]:.4f}, {bounds['max'][2]:.4f}]")
        size = bounds['max'] - bounds['min']
        print(f"    Size: [{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}]")


def find_same_object_views(dataset_path, food_item):
    """
    Find different views of the same food item
    
    Args:
        dataset_path: Path to dataset
        food_item: Name of food item
        
    Returns:
        list: List of view directories containing voxel predictions
    """
    food_item_path = Path(dataset_path) / food_item
    
    if not food_item_path.exists():
        print(f"Food item not found: {food_item_path}")
        return []
    
    # Look for SAM3D prediction directories
    pred_dirs = []
    sam3d_dir = food_item_path / 'SAM3D_singleview_prediction'
    
    if sam3d_dir.exists():
        for view_dir in sam3d_dir.iterdir():
            if view_dir.is_dir():
                voxel_file = view_dir / 'sam3d_voxels_normalized.npy'
                if voxel_file.exists():
                    pred_dirs.append(str(view_dir))
    
    return pred_dirs


def main():
    parser = argparse.ArgumentParser(description='Simple voxel fusion for two views of the same object')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--food_item', type=str, required=True, help='Food item name (e.g., id-101-steak-piece-28g)')
    parser.add_argument('--view1', type=str, help='First view directory (e.g., 000)')
    parser.add_argument('--view2', type=str, help='Second view directory (e.g., 001)')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--no_ply', action='store_true', help='Disable PLY file generation')
    
    args = parser.parse_args()
    
    print(f"Simple Voxel Fusion")
    print(f"Dataset: {args.dataset_path}")
    print(f"Food item: {args.food_item}")
    
    # Find available views
    available_views = find_same_object_views(args.dataset_path, args.food_item)
    
    if len(available_views) < 2:
        print(f"Error: Need at least 2 views, found {len(available_views)}")
        if available_views:
            print("Available views:", available_views)
        return
    
    print(f"Found {len(available_views)} available views")
    
    # Select views
    if args.view1 and args.view2:
        view1_path = Path(args.dataset_path) / args.food_item / 'SAM3D_singleview_prediction' / args.view1
        view2_path = Path(args.dataset_path) / args.food_item / 'SAM3D_singleview_prediction' / args.view2
        
        if not view1_path.exists() or not view2_path.exists():
            print(f"Error: Specified views not found")
            return
            
        voxel1_path = view1_path / 'sam3d_voxels_normalized.npy'
        voxel2_path = view2_path / 'sam3d_voxels_normalized.npy'
    else:
        # Use first two available views
        voxel1_path = Path(available_views[0]) / 'sam3d_voxels_normalized.npy'
        voxel2_path = Path(available_views[1]) / 'sam3d_voxels_normalized.npy'
        print(f"Using views: {Path(available_views[0]).name} and {Path(available_views[1]).name}")
    
    # Load voxels
    print(f"\nLoading voxels...")
    voxels1 = load_normalized_voxels(voxel1_path)
    voxels2 = load_normalized_voxels(voxel2_path)
    
    if voxels1 is None or voxels2 is None:
        print("Error: Failed to load voxels")
        return
    
    # Perform fusion
    start_time = time.time()
    fusion_results = simple_voxel_fusion(voxels1, voxels2)
    fusion_time = time.time() - start_time
    
    fusion_results['statistics']['fusion_time_seconds'] = fusion_time
    
    # Print statistics
    print_fusion_statistics(fusion_results)
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('voxel_fusion_results') / args.food_item
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_file = output_dir / 'fusion_results.json'
    
    # Prepare JSON-serializable results
    json_results = {
        'food_item': args.food_item,
        'view1_path': str(voxel1_path),
        'view2_path': str(voxel2_path),
        'fusion_time_seconds': fusion_time,
        'statistics': {}
    }
    
    # Convert numpy types for JSON serialization
    for key, value in fusion_results['statistics'].items():
        if isinstance(value, np.ndarray):
            json_results['statistics'][key] = value.tolist()
        elif isinstance(value, dict):
            json_results['statistics'][key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    json_results['statistics'][key][k] = v.tolist()
                else:
                    json_results['statistics'][key][k] = float(v) if isinstance(v, np.floating) else v
        else:
            json_results['statistics'][key] = float(value) if isinstance(value, np.floating) else value
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create PLY visualization
    if not args.no_ply:
        print("\nGenerating colored PLY file...")
        ply_file = save_fusion_ply(fusion_results, output_dir)
        if ply_file:
            json_results['ply_file'] = ply_file
    
    print(f"\nFusion completed in {fusion_time:.2f} seconds")


if __name__ == "__main__":
    main()