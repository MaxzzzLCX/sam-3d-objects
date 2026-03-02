#!/usr/bin/env python3
"""
Batch Evaluation Script for SAM3D Pipeline

This script:
1. Iterates through all food items in the NutritionVerse-3D dataset
2. Runs SAM3D prediction for each rendered view
3. Calculates Chamfer distance between predictions and ground truth
4. Saves evaluation results in a comprehensive CSV report
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

# Import our refactored modules
from sam3d_single_view_prediction import run_sam3d_prediction
from chamfer_distance_evaluation import calculate_chamfer_distance_for_files
from surface_completeness_evaluation import evaluate_surface_completeness_for_files, evaluate_surface_completeness


def find_food_items(dataset_path):
    """
    Find all food item directories with normalized meshes
    
    Args:
        dataset_path: Path to NutritionVerse-3D dataset
    
    Returns:
        list: List of food item directory paths
    """
    food_items = []
    
    # Look for directories that contain normalized_mesh.obj
    for item_dir in glob.glob(os.path.join(dataset_path, "*")):
        if os.path.isdir(item_dir):
            normalized_mesh = os.path.join(item_dir, "normalized_mesh.obj")
            if os.path.exists(normalized_mesh):
                food_items.append(item_dir)
    
    return sorted(food_items)


def find_rendered_images(food_item_dir):
    """
    Find all rendered images for a food item
    
    Args:
        food_item_dir: Path to food item directory
    
    Returns:
        list: List of rendered image paths
    """
    rendered_images = []
    
    # Look for rendered-test-example directory with render_*.png files
    render_dir = os.path.join(food_item_dir, "rendered-test-example")
    if os.path.exists(render_dir):
        render_files = glob.glob(os.path.join(render_dir, "render_*.png"))
        rendered_images.extend(sorted(render_files))
    
    return rendered_images


def evaluate_single_item(food_item_dir, config_path, max_views=None, sampling_strategy='fixed_n'):
    """
    Evaluate a single food item with all its rendered views
    
    Args:
        food_item_dir: Path to food item directory
        config_path: Path to SAM3D config file
        max_views: Maximum number of views to process (None for all)
        sampling_strategy: 'fixed_n' (default) or 'adaptive' sampling method
    
    Returns:
        list: List of evaluation results for each view
    """
    food_item_name = os.path.basename(food_item_dir)
    print(f"\n{'='*60}")
    print(f"Evaluating: {food_item_name}")
    print(f"{'='*60}")
    
    # Find rendered images
    rendered_images = find_rendered_images(food_item_dir)
    
    if not rendered_images:
        print(f"No rendered images found for {food_item_name}")
        return []
    
    if max_views:
        rendered_images = rendered_images[:max_views]
    
    print(f"Found {len(rendered_images)} rendered views")
    
    results = []
    
    for i, image_path in enumerate(rendered_images):
        view_name = Path(image_path).stem  # e.g., "render_000"
        print(f"\nProcessing view {i+1}/{len(rendered_images)}: {view_name}")
        
        # Run SAM3D prediction
        print("Running SAM3D prediction...")
        normalized_voxels, output_dir = run_sam3d_prediction(
            food_item_dir, image_path, config_path, sampling_strategy
        )
        
        if normalized_voxels is None:
            print(f"SAM3D prediction failed for {view_name}")
            results.append({
                'food_item': food_item_name,
                'view': view_name,
                'sam3d_success': False,
                'num_voxels': 0,
                'chamfer_distance': None,
                'surface_complete': None,
                'error': 'SAM3D prediction failed'
            })
            continue
        
        # Calculate Chamfer distance
        print("Calculating Chamfer distance...")
        gt_path = os.path.join(output_dir, "ground_truth_surface_points.ply")
        pred_path = os.path.join(output_dir, "sam3d_voxels_pointcloud.ply")
        
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"Point cloud files not found for {view_name}")
            results.append({
                'food_item': food_item_name,
                'view': view_name,
                'sam3d_success': True,
                'num_voxels': len(normalized_voxels),
                'chamfer_distance': None,
                'surface_complete': None,
                'error': 'Point cloud files not found'
            })
            continue
        
        chamfer_results = calculate_chamfer_distance_for_files(gt_path, pred_path, output_dir)
        
        if chamfer_results is None:
            print(f"Chamfer distance calculation failed for {view_name}")
            results.append({
                'food_item': food_item_name,
                'view': view_name,
                'sam3d_success': True,
                'num_voxels': len(normalized_voxels),
                'chamfer_distance': None,
                'surface_complete': None,
                'error': 'Chamfer distance calculation failed'
            })
            continue
        
        # Calculate surface completeness using raw voxel coordinates (more accurate)
        print("Evaluating surface completeness...")
        try:
            # Look for raw voxels file first (preferred for accuracy)
            raw_voxels_file = os.path.join(output_dir, "sam3d_voxels_raw.npy")
            
            if os.path.exists(raw_voxels_file):
                raw_voxels = np.load(raw_voxels_file)
                completeness_results = evaluate_surface_completeness(raw_voxels)
                print(f"  Using raw voxels for surface completeness ({len(raw_voxels)} voxels)")
            else:
                # Fallback to normalized voxels if raw not available
                completeness_results = evaluate_surface_completeness(normalized_voxels, grid_size=64)
                print(f"  Using normalized voxels for surface completeness ({len(normalized_voxels)} voxels)")
            
            surface_complete = completeness_results['is_complete']
            flood_volume_ratio = completeness_results['flood_volume_ratio']
            completeness_time = completeness_results['evaluation_time']
            
            # Save completeness results to output directory
            import json
            completeness_file = os.path.join(output_dir, 'surface_completeness_results.json')
            with open(completeness_file, 'w') as f:
                json.dump(completeness_results, f, indent=2)
                
        except Exception as e:
            print(f"Surface completeness evaluation failed for {view_name}: {e}")
            surface_complete = None
            flood_volume_ratio = None
            completeness_time = None
        
        # Success - record all results
        result = {
            'food_item': food_item_name,
            'view': view_name,
            'sam3d_success': True,
            'num_voxels': len(normalized_voxels),
            'chamfer_distance': chamfer_results['chamfer_distance'],  # Keep for backward compatibility
            'bidirectional_chamfer_distance': chamfer_results.get('bidirectional_chamfer_distance', chamfer_results['chamfer_distance']),
            'unidirectional_chamfer_distance': chamfer_results.get('unidirectional_chamfer_distance', None),
            'gt_points_count': chamfer_results['gt_points_count'],
            'pred_points_count': chamfer_results['pred_points_count'],
            'calculation_time': chamfer_results['calculation_time_seconds'],
            'surface_complete': surface_complete,
            'flood_volume_ratio': flood_volume_ratio,
            'completeness_evaluation_time': completeness_time,
            'error': None,
            'output_dir': output_dir
        }
        
        results.append(result)
        
        print(f"✓ Success: Bidirectional Chamfer = {chamfer_results.get('bidirectional_chamfer_distance', chamfer_results['chamfer_distance']):.6f}, Unidirectional Chamfer = {chamfer_results.get('unidirectional_chamfer_distance', 'N/A'):.6f if chamfer_results.get('unidirectional_chamfer_distance') is not None else 'N/A'}, Surface complete = {surface_complete}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation of SAM3D pipeline on NutritionVerse-3D dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to NutritionVerse-3D dataset')
    parser.add_argument('--config_path', type=str, 
                       default='/scratch/cl927/sam-3d-objects/checkpoints/hf/pipeline.yaml',
                       help='Path to SAM3D config file')
    parser.add_argument('--max_items', type=int, help='Maximum number of food items to process')
    parser.add_argument('--max_views', type=int, help='Maximum number of views per food item')
    parser.add_argument('--sampling_strategy', type=str, choices=['fixed_n', 'adaptive'], 
                       default='fixed_n', help='Point sampling strategy: fixed_n (default) or adaptive')
    parser.add_argument('--output_file', type=str, help='Output CSV file path')
    parser.add_argument('--resume_from', type=str, help='Food item to resume from (by name)')
    parser.add_argument('--start_index', type=int, help='Starting index in the food items list (0-based)')
    
    args = parser.parse_args()
    
    print("SAM3D Batch Evaluation Pipeline")
    print("=" * 60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"SAM3D config: {args.config_path}")
    print(f"Sampling strategy: {args.sampling_strategy}")
    print(f"Max items: {args.max_items or 'All'}")
    print(f"Max views per item: {args.max_views or 'All'}")
    
    # Check if SAM3D config exists
    if not os.path.exists(args.config_path):
        print(f"Error: SAM3D config not found at {args.config_path}")
        return
    
    # Find all food items
    food_items = find_food_items(args.dataset_path)
    print(f"\nFound {len(food_items)} food items with normalized meshes")
    
    if not food_items:
        print("No food items found! Check dataset path and ensure normalized meshes exist.")
        return
    
    # Apply resume from filter or start index
    if args.resume_from:
        resume_index = None
        for i, item_dir in enumerate(food_items):
            if args.resume_from in os.path.basename(item_dir):
                resume_index = i
                break
        if resume_index is not None:
            food_items = food_items[resume_index:]
            print(f"Resuming from item {resume_index}: {os.path.basename(food_items[0])}")
        else:
            print(f"Resume item '{args.resume_from}' not found")
            return
    elif args.start_index is not None:
        if args.start_index >= len(food_items):
            print(f"Error: Start index {args.start_index} is >= total items ({len(food_items)})")
            return
        food_items = food_items[args.start_index:]
        print(f"Starting from index {args.start_index}: {os.path.basename(food_items[0])}")
    
    # Apply max items limit
    if args.max_items:
        food_items = food_items[:args.max_items]
        print(f"Limited to {len(food_items)} food items")
    
    # Process each food item
    all_results = []
    
    for i, food_item_dir in enumerate(food_items):
        food_item_name = os.path.basename(food_item_dir)
        print(f"\nProgress: {i+1}/{len(food_items)} - {food_item_name}")
        
        try:
            item_results = evaluate_single_item(food_item_dir, args.config_path, args.max_views, args.sampling_strategy)
            all_results.extend(item_results)
            
            # Save intermediate results every 5 items
            if (i + 1) % 5 == 0 or i == len(food_items) - 1:
                df = pd.DataFrame(all_results)
                
                # Generate output filename if not provided
                if args.output_file:
                    output_file = args.output_file
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"sam3d_batch_evaluation_{timestamp}.csv"
                
                df.to_csv(output_file, index=False)
                print(f"\nIntermediate results saved to: {output_file}")
                
                # Print summary statistics
                successful = df[df['sam3d_success'] == True]
                valid_chamfer = successful[successful['chamfer_distance'].notna()]
                valid_completeness = successful[successful['surface_complete'].notna()]
                
                print(f"\nCurrent Statistics:")
                print(f"  Total evaluations: {len(df)}")
                print(f"  SAM3D successes: {len(successful)} ({len(successful)/len(df)*100:.1f}%)")
                print(f"  Valid Chamfer distances: {len(valid_chamfer)} ({len(valid_chamfer)/len(df)*100:.1f}%)")
                print(f"  Valid surface completeness: {len(valid_completeness)} ({len(valid_completeness)/len(df)*100:.1f}%)")
                
                if len(valid_chamfer) > 0:
                    mean_chamfer = valid_chamfer['chamfer_distance'].mean()
                    median_chamfer = valid_chamfer['chamfer_distance'].median()
                    std_chamfer = valid_chamfer['chamfer_distance'].std()
                    print(f"  Mean Chamfer distance: {mean_chamfer:.6f} ± {std_chamfer:.6f}")
                    print(f"  Median Chamfer distance: {median_chamfer:.6f}")
                    print(f"  Min Chamfer distance: {valid_chamfer['chamfer_distance'].min():.6f}")
                    print(f"  Max Chamfer distance: {valid_chamfer['chamfer_distance'].max():.6f}")
                
                if len(valid_completeness) > 0:
                    complete_count = valid_completeness['surface_complete'].sum()
                    completion_rate = complete_count / len(valid_completeness) * 100
                    mean_flood_ratio = valid_completeness['flood_volume_ratio'].mean()
                    print(f"  Surface completion rate: {complete_count}/{len(valid_completeness)} ({completion_rate:.1f}%)")
                    print(f"  Mean flood volume ratio: {mean_flood_ratio:.3f}")
        
        except Exception as e:
            print(f"Error processing {food_item_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Record the error
            all_results.append({
                'food_item': food_item_name,
                'view': 'unknown',
                'sam3d_success': False,
                'num_voxels': 0,
                'chamfer_distance': None,
                'surface_complete': None,
                'error': str(e)
            })
    
    # Final results
    print("\n" + "=" * 60)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 60)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Generate final output filename if not provided
        if args.output_file:
            output_file = args.output_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"sam3d_batch_evaluation_{timestamp}.csv"
        
        df.to_csv(output_file, index=False)
        print(f"Final results saved to: {output_file}")
        
        # Print final summary statistics
        successful = df[df['sam3d_success'] == True]
        valid_chamfer = successful[successful['chamfer_distance'].notna()]
        
        print(f"\nFinal Statistics:")
        print(f"  Total evaluations: {len(df)}")
        print(f"  SAM3D successes: {len(successful)} ({len(successful)/len(df)*100:.1f}%)")
        print(f"  Valid Chamfer distances: {len(valid_chamfer)} ({len(valid_chamfer)/len(df)*100:.1f}%)")
        
        if len(valid_chamfer) > 0:
            mean_chamfer = valid_chamfer['chamfer_distance'].mean()
            median_chamfer = valid_chamfer['chamfer_distance'].median()
            std_chamfer = valid_chamfer['chamfer_distance'].std()
            print(f"  Mean Chamfer distance: {mean_chamfer:.6f} ± {std_chamfer:.6f}")
            print(f"  Median Chamfer distance: {median_chamfer:.6f}")
            print(f"  Min Chamfer distance: {valid_chamfer['chamfer_distance'].min():.6f}")
            print(f"  Max Chamfer distance: {valid_chamfer['chamfer_distance'].max():.6f}")
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()