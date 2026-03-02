#!/usr/bin/env python3
"""
Batch Surface Completeness Evaluation Script

This script:
1. Iterates through all food items in the NutritionVerse-3D dataset  
2. For each rendered view, evaluates surface completeness on both:
   - Raw voxel coordinates (sam3d_voxels_raw.npy)
   - Normalized voxel coordinates (sam3d_voxels_normalized.npy)
3. Saves detailed results in CSV format
4. Provides statistics on completeness rates and method contradictions
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

# Import our surface completeness evaluation
from surface_completeness_evaluation import (
    evaluate_surface_completeness_auto,
    evaluate_surface_completeness_force_normalized
)


def find_food_items(dataset_path):
    """
    Find all food item directories with SAM3D predictions
    
    Args:
        dataset_path: Path to NutritionVerse-3D dataset
    
    Returns:
        list: List of food item directory paths
    """
    food_items = []
    
    # Look for directories that contain SAM3D predictions
    for item_dir in glob.glob(os.path.join(dataset_path, "*")):
        if os.path.isdir(item_dir):
            sam3d_dir = os.path.join(item_dir, "SAM3D_singleview_prediction")
            if os.path.exists(sam3d_dir) and os.path.isdir(sam3d_dir):
                food_items.append(item_dir)
    
    return sorted(food_items)


def find_sam3d_predictions(food_item_dir):
    """
    Find all SAM3D prediction directories for a food item
    
    Args:
        food_item_dir: Path to food item directory
        
    Returns:
        list: List of prediction directory paths
    """
    sam3d_dir = os.path.join(food_item_dir, "SAM3D_singleview_prediction")
    
    if not os.path.exists(sam3d_dir):
        return []
    
    prediction_dirs = []
    for pred_dir in glob.glob(os.path.join(sam3d_dir, "*")):
        if os.path.isdir(pred_dir):
            # Check if this directory has the required voxel files
            raw_file = os.path.join(pred_dir, "sam3d_voxels_raw.npy")
            norm_file = os.path.join(pred_dir, "sam3d_voxels_normalized.npy")
            
            if os.path.exists(raw_file) and os.path.exists(norm_file):
                prediction_dirs.append(pred_dir)
    
    return sorted(prediction_dirs)


def evaluate_completeness_single_prediction(pred_dir, view_name):
    """
    Evaluate surface completeness for a single prediction directory
    
    Args:
        pred_dir: Path to prediction directory
        view_name: Name of the view (e.g., "000", "001")
    
    Returns:
        dict: Results for both raw and normalized evaluations
    """
    results = {
        'view': view_name,
        'pred_dir': pred_dir
    }
    
    raw_file = os.path.join(pred_dir, "sam3d_voxels_raw.npy")
    norm_file = os.path.join(pred_dir, "sam3d_voxels_normalized.npy")
    
    # Evaluate raw voxels
    try:
        raw_voxels = np.load(raw_file)
        raw_results = evaluate_surface_completeness_auto(raw_voxels, prefer_discrete=True)
        
        results.update({
            'raw_num_voxels': len(raw_voxels),
            'raw_method': raw_results['method'],
            'raw_is_complete': raw_results['is_complete'],
            'raw_flood_ratio': raw_results['flood_volume_ratio'],
            'raw_evaluation_time': raw_results['evaluation_time'],
            'raw_success': True,
            'raw_error': None
        })
        
        if raw_results['method'] == 'discrete':
            results['raw_detected_grid_size'] = raw_results['detected_grid_size']
        else:
            results['raw_detected_grid_size'] = raw_results.get('requested_grid_size', 64)
            
    except Exception as e:
        results.update({
            'raw_num_voxels': 0,
            'raw_method': None,
            'raw_is_complete': None,
            'raw_flood_ratio': None,
            'raw_evaluation_time': None,
            'raw_detected_grid_size': None,
            'raw_success': False,
            'raw_error': str(e)
        })
    
    # Evaluate normalized voxels
    try:
        norm_voxels = np.load(norm_file)
        norm_results = evaluate_surface_completeness_auto(norm_voxels, prefer_discrete=True)
        
        results.update({
            'norm_num_voxels': len(norm_voxels),
            'norm_method': norm_results['method'],
            'norm_is_complete': norm_results['is_complete'],
            'norm_flood_ratio': norm_results['flood_volume_ratio'],
            'norm_evaluation_time': norm_results['evaluation_time'],
            'norm_success': True,
            'norm_error': None
        })
        
        if norm_results['method'] == 'discrete':
            results['norm_detected_grid_size'] = norm_results['detected_grid_size']
        else:
            results['norm_detected_grid_size'] = norm_results.get('requested_grid_size', 64)
            
    except Exception as e:
        results.update({
            'norm_num_voxels': 0,
            'norm_method': None,
            'norm_is_complete': None,
            'norm_flood_ratio': None,
            'norm_evaluation_time': None,
            'norm_detected_grid_size': None,
            'norm_success': False,
            'norm_error': str(e)
        })
    
    # Check for contradictions
    if results['raw_is_complete'] is not None and results['norm_is_complete'] is not None:
        results['contradiction'] = results['raw_is_complete'] != results['norm_is_complete']
    else:
        results['contradiction'] = None
    
    return results


def evaluate_food_item_completeness(food_item_dir, max_views=None):
    """
    Evaluate surface completeness for all predictions of a food item
    
    Args:
        food_item_dir: Path to food item directory
        max_views: Maximum number of views to process (None for all)
    
    Returns:
        list: List of evaluation results for each view
    """
    food_item_name = os.path.basename(food_item_dir)
    print(f"\n{'='*60}")
    print(f"Evaluating: {food_item_name}")
    print(f"{'='*60}")
    
    # Find prediction directories
    pred_dirs = find_sam3d_predictions(food_item_dir)
    
    if not pred_dirs:
        print(f"No SAM3D predictions found for {food_item_name}")
        return []
    
    if max_views:
        pred_dirs = pred_dirs[:max_views]
    
    print(f"Found {len(pred_dirs)} prediction directories")
    
    results = []
    
    for pred_dir in pred_dirs:
        view_name = os.path.basename(pred_dir)
        print(f"Processing view {view_name}...")
        
        result = evaluate_completeness_single_prediction(pred_dir, view_name)
        result['food_item'] = food_item_name
        results.append(result)
        
        # Print summary
        if result['raw_success'] and result['norm_success']:
            raw_complete = result['raw_is_complete']
            norm_complete = result['norm_is_complete']
            contradiction = result['contradiction']
            
            print(f"  Raw: {raw_complete} | Norm: {norm_complete} | Contradiction: {contradiction}")
        else:
            print(f"  Errors - Raw: {'✓' if result['raw_success'] else '✗'} | Norm: {'✓' if result['norm_success'] else '✗'}")
    
    return results


def calculate_statistics(results_df):
    """
    Calculate comprehensive statistics from results
    
    Args:
        results_df: DataFrame with evaluation results
        
    Returns:
        dict: Statistics summary
    """
    stats = {}
    
    # Overall statistics
    total_evaluations = len(results_df)
    successful_evaluations = len(results_df[
        (results_df['raw_success'] == True) & 
        (results_df['norm_success'] == True)
    ])
    
    stats['total_evaluations'] = total_evaluations
    stats['successful_evaluations'] = successful_evaluations
    stats['success_rate'] = successful_evaluations / total_evaluations * 100 if total_evaluations > 0 else 0
    
    if successful_evaluations == 0:
        return stats
    
    # Filter to successful evaluations only
    valid_df = results_df[
        (results_df['raw_success'] == True) & 
        (results_df['norm_success'] == True)
    ]
    
    # Completeness rates
    raw_complete_count = valid_df['raw_is_complete'].sum()
    norm_complete_count = valid_df['norm_is_complete'].sum()
    
    stats['raw_completeness_rate'] = raw_complete_count / len(valid_df) * 100
    stats['norm_completeness_rate'] = norm_complete_count / len(valid_df) * 100
    
    # Contradiction analysis
    contradictions = valid_df['contradiction'].sum()
    stats['contradictions_count'] = contradictions
    stats['contradiction_rate'] = contradictions / len(valid_df) * 100
    
    # Method usage analysis
    stats['raw_method_discrete_rate'] = (valid_df['raw_method'] == 'discrete').sum() / len(valid_df) * 100
    stats['norm_method_discrete_rate'] = (valid_df['norm_method'] == 'discrete').sum() / len(valid_df) * 100
    
    # Per-object statistics
    object_stats = []
    for food_item in valid_df['food_item'].unique():
        item_df = valid_df[valid_df['food_item'] == food_item]
        
        item_stat = {
            'food_item': food_item,
            'total_views': len(item_df),
            'raw_complete_count': item_df['raw_is_complete'].sum(),
            'norm_complete_count': item_df['norm_is_complete'].sum(),
            'contradiction_count': item_df['contradiction'].sum(),
            'raw_complete_rate': item_df['raw_is_complete'].sum() / len(item_df) * 100,
            'norm_complete_rate': item_df['norm_is_complete'].sum() / len(item_df) * 100,
            'contradiction_rate': item_df['contradiction'].sum() / len(item_df) * 100
        }
        object_stats.append(item_stat)
    
    stats['per_object_stats'] = object_stats
    
    return stats


def print_statistics(stats):
    """Print comprehensive statistics"""
    
    print(f"\n{'='*70}")
    print("SURFACE COMPLETENESS EVALUATION STATISTICS")
    print(f"{'='*70}")
    
    print(f"\nOverall Results:")
    print(f"  Total evaluations: {stats['total_evaluations']}")
    print(f"  Successful evaluations: {stats['successful_evaluations']}")
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    
    if stats['successful_evaluations'] == 0:
        print("  No successful evaluations to analyze!")
        return
    
    print(f"\nCompleteness Rates:")
    print(f"  Raw voxels complete: {stats['raw_completeness_rate']:.1f}%")
    print(f"  Normalized voxels complete: {stats['norm_completeness_rate']:.1f}%")
    
    print(f"\nMethod Usage:")
    print(f"  Raw voxels using discrete method: {stats['raw_method_discrete_rate']:.1f}%")
    print(f"  Normalized voxels using discrete method: {stats['norm_method_discrete_rate']:.1f}%")
    
    print(f"\nContradiction Analysis:")
    print(f"  Total contradictions: {stats['contradictions_count']}")
    print(f"  Contradiction rate: {stats['contradiction_rate']:.1f}%")
    
    print(f"\nPer-Object Statistics:")
    print(f"{'Food Item':<40} {'Views':<6} {'Raw%':<6} {'Norm%':<6} {'Contr%':<6}")
    print(f"{'-'*40} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
    
    for obj_stat in stats['per_object_stats']:
        name = obj_stat['food_item'][:38] + '..' if len(obj_stat['food_item']) > 40 else obj_stat['food_item']
        print(f"{name:<40} {obj_stat['total_views']:<6} "
              f"{obj_stat['raw_complete_rate']:5.1f} {obj_stat['norm_complete_rate']:5.1f} "
              f"{obj_stat['contradiction_rate']:5.1f}")


def main():
    parser = argparse.ArgumentParser(description='Batch surface completeness evaluation on NutritionVerse-3D dataset')
    parser.add_argument('--dataset_path', type=str, required=True, 
                       help='Path to NutritionVerse-3D dataset')
    parser.add_argument('--max_objects', type=int, 
                       help='Maximum number of food items to process')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Starting index in the food items list (0-based)')
    parser.add_argument('--max_views', type=int,
                       help='Maximum number of views per food item')
    parser.add_argument('--output_file', type=str,
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    print("Surface Completeness Batch Evaluation")
    print("=" * 60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Max objects: {args.max_objects or 'All'}")
    print(f"Start index: {args.start_index}")
    print(f"Max views per object: {args.max_views or 'All'}")
    
    # Find all food items
    food_items = find_food_items(args.dataset_path)
    
    if not food_items:
        print(f"No food items found in {args.dataset_path}")
        return
    
    print(f"Found {len(food_items)} total food items")
    
    # Apply start index and max objects filtering
    if args.start_index >= len(food_items):
        print(f"Start index {args.start_index} is beyond available items ({len(food_items)})")
        return
    
    food_items = food_items[args.start_index:]
    
    if args.max_objects:
        food_items = food_items[:args.max_objects]
    
    print(f"Processing {len(food_items)} food items (starting from index {args.start_index})")
    
    # Process all food items
    all_results = []
    
    for i, food_item_dir in enumerate(food_items):
        food_item_name = os.path.basename(food_item_dir)
        actual_index = args.start_index + i
        print(f"\nProgress: {i+1}/{len(food_items)} (global index {actual_index}) - {food_item_name}")
        
        try:
            item_results = evaluate_food_item_completeness(food_item_dir, args.max_views)
            all_results.extend(item_results)
            
            # Print progress update
            if (i + 1) % 5 == 0 or i == len(food_items) - 1:
                print(f"\nProgress update: processed {i+1}/{len(food_items)} items, total results: {len(all_results)}")
                
        except Exception as e:
            print(f"Error processing {food_item_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final results
    print(f"\n{'='*70}")
    print("BATCH EVALUATION COMPLETE")
    print(f"{'='*70}")
    
    if all_results:
        final_df = pd.DataFrame(all_results)
        
        # Generate final output filenames
        if args.output_file:
            output_file = args.output_file
            stats_file = args.output_file.replace('.csv', '_statistics.json')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            start_suffix = f"_start{args.start_index}" if args.start_index > 0 else ""
            max_suffix = f"_max{args.max_objects}" if args.max_objects else ""
            output_file = f"surface_completeness_evaluation{start_suffix}{max_suffix}_{timestamp}.csv"
            stats_file = f"surface_completeness_statistics{start_suffix}{max_suffix}_{timestamp}.json"
        
        # Save final CSV results
        final_df.to_csv(output_file, index=False)
        print(f"Final results saved to: {output_file}")
        
        # Calculate and print final comprehensive statistics
        final_stats = calculate_statistics(final_df)
        print_statistics(final_stats)
        
        # Save statistics to JSON file
        import json
        import numpy as np
        
        def convert_numpy_types(obj):
            """Convert NumPy types to JSON-serializable Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert statistics to JSON-serializable format
        json_compatible_stats = convert_numpy_types(final_stats)
        
        with open(stats_file, 'w') as f:
            json.dump(json_compatible_stats, f, indent=2)
        print(f"Final statistics saved to: {stats_file}")
        
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()