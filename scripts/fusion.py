#!/usr/bin/env python3
"""
Simple fusion of aligned SAM3D occupancy grids
"""

import numpy as np
import trimesh
import os
from pathlib import Path

def simple_average_fusion(grid1, grid2, method="arithmetic"):
    """
    Simple fusion by averaging two aligned occupancy grids
    
    Args:
        grid1: First occupancy grid (64x64x64) with probability logits
        grid2: Second occupancy grid (64x64x64) with probability logits  
        method: "arithmetic", "geometric", or "min_entropy" averaging
    
    Returns:
        fused_grid: Averaged occupancy grid
    """
    if method == "arithmetic":
        # Simple arithmetic mean
        fused_grid = (grid1 + grid2) / 2.0
    elif method == "geometric":
        # Geometric mean (for logits, this is more principled)
        # Convert logits to probabilities, geometric mean, back to logits
        prob1 = 1.0 / (1.0 + np.exp(-grid1))  # sigmoid
        prob2 = 1.0 / (1.0 + np.exp(-grid2))  # sigmoid
        fused_prob = np.sqrt(prob1 * prob2)    # geometric mean
        # Back to logits (avoid division by zero)
        fused_prob = np.clip(fused_prob, 1e-7, 1-1e-7)
        fused_grid = np.log(fused_prob / (1.0 - fused_prob))
    elif method == "min_entropy":
        # Minimum entropy fusion: select voxel value from view with lower entropy (higher confidence)
        fused_grid = minimum_entropy_fusion(grid1, grid2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return fused_grid

def minimum_entropy_fusion(grid1, grid2):
    """
    Minimum entropy fusion: for each voxel, select the value from the view with lower entropy
    
    Args:
        grid1: First occupancy grid (64x64x64) with probability logits
        grid2: Second occupancy grid (64x64x64) with probability logits
    
    Returns:
        fused_grid: Grid with values from the view with minimum entropy per voxel
    """
    print("Performing minimum entropy fusion...")
    
    # Convert logits to probabilities
    prob1 = 1.0 / (1.0 + np.exp(-grid1))  # sigmoid
    prob2 = 1.0 / (1.0 + np.exp(-grid2))  # sigmoid
    
    # Clip probabilities to avoid log(0)
    prob1_clipped = np.clip(prob1, 1e-7, 1-1e-7)
    prob2_clipped = np.clip(prob2, 1e-7, 1-1e-7)
    
    # Calculate entropy: H = -p*log(p) - (1-p)*log(1-p)
    entropy1 = -(prob1_clipped * np.log(prob1_clipped) + (1-prob1_clipped) * np.log(1-prob1_clipped))
    entropy2 = -(prob2_clipped * np.log(prob2_clipped) + (1-prob2_clipped) * np.log(1-prob2_clipped))
    
    # Select grid1 values where entropy1 < entropy2, otherwise grid2
    use_grid1 = entropy1 < entropy2
    fused_grid = np.where(use_grid1, grid1, grid2)
    
    # Print some statistics
    total_voxels = grid1.size
    grid1_selected = np.sum(use_grid1)
    grid2_selected = total_voxels - grid1_selected
    
    print(f"  Selected from view 1: {grid1_selected:,} voxels ({100*grid1_selected/total_voxels:.1f}%)")
    print(f"  Selected from view 2: {grid2_selected:,} voxels ({100*grid2_selected/total_voxels:.1f}%)")
    print(f"  Average entropy - View 1: {entropy1.mean():.4f}, View 2: {entropy2.mean():.4f}")
    print(f"  Selected entropy range: [{entropy1[use_grid1].mean():.4f}, {entropy2[~use_grid1].mean():.4f}]")
    
    return fused_grid

def save_aligned_grids(view1_grid, view2_grid, output_file="aligned_grids.npz"):
    """Save aligned occupancy grids for fusion"""
    np.savez(output_file,
             view1_grid=view1_grid,
             view2_grid=view2_grid)
    print(f"Saved aligned grids to {output_file}")

def load_aligned_grids(input_file="aligned_grids.npz"):
    """Load aligned occupancy grids"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Aligned grids file not found: {input_file}")
    
    data = np.load(input_file)
    view1_grid = data['view1_grid']
    view2_grid = data['view2_grid']
    
    print(f"Loaded aligned grids from {input_file}")
    print(f"  View1 grid: shape={view1_grid.shape}, range=[{view1_grid.min():.6f}, {view1_grid.max():.6f}]")
    print(f"  View2 grid: shape={view2_grid.shape}, range=[{view2_grid.min():.6f}, {view2_grid.max():.6f}]")
    
    return view1_grid, view2_grid

def visualize_fusion_result(fused_grid, threshold=0.0, output_file="fusion_result.ply"):
    """Convert fused grid to point cloud for visualization"""
    
    # Threshold at logit > 0 (probability > 0.5)
    occupied_coords = np.array(np.where(fused_grid > threshold)).T
    
    if len(occupied_coords) == 0:
        print("Warning: No occupied voxels in fusion result!")
        return
    
    # Convert to normalized coordinates [-0.5, 0.5]
    points = (occupied_coords / 63.0) - 0.5
    
    # Color by probability/confidence
    logits = fused_grid[occupied_coords[:, 0], occupied_coords[:, 1], occupied_coords[:, 2]]
    probabilities = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    
    # Map probabilities to colors (blue=low confidence, red=high confidence)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = (probabilities * 255).astype(np.uint8)  # Red channel
    colors[:, 2] = ((1 - probabilities) * 255).astype(np.uint8)  # Blue channel
    colors = colors.astype(np.uint8)
    
    # Save point cloud
    pc = trimesh.PointCloud(points, colors=colors)
    pc.export(output_file)
    
    print(f"Saved fusion result: {output_file}")
    print(f"  {len(points)} occupied voxels")
    print(f"  Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    print(f"  Colors: blue=low confidence, red=high confidence")

def analyze_fusion_quality(grid1, grid2, fused_grid):
    """Analyze the quality of fusion results"""
    
    print("\n=== Fusion Quality Analysis ===")
    
    # Count occupied voxels
    occ1 = np.sum(grid1 > 0)
    occ2 = np.sum(grid2 > 0) 
    occ_fused = np.sum(fused_grid > 0)
    
    print(f"Occupied voxels:")
    print(f"  View 1: {occ1}")
    print(f"  View 2: {occ2}")
    print(f"  Fused:  {occ_fused}")
    print(f"  Union (expected): {np.sum((grid1 > 0) | (grid2 > 0))}")
    
    # Overlap analysis
    both_occupied = np.sum((grid1 > 0) & (grid2 > 0))
    only_view1 = np.sum((grid1 > 0) & (grid2 <= 0))
    only_view2 = np.sum((grid1 <= 0) & (grid2 > 0))
    
    print(f"\nOverlap analysis:")
    print(f"  Both views occupied: {both_occupied}")
    print(f"  Only view 1: {only_view1}")
    print(f"  Only view 2: {only_view2}")
    print(f"  Overlap ratio: {both_occupied / max(occ1, occ2):.3f}")
    
    # Value distribution
    print(f"\nValue distributions:")
    print(f"  View 1 logits: mean={grid1.mean():.3f}, std={grid1.std():.3f}")
    print(f"  View 2 logits: mean={grid2.mean():.3f}, std={grid2.std():.3f}")
    print(f"  Fused logits:  mean={fused_grid.mean():.3f}, std={fused_grid.std():.3f}")

def run_simple_fusion(aligned_grids_file="aligned_grids.npz", fusion_method="arithmetic"):
    """
    Main fusion pipeline using simple averaging
    
    Args:
        aligned_grids_file: Path to aligned grids .npz file
        fusion_method: "arithmetic" or "geometric" averaging
    """
    
    print("=== Simple Average Fusion ===")
    print(f"Method: {fusion_method}")
    
    # Load aligned grids
    try:
        view1_grid, view2_grid = load_aligned_grids(aligned_grids_file)
    except FileNotFoundError:
        print(f"Error: {aligned_grids_file} not found!")
        print("Run alignment first to generate aligned grids.")
        return
    
    # Perform fusion
    print(f"\nPerforming {fusion_method} fusion...")
    fused_grid = simple_average_fusion(view1_grid, view2_grid, method=fusion_method)
    
    # Analyze results
    analyze_fusion_quality(view1_grid, view2_grid, fused_grid)
    
    # Save results
    output_file = f"fusion_{fusion_method}.ply"
    visualize_fusion_result(fused_grid, output_file=output_file)
    
    # Save fused grid for further processing
    np.savez(f"fusion_{fusion_method}.npz", 
             fused_grid=fused_grid,
             method=fusion_method)
    print(f"Saved fused grid to fusion_{fusion_method}.npz")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    fusion_method = "arithmetic"
    aligned_grids_file = "aligned_grids.npz"
    
    if len(sys.argv) > 1:
        fusion_method = sys.argv[1]  # "arithmetic" or "geometric"
    if len(sys.argv) > 2:
        aligned_grids_file = sys.argv[2]
    
    run_simple_fusion(aligned_grids_file, fusion_method)
