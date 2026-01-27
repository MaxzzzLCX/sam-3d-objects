# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import numpy as np
import trimesh
import torch

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image
image = load_image("/scratch/cl927/scenes_sam3dvggt/apple_new/resized_images/resized_02.png")
mask = load_mask("/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_02_mask_1.png")

# run model (generates a map)
output = inference(image, mask, seed=42, stage1_only=True) # Only run stage 1 for occupancy grid


def analyze_full_pipeline_data():
    """
    Comprehensive analysis of SAM3D data flow to understand surface vs volume
    """
    print("=" * 80)
    print("COMPREHENSIVE SAM3D DATA FLOW ANALYSIS")
    print("=" * 80)
    
    # 1. RAW OCCUPANCY GRID (before any filtering)
    occupancy_grid = output["occupancy_grid"].cpu().numpy().squeeze()  # Full 64³ grid
    print(f"\n1. RAW OCCUPANCY GRID:")
    print(f"   Shape: {occupancy_grid.shape}")
    print(f"   Values: min={occupancy_grid.min():.6f}, max={occupancy_grid.max():.6f}, mean={occupancy_grid.mean():.6f}")
    
    # Count occupied voxels in full grid (occupancy > 0)
    occupied_mask = occupancy_grid > 0
    total_occupied = np.sum(occupied_mask)
    total_possible = occupancy_grid.size
    occupancy_ratio = total_occupied / total_possible
    
    print(f"   Total occupied voxels (>0): {total_occupied}")
    print(f"   Total grid size: {total_possible}")
    print(f"   Occupancy ratio: {occupancy_ratio:.6f} ({occupancy_ratio*100:.2f}%)")
    
    # Analyze occupancy value distribution
    occupied_values = occupancy_grid[occupied_mask]
    if len(occupied_values) > 0:
        print(f"   Occupied voxel values: min={occupied_values.min():.6f}, max={occupied_values.max():.6f}")
        print(f"   Value distribution: {np.percentile(occupied_values, [25, 50, 75, 95])}")
    
    # 2. COORDS_ORIGINAL (filtered coordinates)
    coords_original = output["coords_original"].cpu().numpy()[:, 1:]  # Remove batch index
    print(f"\n2. COORDS_ORIGINAL (after torch.argwhere(ss > 0)):")
    print(f"   Number of coordinates: {len(coords_original)}")
    print(f"   Coordinate range: x=[{coords_original[:, 0].min()}, {coords_original[:, 0].max()}]")
    print(f"   Coordinate range: y=[{coords_original[:, 1].min()}, {coords_original[:, 1].max()}]")  
    print(f"   Coordinate range: z=[{coords_original[:, 2].min()}, {coords_original[:, 2].max()}]")
    
    # 3. Compare full grid vs coords
    print(f"\n3. COMPARISON:")
    print(f"   Occupied voxels in full grid: {total_occupied}")
    print(f"   Coordinates extracted: {len(coords_original)}")
    print(f"   Match: {'✓' if total_occupied == len(coords_original) else '✗'}")
    
    if total_occupied == len(coords_original):
        print("   → coords_original contains ALL occupied voxels from full grid")
    else:
        print("   → coords_original is filtered/processed version")
        
    # 4. Verify by reconstructing occupancy from coords
    reconstructed_grid = np.zeros((64, 64, 64))
    reconstructed_grid[coords_original[:, 0], coords_original[:, 1], coords_original[:, 2]] = 1
    reconstructed_occupied = np.sum(reconstructed_grid)
    
    print(f"\n4. RECONSTRUCTION TEST:")
    print(f"   Reconstructed occupied count: {reconstructed_occupied}")
    print(f"   Original coords count: {len(coords_original)}")
    print(f"   Perfect match: {'✓' if reconstructed_occupied == len(coords_original) else '✗'}")
    
    return occupancy_grid, coords_original, occupied_mask


def analyze_surface_vs_volume(occupancy_grid, coords_original):
    """
    Determine if SAM3D produces surface or volume representation
    """
    print(f"\n" + "=" * 50)
    print("SURFACE VS VOLUME ANALYSIS")  
    print("=" * 50)
    
    # Method 1: Use full occupancy grid
    print(f"\nMETHOD 1: Full Occupancy Grid Analysis")
    occupied_mask = occupancy_grid > 0
    
    surface_count = 0
    interior_count = 0
    
    occupied_coords = np.argwhere(occupied_mask)
    
    for coord in occupied_coords:
        i, j, k = coord
        
        # Check 6-connected neighbors
        neighbors = [
            (i+1, j, k), (i-1, j, k),
            (i, j+1, k), (i, j-1, k),
            (i, j, k+1), (i, j, k-1)
        ]
        
        is_surface = False
        for ni, nj, nk in neighbors:
            if 0 <= ni < 64 and 0 <= nj < 64 and 0 <= nk < 64:
                if occupancy_grid[ni, nj, nk] <= 0:  # Empty neighbor
                    is_surface = True
                    break
            else:
                is_surface = True  # Edge of grid
                
        if is_surface:
            surface_count += 1
        else:
            interior_count += 1
    
    print(f"   Surface voxels: {surface_count}")
    print(f"   Interior voxels: {interior_count}") 
    print(f"   Surface ratio: {surface_count/len(occupied_coords):.3f}")
    
    # Method 2: Analyze occupancy values for thickness
    print(f"\nMETHOD 2: Occupancy Value Analysis")
    occupied_values = occupancy_grid[occupied_mask]
    
    # If it's a surface, values should be relatively uniform
    # If it's volume, interior might have higher values
    print(f"   Value statistics:")
    print(f"     Min: {occupied_values.min():.6f}")
    print(f"     Max: {occupied_values.max():.6f}")
    print(f"     Mean: {occupied_values.mean():.6f}")
    print(f"     Std: {occupied_values.std():.6f}")
    
    # Method 3: Expected counts analysis
    print(f"\nMETHOD 3: Expected Count Analysis")
    total_occupied = len(occupied_coords)
    
    # For sphere r=20 in 64³ grid
    expected_volume = int((4/3) * np.pi * (20**3))  # ~33,500
    expected_surface = int(4 * np.pi * (20**2))     # ~5,000
    
    print(f"   Actual occupied: {total_occupied}")
    print(f"   Expected solid sphere: ~{expected_volume}")
    print(f"   Expected surface sphere: ~{expected_surface}")
    
    # Conclusion
    print(f"\nCONCLUSION:")
    if interior_count == 0:
        print("   🔍 PURE SURFACE representation")
        result = "surface"
    elif interior_count < total_occupied * 0.1:  # < 10% interior
        print("   🔍 MOSTLY SURFACE with minimal interior")
        result = "mostly_surface"
    elif surface_count > total_occupied * 0.8:   # > 80% surface
        print("   🔍 THICK SURFACE shell")
        result = "thick_surface"
    else:
        print("   🔍 VOLUME representation")
        result = "volume"
        
    return result, surface_count, interior_count


def save_visualizations(occupancy_grid, coords_original):
    """
    Save different representations for visual comparison
    """
    print(f"\n" + "=" * 50)
    print("SAVING VISUALIZATIONS")
    print("=" * 50)
    
    # 1. Full occupancy grid (all occupied voxels)
    occupied_coords = np.argwhere(occupancy_grid > 0)
    occupied_coords_normalized = (occupied_coords / 63.0) - 0.5
    
    pc_full = trimesh.PointCloud(occupied_coords_normalized)
    pc_full.export("full_occupancy_grid.ply")
    print(f"   Saved full occupancy grid: {len(occupied_coords)} points → full_occupancy_grid.ply")
    
    # 2. Coords original 
    coords_normalized = (coords_original / 63.0) - 0.5
    pc_coords = trimesh.PointCloud(coords_normalized)
    pc_coords.export("coords_original.ply")  
    print(f"   Saved coords_original: {len(coords_original)} points → coords_original.ply")
    
    # 3. Occupancy values visualization (color by intensity)
    if len(occupied_coords) > 0:
        occupied_values = occupancy_grid[occupancy_grid > 0]
        # Normalize values to 0-255 for colors
        normalized_values = ((occupied_values - occupied_values.min()) / 
                           (occupied_values.max() - occupied_values.min()) * 255)
        
        # Create color map (blue=low, red=high)  
        colors = np.zeros((len(occupied_coords), 3))
        colors[:, 0] = normalized_values  # Red channel
        colors[:, 2] = 255 - normalized_values  # Blue channel
        
        pc_values = trimesh.PointCloud(occupied_coords_normalized, colors=colors.astype(np.uint8))
        pc_values.export("occupancy_values_colored.ply")
        print(f"   Saved value-colored grid: occupancy_values_colored.ply")
    
    # 4. Compare counts
    print(f"\n   COMPARISON:")
    print(f"     Full grid occupied: {len(occupied_coords)}")
    print(f"     Coords original: {len(coords_original)}")
    print(f"     Match: {'✓' if len(occupied_coords) == len(coords_original) else '✗'}")


# Run comprehensive analysis
occupancy_grid, coords_original, occupied_mask = analyze_full_pipeline_data()
result, surface_count, interior_count = analyze_surface_vs_volume(occupancy_grid, coords_original)
save_visualizations(occupancy_grid, coords_original)

print(f"\n" + "=" * 80)
print("FINAL SUMMARY") 
print("=" * 80)
print(f"SAM3D Representation Type: {result.upper()}")
print(f"Surface voxels: {surface_count}")
print(f"Interior voxels: {interior_count}")
print(f"coords_original contains: {'ALL occupied voxels' if len(coords_original) == np.sum(occupied_mask) else 'FILTERED subset'}")

print(f"\nIMPLICATIONS FOR ALIGNMENT:")
if result in ["surface", "mostly_surface"]:
    print("✅ Can use coords_original directly for ICP - no surface extraction needed!")
    print("✅ SAM3D and VGGT both represent surfaces - perfect match!")
elif result == "thick_surface":
    print("⚠️  Surface extraction recommended but not critical")
else:
    print("❌ Need surface extraction for proper ICP alignment")

print("=" * 80)