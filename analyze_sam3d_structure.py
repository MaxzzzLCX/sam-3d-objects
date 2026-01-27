# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import numpy as np
import trimesh


# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask, load_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("/scratch/cl927/scenes_sam3dvggt/apple_new/resized_images/resized_02.png")
mask = load_mask("/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_02_mask_1.png")

# run model (generates a map)
output = inference(image, mask, seed=42, stage1_only=True) # Only run stage 1 for occupancy grid

# export occupancy grid
scale = output["scale"].cpu().numpy().squeeze() # Model prediction of object scale
shift = output["translation"].cpu().numpy().squeeze() # Model prediction of object translation
occupancy_grid = output["occupancy_grid"].cpu().numpy().squeeze() # NxNxN occupancy grid
print(f"Shape of occupancy grid is {occupancy_grid.shape}")
print(f"Max value {occupancy_grid.max()}, Min value {occupancy_grid.min()}, Average value {occupancy_grid.mean()}")

coords_original = output["coords_original"].cpu().numpy()
coords_original = coords_original[:, 1:] # Keep only x, y, z coordinates, discard batch_index


def visualize_voxels(voxels):
    """
    Visualize the voxel coordinates as a point cloud

    :param voxels: Nx3 numpy array of voxel coordinates
    """
    # Normalize coordinates from [0, 63] to [-0.5, 0.5]
    voxels_normalized = (voxels / 63.0) - 0.5
    
    # For visualization purpose
    pc = trimesh.PointCloud(voxels_normalized)
    pc.export(f"sam3d_structure_analysis.ply")
    print(f"Saved {len(voxels)} points to sam3d_structure_analysis.ply")


def analyze_occupancy_structure(voxels):
    """
    Analyze whether voxels represent surface or volume
    """
    num_occupied = len(voxels)
    total_voxels = 64 ** 3  # 262,144
    ratio = num_occupied / total_voxels
    
    print(f"\n=== OCCUPANCY ANALYSIS ===")
    print(f"Total occupied voxels: {num_occupied}")
    print(f"Total possible voxels: {total_voxels}")
    print(f"Occupancy ratio: {ratio:.6f} ({ratio*100:.2f}%)")
    
    # Check if this looks like surface vs volume
    # For a solid sphere: volume = (4/3)πr³, surface ≈ 4πr²  
    # For 64³ grid, sphere radius ~20 voxels
    expected_volume_sphere = int((4/3) * np.pi * (20**3))  # ~33,500 voxels
    expected_surface_sphere = int(4 * np.pi * (20**2))     # ~5,000 voxels
    
    print(f"Expected for solid sphere (~r=20): {expected_volume_sphere} voxels")
    print(f"Expected for surface sphere (~r=20): {expected_surface_sphere} voxels")
    
    if num_occupied < 8000:
        print("📊 ANALYSIS: This looks like SURFACE representation")
    elif num_occupied > 25000:
        print("📊 ANALYSIS: This looks like VOLUME representation")  
    else:
        print("📊 ANALYSIS: Unclear - could be thick surface or partial volume")
    
    return ratio


def check_interior_vs_surface(voxels):
    """
    Check if voxels are mostly on surface or fill interior
    """
    print(f"\n=== SURFACE VS INTERIOR CHECK ===")
    
    # Create 3D grid
    grid = np.zeros((64, 64, 64), dtype=bool)
    grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
    
    # Count surface vs interior voxels
    surface_count = 0
    interior_count = 0
    
    for i, j, k in voxels:
        # Check 6-connected neighbors
        neighbors = [
            (i+1, j, k), (i-1, j, k),
            (i, j+1, k), (i, j-1, k),
            (i, j, k+1), (i, j, k-1)
        ]
        
        # If any neighbor is empty, this is a surface voxel
        is_surface = False
        for ni, nj, nk in neighbors:
            if (0 <= ni < 64 and 0 <= nj < 64 and 0 <= nk < 64):
                if not grid[ni, nj, nk]:
                    is_surface = True
                    break
            else:
                is_surface = True  # Edge of grid
                
        if is_surface:
            surface_count += 1
        else:
            interior_count += 1
    
    print(f"Surface voxels: {surface_count}")
    print(f"Interior voxels: {interior_count}")
    print(f"Surface ratio: {surface_count/len(voxels):.3f}")
    
    if interior_count == 0:
        print("🔍 RESULT: PURE SURFACE - no interior voxels detected!")
    elif surface_count / len(voxels) > 0.8:
        print("🔍 RESULT: mostly surface with thin shell")
    else:
        print("🔍 RESULT: mixed surface + volume representation")
    
    return surface_count, interior_count


def analyze_thickness(voxels):
    """
    Analyze the thickness of the voxel representation
    """
    print(f"\n=== THICKNESS ANALYSIS ===")
    
    # Create 3D grid
    grid = np.zeros((64, 64, 64), dtype=bool)
    grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
    
    # Find center of mass
    center = voxels.mean(axis=0)
    print(f"Center of mass: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")
    
    # Calculate distances from center for occupied voxels
    distances = np.sqrt(np.sum((voxels - center)**2, axis=1))
    
    print(f"Distance from center: min={distances.min():.2f}, max={distances.max():.2f}, mean={distances.mean():.2f}")
    print(f"Radius range: {distances.max() - distances.min():.2f} voxels")
    
    # If it's a thin shell, most points should be at similar distance from center
    distance_std = np.std(distances)
    print(f"Distance standard deviation: {distance_std:.2f}")
    
    if distance_std < 2.0:
        print("🔍 THICKNESS: Very thin shell (std < 2.0)")
    elif distance_std < 5.0:
        print("🔍 THICKNESS: Moderately thick shell")
    else:
        print("🔍 THICKNESS: Thick or solid object")


# Run all analyses
visualize_voxels(coords_original)

# Analyze the structure
ratio = analyze_occupancy_structure(coords_original)
surface_count, interior_count = check_interior_vs_surface(coords_original)
analyze_thickness(coords_original)

print(f"\n=== IMPLICATIONS FOR ICP ===")
if interior_count == 0:
    print("✅ SAM3D is SURFACE-only - perfect for ICP with VGGT!")
    print("✅ No need for surface extraction - can use voxels directly")
    print("✅ This explains why our ICP worked so well!")
    print("✅ Can simplify alignment pipeline significantly")
elif surface_count / len(coords_original) > 0.9:
    print("⚠️  SAM3D is mostly surface with minimal interior")
    print("⚠️  Surface extraction might still help but not critical")
else:
    print("❌ SAM3D includes significant interior - surface extraction needed")

print(f"\n=== SAM3D PREDICTIONS ===")
print(f"Scale: {scale}")
print(f"Shift: {shift}")

# Applying scale to the occupancy
grid_volume = 1 * scale[0] * scale[1] * scale[2]
print(f"Estimated object volume: {grid_volume:.6f} cubic units")