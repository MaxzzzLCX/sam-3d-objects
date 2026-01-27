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
# image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
# mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)
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

# pointmap_scale = output["pointmap_scale"].cpu().numpy()
# pointmap_shift = output["pointmap_shift"].cpu().numpy()

coords_original = output["coords_original"].cpu().numpy()
coords_original = coords_original[:, 1:] # Keep only x, y, z coordinates, discard batch_index


def visualize_voxels(voxels):
    """
    Visualize the voxel coordinates as a point cloud

    :param voxels: Nx3 numpy array of voxel coordinates
    """
    # Normalize coordinates from [0, 63] to [-0.5, 0.5]
    voxels_normalized = (voxels / 63.0) - 0.5
    
    # For visiualization purpose
    pc = trimesh.PointCloud(voxels_normalized)
    pc.export(f"occupancy_grid_raw.ply")
    print(f"Saved {len(voxels)} points to occupancy_grid.ply")


def calculate_occupancy_ratio(voxels):
    """
    Calculate the ratio occupied by the voxels
    """
    num_occupied = len(coords_original)
    total_voxels = 64 ** 3  # 262,144
    ratio = num_occupied / total_voxels
    return ratio

visualize_voxels(coords_original)
ratio = calculate_occupancy_ratio(coords_original)
print(f"Occupancy ratio: {ratio:.6f}")
print(f"Scale: {scale}, Shift: {shift}")

# Applying scale to the occupancy
grid_volume = 1 * scale[0] * scale[1] * scale[2]
print(f"Estimated object volume: {grid_volume:.6f} cubic units")
