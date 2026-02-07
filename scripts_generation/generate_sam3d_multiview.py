# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import argparse

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    print(f"Using GPU from environment: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

import sys
import numpy as np
import trimesh

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask

def generate_sam3d_outputs(object_path, image_paths, mask_paths):
    """Generate SAM3D occupancy grids for multiple views"""
    
    # load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)
    
    output_dir = os.path.join(object_path, "sam3d_singleview_predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        print(f"Processing view {i+1}: {os.path.basename(img_path)}")
        
        # load image and mask
        image = load_image(img_path)
        mask = load_mask(mask_path)
        
        # run model
        output = inference(image, mask, seed=42, stage1_only=True)
    
        # extract outputs
        scale = output["scale"].cpu().numpy().squeeze()
        shift = output["translation"].cpu().numpy().squeeze()
        rotation = output["6drotation_normalized"].cpu().numpy().squeeze()
        coords_original = output["coords_original"].cpu().numpy()[:, 1:]  # Remove batch index
        occupancy_grid = output["occupancy_grid"].cpu().numpy().squeeze()  # Full probability grid
        

        # Convert from voxel indices [0, 63] to world coordinates [-0.5, 0.5] (like demo_occupancy.py)
        coords_normalized = (coords_original / 63.0) - 0.5
        print(f" Mean of original coords: {coords_original.mean(axis=0)}; Max: {coords_original.max(axis=0)}; Min: {coords_original.min(axis=0)}")
        print(f" Mean of normalized coords: {coords_normalized.mean(axis=0)}; Max: {coords_normalized.max(axis=0)}; Min: {coords_normalized.min(axis=0)}")


        # Save output data
        image_name = img_path.split("/")[-1].split(".")[0]
        data_file = os.path.join(output_dir, f"{image_name}.npz")
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        np.savez(data_file, 
                coords_original=coords_original,
                coords_normalized=coords_normalized,
                occupancy_grid=occupancy_grid,  # Save full probability grid
                scale=scale, 
                shift=shift,
                rotation=rotation,
                image_path=img_path,
                mask_path=mask_path)

        # Save normalized voxels as point cloud for visualization
        pc = trimesh.PointCloud(coords_normalized)
        pc_path = os.path.join(output_dir, f"{image_name}_voxels.ply")
        os.makedirs(os.path.dirname(pc_path), exist_ok=True)
        pc.export(pc_path)
        
        print(f"  Saved SAM3D output to {data_file}")
        print(f"  Saved voxel point cloud to {pc_path}")
        print(f"  Occupancy grid shape: {occupancy_grid.shape}")
        print(f"  Occupancy values: min={occupancy_grid.min():.6f}, max={occupancy_grid.max():.6f}, mean={occupancy_grid.mean():.6f}")
        print(f"  Scale: {scale}, Shift: {shift}")

def main():

    argparser = argparse.ArgumentParser(description="Generate SAM3D multiview outputs")


    # For dataset
    # object_path = "/scratch/cl927/nutritionverse-3d-new/_NEWCODE_test_id-11-red-apple"
    # object_path = "/scratch/cl927/nutritionverse-3d-new/_NEWCODE_id-26-chicken-leg-133g"
    object_path = "/scratch/cl927/Toys4k/renders/_test"
    
    # Generate SAM3D outputs for first two views
    # image_paths = [
    #     "/scratch/cl927/nutritionverse-3d-new/_NEWCODE_test_id-11-red-apple/rendered-test-example/render_000.png",
    #     "/scratch/cl927/nutritionverse-3d-new/_NEWCODE_test_id-11-red-apple/rendered-test-example/render_001.png"
    # ]
    # image_paths = [
    #     "/scratch/cl927/nutritionverse-3d-new/_NEWCODE_id-26-chicken-leg-133g/rendered-test-example/render_000.png",
    #     "/scratch/cl927/nutritionverse-3d-new/_NEWCODE_id-26-chicken-leg-133g/rendered-test-example/render_001.png"
    # ]
    image_paths = [
        "/scratch/cl927/Toys4k/renders/_test/000.png",
        "/scratch/cl927/Toys4k/renders/_test/001.png"
    ]

    # Use image as mask for now
    # mask_paths = [
    #     "/scratch/cl927/nutritionverse-3d-new/_NEWCODE_test_id-11-red-apple/rendered-test-example/render_000.png", 
    #     "/scratch/cl927/nutritionverse-3d-new/_NEWCODE_test_id-11-red-apple/rendered-test-example/render_001.png"
    # ]
    # mask_paths = [
    #     "/scratch/cl927/nutritionverse-3d-new/_NEWCODE_id-26-chicken-leg-133g/rendered-test-example/render_000.png", 
    #     "/scratch/cl927/nutritionverse-3d-new/_NEWCODE_id-26-chicken-leg-133g/rendered-test-example/render_001.png"
    # ]
    mask_paths = [
        "/scratch/cl927/Toys4k/renders/_test/000.png", 
        "/scratch/cl927/Toys4k/renders/_test/001.png"
    ]
    
    generate_sam3d_outputs(object_path, image_paths, mask_paths)


if __name__ == "__main__":
    main()