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

def generate_sam3d_outputs(object_path, image_paths, mask_paths, inference, stage1_only=True):
    """Generate SAM3D occupancy grids for multiple views"""
    
    if inference is None:
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
        output = inference(image, mask, seed=42, stage1_only=stage1_only)
        # for key in output.keys():
        #     print(f"{key}: type {type(output[key])}; {output[key]}")
        
        # extract outputs
        scale = output["scale"].cpu().numpy().squeeze()
        translation = output["translation"].cpu().numpy().squeeze()
        translation_scale = output["translation_scale"].cpu().numpy().squeeze()
        rotation = output["6drotation_normalized"].cpu().numpy().squeeze()
        coords_original = output["coords_original"].cpu().numpy()[:, 1:]  # Remove batch index
        occupancy_grid = output["occupancy_grid"].cpu().numpy().squeeze()  # Full probability grid

        # Convert from voxel indices [0, 63] to world coordinates [-0.5, 0.5] (like demo_occupancy.py)
        coords_normalized = (coords_original / 63.0) - 0.5
        print(f" Mean of original coords: {coords_original.mean(axis=0)}; Max: {coords_original.max(axis=0)}; Min: {coords_original.min(axis=0)}")
        print(f" Mean of normalized coords: {coords_normalized.mean(axis=0)}; Max: {coords_normalized.max(axis=0)}; Min: {coords_normalized.min(axis=0)}")
        
        # Save outputs
        image_name = img_path.split("/")[-1].split(".")[0]

        if not stage1_only:
            mesh = output["mesh"][0]
            glb = output["glb"]
            print(f"Mesh vertices shape: {mesh.vertices.shape}, faces shape: {mesh.faces.shape}")
            print(f"GLB vertices shape: {glb.vertices.shape}, faces shape: {glb.faces.shape}")
            # Save mesh as GLB and PLY using the trimesh object
            glb.export(os.path.join(output_dir, f"{image_name}_mesh.glb"))
            glb.export(os.path.join(output_dir, f"{image_name}_mesh.ply"))
            print(f"Saved mesh for view {i+1} to {output_dir}")

        # Save output data
        data_file = os.path.join(output_dir, f"{image_name}_sam3d_outputs.npz")
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        np.savez(data_file, 
                coords_original=coords_original,
                coords_normalized=coords_normalized,
                occupancy_grid=occupancy_grid,  # Save full probability grid
                scale=scale, 
                translation=translation,
                translation_scale=translation_scale,
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
        print(f"  Scale: {scale}, Translation: {translation}, Translation Scale: {translation_scale}")
    
    results = {
        "num_views": len(image_paths),
        "output_dir": output_dir,
    }
    return results

def main():

    argparser = argparse.ArgumentParser(description="Generate SAM3D multiview outputs")
    argparser.add_argument("--object_path", type=str, required=True, help="Path to object folder containing images and masks")
    argparser.add_argument("--image_folder", type=str, required=True, help="Folder path containing input images")
    argparser.add_argument("--mask_folder", type=str, required=False, help="Folder path containing input masks")

    args = argparser.parse_args()

    image_paths = sorted([os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder) if f.endswith(".png")])
    masks_paths = image_paths if not args.mask_folder else sorted([os.path.join(args.mask_folder, f) for f in os.listdir(args.mask_folder) if f.endswith(".png")])

    # load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    generate_sam3d_outputs(args.object_path, image_paths, masks_paths, inference, stage1_only=False)


if __name__ == "__main__":
    main()