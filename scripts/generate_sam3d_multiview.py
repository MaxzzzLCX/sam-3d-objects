# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import numpy as np
import trimesh

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask

def generate_sam3d_outputs(image_paths, mask_paths, output_dir="sam3d_outputs"):
    """Generate SAM3D occupancy grids for multiple views"""
    
    # load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)
    
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
        coords_original = output["coords_original"].cpu().numpy()[:, 1:]  # Remove batch index
        
        # save outputs
        view_file = os.path.join(output_dir, f"view_{i+1:02d}.npz")
        np.savez(view_file, 
                coords=coords_original,
                scale=scale, 
                shift=shift,
                image_path=img_path,
                mask_path=mask_path)
        
        print(f"  Saved {len(coords_original)} voxels to {view_file}")
        print(f"  Scale: {scale}, Shift: {shift}")

if __name__ == "__main__":
    # Generate SAM3D outputs for first two views
    image_paths = [
        "/scratch/cl927/scenes_sam3dvggt/apple_new/resized_images/resized_01.png",
        "/scratch/cl927/scenes_sam3dvggt/apple_new/resized_images/resized_02.png"
    ]
    mask_paths = [
        "/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_01_mask_1.png", 
        "/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_02_mask_1.png"
    ]
    
    generate_sam3d_outputs(image_paths, mask_paths)