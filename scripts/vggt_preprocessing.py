#!/usr/bin/env python3
"""
VGGT Preprocessing Script
========================

Preprocesses images and generates masks for VGGT inference to ensure consistent scaling.

This script performs two key steps:
1. Resizes images to 518x518 (VGGT's native resolution) to avoid internal scaling
2. Generates binary masks from the resized images to ensure mask-image alignment

Usage:
    python scripts/vggt_preprocessing.py --scene_dir /path/to/scene --image_dir rendered-test-example --output_dir vggt_preprocessed

Output Structure:
    scene_dir/
    └── SAM3D_multiview_prediction/
        └── vggt_preprocessed/
            ├── images/           # 518x518 resized images  
            └── masks/            # Corresponding binary masks
"""

import os
import argparse
import numpy as np
from PIL import Image
import glob
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess images and masks for VGGT")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory name containing the input images (relative to scene_dir)")
    parser.add_argument("--output_dir", type=str, default="vggt_preprocessed", help="Output directory name (relative to scene_dir)")
    parser.add_argument("--target_resolution", type=int, default=518, help="Target resolution for VGGT (default: 518)")
    parser.add_argument("--background_threshold", type=int, default=0, help="Pixel threshold for background detection (default: 0 - any non-zero pixel is foreground)")
    parser.add_argument("--select_images", type=str, default=None, help="Comma-separated list of image names or indices (e.g., 'render_000,render_001' or '0,1,5')")
    return parser.parse_args()


def get_image_paths(image_dir):
    """Get all image file paths from directory."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_path_list = []
    for ext in image_extensions:
        image_path_list.extend(glob.glob(os.path.join(image_dir, ext)))
        image_path_list.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    image_path_list.sort()
    return image_path_list


def filter_images_by_selection(image_path_list, select_images):
    """Filter image list based on selection criteria."""
    if not select_images:
        return image_path_list
    
    selected_images = []
    selections = [s.strip() for s in select_images.split(',')]
    
    for selection in selections:
        if selection.isdigit():
            # Selection by index (e.g., "0", "1", "5")
            idx = int(selection)
            if 0 <= idx < len(image_path_list):
                selected_images.append(image_path_list[idx])
            else:
                print(f"Warning: Index {idx} out of range (0-{len(image_path_list)-1})")
        else:
            # Selection by name (e.g., "render_000", "render_001")
            found = False
            for img_path in image_path_list:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                if img_name == selection or os.path.basename(img_path) == selection:
                    selected_images.append(img_path)
                    found = True
                    break
            if not found:
                print(f"Warning: Image '{selection}' not found in directory")
    
    if selected_images:
        print(f"Selected {len(selected_images)} images: {[os.path.basename(p) for p in selected_images]}")
        return selected_images
    else:
        print("Warning: No valid images selected, using all images")
        return image_path_list

def mask_without_resizing(img_path, output_masks_dir, background_threshold=0):
    """Generate binary mask without resizing."""

    img_pil = Image.open(img_path).convert('RGB')
    img_np = np.array(img_pil)
    mask = np.any(img_np > background_threshold, axis=2).astype(np.uint8) * 255
    
    mask_name = "mask_without_resizing.png"
    mask_path = os.path.join(output_masks_dir, mask_name)
    mask_img = Image.fromarray(mask, 'L')
    mask_img.save(mask_path)
    return mask_path

def preprocess_image_and_mask(img_path, output_images_dir, output_masks_dir, target_resolution, background_threshold):
    """
    Process a single image: resize to target resolution and generate binary mask.
    
    Args:
        img_path: Path to input image
        output_images_dir: Directory to save processed images
        output_masks_dir: Directory to save generated masks
        target_resolution: Target size (518 for VGGT)
        background_threshold: Threshold for background detection
    """
    # Load image
    img_pil = Image.open(img_path).convert('RGB')
    original_size = img_pil.size
    
    # Resize image to target resolution using bilinear for good balance of quality and speed
    img_resized = img_pil.resize((target_resolution, target_resolution), Image.BILINEAR)
    
    # Convert to numpy for mask generation
    img_np = np.array(img_resized)
    
    # Debug: check background pixel values
    img_basename = os.path.basename(img_path)
    print(f"\nImage: {img_basename}")
    print(f"Min pixel values (R,G,B): {np.min(img_np, axis=(0,1))}")
    print(f"Max pixel values (R,G,B): {np.max(img_np, axis=(0,1))}")
    print(f"Unique values in corners (should be background):")
    corner_pixels = [
        img_np[0, 0], img_np[0, -1], img_np[-1, 0], img_np[-1, -1]
    ]
    for i, pixel in enumerate(corner_pixels):
        print(f"  Corner {i}: {pixel}")
    
    # Generate binary mask: any pixel with any channel > threshold is foreground
    mask = np.any(img_np > background_threshold, axis=2).astype(np.uint8) * 255
    
    # Debug mask statistics
    binary_mask = mask > 0
    total_pixels = binary_mask.size
    foreground_pixels = np.sum(binary_mask)
    background_pixels = total_pixels - foreground_pixels
    print(f"Mask stats: {foreground_pixels}/{total_pixels} foreground ({100*foreground_pixels/total_pixels:.1f}%)")
    print(f"Background threshold: {background_threshold}")
    
    # Save processed image
    img_basename = os.path.basename(img_path)
    img_name, img_ext = os.path.splitext(img_basename)
    
    output_img_path = os.path.join(output_images_dir, img_basename)
    img_resized.save(output_img_path)
    
    # Save mask with VGGT expected naming convention
    mask_name = f"{img_name}_mask_1.png"
    mask_path = os.path.join(output_masks_dir, mask_name)
    mask_img = Image.fromarray(mask, 'L')
    mask_img.save(mask_path)
    
    print(f"Processed {img_basename} ({original_size[0]}x{original_size[1]} -> {target_resolution}x{target_resolution})")
    
    return output_img_path, mask_path


def main():
    args = parse_args()
    
    # Setup directories - save under SAM3D_multiview_prediction
    image_dir = os.path.join(args.scene_dir, args.image_dir)
    output_base_dir = os.path.join(args.scene_dir, "SAM3D_multiview_prediction", args.output_dir)
    output_images_dir = os.path.join(output_base_dir, "images_resized")
    output_masks_dir = os.path.join(output_base_dir, "masks_resized")
    masks_dir_noresize = os.path.join(output_base_dir, "masks_noresize")
    
    # Create output directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    os.makedirs(masks_dir_noresize, exist_ok=True)
    
    print(f"VGGT Preprocessing")
    print(f"=================")
    print(f"Input directory: {image_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Target resolution: {args.target_resolution}x{args.target_resolution}")
    print(f"Background threshold: {args.background_threshold}")
    print()
    
    # Get image paths
    image_path_list = get_image_paths(image_dir)
    print(f"Found {len(image_path_list)} images in {image_dir}")
    
    # Filter images if selection specified
    image_path_list = filter_images_by_selection(image_path_list, args.select_images)
    
    # Process each image
    processed_images = []
    processed_masks = []
    
    print(f"\nProcessing {len(image_path_list)} images...")
    for img_path in image_path_list:
        try:
            # For debugging
            mask_noresize_path = mask_without_resizing(img_path, masks_dir_noresize, args.background_threshold)

            output_img_path, mask_path = preprocess_image_and_mask(
                img_path, output_images_dir, output_masks_dir, 
                args.target_resolution, args.background_threshold
            )
            processed_images.append(output_img_path)
            processed_masks.append(mask_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\nPreprocessing complete!")
    print(f"Processed images: {len(processed_images)}")
    print(f"Generated masks: {len(processed_masks)}")
    print(f"Images saved to: {output_images_dir}")
    print(f"Masks saved to: {output_masks_dir}")
    print()
    print("Usage with VGGT:")
    print(f"python scripts/vggt_runner.py --scene_dir {args.scene_dir} --image_dir SAM3D_multiview_prediction/{args.output_dir}/images_resized --mask_dir SAM3D_multiview_prediction/{args.output_dir}/masks_resized")


if __name__ == "__main__":
    main()