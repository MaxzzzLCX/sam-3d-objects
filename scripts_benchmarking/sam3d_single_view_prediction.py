#!/usr/bin/env python3
"""
SAM3D Single View Prediction Script

This script:
1. Takes a rendered image and generates a binary mask
2. Runs SAM3D inference to get voxel predictions
3. Normalizes the voxel grid (center at origin, unit cube)
4. Saves normalized voxels and comparison visualizations with ground truth mesh
"""

import os
import sys
import numpy as np
import trimesh
from PIL import Image
import json
from pathlib import Path

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import SAM3D inference code
sys.path.append("/scratch/cl927/sam-3d-objects/notebook")
from inference import Inference, load_image, load_single_mask, load_mask

def create_mask_from_rendered_image(image_path):
    """
    Create a binary mask from a rendered image with transparent background
    
    Args:
        image_path: Path to rendered PNG image with RGBA channels
    
    Returns:
        numpy array: Binary mask in the format expected by SAM3D (HxW)
    """
    # Load RGBA image
    img = Image.open(image_path).convert("RGBA")
    img_array = np.array(img)
    
    # Create mask from alpha channel
    # Alpha > 0 means object pixel, Alpha = 0 means background
    alpha = img_array[:, :, 3]  # Get alpha channel (HxW)
    mask = (alpha > 0).astype(np.uint8) * 255  # Convert to binary mask (0 or 255)
    
    # Return as 2D numpy array (HxW) - this is what SAM3D expects
    return mask

def normalize_voxel_coordinates(voxel_coords):
    """
    Normalize voxel coordinates and apply coordinate system transformation
    
    1. Apply rotation from SAM3D coordinate system (Y-up) to Blender (Z-up)
    2. Center bounding box at origin and scale to unit cube
    
    Args:
        voxel_coords: Nx3 numpy array of voxel coordinates
    
    Returns:
        Nx3 numpy array of transformed and normalized coordinates
    """
    if len(voxel_coords) == 0:
        return voxel_coords
    
    # Step 1: Apply coordinate system transformation from SAM3D (Y-up) to Blender (Z-up)
    # -90-degree rotation around X-axis: Y -> -Z, Z -> Y
    
    # Rotation matrix for -90° around X-axis:
    rotation_matrix = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ], dtype=float)
    
    # Apply rotation to convert Y-up (SAM3D) to Z-up (Blender)
    rotated_coords = voxel_coords @ rotation_matrix.T
    
    # Step 2: Normalize to unit cube centered at origin
    # Get bounding box
    min_coords = np.min(rotated_coords, axis=0)
    max_coords = np.max(rotated_coords, axis=0)
    
    # Center bounding box at origin
    bbox_center = (min_coords + max_coords) / 2.0
    centered_coords = rotated_coords - bbox_center
    
    # Scale so longest dimension spans [-0.5, 0.5]
    extents = max_coords - min_coords
    max_extent = np.max(extents)
    
    if max_extent > 0:
        normalized_coords = centered_coords / max_extent
    else:
        normalized_coords = centered_coords
    
    return normalized_coords

def run_sam3d_inference(image_path, mask_path, config_path):
    """
    Run SAM3D inference on image and mask
    
    Args:
        image_path: Path to input image
        mask_path: Path to mask file (can be same as image_path for RGBA images)
        config_path: Path to SAM3D config file
    
    Returns:
        dict: SAM3D inference output containing occupancy grid and coordinates
    """
    # Load model
    inference = Inference(config_path, compile=False)
    
    # Load image and mask using SAM3D's built-in functions
    image = load_image(image_path)
    
    # If mask_path is the same as image_path, we're using the image itself as mask
    if mask_path == image_path:
        print("Using rendered image itself as mask (extracting from alpha channel)")
        mask = load_mask(image_path)  # SAM3D can extract mask from RGBA images
    else:
        mask = load_mask(mask_path)
    
    # Run inference (stage 1 only for occupancy grid)
    output = inference(image, mask, seed=42, stage1_only=True)
    
    return output

def save_voxel_predictions(voxel_coords, output_dir, raw_voxel_coords=None, filename_prefix="sam3d_voxels"):
    """
    Save voxel coordinates as point cloud
    
    Args:
        voxel_coords: Nx3 numpy array of normalized voxel coordinates
        output_dir: Directory to save files
        raw_voxel_coords: Nx3 numpy array of raw/unnormalized voxel coordinates (optional)
        filename_prefix: Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if len(voxel_coords) == 0:
        print("Warning: No voxels to save")
        return
    
    # Save normalized voxels
    pc = trimesh.PointCloud(voxel_coords)
    ply_path = os.path.join(output_dir, f"{filename_prefix}_normalized.ply")
    pc.export(ply_path)
    
    npy_path = os.path.join(output_dir, f"{filename_prefix}_normalized.npy")
    np.save(npy_path, voxel_coords)
    
    print(f"Saved {len(voxel_coords)} normalized voxels to:")
    print(f"  - {ply_path}")
    print(f"  - {npy_path}")
    
    # Save raw/unnormalized voxels if provided
    if raw_voxel_coords is not None and len(raw_voxel_coords) > 0:
        raw_pc = trimesh.PointCloud(raw_voxel_coords)
        raw_ply_path = os.path.join(output_dir, f"{filename_prefix}_raw.ply")
        raw_pc.export(raw_ply_path)
        
        raw_npy_path = os.path.join(output_dir, f"{filename_prefix}_raw.npy")
        np.save(raw_npy_path, raw_voxel_coords)
        
        print(f"Saved {len(raw_voxel_coords)} raw voxels to:")
        print(f"  - {raw_ply_path}")
        print(f"  - {raw_npy_path}")
        
        # Print statistics for debugging
        print(f"Raw voxel statistics:")
        print(f"  Min: {raw_voxel_coords.min(axis=0)}")
        print(f"  Max: {raw_voxel_coords.max(axis=0)}")
        print(f"  Extent: {raw_voxel_coords.max(axis=0) - raw_voxel_coords.min(axis=0)}")
        print(f"  Center: {(raw_voxel_coords.min(axis=0) + raw_voxel_coords.max(axis=0)) / 2}")
        
        print(f"Normalized voxel statistics:")
        print(f"  Min: {voxel_coords.min(axis=0)}")
        print(f"  Max: {voxel_coords.max(axis=0)}")
        print(f"  Extent: {voxel_coords.max(axis=0) - voxel_coords.min(axis=0)}")
        print(f"  Center: {(voxel_coords.min(axis=0) + voxel_coords.max(axis=0)) / 2}")

def create_comparison_visualization(normalized_voxels, ground_truth_mesh_path, output_dir):
    """
    Create visualization comparing SAM3D voxels with ground truth mesh surface points
    Both are saved as point clouds with different colors in the same file
    
    Args:
        normalized_voxels: Nx3 numpy array of normalized voxel coordinates
        ground_truth_mesh_path: Path to ground truth normalized_mesh.obj
        output_dir: Directory to save comparison visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth mesh
    try:
        gt_mesh = trimesh.load(ground_truth_mesh_path)
        if isinstance(gt_mesh, trimesh.Scene):
            gt_mesh = gt_mesh.dump(concatenate=True)
        print(f"Loaded ground truth mesh: {gt_mesh.vertices.shape[0]} vertices, {gt_mesh.faces.shape[0]} faces")
    except Exception as e:
        print(f"Error loading ground truth mesh: {e}")
        return
    
    # Sample points from mesh surface
    n_surface_points = min(10000, len(normalized_voxels) * 2) if len(normalized_voxels) > 0 else 10000
    try:
        surface_points, _ = trimesh.sample.sample_surface(gt_mesh, n_surface_points)
        print(f"Sampled {len(surface_points)} points from mesh surface")
    except Exception as e:
        print(f"Error sampling mesh surface, using vertices instead: {e}")
        surface_points = gt_mesh.vertices
    
    # Print coordinate statistics for debugging orientation issues
    print("\n=== COORDINATE SYSTEM ANALYSIS ===")
    print("Ground Truth Mesh Surface Points:")
    print(f"  X range: [{surface_points[:, 0].min():.3f}, {surface_points[:, 0].max():.3f}]")
    print(f"  Y range: [{surface_points[:, 1].min():.3f}, {surface_points[:, 1].max():.3f}]") 
    print(f"  Z range: [{surface_points[:, 2].min():.3f}, {surface_points[:, 2].max():.3f}]")
    print(f"  Center: [{surface_points[:, 0].mean():.3f}, {surface_points[:, 1].mean():.3f}, {surface_points[:, 2].mean():.3f}]")
    
    if len(normalized_voxels) > 0:
        print("SAM3D Predicted Voxels:")
        print(f"  X range: [{normalized_voxels[:, 0].min():.3f}, {normalized_voxels[:, 0].max():.3f}]")
        print(f"  Y range: [{normalized_voxels[:, 1].min():.3f}, {normalized_voxels[:, 1].max():.3f}]")
        print(f"  Z range: [{normalized_voxels[:, 2].min():.3f}, {normalized_voxels[:, 2].max():.3f}]")
        print(f"  Center: [{normalized_voxels[:, 0].mean():.3f}, {normalized_voxels[:, 1].mean():.3f}, {normalized_voxels[:, 2].mean():.3f}]")
        
        # Check for potential 90-degree rotation patterns
        print("\nPOTENTIAL COORDINATE SYSTEM ISSUES:")
        # Check if voxel Y matches mesh Z (common XY<->XZ rotation)
        if abs(normalized_voxels[:, 1].mean() - surface_points[:, 2].mean()) < 0.1:
            print("  WARNING: Voxel Y ≈ Mesh Z (possible XY-plane vs XZ-plane issue)")
        if abs(normalized_voxels[:, 2].mean() - surface_points[:, 1].mean()) < 0.1:
            print("  WARNING: Voxel Z ≈ Mesh Y (possible coordinate axis swap)")
            
        # Check extent alignment
        voxel_extents = normalized_voxels.max(axis=0) - normalized_voxels.min(axis=0)
        mesh_extents = surface_points.max(axis=0) - surface_points.min(axis=0)
        print(f"Mesh extents (X,Y,Z): [{mesh_extents[0]:.3f}, {mesh_extents[1]:.3f}, {mesh_extents[2]:.3f}]")
        print(f"Voxel extents (X,Y,Z): [{voxel_extents[0]:.3f}, {voxel_extents[1]:.3f}, {voxel_extents[2]:.3f}]")
    
    # Create combined point cloud with different colors
    all_points = []
    all_colors = []
    
    # Add ground truth surface points (blue)
    all_points.append(surface_points)
    mesh_colors = np.tile([0, 100, 255, 255], (len(surface_points), 1))  # Blue
    all_colors.append(mesh_colors)
    
    # Add SAM3D voxel points (red)
    if len(normalized_voxels) > 0:
        all_points.append(normalized_voxels)
        voxel_colors = np.tile([255, 0, 0, 255], (len(normalized_voxels), 1))  # Red
        all_colors.append(voxel_colors)
    
    # Combine all points and colors
    if len(all_points) > 0:
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        # Create combined point cloud
        comparison_pc = trimesh.PointCloud(vertices=combined_points, colors=combined_colors)
        comparison_path = os.path.join(output_dir, "comparison_pointclouds.ply")
        comparison_pc.export(comparison_path)
        
        print(f"\nSaved comparison point clouds to: {comparison_path}")
        print(f"  Blue points: {len(surface_points)} ground truth surface points")
        if len(normalized_voxels) > 0:
            print(f"  Red points: {len(normalized_voxels)} SAM3D predicted voxels")
    
    # Also save separate point clouds for individual analysis
    gt_pc = trimesh.PointCloud(vertices=surface_points)
    gt_path = os.path.join(output_dir, "ground_truth_surface_points.ply")
    gt_pc.export(gt_path)
    print(f"Saved ground truth surface points to: {gt_path}")
    
    if len(normalized_voxels) > 0:
        voxel_pc = trimesh.PointCloud(vertices=normalized_voxels)
        voxel_path = os.path.join(output_dir, "sam3d_voxels_pointcloud.ply")
        voxel_pc.export(voxel_path)
        print(f"Saved SAM3D voxels to: {voxel_path}")
        
    # Save original mesh for reference
    gt_separate_path = os.path.join(output_dir, "ground_truth_mesh_reference.ply")
    gt_mesh.export(gt_separate_path)
    print(f"Saved ground truth mesh to: {gt_separate_path}")
    
    print("=" * 40)

def process_single_image(image_path, food_item_dir, config_path):
    """
    Process a single rendered image through the SAM3D pipeline
    
    Args:
        image_path: Path to rendered image
        food_item_dir: Path to food item directory (contains normalized_mesh.obj)
        config_path: Path to SAM3D config file
    """
    print(f"Processing: {image_path}")
    
    # Use the rendered image itself as the mask (since it contains transparency)
    # The SAM3D load_mask function can extract segmentation from RGBA images
    mask_path = image_path  # Use the same image as both input and mask
    
    print(f"Using rendered image as mask: {mask_path}")
    
    # Run SAM3D inference using the image as both input and mask
    print("Running SAM3D inference...")
    output = run_sam3d_inference(image_path, mask_path, config_path)
    
    # Extract voxel coordinates
    coords_original = output["coords_original"].cpu().numpy()
    coords_original = coords_original[:, 1:]  # Keep only x, y, z coordinates
    
    print(f"\n=== SAM3D COORDINATE SYSTEM ANALYSIS ===")
    print("Raw SAM3D coordinates (voxel indices [0,63]):")
    print(f"  Shape: {coords_original.shape}")
    print(f"  X range: [{coords_original[:, 0].min():.1f}, {coords_original[:, 0].max():.1f}]")
    print(f"  Y range: [{coords_original[:, 1].min():.1f}, {coords_original[:, 1].max():.1f}]") 
    print(f"  Z range: [{coords_original[:, 2].min():.1f}, {coords_original[:, 2].max():.1f}]")
    
    # Convert from voxel indices [0, 63] to world coordinates [-0.5, 0.5] (like demo_occupancy.py)
    voxels_world = (coords_original / 63.0) - 0.5
    
    print("SAM3D world coordinates ([-0.5, 0.5]):")
    print(f"  X range: [{voxels_world[:, 0].min():.3f}, {voxels_world[:, 0].max():.3f}]")
    print(f"  Y range: [{voxels_world[:, 1].min():.3f}, {voxels_world[:, 1].max():.3f}]")
    print(f"  Z range: [{voxels_world[:, 2].min():.3f}, {voxels_world[:, 2].max():.3f}]")
    
    # Check SAM3D convention - which axis is "up"?
    center = voxels_world.mean(axis=0)
    extents = voxels_world.max(axis=0) - voxels_world.min(axis=0)
    print(f"SAM3D voxel center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"SAM3D voxel extents: [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]")
    
    # Identify dominant axis (likely the "up" direction)
    dominant_axis = np.argmax(extents)
    axis_names = ['X', 'Y', 'Z']
    print(f"Dominant axis: {axis_names[dominant_axis]} (extent: {extents[dominant_axis]:.3f})")
    print("=" * 50)
    
    # Further normalize to match ground truth normalization (bounding box centered, unit cube)
    normalized_voxels = normalize_voxel_coordinates(voxels_world)
    
    print(f"SAM3D prediction: {len(normalized_voxels)} occupied voxels")
    
    # Create output directory with subfolder based on image name
    image_name = Path(image_path).stem  # Get filename without extension (e.g., "render_001")
    view_id = image_name.split('_')[-1]  # Extract "001" from "render_001"
    output_dir = os.path.join(food_item_dir, "SAM3D_singleview_prediction", view_id)
    
    # Save voxel predictions (both raw world coordinates and normalized)
    save_voxel_predictions(normalized_voxels, output_dir, raw_voxel_coords=voxels_world)
    
    # Create comparison with ground truth
    ground_truth_path = os.path.join(food_item_dir, "normalized_mesh.obj")
    if os.path.exists(ground_truth_path):
        print("Creating comparison visualization...")
        create_comparison_visualization(normalized_voxels, ground_truth_path, output_dir)
    else:
        print(f"Warning: Ground truth mesh not found at {ground_truth_path}")
    
    # Save inference metadata
    metadata = {
        "input_image": str(image_path),
        "num_voxels": len(normalized_voxels),
        "scale": output["scale"].cpu().numpy().squeeze().tolist(),
        "translation": output["translation"].cpu().numpy().squeeze().tolist(),
        "occupancy_ratio": len(coords_original) / (64**3),
        "normalization_method": "bounding_box_center_unit_cube"
    }
    
    metadata_path = os.path.join(output_dir, "sam3d_inference_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    return normalized_voxels, output_dir

def main():
    # Configuration
    config_path = "/scratch/cl927/sam-3d-objects/checkpoints/hf/pipeline.yaml"
    
    # Example: Process second rendered image (render_001.png) which has different viewpoint
    food_item_dir = "/scratch/cl927/nutritionverse-3d-new/id-1-salad-chicken-strip-7g"
    # food_item_dir = "/scratch/cl927/nutritionverse-3d-new/id-11-red-apple-145g"
    rendered_image_path = os.path.join(food_item_dir, "rendered-test-example", "render_000.png")
    
    print("SAM3D Single View Prediction")
    print("=" * 60)
    print(f"Food item directory: {food_item_dir}")
    print(f"Input image: {rendered_image_path}")
    print(f"SAM3D config: {config_path}")
    
    # Check if files exist
    if not os.path.exists(rendered_image_path):
        print(f"Error: Rendered image not found at {rendered_image_path}")
        print("Run the rendering script first!")
        return
    
    if not os.path.exists(config_path):
        print(f"Error: SAM3D config not found at {config_path}")
        return
    
    # Process the image
    try:
        normalized_voxels, output_dir = process_single_image(
            rendered_image_path, food_item_dir, config_path
        )
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print(f"Generated {len(normalized_voxels)} normalized voxels")
        print(f"Results saved to: {output_dir}")
        print("\nFiles generated:")
        print("- sam3d_voxels_normalized.ply (normalized voxel point cloud)")
        print("- sam3d_voxels_normalized.npy (normalized numpy array)")
        print("- sam3d_voxels_raw.ply (raw SAM3D voxel point cloud)")  
        print("- sam3d_voxels_raw.npy (raw SAM3D numpy array)")
        print("- comparison_pointclouds.ply (BLUE=ground truth surface, RED=SAM3D voxels)")
        print("- ground_truth_surface_points.ply (mesh surface points)")
        print("- sam3d_voxels_pointcloud.ply (voxels as point cloud)")
        print("- ground_truth_mesh_reference.ply (ground truth mesh)")
        print("- sam3d_inference_metadata.json (inference details)")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()