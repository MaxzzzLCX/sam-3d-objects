#!/usr/bin/env python3
"""
Render normalized meshes from NutritionVerse-3D dataset for testing

This script:
1. Finds all normalized_mesh.obj files in the dataset
2. Renders 10 views of each mesh using Blender
3. Saves rendered images to "rendered-test-example" folder in each item directory
4. Uses fixed camera at radius 2.0 (suitable for normalized meshes in unit cube)
"""

import bpy
import bmesh
import sys
import os
import argparse
import json
import numpy as np
import glob
from pathlib import Path
from mathutils import Matrix, Vector

def clear_scene():
    """Clear all mesh objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

def setup_camera_and_lighting():
    """Setup camera at radius 2.0 and basic lighting"""
    # Add camera at radius 2.0
    bpy.ops.object.camera_add(location=(0, 0, 2.0))
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera
    
    # Configure camera for normalized meshes
    camera.data.lens = 35  # 35mm focal length
    camera.data.sensor_width = 32  # 32mm sensor
    
    # Add area light
    bpy.ops.object.light_add(type='AREA', location=(2, 2, 3))
    light = bpy.context.active_object
    light.data.energy = 5.0
    light.data.size = 2.0
    light.data.size_y = 2.0
    
    # Add fill light
    bpy.ops.object.light_add(type='AREA', location=(-2, -1, 2))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 2.0
    fill_light.data.size = 1.5
    fill_light.data.size_y = 1.5
    
    return camera

def render_object_mask(mesh_obj, mask_path):
    """Render binary mask by using transparency and post-processing"""
    import os
    
    scene = bpy.context.scene
    
    # Store original settings
    original_film_transparent = scene.render.film_transparent
    original_color_mode = scene.render.image_settings.color_mode
    original_file_format = scene.render.image_settings.file_format
    
    # Enable transparency to get alpha channel
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    
    # Create temporary path for RGBA render
    temp_rgba_path = mask_path.replace('.png', '_temp_rgba.png')
    
    # Render with transparency
    scene.render.filepath = temp_rgba_path
    bpy.ops.render.render(write_still=True)
    
    # Post-process using Blender's image processing
    # Load the RGBA image
    bpy.ops.image.open(filepath=temp_rgba_path)
    rgba_image = bpy.data.images.get(os.path.basename(temp_rgba_path))
    
    if rgba_image:
        # Get pixel data (RGBA)
        pixels = list(rgba_image.pixels)
        width, height = rgba_image.size
        
        # Create binary mask from alpha channel
        # pixels are in format [R,G,B,A, R,G,B,A, ...]
        binary_pixels = []
        for i in range(0, len(pixels), 4):
            alpha = pixels[i + 3]  # Alpha channel
            # Set to 1.0 (white) if alpha > 0, else 0.0 (black)
            mask_value = 1.0 if alpha > 0 else 0.0
            binary_pixels.extend([mask_value, mask_value, mask_value, 1.0])  # RGBA for consistency
        
        # Create new image for mask
        mask_image = bpy.data.images.new("mask_temp", width=width, height=height, alpha=False)
        mask_image.pixels = binary_pixels
        
        # Save as grayscale
        mask_image.file_format = 'PNG'
        mask_image.filepath_raw = mask_path
        mask_image.save()
        
        # Clean up
        bpy.data.images.remove(rgba_image)
        bpy.data.images.remove(mask_image)
    
    # Clean up temporary file
    if os.path.exists(temp_rgba_path):
        os.remove(temp_rgba_path)
    
    # Restore original render settings
    scene.render.film_transparent = original_film_transparent
    scene.render.image_settings.color_mode = original_color_mode
    scene.render.image_settings.file_format = original_file_format

def setup_render_settings():
    """Setup Cycles render settings optimized for speed"""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Use GPU if available
    scene.cycles.device = 'GPU'
    
    # Fast render settings
    scene.cycles.samples = 64  # Lower samples for faster rendering
    scene.cycles.max_bounces = 4
    scene.cycles.min_bounces = 1
    scene.cycles.transparent_max_bounces = 4
    scene.cycles.transparent_min_bounces = 1
    
    # Resolution
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    
    # Image settings
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True
    
    # Enable denoising for cleaner results
    scene.view_layers[0].cycles.use_denoising = True

def load_normalized_mesh(mesh_path):
    """Load normalized mesh from .obj file"""
    # Import mesh
    bpy.ops.import_scene.obj(filepath=mesh_path)
    
    # Find the imported mesh object
    mesh_obj = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break
    
    if mesh_obj is None:
        raise ValueError(f"No mesh found in {mesh_path}")
    
    # Set as active object
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    
    # Add material for better rendering
    if not mesh_obj.data.materials:
        mat = bpy.data.materials.new(name="FoodMaterial")
        mat.use_nodes = True
        
        # Simple principled BSDF material
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = (0.8, 0.7, 0.6, 1.0)  # Base color
        bsdf.inputs[7].default_value = 0.3  # Roughness
        bsdf.inputs[12].default_value = 1.0  # Specular
        
        mesh_obj.data.materials.append(mat)
    
    return mesh_obj

def generate_camera_positions(n_views=10):
    """Generate camera positions around the object at radius 2.0"""
    positions = []
    
    for i in range(n_views):
        # Distribute azimuth angles evenly around the object
        azimuth = (360 / n_views) * i
        
        # Use varying elevation angles for more interesting views
        if n_views == 10:
            elevations = [30, 45, 30, 60, 45, 30, 45, 60, 30, 45]
            elevation = elevations[i]
        else:
            elevation = 30 + (i % 3) * 15  # Vary between 30, 45, 60 degrees
        
        # Convert to radians
        azim_rad = np.radians(azimuth)
        elev_rad = np.radians(elevation)
        
        # Calculate position on sphere of radius 2.0
        radius = 2.0
        x = radius * np.cos(elev_rad) * np.sin(azim_rad)
        y = radius * np.cos(elev_rad) * np.cos(azim_rad)
        z = radius * np.sin(elev_rad)
        
        positions.append((x, y, z, azimuth, elevation))
    
    return positions

def render_views(mesh_obj, output_dir, n_views=10):
    """Render multiple views of the mesh with masks and parameter logging"""
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = output_dir.replace("rendered-test-example", "rendered-test-example-masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    camera = bpy.context.scene.camera
    positions = generate_camera_positions(n_views)
    
    # Prepare camera parameters JSON
    camera_params = {
        "camera_radius": 2.0,
        "focal_length_mm": camera.data.lens,
        "sensor_width_mm": camera.data.sensor_width,
        "views": []
    }
    
    print(f"  Rendering {n_views} views to {output_dir}")
    print(f"  Saving masks to {masks_dir}")
    
    for i, (x, y, z, azimuth, elevation) in enumerate(positions):
        # Position camera
        camera.location = (x, y, z)
        
        # Point camera at origin (where the normalized mesh is)
        direction = Vector((0, 0, 0)) - Vector((x, y, z))
        camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        
        # Update scene
        bpy.context.view_layer.update()
        
        # Record camera parameters
        camera_params["views"].append({
            "view_id": i,
            "filename": f"render_{i:03d}.png",
            "camera_position": [x, y, z],
            "azimuth_deg": float(azimuth),
            "elevation_deg": float(elevation),
            "camera_rotation_euler": [float(r) for r in camera.rotation_euler]
        })
        
        # Render RGB image
        output_path = os.path.join(output_dir, f"render_{i:03d}.png")
        bpy.context.scene.render.filepath = output_path
        
        print(f"    View {i+1}/{n_views}: azim={azimuth:.0f}°, elev={elevation:.0f}°")
        bpy.ops.render.render(write_still=True)
        
        # Generate mask by rendering object ID
        mask_path = os.path.join(masks_dir, f"render_{i:03d}.png")
        render_object_mask(mesh_obj, mask_path)
    
    # Save camera parameters to JSON file
    params_file = os.path.join(output_dir, "camera_parameters.json")
    with open(params_file, 'w') as f:
        json.dump(camera_params, f, indent=2)
    print(f"  Saved camera parameters to {params_file}")

def process_dataset(dataset_path, n_views=10, limit=None):
    """Process all normalized meshes in the dataset"""
    # Find all normalized mesh files
    normalized_meshes = glob.glob(os.path.join(dataset_path, "**/normalized_mesh.obj"), recursive=True)
    
    # Sort for consistent ordering
    normalized_meshes.sort()
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        normalized_meshes = normalized_meshes[:limit]
        print(f"Found {len(glob.glob(os.path.join(dataset_path, '**/normalized_mesh.obj'), recursive=True))} total meshes, processing first {len(normalized_meshes)} (limit={limit})")
    else:
        print(f"Found {len(normalized_meshes)} normalized meshes to render")
    
    # Setup render environment once
    clear_scene()
    setup_camera_and_lighting()
    setup_render_settings()
    
    success_count = 0
    error_count = 0
    
    for i, mesh_path in enumerate(normalized_meshes):
        try:
            # Get food item folder
            food_item_dir = Path(mesh_path).parent
            food_item_name = food_item_dir.name
            
            print(f"\n[{i+1}/{len(normalized_meshes)}] Processing: {food_item_name}")
            
            # Create output directory
            output_dir = food_item_dir / "rendered-test-example"
            
            # Clear previous mesh objects
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                    bpy.data.objects.remove(obj, do_unlink=True)
            
            # Load mesh
            mesh_obj = load_normalized_mesh(mesh_path)
            
            # Render views
            render_views(mesh_obj, str(output_dir), n_views)
            
            print(f"  ✓ Successfully rendered {n_views} views")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {mesh_path}: {e}")
            error_count += 1
    
    print(f"\n" + "="*60)
    print(f"RENDERING COMPLETE!")
    print(f"Successfully processed: {success_count} food items")
    print(f"Errors encountered: {error_count} food items")
    print(f"Total rendered images: {success_count * n_views}")

def main():
    # Parse command line arguments
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser(description="Render normalized meshes from NutritionVerse-3D dataset")
    parser.add_argument("--dataset_path", type=str, default="/scratch/cl927/nutritionverse-3d-new",
                        help="Path to the nutritionverse dataset")
    parser.add_argument("--n_views", type=int, default=10,
                        help="Number of views to render per mesh")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit processing to first N food items (useful for testing)")
    parser.add_argument("--test_single", type=str, default=None,
                        help="Test on single food item folder name")
    
    args = parser.parse_args(argv)
    
    print("NutritionVerse-3D Mesh Renderer")
    print("="*60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Views per mesh: {args.n_views}")
    if args.limit:
        print(f"Processing limit: {args.limit} food items")
    print(f"Camera radius: 2.0 (fixed for normalized meshes)")
    
    if args.test_single:
        # Test on single item
        test_mesh_path = os.path.join(args.dataset_path, args.test_single, "normalized_mesh.obj")
        if os.path.exists(test_mesh_path):
            print(f"Testing on single item: {args.test_single}")
            
            clear_scene()
            setup_camera_and_lighting()
            setup_render_settings()
            
            food_item_dir = Path(test_mesh_path).parent
            output_dir = food_item_dir / "rendered-test-example"
            
            mesh_obj = load_normalized_mesh(test_mesh_path)
            render_views(mesh_obj, str(output_dir), args.n_views)
            
            print(f"Test complete! Check: {output_dir}")
        else:
            print(f"Error: {test_mesh_path} not found")
    else:
        # Process entire dataset
        process_dataset(args.dataset_path, args.n_views, args.limit)

if __name__ == "__main__":
    main()