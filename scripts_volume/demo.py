# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import os
import numpy as np

# import inference code
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "..", "notebook")
sys.path.append(parent_dir)
from inference import Inference, load_image, load_single_mask, load_masks, make_scene

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

image = load_image("/scratch/cl927/sam-3d-objects/scripts_volume/images/brunch.jpg")
mask_plate = load_single_mask("/scratch/cl927/sam-3d-objects/scripts_volume/masks", index=0)
mask_hashbrown = load_single_mask("/scratch/cl927/sam-3d-objects/scripts_volume/masks", index=1)

raw_volumes = np.zeros(2)
raw_spans = np.zeros((2, 3))
scales = np.zeros((2, 3))
volumes = np.zeros(2)
spans = np.zeros((2, 3))

# run model (generates a map)
for i, mask in enumerate([mask_plate, mask_hashbrown]):
    print(f"Running inference with mask {i}...")
    output = inference(image, mask, seed=42)
    # 6drotation_normalized, scale, shape, translation, translation_scale, coords_original, 
    # mesh (sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh.MeshExtractResult) 
    # gaussian (sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model.Gaussian)
    # glb: <trimesh.Trimesh(vertices.shape=(486892, 3), faces.shape=(973880, 3))>
    # gs: <sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model.Gaussian
    # pointmap, pointmap_colors

    # export gaussian splat and mesh
    print(f"The raw mesh object is of type {type(output['mesh'])}")
    print(output["mesh"])
    # output["mesh"].export(f"mesh_raw.glb")
    # output["mesh"].export(f"mesh_raw.ply")

    output["glb"].export(f"scripts_volume/mesh_trimesh_{i}.glb")
    output["glb"].export(f"scripts_volume/mesh_trimesh_{i}.ply")

    # Check if mesh is watertight using the trimesh object
    glb_mesh = output["glb"]
    is_watertight = glb_mesh.is_watertight
    scale = output["scale"].squeeze().cpu().numpy()
    print(f"scale {scale} is type {type(scale)}, length {len(scale)}")

    
    print(f"Mesh watertight: {is_watertight}")
    scales[i] = scale
    raw_volumes[i] = glb_mesh.volume

    # Span in each dimension
    span = glb_mesh.bounds[1] - glb_mesh.bounds[0]
    raw_spans[i] = span
    print(f"Span in each dimension: {span}")

    if scale[0] == scale[1] == scale[2]:
        print(f"Mesh scale: {scale[0]:.4f}")
        volume = glb_mesh.volume * (scale[0] ** 3)
        print(f"Mesh volume (scale={scale[0]:.4f}): {volume:.4f}")
    else:
        print(f"Mesh scale: {scale}. Not same scale across dimensions.")
        print(f"Volume calculation may be inaccurate due to non-uniform scaling.")
        volumes[i] = glb_mesh.volume * (scale[0] * scale[1] * scale[2])
        print(f"Mesh volume: {glb_mesh.volume * (scale[0] * scale[1] * scale[2]):.4f}")

    volumes[i] = volume
    spans[i] = span * scale

    output["gs"].save_ply(f"scripts_volume/mesh_splat_{i}.ply")
    print(f"Your reconstruction has been saved to scripts_volume/mesh_splat_{i}.ply")

# Final volume results
print(f"Raw volumes: {raw_volumes}")
print(f"Raw spans: {raw_spans}")
print(f"Scales: {scales}")
print(f"Volume after scaling: {volumes}")
print(f"Spans after scaling: {spans}")

actual_plate_diameter = 25  # cm

mesh_plate_diameter = max(spans[0])
metric_factor = actual_plate_diameter / mesh_plate_diameter # The length convertion factor from mesh units to real-world units

# Convert the mesh volume to real-world units using the scale factor
mesh_plate_volume = volumes[0]
real_world_plate_volume = mesh_plate_volume * (metric_factor ** 3)
print(f"Estimated plate volume in real-world units: {real_world_plate_volume:.6f} cubic centimeters")

real_world_hashbrown_volume = volumes[1] * (metric_factor ** 3)
print(f"Estimated hashbrown volume in real-world units: {real_world_hashbrown_volume:.6f} cubic centimeters (mL)")