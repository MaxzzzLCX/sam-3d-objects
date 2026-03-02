# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import os
import numpy as np
import json
import csv

import open3d as o3d
import torch
import argparse
import trimesh

# import inference code
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "..", "notebook")
sys.path.append(parent_dir)
from inference import Inference, load_image, load_single_mask, load_masks, make_scene


def run_volume_estimation_experiments(dataset_folder, volume_est_method, start_index=None, end_index=None):
    """
    Run tests on real dataset for volume estimation
    """
    dataset_folder_image = os.path.join(dataset_folder, "resized_images")
    image_paths = sorted([os.path.join(dataset_folder_image, f) for f in os.listdir(dataset_folder_image) if f.endswith(".jpeg") or f.endswith(".jpg") or f.endswith(".png")])
    if start_index is not None and end_index is not None:
        image_paths = image_paths[start_index:end_index]
    print(f"Dataset length {len(image_paths)} \n Image paths: {image_paths}")

    # Read the dataset properties from json
    json_path = os.path.join(dataset_folder, "plate_diameters.json")
    with open(json_path, "r") as f:
        dataset_properties = json.load(f)

    
    # Initialize csv file to save results
    csv_path = f"{dataset_folder}/volume_estimation_results.csv"
    with open(csv_path, "w", newline="") as csvfile:
        # fieldnames = ["image_name", "gt_volume_ml", "predicted_food_volume_ml(convex_hull)", "abs_error", "error_percent"]
        fieldnames = ["image_name", "gt_volume_ml", f"predicted_food_volume_ml({volume_est_method})", "volume_error", "error_percent", "volume_with_gt_relative_scale", "error_percent_with_gt_relative_scale", "relative_scale_error", "relative_scale_error_percent"]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    errors = []
    errors_with_gt_relative_scale = []
    relative_scale_errors = []
    raw_results = {}

    for i, image_path in enumerate(image_paths):
        IMAGE_NAME = image_path.split("/")[-1].split(".")[0]
        mask_folder = os.path.join(dataset_folder, f"masks_{IMAGE_NAME}")

        diameter = dataset_properties[IMAGE_NAME]["diameter"]
        gt_volume_ml = dataset_properties[IMAGE_NAME]["gt_volume_ml"]
        gt_relative_scale = dataset_properties[IMAGE_NAME]["gt_relative_scale"]

        volume_estimation_results = estimate_volumes(image_path, mask_folder, volume_est_method, ACTUAL_PLATE_DIAMETER=diameter, GT_RELATIVE_SCALES=gt_relative_scale)   
        predicted_volume = volume_estimation_results[f"food_volume_ml({volume_est_method})"]
        predicted_volume_with_gt_relative_scale = volume_estimation_results[f"food_volume_gt_relative_scale_ml({volume_est_method})"]
        predicted_relative_scale = volume_estimation_results["relative_scale_plate_food"]

        raw_results[i] = volume_estimation_results

        print(f"\n=== SUMMARY FOR {IMAGE_NAME} ===")
        print(f"Ground truth volume (ml): {gt_volume_ml}")
        print(f"Predicted food volume (ml): {predicted_volume:.2f} mL")
        print(f"Absolute error (ml): {abs(predicted_volume - gt_volume_ml):.2f} mL")
        print(f"Relative error: {abs(predicted_volume - gt_volume_ml)/gt_volume_ml:.2%}")

        # Save results to csv
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                "image_name": IMAGE_NAME,
                "gt_volume_ml": gt_volume_ml,
                f"predicted_food_volume_ml({volume_est_method})": f"{predicted_volume:.2f}",
                "volume_error": f"{(predicted_volume - gt_volume_ml):.2f}",
                "error_percent": f"{abs(predicted_volume - gt_volume_ml)/gt_volume_ml:.2%}",
                f"volume_with_gt_relative_scale": f"{predicted_volume_with_gt_relative_scale:.2f}",
                f"error_percent_with_gt_relative_scale": f"{abs(predicted_volume_with_gt_relative_scale - gt_volume_ml)/gt_volume_ml:.2%}",
                "relative_scale_error": f"{predicted_relative_scale - gt_relative_scale:.4f}",
                "relative_scale_error_percent": f"{abs(predicted_relative_scale - gt_relative_scale):.2%}"
            })
        errors.append(abs(predicted_volume - gt_volume_ml)/gt_volume_ml)
        errors_with_gt_relative_scale.append(abs(predicted_volume_with_gt_relative_scale - gt_volume_ml)/gt_volume_ml)
        relative_scale_errors.append(abs(predicted_relative_scale - gt_relative_scale)/gt_relative_scale)
        
    print(f"FINISHED {len(image_paths)} EXPERIMENTS. RESULTS SAVED TO {csv_path}")

    # Save a json file
    json_path = f"{dataset_folder}/{volume_est_method}_volume_estimation_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "dataset_folder": dataset_folder,
            "start_index": start_index,
            "end_index": end_index,
            "average_relative_error": np.mean(errors),
            "errors": errors,
            "average_relative_error_with_gt_relative_scale": np.mean(errors_with_gt_relative_scale),
            "errors_with_gt_relative_scale": errors_with_gt_relative_scale,
            "average_relative_scale_error": np.mean(relative_scale_errors),
            "relative_scale_errors": relative_scale_errors,
            "raw_result": raw_results
        }, f, indent=4)

    print(f"Average relative error: {np.mean(errors):.2%}")
    print(f"Average relative error with GT relative scale: {np.mean(errors_with_gt_relative_scale):.2%}")
    print(f"Average relative scale error: {np.mean(relative_scale_errors):.2%}")
    return csv_path


def run_volume_estimation_experiments_with_pointmaps(dataset_folder, volume_est_method, start_index=None, end_index=None):
    """
    Run tests on real dataset for volume estimation
    """
    dataset_folder_image = os.path.join(dataset_folder, "resized_images")
    dataset_folder_pointmaps = os.path.join(dataset_folder, "pointmaps")
    image_paths = sorted([os.path.join(dataset_folder_image, f) for f in os.listdir(dataset_folder_image) if f.endswith(".jpeg") or f.endswith(".jpg") or f.endswith(".png")])
    pointmap_paths = sorted([os.path.join(dataset_folder_pointmaps, f) for f in os.listdir(dataset_folder_pointmaps) if f.endswith(".npy")])
    
    if start_index is not None and end_index is not None:
        image_paths = image_paths[start_index:end_index]
        pointmap_paths = pointmap_paths[start_index:end_index]
    print(f"Dataset length {len(image_paths)} \n Image paths: {image_paths}")

    # Read the dataset properties from json
    json_path = os.path.join(dataset_folder, "plate_diameters.json")
    with open(json_path, "r") as f:
        dataset_properties = json.load(f)

    # Load VGGT camera poses including intrinsics
    vggt_poses_path = os.path.join(dataset_folder, "vggt_camera_poses.npz")
    vggt_intrinsics = None
    if os.path.exists(vggt_poses_path):
        vggt_data = np.load(vggt_poses_path)
        vggt_intrinsics = vggt_data['intrinsic']  # Shape: (N, 3, 3)
        print(f"Loaded VGGT intrinsics from {vggt_poses_path}")
        print(f"Intrinsics shape: {vggt_intrinsics.shape}")
    else:
        print(f"Warning: VGGT camera poses not found at {vggt_poses_path}. Intrinsics will be inferred.")

    
    # Initialize csv file to save results
    csv_path = f"{dataset_folder}/{volume_est_method}_volume_estimation_results_with_pointmaps.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["image_name", "gt_volume_ml", f"predicted_food_volume_ml({volume_est_method})", "volume_error", "error_percent", "volume_with_gt_relative_scale", "error_percent_with_gt_relative_scale", "relative_scale_error", "relative_scale_error_percent"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    errors = []
    errors_with_gt_relative_scale = []
    relative_scale_errors = []
    raw_results = {}

    for i, (image_path, pointmap_path) in enumerate(zip(image_paths, pointmap_paths)):
        IMAGE_NAME = image_path.split("/")[-1].split(".")[0]
        mask_folder = os.path.join(dataset_folder, f"masks_{IMAGE_NAME}")

        diameter = dataset_properties[IMAGE_NAME]["diameter"]
        gt_volume_ml = dataset_properties[IMAGE_NAME]["gt_volume_ml"]
        gt_relative_scale = dataset_properties[IMAGE_NAME]["gt_relative_scale"]

        # Read pointmap from the pointmap path
        pointmap = None
        intrinsics = None
        if os.path.exists(pointmap_path):
            pointmap = np.load(pointmap_path).astype(np.float32)
            pointmap = torch.from_numpy(pointmap)  # Convert to torch tensor if needed for the inference function
            print(f"Shape of pointmap: {pointmap.shape}")
            print(f"Data type of pointmap: {pointmap.dtype}")
            print(f"Loaded point map from {pointmap_path}")
            
            # Get corresponding intrinsics for this image
            if vggt_intrinsics is not None:
                intrinsics = vggt_intrinsics[i]  # Get intrinsics for image i
                print(f"Using VGGT intrinsics for image {i}: fx={intrinsics[0,0]:.2f}, fy={intrinsics[1,1]:.2f}")
        else:
            print(f"Point map not found at {pointmap_path}. Proceeding without point map for this sample.")


        volume_estimation_results = estimate_volumes(image_path, mask_folder, ACTUAL_PLATE_DIAMETER=diameter, GT_RELATIVE_SCALES=gt_relative_scale, pointmap=pointmap, intrinsics=intrinsics, volume_est_method=volume_est_method)   
        predicted_volume = volume_estimation_results[f"food_volume_ml({volume_est_method})"]
        predicted_relative_scale = volume_estimation_results["relative_scale_plate_food"]
        predicted_volume_with_gt_relative_scale = volume_estimation_results[f"food_volume_gt_relative_scale_ml({volume_est_method})"]

        raw_results[i] = volume_estimation_results

        print(f"\n=== SUMMARY FOR {IMAGE_NAME} ===")
        print(f"Ground truth volume (ml): {gt_volume_ml}")
        print(f"Predicted food convex hull volume (ml): {predicted_volume:.2f} mL")
        print(f"Predicted food convex hull volume with GT relative scale (ml): {predicted_volume_with_gt_relative_scale:.2f} mL")
        print(f"Relative error: {abs(predicted_volume - gt_volume_ml)/gt_volume_ml:.2%}")

        print(f"Ground truth relative scale: {gt_relative_scale}")
        print(f"Predicted relative scale: {predicted_relative_scale}")
        print(f"Relative scale error: {abs(predicted_relative_scale - gt_relative_scale)/gt_relative_scale:.2%}")


        # Save results to csv
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                "image_name": IMAGE_NAME,
                "gt_volume_ml": gt_volume_ml,
                f"predicted_food_volume_ml({volume_est_method})": f"{predicted_volume:.2f}",
                "volume_error": f"{(predicted_volume - gt_volume_ml):.2f}",
                "error_percent": f"{abs(predicted_volume - gt_volume_ml)/gt_volume_ml:.2%}",
                "volume_with_gt_relative_scale": f"{predicted_volume_with_gt_relative_scale:.2f}",
                "error_percent_with_gt_relative_scale": f"{abs(predicted_volume_with_gt_relative_scale - gt_volume_ml)/gt_volume_ml:.2%}",
                "relative_scale_error": f"{(predicted_relative_scale - gt_relative_scale):.4f}",
                "relative_scale_error_percent": f"{abs(predicted_relative_scale - gt_relative_scale)/gt_relative_scale:.2%}"
            })
        errors.append(abs(predicted_volume - gt_volume_ml)/gt_volume_ml)
        errors_with_gt_relative_scale.append(abs(predicted_volume_with_gt_relative_scale - gt_volume_ml)/gt_volume_ml)
        relative_scale_errors.append(abs(predicted_relative_scale - gt_relative_scale)/gt_relative_scale)
    
    print(f"FINISHED {len(image_paths)} EXPERIMENTS. RESULTS SAVED TO {csv_path}")

    # Save a json file
    json_path = f"{dataset_folder}/volume_estimation_results_with_pointmaps.json"
    with open(json_path, "w") as f:
        json.dump({
            "dataset_folder": dataset_folder,
            "start_index": start_index,
            "end_index": end_index,
            "average_relative_error": np.mean(errors),
            "errors": errors,
            "average_relative_error_with_gt_relative_scale": np.mean(errors_with_gt_relative_scale),
            "errors_with_gt_relative_scale": errors_with_gt_relative_scale,
            "average_relative_scale_error": np.mean(relative_scale_errors),
            "relative_scale_errors": relative_scale_errors,
            "raw_result": raw_results
        }, f, indent=4)

    print(f"Average relative error: {np.mean(errors):.2%}")
    print(f"Average relative error with GT relative scale: {np.mean(errors_with_gt_relative_scale):.2%}")
    print(f"Average relative scale error: {np.mean(relative_scale_errors):.2%}")

    return csv_path


def estimate_volumes(
        image_path, mask_path, volume_est_method, ACTUAL_PLATE_DIAMETER, GT_RELATIVE_SCALES=None, 
        pointmap=None, intrinsics=None
    ):
    image = load_image(image_path)
    IMAGE_NAME = image_path.split("/")[-1].split(".")[0]
    mask_plate = load_single_mask(mask_path, index=0)
    mask_food = load_single_mask(mask_path, index=1)

    if pointmap is not None:
        GEN_OUTPUT_DIR = f"{os.path.dirname(mask_path)}/generations_with_pointmaps/{IMAGE_NAME}"
    else:
        GEN_OUTPUT_DIR = f"{os.path.dirname(mask_path)}/generations_no_pointmaps/{IMAGE_NAME}"
    # GEN_OUTPUT_DIR = f"scripts_volume/real_dataset/generations/{IMAGE_NAME}"
    os.makedirs(GEN_OUTPUT_DIR, exist_ok=True)

    # image = load_image("/scratch/cl927/sam-3d-objects/scripts_volume/images/brunch.jpg")
    # mask_plate = load_single_mask("/scratch/cl927/sam-3d-objects/scripts_volume/masks", index=0)
    # mask_food = load_single_mask("/scratch/cl927/sam-3d-objects/scripts_volume/masks", index=1)

    volume_estimation_results = {}
    raw_volumes = np.zeros(2)
    raw_spans = np.zeros((2, 3))
    scales = np.zeros((2, 3))
    volumes = np.zeros(2)
    raw_volumes_with_estimation_method = np.zeros(2)
    volumes_with_estimation_method = np.zeros(2)
    spans = np.zeros((2, 3))

    # run model (generates a map)
    outputs = []
    objects = ["plate", "food"]
    for i, mask in enumerate([mask_plate, mask_food]):
        print(f"Running inference with mask {i}...")
        output = inference(
            image=image, 
            mask=mask,
            seed=42,
            pointmap=pointmap,
            intrinsics=intrinsics
        )

        outputs.append(output)
        torch.cuda.empty_cache()
        print("Cleared GPU cache from previous inference")
        # 6drotation_normalized, scale, shape, translation, translation_scale, coords_original, 
        # mesh (sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh.MeshExtractResult) 
        # gaussian (sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model.Gaussian)
        # glb: <trimesh.Trimesh(vertices.shape=(486892, 3), faces.shape=(973880, 3))>
        # gs: <sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model.Gaussian
        # pointmap, pointmap_colors

    for idx, (object, output) in enumerate(zip(objects, outputs)):
        
        # export gaussian splat and mesh
        print("==================================")
        print(f"Object {object}")
        # print(output["mesh"])
        # output["mesh"].export(f"mesh_raw.glb")
        # output["mesh"].export(f"mesh_raw.ply")

        output["glb"].export(f"{GEN_OUTPUT_DIR}/{IMAGE_NAME}_{object}.glb")
        output["glb"].export(f"{GEN_OUTPUT_DIR}/{IMAGE_NAME}_{object}.ply")
        
        # Check if mesh is watertight using the trimesh object
        glb_mesh = output["glb"]
        # is_watertight = glb_mesh.is_watertight
        # print(f"Mesh watertight: {is_watertight}")
        
        ### Compute convex hull volume for the mesh
        # (because the mesh itself is a thin surface shell, not the entire object)
        if volume_est_method == "convex_hull":
            convex_hull = glb_mesh.convex_hull
            convex_volume = convex_hull.volume
            print(f"Raw mesh volume: {glb_mesh.volume:.6f}")
            print(f"Convex hull volume: {convex_volume:.6f}")
            print(f"Mesh volume / Convex hull volume: {glb_mesh.volume/convex_volume:.2%}")
            
            # Export convex hull for visualization
            convex_hull.export(f"{GEN_OUTPUT_DIR}/{IMAGE_NAME}_convex_hull_{object}.ply")

            raw_volumes_with_estimation_method[idx] = convex_volume

        elif volume_est_method == "voxelize":
            mesh = trimesh.load(f"{GEN_OUTPUT_DIR}/{IMAGE_NAME}_{object}.ply")
            voxel_pitch = glb_mesh.extents.max() / 150
            voxelized = mesh.voxelized(pitch=voxel_pitch)
            voxelized = voxelized.fill()  # Fill the interior of the voxel grid to get a solid volume
            
            voxelized_mesh = voxelized.as_boxes()  # Convert to a mesh of boxes for volume calculation
            voxelized_volume = voxelized.volume

            voxelized_mesh.export(f"{GEN_OUTPUT_DIR}/{IMAGE_NAME}_voxelized_{object}.ply")

            raw_volumes_with_estimation_method[idx] = voxelized_volume


        
        scale = output["scale"].squeeze().cpu().numpy()
        span = glb_mesh.bounds[1] - glb_mesh.bounds[0]
        bbox_volume = span[0] * span[1] * span[2]
        print(f"Bounding box volume: {bbox_volume:.6f}")
        print(f"Mesh volume / BBox volume (occupancy): {glb_mesh.volume/bbox_volume:.2%}")
        print(f"Scale: {scale}")
        print(f"Span: {span}")

        
        scales[idx] = scale
        raw_volumes[idx] = glb_mesh.volume
        raw_spans[idx] = span

        
        if scale[0] == scale[1] == scale[2]:
            volume = glb_mesh.volume * (scale[0] ** 3)
            volume_with_est_method = raw_volumes_with_estimation_method[idx] * (scale[0] ** 3)
        else:
            volume = glb_mesh.volume * (scale[0] * scale[1] * scale[2])
            volume_with_est_method = raw_volumes_with_estimation_method[idx] * (scale[0] * scale[1] * scale[2])

        volumes[idx] = volume
        volumes_with_estimation_method[idx] = volume_with_est_method
        spans[idx] = span * scale

    # Save the two objects in a single scene for visual comparison
    scene_gs = make_scene(*outputs)
    scene_gs.save_ply(f"{GEN_OUTPUT_DIR}/{IMAGE_NAME}_combined.ply")

    
    # Diagnostic: Check real-world dimensions
    print(f"\n=== DIAGNOSTIC ===")
    print(f"Plate mesh diameter: {max(spans[0]):.4f} mesh units")
    print(f"Food mesh dimensions: X={spans[1][0]:.4f}, Y={spans[1][1]:.4f}, Z={spans[1][2]:.4f} mesh units")


    mesh_plate_diameter = max(spans[0])
    metric_factor = ACTUAL_PLATE_DIAMETER / mesh_plate_diameter # The length convertion factor from mesh units to real-world units
    print(f"Metric conversion factor: {metric_factor:.4f} (cm per mesh unit)")
    print(f"Food real-world dimensions: X={spans[1][0]*metric_factor:.2f}cm, Y={spans[1][1]*metric_factor:.2f}cm, Z={spans[1][2]*metric_factor:.2f}cm")

    # Convert the mesh volume to real-world units using the scale factor
    mesh_plate_volume = volumes[0]
    real_world_plate_volume = mesh_plate_volume * (metric_factor ** 3)
    print(f"Estimated plate volume in real-world units: {real_world_plate_volume:.6f} cubic centimeters")


    real_world_food_volume = volumes[1] * (metric_factor ** 3)
    real_world_food_volume_with_estimation_method = volumes_with_estimation_method[1] * (metric_factor ** 3)

    # DEBUG: If use the ground truth relative scale to calculate food volume. How does it compare?
    plate_scale = scales[0]
    food_scale_gt = plate_scale / GT_RELATIVE_SCALES
    food_volume_gt_relative_scale = raw_volumes_with_estimation_method[1] * (food_scale_gt[0] * food_scale_gt[1] * food_scale_gt[2])
    real_world_food_volume_gt_relative_scale = food_volume_gt_relative_scale * (metric_factor ** 3)

    # print(f"Estimated food volume in mesh units: {volumes[1]:.6f}")
    print(f"Estimated food volume with estimation method in mesh units: {volumes_with_estimation_method[1]:.6f}")
    # print(f"Mesh to convex hull volume ratio: {volumes[1]/convex_hull_volumes[1]:.6f}")
    # print(f"Estimated food volume in real-world units: {real_world_food_volume:.6f} cubic centimeters (mL)")
    print(f"Estimated food volume with estimation method in real-world units: {real_world_food_volume_with_estimation_method:.6f} cubic centimeters (mL)")
    print(f"Estimated food volume using GT relative scale: {real_world_food_volume_gt_relative_scale:.6f} cubic centimeters (mL)")

    # Final volume results
    print("\n=== FINAL RESULTS SUMMARY ===")
    print(f"METHOD: {volume_est_method}")
    print(f"Estimated food volume with estimation method (ml): {real_world_food_volume_with_estimation_method:.2f} mL")
    print(f"Estimated food volume using GT relative scale (ml): {real_world_food_volume_gt_relative_scale:.2f} mL")
    # print(f"Estimated food volume (ml): {real_world_food_volume:.2f} mL")
    # print(f"Raw volumes: {raw_volumes}")
    # print(f"Raw convex hull volumes: {raw_convex_hull_volumes}")
    # print(f"Raw spans: {raw_spans}")
    # print(f"Scales: {scales}")
    # print(f"Volume after scaling: {volumes}")
    # print(f"Convex hull volume after scaling: {convex_hull_volumes}")
    # print(f"Spans after scaling: {spans}")

    volume_estimation_results = {
        "plate_diameter": ACTUAL_PLATE_DIAMETER,
        "metric_conversion_factor": metric_factor,
        "relative_scale_plate_food": scales[0][0] / scales[1][0],
        "gt_relative_scale_plate_food": GT_RELATIVE_SCALES,
        f"food_volume_ml({volume_est_method})": real_world_food_volume_with_estimation_method,
        f"food_volume_gt_relative_scale_ml({volume_est_method})": real_world_food_volume_gt_relative_scale,
        # "food_volume_ml": real_world_food_volume,
        # "raw_volumes": raw_volumes.tolist(),
        # "raw_convex_hull_volumes": raw_convex_hull_volumes.tolist(),
        "raw_spans": raw_spans.tolist(),
        "spans": spans.tolist(),
        "scales": scales.tolist(),
        # "volumes": volumes.tolist(),
        # "convex_hull_volumes": convex_hull_volumes.tolist(),
    }

    return volume_estimation_results


class VoxelizeVolumeEstimator:
    """
    If we have the SAM3D results (mesh, scale, etc.)
    We can directly try voxelization-based volume estimation on mesh, without needed to regenerate.
    """

    @staticmethod
    def run_estimation_on_folder(dataset_folder, prev_json_path, generation_method="generations_no_pointmaps"):
        """
        Run voxelization-based volume estimation on all pre-generated meshes in a folder.
        
        Args:
            dataset_folder: Path to the dataset folder (e.g., /scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/pepper_bowl)
            prev_json_path: Path to the previous volume estimation results JSON file
            generation_method: Name of the generation folder (e.g., "generations_no_pointmaps" or "generations_with_pointmaps")
        """
        # Load previous results
        with open(prev_json_path, "r") as f:
            prev_results = json.load(f)
        
        # Get all generation folders
        dataset_folder_image = os.path.join(dataset_folder, "resized_images")
        image_paths = sorted([os.path.join(dataset_folder_image, f) for f in os.listdir(dataset_folder_image) 
                             if f.endswith(".jpeg") or f.endswith(".jpg") or f.endswith(".png")])
        
        start_index = prev_results.get("start_index", 0)
        end_index = prev_results.get("end_index", len(image_paths))
        
        if start_index is not None and end_index is not None:
            image_paths = image_paths[start_index:end_index]
        
        # Prepare output
        errors = []
        errors_with_gt_relative_scale = []
        relative_scale_errors = []
        raw_results = {}
        
        # Read ground truth data
        json_path = os.path.join(dataset_folder, "plate_diameters.json")
        with open(json_path, "r") as f:
            dataset_properties = json.load(f)
        
        # Process each image
        for i, image_path in enumerate(image_paths):
            IMAGE_NAME = image_path.split("/")[-1].split(".")[0]
            
            # Paths to pre-generated meshes
            gen_folder = os.path.join(dataset_folder, generation_method, IMAGE_NAME)
            plate_mesh_path = os.path.join(gen_folder, f"{IMAGE_NAME}_plate.ply")
            food_mesh_path = os.path.join(gen_folder, f"{IMAGE_NAME}_food.ply")
            
            # Get previous results for this sample
            prev_sample_results = prev_results["raw_result"][str(i)]
            
            # Get ground truth
            gt_volume_ml = dataset_properties[IMAGE_NAME]["gt_volume_ml"]
            gt_relative_scale = dataset_properties[IMAGE_NAME]["gt_relative_scale"]
            
            # Estimate voxelized volume
            voxel_results = VoxelizeVolumeEstimator.estimate_voxelized_volume(
                food_mesh_path, plate_mesh_path, prev_sample_results, gt_relative_scale
            )
            
            # Calculate errors
            predicted_volume = voxel_results["food_voxelized_volume_ml"]
            predicted_volume_with_gt_scale = voxel_results["food_voxelized_volume_gt_relative_scale_ml"]
            predicted_relative_scale = voxel_results["relative_scale_plate_food"]
            
            errors.append(abs(predicted_volume - gt_volume_ml) / gt_volume_ml)
            errors_with_gt_relative_scale.append(abs(predicted_volume_with_gt_scale - gt_volume_ml) / gt_volume_ml)
            relative_scale_errors.append(abs(predicted_relative_scale - gt_relative_scale))
            
            raw_results[i] = voxel_results
            
            print(f"\n=== SUMMARY FOR {IMAGE_NAME} ===")
            print(f"Ground truth volume (ml): {gt_volume_ml}")
            print(f"Predicted voxelized food volume (ml): {predicted_volume:.2f} mL")
            print(f"Predicted voxelized food volume with GT scale (ml): {predicted_volume_with_gt_scale:.2f} mL")
            print(f"Relative error: {abs(predicted_volume - gt_volume_ml)/gt_volume_ml:.2%}")
        
        # Save results
        if generation_method == "generations_no_pointmaps":
            output_json_path = os.path.join(dataset_folder, "voxelize_volume_estimation_results.json")
        elif generation_method == "generations_with_pointmaps":
            output_json_path = os.path.join(dataset_folder, "voxelize_volume_estimation_results_with_pointmaps.json")

        with open(output_json_path, "w") as f:
            json.dump({
                "dataset_folder": dataset_folder,
                "generation_method": generation_method,
                "start_index": start_index,
                "end_index": end_index,
                "average_relative_error": np.mean(errors),
                "errors": errors,
                "average_relative_error_with_gt_relative_scale": np.mean(errors_with_gt_relative_scale),
                "errors_with_gt_relative_scale": errors_with_gt_relative_scale,
                "average_relative_scale_error": np.mean(relative_scale_errors),
                "relative_scale_errors": relative_scale_errors,
                "raw_result": raw_results
            }, f, indent=4)
        
        print(f"\n=== OVERALL RESULTS ===")
        print(f"Average relative error: {np.mean(errors):.2%}")
        print(f"Average relative error with GT relative scale: {np.mean(errors_with_gt_relative_scale):.2%}")
        print(f"Average relative scale error: {np.mean(relative_scale_errors):.2%}")
        print(f"Results saved to {output_json_path}")
        
        return output_json_path
    
    @staticmethod
    def estimate_voxelized_volume(
            food_mesh_path, plate_mesh_path, prev_volume_estimation_results, gt_relative_scale=None
        ):
        """
        Estimate volume using voxelization on pre-generated meshes.
        
        Args:
            food_mesh_path: Path to the food mesh PLY file
            plate_mesh_path: Path to the plate mesh PLY file
            prev_volume_estimation_results: Dictionary containing previous SAM3D results (scales, metric conversion, etc.)
            gt_relative_scale: Ground truth relative scale between plate and food (optional)
        
        Returns:
            Dictionary with voxelized volume estimation results
        """
        # Load meshes
        food_mesh = trimesh.load(food_mesh_path)
        plate_mesh = trimesh.load(plate_mesh_path)
        
        # Extract previous results
        scales = np.array(prev_volume_estimation_results["scales"])
        metric_conversion_factor = prev_volume_estimation_results["metric_conversion_factor"]
        relative_scale_plate_food = prev_volume_estimation_results["relative_scale_plate_food"]
        
        plate_scale = scales[0]
        food_scale = scales[1]
        
        print(f"\n=== VOXELIZATION VOLUME ESTIMATION ===")
        print(f"Food mesh path: {food_mesh_path}")
        print(f"Loaded food mesh with {len(food_mesh.vertices)} vertices")
        print(f"Food scale from SAM3D: {food_scale}")
        print(f"Metric conversion factor: {metric_conversion_factor:.4f} cm/mesh_unit")
        
        # Voxelize the food mesh
        voxel_pitch = food_mesh.extents.max() / 150
        print(f"Voxel pitch: {voxel_pitch:.6f}")
        
        voxelized = food_mesh.voxelized(pitch=voxel_pitch)
        voxelized = voxelized.fill()  # Fill the interior
        
        # Export voxelized mesh for debugging
        voxelized_mesh = voxelized.as_boxes()
        voxelized_output_path = food_mesh_path.replace(".ply", "_voxelized.ply")
        voxelized_mesh.export(voxelized_output_path)
        print(f"Voxelized mesh saved to: {voxelized_output_path}")
        
        # Get voxelized volume
        raw_voxelized_volume = voxelized.volume
        print(f"Raw voxelized volume (mesh units^3): {raw_voxelized_volume:.6f}")
        
        # Apply SAM3D scale
        if food_scale[0] == food_scale[1] == food_scale[2]:
            voxelized_volume_scaled = raw_voxelized_volume * (food_scale[0] ** 3)
        else:
            voxelized_volume_scaled = raw_voxelized_volume * (food_scale[0] * food_scale[1] * food_scale[2])
        
        print(f"Voxelized volume after SAM3D scaling (mesh units^3): {voxelized_volume_scaled:.6f}")
        
        # Convert to real-world units (mL)
        real_world_voxelized_volume = voxelized_volume_scaled * (metric_conversion_factor ** 3)
        print(f"Voxelized volume in real-world units (mL): {real_world_voxelized_volume:.2f}")
        
        # Calculate volume using ground truth relative scale (if provided)
        real_world_voxelized_volume_gt_scale = None
        if gt_relative_scale is not None:
            food_scale_gt = plate_scale / gt_relative_scale
            voxelized_volume_gt_scale = raw_voxelized_volume * (food_scale_gt[0] * food_scale_gt[1] * food_scale_gt[2])
            real_world_voxelized_volume_gt_scale = voxelized_volume_gt_scale * (metric_conversion_factor ** 3)
            print(f"Voxelized volume using GT relative scale (mL): {real_world_voxelized_volume_gt_scale:.2f}")
        
        # Return results
        results = {
            "food_mesh_path": food_mesh_path,
            "plate_mesh_path": plate_mesh_path,
            "metric_conversion_factor": metric_conversion_factor,
            "relative_scale_plate_food": relative_scale_plate_food,
            "gt_relative_scale_plate_food": gt_relative_scale,
            "raw_voxelized_volume": raw_voxelized_volume,
            "voxelized_volume_scaled": voxelized_volume_scaled,
            "food_voxelized_volume_ml": real_world_voxelized_volume,
            "food_voxelized_volume_gt_relative_scale_ml": real_world_voxelized_volume_gt_scale,
            "scales": scales.tolist(),
            "voxel_pitch": voxel_pitch
        }
        
        return results

def main():

    dataset_folders = [
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/orange_plate",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/orange_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/mango_plate",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/mango_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/box_plate",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/box_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/gum_plate",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/gum_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/avocado_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/avocado_plate",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/egg_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/egg_plate",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/pepper_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/pepper_plate",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/potato_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/potato_plate"
    ]

    for idx, dataset_folder in enumerate(dataset_folders):
        print(f"\n\n==============================")
        print(f"RUNNING VOLUME ESTIMATION EXPERIMENTS ON DATASET {dataset_folder}")

    
        if USE_PRECOMPUTED_POINTMAPS: 
            print(f"RUNNING WITH POINTMAPS")
            run_volume_estimation_experiments_with_pointmaps(
                dataset_folder = dataset_folder,
                volume_est_method = "voxelize",
                start_index=0,
                end_index=1
            )            
        else:   
            print(f"RUNNING WITHOUT POINTMAP")
            run_volume_estimation_experiments(
                dataset_folder = dataset_folder,
                volume_est_method = "voxelize",
                start_index=0,
                end_index=1
            )

    

    # image_path = "/scratch/cl927/sam-3d-objects/scripts_volume/real_dataset/images/potato_1.jpeg"
    # mask_folder_path = "/scratch/cl927/sam-3d-objects/scripts_volume/real_dataset/masks_potato_1"

    # output = estimate_volumes(image_path, mask_folder_path, ACTUAL_PLATE_DIAMETER=20.5)
    # print("\n=== VOLUME ESTIMATION RESULTS ===")
    # for key, value in output.items():
    #     print(f"{key}: {value}")

def test():
    
    dataset_folders = [
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/orange_plate",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/orange_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/mango_plate",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/mango_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/box_plate",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/box_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/gum_plate",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/gum_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/avocado_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/avocado_plate",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/egg_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/egg_plate",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/pepper_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/pepper_plate",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/potato_bowl",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/potato_plate"
    ]

    total_folders = len(dataset_folders)

    for idx, dataset_folder in enumerate(dataset_folders):
        print(f"\n\n==============================")
        print(f"[{idx+1}/{total_folders}]RUNNING VOLUME ESTIMATION EXPERIMENTS ON DATASET {dataset_folder}")


        # For a single mesh pair
        VoxelizeVolumeEstimator.run_estimation_on_folder(
            dataset_folder=dataset_folder,
            prev_json_path=f"{dataset_folder}/volume_estimation_results.json",
            generation_method="generations_no_pointmaps"
        )

        VoxelizeVolumeEstimator.run_estimation_on_folder(
            dataset_folder=dataset_folder,
            prev_json_path=f"{dataset_folder}/volume_estimation_results_with_pointmaps.json",
            generation_method="generations_with_pointmaps"
        )

    


    # # For a single mesh pair
    # VoxelizeVolumeEstimator.run_estimation_on_folder(
    #     dataset_folder="/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/egg_plate",
    #     prev_json_path="/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_new/egg_plate/volume_estimation_results_with_pointmaps.json",
    #     generation_method="generations_with_pointmaps"
    # )

if __name__ == "__main__":
    # load model
    tag = "hf"

    argparser = argparse.ArgumentParser(description="Run volume estimation experiments on a dataset.")
    argparser.add_argument("--pointmap", action="store_true", help="Whether to use precomputed pointmaps for volume estimation.")
    args = argparser.parse_args()

    
    # OPTIMIZATION: Use config without depth model when only using pointmaps
    # This saves ~6.5GB GPU memory by not loading MoGE
    if args.pointmap:
        USE_PRECOMPUTED_POINTMAPS = True
    else:
        USE_PRECOMPUTED_POINTMAPS = False
    
    if USE_PRECOMPUTED_POINTMAPS:
        config_path = f"checkpoints/{tag}/pipeline_no_depth.yaml"
        print("Using pipeline_no_depth.yaml - MoGE depth model will NOT be loaded.")
        print("NOTE: You MUST provide pointmaps to inference, or it will fail.")
    else:
        config_path = f"checkpoints/{tag}/pipeline.yaml"
        print("Using pipeline.yaml - MoGE depth model will be loaded.")
    
    inference = Inference(config_path, compile=False)

    # main()
    test()