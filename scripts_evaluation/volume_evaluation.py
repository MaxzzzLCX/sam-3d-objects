"""
This script is for volume evaluation of generated 3D meshes.
NOTE: volume evaluation for SAM3D is different from TRELLIS. Because TRELLIS's mesh is already closed interior mesh.
"""
import argparse
import trimesh
import os
import json
import csv
import numpy as np

def get_volume(mesh_path, method='original', alpha=None, voxel_pitch=None):
    """
    Read a mesh from the given path and calculate its volume
    
    Args:
        mesh_path: Path to the mesh file
        method: Method to use for volume calculation
            - 'original': Use the mesh as-is
            - 'convex_hull': Create convex hull of the mesh
            - 'alpha_shape': Create alpha shape (tighter fit than convex hull)
            - 'voxelize': Fill interior using voxelization
        alpha: Alpha value for alpha shape (larger = looser fit). Default: auto-determined
        voxel_pitch: Voxel size for voxelization. Default: auto-determined from mesh bounds
    """
    mesh = trimesh.load(mesh_path)
    original_mesh = mesh.copy()
    
    if method == 'convex_hull':
        mesh = mesh.convex_hull
        save_path = mesh_path.replace(".ply", "_convex_hull.ply")
        mesh.export(save_path)
        print(f"Saved convex hull mesh to {save_path}")
        
    elif method == 'alpha_shape':
        # Alpha shape works on point clouds, so we sample points from the mesh
        points = mesh.sample(10000)  # Sample points from mesh surface
        if alpha is None:
            # Auto-determine alpha based on average edge length
            alpha = mesh.extents.max() * 0.03

        
        try:
            # Try Open3D first (most widely used and well-maintained)
            import open3d as o3d
            
            # Convert to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Estimate normals for better reconstruction
            pcd.estimate_normals()
            
            # Create alpha shape
            alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            
            # Convert back to trimesh
            vertices = np.asarray(alpha_mesh.vertices)
            faces = np.asarray(alpha_mesh.triangles)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            print(f"Using Open3D alpha shape with alpha={alpha:.6f}")
            
        except Exception as e:
            print(f"Warning: Alpha shape failed ({e}), using original mesh")
            mesh = original_mesh
        
        save_path = mesh_path.replace(".ply", "_alpha_shape.ply")
        mesh.export(save_path)
        print(f"Saved alpha shape mesh to {save_path}")
        
    elif method == 'voxelize':
        # Voxelize the mesh and fill the entire interior volume
        if voxel_pitch is None:
            # Auto-determine voxel pitch based on mesh size
            voxel_pitch = mesh.extents.max() / 150  # 100 voxels along longest axis
        
        try:
            # First voxelize the surface
            voxelized = mesh.voxelized(pitch=voxel_pitch)
            
            # Fill the interior (this is the key step!)
            voxelized = voxelized.fill()
            
            # Convert back to mesh
            mesh = voxelized.as_boxes()
            save_path = mesh_path.replace(".ply", "_voxelized.ply")
            mesh.export(save_path)
            print(f"Saved voxelized mesh (interior filled) to {save_path}")
            print(f"Voxel pitch: {voxel_pitch:.6f}")
        except Exception as e:
            print(f"Warning: Voxelization failed ({e}), using original mesh")
            mesh = original_mesh

    return mesh.volume

def volume_evaluation(generated_mesh_path, ground_truth_mesh_path, method='alpha_shape'):
    """
    Evaluate the volume of the generated mesh against the ground truth mesh.
    
    Args:
        generated_mesh_path: Path to generated mesh
        ground_truth_mesh_path: Path to ground truth mesh
        method: Method to use for volume calculation ('original', 'convex_hull', 'alpha_shape', 'voxelize')
    """
    # Try different methods
    generated_volume_original = get_volume(generated_mesh_path, method='original')
    generated_volume = get_volume(generated_mesh_path, method=method)
    gt_volume = get_volume(ground_truth_mesh_path, method='original')
    gt_volume_method = get_volume(ground_truth_mesh_path, method=method)

    volume_error = abs(generated_volume - gt_volume)
    volume_error_percentage = (volume_error / gt_volume) * 100 if gt_volume > 0 else float('inf')
    volume_error_method = abs(generated_volume - gt_volume_method)
    volume_error_percentage_method = (volume_error_method / gt_volume_method) * 100 if gt_volume_method > 0 else float('inf')

    # print(f"Generated Mesh Volume (Original): {generated_volume_original:.6f}")
    print(f"Generated Mesh Volume ({method}): {generated_volume:.6f}")
    print(f"Ground Truth Mesh Volume: {gt_volume:.6f}")
    print(f"Ground Truth Mesh Volume ({method}): {gt_volume_method:.6f}")
    print(f"Volume Error (compared to original gt): {volume_error:.6f}")
    print(f"Volume Error Percentage (compared to original gt): {volume_error_percentage:.2f}%")
    print(f"Volume Error (compared to {method} gt): {volume_error_method:.6f}")
    print(f"Volume Error Percentage (compared to {method} gt): {volume_error_percentage_method:.2f}%")

    return {
        "generated_volume": generated_volume,
        "ground_truth_volume": gt_volume,
        "ground_truth_volume_method": gt_volume_method,
        "volume_error": volume_error,
        "volume_error_percentage": volume_error_percentage,
        "volume_error_method": volume_error_method,
        "volume_error_percentage_method": volume_error_percentage_method,
        "method": method
    }

def folder_volume_evaluation(generation_output_dir, ground_truth_mesh_path, method='alpha_shape'):
    """
    Evaluate the volume for all generated meshes in a folder against the ground truth mesh.
    """
    generated_meshes = sorted([f for f in os.listdir(generation_output_dir) if f.endswith("mesh.ply")])
    
    predicted_volumes = []
    gt_volumes = []
    gt_volumes_method = []
    volume_errors = []
    volume_error_percentages = []
    volume_errors_method = []
    volume_error_percentages_method = []

    for mesh_file in generated_meshes:
        print(f"Evaluating volume for {mesh_file}...")
        generated_mesh_path = os.path.join(generation_output_dir, mesh_file)
        mesh_vol_result = volume_evaluation(generated_mesh_path, ground_truth_mesh_path, method=method)
        predicted_volumes.append(mesh_vol_result["generated_volume"])
        gt_volumes.append(mesh_vol_result["ground_truth_volume"])
        gt_volumes_method.append(mesh_vol_result["ground_truth_volume_method"])
        volume_errors.append(mesh_vol_result["volume_error"])
        volume_error_percentages.append(mesh_vol_result["volume_error_percentage"])
        volume_errors_method.append(mesh_vol_result["volume_error_method"])
        volume_error_percentages_method.append(mesh_vol_result["volume_error_percentage_method"])
    
    # Write in JSON in this folder
    json_output_path = os.path.join(generation_output_dir, "volume_evaluation_results.json")
    with open(json_output_path, "w") as f:
        json.dump({
            "mean percent error": np.mean(volume_error_percentages),
            "std percent error": np.std(volume_error_percentages),
            "mean percent error (compared to method gt)": np.mean(volume_error_percentages_method),
            "std percent error (compared to method gt)": np.std(volume_error_percentages_method),
            "predicted_volumes": predicted_volumes,
            "gt_volumes": gt_volumes,
            "gt_volumes_method": gt_volumes_method,
            "volume_errors": volume_errors,
            "volume_errors_method": volume_errors_method,
            "volume_error_percentages": volume_error_percentages,
            "volume_error_percentages_method": volume_error_percentages_method
        }, f, indent=2)
    
    print(f"Saved volume evaluation results in {json_output_path}")
    print(f"Mean Volume Error Percentage: {np.mean(volume_error_percentages):.2f}%")
    print(f"Std Volume Error Percentage: {np.std(volume_error_percentages):.2f}%")
    return {
        "predicted_volumes": predicted_volumes,
        "gt_volumes": gt_volumes,
        "gt_volumes_method": gt_volumes_method,   
        "volume_errors": volume_errors,
        "volume_errors_method": volume_errors_method,
        "volume_error_percentages": volume_error_percentages,
        "volume_error_percentages_method": volume_error_percentages_method,
    }

def batch_volume_evaluation(
        dataset_folder, start_index=0, end_index=None, 
        method='alpha_shape', 
        generation_method="sam3d_singleview_predictions", 
        overall_json_output_path=None
    ):
    """
    Batch evaluate the volume for multiple generated mesh folders.
    """
    generation_output_dirs = sorted([f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))])
    if start_index is not None and end_index is not None:
        generation_output_dirs = generation_output_dirs[start_index:end_index]
    print(f"Evaluating volume for folders (start {start_index}, end {end_index}): {generation_output_dirs}...")
    print(f"Total length of generation_output_dirs: {len(generation_output_dirs)}")
    
    all_results = {}
    for generation_output_dir in generation_output_dirs:
        print(f"===============================")
        print(f"Evaluating volume for folder: {generation_output_dir}...")
        folder_results = folder_volume_evaluation(
            os.path.join(dataset_folder, f"{generation_output_dir}/{generation_method}"), 
            os.path.join(dataset_folder, f"{generation_output_dir}/mesh.ply"),
            method=method
        )
        all_results[generation_output_dir] = folder_results
    
    # Write overall results in a JSON file
    with open(overall_json_output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate the volume of generated 3D meshes against ground truth.")
    parser.add_argument("--generation_output_dirs", nargs="+", help="List of directories containing generated mesh files to evaluate.")
    parser.add_argument("--ground_truth_mesh_path", type=str, help="Path to the ground truth mesh file.")
    args = parser.parse_args()


    # folder_volume_evaluation(
    #     # args.generation_output_dirs, 
    #     # args.ground_truth_mesh_path
    #     "/scratch/cl927/datasets/Toys4k/debug/000/sam3d_singleview_predictions",
    #     "/scratch/cl927/datasets/Toys4k/debug/000/mesh.ply",
    #     method="voxelize"
    # )
    dataset_folder = "/scratch/cl927/datasets/Toys4k/subset_foodlike"
    generation_method = "trellis_multiimage_outputs"
    start_index = 14
    end_index = 22
    batch_volume_evaluation(
        dataset_folder=dataset_folder,
        start_index=start_index,
        end_index=end_index,
        method="voxelize",
        # generation_method="sam3d_singleview_predictions",
        generation_method=generation_method,
        overall_json_output_path=os.path.join(dataset_folder, f"{generation_method}_overall_volume_evaluation_results_{start_index}_{end_index}.json")
    )

if __name__ == "__main__":
    main()