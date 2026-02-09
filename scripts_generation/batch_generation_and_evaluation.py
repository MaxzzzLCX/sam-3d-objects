import argparse
import os
import open3d as o3d
import numpy as np
import json
from datetime import datetime

from generate_sam3d_multiview import generate_sam3d_outputs
from chamfer_distance_prep import sample_points_from_mesh, count_sample_points
from chamfer_distance_evaluation import load_point_cloud, chamfer_distance_evaluation


def main():
    argparser = argparse.ArgumentParser(description="Generate SAM3D multiview outputs")
    argparser.add_argument("--dataset_folder", type=str, required=True, help="Path to dataset folder")
    argparser.add_argument("--start_index", type=int, default=0, help="Start index for processing dataset folders")
    argparser.add_argument("--end_index", type=int, default=None, help="End index for processing dataset folders")

    args = argparser.parse_args()

    dataset_folders = sorted([os.path.join(args.dataset_folder, f) for f in os.listdir(args.dataset_folder) if os.path.isdir(os.path.join(args.dataset_folder, f))])
    if args.start_index is not None and args.end_index is not None:
        dataset_folders = dataset_folders[args.start_index:args.end_index]

    print(f"dataset_folders (length {len(dataset_folders)}): {dataset_folders}")

    for dataset_folder in dataset_folders:
        print(f"Processing dataset folder: {dataset_folder}")

        image_paths = sorted([os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".png")])
        masks_paths = image_paths

        output_info = generate_sam3d_outputs(dataset_folder, image_paths, masks_paths)
        generation_output_dir = output_info['output_dir']

        
        ground_truth_mesh_path = os.path.join(dataset_folder, "mesh.ply")
        gt_mesh = o3d.io.read_triangle_mesh(ground_truth_mesh_path)

        generated_voxels = sorted([f for f in os.listdir(generation_output_dir) if f.endswith("voxels.ply")])
        generated_sam3d_info = sorted([f for f in os.listdir(generation_output_dir) if f.endswith("sam3d_outputs.npz")])

        # Evaluate chamfer for each generated voxel file
        total_chamfer_results = {
                "average_bidirectional_chamfer_distance": 0.0,
                "average_unidirectional_chamfer_distance": 0.0,
                "average_number_of_points": 0.0
            }
        total_bidirectional_cd = 0.0
        total_unidirectional_cd = 0.0
        total_number_of_points = 0
        
        for voxels_file, sam3d_info_file in zip(generated_voxels, generated_sam3d_info):

            print(f"Calculating Chamfer distance for {voxels_file}...")

            

            pred_path = os.path.join(generation_output_dir, voxels_file)
            sam3d_info_path = os.path.join(generation_output_dir, sam3d_info_file)

            pred_pc = load_point_cloud(pred_path)
            # Sample the same number of points from the GT mesh as the predicted point cloud
            gt_pc = sample_points_from_mesh(gt_mesh, num_points=pred_pc.shape[0])
            chamfer_results = chamfer_distance_evaluation(
                gt_pc, pred_pc, output_dir=generation_output_dir, debug=False
            )

            
            voxel_file_name = voxels_file.split("/")[-1].split(".")[0]
            print(f"voxels_file_name: {voxel_file_name}")

            # Add the stats to the overall results dictionary
            total_chamfer_results[voxel_file_name] = chamfer_results
            total_bidirectional_cd += chamfer_results['bidirectional_chamfer_distance']
            total_unidirectional_cd += chamfer_results['unidirectional_chamfer_distance']
            total_number_of_points += pred_pc.shape[0]
        
        # Calculate average Chamfer distances across all views
        num_views = len(generated_voxels)
        total_chamfer_results["average_bidirectional_chamfer_distance"] = total_bidirectional_cd / num_views
        total_chamfer_results["average_unidirectional_chamfer_distance"] = total_unidirectional_cd / num_views
        total_chamfer_results["average_number_of_points"] = total_number_of_points / num_views

        # Save overall results for the object
        chamfer_distance_eval_file = os.path.join(generation_output_dir, "chamfer_distance_evaluation.json")
        with open(chamfer_distance_eval_file, 'w') as f:
            json.dump(total_chamfer_results, f, indent=2)

        print(f"Finished processing {dataset_folder}")
    
    print(f"FINISHED (start_idx {args.start_index}, end_idx {args.end_index}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



if __name__ == "__main__":
    main()