import argparse
import os
import sys
import open3d as o3d
import numpy as np
import json
from datetime import datetime

from chamfer_distance_prep import sample_points_from_mesh, count_sample_points
from chamfer_distance_evaluation import load_point_cloud, chamfer_distance_evaluation

sys.path.append("scripts_volume")
from render_from_poses import reproject_folder_and_evaluate_iou

def main():
    argparser = argparse.ArgumentParser(description="Evaluate SAM3D multiview outputs")
    argparser.add_argument("--dataset_folder", type=str, required=True, help="Path to dataset folder")
    argparser.add_argument("--start_index", type=int, default=0, help="Start index for processing dataset folders")
    argparser.add_argument("--end_index", type=int, default=None, help="End index for processing dataset folders")
    argparser.add_argument("--chamfer", action='store_true', help="Whether to run Chamfer distance evaluation")
    argparser.add_argument("--reprojection", action='store_true', help="Whether to run reprojection evaluation")

    args = argparser.parse_args()

    dataset_folders = sorted([os.path.join(args.dataset_folder, f) for f in os.listdir(args.dataset_folder) if os.path.isdir(os.path.join(args.dataset_folder, f))])
    if args.start_index is not None and args.end_index is not None:
        dataset_folders = dataset_folders[args.start_index:args.end_index]

    print(f"dataset_folders (length {len(dataset_folders)}): {dataset_folders}")

    for dataset_idx, dataset_folder in enumerate(dataset_folders):
        print(f"==================================")
        print(f"[{dataset_idx}/{len(dataset_folders)}] Processing dataset folder: {dataset_folder}")
        print(f"==================================")
        
        # Assume sam3d_singleview_predictions folder already exists
        generation_output_dir = os.path.join(dataset_folder, "sam3d_singleview_predictions")
        if not os.path.exists(generation_output_dir):
            print(f"ERROR: Generation output not found at {generation_output_dir}")
            print(f"Skipping {dataset_folder}")
            continue

        ground_truth_mesh_path = os.path.join(dataset_folder, "mesh.ply")
        gt_mesh = o3d.io.read_triangle_mesh(ground_truth_mesh_path)

        generated_voxels = sorted([f for f in os.listdir(generation_output_dir) if f.endswith("voxels.ply")])
        # generated_meshes = sorted([f for f in os.listdir(generation_output_dir) if f.endswith("mesh.glb")])
        generated_meshes = sorted([f for f in os.listdir(generation_output_dir) if f.endswith("mesh.ply")])
        generated_sam3d_info = sorted([f for f in os.listdir(generation_output_dir) if f.endswith("sam3d_outputs.npz")])

        # Evaluate chamfer for each generated mesh file
        total_chamfer_results = {
                "average_bidirectional_chamfer_distance": 0.0,
                "average_unidirectional_chamfer_distance": 0.0,
                "average_number_of_points": 0.0
            }
        total_bidirectional_cd = 0.0
        total_unidirectional_cd = 0.0
        total_number_of_points = 0

        # Initial Rotations to align generation with ground truth
        initial_rotations = []  

        if args.chamfer:
            
            # Mesh Chamfer Distance Evaluation
            for mesh_file, sam3d_info_file in zip(generated_meshes, generated_sam3d_info):
                num_points = 10000
                print(f"Calculating Chamfer distance for {mesh_file}...")

                pred_path = os.path.join(generation_output_dir, mesh_file)
                pred_mesh = o3d.io.read_triangle_mesh(pred_path)

                pred_pc = sample_points_from_mesh(pred_mesh, num_points = num_points)
                # Sample the same number of points from the GT mesh as the predicted point cloud
                gt_pc = sample_points_from_mesh(gt_mesh, num_points = num_points)
                
                chamfer_results, best_initial_rotation = chamfer_distance_evaluation(
                    gt_pc, pred_pc, output_dir=generation_output_dir, debug=False
                )

                if not np.allclose(best_initial_rotation, np.eye(3)):
                    initial_rotations.append(best_initial_rotation)
                else:
                    initial_rotations.append(None)

                mesh_file_name = mesh_file.split("/")[-1].split(".")[0]
                print(f"mesh_file_name: {mesh_file_name}")

                # Add the stats to the overall results dictionary
                total_chamfer_results[mesh_file_name] = chamfer_results
                total_bidirectional_cd += chamfer_results['bidirectional_chamfer_distance']
                total_unidirectional_cd += chamfer_results['unidirectional_chamfer_distance']
                total_number_of_points += pred_pc.shape[0]
            
            # Calculate average Chamfer distances across all views
            num_views = len(generated_meshes)
            total_chamfer_results["average_bidirectional_chamfer_distance"] = total_bidirectional_cd / num_views
            total_chamfer_results["average_unidirectional_chamfer_distance"] = total_unidirectional_cd / num_views
            total_chamfer_results["average_number_of_points"] = total_number_of_points / num_views

            # Save overall results for the object
            chamfer_distance_eval_file = os.path.join(generation_output_dir, "chamfer_distance_evaluation.json")
            with open(chamfer_distance_eval_file, 'w') as f:
                json.dump(total_chamfer_results, f, indent=2)

            print(f"Finished Chamfer evaluation for {dataset_folder}")

            initial_rotations_file = os.path.join(generation_output_dir, "best_initial_rotations.json")
            with open(initial_rotations_file, 'w') as f:
                json.dump({"initial_rotations": [rotation.tolist() if rotation is not None else None for rotation in initial_rotations]}, f, indent=2)
        

        ## Reprojection Evaluation
        if args.reprojection:
            print(f"===REPROJECTION EVALUATION=== for {dataset_folder}...")
            print(f"Initial rotations are {initial_rotations}")
            rendering_output_folder = os.path.join(generation_output_dir, "reprojection")
            reproject_folder_and_evaluate_iou(
                generation_output_dir, 
                os.path.join(dataset_folder, "transforms.json"), 
                rendering_output_folder,
                initial_transforms=initial_rotations,
                evaluate_iou = True,
            )
        
        ## Volume Evaluation
        # Placeholder

    print(f"FINISHED (start_idx {args.start_index}, end_idx {args.end_index}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



if __name__ == "__main__":
    main()
