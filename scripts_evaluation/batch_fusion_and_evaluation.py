"""
This script is for batch multiview fusion of single-view predictions.
"""

import argparse
import os
import open3d as o3d
import numpy as np
import json
import trimesh

from align_without_vggt import two_view_fusion
from chamfer_distance_prep import sample_points_from_mesh, count_sample_points
from chamfer_distance_evaluation import load_point_cloud, chamfer_distance_evaluation

# Helper function to load single view evaluation results
def load_single_view_evaluation_results(object_folder, view_index):
    evaluation_results_path = os.path.join(object_folder, f"sam3d_singleview_predictions/chamfer_distance_evaluation.json")
    with open(evaluation_results_path, "r") as f:
        evaluation_results = json.load(f)
    
    view_key = f"00{view_index}_voxels"

    filtered_attributes = ["bidirectional_chamfer_distance", "unidirectional_chamfer_distance", "num_points"]
    filtered_results = {
        attr: evaluation_results.get(view_key, {}).get(attr, None) for attr in filtered_attributes
    }
    return filtered_results

    
def main():
    argparser = argparse.ArgumentParser(description="Batch fusion and evaluation of SAM3D single-view predictions")
    argparser.add_argument("--dataset_folder", type=str, required=True, help="Path to dataset folder containing individual evaluation results")
    argparser.add_argument("--start_index", type=int, default=0, help="Start index for processing dataset folders")
    argparser.add_argument("--end_index", type=int, default=None, help="End index for processing dataset folders")
    argparser.add_argument("--num_views", type=int, required=True, help="Number of views for evaluation")
    argparser.add_argument("--view1", type=int, required=True, help="Which view to use the first view for fusion (starting from 0)")
    argparser.add_argument("--view2", type=int, required=True, help="Which view to use the second view for fusion (starting from 0)")
    args = argparser.parse_args()

    dataset_folders = sorted([os.path.join(args.dataset_folder, f) for f in os.listdir(args.dataset_folder) if os.path.isdir(os.path.join(args.dataset_folder, f))])
    if args.start_index is not None and args.end_index is not None:
        dataset_folders = dataset_folders[args.start_index:args.end_index]
    print(f"dataset_folders (length {len(dataset_folders)}): {dataset_folders}")

    for i, dataset_folder in enumerate(dataset_folders):
        print(f"========Progress {i+1}/{len(dataset_folders)}========")
        print(f"Processing dataset folder: {dataset_folder}")

        # Fusion
        fusion_output_folder = two_view_fusion(
            voxel_one_path=os.path.join(dataset_folder, f"sam3d_singleview_predictions/00{args.view1}_voxels.ply"),
            voxel_two_path=os.path.join(dataset_folder, f"sam3d_singleview_predictions/00{args.view2}_voxels.ply"),
            voxel_one_npz_path=os.path.join(dataset_folder, f"sam3d_singleview_predictions/00{args.view1}_sam3d_outputs.npz"),
            voxel_two_npz_path=os.path.join(dataset_folder, f"sam3d_singleview_predictions/00{args.view2}_sam3d_outputs.npz"),
            output_dir=os.path.join(dataset_folder, "sam3d_fusion"),
            view1=args.view1,
            view2=args.view2,
            debug=False
        )

        # Evaluation
        ground_truth_mesh_path = os.path.join(dataset_folder, "mesh.ply")
        gt_mesh = o3d.io.read_triangle_mesh(ground_truth_mesh_path)

        avg_voxels = f"{fusion_output_folder}/fused_average_voxels_views{args.view1}_{args.view2}.ply"
        min_entropy_voxels = f"{fusion_output_folder}/fused_min_entropy_voxels_views{args.view1}_{args.view2}.ply"
        fused_voxels = [avg_voxels, min_entropy_voxels]
        fusion_methods = ["average", "min_entropy"]
        
        eval_results = {}

        # Fetch single view evaluation results for the two views used in fusion
        single_view_eval_results_view1 = load_single_view_evaluation_results(dataset_folder, args.view1)
        single_view_eval_results_view2 = load_single_view_evaluation_results(dataset_folder, args.view2)
        if single_view_eval_results_view1 is None or single_view_eval_results_view2 is None:
            raise ValueError(f"Could not find single view evaluation results for views {args.view1} and {args.view2} in {dataset_folder}")
        
        eval_results[f"view_00{args.view1}"] = single_view_eval_results_view1
        eval_results[f"view_00{args.view2}"] = single_view_eval_results_view2

        for voxel_file, fusion_method in zip(fused_voxels, fusion_methods):

            print(f"Calculating Chamfer distance for {voxel_file}...")

            pred_pc = load_point_cloud(voxel_file)
            # Sample the same number of points from the GT mesh as the predicted point cloud
            gt_pc = sample_points_from_mesh(gt_mesh, num_points=pred_pc.shape[0])
            chamfer_results = chamfer_distance_evaluation(
                gt_pc, pred_pc, output_dir=fusion_output_folder, debug=False
            )

            
            voxel_file_name = voxel_file.split("/")[-1].split(".")[0]
            print(f"voxels_file_name: {voxel_file_name}")

            eval_results[f"{fusion_method}_views{args.view1}_{args.view2}"] = {
                "bidirectional_chamfer_distance": chamfer_results["bidirectional_chamfer_distance"],
                "unidirectional_chamfer_distance": chamfer_results["unidirectional_chamfer_distance"],
                "num_points": chamfer_results["num_points"]
            }

    
        # Save overall results for the object
        chamfer_distance_eval_file = os.path.join(fusion_output_folder, f"chamfer_distance_evaluation_view{args.view1}_{args.view2}.json")
        with open(chamfer_distance_eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)

        print(f"Finished processing {dataset_folder}")
    


        
if __name__ == "__main__":
    main()