"""
After fusion evaluation is run. This script aggregates the evaluation stats.
"""

import argparse
import os
import csv
import json
import numpy as np
from scipy import stats

def stats_in_csv(dataset_folder_path, start_index, end_index, csv_file_path):
    """
    Aggregates evaluation stats from inidividual folders to a single CSV file.
    """

    # CSV header
    # csv_file_path = os.path.join(dataset_folder_path, "evaluation_stats.csv")
    csv_header = ["view1_idx", "view2_idx", "view_1_bcd", "view_2_bcd", "fuse_avg_bcd", "fuse_avg_bcd_change", "fuse_min_entropy_bcd", "fuse_min_entropy_bcd_change", "view_1_ucd", "view_2_ucd", "fuse_avg_ucd", "fuse_avg_ucd_change", "fuse_min_entropy_ucd", "fuse_min_entropy_ucd_change", "dataset_folder"]
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)

    dataset_folders = sorted([os.path.join(dataset_folder_path, f) for f in os.listdir(dataset_folder_path) if os.path.isdir(os.path.join(dataset_folder_path, f))])
    if start_index is not None and end_index is not None:
        dataset_folders = dataset_folders[start_index:end_index]
    
    print(f"dataset_folders (length {len(dataset_folders)}): {dataset_folders}")

    for dataset_folder in dataset_folders:

        fusion_eval_result_folder = os.path.join(dataset_folder, "sam3d_fusion")
        fusion_eval_result_files = sorted([f for f in os.listdir(fusion_eval_result_folder) if f.startswith("chamfer_distance_evaluation_view") and f.endswith(".json")])
        print(f"fusion_eval_result_files length: {len(fusion_eval_result_files)}")

        for fusion_eval_file in fusion_eval_result_files:
            fusion_eval_file_path = os.path.join(fusion_eval_result_folder, fusion_eval_file)

            view1_idx = fusion_eval_file.split("view")[1].split("_")[0]
            view2_idx = fusion_eval_file[-6] # Last char before ".json"
            with open(fusion_eval_file_path, "r") as f:
                fuse_eval_results = json.load(f)

            # Calculate improvements over single view
            best_bcd_single_view = min(fuse_eval_results[f"view_00{view1_idx}"]["bidirectional_chamfer_distance"], fuse_eval_results[f"view_00{view2_idx}"]["bidirectional_chamfer_distance"])
            best_ucd_single_view = min(fuse_eval_results[f"view_00{view1_idx}"]["unidirectional_chamfer_distance"], fuse_eval_results[f"view_00{view2_idx}"]["unidirectional_chamfer_distance"])

        

            # Save stats to CSV
            with open(csv_file_path, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    view1_idx, 
                    view2_idx, 
                    fuse_eval_results[f"view_00{view1_idx}"]["bidirectional_chamfer_distance"],
                    fuse_eval_results[f"view_00{view2_idx}"]["bidirectional_chamfer_distance"],
                    fuse_eval_results[f"average_views{view1_idx}_{view2_idx}"]["bidirectional_chamfer_distance"],
                    (fuse_eval_results[f"average_views{view1_idx}_{view2_idx}"]["bidirectional_chamfer_distance"] - best_bcd_single_view) / best_bcd_single_view,
                    fuse_eval_results[f"min_entropy_views{view1_idx}_{view2_idx}"]["bidirectional_chamfer_distance"],
                    (fuse_eval_results[f"min_entropy_views{view1_idx}_{view2_idx}"]["bidirectional_chamfer_distance"] - best_bcd_single_view) / best_bcd_single_view,
                    fuse_eval_results[f"view_00{view1_idx}"]["unidirectional_chamfer_distance"],
                    fuse_eval_results[f"view_00{view2_idx}"]["unidirectional_chamfer_distance"],
                    fuse_eval_results[f"average_views{view1_idx}_{view2_idx}"]["unidirectional_chamfer_distance"],
                    (fuse_eval_results[f"average_views{view1_idx}_{view2_idx}"]["unidirectional_chamfer_distance"] - best_ucd_single_view) / best_ucd_single_view,
                    fuse_eval_results[f"min_entropy_views{view1_idx}_{view2_idx}"]["unidirectional_chamfer_distance"],
                    (fuse_eval_results[f"min_entropy_views{view1_idx}_{view2_idx}"]["unidirectional_chamfer_distance"] - best_ucd_single_view) / best_ucd_single_view,
                    dataset_folder
                ])    

    return csv_file_path

def overall_stats(csv_file_path):

    json_file_path = csv_file_path.replace(".csv", "_overall_stats.json")

    count = 0
    fuse_avg_bcd = []
    fuse_avg_ucd = []
    fuse_min_entropy_bcd = []
    fuse_min_entropy_ucd = []
    fuse_avg_bcd_change = []
    fuse_avg_ucd_change = []
    fuse_min_entropy_bcd_change = []
    fuse_min_entropy_ucd_change = []


    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            count += 1
            fuse_avg_bcd.append(float(row["fuse_avg_bcd"]))
            fuse_avg_ucd.append(float(row["fuse_avg_ucd"]))
            fuse_min_entropy_bcd.append(float(row["fuse_min_entropy_bcd"]))
            fuse_min_entropy_ucd.append(float(row["fuse_min_entropy_ucd"]))
            fuse_avg_bcd_change.append(float(row["fuse_avg_bcd_change"]))
            fuse_avg_ucd_change.append(float(row["fuse_avg_ucd_change"]))
            fuse_min_entropy_bcd_change.append(float(row["fuse_min_entropy_bcd_change"]))
            fuse_min_entropy_ucd_change.append(float(row["fuse_min_entropy_ucd_change"]))

    overall_stats = {
        "total_entries": count,
        "chamfer_distance":{
            "fuse_avg_bcd(mean)": np.mean(fuse_avg_bcd),
            "fuse_avg_bcd(std)": np.std(fuse_avg_bcd),
            "fuse_min_entropy_bcd(mean)": np.mean(fuse_min_entropy_bcd),
            "fuse_min_entropy_bcd(std)": np.std(fuse_min_entropy_bcd),
            "fuse_avg_ucd(mean)": np.mean(fuse_avg_ucd),
            "fuse_avg_ucd(std)": np.std(fuse_avg_ucd),
            "fuse_min_entropy_ucd(mean)": np.mean(fuse_min_entropy_ucd),
            "fuse_min_entropy_ucd(std)": np.std(fuse_min_entropy_ucd)
        },
        "comparison_to_single_views": {
            "average_fuse_bcd_change(mean)": np.mean(fuse_avg_bcd_change),
            "average_fuse_bcd_change(std)": np.std(fuse_avg_bcd_change),
            "average_fuse_min_entropy_bcd_change(mean)": np.mean(fuse_min_entropy_bcd_change),
            "average_fuse_min_entropy_bcd_change(std)": np.std(fuse_min_entropy_bcd_change),
            "average_fuse_ucd_change(mean)": np.mean(fuse_avg_ucd_change),
            "average_fuse_ucd_change(std)": np.std(fuse_avg_ucd_change),
            "average_fuse_min_entropy_ucd_change(mean)": np.mean(fuse_min_entropy_ucd_change),
            "average_fuse_min_entropy_ucd_change(std)": np.std(fuse_min_entropy_ucd_change)
        }
    }

    with open(json_file_path, "w") as f:
        json.dump(overall_stats, f, indent=4)
        
    return overall_stats




def main():
    argparser = argparse.ArgumentParser(description="Aggregate evaluation stats from individual folders to a single CSV file")
    argparser.add_argument("--dataset_folder_path", type=str, required=True, help="Path to dataset folder containing individual evaluation results")
    argparser.add_argument("--start_index", type=int, default=0, help="Start index for processing dataset folders")
    argparser.add_argument("--end_index", type=int, default=None, help="End index for processing dataset folders")
    argparser.add_argument("--output_dir", type=str, required=True, help="Directory where output csv or json files will be saved")
    args = argparser.parse_args()

    
    # Read the dataset folders
    csv_file_path = os.path.join(args.output_dir, "fusion_evaluation_stats.csv")
    csv_file_path = stats_in_csv(args.dataset_folder_path, args.start_index, args.end_index, csv_file_path)
    print(f"Saved aggregated evaluation stats to {csv_file_path}")

    # Calculate overall stats
    overall_stats_result = overall_stats(csv_file_path)
    print(f"Overall Stats: {overall_stats_result}")



if __name__ == "__main__":
    main()