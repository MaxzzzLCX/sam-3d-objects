"""
Calculates the overall stats of evaluation over batch of dataset.
"""
import json
import argparse
import os
import csv
import numpy as np

def evaluate_std(chamfer_distance_evaluation_results, num_views):
    """
    Evaluates the standard deviation of Chamfer distance across multiple views.
    (If std was not calculated when data was collected)
    """
    bcds = np.zeros(num_views)
    ucds = np.zeros(num_views)
    for i in range(num_views):
        view_key = f"00{i}_mesh" if i<10 else f"0{i}_mesh"
        
        if view_key not in chamfer_distance_evaluation_results:
            raise KeyError(f"Missing evaluation results for view {i}: expected key '{view_key}' not found in results.")
        
        bcds[i] = chamfer_distance_evaluation_results[view_key]['bidirectional_chamfer_distance']
        ucds[i] = chamfer_distance_evaluation_results[view_key]['unidirectional_chamfer_distance']

    average_bcd = np.mean(bcds)
    average_ucd = np.mean(ucds)
    std_bcd = np.std(bcds)
    std_ucd = np.std(ucds)
    stats = {
        "average_bcd": average_bcd,
        "average_ucd": average_ucd,
        "std_bcd": std_bcd,
        "std_ucd": std_ucd
    }
    return stats


def chamfer_stats_in_csv(dataset_folder_path, start_index, end_index, num_views, generation_method, csv_file_path):
    """
    Aggregates evaluation stats from inidividual folders to a single CSV file.
    """

    # CSV header
    # csv_file_path = os.path.join(dataset_folder_path, "evaluation_stats.csv")
    csv_header = ["dataset_folder", "num_views", "average_bcd", "std_bcd", "average_ucd", "std_ucd"]
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)

    dataset_folders = sorted([os.path.join(dataset_folder_path, f) for f in os.listdir(dataset_folder_path) if os.path.isdir(os.path.join(dataset_folder_path, f))])
    if start_index is not None and end_index is not None:
        dataset_folders = dataset_folders[start_index:end_index]
    
    print(f"dataset_folders (length {len(dataset_folders)}): {dataset_folders}")

    for dataset_folder in dataset_folders:

        evaluation_results_path = os.path.join(dataset_folder, f"{generation_method}/chamfer_distance_evaluation.json")
        
        if generation_method == "trellis_multiimage_outputs":
            with open(evaluation_results_path, "r") as f:
                evaluation_results = json.load(f)

                stats = evaluation_results["multi_0_1_mesh"]

            # Save stats to CSV
            with open(csv_file_path, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([dataset_folder, num_views, stats["bidirectional_chamfer_distance"], 0, stats["unidirectional_chamfer_distance"], 0])

        else:
            with open(evaluation_results_path, "r") as f:
                evaluation_results = json.load(f)

                stats = evaluate_std(evaluation_results, num_views=num_views)
        
            # Save stats to CSV
            with open(csv_file_path, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([dataset_folder, num_views, stats["average_bcd"], stats["std_bcd"], stats["average_ucd"], stats["std_ucd"]])

    return csv_file_path

def reprojection_stats_in_csv(dataset_folder_path, start_index, end_index, num_views, generation_method, csv_file_path):
    """
    Aggregates evaluation stats from inidividual folders to a single CSV file.
    """

    # CSV header
    # csv_file_path = os.path.join(dataset_folder_path, "evaluation_stats.csv")
    csv_header = ["dataset_folder", "num_views", "main_view_iou", "other_views_average_iou"]
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)

    dataset_folders = sorted([os.path.join(dataset_folder_path, f) for f in os.listdir(dataset_folder_path) if os.path.isdir(os.path.join(dataset_folder_path, f))])
    if start_index is not None and end_index is not None:
        dataset_folders = dataset_folders[start_index:end_index]
    
    print(f"dataset_folders (length {len(dataset_folders)}): {dataset_folders}")

    for dataset_folder in dataset_folders:

        reprojection_folder = os.path.join(dataset_folder, f"{generation_method}/reprojection")
        reprojected_views = sorted([f for f in os.listdir(reprojection_folder) if os.path.isdir(os.path.join(reprojection_folder, f))]) # ["reprojection/view00_mesh/", "reprojection/view01_mesh", ...]
        for reprojection_view_folder in reprojected_views:
            view_folder_path = os.path.join(reprojection_folder, reprojection_view_folder)
            iou_evaluation_path = os.path.join(view_folder_path, "iou/iou_results.json")
            if not os.path.exists(iou_evaluation_path):
                print(f"Skipping {iou_evaluation_path} (does not exist)")
                continue
            
            print(f"Processing IoU evaluation results from {iou_evaluation_path}...")
            with open(iou_evaluation_path, "r") as f:
                iou_evaluation_results = json.load(f)
                main_view_iou = iou_evaluation_results["main_view_iou"]
                other_views_average_iou = iou_evaluation_results["other_views_average_iou"]

            # Save stats to CSV
            with open(csv_file_path, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([dataset_folder, num_views, main_view_iou, other_views_average_iou])
    
    return csv_file_path

def chamfer_overall_stats(csv_file_path):
    """
    Reads the CSV file and calculates overall stats across all datasets.
    """
    output_json_file = csv_file_path.replace(".csv", "_overall_stats.json")
    bcds = []
    ucds = []
    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            bcds.append(float(row["average_bcd"]))
            ucds.append(float(row["average_ucd"]))

    overall_stats = {
        "overall_average_bcd": np.mean(bcds),
        "overall_std_bcd": np.std(bcds),
        "overall_average_ucd": np.mean(ucds),
        "overall_std_ucd": np.std(ucds)
    }
    with open(output_json_file, "w") as f:
        json.dump(overall_stats, f, indent=4)
    return overall_stats

def iou_overall_stats(csv_file_path):
    """
    Reads the CSV file and calculates overall stats across all datasets.
    """
    output_json_file = csv_file_path.replace(".csv", "_overall_stats.json")
    main_view_ious = []
    other_views_average_ious = []
    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            main_view_ious.append(float(row["main_view_iou"]))
            other_views_average_ious.append(float(row["other_views_average_iou"]))

    overall_stats = {
        "overall_main_view_iou": np.mean(main_view_ious),
        "overall_main_view_iou_std": np.std(main_view_ious),
        "overall_other_views_average_iou": np.mean(other_views_average_ious),
        "overall_other_views_average_iou_std": np.std(other_views_average_ious)
    }
    with open(output_json_file, "w") as f:
        json.dump(overall_stats, f, indent=4)
    return overall_stats  


def main():
    argparser = argparse.ArgumentParser(description="Calculate overall evaluation stats for a batch of dataset")
    argparser.add_argument("--dataset_folder", type=str, required=True, help="Path to dataset folder containing individual evaluation results")
    argparser.add_argument("--generation_method", type=str, required=True, 
                           choices=["sam3d_singleview_predictions", "trellis_singleview_outputs", "trellis_multiimage_outputs"], help="Generation method (e.g. sam3d_singleview_predictions)")
    argparser.add_argument("--output_dir", type=str, required=True, help="Directory where output csv or json files will be saved")
    argparser.add_argument("--start_index", type=int, default=0, help="Start index for processing dataset folders")
    argparser.add_argument("--end_index", type=int, default=None, help="End index for processing dataset folders")
    argparser.add_argument("--num_views", type=int, required=True, help="Number of views for evaluation")
    args = argparser.parse_args()

    csv_file_path = chamfer_stats_in_csv(
        args.dataset_folder, 
        args.start_index, 
        args.end_index, 
        args.num_views,
        args.generation_method, 
        os.path.join(args.output_dir, f"{args.generation_method}_chamfer_evaluation_stats.csv")
    )
    iou_file_path = reprojection_stats_in_csv(
        args.dataset_folder, 
        args.start_index, 
        args.end_index, 
        args.num_views, 
        args.generation_method, 
        os.path.join(args.output_dir, f"{args.generation_method}_reprojection_evaluation_stats.csv")
    )
    chamfer_overall_stats_result = chamfer_overall_stats(csv_file_path)
    iou_overall_stats_result = iou_overall_stats(iou_file_path)
    print(f"Chamfer Overall Stats: {chamfer_overall_stats_result}")
    print(f"IOU Overall Stats: {iou_overall_stats_result}")
    
if __name__ == "__main__":
    main()