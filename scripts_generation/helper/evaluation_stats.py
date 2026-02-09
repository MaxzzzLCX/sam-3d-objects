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
        view_key = f"00{i}_voxels" if i<10 else f"0{i}_voxels"
        
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


def stats_in_csv(dataset_folder_path, start_index, end_index, num_views, csv_file_path):
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

        evaluation_results_path = os.path.join(dataset_folder, "sam3d_singleview_predictions/chamfer_distance_evaluation.json")
        
        with open(evaluation_results_path, "r") as f:
            evaluation_results = json.load(f)

            stats = evaluate_std(evaluation_results, num_views=num_views)
    
        # Save stats to CSV
        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([dataset_folder, num_views, stats["average_bcd"], stats["std_bcd"], stats["average_ucd"], stats["std_ucd"]])
    

    return csv_file_path

def overall_stats(csv_file_path):
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
        


def main():
    argparser = argparse.ArgumentParser(description="Calculate overall evaluation stats for a batch of dataset")
    argparser.add_argument("--dataset_folder", type=str, required=True, help="Path to dataset folder containing individual evaluation results")
    argparser.add_argument("--output_dir", type=str, required=True, help="Directory where output csv or json files will be saved")
    argparser.add_argument("--start_index", type=int, default=0, help="Start index for processing dataset folders")
    argparser.add_argument("--end_index", type=int, default=None, help="End index for processing dataset folders")
    argparser.add_argument("--num_views", type=int, required=True, help="Number of views for evaluation")
    args = argparser.parse_args()

    csv_file_path = stats_in_csv(args.dataset_folder, args.start_index, args.end_index, args.num_views, os.path.join(args.output_dir, "evaluation_stats.csv"))
    overall_stats_result = overall_stats(csv_file_path)
    print(f"Overall Stats: {overall_stats_result}")
    
if __name__ == "__main__":
    main()