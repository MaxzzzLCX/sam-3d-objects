"""
This script is responsible for collecting volume evaluation results across
multiple objects and methods of generations. 
"""

import os
import json
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

class Toys4kVolumeEvaluationCollector:
    def __init__(self, dataset_folder_path, gen_method):
        self.dataset_folder_path = dataset_folder_path
        self.gen_method = gen_method
    
    def collect_volume_evaluation_results(self):

        object_folder_paths = sorted([os.path.join(self.dataset_folder_path, f) for f in os.listdir(self.dataset_folder_path) if os.path.isdir(os.path.join(self.dataset_folder_path, f))])

        mean_error_percentages = []
        std_error_percentages = []
        num_views_overall = []

        for object_folder_path in object_folder_paths:

            gen_folder = os.path.join(object_folder_path, self.gen_method, "volume_evaluation_results.json")

            with open(gen_folder, "r") as f:
                gen_results = json.load(f)
                num_views = len(gen_results["predicted_volumes"])
            
            mean_error_percentages.append(gen_results["mean percent error"])
            std_error_percentages.append(gen_results["std percent error"])
            num_views_overall.append(num_views)
        
        # Create a DataFrame to store the results
        df = pd.DataFrame({
            "Object": [os.path.basename(path) for path in object_folder_paths],
            "Num Views": num_views_overall,
            "Mean Percent Error": mean_error_percentages,
            "Std Percent Error": std_error_percentages
        })

        # Save the DataFrame to a CSV file
        output_csv_path = os.path.join("/scratch/cl927/sam-3d-objects/results/20250225_volumes", f"foodlike_{self.gen_method}_volume_evaluation_summary.csv")
        df.to_csv(output_csv_path, index=False)
        print(f"Volume evaluation summary saved to {output_csv_path}")

        # Show stats
        overall_mean_error = np.mean(mean_error_percentages)
        overall_std_error = np.std(mean_error_percentages)
        print(f"Overall Mean Percent Error for {self.gen_method}: {overall_mean_error:.2f}%")
        print(f"Overall Std Percent Error for {self.gen_method}: {overall_std_error:.2f}%")
    
    def dataset_volume_eval_stats(self):
        object_folder_paths = sorted([os.path.join(self.dataset_folder_path, f) for f in os.listdir(self.dataset_folder_path) if os.path.isdir(os.path.join(self.dataset_folder_path, f))])

        instance_error_percentage = []

        for object_folder_path in object_folder_paths:

            gen_folder = os.path.join(object_folder_path, self.gen_method, "volume_evaluation_results.json")

            with open(gen_folder, "r") as f:
                gen_results = json.load(f)
                error_percentages = gen_results["volume_error_percentages"]
            
            instance_error_percentage.extend(error_percentages)
        
        overall_mean_error = np.mean(instance_error_percentage)
        overall_std_error = np.std(instance_error_percentage)
        print(f"Overall Mean Percent Error for {self.gen_method}: {overall_mean_error:.2f}%")
        print(f"Overall Std Percent Error for {self.gen_method}: {overall_std_error:.2f}%")
        
        # Plot a histogram of error percentages
        # Bin every 10% error increments up to the max error percentage
        bins = np.arange(0, max(instance_error_percentage) + 10, 10)
        plt.figure(figsize=(10, 6))
        weights = np.ones_like(instance_error_percentage) / len(instance_error_percentage) * 100
        plt.hist(instance_error_percentage, bins=bins, weights=weights, edgecolor='black')
        plt.title(f"Distribution of Volume Error Percentages for {self.gen_method}")
        plt.xlabel("Volume Error Percentage (%)")
        plt.ylabel("Percentage of Samples (%)")
        output_plot_path = os.path.join("/scratch/cl927/sam-3d-objects/results/20250225_volumes", f"foodlike_{self.gen_method}_volume_error_distribution.png")
        plt.savefig(output_plot_path)
        print(f"Volume error distribution plot saved to {output_plot_path}")

class RealDataVolumeEvaluationCollector:
    def __init__(self, dataset_folder_path, gen_method, white_list_food, black_list_views, with_pointmaps):
        self.dataset_folder_path = dataset_folder_path
        self.gen_method = gen_method
        self.white_list_food = white_list_food
        self.black_list_views = black_list_views
        self.with_pointmaps = with_pointmaps
    
    def collect_volume_evaluation_results(self, json_file_name, save_result_folder, with_blaclist_filtering=False):
        object_folder_paths = sorted(
            [os.path.join(self.dataset_folder_path, f) 
                for f in os.listdir(self.dataset_folder_path) 
                if os.path.isdir(os.path.join(self.dataset_folder_path, f)) and f.split("_")[0] in self.white_list_food]
            )

        
        all_errors = []
        all_errors_with_gt_scale = []
        all_scale_errors = []
        mean_error_percentages = []
        std_error_percentages = []
        mean_error_percentages_with_gt_scale = []
        std_error_percentages_with_gt_scale = []
        mean_scale_error_percentages = []
        std_scale_error_percentages = []
        num_views_overall = []

        for object_folder_path in object_folder_paths:                

            gen_folder = os.path.join(object_folder_path, json_file_name)

            with open(gen_folder, "r") as f:
                gen_results = json.load(f)
                num_views = len(gen_results["errors"])
            
            errors = gen_results["errors"]
            errors_with_gt_scale = gen_results["errors_with_gt_relative_scale"]
            scale_errors = gen_results["relative_scale_errors"]

            if with_blaclist_filtering:

                # Filter out blacklisted views
                object_name = os.path.basename(object_folder_path)
                black_listed_views_for_object = self.black_list_views.get(object_name, [])
                print(f"Object {object_name} has blacklisted views: {black_listed_views_for_object}")

                filtered_errors = []
                filtered_errors_with_gt_scale = []
                filtered_scale_errors = []
                for i, error in enumerate(errors):
                    if i not in black_listed_views_for_object:
                        filtered_errors.append(error)
                        filtered_errors_with_gt_scale.append(errors_with_gt_scale[i])
                        filtered_scale_errors.append(scale_errors[i])

                errors = filtered_errors
                errors_with_gt_scale = filtered_errors_with_gt_scale
                scale_errors = filtered_scale_errors

            all_errors.extend(errors)
            all_errors_with_gt_scale.extend(errors_with_gt_scale)
            all_scale_errors.extend(scale_errors)

            mean_error = np.mean(errors)
            std_error = np.std(errors)

            mean_error_percentages.append(mean_error)
            std_error_percentages.append(std_error)
            mean_error_percentages_with_gt_scale.append(np.mean(errors_with_gt_scale))
            std_error_percentages_with_gt_scale.append(np.std(errors_with_gt_scale))
            mean_scale_error_percentages.append(np.mean(scale_errors))
            std_scale_error_percentages.append(np.std(scale_errors))
            num_views_overall.append(num_views)
        
        # Create a DataFrame to store the results
        df = pd.DataFrame({
            "Object": [os.path.basename(path) for path in object_folder_paths],
            "Num Views": num_views_overall,
            "Mean Percent Error": mean_error_percentages,
            "Std Percent Error": std_error_percentages,
            "Mean Percent Error With GT Scale": mean_error_percentages_with_gt_scale,
            "Std Percent Error With GT Scale": std_error_percentages_with_gt_scale,
            "Mean Scale Error": mean_scale_error_percentages,
            "Std Scale Error": std_scale_error_percentages
        })

        # Save the DataFrame to a CSV file
        if self.with_pointmaps:
            if with_blaclist_filtering:
                output_csv_path = os.path.join("/scratch/cl927/sam-3d-objects/results", save_result_folder, f"foodlike_voxelized_filtered_{self.gen_method}_volume_evaluation_summary_with_pointmaps.csv")
            else:
                output_csv_path = os.path.join("/scratch/cl927/sam-3d-objects/results", save_result_folder, f"foodlike_voxelized_{self.gen_method}_volume_evaluation_summary_with_pointmaps.csv")
        else:
            if with_blaclist_filtering:
                output_csv_path = os.path.join("/scratch/cl927/sam-3d-objects/results", save_result_folder, f"foodlike_voxelized_filtered_{self.gen_method}_volume_evaluation_summary.csv")
            else:
                output_csv_path = os.path.join("/scratch/cl927/sam-3d-objects/results/", save_result_folder, f"foodlike_voxelized_{self.gen_method}_volume_evaluation_summary.csv")
        df.to_csv(output_csv_path, index=False)
        print(f"Volume evaluation summary saved to {output_csv_path}")

        # Show stats
        overall_mean_error = np.mean(mean_error_percentages)
        overall_std_error = np.std(mean_error_percentages)
        print(f"Overall Mean Percent Error for {self.gen_method}: {overall_mean_error*100:.4f}%")
        print(f"Overall Std Percent Error (between objects) for {self.gen_method}: {overall_std_error*100:.4f}%")
        print(f"Overall Std Percent Error (between samples) for {self.gen_method}: {np.std(all_errors)*100:.4f}%")
        print(f"Volume estimation error with GT scale: {np.mean(mean_error_percentages_with_gt_scale)*100:.4f}% ± {np.std(mean_error_percentages_with_gt_scale)*100:.4f}%")
        print(f"Relative scale error: {np.mean(mean_scale_error_percentages)*100:.4f}% ± {np.std(mean_scale_error_percentages)*100:.4f}%")

        # Save the overall stats to a text file
        overall_stats_path = output_csv_path.replace(".csv", ".txt")
        with open(overall_stats_path, "w") as f:
            f.write(f"Overall Mean Percent Error for {self.gen_method}: {overall_mean_error*100:.4f}%\n")
            f.write(f"Overall Std Percent Error (between objects) for {self.gen_method}: {overall_std_error*100:.4f}%\n")
            f.write(f"Overall Std Percent Error (between samples) for {self.gen_method}: {np.std(all_errors)*100:.4f}%\n")
            f.write(f"Volume estimation error with GT scale: {np.mean(mean_error_percentages_with_gt_scale)*100:.4f}% ± {np.std(mean_error_percentages_with_gt_scale)*100:.4f}%\n")
            f.write(f"Relative scale error: {np.mean(mean_scale_error_percentages)*100:.4f}% ± {np.std(mean_scale_error_percentages)*100:.4f}%\n")


    def filter_and_collect_volume_evaluation_results(self, json_file_name):
        object_folder_paths = sorted(
            [os.path.join(self.dataset_folder_path, f) 
                for f in os.listdir(self.dataset_folder_path) 
                if os.path.isdir(os.path.join(self.dataset_folder_path, f)) and f.split("_")[0] in self.white_list_food]
            )
        
        discarded_bad_instances = []

        all_errors = []
        all_errors_with_gt_scale = []
        all_scale_errors = []
        mean_error_percentages = []
        std_error_percentages = []
        mean_error_percentages_with_gt_scale = []
        std_error_percentages_with_gt_scale = []
        mean_scale_error_percentages = []
        std_scale_error_percentages = []
        num_views_overall = []

        for object_folder_path in object_folder_paths:

            gen_folder = os.path.join(object_folder_path, json_file_name)

            with open(gen_folder, "r") as f:
                gen_results = json.load(f)
                num_views = len(gen_results["errors"])
            
            errors = gen_results["errors"]
            errors_with_gt_scale = gen_results["errors_with_gt_relative_scale"]
            scale_errors = gen_results["relative_scale_errors"]

            # Filter out all errors that are above the mean error percentage for this object
            valid_error_indices = np.where(np.array(errors) <= np.mean(errors))[0]
            filtered_errors = np.array(errors)[valid_error_indices]
            filtered_errors_with_gt_scale = np.array(errors_with_gt_scale)[valid_error_indices]
            filtered_scale_errors = np.array(scale_errors)[valid_error_indices]

            # Save the discarded bad instances for inspection
            discarded_indices = np.where(np.array(errors) > np.mean(errors))[0]
            discarded_bad_instances.append({
                "object": os.path.basename(object_folder_path),
                "discarded_indices": discarded_indices.tolist(),
            })

            all_errors.extend(filtered_errors)
            all_errors_with_gt_scale.extend(filtered_errors_with_gt_scale)
            all_scale_errors.extend(filtered_scale_errors)

            mean_error = np.mean(filtered_errors)
            std_error = np.std(filtered_errors)

            mean_error_percentages.append(mean_error)
            std_error_percentages.append(std_error)
            mean_error_percentages_with_gt_scale.append(np.mean(filtered_errors_with_gt_scale))
            std_error_percentages_with_gt_scale.append(np.std(filtered_errors_with_gt_scale))
            mean_scale_error_percentages.append(np.mean(filtered_scale_errors))
            std_scale_error_percentages.append(np.std(filtered_scale_errors))
            num_views_overall.append(num_views)
        
        # Create a DataFrame to store the results
        df = pd.DataFrame({
            "Object": [os.path.basename(path) for path in object_folder_paths],
            "Num Views": num_views_overall,
            "Mean Percent Error": mean_error_percentages,
            "Std Percent Error": std_error_percentages,
            "Mean Percent Error With GT Scale": mean_error_percentages_with_gt_scale,
            "Std Percent Error With GT Scale": std_error_percentages_with_gt_scale,
            "Mean Scale Error": mean_scale_error_percentages,
            "Std Scale Error": std_scale_error_percentages
        })

        # Save the DataFrame to a CSV file
        if self.with_pointmaps:
            output_csv_path = os.path.join("/scratch/cl927/sam-3d-objects/results/realdata", f"foodlike_voxelized_filtered_{self.gen_method}_volume_evaluation_summary_with_pointmaps.csv")
        else:
            output_csv_path = os.path.join("/scratch/cl927/sam-3d-objects/results/realdata", f"foodlike_voxelized_filtered_{self.gen_method}_volume_evaluation_summary.csv")
        df.to_csv(output_csv_path, index=False)
        print(f"Volume evaluation summary saved to {output_csv_path}")

        # Show stats
        overall_mean_error = np.mean(mean_error_percentages)
        overall_std_error = np.std(mean_error_percentages)
        print(f"Overall Mean Percent Error for {self.gen_method}: {overall_mean_error*100:.4f}%")
        print(f"Overall Std Percent Error (between objects) for {self.gen_method}: {overall_std_error*100:.4f}%")
        print(f"Overall Std Percent Error (between samples) for {self.gen_method}: {np.std(all_errors)*100:.4f}%")
        print(f"Volume estimation error with GT scale: {np.mean(mean_error_percentages_with_gt_scale)*100:.4f}% ± {np.std(mean_error_percentages_with_gt_scale)*100:.4f}%")
        print(f"Relative scale error: {np.mean(mean_scale_error_percentages)*100:.4f}% ± {np.std(mean_scale_error_percentages)*100:.4f}%")

        print(f"Discarded bad instances: {json.dumps(discarded_bad_instances, indent=2)}")

if __name__ == "__main__":
    # dataset_folder_path = "/scratch/cl927/datasets/Toys4k/subset_foodlike"
    # gen_method = "sam3d_singleview_predictions"
    # toys4k_volume_collector = Toys4kVolumeEvaluationCollector(dataset_folder_path, gen_method)
    # toys4k_volume_collector.collect_volume_evaluation_results()
    # toys4k_volume_collector.dataset_volume_eval_stats()

    real_data_folder_path = "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt"
    real_data_gen_method = "sam3d_singleview_predictions"
    with_pointmaps = False
    if with_pointmaps:
        json_file = "voxelize_volume_estimation_results_with_pointmaps.json"
    else:        
        json_file = "voxelize_volume_estimation_results.json"

    # Manual Filtering of bad views
    black_list_views = {
        "egg_bowl": [5],
        "egg_plate": [],
        "orange_bowl": [],
        "orange_plate": [5],
        # "pepper_bowl": [],
        # "pepper_plate": [2, 4, 5],
        "potato_bowl": [5],
        "potato_plate": [5],
        "strawberry_bowl": [],
        "strawberry_plate": [],
        "avocado_plate": [],
    }

    real_data_volume_collector = RealDataVolumeEvaluationCollector(
        real_data_folder_path, 
        real_data_gen_method,
        white_list_food=["egg", "orange", "potato", "strawberry", "avocado"],
        black_list_views=black_list_views,
        with_pointmaps=with_pointmaps
        # white_list_food=["box", "egg", "orange", "pepper", "potato"]
    )

    real_data_volume_collector.collect_volume_evaluation_results(
        json_file_name=json_file,
        save_result_folder="20250305",
        with_blaclist_filtering=True
    )


    # real_data_volume_collector.filter_and_collect_volume_evaluation_results(
    #     json_file_name=json_file,
    # )
