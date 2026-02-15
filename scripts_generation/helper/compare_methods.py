import json
import csv
import os


def main():

    final_output_csv = os.path.join("/scratch/cl927/sam-3d-objects/results", "final_results.csv")
    generation_methods = ["sam3d_singleview_predictions", "trellis_singleview_outputs", "trellis_multiimage_outputs"]
    evaluation_types = ["chamfer", "reprojection"]

    with open(final_output_csv, 'w', newline='') as csvfile:
        fieldnames = ['generation_method', 'main_view_iou', 'other_views_average_iou', "chamfer_bcd", "chamfer_ucd"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for generation_method in generation_methods:
            entry = [generation_method]

            iou_json = os.path.join("/scratch/cl927/sam-3d-objects/results", f"{generation_method}_reprojection_evaluation_stats_overall_stats.json")
            chamfer_json = os.path.join("/scratch/cl927/sam-3d-objects/results", f"{generation_method}_chamfer_evaluation_stats_overall_stats.json")
            with open(iou_json, 'r') as f:
                iou_stats = json.load(f)
                entry.append(f"{iou_stats['overall_main_view_iou']:.3f} ± ({iou_stats['overall_main_view_iou_std']:.3f})")
                entry.append(f"{iou_stats['overall_other_views_average_iou']:.3f} ± ({iou_stats['overall_other_views_average_iou_std']:.3f})")

            with open(chamfer_json, 'r') as f:
                chamfer_stats = json.load(f)
                entry.append(f"{chamfer_stats['overall_average_bcd']:.5f} ± ({chamfer_stats['overall_std_bcd']:.5f})")
                entry.append(f"{chamfer_stats['overall_average_ucd']:.5f} ± ({chamfer_stats['overall_std_ucd']:.5f})")

            writer.writerow({
                'generation_method': entry[0],
                'main_view_iou': entry[1],
                'other_views_average_iou': entry[2],
                'chamfer_bcd': entry[3],
                'chamfer_ucd': entry[4]
            })

if __name__ == "__main__":
    main()