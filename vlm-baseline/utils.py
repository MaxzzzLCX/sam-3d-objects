"""
Utility functions for Gemini Minimal Multimodal experiment.
"""
from pathlib import Path
import json
import numpy as np


def overall_results(results_dir, model, num_values=[1, 6]):

    overall_results = {}
    results_dir = Path(results_dir)
    
    for num in num_values:
        target_files = list(results_dir.glob(f"*-num{num}-results.json"))
        if not target_files:
            print(f"Warning: No result files found for num={num} in {results_dir}")

        all_results = []
        # Collect overall average and std in volume estimation error
        for result_file in target_files:
            try:
                raw_result_data = json.loads(result_file.read_text(encoding="utf-8"))
                all_results.append(raw_result_data["results"])
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON from {result_file}: {e}")
        
        error_values = [
            r["volume_error_mean"]
            for r in all_results
            if "volume_error_mean" in r and r["volume_error_mean"] is not None
        ]
        sample_counts = [
            len(r["volume_estimations"])
            for r in all_results
            if "volume_estimations" in r and r["volume_estimations"] is not None
        ]

        average_error = float(np.mean(error_values)) if error_values else None
        std_error = float(np.std(error_values)) if error_values else None
        total_samples = int(np.sum(sample_counts)) if sample_counts else 0
        overall_results[num] = {
            "average_volume_error": average_error,
            "std_volume_error": std_error,
            "total_samples": total_samples
        }

    print("Overall Results:")
    for num, metrics in overall_results.items():
        print(f"Num: {num}")
        print(f"  Average Volume Error: {metrics['average_volume_error']:.6f}")
        print(f"  Std Volume Error: {metrics['std_volume_error']:.6f}")
        print(f"  Total Samples: {metrics['total_samples']}")
    
    # Save overall results to a JSON file
    json_output_path = f"/scratch/cl927/sam-3d-objects/vlm-baseline/results/{model}_overall_results.json"
    with open(json_output_path, "w") as f:
        json.dump(overall_results, f, indent=2)
    print(f"Saved overall results to {json_output_path}")
    
    

def main():
    model = "gemini-2.5-flash" # "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-3-flash-preview"
    results_dir = f"/scratch/cl927/sam-3d-objects/vlm-baseline/outputs/{model}/"
    overall_results(results_dir, model)

if __name__ == "__main__":
    main()