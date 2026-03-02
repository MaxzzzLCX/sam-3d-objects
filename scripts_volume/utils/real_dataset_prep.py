"""
Help to create correspondence for image names to real plate diameters.
"""

import json
import os
from datetime import datetime

def create_image_to_diameter_mapping(dataset_folder):
    mapping = {
        "potato_1": {
            "diameter": 20.5,
            "gt_volume_ml": 183
        },
        "potato_2": {
            "diameter": 20.5,
            "gt_volume_ml": 183
        },
        "potato_3": {
            "diameter": 15.3,
            "gt_volume_ml": 183
        },
        "potato_4": {
            "diameter": 17.5,
            "gt_volume_ml": 183
        },
        "egg_1": {
            "diameter": 20.5,
            "gt_volume_ml": 76.2
        },
        "egg_2": {
            "diameter": 15.3,
            "gt_volume_ml": 76.2
        },
        "egg_3": {
            "diameter": 17.5,
            "gt_volume_ml": 76.2
        },
        "cucumber_1": {
            "diameter": 20.5,
            "gt_volume_ml": 52.8
        },
        "cucumber_2": {
            "diameter": 15.3,
            "gt_volume_ml": 52.8
        },
        "cucumber_3": {
            "diameter": 17.5,
            "gt_volume_ml": 52.8
        },
    }

    # Current time
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(dataset_folder, f"generations/plate_diameters_{timestamp}.json")

    with open(json_path, "w") as f:
        json.dump(mapping, f, indent=4)

def main():
    dataset_folder = "/scratch/cl927/sam-3d-objects/scripts_volume/real_dataset"
    create_image_to_diameter_mapping(dataset_folder)

if __name__ == "__main__":
    main()