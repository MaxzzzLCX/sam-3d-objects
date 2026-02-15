"""
Evaluates the rerendered image against the original image
"""
import argparse
import os
from PIL import Image
import numpy as np
import json


def visualize_iou(image1_array, image2_array, intersection, union):
    """
    On top of the original (image1_array)
    Shade the intersection area in green and the union area in red
    """
    vis_image = image1_array.copy().astype(np.float32)
    
    # Overlay green tint on intersection areas (blend with original)
    green_overlay = np.array([0, 255, 0, 255], dtype=np.float32)
    vis_image[intersection] = vis_image[intersection] * 0.7 + green_overlay * 0.3
    
    # Overlay red tint on union-only areas (blend with original)
    red_overlay = np.array([255, 0, 0, 255], dtype=np.float32)
    vis_image[union & ~intersection] = vis_image[union & ~intersection] * 0.7 + red_overlay * 0.3
    
    return vis_image.astype(np.uint8)


    

def calculate_iou(image1_path, image2_path, output_dir, visualize=False):
    """Calculate IoU between two images based on non-transparent pixels"""
    # Load images with alpha channel
    image1 = Image.open(image1_path).convert('RGBA')
    image2 = Image.open(image2_path).convert('RGBA')
    
    # Convert to numpy arrays
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    
    # Create binary masks from alpha channel (non-transparent = object)
    mask1 = arr1[:, :, 3] > 128  # Alpha > 128 means non-transparent
    mask2 = arr2[:, :, 3] > 128
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    
    iou = intersection_area / union_area if union_area > 0 else 0

    if visualize:
        # Save visualization
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        vis_image = visualize_iou(arr1, arr2, intersection, union)
        Image.fromarray(vis_image).save(output_dir)
        print("Saved visualized IoU image.")


        # vis_intersection = np.zeros_like(arr1)
        # vis_intersection[intersection] = [255, 255, 255, 255]
        # vis_union = np.zeros_like(arr1)
        # vis_union[union] = [255, 255, 255, 255]
        
        # Image.fromarray(vis_intersection).save('scripts_volume/intersection.png')
        # Image.fromarray(vis_union).save('scripts_volume/union.png')
        # print("Saved intersection and union images.")

    return iou

def diagnose_images(image1_path, image2_path):
    """
    Check the sizes and pixel values of image pixels
    """
    image1 = Image.open(image1_path).convert('RGBA')
    image2 = Image.open(image2_path).convert('RGBA')

    print(f"Image 1 size: {image1.size}, mode: {image1.mode}")
    print(f"Image 2 size: {image2.size}, mode: {image2.mode}")

    # Convert to numpy arrays
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    print(f"Image 1 pixel value range: {arr1.min()} to {arr1.max()}")
    print(f"Image 2 pixel value range: {arr2.min()} to {arr2.max()}")

def evaluate_folder_iou(original_folder, rendered_folder, main_view_index=0):
    """
    Evaluate IoU for all image pairs in the given folders
    Assumes images have the same names in both folders
    """
    original_images = sorted([f for f in os.listdir(original_folder) if f.endswith('.png')])
    rendered_images = sorted([f for f in os.listdir(rendered_folder) if f.endswith('.png')])
    
    print(f"Original folder: {original_folder}, Rendered folder: {rendered_folder}")
    print(f"Original images length: {len(original_images)}")
    print(f"Rendered images length: {len(rendered_images)}")

    ious = []
    for i, (orig_img, rend_img) in enumerate(zip(original_images, rendered_images)):
        orig_path = os.path.join(original_folder, orig_img)
        rend_path = os.path.join(rendered_folder, rend_img)
        iou = calculate_iou(orig_path, rend_path, f"{rendered_folder}/iou/visualize_iou_{i}.png", visualize=True)
        ious.append(iou)
        print(f"IoU for {orig_img}: {iou:.4f}")
    
    other_views_mean_iou = np.mean([iou for j, iou in enumerate(ious) if j != main_view_index])
    avg_iou = np.mean(ious) if ious else 0

    print(f"IoU {ious}")
    output = {
        "main_view_iou": ious[main_view_index],
        "other_views_average_iou": other_views_mean_iou,
        "ious": {
            idx: iou for idx, iou in enumerate(ious)
        }
    }

    output_json_path = os.path.join(rendered_folder, "iou", "iou_results.json")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"Average IoU across all images: {avg_iou:.4f}")

def evaluate_folder_iou_multiview(original_folder, rendered_folder, main_view_indices=[0,1]):
    """
    Evaluate IoU for all image pairs in the given folders
    Assumes images have the same names in both folders
    """
    original_images = sorted([f for f in os.listdir(original_folder) if f.endswith('.png')])
    rendered_images = sorted([f for f in os.listdir(rendered_folder) if f.endswith('.png')])
    
    print(f"Original folder: {original_folder}, Rendered folder: {rendered_folder}")
    print(f"Original images length: {len(original_images)}")
    print(f"Rendered images length: {len(rendered_images)}")

    ious = []
    for i, (orig_img, rend_img) in enumerate(zip(original_images, rendered_images)):
        orig_path = os.path.join(original_folder, orig_img)
        rend_path = os.path.join(rendered_folder, rend_img)
        iou = calculate_iou(orig_path, rend_path, f"{rendered_folder}/iou/visualize_iou_{i}.png", visualize=True)
        ious.append(iou)
        print(f"IoU for {orig_img}: {iou:.4f}")
    
    main_view_mean_iou = np.mean([iou for j, iou in enumerate(ious) if j in main_view_indices])
    other_views_mean_iou = np.mean([iou for j, iou in enumerate(ious) if j not in main_view_indices])
    avg_iou = np.mean(ious) if ious else 0

    print(f"IoU {ious}")
    output = {
        "used_views": main_view_indices,
        "main_view_iou": main_view_mean_iou,
        "other_views_average_iou": other_views_mean_iou,
        "ious": {
            idx: iou for idx, iou in enumerate(ious)
        }
    }

    output_json_path = os.path.join(rendered_folder, "iou", "iou_results.json")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"Average IoU across all images: {avg_iou:.4f}")

def main():
    argparser = argparse.ArgumentParser(description="Evaluate rendered images against original images")
    argparser.add_argument("--original_image", type=str, required=True)
    argparser.add_argument("--rendered_image", type=str, required=True)
    args = argparser.parse_args()
    
    # print("Diagnosing images...")
    # diagnose_images(args.original_image, args.rendered_image)

    # iou = calculate_iou(args.original_image, args.rendered_image, os.path.join(args.rendered_image, "visualize_iou"), visualize=True)
    # print(f"IoU between original and rendered image: {iou:.4f}")

    evaluate_folder_iou(args.original_image, args.rendered_image)



if __name__ == '__main__':
    main()