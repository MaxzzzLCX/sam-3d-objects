import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def project_voxels_to_image(voxels, scale, shift, intrinsic, image_path, mask_path, output_path):
    """
    Project SAM3D voxels back to image coordinates to check layout accuracy
    
    Args:
        voxels: (N, 3) voxel coordinates [0, 63] 
        scale: (3,) SAM3D predicted scale
        shift: (3,) SAM3D predicted translation
        intrinsic: (3, 3) camera intrinsic matrix
        image_path: path to original image
        mask_path: path to mask
        output_path: where to save visualization
    """
    
    # Step 1: Convert voxels to camera coordinates (same as alignment script)
    voxels_normalized = (voxels / 63.0) - 0.5  # [0,63] -> [-0.5, 0.5]
    voxels_camera = voxels_normalized * scale + shift  # Apply SAM3D layout
    
    # Step 2: Project 3D points to 2D image coordinates
    # Camera coordinates: X right, Y down, Z forward (OpenCV convention)
    points_3d = voxels_camera  # (N, 3)
    
    # Project using intrinsic matrix: [u, v, 1] = K * [X, Y, Z]
    points_2d_homo = (intrinsic @ points_3d.T).T  # (N, 3)
    
    # Convert from homogeneous to pixel coordinates
    valid_depth = points_3d[:, 2] > 0  # Only points in front of camera
    points_2d = points_2d_homo[valid_depth]
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]  # [u/z, v/z]
    
    # Step 3: Load image and mask for visualization
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path).convert('L')) > 128
    
    # Step 4: Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image with mask overlay
    axes[0].imshow(image)
    axes[0].imshow(mask, alpha=0.3, cmap='Reds')
    axes[0].set_title('Original Image + Mask')
    axes[0].axis('off')
    
    # Projected voxels on image
    axes[1].imshow(image)
    
    # Filter points within image bounds
    h, w = image.shape[:2]
    valid_proj = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                 (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    valid_points = points_2d[valid_proj]
    
    if len(valid_points) > 0:
        axes[1].scatter(valid_points[:, 0], valid_points[:, 1], 
                       c='cyan', s=1, alpha=0.7, label=f'{len(valid_points)} projected voxels')
    axes[1].set_title('SAM3D Layout Projection')
    axes[1].legend()
    axes[1].axis('off')
    
    # Overlay: mask + projected points
    axes[2].imshow(image)
    axes[2].imshow(mask, alpha=0.3, cmap='Reds', label='Ground truth mask')
    if len(valid_points) > 0:
        axes[2].scatter(valid_points[:, 0], valid_points[:, 1], 
                       c='cyan', s=1, alpha=0.7, label='Projected voxels')
    axes[2].set_title('Overlay: Mask + Projection')
    axes[2].legend()
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate alignment metrics
    mask_pixels = np.where(mask)
    mask_center = [np.mean(mask_pixels[1]), np.mean(mask_pixels[0])]  # [x, y]
    
    if len(valid_points) > 0:
        proj_center = np.mean(valid_points, axis=0)
        center_error = np.linalg.norm(np.array(mask_center) - proj_center)
        
        print(f"Layout accuracy metrics:")
        print(f"  Mask center: ({mask_center[0]:.1f}, {mask_center[1]:.1f})")
        print(f"  Projected center: ({proj_center[0]:.1f}, {proj_center[1]:.1f})")
        print(f"  Center error: {center_error:.1f} pixels")
        print(f"  Projected points in image: {len(valid_points)}/{len(points_2d)}")
        
        return center_error
    else:
        print("No valid projections found - layout may be severely wrong")
        return float('inf')


def check_layout_accuracy():
    """Check SAM3D layout accuracy for both views"""
    
    # Load VGGT data
    vggt_data = np.load("/scratch/cl927/scenes_sam3dvggt/apple_new/vggt_camera_poses.npz")
    intrinsics = vggt_data['intrinsic']
    image_names = vggt_data['image_names']
    
    # Load SAM3D outputs
    sam3d_dir = "sam3d_outputs"
    
    for view_idx in [1, 2]:
        print(f"\n=== Checking View {view_idx} Layout ===")
        
        # Load SAM3D data
        view_data = np.load(f"{sam3d_dir}/view_{view_idx:02d}.npz")
        voxels = view_data['coords']
        scale = view_data['scale']
        shift = view_data['shift']
        
        # Get corresponding image paths
        image_path = f"/scratch/cl927/scenes_sam3dvggt/apple_new/resized_images/resized_{view_idx:02d}.png"
        mask_path = f"/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_{view_idx:02d}_mask_1.png"
        
        print(f"Image: {image_path}")
        print(f"Scale: {scale}")
        print(f"Shift: {shift}")
        print(f"Voxels: {len(voxels)}")
        
        # Project and visualize
        error = project_voxels_to_image(
            voxels=voxels,
            scale=scale, 
            shift=shift,
            intrinsic=intrinsics[view_idx-1],  # 0-indexed
            image_path=image_path,
            mask_path=mask_path,
            output_path=f"layout_check_view_{view_idx}.png"
        )


if __name__ == "__main__":
    check_layout_accuracy()