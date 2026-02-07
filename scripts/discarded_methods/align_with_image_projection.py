import numpy as np
import trimesh
from PIL import Image

def correct_sam3d_layout(voxel_coords, intrinsic, image_path, mask_path):
    """
    Correct SAM3D layout by fitting voxel projection to image mask
    
    Returns corrected scale and shift
    """
    # Load mask to get object bounds
    mask = np.array(Image.open(mask_path).convert('L')) > 128
    mask_pixels = np.where(mask)
    
    if len(mask_pixels[0]) == 0:
        return np.ones(3), np.zeros(3)  # Default if no mask
    
    # Get mask bounding box
    mask_bbox = [
        mask_pixels[1].min(), mask_pixels[0].min(),  # x_min, y_min
        mask_pixels[1].max(), mask_pixels[0].max()   # x_max, y_max  
    ]
    mask_center = [(mask_bbox[0] + mask_bbox[2]) / 2, (mask_bbox[1] + mask_bbox[3]) / 2]
    mask_size = [mask_bbox[2] - mask_bbox[0], mask_bbox[3] - mask_bbox[1]]
    
    # Normalize voxel coordinates
    voxels_norm = (voxel_coords / 63.0) - 0.5  # [-0.5, 0.5]
    
    # Estimate depth from mask size (rough heuristic)
    # Larger objects in image are closer to camera
    focal_length = (intrinsic[0, 0] + intrinsic[1, 1]) / 2  # Average focal length
    estimated_depth = focal_length * 0.1 / np.mean(mask_size)  # Rough estimate
    estimated_depth = np.clip(estimated_depth, 0.3, 3.0)  # Reasonable bounds
    
    # Back-project mask center to 3D
    mask_center_3d = np.array([
        (mask_center[0] - intrinsic[0, 2]) * estimated_depth / intrinsic[0, 0],
        (mask_center[1] - intrinsic[1, 2]) * estimated_depth / intrinsic[1, 1], 
        estimated_depth
    ])
    
    # Estimate scale from mask size
    voxel_size_2d = np.array([
        voxels_norm[:, 0].max() - voxels_norm[:, 0].min(),
        voxels_norm[:, 1].max() - voxels_norm[:, 1].min()
    ])
    
    if voxel_size_2d[0] > 0 and voxel_size_2d[1] > 0:
        # Scale to match projected size
        scale_xy = np.array(mask_size) * estimated_depth / (focal_length * voxel_size_2d)
        scale_z = np.mean(scale_xy)  # Assume isotropic scaling for depth
        corrected_scale = np.array([scale_xy[0], scale_xy[1], scale_z])
    else:
        corrected_scale = np.array([0.1, 0.1, 0.1])  # Default small scale
    
    # Center the voxels at the estimated 3D position
    voxel_center = voxels_norm.mean(axis=0)
    corrected_shift = mask_center_3d - voxel_center * corrected_scale
    
    return corrected_scale, corrected_shift


def align_with_corrected_layout():
    """Align SAM3D voxels with corrected layout predictions"""
    
    # Load VGGT poses
    vggt_data = np.load("/scratch/cl927/scenes_sam3dvggt/apple_new/vggt_camera_poses.npz")
    extrinsic = vggt_data['extrinsic']
    intrinsic = vggt_data['intrinsic']
    
    # Load SAM3D data
    view1_data = np.load("sam3d_outputs/view_01.npz")
    view2_data = np.load("sam3d_outputs/view_02.npz")
    
    # Correct layouts
    scale1_corr, shift1_corr = correct_sam3d_layout(
        view1_data['coords'], intrinsic[0],
        "/scratch/cl927/scenes_sam3dvggt/apple_new/resized_images/resized_01.png",
        "/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_01_mask_1.png"
    )
    
    scale2_corr, shift2_corr = correct_sam3d_layout(
        view2_data['coords'], intrinsic[1], 
        "/scratch/cl927/scenes_sam3dvggt/apple_new/resized_images/resized_02.png",
        "/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_02_mask_1.png"
    )
    
    print(f"Original scale1: {view1_data['scale']}, corrected: {scale1_corr}")
    print(f"Original scale2: {view2_data['scale']}, corrected: {scale2_corr}")
    print(f"Original shift1: {view1_data['shift']}, corrected: {shift1_corr}")
    print(f"Original shift2: {view2_data['shift']}, corrected: {shift2_corr}")
    
    # Now use the corrected layout in the alignment function
    from scripts.alignment_raw_layout_predictions import align_multiview_voxels
    
    align_multiview_voxels(
        view1_voxels=view1_data['coords'],
        view2_voxels=view2_data['coords'],
        view1_scale=scale1_corr,  # Use corrected layout
        view1_shift=shift1_corr,
        view2_scale=scale2_corr,
        view2_shift=shift2_corr,
        pose1=extrinsic[0],
        pose2=extrinsic[1],
        output_path="corrected_layout_aligned.ply"
    )


if __name__ == "__main__":
    align_with_corrected_layout()