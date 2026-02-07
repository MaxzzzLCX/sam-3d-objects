import numpy as np
import trimesh
from PIL import Image

def improved_sam3d_layout(voxel_coords, intrinsic, vggt_depth, mask_path):
    """
    Use VGGT depth map to get accurate depth instead of heuristic estimation
    """
    # Load mask
    mask = np.array(Image.open(mask_path).convert('L')) > 128
    
    # Resize mask to match VGGT depth resolution (518x518)
    from PIL import Image as PILImage  
    mask_resized = np.array(PILImage.fromarray(mask.astype(np.uint8) * 255).resize((518, 518), PILImage.NEAREST)) > 128
    
    # Get depth values within mask region
    depth_masked = vggt_depth[mask_resized]
    if len(depth_masked) == 0:
        return np.ones(3), np.zeros(3)
        
    # Use median depth from VGGT (more robust than mean)
    estimated_depth = np.median(depth_masked[depth_masked > 0])
    
    # Get mask center in original resolution
    mask_pixels = np.where(mask)
    mask_center = [np.mean(mask_pixels[1]), np.mean(mask_pixels[0])]  # [x, y]
    mask_size = [mask_pixels[1].max() - mask_pixels[1].min(), 
                 mask_pixels[0].max() - mask_pixels[0].min()]
    
    # Back-project to 3D using accurate depth
    mask_center_3d = np.array([
        (mask_center[0] - intrinsic[0, 2]) * estimated_depth / intrinsic[0, 0],
        (mask_center[1] - intrinsic[1, 2]) * estimated_depth / intrinsic[1, 1], 
        estimated_depth
    ])
    
    # Estimate scale from mask size and depth
    voxels_norm = (voxel_coords / 63.0) - 0.5
    voxel_size_2d = np.array([
        voxels_norm[:, 0].max() - voxels_norm[:, 0].min(),
        voxels_norm[:, 1].max() - voxels_norm[:, 1].min()
    ])
    
    if voxel_size_2d[0] > 0 and voxel_size_2d[1] > 0:
        scale_xy = np.array(mask_size) * estimated_depth / (intrinsic[[0,1], [0,1]] * voxel_size_2d)
        scale_z = np.mean(scale_xy)
        corrected_scale = np.array([scale_xy[0], scale_xy[1], scale_z])
    else:
        corrected_scale = np.array([0.1, 0.1, 0.1])
    
    voxel_center = voxels_norm.mean(axis=0) 
    corrected_shift = mask_center_3d - voxel_center * corrected_scale
    
    return corrected_scale, corrected_shift


def align_with_vggt_depth():
    """Use VGGT depth for better layout correction"""
    
    # Load VGGT data including depth maps
    vggt_data = np.load("/scratch/cl927/scenes_sam3dvggt/apple_new/vggt_camera_poses.npz")
    extrinsic = vggt_data['extrinsic'] 
    intrinsic = vggt_data['intrinsic']
    depth_maps = vggt_data['depth_map']  # (N, H, W)
    
    # Load SAM3D data
    view1_data = np.load("sam3d_outputs/view_01.npz")
    view2_data = np.load("sam3d_outputs/view_02.npz")
    
    # Use VGGT depth for accurate layout correction
    scale1_corr, shift1_corr = improved_sam3d_layout(
        view1_data['coords'], intrinsic[0], depth_maps[0],
        "/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_01_mask_1.png"
    )
    
    scale2_corr, shift2_corr = improved_sam3d_layout(
        view2_data['coords'], intrinsic[1], depth_maps[1],
        "/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_02_mask_1.png"
    )
    
    print(f"VGGT-depth corrected scale1: {scale1_corr}, shift1: {shift1_corr}")
    print(f"VGGT-depth corrected scale2: {scale2_corr}, shift2: {shift2_corr}")
    
    # Use corrected layout in alignment
    from scripts.alignment_raw_layout_predictions import align_multiview_voxels
    
    align_multiview_voxels(
        view1_voxels=view1_data['coords'],
        view2_voxels=view2_data['coords'], 
        view1_scale=scale1_corr,
        view1_shift=shift1_corr,
        view2_scale=scale2_corr,
        view2_shift=shift2_corr,
        pose1=extrinsic[0],
        pose2=extrinsic[1], 
        output_path="vggt_depth_corrected_aligned.ply"
    )


if __name__ == "__main__":
    align_with_vggt_depth()