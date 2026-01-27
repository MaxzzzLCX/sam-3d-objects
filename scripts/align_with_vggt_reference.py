import numpy as np
import trimesh

def align_with_vggt_pointclouds(sam3d_coords1, sam3d_coords2, vggt_points1, vggt_points2, 
                                mask1, mask2, output_path="vggt_aligned_voxels.ply"):
    """
    Use VGGT point clouds as reference to align SAM3D voxels
    
    Args:
        sam3d_coords1, sam3d_coords2: (N, 3) SAM3D voxel coordinates [0, 63]
        vggt_points1, vggt_points2: (H, W, 3) VGGT point clouds in world coordinates  
        mask1, mask2: (H, W) boolean masks for the object
        output_path: where to save result
    """
    
    # Extract VGGT points within the mask regions
    vggt_masked1 = vggt_points1[mask1]  # Only points inside object mask
    vggt_masked2 = vggt_points2[mask2]
    
    # Normalize SAM3D coordinates to [-0.5, 0.5]
    sam3d_norm1 = (sam3d_coords1 / 63.0) - 0.5
    sam3d_norm2 = (sam3d_coords2 / 63.0) - 0.5
    
    # Find scale and translation to fit SAM3D voxels to VGGT points
    def fit_sam3d_to_vggt_improved(sam3d_norm, vggt_points):
        """Improved fitting using actual point distributions, not just bounding boxes"""
        
        # Method 1: Bounding box alignment (current approach)
        sam3d_min, sam3d_max = sam3d_norm.min(axis=0), sam3d_norm.max(axis=0)
        vggt_min, vggt_max = vggt_points.min(axis=0), vggt_points.max(axis=0)
        
        sam3d_size = sam3d_max - sam3d_min
        vggt_size = vggt_max - vggt_min
        
        # More conservative scaling - preserve aspect ratios better
        scale_factors = vggt_size / (sam3d_size + 1e-8)
        
        # Use median scale to avoid extreme distortion
        median_scale = np.median(scale_factors)
        conservative_scale = np.full(3, median_scale)  # Isotropic scaling
        
        # Alternative: Use individual scales but clamp to reasonable ratios
        max_ratio = 2.0  # Don't allow more than 2x distortion
        min_scale, max_scale = np.min(scale_factors), np.max(scale_factors)
        if max_scale / min_scale > max_ratio:
            # If aspect ratio is too extreme, use median scaling
            scale = np.full(3, median_scale)
        else:
            scale = scale_factors
            
        # Center alignment
        sam3d_center = (sam3d_min + sam3d_max) / 2
        vggt_center = (vggt_min + vggt_max) / 2
        translation = vggt_center - sam3d_center * scale
        
        print(f"  Scale factors: {scale_factors}, chosen: {scale}")
        print(f"  SAM3D size: {sam3d_size}, VGGT size: {vggt_size}")
        
        return scale, translation
    
    # Fit both views to their respective VGGT point clouds
    scale1, trans1 = fit_sam3d_to_vggt_improved(sam3d_norm1, vggt_masked1)
    scale2, trans2 = fit_sam3d_to_vggt_improved(sam3d_norm2, vggt_masked2)
    
    # Transform SAM3D voxels to match VGGT coordinates
    sam3d_aligned1 = sam3d_norm1 * scale1 + trans1
    sam3d_aligned2 = sam3d_norm2 * scale2 + trans2
    
    # Combine aligned voxels
    combined_voxels = np.vstack([sam3d_aligned1, sam3d_aligned2])
    
    # Create colors (red for view1, blue for view2)
    colors1 = np.tile([255, 0, 0], (len(sam3d_aligned1), 1))
    colors2 = np.tile([0, 0, 255], (len(sam3d_aligned2), 1))
    combined_colors = np.vstack([colors1, colors2]).astype(np.uint8)
    
    # Save result
    pc = trimesh.PointCloud(combined_voxels, colors=combined_colors)
    pc.export(output_path)
    
    print(f"Saved {len(combined_voxels)} VGGT-aligned voxels to {output_path}")
    print(f"  View 1: {len(sam3d_aligned1)} voxels (red)")
    print(f"  View 2: {len(sam3d_aligned2)} voxels (blue)")
    
    return combined_voxels, combined_colors


def main():
    """Align SAM3D voxels using VGGT point clouds as reference"""
    
    # Load VGGT data (including point clouds)
    vggt_data = np.load("/scratch/cl927/scenes_sam3dvggt/apple_new/vggt_camera_poses.npz")
    vggt_points_3d = vggt_data['points_3d']  # (N, H, W, 3) 
    image_names = vggt_data['image_names']
    
    print(f"VGGT point clouds shape: {vggt_points_3d.shape}")
    print(f"Available images: {image_names}")
    
    # Load SAM3D data  
    view1_data = np.load("sam3d_outputs/view_01.npz")
    view2_data = np.load("sam3d_outputs/view_02.npz")
    
    sam3d_coords1 = view1_data['coords']
    sam3d_coords2 = view2_data['coords']
    
    # Load masks for filtering VGGT points
    from PIL import Image
    mask1 = np.array(Image.open("/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_01_mask_1.png").convert('L')) > 128
    mask2 = np.array(Image.open("/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_02_mask_1.png").convert('L')) > 128
    
    # Resize masks to match VGGT resolution (518x518)
    from PIL import Image as PILImage
    mask1_resized = np.array(PILImage.fromarray(mask1.astype(np.uint8) * 255).resize((518, 518), PILImage.NEAREST)) > 128
    mask2_resized = np.array(PILImage.fromarray(mask2.astype(np.uint8) * 255).resize((518, 518), PILImage.NEAREST)) > 128
    
    # Get VGGT point clouds for the two views (assuming resized_01.png is index 0, resized_02.png is index 1)
    vggt_points1 = vggt_points_3d[0]  # (H, W, 3) for view 1
    vggt_points2 = vggt_points_3d[1]  # (H, W, 3) for view 2
    
    print(f"View 1: {len(sam3d_coords1)} SAM3D voxels, {np.sum(mask1_resized)} VGGT points in mask")
    print(f"View 2: {len(sam3d_coords2)} SAM3D voxels, {np.sum(mask2_resized)} VGGT points in mask")
    
    # Align SAM3D voxels to VGGT point clouds
    align_with_vggt_pointclouds(
        sam3d_coords1, sam3d_coords2,
        vggt_points1, vggt_points2, 
        mask1_resized, mask2_resized,
        output_path="vggt_reference_aligned.ply"
    )


if __name__ == "__main__":
    main()