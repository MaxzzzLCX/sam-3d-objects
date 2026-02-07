import numpy as np
import trimesh
import open3d as o3d
from skimage import measure
from scipy import ndimage
import os

def extract_voxel_surface(voxel_coords, grid_size=64):
    """Extract only surface voxels (those with empty neighbors)"""
    
    # Create 3D occupancy grid
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = True
    
    surface_voxels = []
    
    for i, j, k in voxel_coords:
        # Check 6-connected neighbors  
        neighbors = [
            (i+1, j, k), (i-1, j, k),
            (i, j+1, k), (i, j-1, k),
            (i, j, k+1), (i, j, k-1)
        ]
        
        # If any neighbor is empty, this is a surface voxel
        is_surface = False
        for ni, nj, nk in neighbors:
            if (0 <= ni < grid_size and 0 <= nj < grid_size and 0 <= nk < grid_size):
                if not grid[ni, nj, nk]:
                    is_surface = True
                    break
            else:
                is_surface = True  # Edge of grid
                
        if is_surface:
            surface_voxels.append([i, j, k])
    
    return np.array(surface_voxels)


def extract_marching_cubes_surface(voxel_coords, grid_size=64, num_points=5000):
    """Use marching cubes to extract smooth surface points"""
    
    # Create occupancy grid
    grid = np.zeros((grid_size, grid_size, grid_size))
    if len(voxel_coords) > 0:
        grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1
    
    # Smooth with Gaussian filter for better surface
    grid_smooth = ndimage.gaussian_filter(grid, sigma=0.8)
    
    try:
        # Extract surface mesh using marching cubes
        verts, faces, normals, values = measure.marching_cubes(grid_smooth, level=0.5)
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Sample points uniformly on surface
        if len(mesh.faces) > 0:
            surface_points, face_indices = mesh.sample(num_points, return_index=True)
            surface_normals = mesh.face_normals[face_indices]
            return surface_points, surface_normals
        else:
            return np.array([]), np.array([])
            
    except (ValueError, RuntimeError) as e:
        print(f"    Marching cubes failed: {e}")
        return np.array([]), np.array([])


def upsample_surface_voxels(surface_voxels, points_per_voxel=8, noise_std=0.15):
    """Generate multiple points per surface voxel with small random offset"""
    
    if len(surface_voxels) == 0:
        return np.array([])
        
    upsampled_points = []
    
    for voxel in surface_voxels:
        # Generate multiple points around each voxel center
        for _ in range(points_per_voxel):
            # Add small random offset (within voxel boundaries)
            noise = np.random.normal(0, noise_std, 3)
            point = voxel.astype(float) + noise
            upsampled_points.append(point)
    
    return np.array(upsampled_points)


def open3d_icp_alignment(source_points, target_points, method="point_to_plane", threshold=0.05):
    """Use Open3D's robust ICP implementation"""
    
    # Convert to Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    
    # Estimate normals for point-to-plane ICP
    if method == "point_to_plane":
        source_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    
    # Set ICP parameters
    print(f"    Using ICP threshold: {threshold}")
    
    if method == "point_to_plane":

        # Point-to-plane ICP (better for surfaces)
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
    else:
        # Point-to-point ICP
        # Debugging
        # Save the point clouds before registration in two colors. In one file.
        source_temp = source_pcd.paint_uniform_color([1, 0, 0])  # Red
        target_temp = target_pcd.paint_uniform_color([0, 1, 0])  # Green

        o3d.io.write_point_cloud("icp_debug_before.ply", source_temp + target_temp)
        print(f"Saved a debug point cloud before ICP to 'icp_debug_before.ply'")
        # raise NotImplementedError("Debugging ICP - stop here")
    
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
    
    # Apply transformation
    source_pcd.transform(result.transformation)
    aligned_points = np.asarray(source_pcd.points)
    
    # Debug: Print available attributes
    print(f"    Result type: {type(result)}")
    print(f"    Available attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
    
    # Check if result has the expected attributes (different Open3D versions)
    converged = getattr(result, 'converged', True)  # Default to True if not available
    rmse = getattr(result, 'inlier_rmse', 0.0)     # Default to 0.0 if not available
    correspondences = getattr(result, 'correspondence_set', [])  # Default to empty list
    
    print(f"    Open3D ICP converged: {converged}")
    print(f"    Final RMSE: {rmse:.6f}")
    print(f"    Correspondences: {len(correspondences)}")
    
    return aligned_points, result.transformation


def align_with_open3d_surface(method="surface_voxels"):
    """Use Open3D ICP with proper surface extraction
    
    Args:
        method: One of ["surface_voxels", "upsampled_surface"]
    """
    
    valid_methods = ["surface_voxels", "upsampled_surface"]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
    
    print(f"Using surface extraction method: {method}")
    
    """
    # Load data
    vggt_data = np.load("/scratch/cl927/nutritionverse-3d-new/_test_id-11-red-apple-145g/SAM3D_multiview_prediction/vggt_camera_poses.npz")
    vggt_points_3d = vggt_data['points_3d']
    print(f"Loaded OLD VGGT points: {vggt_points_3d.shape}")

    # DEBUG: Save the loaded VGGT points for debugging
    # (N, 518, 518, 3) -> (N*518*518, 3)
    combined_vggt_points = vggt_points_3d.reshape(-1, 3)
    
    # Remove invalid points (NaN, inf, or zeros)
    # valid_mask = np.isfinite(combined_vggt_points).all(axis=1) & (np.abs(combined_vggt_points).sum(axis=1) > 1e-6)
    # combined_vggt_points = combined_vggt_points[valid_mask]
    
    print(f"Shape of combined VGGT points: {combined_vggt_points.shape}")
    print(f"Valid points after filtering: {len(combined_vggt_points)}")
    
    # vggt_pcd = o3d.geometry.PointCloud()
    # vggt_pcd.points = o3d.utility.Vector3dVector(combined_vggt_points)
    # o3d.io.write_point_cloud("NEW_debug_vggt_points.ply", vggt_pcd)
    pc = trimesh.PointCloud(combined_vggt_points)
    pc.export("NEW_debug_vggt_points.ply")
    print(f"Saved debug VGGT points to 'NEW_debug_vggt_points.ply'")
    raise NotImplementedError("Debugging VGGT points - stop here")
    """

    vggt_data = o3d.io.read_point_cloud("/scratch/cl927/nutritionverse-3d-new/_test_id-11-red-apple-145g/SAM3D_multiview_prediction/VGGT_view_0-1_masked_unscaled_conf0.0/points.ply")
    vggt_points_3d = np.asarray(vggt_data.points)  # numpy array
    print(f"Loaded VGGT points: {vggt_points_3d.shape}")

    # DEBUG: Save the loaded VGGT points for debugging
    pc = trimesh.PointCloud(vggt_points_3d)
    pc.export("NEWMETHOD_debug_vggt_points.ply")
    print(f"Saved debug VGGT points to 'NEWMETHOD_debug_vggt_points.ply'")
    # raise NotImplementedError("Debugging VGGT points - stop here")
    
    view1_data = np.load("/scratch/cl927/nutritionverse-3d-new/_test_id-11-red-apple-145g/SAM3D_singleview_prediction/000/sam3d_voxels_normalized.npy")
    view2_data = np.load("/scratch/cl927/nutritionverse-3d-new/_test_id-11-red-apple-145g/SAM3D_singleview_prediction/001/sam3d_voxels_normalized.npy")
    
    # Redo the rotation when we saved the SAM3D voxels
    # The rotation before was -90˚ around X; so we apply +90˚ here 
    rotation_matrix = np.array([
        [1,  0,  0],
        [0,  0,  -1],
        [0, 1,  0]
    ], dtype=float)
    view1_data = view1_data @ rotation_matrix.T
    view2_data = view2_data @ rotation_matrix.T
    

    # # Denormalize the SAM3D voxel coordinates to [0, 63]
    view1_data = ((view1_data + 0.5) * 63.0).astype(int)
    view2_data = ((view2_data + 0.5) * 63.0).astype(int)

    # # Load masks
    # from PIL import Image as PILImage
    # mask1 = np.array(PILImage.open("/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_01_mask_1.png").convert('L')) > 128
    # mask2 = np.array(PILImage.open("/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_02_mask_1.png").convert('L')) > 128
    
    # mask1_resized = np.array(PILImage.fromarray(mask1.astype(np.uint8) * 255).resize((518, 518), PILImage.NEAREST)) > 128
    # mask2_resized = np.array(PILImage.fromarray(mask2.astype(np.uint8) * 255).resize((518, 518), PILImage.NEAREST)) > 128
    
    # # Extract VGGT surface points
    # vggt_surface1 = vggt_points_3d[0][mask1_resized]
    # vggt_surface2 = vggt_points_3d[1][mask2_resized]
    # vggt_surfaces_all = np.vstack([vggt_surface1, vggt_surface2])
    vggt_surfaces_all = vggt_points_3d
    
    # print(f"Total VGGT points after masking: {len(vggt_surfaces_all)}")
    
    # Process each SAM3D view
    results = []
    
    for view_idx, view_data in enumerate([view1_data, view2_data]):
        print(f"\nProcessing view {view_idx + 1}:")
        print(f"Original voxels: {len(view_data)}")
        # print(f"  Original voxels: {len(view_data['coords'])}")
        
        # Extract surface based on chosen method
        if method == "surface_voxels":
            surface_voxels = extract_voxel_surface(view_data, grid_size=64)
            print(f"  Surface voxels: {len(surface_voxels)}")
            surface_points = surface_voxels
            
        elif method == "upsampled_surface":
            surface_voxels = extract_voxel_surface(view_data, grid_size=64)
            surface_points = upsample_surface_voxels(surface_voxels, points_per_voxel=5)
            print(f"  Upsampled surface points: {len(surface_points)}")
        
        if len(surface_points) == 0:
            print(f"    No surface points generated for view {view_idx + 1}")
            results.append(np.array([]))
            continue
            
        # Convert to world coordinates with initial scaling
        surface_norm = (surface_points / 63.0) - 0.5
        
        # Debug coordinate ranges
        print(f"  SAM3D surface_norm range: [{surface_norm.min(axis=0)}, {surface_norm.max(axis=0)}]")
        print(f"  VGGT surface range: [{vggt_surfaces_all.min(axis=0)}, {vggt_surfaces_all.max(axis=0)}]")
        
        # Initial scale estimation
        sam3d_size = surface_norm.max(axis=0) - surface_norm.min(axis=0) 
        vggt_size = vggt_surfaces_all.max(axis=0) - vggt_surfaces_all.min(axis=0)
        
        print(f"  SAM3D size: {sam3d_size}")
        print(f"  VGGT size: {vggt_size}")
        
        size_ratios = vggt_size / (sam3d_size + 1e-8)
        initial_scale = np.median(size_ratios)
        
        print(f"  Size ratios: {size_ratios}")
        print(f"  Initial scale factor: {initial_scale}")
        
        surface_scaled = surface_norm * initial_scale
        
        print(f"  SAM3D after scaling range: [{surface_scaled.min(axis=0)}, {surface_scaled.max(axis=0)}]")
        
        # Check distance between centroids
        sam3d_centroid = surface_scaled.mean(axis=0)
        vggt_centroid = vggt_surfaces_all.mean(axis=0)
        centroid_distance = np.linalg.norm(sam3d_centroid - vggt_centroid)
        
        print(f"  SAM3D centroid: {sam3d_centroid}")
        print(f"  VGGT centroid: {vggt_centroid}")
        print(f"  Centroid distance: {centroid_distance}")
        
        # Try centering both point clouds first
        surface_centered = surface_scaled - sam3d_centroid
        vggt_centered = vggt_surfaces_all - vggt_centroid
        
        print(f"  After centering - SAM3D range: [{surface_centered.min(axis=0)}, {surface_centered.max(axis=0)}]")
        print(f"  After centering - VGGT range: [{vggt_centered.min(axis=0)}, {vggt_centered.max(axis=0)}]")
        
        try:
            # Use Open3D ICP to find transformation  
            # Use centered point clouds and smaller, more precise threshold
            icp_method = "point_to_point"
            # Use much smaller threshold for precision - around 2-5% of object size
            object_size = max(np.linalg.norm(vggt_size), np.linalg.norm(sam3d_size * initial_scale))
            threshold = object_size * 0.10  # 10% of object size for tight alignment
            threshold = max(0.005, min(threshold, 0.2))  # Clamp between 0.005 and 0.1
            print(f"  Using ICP threshold: {threshold} (object size: {object_size})")
            
            aligned_surface_points, transformation = open3d_icp_alignment(
                surface_centered, vggt_centered, method=icp_method, threshold=threshold
            )
            
            # Debug: Check actual alignment
            print(f"  Transformation matrix:")
            print(f"    {transformation}")
            print(f"  Surface points before ICP: {surface_centered.mean(axis=0)} (centroid)")
            print(f"  Surface points after ICP: {aligned_surface_points.mean(axis=0)} (centroid)")
            print(f"  VGGT target centroid: {vggt_centered.mean(axis=0)}")
            print(f"  Distance between aligned SAM3D and VGGT centroids: {np.linalg.norm(aligned_surface_points.mean(axis=0) - vggt_centered.mean(axis=0))}")
            
            # Calculate final RMSE for surface alignment quality
            from scipy.spatial.distance import cdist
            distances = cdist(aligned_surface_points, vggt_centered)
            rmse = np.sqrt(np.mean(np.min(distances, axis=1)**2))
            print(f"  {method} surface RMSE: {rmse:.6f}")
            print(f"  Average distance to closest VGGT point: {np.mean(np.min(distances, axis=1)):.6f}")
            
            # Check if ICP actually moved the points significantly
            original_distances = cdist(surface_centered, vggt_centered)
            original_rmse = np.sqrt(np.mean(np.min(original_distances, axis=1)**2))
            print(f"  RMSE before ICP: {original_rmse:.6f}")
            print(f"  RMSE improvement: {((original_rmse - rmse) / original_rmse * 100):.1f}%")
            
            # Transform back to world coordinates and apply to original voxels
            # The transformation was computed on centered data, so we need to account for that
            original_voxels_norm = (view_data / 63.0) - 0.5
            original_voxels_scaled = original_voxels_norm * initial_scale
            original_voxels_centered = original_voxels_scaled - sam3d_centroid
            
            # Apply ICP transformation
            original_voxels_homogeneous = np.hstack([
                original_voxels_centered, 
                np.ones((len(original_voxels_centered), 1))
            ])
            transformed_voxels_homogeneous = (transformation @ original_voxels_homogeneous.T).T
            transformed_original_voxels = transformed_voxels_homogeneous[:, :3]
            
            # Translate back to VGGT coordinate system
            transformed_original_voxels += vggt_centroid
            
            print(f"  Applied transformation to {len(transformed_original_voxels)} original voxels")
            results.append(transformed_original_voxels)
            
        except Exception as e:
            print(f"  {method} failed: {e}")
            results.append(np.array([]))
    
    # Combine results
    if len(results) == 2 and len(results[0]) > 0 and len(results[1]) > 0:
        combined_points = np.vstack(results)
        
        # Create colors
        colors1 = np.tile([255, 0, 0], (len(results[0]), 1))
        colors2 = np.tile([0, 0, 255], (len(results[1]), 1)) 
        combined_colors = np.vstack([colors1, colors2]).astype(np.uint8)
        
        # Save result
        pc = trimesh.PointCloud(combined_points, colors=combined_colors)
        pc_path = f"/scratch/cl927/nutritionverse-3d-new/_test_id-11-red-apple-145g/SAM3D_multiview_prediction/aligned_outputs/open3d_{method}_aligned.ply"
        os.makedirs(os.path.dirname(pc_path), exist_ok=True)
        pc.export(pc_path)
        print(f"\nSaved combined to {pc_path}")
        
        print(f"\nSaved transformed original SAM3D voxels: {len(combined_points)} points")
        print(f"  Output: open3d_{method}_aligned.ply")
        print(f"  View 1: {len(results[0])} original voxels (red)")
        print(f"  View 2: {len(results[1])} original voxels (blue)")
        print(f"  Note: Transformation found using {method} surface extraction, applied to all original voxels")
        
        # Save individual view alignments for visualization
        # View 1 + VGGT
        view1_vggt_points = np.vstack([results[0], vggt_surfaces_all])
        view1_sam3d_colors = np.tile([255, 0, 0], (len(results[0]), 1))  # Red for SAM3D
        view1_vggt_colors = np.tile([0, 255, 0], (len(vggt_surfaces_all), 1))  # Green for VGGT
        view1_vggt_combined_colors = np.vstack([view1_sam3d_colors, view1_vggt_colors]).astype(np.uint8)
        
        pc_view1_vggt = trimesh.PointCloud(view1_vggt_points, colors=view1_vggt_combined_colors)
        view1_vggt_path = f"/scratch/cl927/nutritionverse-3d-new/_test_id-11-red-apple-145g/SAM3D_multiview_prediction/aligned_outputs/open3d_{method}_view1_vggt.ply"
        pc_view1_vggt.export(view1_vggt_path)
        
        # View 2 + VGGT
        view2_vggt_points = np.vstack([results[1], vggt_surfaces_all])
        view2_sam3d_colors = np.tile([0, 0, 255], (len(results[1]), 1))  # Blue for SAM3D
        view2_vggt_colors = np.tile([0, 255, 0], (len(vggt_surfaces_all), 1))  # Green for VGGT
        view2_vggt_combined_colors = np.vstack([view2_sam3d_colors, view2_vggt_colors]).astype(np.uint8)
        
        pc_view2_vggt = trimesh.PointCloud(view2_vggt_points, colors=view2_vggt_combined_colors)
        view2_vggt_path = f"/scratch/cl927/nutritionverse-3d-new/_test_id-11-red-apple-145g/SAM3D_multiview_prediction/aligned_outputs/open3d_{method}_view2_vggt.ply"
        pc_view2_vggt.export(view2_vggt_path)
        
        print(f"  Alignment visualization files:")
        print(f"    View1+VGGT: open3d_{method}_view1_vggt.ply (SAM3D=red, VGGT=green)")
        print(f"    View2+VGGT: open3d_{method}_view2_vggt.ply (SAM3D=blue, VGGT=green)")
    else:
        print("Failed to align both views")


if __name__ == "__main__":
    import sys
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--method', type=str, default='surface_voxels',
    #                     choices=['surface_voxels', 'upsampled_surface'],
    #                     help='Surface extraction method for ICP alignment')
    # parser.add_argument('--scene_dir', type=str, required=True)
    # parser.add_argument('--vggt', type=str, default=None, help='Path to VGGT camera poses .npz file')



    
    # Default method
    method = "surface_voxels"
    
    # Parse command line argument if provided
    if len(sys.argv) > 1:
        method = sys.argv[1]
    
    align_with_open3d_surface(method)