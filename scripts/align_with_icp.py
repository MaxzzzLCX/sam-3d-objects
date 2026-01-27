import numpy as np
import trimesh
import open3d as o3d
from skimage import measure
from scipy import ndimage

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


def open3d_icp_alignment(source_points, target_points, method="point_to_plane"):
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
    threshold = 0.05  # Distance threshold
    
    if method == "point_to_plane":
        # Point-to-plane ICP (better for surfaces)
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
    else:
        # Point-to-point ICP
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


def align_with_open3d_surface(method="marching_cubes"):
    """Use Open3D ICP with proper surface extraction
    
    Args:
        method: One of ["surface_voxels", "marching_cubes", "upsampled_surface"]
    """
    
    valid_methods = ["surface_voxels", "marching_cubes", "upsampled_surface"]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
    
    print(f"Using surface extraction method: {method}")
    
    # Load data
    vggt_data = np.load("/scratch/cl927/scenes_sam3dvggt/apple_new/vggt_camera_poses.npz")
    vggt_points_3d = vggt_data['points_3d']
    
    view1_data = np.load("sam3d_outputs/view_01.npz")
    view2_data = np.load("sam3d_outputs/view_02.npz")
    
    # Load masks
    from PIL import Image as PILImage
    mask1 = np.array(PILImage.open("/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_01_mask_1.png").convert('L')) > 128
    mask2 = np.array(PILImage.open("/scratch/cl927/scenes_sam3dvggt/apple_new/masks/resized_02_mask_1.png").convert('L')) > 128
    
    mask1_resized = np.array(PILImage.fromarray(mask1.astype(np.uint8) * 255).resize((518, 518), PILImage.NEAREST)) > 128
    mask2_resized = np.array(PILImage.fromarray(mask2.astype(np.uint8) * 255).resize((518, 518), PILImage.NEAREST)) > 128
    
    # Extract VGGT surface points
    vggt_surface1 = vggt_points_3d[0][mask1_resized]
    vggt_surface2 = vggt_points_3d[1][mask2_resized]
    
    print(f"VGGT surface points: View1={len(vggt_surface1)}, View2={len(vggt_surface2)}")
    
    # Process each SAM3D view
    results = []
    
    for view_idx, (view_data, vggt_surface) in enumerate([(view1_data, vggt_surface1), (view2_data, vggt_surface2)]):
        print(f"\nProcessing view {view_idx + 1}:")
        print(f"  Original voxels: {len(view_data['coords'])}")
        
        # Extract surface based on chosen method
        if method == "surface_voxels":
            surface_voxels = extract_voxel_surface(view_data['coords'], grid_size=64)
            print(f"  Surface voxels: {len(surface_voxels)}")
            surface_points = surface_voxels
            
        elif method == "marching_cubes":
            surface_points, normals = extract_marching_cubes_surface(view_data['coords'], num_points=3000)
            print(f"  Marching cubes points: {len(surface_points)}")
            
        elif method == "upsampled_surface":
            surface_voxels = extract_voxel_surface(view_data['coords'], grid_size=64)
            surface_points = upsample_surface_voxels(surface_voxels, points_per_voxel=5)
            print(f"  Upsampled surface points: {len(surface_points)}")
        
        if len(surface_points) == 0:
            print(f"    No surface points generated for view {view_idx + 1}")
            results.append(np.array([]))
            continue
            
        # Convert to world coordinates with initial scaling
        surface_norm = (surface_points / 63.0) - 0.5
        
        # Initial scale estimation
        sam3d_size = surface_norm.max(axis=0) - surface_norm.min(axis=0) 
        vggt_size = vggt_surface.max(axis=0) - vggt_surface.min(axis=0)
        initial_scale = np.median(vggt_size / (sam3d_size + 1e-8))
        
        surface_scaled = surface_norm * initial_scale
        
        try:
            # Use Open3D ICP to find transformation
            icp_method = "point_to_plane" if method == "marching_cubes" else "point_to_point"
            aligned_surface_points, transformation = open3d_icp_alignment(
                surface_scaled, vggt_surface, method=icp_method
            )
            
            # Calculate final RMSE for surface alignment quality
            from scipy.spatial.distance import cdist
            distances = cdist(aligned_surface_points, vggt_surface)
            rmse = np.sqrt(np.mean(np.min(distances, axis=1)**2))
            print(f"  {method} surface RMSE: {rmse:.6f}")
            
            # Apply the SAME transformation to original SAM3D voxels
            # Convert original voxels to world coordinates with same initial scaling
            original_voxels_norm = (view_data['coords'] / 63.0) - 0.5
            original_voxels_scaled = original_voxels_norm * initial_scale
            
            # Apply ICP transformation to original voxels
            original_voxels_homogeneous = np.hstack([
                original_voxels_scaled, 
                np.ones((len(original_voxels_scaled), 1))
            ])
            transformed_voxels_homogeneous = (transformation @ original_voxels_homogeneous.T).T
            transformed_original_voxels = transformed_voxels_homogeneous[:, :3]
            
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
        pc.export(f"open3d_{method}_aligned.ply")
        
        print(f"\nSaved transformed original SAM3D voxels: {len(combined_points)} points")
        print(f"  Output: open3d_{method}_aligned.ply")
        print(f"  View 1: {len(results[0])} original voxels (red)")
        print(f"  View 2: {len(results[1])} original voxels (blue)")
        print(f"  Note: Transformation found using {method} surface extraction, applied to all original voxels")
    else:
        print("Failed to align both views")


if __name__ == "__main__":
    import sys
    
    # Default method
    method = "marching_cubes"
    
    # Parse command line argument if provided
    if len(sys.argv) > 1:
        method = sys.argv[1]
    
    align_with_open3d_surface(method)