import numpy as np
import trimesh
import open3d as o3d
from skimage import measure
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

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


def calculate_icp_transformations(method="marching_cubes"):
    """Calculate ICP transformations for both views without saving results
    
    Args:
        method: One of ["surface_voxels", "marching_cubes", "upsampled_surface"]
        
    Returns:
        tuple: (transform_v1_to_world, transform_v2_to_world, view1_data, view2_data, vggt_surfaces)
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
    
    # Process each SAM3D view to get transformations
    transformations = []
    
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
            transformations.append(None)
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
            
            transformations.append(transformation)
            
        except Exception as e:
            print(f"  {method} failed: {e}")
            transformations.append(None)
    
    # Return transformation matrices and data for next stage
    if len(transformations) == 2 and transformations[0] is not None and transformations[1] is not None:
        print("\nSuccessfully calculated transformations for both views")
        return transformations[0], transformations[1], view1_data, view2_data, (vggt_surface1, vggt_surface2)
    else:
        print("Failed to calculate transformations for both views")
        return None, None, None, None, None


def create_occupancy_grid(view_data, grid_size=64):
    """Load occupancy grid from SAM3D data with probability values"""
    if 'occupancy_grid' in view_data:
        # Use saved probability grid
        grid = view_data['occupancy_grid'].astype(np.float32)
        print(f"    Using saved occupancy grid with values: min={grid.min():.6f}, max={grid.max():.6f}, mean={grid.mean():.6f}")
    else:
        # Fallback: create binary grid from coordinates (for backwards compatibility)
        print(f"    Warning: No occupancy_grid found, creating binary grid from coordinates")
        grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        if len(view_data['coords']) > 0:
            grid[view_data['coords'][:, 0], view_data['coords'][:, 1], view_data['coords'][:, 2]] = 1.0
    return grid


def trilinear_interpolate_grid(source_grid, transformation_matrix, target_grid_shape=(64, 64, 64)):
    """
    Interpolate source grid into target coordinate system using transformation matrix
    
    Args:
        source_grid: 3D numpy array (64, 64, 64) - source occupancy grid
        transformation_matrix: 4x4 transformation matrix from ICP
        target_grid_shape: shape of target grid (default 64x64x64)
    
    Returns:
        interpolated_grid: 3D numpy array with same shape as target
    """
    # Create interpolator for source grid
    x = np.arange(source_grid.shape[0])
    y = np.arange(source_grid.shape[1]) 
    z = np.arange(source_grid.shape[2])
    interpolator = RegularGridInterpolator((x, y, z), source_grid, 
                                         bounds_error=False, fill_value=0.0)
    
    # Create target grid coordinates
    target_coords = np.mgrid[0:target_grid_shape[0], 0:target_grid_shape[1], 0:target_grid_shape[2]]
    target_points = np.stack([target_coords[0].ravel(), 
                             target_coords[1].ravel(), 
                             target_coords[2].ravel()], axis=1).astype(np.float32)
    
    # Convert target voxel indices to normalized coordinates [-0.5, 0.5]
    target_points_norm = (target_points / 63.0) - 0.5
    
    # Apply inverse transformation to map target points back to source coordinate system
    # First we need to undo the initial scaling used in ICP
    inv_transformation = np.linalg.inv(transformation_matrix)
    
    # Convert to homogeneous coordinates
    target_points_homo = np.hstack([target_points_norm, np.ones((len(target_points_norm), 1))])
    
    # Apply inverse transformation
    source_points_homo = (inv_transformation @ target_points_homo.T).T
    source_points_norm = source_points_homo[:, :3]
    
    # Convert back to voxel indices [0, 63]
    source_voxel_coords = (source_points_norm + 0.5) * 63.0
    
    # Interpolate source grid at these coordinates
    interpolated_values = interpolator(source_voxel_coords)
    
    # Reshape back to grid
    interpolated_grid = interpolated_values.reshape(target_grid_shape)
    
    return interpolated_grid


def align_with_axis_interpolation(transform_v1_to_world, transform_v2_to_world, 
                                  view1_data, view2_data, vggt_surfaces, method="marching_cubes"):
    """
    Align two views using pre-calculated transformations and interpolate view2 into view1's coordinate system
    
    Args:
        transform_v1_to_world: 4x4 transformation matrix from view1 to world
        transform_v2_to_world: 4x4 transformation matrix from view2 to world
        view1_data: SAM3D data for view 1
        view2_data: SAM3D data for view 2
        vggt_surfaces: Tuple of (vggt_surface1, vggt_surface2)
        method: Surface extraction method used in ICP
    """
    print(f"Starting axis-aligned interpolation with method: {method}")
    
    # Unpack VGGT surfaces
    vggt_surface1, vggt_surface2 = vggt_surfaces
    
    print(f"Using pre-calculated transformations")
    print(f"VGGT surface points: View1={len(vggt_surface1)}, View2={len(vggt_surface2)}")
    
    # Create occupancy grids for both views (with probability values)
    view1_grid = create_occupancy_grid(view1_data)
    view2_grid = create_occupancy_grid(view2_data)
    
    print(f"Created occupancy grids: View1 occupancy={view1_grid.sum()}, View2 occupancy={view2_grid.sum()}")
    
    # Use pre-calculated transformations: view2 -> world -> view1
    print("\nUsing pre-calculated transformations (View2 -> View1):")
    
    try:
        # Compose transformations: view2 -> world -> view1
        transform_world_to_v1 = np.linalg.inv(transform_v1_to_world)
        transform_v2_to_v1 = transform_world_to_v1 @ transform_v2_to_world
        
        print("Successfully computed view2 -> view1 transformation")
        
        # Interpolate view2 grid into view1 coordinate system
        print("Interpolating view2 grid into view1 coordinate system...")
        view2_interpolated = trilinear_interpolate_grid(view2_grid, transform_v2_to_v1)
        
        print(f"Interpolated grid occupancy: {view2_interpolated.sum()}")
        
        # Convert both grids to point clouds for visualization (threshold logits at 0)
        view1_coords = np.array(np.where(view1_grid > 0)).T
        view2_interp_coords = np.array(np.where(view2_interpolated > 0)).T
        
        print(f"Axis-aligned results:")
        print(f"  View1 (original): {len(view1_coords)} voxels")
        print(f"  View2 (interpolated): {len(view2_interp_coords)} voxels")
        
        # Save results
        if len(view1_coords) > 0 and len(view2_interp_coords) > 0:
            # Convert to normalized coordinates for visualization
            view1_points = (view1_coords / 63.0) - 0.5
            view2_interp_points = (view2_interp_coords / 63.0) - 0.5
            
            # Combine points
            combined_points = np.vstack([view1_points, view2_interp_points])
            
            # Create colors (red for view1, blue for interpolated view2)
            colors1 = np.tile([255, 0, 0], (len(view1_coords), 1))
            colors2 = np.tile([0, 0, 255], (len(view2_interp_coords), 1))
            combined_colors = np.vstack([colors1, colors2]).astype(np.uint8)
            
            pc = trimesh.PointCloud(combined_points, colors=combined_colors)
            pc.export(f"axis_aligned_{method}.ply")
            
            # Save aligned grids as NPZ for fusion
            np.savez(f"aligned_grids_{method}.npz",
                     view1_grid=view1_grid,
                     view2_grid=view2_interpolated,
                     method=method)
            print(f"Saved aligned grids: aligned_grids_{method}.npz")
            
            print(f"Saved axis-aligned result: axis_aligned_{method}.ply")
            print(f"  View1: {len(view1_coords)} voxels (red)")
            print(f"  View2 interpolated: {len(view2_interp_coords)} voxels (blue)")
            print(f"  Note: Both views now share the same coordinate system")
            print(f"  Ready for fusion: use aligned_grids_{method}.npz")
        else:
            print("No voxels in aligned result!")
            
    except Exception as e:
        print(f"Axis alignment failed: {e}")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    method = "marching_cubes"
    if len(sys.argv) > 1:
        method = sys.argv[1]
    
    print("=== Stage 1: Calculating ICP Transformations ===")
    transform_v1, transform_v2, view1_data, view2_data, vggt_surfaces = calculate_icp_transformations(method)
    
    if transform_v1 is not None and transform_v2 is not None:
        print("\n=== Stage 2: Axis Alignment with Trilinear Interpolation ===")
        align_with_axis_interpolation(transform_v1, transform_v2, view1_data, view2_data, vggt_surfaces, method)
    else:
        print("\nFailed to calculate transformations, skipping axis alignment")