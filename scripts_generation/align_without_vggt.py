import numpy as np
import trimesh
import open3d as o3d
from skimage import measure
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import argparse
import os

from chamfer_distance_evaluation import calculate_chamfer_distance


def normalize_point_cloud(points):
    """ Normalize point cloud to fit within unit cube centered at origin. """
    points = points.copy()  # Don't modify original
    
    # Step 1: Center the point cloud at origin
    center = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
    points -= center
    
    # Step 2: Scale to fit within unit cube [-0.5, 0.5]^3
    max_coords = np.max(points, axis=0)
    min_coords = np.min(points, axis=0)
    max_extent = np.max(max_coords - min_coords)  # Largest dimension
    
    if max_extent > 0:  # Avoid division by zero
        points /= max_extent  # Scale so largest dimension is 1.0
    
    return points


def icp_alignment(voxel_one_path, voxel_two_path, output_dir, threshold=0.15, max_iterations=100):
    """
    Perform ICP alignment of predicted points to ground truth points.
    """

    # Load point clouds
    voxel_one_pc = o3d.io.read_point_cloud(voxel_one_path)
    voxel_two_pc = o3d.io.read_point_cloud(voxel_two_path)
    voxel_one = np.asarray(voxel_one_pc.points)
    voxel_two = np.asarray(voxel_two_pc.points)

    # Randomly sample the same number of points as pred points from gt_points
    print("Number of Voxel points:", len(voxel_one))
    print("Number of Pred points:", len(voxel_two))
    if len(voxel_one) > len(voxel_two):
        indices = np.random.choice(len(voxel_one), size=len(voxel_two), replace=False)
        voxel_one = voxel_one[indices]
        print(f"Sampled {len(voxel_two)} points from Voxel one for ICP alignment.")

    # Before ICP, calculate Chamfer distance
    chamfer_results = calculate_chamfer_distance(voxel_one, voxel_two)
    print(f"Chamfer distance before ICP: {chamfer_results['bidirectional_chamfer']}")


    # Convert np.ndarrays to Open3D point clouds
    voxel_one_pcd = o3d.geometry.PointCloud()
    voxel_one_pcd.points = o3d.utility.Vector3dVector(voxel_one)
    voxel_two_pcd = o3d.geometry.PointCloud()
    voxel_two_pcd.points = o3d.utility.Vector3dVector(voxel_two)
    
    # Perform ICP alignment
    result = o3d.pipelines.registration.registration_icp(
        voxel_two_pcd, voxel_one_pcd, max_correspondence_distance=threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    transformation = result.transformation
    voxel_two_pcd.transform(transformation)
    voxel_two_aligned = np.asarray(voxel_two_pcd.points)

    # After ICP, calculate Chamfer distance
    chamfer_results = calculate_chamfer_distance(voxel_one, voxel_two_aligned)
    print(f"Chamfer distance after ICP: {chamfer_results['bidirectional_chamfer']}")

    # Save as combined point cloud
    icp_alignment_path = f"{output_dir}/icp_aligned_combined.ply"
    os.makedirs(os.path.dirname(icp_alignment_path), exist_ok=True)

    combined_points = np.concatenate([voxel_one, voxel_two_aligned], axis=0)
    voxel_two_colors = np.tile([1.0, 0.0, 0.0], (len(voxel_two_aligned), 1))  # Red
    voxel_one_colors = np.tile([0.0, 0.0, 1.0], (len(voxel_one), 1))      # Blue

    combined_colors = np.concatenate([voxel_one_colors, voxel_two_colors], axis=0)
    combined_pc = o3d.geometry.PointCloud()
    combined_pc.points = o3d.utility.Vector3dVector(combined_points)
    combined_pc.colors = o3d.utility.Vector3dVector(combined_colors)
    o3d.io.write_point_cloud(icp_alignment_path, combined_pc)
    print(f"Saved ICP-aligned combined point cloud to {icp_alignment_path}")

    return voxel_one, voxel_two_aligned, transformation


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

def fuse_by_average(grid_one, grid_two):
    
    # grid_one and grid_two are logits
    # Average their probability with numerical stability
    prob_one = 1 / (1 + np.exp(-grid_one))
    prob_two = 1 / (1 + np.exp(-grid_two))
    avg_prob = (prob_one + prob_two) / 2
    
    # Clip to avoid division by zero when converting back to logits
    epsilon = 1e-7
    avg_prob = np.clip(avg_prob, epsilon, 1 - epsilon)
    
    # Convert back to logits
    avg_logit = np.log(avg_prob / (1 - avg_prob))
    return avg_logit

def fuse_by_min_entropy(grid_one, grid_two):
    """
    Minimum entropy fusion: for each voxel, select the value from the view with lower entropy
    
    Args:
        grid1: First occupancy grid (64x64x64) with probability logits
        grid2: Second occupancy grid (64x64x64) with probability logits
    
    Returns:
        fused_grid: Grid with values from the view with minimum entropy per voxel
    """
    print("Performing minimum entropy fusion...")
    
    # Convert logits to probabilities
    prob1 = 1.0 / (1.0 + np.exp(-grid_one))  # sigmoid
    prob2 = 1.0 / (1.0 + np.exp(-grid_two))  # sigmoid
    
    # Clip probabilities to avoid log(0)
    prob1_clipped = np.clip(prob1, 1e-7, 1-1e-7)
    prob2_clipped = np.clip(prob2, 1e-7, 1-1e-7)
    
    # Calculate entropy: H = -p*log(p) - (1-p)*log(1-p)
    entropy1 = -(prob1_clipped * np.log(prob1_clipped) + (1-prob1_clipped) * np.log(1-prob1_clipped))
    entropy2 = -(prob2_clipped * np.log(prob2_clipped) + (1-prob2_clipped) * np.log(1-prob2_clipped))
    
    # Select grid1 values where entropy1 < entropy2, otherwise grid2
    use_grid1 = entropy1 < entropy2
    fused_grid = np.where(use_grid1, grid_one, grid_two)
    
    # Print some statistics
    total_voxels = grid_one.size
    grid1_selected = np.sum(use_grid1)
    grid2_selected = total_voxels - grid1_selected
    
    print(f"  Selected from view 1: {grid1_selected:,} voxels ({100*grid1_selected/total_voxels:.1f}%)")
    print(f"  Selected from view 2: {grid2_selected:,} voxels ({100*grid2_selected/total_voxels:.1f}%)")
    print(f"  Average entropy - View 1: {entropy1.mean():.4f}, View 2: {entropy2.mean():.4f}")
    print(f"  Selected entropy range: [{entropy1[use_grid1].mean():.4f}, {entropy2[~use_grid1].mean():.4f}]")
    
    return fused_grid


def main():
    parser = argparse.ArgumentParser(description="Align with ICP")
    parser.add_argument("--voxel_one", type=str, required=True, help="Path to view one PLY file")
    parser.add_argument("--voxel_two", type=str, required=True, help="Path to view two PLY file")
    parser.add_argument("--voxel_one_npz", type=str, required=True, help="Path to view one voxel npz file")
    parser.add_argument("--voxel_two_npz", type=str, required=True, help="Path to view two voxel npz file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save aligned point clouds")
    args = parser.parse_args() 

    # For debugging, just visualize the voxel with the VGGT predictions
    # Save them in one file, use different colors to denote points
    voxel_one, voxel_two_aligned, transformation = icp_alignment(args.voxel_one, args.voxel_two, args.output_dir)

    # Fusion
    # Extract raw occupancy grids from npz files
    voxel_one_data = np.load(args.voxel_one_npz)
    voxel_two_data = np.load(args.voxel_two_npz)
    occupancy_grid_one = voxel_one_data['occupancy_grid']
    occupancy_grid_two = voxel_two_data['occupancy_grid']

    occupancy_grid_two_interpolated = trilinear_interpolate_grid(occupancy_grid_two, transformation)

    # Visualize the two voxel grids after interpolation
    # Grid values are logits
    active_voxels_coords = np.argwhere(occupancy_grid_two_interpolated > 0.0)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(active_voxels_coords)
    o3d.io.write_point_cloud(f"{args.output_dir}/voxel_two_interpolated.ply", pc)
    print(f"Saved interpolated voxel two point cloud to {args.output_dir}/voxel_two_interpolated.ply")
    
    # Fuse by averaging probabilities
    fused_occupancy_grid = fuse_by_average(occupancy_grid_one, occupancy_grid_two_interpolated)
    # Save fused point clouds
    fused_active_voxels_coords = np.argwhere(fused_occupancy_grid > 0.0)
    pc_fused = o3d.geometry.PointCloud()
    pc_fused.points = o3d.utility.Vector3dVector(fused_active_voxels_coords)
    o3d.io.write_point_cloud(f"{args.output_dir}/fused_average_voxels.ply", pc_fused)
    print(f"Saved fused voxel point cloud to {args.output_dir}/fused_average_voxels.ply")

    # Fuse by minimum entropy
    fused_occupancy_grid_min_entropy = fuse_by_min_entropy(occupancy_grid_one, occupancy_grid_two_interpolated)
    # Save fused point clouds
    fused_active_voxels_coords_min_entropy = np.argwhere(fused_occupancy_grid_min_entropy > 0.0)
    pc_fused_min_entropy = o3d.geometry.PointCloud()
    pc_fused_min_entropy.points = o3d.utility.Vector3dVector(fused_active_voxels_coords_min_entropy)
    o3d.io.write_point_cloud(f"{args.output_dir}/fused_min_entropy_voxels.ply", pc_fused_min_entropy)
    print(f"Saved fused (min entropy) voxel point cloud to {args.output_dir}/fused_min_entropy_voxels.ply")

if __name__ == "__main__":
    main()