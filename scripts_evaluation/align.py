import numpy as np
import trimesh
import open3d as o3d
from skimage import measure
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import argparse
import os

from chamfer_distance_evaluation import calculate_chamfer_distance


def save_pre_alignment_combined_point_cloud(gt_path, pred_path, output_dir):
    """
    Save a combined point cloud visualizing pre-ICP alignment of ground truth and predicted point clouds.
    
    Args:
        gt_path: Path to ground truth PLY file
        pred_path: Path to prediction PLY file
        output_dir: Directory to save the combined point cloud
    """
    # For debugging, just visualize the voxel with the VGGT predictions
    # Save them in one file, use different colors to denote points
    pred_pc = o3d.io.read_point_cloud(pred_path)
    gt_pc = o3d.io.read_point_cloud(gt_path)

    pre_alignment_path = f"{output_dir}/pre_icp_combined.ply"
    os.makedirs(os.path.dirname(pre_alignment_path), exist_ok=True)
    
    # Combine points from both point clouds
    pred_points = np.asarray(pred_pc.points)
    gt_points = np.asarray(gt_pc.points)
    combined_points = np.concatenate([pred_points, gt_points], axis=0)
    
    # Create different colors for each point cloud
    # Red for predicted points, Blue for ground truth points
    pred_colors = np.tile([1.0, 0.0, 0.0], (len(pred_points), 1))  # Red
    gt_colors = np.tile([0.0, 0.0, 1.0], (len(gt_points), 1))      # Blue
    combined_colors = np.concatenate([pred_colors, gt_colors], axis=0)
    
    # Create combined point cloud
    combined_pc = o3d.geometry.PointCloud()
    combined_pc.points = o3d.utility.Vector3dVector(combined_points)
    combined_pc.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # Save the combined point cloud
    o3d.io.write_point_cloud(pre_alignment_path, combined_pc)

    print(f"Saved pre-ICP combined point cloud to {pre_alignment_path}")

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

def save_coarse_alignment(gt_path, pred_path, output_dir):
    """
    Coarse alignment using SAM3D layout prediction
    """

    # Coarse alignment step
    data_path = pred_path.replace('_voxels.ply', '.npz')
    sam3d_data = np.load(data_path)
    scale = sam3d_data['scale']
    shift = sam3d_data['shift']
    rotation_6d = sam3d_data['rotation']

    # Covnert 6d rotation to rotation matrix
    def rotation_6d_to_matrix(rots_6d):
        a1 = rots_6d[:, 0:3]
        a2 = rots_6d[:, 3:6]

        b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
        b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
        b3 = np.cross(b1, b2)

        rot_mats = np.stack([b1, b2, b3], axis=-1)  # Shape (N, 3, 3)
        return rot_mats
    
    rotation_matrix = rotation_6d_to_matrix(rotation_6d[np.newaxis, :])[0]
    print(f"Rotation matrix:\n{rotation_matrix}")

    print(f"Scale: {scale}, Shift: {shift}")

    # For debugging, just visualize the voxel with the VGGT predictions
    # Save them in one file, use different colors to denote points
    pred_pc = o3d.io.read_point_cloud(pred_path)
    gt_pc = o3d.io.read_point_cloud(gt_path)

    coarse_alignment_path = f"{output_dir}/coarse_alignment.ply"
    os.makedirs(os.path.dirname(coarse_alignment_path), exist_ok=True)
    
    # Combine points from both point clouds
    pred_points = np.asarray(pred_pc.points)
    pred_transformed = pred_points @ rotation_matrix # Apply rotation
    # pred_transformed = (rotation_matrix @ pred_points.T).T  # Apply rotation
    # pred_transformed = pred_points # DEBUG


    gt_points = np.asarray(gt_pc.points)
    gt_normalized = normalize_point_cloud(gt_points)
    combined_points = np.concatenate([pred_transformed, gt_normalized], axis=0)


    # Check the scale of two point clouds
    print(f"Voxel point clouds scale: Pred min {pred_transformed.min(axis=0)}, max {pred_transformed.max(axis=0)}")
    print(f"Length in each axis: Pred {pred_transformed.max(axis=0) - pred_transformed.min(axis=0)}")
    print(f"Voxel point clouds scale: GT min {gt_normalized.min(axis=0)}, max {gt_normalized.max(axis=0)}")
    print(f"Length in each axis: GT {gt_normalized.max(axis=0) - gt_normalized.min(axis=0)}")

    # Create different colors for each point cloud
    # Red for predicted points, Blue for ground truth points
    pred_colors = np.tile([1.0, 0.0, 0.0], (len(pred_transformed), 1))  # Red
    gt_colors = np.tile([0.0, 0.0, 1.0], (len(gt_points), 1))      # Blue
    combined_colors = np.concatenate([pred_colors, gt_colors], axis=0)
    
    # Create combined point cloud
    combined_pc = o3d.geometry.PointCloud()
    combined_pc.points = o3d.utility.Vector3dVector(combined_points)
    combined_pc.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # Save the combined point cloud
    o3d.io.write_point_cloud(coarse_alignment_path, combined_pc)

    print(f"Saved coarse-aligned combined point cloud to {coarse_alignment_path}")

    return gt_normalized, pred_transformed

def icp_alignment(gt_points, pred_points, output_dir, threshold=0.15, max_iterations=100):
    """
    Perform ICP alignment of predicted points to ground truth points.
    """

    # Randomly sample the same number of points as pred points from gt_points
    print("Number of GT points:", len(gt_points))
    print("Number of Pred points:", len(pred_points))
    if len(gt_points) > len(pred_points):
        indices = np.random.choice(len(gt_points), size=len(pred_points), replace=False)
        gt_points = gt_points[indices]
        print(f"Sampled {len(pred_points)} points from GT for ICP alignment.")

    # Before ICP, calculate Chamfer distance
    chamfer_results = calculate_chamfer_distance(gt_points, pred_points)
    print(f"Chamfer distance before ICP: {chamfer_results['bidirectional_chamfer']}")


    # Convert np.ndarrays to Open3D point clouds
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
    
    # Perform ICP alignment
    result = o3d.pipelines.registration.registration_icp(
        pred_pcd, gt_pcd, max_correspondence_distance=threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    transformation = result.transformation
    pred_pcd.transform(transformation)
    aligned_pred_points = np.asarray(pred_pcd.points)

    # After ICP, calculate Chamfer distance
    chamfer_results = calculate_chamfer_distance(gt_points, aligned_pred_points)
    print(f"Chamfer distance after ICP: {chamfer_results['bidirectional_chamfer']}")

    # Save as combined point cloud
    icp_alignment_path = f"{output_dir}/icp_aligned_combined.ply"
    os.makedirs(os.path.dirname(icp_alignment_path), exist_ok=True)
    combined_points = np.concatenate([aligned_pred_points, gt_points], axis=0)
    pred_colors = np.tile([1.0, 0.0, 0.0], (len(aligned_pred_points), 1))  # Red
    gt_colors = np.tile([0.0, 0.0, 1.0], (len(gt_points), 1))      # Blue
    combined_colors = np.concatenate([pred_colors, gt_colors], axis=0)
    combined_pc = o3d.geometry.PointCloud()
    combined_pc.points = o3d.utility.Vector3dVector(combined_points)
    combined_pc.colors = o3d.utility.Vector3dVector(combined_colors)
    o3d.io.write_point_cloud(icp_alignment_path, combined_pc)
    print(f"Saved ICP-aligned combined point cloud to {icp_alignment_path}")

    return gt_points, aligned_pred_points

def main():
    parser = argparse.ArgumentParser(description="Align with ICP")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground truth PLY file")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted PLY file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save aligned point clouds")
    args = parser.parse_args() 

    # For debugging, just visualize the voxel with the VGGT predictions
    # Save them in one file, use different colors to denote points
    save_pre_alignment_combined_point_cloud(args.gt_path, args.pred_path, args.output_dir)

    gt_normalized, pred_transformed = save_coarse_alignment(args.gt_path, args.pred_path, args.output_dir)

    gt_aligned_points, pred_aligned_points = icp_alignment(gt_normalized, pred_transformed, args.output_dir)


    


if __name__ == "__main__":
    main()