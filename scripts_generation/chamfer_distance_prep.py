import open3d as o3d
import numpy as np
import argparse
import trimesh

def sample_points_from_mesh(mesh, num_points=10000):
    """Sample points uniformly from a mesh surface."""
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    # Rotation the sampled mesh point clouds to align with expected coordinate system of SAM3D
    # rotation_matrix = np.array([
    #     [1,  0,  0],
    #     [0,  0,  -1],
    #     [0,  1,  0]
    # ], dtype=float)
    # pcd.rotate(rotation_matrix, center=(0, 0, 0))
    
    return np.asarray(pcd.points)

def count_sample_points(point_cloud):
    """Count number of points in the sampled voxel point cloud."""
    return point_cloud.shape[0]

def main():
    print(f"Code Start")

    parser = argparse.ArgumentParser(description="Calculate Chamfer distance between ground truth and predicted point clouds with optional ICP alignment.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground truth mesh file")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted point cloud file")
    parser.add_argument("--output_dir", type=str, required=True, help="folder to save output point clouds")
    args = parser.parse_args()
    
    # Count the number of active voxels
    # The pred_path is a .ply not a mesh
    pred_mesh = o3d.io.read_point_cloud(args.pred_path)
    pred_points = np.asarray(pred_mesh.points)
    num_points = count_sample_points(pred_points)
    print(f"{args.pred_path} has {num_points} points")
    


    gt_mesh = o3d.io.read_triangle_mesh(args.gt_path)
    gt_points = sample_points_from_mesh(gt_mesh, num_points=num_points)
    np.save(f"{args.output_dir}/gt_points.npy", gt_points)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(gt_points)
    o3d.io.write_point_cloud(f"{args.output_dir}/gt_points.ply", pc)
    print(f"Saved ground truth points to {args.output_dir}/gt_points.npy and {args.output_dir}/gt_points.ply")

if __name__ == "__main__":
    main()