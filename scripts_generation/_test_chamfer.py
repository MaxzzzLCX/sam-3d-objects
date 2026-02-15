from chamfer_distance_evaluation import chamfer_distance_evaluation
from chamfer_distance_prep import sample_points_from_mesh
import open3d as o3d

def main():
    gt_mesh_path = "/scratch/cl927/datasets/helper/mesh_gt.ply"
    pred_mesh_path = "/scratch/cl927/datasets/helper/002_mesh_new_double_rotated.ply"

    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    pred_mesh = o3d.io.read_triangle_mesh(pred_mesh_path)

    pred_pc = sample_points_from_mesh(pred_mesh, num_points = 10000)
    # Sample the same number of points from the GT mesh as the predicted point cloud
    gt_pc = sample_points_from_mesh(gt_mesh, num_points = 10000)

    chamfer_results, best_initial_rotation = chamfer_distance_evaluation(
        gt_pc, pred_pc, output_dir="/scripts_generation", debug=False
    )

    print("Chamfer Distance Results:", chamfer_results)
    print("Best Initial Rotation:\n", best_initial_rotation)

if __name__ == "__main__":
    main()