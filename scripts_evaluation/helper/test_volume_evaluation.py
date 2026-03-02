"""
In doing volume evaluation, I noticed that voxelization gives a valid volume, but introduces large errors.
"""
import trimesh
import matplotlib.pyplot as plt
import os
import open3d as o3d
import numpy as np

def test_volume_evaluation_voxels():
    gt_mesh_path = "/scratch/cl927/datasets/Toys4k/debug/TEST_VOLUME_6/mesh.ply"
    gt_volume = trimesh.load(gt_mesh_path).volume
    print(f"Ground Truth Volume: {gt_volume:.6f}")
    
    resolutions = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    results = []

    # Voxelize and fill the gt mesh. See what volume that is
    for voxel_resolutions in resolutions:
        print(f"Testing voxelization with resolution {voxel_resolutions}...")
        gt_mesh = trimesh.load(gt_mesh_path)
        voxelized_mesh = gt_mesh.voxelized(pitch=gt_mesh.extents.max() / voxel_resolutions).fill()
        voxelized_volume = voxelized_mesh.volume
        volume_error = abs(voxelized_volume - gt_volume)
        volume_error_percentage = (volume_error / gt_volume) * 100 if gt_volume > 0 else float('inf')
        results.append(volume_error_percentage)

        # Save the voxelized mesh for inspection
        save_path = gt_mesh_path.replace(".ply", f"_voxelized_{voxel_resolutions}.ply")
        voxelized_mesh.as_boxes().export(save_path)

        print(f"Voxelized Mesh Volume: {voxelized_volume:.6f}")
        print(f"Volume Error: {volume_error:.6f}")
        print(f"Volume Error Percentage: {volume_error_percentage:.2f}%")
        print()
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(resolutions, results, marker='o')
    plt.title("Volume Error Percentage vs Voxel Resolution")
    plt.xlabel("Voxel Resolution (number of voxels along longest axis)")
    plt.ylabel("Volume Error Percentage (%)")
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(gt_mesh_path), "volume_error_vs_voxel_resolution.png"))

def test_volume_evaluation_alpha():
    gt_mesh_path = "/scratch/cl927/datasets/Toys4k/debug/TEST_VOLUME_6/mesh.ply"
    gt_volume = trimesh.load(gt_mesh_path).volume
    print(f"Ground Truth Volume: {gt_volume:.6f}")
    
    ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    # ratios = [0.01]
    results = []

    # Use alpha shape to fill the interior
    for ratio in ratios:
        
        gt_mesh = trimesh.load(gt_mesh_path)
        alpha = gt_mesh.extents.max() * ratio
        print(f"Testing alpha shape with alpha={alpha}...")

        # Convert to Open3D point cloud
        points = gt_mesh.sample(10000)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        print(f"Mesh has {len(pcd.points)} points")
        
        # Estimate normals for better reconstruction
        pcd.estimate_normals()
        
        # Create alpha shape
        alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        
        # Convert back to trimesh
        vertices = np.asarray(alpha_mesh.vertices)
        faces = np.asarray(alpha_mesh.triangles)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        print(f"Using Open3D alpha shape with alpha={alpha:.6f}")

        # Save the alpha shape mesh for inspection
        save_path = gt_mesh_path.replace(".ply", f"_alpha_{alpha:.2f}.ply")
        mesh.export(save_path)

        # Compute volume error
        mesh_volume = mesh.volume
        volume_error = abs(mesh_volume - gt_volume)
        volume_error_percentage = (volume_error / gt_volume) * 100 if gt_volume > 0 else float('inf')
        results.append(volume_error_percentage)
        print(f"Alpha Shape Mesh Volume: {mesh_volume:.6f}")
        print(f"Ground Truth Volume: {gt_volume:.6f}")
        print(f"Volume Error: {volume_error:.6f}")
        print(f"Volume Error Percentage: {volume_error_percentage:.2f}%")
        print()
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, results, marker='o')
    plt.title("Volume Error Percentage vs Alpha Shape Parameter")
    plt.xlabel("Alpha Shape Parameter (\% of span)")
    plt.ylabel("Volume Error Percentage (%)")
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(gt_mesh_path), "volume_error_vs_alpha_shape.png"))

if __name__ == "__main__":
    test_volume_evaluation_voxels()
    # test_volume_evaluation_alpha()