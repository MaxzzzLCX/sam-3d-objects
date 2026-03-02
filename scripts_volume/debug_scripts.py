
import trimesh
import open3d as o3d


def scale_of_objects():

    # Measure the span of the object in all three directions
    file_path = "/scratch/cl927/sam-3d-objects/scripts_volume/online_playground_demo/object_1.ply"

    mesh = trimesh.load(file_path)
    span = mesh.bounds[1] - mesh.bounds[0]
    print(f"Span in each dimension: {span}")

def test_surface_reconstruction_of_gs(method="poisson"):
    """
    I am curious whether surface reconstruction can directly work on GS centers (treat as point clouds)
    """
    file_path = "/scratch/cl927/sam-3d-objects/scripts_volume/online_playground_demo/object_0.ply"

    # Load the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(file_path)
    print(pcd)
    
    # DIAGNOSTIC: Check point cloud validity
    import numpy as np
    points = np.asarray(pcd.points)
    print(f"Point cloud shape: {points.shape}")
    print(f"Point cloud bounds: min={points.min(axis=0)}, max={points.max(axis=0)}")
    print(f"Has NaN: {np.isnan(points).any()}")
    print(f"Has Inf: {np.isinf(points).any()}")

    if method == "poisson":
        # No normal, have to estimate normal
        # if not pcd.has_normals():
        print("Estimating normals for the point cloud...")
        # Increase search radius - GS centers might be sparse
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        
        # DIAGNOSTIC: Check normals
        normals = np.asarray(pcd.normals)
        print(f"Normals shape: {normals.shape}")
        print(f"Normals have NaN: {np.isnan(normals).any()}")
        print(f"Normals have Inf: {np.isinf(normals).any()}")
        print(f"Zero-length normals: {(np.linalg.norm(normals, axis=1) < 1e-6).sum()}")
        
        # Orient normals consistently (important for Poisson)
        pcd.orient_normals_consistent_tangent_plane(k=15)


        # Perform surface reconstruction using Poisson reconstruction
        print("Performing Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        print(mesh)
        print(f"Is mesh watertight? {mesh.is_watertight()}")

        # Save the reconstructed mesh
        output_mesh_path = "/scratch/cl927/sam-3d-objects/scripts_volume/online_playground_demo/psr_object_0.ply"
        o3d.io.write_triangle_mesh(output_mesh_path, mesh)
        print(f"Reconstructed mesh saved to {output_mesh_path}")

    elif method == "alpha":
        print("Performing Alpha Shape surface reconstruction...")
        alpha = 0.1  # Adjust alpha as needed
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        print(mesh)
        print(f"Is mesh watertight? {mesh.is_watertight()}")

        # Save the reconstructed mesh
        output_mesh_path = "/scratch/cl927/sam-3d-objects/scripts_volume/online_playground_demo/alpha_object_0.ply"
        o3d.io.write_triangle_mesh(output_mesh_path, mesh)
        print(f"Reconstructed mesh saved to {output_mesh_path}")

def main():
    # scale_of_objects()
    test_surface_reconstruction_of_gs(method="alpha")

if __name__ == "__main__":
    main()

    