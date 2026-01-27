
import numpy as np
import trimesh

def align_multiview_voxels(view1_voxels, view2_voxels, view1_scale, view1_shift, 
                          view2_scale, view2_shift, pose1, pose2, output_path="aligned_voxels.ply"):
    """
    Takes in two sets of normalized occupancy grid voxels from two different views,
    apply the predicted camera poses from VGGT to align them in a common coordinate system,
    and save the aligned point cloud to a .ply file.
    
    Args:
        view1_voxels: (N1, 3) voxel coordinates from view 1 [0, 63] grid indices
        view2_voxels: (N2, 3) voxel coordinates from view 2 [0, 63] grid indices  
        view1_scale: (3,) SAM3D predicted scale for view 1
        view1_shift: (3,) SAM3D predicted translation for view 1
        view2_scale: (3,) SAM3D predicted scale for view 2
        view2_shift: (3,) SAM3D predicted translation for view 2
        pose1: (3, 4) VGGT camera extrinsic matrix for view 1 [R|t] format (world to camera)
        pose2: (3, 4) VGGT camera extrinsic matrix for view 2 [R|t] format (world to camera)
        output_path: where to save the aligned point cloud
    """
    
    # Step 1: Convert grid indices [0, 63] to normalized coordinates [-0.5, 0.5]
    view1_normalized = (view1_voxels / 63.0) - 0.5  # (N1, 3)
    view2_normalized = (view2_voxels / 63.0) - 0.5  # (N2, 3)
    
    # Step 2: Apply SAM3D scale and shift to get object coordinates in camera frame
    view1_camera = view1_normalized * view1_scale + view1_shift  # (N1, 3)
    view2_camera = view2_normalized * view2_scale + view2_shift  # (N2, 3)
    
    # Step 3: Transform from camera coordinates to world coordinates
    # VGGT extrinsic is (3, 4) in [R|t] format, need to convert to (4, 4) homogeneous matrix
    def to_homogeneous_matrix(extrinsic_3x4):
        """Convert (3, 4) [R|t] matrix to (4, 4) homogeneous transformation matrix"""
        homogeneous = np.eye(4)
        homogeneous[:3, :] = extrinsic_3x4
        return homogeneous
    
    pose1_4x4 = to_homogeneous_matrix(pose1)  # Convert to 4x4
    pose2_4x4 = to_homogeneous_matrix(pose2)  # Convert to 4x4
    
    # pose is world-to-camera transform (camera from world), so we need camera-to-world inverse
    pose1_inv = np.linalg.inv(pose1_4x4)  # camera to world
    pose2_inv = np.linalg.inv(pose2_4x4)  # camera to world
    
    # Convert to homogeneous coordinates
    view1_homo = np.hstack([view1_camera, np.ones((len(view1_camera), 1))])  # (N1, 4)
    view2_homo = np.hstack([view2_camera, np.ones((len(view2_camera), 1))])  # (N2, 4)
    
    # Transform to world coordinates
    view1_world = (pose1_inv @ view1_homo.T).T[:, :3]  # (N1, 3)
    view2_world = (pose2_inv @ view2_homo.T).T[:, :3]  # (N2, 3)
    
    # Step 4: Combine both views in world coordinate system
    combined_points = np.vstack([view1_world, view2_world])  # (N1+N2, 3)
    
    # Create colors to distinguish the two views (red for view1, blue for view2)
    colors1 = np.tile([255, 0, 0], (len(view1_world), 1))    # Red
    colors2 = np.tile([0, 0, 255], (len(view2_world), 1))    # Blue
    combined_colors = np.vstack([colors1, colors2]).astype(np.uint8)
    
    # Step 5: Save combined point cloud
    pc = trimesh.PointCloud(combined_points, colors=combined_colors)
    pc.export(output_path)
    print(f"Saved {len(combined_points)} aligned points to {output_path}")
    print(f"  View 1: {len(view1_world)} points (red)")
    print(f"  View 2: {len(view2_world)} points (blue)")
    
    return combined_points, combined_colors


def main():
    # Load VGGT camera poses
    vggt_data = np.load("/scratch/cl927/scenes_sam3dvggt/apple_new/vggt_camera_poses.npz")
    extrinsic = vggt_data['extrinsic']  # (N, 4, 4)
    intrinsic = vggt_data['intrinsic']  # (N, 3, 3)
    image_names = vggt_data['image_names']
    
    print("Available images:", image_names)
    
    # Load SAM3D outputs for two views
    sam3d_dir = "sam3d_outputs"
    view1_data = np.load(f"{sam3d_dir}/view_01.npz")
    view2_data = np.load(f"{sam3d_dir}/view_02.npz")
    
    # Extract data
    view1_coords = view1_data['coords']
    view1_scale = view1_data['scale'] 
    view1_shift = view1_data['shift']
    
    view2_coords = view2_data['coords']
    view2_scale = view2_data['scale']
    view2_shift = view2_data['shift']
    
    print(f"View 1: {len(view1_coords)} voxels, scale: {view1_scale}, shift: {view1_shift}")
    print(f"View 2: {len(view2_coords)} voxels, scale: {view2_scale}, shift: {view2_shift}")
    
    # Align using VGGT poses (assuming resized_01.png is index 0, resized_02.png is index 1)
    align_multiview_voxels(
        view1_voxels=view1_coords,
        view2_voxels=view2_coords,
        view1_scale=view1_scale,
        view1_shift=view1_shift,
        view2_scale=view2_scale,
        view2_shift=view2_shift,
        pose1=extrinsic[0],  # First image pose
        pose2=extrinsic[1],  # Second image pose  
        output_path="aligned_apple_views.ply"
    )

if __name__ == "__main__":
    main()