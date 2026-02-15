"""
For checking multiview consistency of SAM-3D-Objects. 
"""
import os
import sys
import numpy as np
import trimesh
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Import SAM3D inference code
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "..", "notebook")
sys.path.append(parent_dir)
from inference import Inference, load_image, load_mask


def apply_transformation(voxels, scale, rotation_6d, translation, translation_scale, apply_translation=False):
    """
    Apply SAM3D layout transformations to mesh vertices
    
    Args:
        voxels: point cloud (trimesh.PointCloud) or numpy array
        scale: np.array of shape (3,), scale factors
        rotation_6d: np.array of shape (6,), 6D rotation representation
        translation: np.array of shape (3,), translation vector
        translation_scale: np.array of shape (3,), translation scale vector
        apply_translation: bool, whether to apply translation (usually False for projection)
    Returns:
        transformed_vertices: np.array of shape (N, 3)
    """
    # Convert 6D rotation to rotation matrix
    # SAM3D uses 6D rotation representation (2 orthonormal vectors)
    rotation_matrix = rotation_6d_to_matrix(rotation_6d)
    
    # Get vertices from point cloud or mesh
    if hasattr(voxels, 'vertices'):
        vertices = voxels.vertices.copy()
    else:
        vertices = voxels.copy()
    
    # Apply transformations: vertices_transformed = R * (S * vertices) + T
    # 1. Scale
    vertices = vertices * scale
    
    # 2. Rotate
    vertices = vertices @ rotation_matrix.T
    
    # 3. Translate (usually skip this for projection - it's in world space, not camera space)
    if apply_translation:
        vertices = vertices + translation * translation_scale
    
    return vertices


def rotation_6d_to_matrix(rotation_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix
    SAM3D uses 6D rotation (Gram-Schmidt orthogonalization)
    
    Args:
        rotation_6d: np.array of shape (6,), representing two 3D vectors
    
    Returns:
        rotation_matrix: np.array of shape (3, 3)
    """
    # First two columns of rotation matrix
    x = rotation_6d[:3]
    y = rotation_6d[3:]
    
    # Gram-Schmidt orthogonalization
    x = x / np.linalg.norm(x)
    y = y - np.dot(y, x) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    
    rotation_matrix = np.stack([x, y, z], axis=1)
    return rotation_matrix


def project_to_image(vertices_3d, image_size, fov=40):
    """
    Project 3D vertices to 2D image coordinates using perspective projection
    
    Args:
        vertices_3d: np.array of shape (N, 3), 3D vertices
        image_size: tuple (width, height)
        fov: field of view in degrees
    
    Returns:
        vertices_2d: np.array of shape (N, 2), 2D pixel coordinates
        valid_mask: np.array of shape (N,), boolean mask for visible points (z > 0)
    """
    width, height = image_size
    
    # Simple perspective projection with camera intrinsics
    # Assuming camera at origin looking down -Z axis
    focal_length = width / (2 * np.tan(np.radians(fov) / 2))
    cx, cy = width / 2, height / 2
    
    # Camera intrinsic matrix
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    
    # Filter points behind camera (z <= 0)
    valid_mask = vertices_3d[:, 2] > 0
    
    # Project to 2D: [u, v, 1]^T = K @ [x/z, y/z, 1]^T
    vertices_2d = np.zeros((len(vertices_3d), 2))
    
    if np.any(valid_mask):
        valid_vertices = vertices_3d[valid_mask]
        # Normalize by depth
        normalized = valid_vertices[:, :2] / valid_vertices[:, 2:3]
        # Apply intrinsics
        homogeneous = np.concatenate([normalized, np.ones((len(normalized), 1))], axis=1)
        projected = (K @ homogeneous.T).T
        vertices_2d[valid_mask] = projected[:, :2]
    
    return vertices_2d, valid_mask

def visualize_voxel_projection(image, voxels, vertices_2d, valid_mask):
    """
    Visualize voxel projection overlay on original image
    
    Args:
        image: PIL Image
        voxels: np.array of shape (N, 3), voxel centers
        vertices_2d: np.array of shape (N, 2), projected 2D coordinates
        valid_mask: boolean array indicating visible voxels
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Image with projection overlay
    axes[1].imshow(image)
    
    # Draw visible voxels
    visible_vertices = vertices_2d[valid_mask]
    axes[1].scatter(visible_vertices[:, 0], visible_vertices[:, 1], 
                   c='red', s=5, alpha=0.5)
    
    axes[1].set_title("Image with Voxel Projection Overlay")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("scripts_volume/voxel_projection_visualization.png")

def visualize_mesh_projection(image, mesh, vertices_2d, valid_mask):
    """
    Visualize mesh projection overlay on original image
    
    Args:
        image: PIL Image
        mesh: trimesh.Trimesh
        vertices_2d: np.array of shape (N, 2), projected 2D coordinates
        valid_mask: boolean array indicating visible vertices
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Image with projection overlay
    axes[1].imshow(image)
    
    # Draw mesh edges
    for edge in mesh.edges:
        v1_idx, v2_idx = edge
        # Only draw if both vertices are visible
        if valid_mask[v1_idx] and valid_mask[v2_idx]:
            v1_2d = vertices_2d[v1_idx]
            v2_2d = vertices_2d[v2_idx]
            axes[1].plot([v1_2d[0], v2_2d[0]], [v1_2d[1], v2_2d[1]], 
                        'g-', linewidth=0.5, alpha=0.6)
    
    # Draw visible vertices
    visible_vertices = vertices_2d[valid_mask]
    axes[1].scatter(visible_vertices[:, 0], visible_vertices[:, 1], 
                   c='red', s=1, alpha=0.5)
    
    axes[1].set_title("Image with Mesh Projection Overlay")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("scripts_volume/mesh_projection_visualization.png")


def project_object(image, voxels, scale, rotation_6d, translation, translation_scale, fov=40, apply_transform=False):
    """
    Projects a 3D mesh back to the image plane using predicted layouts
    
    Args:
        image: PIL Image, the original input image
        voxels: point cloud
        scale: np.array of shape (3,), the predicted scale in x, y, z dimensions
        rotation_6d: np.array of shape (6,), 6D rotation representation from SAM3D
        translation: np.array of shape (3,), the predicted translation in x, y, z dimensions
        translation_scale: np.array of shape (3,), translation scale vector from SAM3D
        fov: float, field of view for projection in degrees
        apply_transform: bool, whether to apply SAM3D layout transforms (usually False for visualization)
    """
    # Get vertices
    if hasattr(voxels, 'vertices'):
        vertices_3d = voxels.vertices.copy()
    else:
        vertices_3d = voxels.copy()
    
    print(f"\n=== Debugging Projection ===")
    print(f"Original voxels (canonical space): X[{vertices_3d[:, 0].min():.3f}, {vertices_3d[:, 0].max():.3f}], "
          f"Y[{vertices_3d[:, 1].min():.3f}, {vertices_3d[:, 1].max():.3f}], "
          f"Z[{vertices_3d[:, 2].min():.3f}, {vertices_3d[:, 2].max():.3f}]")
    
    if apply_transform:
        print(f"\nApplying SAM3D layout transforms in OpenGL space:")
        print(f"  Scale: {scale}")
        print(f"  Rotation (6D): {rotation_6d}")
        print(f"  Translation: {translation}")
        print(f"  Translation scale: {translation_scale}")
        
        # Apply SAM3D transforms in OpenGL space (their native space)
        # Scale
        vertices_gl = vertices_3d * scale
        print(f"After scale (OpenGL): X[{vertices_gl[:, 0].min():.3f}, {vertices_gl[:, 0].max():.3f}], "
              f"Y[{vertices_gl[:, 1].min():.3f}, {vertices_gl[:, 1].max():.3f}], "
              f"Z[{vertices_gl[:, 2].min():.3f}, {vertices_gl[:, 2].max():.3f}]")
        
        # Rotation
        R_gl = rotation_6d_to_matrix(rotation_6d)
        vertices_gl = vertices_gl @ R_gl.T
        print(f"After rotation (OpenGL): X[{vertices_gl[:, 0].min():.3f}, {vertices_gl[:, 0].max():.3f}], "
              f"Y[{vertices_gl[:, 1].min():.3f}, {vertices_gl[:, 1].max():.3f}], "
              f"Z[{vertices_gl[:, 2].min():.3f}, {vertices_gl[:, 2].max():.3f}]")
        
        # Translation
        vertices_gl = vertices_gl + translation * translation_scale
        print(f"After translation (OpenGL): X[{vertices_gl[:, 0].min():.3f}, {vertices_gl[:, 0].max():.3f}], "
              f"Y[{vertices_gl[:, 1].min():.3f}, {vertices_gl[:, 1].max():.3f}], "
              f"Z[{vertices_gl[:, 2].min():.3f}, {vertices_gl[:, 2].max():.3f}]")
        
        # Now convert from OpenGL to OpenCV
        vertices_opencv = np.stack([
            vertices_gl[:, 0],   # X stays the same
            -vertices_gl[:, 1],  # Y = -Y
            -vertices_gl[:, 2],   # Z = -Z
        ], axis=1)
        print(f"After coordinate conversion (OpenGL->OpenCV): X[{vertices_opencv[:, 0].min():.3f}, {vertices_opencv[:, 0].max():.3f}], "
              f"Y[{vertices_opencv[:, 1].min():.3f}, {vertices_opencv[:, 1].max():.3f}], "
              f"Z[{vertices_opencv[:, 2].min():.3f}, {vertices_opencv[:, 2].max():.3f}]")
        
        transformed_vertices = vertices_opencv
    else:
        # No SAM3D transforms - just convert canonical voxels to camera space
        vertices_opencv = np.stack([
            vertices_3d[:, 0],   # X stays the same
            -vertices_3d[:, 2],  # Y = -Z
            vertices_3d[:, 1],   # Z = Y
        ], axis=1)
        print(f"After coordinate conversion (OpenGL->OpenCV): X[{vertices_opencv[:, 0].min():.3f}, {vertices_opencv[:, 0].max():.3f}], "
              f"Y[{vertices_opencv[:, 1].min():.3f}, {vertices_opencv[:, 1].max():.3f}], "
              f"Z[{vertices_opencv[:, 2].min():.3f}, {vertices_opencv[:, 2].max():.3f}]")
        
        # Place object at reasonable depth in front of camera
        vertices_opencv[:, 2] += 2.0  # Move 2 units away from camera
        print(f"Placed at camera depth: Z[{vertices_opencv[:, 2].min():.3f}, {vertices_opencv[:, 2].max():.3f}]")
        
        transformed_vertices = vertices_opencv
    
    # Step 2: Project to 2D
    image_size = image.size  # (width, height)
    print(f"Image size: {image_size}, FOV: {fov}")
    
    vertices_2d, valid_mask = project_to_image(transformed_vertices, image_size, fov)
    
    print(f"Visible vertices: {valid_mask.sum()} / {len(valid_mask)}")
    if valid_mask.sum() > 0:
        visible_2d = vertices_2d[valid_mask]
        print(f"2D projection range: X[{visible_2d[:, 0].min():.1f}, {visible_2d[:, 0].max():.1f}], "
              f"Y[{visible_2d[:, 1].min():.1f}, {visible_2d[:, 1].max():.1f}]")
    
    # Step 3: Visualize
    visualize_voxel_projection(image, voxels, vertices_2d, valid_mask)
    
    return vertices_2d, valid_mask

def load_layout_outputs(output_path):
    """
    Load SAM3D layout outputs from a saved file (e.g., .npz or .json)
    
    Args:
        output_path: str, path to the saved layout outputs
    """
    # For this example, we assume outputs are saved in .npz format
    data = np.load(output_path)
    print(f"Loaded data: {data.keys()}")
    print(f"")
    
    # Extract relevant parameters
    layout_outputs = {
        "scale": data["scale"],
        "rotation_6d": data["rotation"],
        "translation": data["translation"],
        "translation_scale": data["translation_scale"],
    }
    
    return layout_outputs


def main():
    # Example: Run SAM3D inference and visualize projection
    image_path = "/scratch/cl927/datasets/Toys4k_test/renders/_test_dataset_volume/_sam3d_room_scene/images/image.png"
    voxels_path = "/scratch/cl927/datasets/Toys4k_test/renders/_test_dataset_volume/_sam3d_room_scene/sam3d_singleview_predictions/image_voxels.ply"
    sam3d_output_path = "/scratch/cl927/datasets/Toys4k_test/renders/_test_dataset_volume/_sam3d_room_scene/sam3d_singleview_predictions/image_sam3d_outputs.npz"
    
    image = Image.open(image_path)
    voxels = trimesh.load(voxels_path)
    
    layout_outputs = load_layout_outputs(sam3d_output_path)
    
    # Extract layout parameters
    scale = layout_outputs["scale"]
    rotation_6d = layout_outputs["rotation_6d"]
    translation = layout_outputs["translation"]
    translation_scale = layout_outputs["translation_scale"]
    print(f"Scale: {scale}")
    print(f"Rotation (6D): {rotation_6d}")
    print(f"Translation: {translation}")
    print(f"Translation Scale: {translation_scale}")
    
    # Project back to image and visualize
    print("Projecting mesh to image...")
    project_object(image, voxels, scale, rotation_6d, translation, translation_scale, fov=40, apply_transform=True)

if __name__ == "__main__":
    main()