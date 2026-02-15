"""
Render a predicted 3D asset from camera poses stored in transforms.json
"""
import os
import json
import argparse
import numpy as np
from subprocess import DEVNULL, call
import trimesh

from evaluate_projection import evaluate_folder_iou, evaluate_folder_iou_multiview


BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'


def install_blender():
    """Install Blender if not already installed"""
    if not os.path.exists(BLENDER_PATH):
        print('Installing Blender...')
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
        print('Blender installed.')


def transform_matrix_to_spherical(transform_matrix):
    """
    Convert a 4x4 transform matrix to spherical coordinates (yaw, pitch, radius)
    
    Args:
        transform_matrix: 4x4 camera-to-world transform matrix
        
    Returns:
        dict with 'yaw', 'pitch', 'radius' keys
    """
    # Extract camera position from the transform matrix
    # The camera position is in the last column (translation part)
    cam_pos = np.array([
        transform_matrix[0][3],
        transform_matrix[1][3],
        transform_matrix[2][3]
    ])
    
    # Calculate radius (distance from origin)
    radius = np.linalg.norm(cam_pos)
    
    # Calculate pitch (elevation angle)
    # pitch = arcsin(z / radius)
    pitch = np.arcsin(cam_pos[2] / radius)
    
    # Calculate yaw (azimuthal angle)
    # yaw = arctan2(y, x)
    yaw = np.arctan2(cam_pos[1], cam_pos[0])
    
    return {
        'yaw': float(yaw),
        'pitch': float(pitch),
        'radius': float(radius)
    }


def render_mesh_from_transforms(mesh_path, transforms_json_path, output_dir, initial_transform=None, resolution=512):
    """
    Render a mesh using camera poses from transforms.json
    
    Args:
        mesh_path: Path to the 3D mesh file (.glb, .ply, .obj, etc.)
        transforms_json_path: Path to transforms.json with camera poses
        output_dir: Directory to save rendered images
        initial_transform: Initial transformation matrix to apply to the mesh
        resolution: Image resolution (default 512)
    """
    print(f"Rendering mesh {mesh_path}")
    # Load transforms.json
    with open(transforms_json_path, 'r') as f:
        transforms = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare camera data for Blender
    views = []
    for frame in transforms['frames']:
        # Convert transform matrix to spherical coordinates
        spherical = transform_matrix_to_spherical(frame['transform_matrix'])
        
        view = {
            'yaw': spherical['yaw'],
            'pitch': spherical['pitch'],
            'radius': spherical['radius'],
            'fov': frame['camera_angle_x']
        }
        views.append(view)
    
    # Call Blender to render
    blender_script = os.path.join(os.path.dirname(__file__), '..', '..', 'TRELLIS', 'dataset_toolkits', 'blender_script', 'render.py')
    print(f"blender script {blender_script}")

    # Check if the blender script exists, otherwise use a relative path
    if not os.path.exists(blender_script):
        blender_script = os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py')
    
    args = [
        BLENDER_PATH, '-b', '-P', blender_script,
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(mesh_path),
        '--resolution', str(resolution),
        '--output_folder', output_dir,
        '--engine', 'CYCLES',
    ]
    
    print(f'Rendering {len(views)} views...')
    result = call(args, stderr=DEVNULL, stdout=DEVNULL) # Suppress output for cleaner logs
    # result = call(args)  # Don't suppress output so we can see errors
    print(f'Blender exit code: {result}')
    print(f'Rendering complete. Output saved to {output_dir}')

def render_voxel_from_transforms(voxel_path, transforms_json_path, output_dir, resolution=512):
    """
    Render a voxel grid using camera poses from transforms.json
    
    Args:
        voxel_path: Path to the voxel grid file (.npy)
        transforms_json_path: Path to transforms.json with camera poses
        output_dir: Directory to save rendered images
        resolution: Image resolution (default 512)
    """
    # Load voxel grid
    voxels = np.load(voxel_path)
    
    # Load transforms.json
    with open(transforms_json_path, 'r') as f:
        transforms = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

def apply_initial_transformation_and_save(orig_mesh_path, initial_transformation):
    """
    Apply an initial transformation to the original mesh and save the transformed mesh as a new file
    This is to align the mesh in canonical space with the ground truth mesh
    """

    # If initial transformation is a 3*3 rotation, convert to 4*4
    if initial_transformation.shape == (3, 3):
        initial_transformation_4x4 = np.eye(4)
        initial_transformation_4x4[:3, :3] = initial_transformation
        initial_transformation = initial_transformation_4x4

    mesh = trimesh.load(orig_mesh_path)

    # Apply transformation
    # NOTE: trimesh.apply_transform(T) applies: new_vertices = (T @ vertices.T).T
    # For rotation R in upper-left 3x3: new_vertices = vertices @ R.T
    # This matches chamfer_distance_evaluation.py: pred_points @ rotation.T
    mesh.apply_transform(initial_transformation)
    
    # Detect actual file extension and create aligned filename
    import os
    base_path, ext = os.path.splitext(orig_mesh_path)
    transformed_mesh_path = f"{base_path}_aligned{ext}"
    
    mesh.export(transformed_mesh_path)
    print(f"DEBUG: Original path: {orig_mesh_path}")
    print(f"DEBUG: Transformed path: {transformed_mesh_path}")
    return transformed_mesh_path



def reproject_folder_and_evaluate_iou(generation_folder, transforms_json_path, rendering_output_folder, initial_transforms=None, evaluate_iou=False, multiview_inputs=None, resolution=512):
    """
    Render a folder of generated meshes or voxels using camera poses from transforms.json
    
    Args:
        generation_folder: Path to the folder containing generated meshes or voxels
        transforms_json_path: Path to transforms.json with camera poses
        initial_transforms: List of initial transformation matrices for each mesh (default None)
        resolution: Image resolution (default 512)
        evaluate_iou: Whether to evaluate IoU after rendering (default False)
        multiview_input: If None, each mesh is conditioned on single input image; otherwise, provide an array, where each element is an array of conditioned view indices
    """
    
    # meshes_pathes = sorted([f for f in os.listdir(generation_folder) if f.endswith('mesh.glb')])
    meshes_pathes = sorted([f for f in os.listdir(generation_folder) if f.endswith('mesh.ply')])
        
    print(f"meshes_pathes: {meshes_pathes}")

    initial_transforms = [None] * len(meshes_pathes) if initial_transforms is None else initial_transforms
    

    for view_idx, (mesh_filename, initial_transform) in enumerate(zip(meshes_pathes, initial_transforms)):
        print(f"Rendering [{view_idx+1}/{len(meshes_pathes)}] mesh")
        if multiview_inputs is None:
            output_dir = os.path.join(rendering_output_folder, f"view{view_idx}_mesh")
        else:
            multiview_input_index = multiview_inputs[view_idx] # This is an array of indices, e.g. [0,1]
            indices_str = "_".join([str(idx) for idx in multiview_input_index])
            output_dir = os.path.join(rendering_output_folder, f"multiimage_{indices_str}_mesh")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create full path to mesh file
        mesh_path = os.path.join(generation_folder, mesh_filename)
        
        if initial_transform is not None:
            print(f"Applying initial transformation to mesh {mesh_path}")
            mesh_path = apply_initial_transformation_and_save(mesh_path, initial_transform)
            print(f"Transformed mesh saved to {mesh_path}")

        render_mesh_from_transforms(
            mesh_path = mesh_path,
            transforms_json_path = transforms_json_path,
            output_dir = output_dir,
            resolution = resolution
        )

        if evaluate_iou:
            # Evaluate IoU for each view
            object_folder = os.path.dirname(generation_folder)
            if multiview_inputs is not None:
                # This mains the generated mesh is conditioned on multiple input images
                evaluate_folder_iou_multiview(
                    object_folder,
                    output_dir,
                    main_view_indices=multiview_inputs[view_idx]
                )
            else:
                evaluate_folder_iou(
                    object_folder,
                    output_dir,
                main_view_index=view_idx
            )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render predicted 3D asset from existing camera poses')
    # parser.add_argument('--mesh', type=str, required=True,
    #                     help='Path to the predicted 3D mesh (.glb, .ply, .obj)')
    parser.add_argument("--mesh_folder", type=str, required=True, 
                        help="Path to folder containing generated meshes")
    parser.add_argument('--transforms', type=str, required=True,
                        help='Path to transforms.json with camera poses')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for rendered images')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Image resolution (default: 512)')
    args = parser.parse_args()
    
    # Check Blender installation
    print('Checking Blender installation...')
    install_blender()
    
    # # Render single mesh
    # render_mesh_from_transforms(
    #     mesh_path=args.mesh,
    #     transforms_json_path=args.transforms,
    #     output_dir=args.output,
    #     resolution=args.resolution
    # )

    # Rendering a folder of meshes
    reproject_folder_and_evaluate_iou(
        generation_folder=args.mesh_folder,
        transforms_json_path=args.transforms,
        rendering_output_folder=args.output,
        evaluate_iou=True,
        resolution=args.resolution
    )
