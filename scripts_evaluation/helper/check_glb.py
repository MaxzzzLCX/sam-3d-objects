"""
This file is used to check what is in a glb file.
"""
import trimesh
import os

def save_glb_as_ply(glb_path):

    # Load the GLB file
    mesh = trimesh.load(glb_path)
    
    # Mesh volume
    volume = mesh.volume
    print(f"Mesh volume: {volume}")

    # Save the mesh as a PLY file
    ply_path = glb_path.replace(".glb", ".ply")
    mesh.export(ply_path)

    print(f"Saved {glb_path} as {ply_path}")

def check_volume(path):
    mesh = trimesh.load(path)
    volume = mesh.volume
    print(f"Mesh volume: {volume}")

if __name__ == "__main__":

    dataset_folder_path = "/scratch/cl927/datasets/Toys4k/subset_foodlike"
    folder_paths = sorted([f for f in os.listdir(dataset_folder_path) if os.path.isdir(os.path.join(dataset_folder_path, f))])[13:]
    print(f"folder_paths: {folder_paths}")

    generation_method = "trellis_multiimage_outputs"
    for folder_path in folder_paths:
        glb_paths = sorted(f for f in os.listdir(os.path.join(dataset_folder_path, folder_path, generation_method)) if f.endswith(".glb"))
        print(f"glb_paths: {glb_paths}")
        for glb_path in glb_paths:
            print(f"Checking {glb_path}...")
            save_glb_as_ply(os.path.join(dataset_folder_path, folder_path, generation_method, glb_path))
