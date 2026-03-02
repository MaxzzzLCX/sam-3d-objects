import trimesh
import os
import numpy as np

def calculate_span(mesh_path):
    if not os.path.exists(mesh_path):
        print(f"File not found: {mesh_path}")
        return
    
    try:
        mesh = trimesh.load(mesh_path)
        bounds = mesh.bounds  # shape (2, 3): [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        span = bounds[1] - bounds[0]
        x_span, y_span, z_span = span[0], span[1], span[2]
        print(f"Mesh: {os.path.basename(mesh_path)}")
        print(f"  X span: {x_span:.6f}, Y span: {y_span:.6f}, Z span: {z_span:.6f}")
        print(f"  Total vertices: {len(mesh.vertices)}, Total faces: {len(mesh.faces)}")
        print()
    except Exception as e:
        print(f"Error loading {mesh_path}: {e}")
        print()

def main():
    mesh_paths = [
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/orange_plate/generations_with_pointmaps/0/0_plate.ply",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/orange_plate/generations_with_pointmaps/0/0_food.ply",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/orange_plate/generations_with_pointmaps/1/1_food.ply",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/orange_plate/generations_with_pointmaps/2/2_food.ply"
    ]
    
    for mesh_path in mesh_paths:
        calculate_span(mesh_path)

if __name__ == "__main__":
    main()