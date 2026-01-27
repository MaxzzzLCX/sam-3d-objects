
# Read a .ply file and print the number of points
import trimesh
mesh = trimesh.load("occupancy_grid.ply")
print(f"Number of points in occupancy grid: {len(mesh.vertices)}")

mesh = trimesh.load("occupancy_grid_raw.ply")
print(f"Number of points in occupancy grid raw: {len(mesh.vertices)}")