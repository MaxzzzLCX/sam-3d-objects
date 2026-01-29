#!/usr/bin/env python3
"""
Surface reconstruction from fused voxel grids using Poisson Surface Reconstruction
"""

import numpy as np
import trimesh
import open3d as o3d
import sys
import os

def extract_surface_points_with_normals(fused_grid, threshold=0.0, use_gradients=True):
    """
    Extract surface points from grid(s) and estimate normals
    
    Args:
        fused_grid: Either a single 64x64x64 grid OR a tuple of (grid1, grid2)
        threshold: Logit threshold for occupancy
        use_gradients: Use gradient-based normal estimation
    
    Returns:
        points: Nx3 surface points in normalized coordinates [-0.5, 0.5]
        normals: Nx3 surface normals
    """
    
    # Handle both single grid and dual grid cases
    if isinstance(fused_grid, tuple):
        # Two separate grids - process each independently
        grid1, grid2 = fused_grid
        print(f"Processing two separate grids with threshold={threshold}")
        
        points_list = []
        normals_list = []
        
        for i, grid in enumerate([grid1, grid2], 1):
            print(f"  Processing grid {i}...")
            occupied_coords = np.array(np.where(grid > threshold)).T
            
            if len(occupied_coords) == 0:
                print(f"    No occupied voxels in grid {i}")
                continue
                
            print(f"    Found {len(occupied_coords)} occupied voxels in grid {i}")
            
            # Convert to normalized coordinates
            points = (occupied_coords.astype(np.float32) / 63.0) - 0.5
            points_list.append(points)
            
            # Estimate normals for this grid
            if use_gradients and len(occupied_coords) > 100:
                normals = estimate_normals_from_gradients(grid, occupied_coords)
            else:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
                )
                normals = np.asarray(pcd.normals)
            
            normals_list.append(normals)
        
        # Concatenate all points and normals
        if points_list:
            combined_points = np.vstack(points_list)
            combined_normals = np.vstack(normals_list)
            print(f"Combined {len(combined_points)} points from {len(points_list)} grids")
            return combined_points, combined_normals
        else:
            print("No points found in any grid!")
            return np.array([]), np.array([])
    
    else:
        # Single fused grid - original logic
        print(f"Processing single fused grid with threshold={threshold}")
        occupied_coords = np.array(np.where(fused_grid > threshold)).T
        
        if len(occupied_coords) == 0:
            print("No occupied voxels found!")
            return np.array([]), np.array([])
        
        print(f"Found {len(occupied_coords)} occupied voxels")
        
        # Convert to normalized coordinates
        points = (occupied_coords.astype(np.float32) / 63.0) - 0.5
        
        if use_gradients and len(occupied_coords) > 100:
            print("Estimating normals using grid gradients...")
            normals = estimate_normals_from_gradients(fused_grid, occupied_coords)
        else:
            print("Estimating normals using Open3D...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
            )
            normals = np.asarray(pcd.normals)
        
        print(f"Generated {len(normals)} normals")
        return points, normals

def estimate_normals_from_gradients(grid, coords):
    """Estimate normals using finite differences on the occupancy grid"""
    
    normals = []
    
    for coord in coords:
        x, y, z = coord
        
        # Compute gradient using central differences
        grad = np.array([0.0, 0.0, 0.0])
        
        # X gradient
        if x > 0 and x < 63:
            grad[0] = grid[x+1, y, z] - grid[x-1, y, z]
        elif x == 0:
            grad[0] = grid[x+1, y, z] - grid[x, y, z]
        else:  # x == 63
            grad[0] = grid[x, y, z] - grid[x-1, y, z]
            
        # Y gradient  
        if y > 0 and y < 63:
            grad[1] = grid[x, y+1, z] - grid[x, y-1, z]
        elif y == 0:
            grad[1] = grid[x, y+1, z] - grid[x, y, z]
        else:  # y == 63
            grad[1] = grid[x, y, z] - grid[x, y-1, z]
            
        # Z gradient
        if z > 0 and z < 63:
            grad[2] = grid[x, y, z+1] - grid[x, y, z-1]
        elif z == 0:
            grad[2] = grid[x, y, z+1] - grid[x, y, z]
        else:  # z == 63
            grad[2] = grid[x, y, z] - grid[x, y, z-1]
        
        # Normalize gradient to get normal (gradient points outward from surface)
        norm = np.linalg.norm(grad)
        if norm > 1e-6:
            normal = grad / norm
        else:
            normal = np.array([0.0, 0.0, 1.0])  # Default normal
            
        normals.append(normal)
    
    return np.array(normals)

def poisson_surface_reconstruction(points, normals, depth=9, width=0, scale=1.1, linear_fit=False):
    """
    Perform Poisson Surface Reconstruction using Open3D
    
    Args:
        points: Nx3 surface points
        normals: Nx3 surface normals  
        depth: Octree depth for PSR (higher = more detail)
        width: Octree width (0 = auto)
        scale: Bounding box scale factor
        linear_fit: Use linear fit for interpolation
    
    Returns:
        mesh: Open3D triangle mesh
    """
    print(f"Running Poisson Surface Reconstruction...")
    print(f"  Points: {len(points)}")
    print(f"  Depth: {depth}, Scale: {scale}")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Orient normals consistently (important for PSR)
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, 
        depth=depth,
        width=width, 
        scale=scale,
        linear_fit=linear_fit
    )
    
    print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    # Remove low-density vertices (optional cleanup)
    if len(densities) > 0:
        density_threshold = np.quantile(np.asarray(densities), 0.01)  # Remove bottom 1%
        vertices_to_remove = np.asarray(densities) < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"After cleanup: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    return mesh

def save_mesh(mesh, output_file="surface_reconstruction.ply", format="ply"):
    """Save mesh to file"""
    
    if format.lower() == "ply":
        success = o3d.io.write_triangle_mesh(output_file, mesh)
    elif format.lower() == "obj":
        success = o3d.io.write_triangle_mesh(output_file, mesh)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    if success:
        print(f"Saved mesh to {output_file}")
    else:
        print(f"Failed to save mesh to {output_file}")

def load_grid_data(grid_file):
    """Load grid data from NPZ file - returns either single grid or tuple of grids"""
    
    if not os.path.exists(grid_file):
        raise FileNotFoundError(f"Grid file not found: {grid_file}")
    
    data = np.load(grid_file)
    print(f"Available keys in {grid_file}: {list(data.keys())}")
    
    # Check if it's a fused grid file
    if 'fused_grid' in data:
        fused_grid = data['fused_grid']
        method = data.get('method', 'unknown')
        print(f"Loaded fused grid from {grid_file}")
        print(f"  Method: {method}")
        print(f"  Grid shape: {fused_grid.shape}, range: [{fused_grid.min():.6f}, {fused_grid.max():.6f}]")
        return fused_grid
    
    # Check if it's an aligned grids file (two separate grids)
    elif 'view1_grid' in data and 'view2_grid' in data:
        view1_grid = data['view1_grid']
        view2_grid = data['view2_grid']
        method = data.get('method', 'unknown')
        
        print(f"Loaded aligned grids from {grid_file}")
        print(f"  Method: {method}")
        print(f"  View1 grid: shape={view1_grid.shape}, range=[{view1_grid.min():.6f}, {view1_grid.max():.6f}]")
        print(f"  View2 grid: shape={view2_grid.shape}, range=[{view2_grid.min():.6f}, {view2_grid.max():.6f}]")
        print("  Returning separate grids for combined point cloud processing")
        return view1_grid, view2_grid
        
    else:
        raise ValueError(f"Unknown file format. Expected 'fused_grid' or 'view1_grid'+'view2_grid' in {grid_file}")
    
    print(f"  Final grid shape: {fused_grid.shape}")
    print(f"  Final grid range: [{fused_grid.min():.6f}, {fused_grid.max():.6f}]")
    print(f"  Occupied voxels: {np.sum(fused_grid > 0)}")
    
    return fused_grid

def run_surface_reconstruction(grid_file, depth=8, threshold=0.0, output_format="ply"):
    """
    Main pipeline for surface reconstruction
    
    Args:
        grid_file: Path to fused grid NPZ file OR aligned grids NPZ file
        depth: Poisson reconstruction depth
        threshold: Occupancy threshold
        output_format: Output file format ("ply" or "obj")
    """
    
    print("=== Poisson Surface Reconstruction ===")
    
    # Load grid (handles both fused and aligned formats)
    try:
        grid_data = load_grid_data(grid_file)
    except FileNotFoundError:
        print(f"Error: {grid_file} not found!")
        print("Run alignment or fusion first to generate grid files.")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Extract surface points and normals (handles both single grid and tuple)
    points, normals = extract_surface_points_with_normals(grid_data, threshold=threshold)
    
    if len(points) == 0:
        print("No surface points found! Try lowering the threshold.")
        return
    
    # Perform Poisson surface reconstruction
    mesh = poisson_surface_reconstruction(points, normals, depth=depth)
    
    if len(mesh.vertices) == 0:
        print("Surface reconstruction failed!")
        return
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(grid_file))[0]
    output_file = f"surface_{base_name}_depth{depth}.{output_format}"
    
    # Save mesh
    save_mesh(mesh, output_file, output_format)
    
    # Print summary
    print(f"\n=== Reconstruction Summary ===")
    print(f"Input: {grid_file}")
    print(f"Output: {output_file}")
    print(f"Points: {len(points)}")
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    print(f"PSR depth: {depth}")
    print(f"Threshold: {threshold}")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python surface_reconstruction.py <grid_file.npz> [depth] [threshold] [format]")
        print("  grid_file.npz can be either:")
        print("    - fusion_*.npz (pre-fused grid)")
        print("    - aligned_grids_*.npz (will fuse on-the-fly)")
        print("Example: python surface_reconstruction.py aligned_grids_surface_voxels.npz 8 0.0 ply")
        print("Example: python surface_reconstruction.py fusion_arithmetic.npz 8 0.0 ply")
        sys.exit(1)
    
    grid_file = sys.argv[1]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    output_format = sys.argv[4] if len(sys.argv) > 4 else "ply"
    
    run_surface_reconstruction(grid_file, depth, threshold, output_format)