#!/usr/bin/env python3
"""
Surface Completeness Evaluation using Flood Fill

This module evaluates whether a surface represented by sparse voxels is complete
by using flood fill from the boundary. If the flood reaches the centroid, 
the surface has holes.
"""

import numpy as np
from collections import deque
import time


def evaluate_surface_completeness_discrete(discrete_coords, grid_size=64):
    """
    Evaluate surface completeness using discrete integer coordinates directly
    
    Args:
        discrete_coords: Nx3 numpy array of discrete voxel coordinates [0, grid_size-1]
        grid_size: Size of the evaluation grid (default 64)
    
    Returns:
        dict: Results containing completeness metrics
    """
    if len(discrete_coords) == 0:
        return {
            'is_complete': False,
            'centroid_reachable': True,
            'flood_volume_ratio': 1.0,
            'evaluation_time': 0.0,
            'total_voxels': 0,
            'surface_voxels': 0
        }
    
    start_time = time.time()
    
    # Step 1: Coordinates are already discrete - no conversion needed!
    # Just ensure they're within bounds
    grid_coords = np.clip(discrete_coords.astype(int), 0, grid_size - 1)
    
    # Step 2: Create occupancy grid and mark surface voxels
    occupancy_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    
    # Mark surface voxels as occupied (solid)
    for coord in grid_coords:
        occupancy_grid[coord[0], coord[1], coord[2]] = True
    
    # Step 3: Calculate centroid in grid coordinates
    centroid = np.mean(grid_coords, axis=0).astype(int)
    centroid = np.clip(centroid, 0, grid_size - 1)  # Ensure within bounds
    
    # Step 4: Perform flood fill from all boundary faces
    flood_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    
    # Initialize boundary voxels for flood fill (6 faces of the cube)
    queue = deque()
    
    # Add all boundary voxels that are empty (not occupied by surface)
    for i in range(grid_size):
        for j in range(grid_size):
            # Front and back faces (z=0 and z=grid_size-1)
            if not occupancy_grid[i, j, 0]:
                flood_grid[i, j, 0] = True
                queue.append((i, j, 0))
            if not occupancy_grid[i, j, grid_size-1]:
                flood_grid[i, j, grid_size-1] = True
                queue.append((i, j, grid_size-1))
            
            # Left and right faces (x=0 and x=grid_size-1)
            if not occupancy_grid[0, i, j]:
                flood_grid[0, i, j] = True
                queue.append((0, i, j))
            if not occupancy_grid[grid_size-1, i, j]:
                flood_grid[grid_size-1, i, j] = True
                queue.append((grid_size-1, i, j))
            
            # Top and bottom faces (y=0 and y=grid_size-1)  
            if not occupancy_grid[i, 0, j]:
                flood_grid[i, 0, j] = True
                queue.append((i, 0, j))
            if not occupancy_grid[i, grid_size-1, j]:
                flood_grid[i, grid_size-1, j] = True
                queue.append((i, grid_size-1, j))
    
    # Step 5: Flood fill using 6-connectivity (no diagonals)
    directions = [
        (1, 0, 0), (-1, 0, 0),   # x-axis neighbors
        (0, 1, 0), (0, -1, 0),   # y-axis neighbors  
        (0, 0, 1), (0, 0, -1)    # z-axis neighbors
    ]
    
    centroid_reachable = False
    
    while queue:
        x, y, z = queue.popleft()
        
        # Check if we reached the centroid region (with small tolerance)
        if abs(x - centroid[0]) <= 1 and abs(y - centroid[1]) <= 1 and abs(z - centroid[2]) <= 1:
            centroid_reachable = True
            # Could break here for efficiency, but let's continue to get full flood volume
        
        # Explore 6 neighbors
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            # Check bounds
            if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
                # If neighbor is empty (not surface) and not yet flooded
                if not occupancy_grid[nx, ny, nz] and not flood_grid[nx, ny, nz]:
                    flood_grid[nx, ny, nz] = True
                    queue.append((nx, ny, nz))
    
    # Step 6: Calculate metrics
    total_voxels = grid_size ** 3
    surface_voxels = len(grid_coords)
    flooded_voxels = np.sum(flood_grid)
    flood_volume_ratio = flooded_voxels / total_voxels
    
    # Surface is complete if centroid is NOT reachable by flood
    is_complete = not centroid_reachable
    
    evaluation_time = time.time() - start_time
    
    return {
        'is_complete': is_complete,
        'centroid_reachable': centroid_reachable, 
        'flood_volume_ratio': flood_volume_ratio,
        'flooded_voxels': int(flooded_voxels),
        'evaluation_time': evaluation_time,
        'total_voxels': total_voxels,
        'surface_voxels': surface_voxels,
        'centroid_grid': centroid.tolist()
    }


def convert_raw_to_discrete(raw_voxel_coords):
    """
    Convert raw SAM3D voxel coordinates to discrete grid indices
    
    Args:
        raw_voxel_coords: Nx3 array of raw voxel coordinates in normalized space
        
    Returns:
        discrete_coords: Nx3 array of integer grid coordinates
        grid_info: dict with grid dimension information
    """
    discrete_coords = []
    grid_info = {}
    
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        axis_coords = raw_voxel_coords[:, axis_idx]
        unique_vals = np.unique(axis_coords)
        
        if len(unique_vals) == 1:
            # Degenerate case - all coordinates are the same
            indices = np.zeros(len(axis_coords), dtype=int)
            grid_size = 1
        else:
            # Map to integer indices based on uniform spacing
            spacing = (unique_vals[-1] - unique_vals[0]) / (len(unique_vals) - 1)
            indices = np.round((axis_coords - unique_vals[0]) / spacing).astype(int)
            grid_size = len(unique_vals)
        
        discrete_coords.append(indices)
        grid_info[axis_name] = {
            'size': grid_size,
            'range': [indices.min(), indices.max()],
            'unique_count': len(unique_vals)
        }
    
    discrete_coords = np.column_stack(discrete_coords)
    
    return discrete_coords, grid_info


def evaluate_surface_completeness_auto(voxel_coords, grid_size=64, prefer_discrete=True):
    """
    Automatically evaluate surface completeness using the best method
    
    Args:
        voxel_coords: Nx3 numpy array of voxel coordinates 
        grid_size: Resolution of evaluation grid (default 64, used only if discrete detection fails)
        prefer_discrete: If True (default), try to detect and use discrete coordinates directly
    
    Returns:
        dict: Results containing completeness metrics plus method info
    """
    
    if prefer_discrete:
        # Try to detect if these are raw discrete coordinates
        try:
            discrete_coords, grid_info = convert_raw_to_discrete(voxel_coords)
            
            # Use the largest dimension found as grid size
            max_grid_size = max(info['size'] for info in grid_info.values())
            
            # If we detected a reasonable discrete structure, use it
            if max_grid_size >= 16:  # Reasonable minimum threshold
                results = evaluate_surface_completeness_discrete(discrete_coords, max_grid_size)
                results['method'] = 'discrete'
                results['detected_grid_size'] = max_grid_size
                results['grid_info'] = grid_info
                return results
                
        except Exception as e:
            # Fall back to normalized method if discrete detection fails
            pass
    
    # Use normalized coordinate method as fallback
    results = evaluate_surface_completeness_normalized(voxel_coords, grid_size)
    results['method'] = 'normalized'
    results['requested_grid_size'] = grid_size
    return results


def evaluate_surface_completeness_normalized(voxel_coords, grid_size=64):
    """
    Evaluate surface completeness using flood fill algorithm
    
    Args:
        voxel_coords: Nx3 numpy array of voxel coordinates (normalized [-0.5, 0.5])
        grid_size: Resolution of the evaluation grid (default 64)
    
    Returns:
        dict: Results containing completeness metrics
    """
    if len(voxel_coords) == 0:
        return {
            'is_complete': False,
            'centroid_reachable': True,
            'flood_volume_ratio': 1.0,
            'evaluation_time': 0.0,
            'total_voxels': 0,
            'surface_voxels': 0
        }
    
    start_time = time.time()
    
    # Step 1: Convert normalized coordinates to grid indices
    # Normalized coords are in [-0.5, 0.5], convert to [0, grid_size-1]
    grid_coords = np.round((voxel_coords + 0.5) * (grid_size - 1)).astype(int)
    
    # Clamp to valid range
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)
    
    # Step 2: Create occupancy grid and mark surface voxels
    occupancy_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    
    # Mark surface voxels as occupied (solid)
    for coord in grid_coords:
        occupancy_grid[coord[0], coord[1], coord[2]] = True
    
    # Step 3: Calculate centroid in grid coordinates
    centroid = np.mean(grid_coords, axis=0).astype(int)
    centroid = np.clip(centroid, 0, grid_size - 1)  # Ensure within bounds
    
    # Step 4: Perform flood fill from all boundary faces
    flood_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    
    # Initialize boundary voxels for flood fill (6 faces of the cube)
    queue = deque()
    
    # Add all boundary voxels that are empty (not occupied by surface)
    for i in range(grid_size):
        for j in range(grid_size):
            # Front and back faces (z=0 and z=grid_size-1)
            if not occupancy_grid[i, j, 0]:
                flood_grid[i, j, 0] = True
                queue.append((i, j, 0))
            if not occupancy_grid[i, j, grid_size-1]:
                flood_grid[i, j, grid_size-1] = True
                queue.append((i, j, grid_size-1))
            
            # Left and right faces (x=0 and x=grid_size-1)
            if not occupancy_grid[0, i, j]:
                flood_grid[0, i, j] = True
                queue.append((0, i, j))
            if not occupancy_grid[grid_size-1, i, j]:
                flood_grid[grid_size-1, i, j] = True
                queue.append((grid_size-1, i, j))
            
            # Top and bottom faces (y=0 and y=grid_size-1)  
            if not occupancy_grid[i, 0, j]:
                flood_grid[i, 0, j] = True
                queue.append((i, 0, j))
            if not occupancy_grid[i, grid_size-1, j]:
                flood_grid[i, grid_size-1, j] = True
                queue.append((i, grid_size-1, j))
    
    # Step 5: Flood fill using 6-connectivity (no diagonals)
    directions = [
        (1, 0, 0), (-1, 0, 0),   # x-axis neighbors
        (0, 1, 0), (0, -1, 0),   # y-axis neighbors  
        (0, 0, 1), (0, 0, -1)    # z-axis neighbors
    ]
    
    centroid_reachable = False
    
    while queue:
        x, y, z = queue.popleft()
        
        # Check if we reached the centroid region (with small tolerance)
        if abs(x - centroid[0]) <= 1 and abs(y - centroid[1]) <= 1 and abs(z - centroid[2]) <= 1:
            centroid_reachable = True

            # Could break here for efficiency, but let's continue to get full flood volume
        
        # Explore 6 neighbors
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            # Check bounds
            if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
                # If neighbor is empty (not surface) and not yet flooded
                if not occupancy_grid[nx, ny, nz] and not flood_grid[nx, ny, nz]:
                    flood_grid[nx, ny, nz] = True
                    queue.append((nx, ny, nz))
    
    # Step 6: Calculate metrics
    total_voxels = grid_size ** 3
    surface_voxels = len(grid_coords)
    flooded_voxels = np.sum(flood_grid)
    flood_volume_ratio = flooded_voxels / total_voxels
    
    # Surface is complete if centroid is NOT reachable by flood
    is_complete = not centroid_reachable
    
    evaluation_time = time.time() - start_time
    
    return {
        'is_complete': is_complete,
        'centroid_reachable': centroid_reachable, 
        'flood_volume_ratio': flood_volume_ratio,
        'flooded_voxels': int(flooded_voxels),
        'evaluation_time': evaluation_time,
        'total_voxels': total_voxels,
        'surface_voxels': surface_voxels,
        'centroid_grid': centroid.tolist()
    }


# Default function - uses raw discrete coordinates when possible
def evaluate_surface_completeness(voxel_coords, grid_size=64, prefer_discrete=True):
    """
    Evaluate surface completeness using the best available method
    
    By default, this function:
    1. Tries to detect discrete coordinate structure (preferred for accuracy)
    2. Falls back to normalized coordinate method if needed
    
    Args:
        voxel_coords: Nx3 numpy array of voxel coordinates
        grid_size: Grid resolution for normalized method (default 64)
        prefer_discrete: If True (default), prefer discrete method for raw coordinates
    
    Returns:
        dict: Surface completeness results with method information
    """
    return evaluate_surface_completeness_auto(voxel_coords, grid_size, prefer_discrete)


def evaluate_surface_completeness_force_normalized(voxel_coords, grid_size=64):
    """
    Force evaluation using normalized coordinate method
    
    Use this when you specifically want the normalized method regardless
    of coordinate structure (e.g., for comparison purposes).
    
    Args:
        voxel_coords: Nx3 numpy array of voxel coordinates 
        grid_size: Resolution of evaluation grid (default 64)
    
    Returns:
        dict: Surface completeness results using normalized method
    """
    return evaluate_surface_completeness_normalized(voxel_coords, grid_size)


def evaluate_surface_completeness_for_files(voxel_file_path, output_dir=None):
    """
    Evaluate surface completeness from a voxel file (PLY or NPY)
    
    Args:
        voxel_file_path: Path to voxel file (PLY or NPY format)
        output_dir: Directory to save results (defaults to same as voxel file)
    
    Returns:
        dict: Surface completeness results, or None if failed
    """
    import os
    from pathlib import Path
    import trimesh
    
    try:
        # Load voxel coordinates
        if voxel_file_path.endswith('.ply'):
            mesh = trimesh.load(voxel_file_path)
            if hasattr(mesh, 'vertices'):
                voxel_coords = np.array(mesh.vertices)
            else:
                print(f"Error: Could not load vertices from {voxel_file_path}")
                return None
        elif voxel_file_path.endswith('.npy'):
            voxel_coords = np.load(voxel_file_path)
        else:
            print(f"Error: Unsupported file format {voxel_file_path}")
            return None
        
        # Evaluate completeness using auto-detection
        results = evaluate_surface_completeness_auto(voxel_coords)
        
        # Set output directory
        if output_dir is None:
            output_dir = Path(voxel_file_path).parent
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = output_dir / 'surface_completeness_results.json'
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"Error evaluating surface completeness: {e}")
        return None


def main():
    """
    Main function for standalone testing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate surface completeness using flood fill')
    parser.add_argument('--voxel_path', type=str, required=True, help='Path to voxel file (PLY or NPY)')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    print(f"Evaluating surface completeness for: {args.voxel_path}")
    
    results = evaluate_surface_completeness_for_files(args.voxel_path, args.output_dir)
    
    if results:
        print(f"\nSurface Completeness Results:")
        print(f"  Is Complete: {results['is_complete']}")
        print(f"  Centroid Reachable: {results['centroid_reachable']}")
        print(f"  Flood Volume Ratio: {results['flood_volume_ratio']:.3f}")
        print(f"  Surface Voxels: {results['surface_voxels']:,}")
        print(f"  Flooded Voxels: {results['flooded_voxels']:,}")
        print(f"  Evaluation Time: {results['evaluation_time']:.3f} seconds")
        
        from pathlib import Path
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.voxel_path).parent
        results_file = output_dir / 'surface_completeness_results.json'
        print(f"  Results saved to: {results_file}")


if __name__ == "__main__":
    main()