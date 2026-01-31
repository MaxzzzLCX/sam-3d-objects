#!/usr/bin/env python3
"""
Test Surface Completeness Evaluation

This script tests the surface completeness evaluation on simple geometric shapes
to verify the flood fill algorithm works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from surface_completeness_evaluation import evaluate_surface_completeness


def create_cube_surface(size=0.4, resolution=32):
    """Create surface voxels for a cube (should be complete)"""
    half_size = size / 2
    voxels = []
    
    # Generate cube faces
    step = size / resolution
    
    # Create 6 faces of the cube
    for i in range(resolution):
        for j in range(resolution):
            x = -half_size + i * step
            y = -half_size + j * step
            
            # Front and back faces (z = ±half_size)
            voxels.append([x, y, -half_size])
            voxels.append([x, y, half_size])
            
            # Left and right faces (x = ±half_size)  
            voxels.append([-half_size, x, y])
            voxels.append([half_size, x, y])
            
            # Top and bottom faces (y = ±half_size)
            voxels.append([x, -half_size, y])
            voxels.append([x, half_size, y])
    
    return np.array(voxels)


def create_sphere_surface(radius=0.4, num_points=2000):
    """Create surface voxels for a sphere (should be complete)"""
    # Generate random points on sphere surface
    phi = np.random.uniform(0, 2*np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    
    theta = np.arccos(costheta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    return np.column_stack([x, y, z])


def create_incomplete_sphere(radius=0.4, num_points=2000, hole_size=0.2):
    """Create sphere with a hole (should be incomplete)"""
    voxels = create_sphere_surface(radius, num_points)
    
    # Remove points in a region to create a hole
    # Remove points where x > radius - hole_size (creates hole on +x side)
    mask = voxels[:, 0] < (radius - hole_size)
    
    return voxels[mask]


def create_open_cylinder(radius=0.3, height=0.6, resolution=32):
    """Create open cylinder (should be incomplete - open ends)"""
    voxels = []
    
    # Generate cylindrical surface (no top/bottom caps)
    for i in range(resolution):
        theta = 2 * np.pi * i / resolution
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        
        # Generate points along height
        for j in range(resolution):
            y = -height/2 + j * (height / resolution)
            voxels.append([x, y, z])
    
    return np.array(voxels)


def visualize_voxels(voxels, title="Voxels"):
    """Visualize voxels in 3D"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(voxels[:, 0], voxels[:, 1], voxels[:, 2], alpha=0.6, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = 0.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    return fig


def test_surface_completeness():
    """Test surface completeness on different geometric shapes"""
    
    print("Testing Surface Completeness Evaluation")
    print("=" * 50)
    
    # Test 1: Complete cube
    print("\n1. Testing complete cube...")
    cube_voxels = create_cube_surface(size=0.4, resolution=16)
    cube_results = evaluate_surface_completeness(cube_voxels, grid_size=64)
    
    print(f"   Cube results:")
    print(f"   - Is complete: {cube_results['is_complete']}")
    print(f"   - Centroid reachable: {cube_results['centroid_reachable']}")
    print(f"   - Flood volume ratio: {cube_results['flood_volume_ratio']:.3f}")
    print(f"   - Surface voxels: {cube_results['surface_voxels']}")
    print(f"   - Evaluation time: {cube_results['evaluation_time']:.3f}s")
    
    # Test 2: Complete sphere  
    print(f"\\n2. Testing complete sphere...")
    sphere_voxels = create_sphere_surface(radius=0.4, num_points=1500)
    sphere_results = evaluate_surface_completeness(sphere_voxels, grid_size=64)
    
    print(f"   Sphere results:")
    print(f"   - Is complete: {sphere_results['is_complete']}")
    print(f"   - Centroid reachable: {sphere_results['centroid_reachable']}")
    print(f"   - Flood volume ratio: {sphere_results['flood_volume_ratio']:.3f}")
    print(f"   - Surface voxels: {sphere_results['surface_voxels']}")
    print(f"   - Evaluation time: {sphere_results['evaluation_time']:.3f}s")
    
    # Test 3: Incomplete sphere (with hole)
    print(f"\\n3. Testing sphere with hole...")
    holey_voxels = create_incomplete_sphere(radius=0.4, num_points=1500, hole_size=0.15)
    holey_results = evaluate_surface_completeness(holey_voxels, grid_size=64)
    
    print(f"   Holey sphere results:")
    print(f"   - Is complete: {holey_results['is_complete']}")
    print(f"   - Centroid reachable: {holey_results['centroid_reachable']}")
    print(f"   - Flood volume ratio: {holey_results['flood_volume_ratio']:.3f}")
    print(f"   - Surface voxels: {holey_results['surface_voxels']}")
    print(f"   - Evaluation time: {holey_results['evaluation_time']:.3f}s")
    
    # Test 4: Open cylinder
    print(f"\\n4. Testing open cylinder...")
    cylinder_voxels = create_open_cylinder(radius=0.3, height=0.6, resolution=24)
    cylinder_results = evaluate_surface_completeness(cylinder_voxels, grid_size=64)
    
    print(f"   Open cylinder results:")
    print(f"   - Is complete: {cylinder_results['is_complete']}")
    print(f"   - Centroid reachable: {cylinder_results['centroid_reachable']}")
    print(f"   - Flood volume ratio: {cylinder_results['flood_volume_ratio']:.3f}")
    print(f"   - Surface voxels: {cylinder_results['surface_voxels']}")
    print(f"   - Evaluation time: {cylinder_results['evaluation_time']:.3f}s")
    
    # Test 5: Empty (edge case)
    print(f"\\n5. Testing empty voxels...")
    empty_results = evaluate_surface_completeness(np.array([]), grid_size=64)
    
    print(f"   Empty results:")
    print(f"   - Is complete: {empty_results['is_complete']}")
    print(f"   - Centroid reachable: {empty_results['centroid_reachable']}")
    print(f"   - Flood volume ratio: {empty_results['flood_volume_ratio']:.3f}")
    print(f"   - Surface voxels: {empty_results['surface_voxels']}")
    
    print(f"\\n" + "=" * 50)
    print("Test Summary:")
    print(f"- Cube (expected complete): {'✓' if cube_results['is_complete'] else '✗'}")
    print(f"- Sphere (expected complete): {'✓' if sphere_results['is_complete'] else '✗'}")
    print(f"- Holey sphere (expected incomplete): {'✓' if not holey_results['is_complete'] else '✗'}")
    print(f"- Open cylinder (expected incomplete): {'✓' if not cylinder_results['is_complete'] else '✗'}")
    print(f"- Empty (expected incomplete): {'✓' if not empty_results['is_complete'] else '✗'}")
    
    # Optional: Create visualizations
    try:
        print(f"\\nGenerating visualizations...")
        
        fig1 = visualize_voxels(cube_voxels, f"Complete Cube (Complete: {cube_results['is_complete']})")
        fig1.savefig('/tmp/test_cube.png', dpi=100, bbox_inches='tight')
        
        fig2 = visualize_voxels(sphere_voxels, f"Complete Sphere (Complete: {sphere_results['is_complete']})")
        fig2.savefig('/tmp/test_sphere.png', dpi=100, bbox_inches='tight')
        
        fig3 = visualize_voxels(holey_voxels, f"Sphere with Hole (Complete: {holey_results['is_complete']})")
        fig3.savefig('/tmp/test_holey_sphere.png', dpi=100, bbox_inches='tight')
        
        fig4 = visualize_voxels(cylinder_voxels, f"Open Cylinder (Complete: {cylinder_results['is_complete']})")
        fig4.savefig('/tmp/test_cylinder.png', dpi=100, bbox_inches='tight')
        
        print("Visualizations saved to /tmp/test_*.png")
        
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    test_surface_completeness()