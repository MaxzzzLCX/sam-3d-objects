#!/usr/bin/env python3
"""
Quick verification of normalized mesh properties
"""

import trimesh
import numpy as np

# Load a normalized mesh
normalized_path = "/scratch/cl927/nutritionverse-3d-new/id-104-chocolate-granola-bar-35g/normalized_mesh.obj"

print("Verifying normalized mesh...")
mesh = trimesh.load(normalized_path)

if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump(concatenate=True)

print(f"Bounds: {mesh.bounds}")
print(f"Bounding box center: {(mesh.bounds[0] + mesh.bounds[1]) / 2.0}")
print(f"Extents: {mesh.extents}")
print(f"Max extent: {np.max(mesh.extents)}")

# Check if properly normalized
bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
max_extent = np.max(mesh.extents)

print(f"\nVerification:")
print(f"✓ Bounding box centered at origin: {np.allclose(bbox_center, [0, 0, 0], atol=1e-10)}")
print(f"✓ Max dimension = 1.0: {np.isclose(max_extent, 1.0)}")
print(f"✓ Longest dimension spans [-0.5, 0.5]: {np.isclose(max_extent, 1.0)}")