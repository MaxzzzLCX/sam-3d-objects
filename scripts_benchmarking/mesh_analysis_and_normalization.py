#!/usr/bin/env python3
"""
Mesh Analysis and Normalization Script for NutritionVerse-3D Dataset

This script:
1. Analyzes all meshes in the dataset to understand their sizes and properties
2. Normalizes meshes using bounding box centering and unit cube scaling
3. Provides benchmark utilities for Chamfer distance computation

Normalization strategy:
- Center bounding box at origin (0,0,0)
- Scale so longest dimension spans [-0.5, 0.5] (unit cube)
"""

import os
import glob
import numpy as np
import trimesh
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_mesh(mesh_path):
    """
    Analyze a single mesh and extract key statistics
    
    Args:
        mesh_path: Path to the mesh file (.obj or .ply)
    
    Returns:
        dict: Dictionary containing mesh statistics
    """
    try:
        mesh = trimesh.load(mesh_path)
        
        # Handle case where multiple meshes are loaded
        if isinstance(mesh, trimesh.Scene):
            # Combine all geometries in the scene
            mesh = mesh.dump(concatenate=True)
        
        # Basic properties
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        extents = mesh.extents  # [extent_x, extent_y, extent_z]
        center = mesh.centroid
        volume = mesh.volume if mesh.is_watertight else None
        area = mesh.area
        
        # Bounding box diagonal
        diagonal = np.linalg.norm(extents)
        
        # Maximum extent (for normalization)
        max_extent = np.max(extents)
        
        stats = {
            'path': mesh_path,
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'bounds_min': bounds[0],
            'bounds_max': bounds[1],
            'extents': extents,
            'center': center,
            'volume': volume,
            'surface_area': area,
            'diagonal': diagonal,
            'max_extent': max_extent,
            'is_watertight': mesh.is_watertight,
            'aspect_ratios': extents / np.min(extents)  # aspect ratios relative to smallest dimension
        }
        
        return stats
        
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        return None


def normalize_mesh(mesh):
    """
    Normalize mesh by centering bounding box at origin and scaling to unit cube [-0.5, 0.5]
    
    Args:
        mesh: trimesh.Trimesh object
    
    Returns:
        trimesh.Trimesh: Normalized mesh
    """
    mesh_normalized = mesh.copy()
    
    # Get bounding box
    bounds = mesh_normalized.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    
    # Step 1: Center bounding box at origin
    bbox_center = (bounds[0] + bounds[1]) / 2.0  # Center of bounding box
    mesh_normalized.vertices -= bbox_center
    
    # Step 2: Scale so longest dimension spans [-0.5, 0.5] (total span = 1.0)
    extents = mesh_normalized.extents  # Recalculate after centering
    max_extent = np.max(extents)
    if max_extent > 0:
        mesh_normalized.vertices /= max_extent
    
    return mesh_normalized


def sample_points_from_mesh(mesh, n_points=2048, method='surface'):
    """
    Sample points from mesh for Chamfer distance computation
    
    Args:
        mesh: trimesh.Trimesh object
        n_points: Number of points to sample
        method: 'surface' for surface sampling, 'volume' for volume sampling (if watertight)
    
    Returns:
        np.ndarray: Sampled points of shape (n_points, 3)
    """
    if method == 'surface':
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
    elif method == 'volume' and mesh.is_watertight:
        points = trimesh.sample.sample_volume(mesh, n_points)
    else:
        # Fallback to surface sampling
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
    
    return points


def analyze_dataset(dataset_path):
    """
    Analyze all meshes in the dataset
    
    Args:
        dataset_path: Path to the dataset root directory
    
    Returns:
        pd.DataFrame: DataFrame containing analysis results
    """
    # Find all .obj files in the dataset (both poly.obj and textured.obj)
    poly_files = glob.glob(os.path.join(dataset_path, "**/poly.obj"), recursive=True)
    textured_files = glob.glob(os.path.join(dataset_path, "**/textured.obj"), recursive=True)
    obj_files = poly_files + textured_files
    
    print(f"Found {len(obj_files)} mesh files in the dataset")
    print(f"  - {len(poly_files)} poly.obj files")
    print(f"  - {len(textured_files)} textured.obj files")
    
    results = []
    
    for obj_file in tqdm(obj_files, desc="Analyzing meshes"):
        stats = analyze_mesh(obj_file)
        if stats is not None:
            # Extract food item ID and weight from path
            folder_name = Path(obj_file).parent.name
            stats['food_item'] = folder_name
            results.append(stats)
    
    df = pd.DataFrame(results)
    return df


def visualize_analysis(df, save_path=None):
    """
    Create visualizations of the mesh analysis
    
    Args:
        df: DataFrame from analyze_dataset
        save_path: Optional path to save plots
    """
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution of max extents
    axes[0, 0].hist(df['max_extent'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Maximum Extent')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Maximum Extents')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution of diagonal lengths
    axes[0, 1].hist(df['diagonal'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Bounding Box Diagonal')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Bounding Box Diagonals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Volume distribution (only for watertight meshes)
    watertight_df = df[df['is_watertight'] == True]
    if len(watertight_df) > 0:
        axes[0, 2].hist(watertight_df['volume'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Volume')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title(f'Distribution of Volumes ({len(watertight_df)} watertight meshes)')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Aspect ratios
    aspect_ratios = np.array([ar for ar in df['aspect_ratios']])
    axes[1, 0].boxplot([aspect_ratios[:, 0], aspect_ratios[:, 1], aspect_ratios[:, 2]])
    axes[1, 0].set_xticklabels(['X/min', 'Y/min', 'Z/min'])
    axes[1, 0].set_ylabel('Aspect Ratio')
    axes[1, 0].set_title('Aspect Ratios Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Number of vertices vs faces
    axes[1, 1].scatter(df['vertices'], df['faces'], alpha=0.6)
    axes[1, 1].set_xlabel('Number of Vertices')
    axes[1, 1].set_ylabel('Number of Faces')
    axes[1, 1].set_title('Vertices vs Faces')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Watertight mesh percentage
    watertight_counts = df['is_watertight'].value_counts()
    # Create labels that match the actual data
    labels = []
    for val in watertight_counts.index:
        if val:
            labels.append('Watertight')
        else:
            labels.append('Non-watertight')
    axes[1, 2].pie(watertight_counts.values, labels=labels, 
                   autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('Watertight vs Non-watertight Meshes')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print summary statistics
    print("\n=== MESH ANALYSIS SUMMARY ===")
    print(f"Total meshes analyzed: {len(df)}")
    print(f"Watertight meshes: {df['is_watertight'].sum()} ({df['is_watertight'].mean()*100:.1f}%)")
    print(f"\nMax extent statistics:")
    print(f"  Mean: {df['max_extent'].mean():.4f}")
    print(f"  Std:  {df['max_extent'].std():.4f}")
    print(f"  Min:  {df['max_extent'].min():.4f}")
    print(f"  Max:  {df['max_extent'].max():.4f}")
    print(f"\nDiagonal statistics:")
    print(f"  Mean: {df['diagonal'].mean():.4f}")
    print(f"  Std:  {df['diagonal'].std():.4f}")
    print(f"  Min:  {df['diagonal'].min():.4f}")
    print(f"  Max:  {df['diagonal'].max():.4f}")
    
    if len(watertight_df) > 0:
        print(f"\nVolume statistics (watertight only):")
        print(f"  Mean: {watertight_df['volume'].mean():.4f}")
        print(f"  Std:  {watertight_df['volume'].std():.4f}")
        print(f"  Min:  {watertight_df['volume'].min():.4f}")
        print(f"  Max:  {watertight_df['volume'].max():.4f}")


def demonstrate_normalization(sample_mesh_path):
    """
    Demonstrate normalization on a sample mesh
    
    Args:
        sample_mesh_path: Path to a sample mesh file
    """
    print(f"\nDemonstrating normalization on: {sample_mesh_path}")
    
    # Load original mesh
    mesh_original = trimesh.load(sample_mesh_path)
    if isinstance(mesh_original, trimesh.Scene):
        mesh_original = mesh_original.dump(concatenate=True)
    
    print(f"Original mesh:")
    print(f"  Bounds: {mesh_original.bounds}")
    print(f"  Extents: {mesh_original.extents}")
    print(f"  Bbox center: {(mesh_original.bounds[0] + mesh_original.bounds[1]) / 2.0}")
    print(f"  Max extent: {np.max(mesh_original.extents):.4f}")
    
    # Normalize
    mesh_normalized = normalize_mesh(mesh_original)
    print(f"\nNormalized mesh:")
    print(f"  Bounds: {mesh_normalized.bounds}")
    print(f"  Extents: {mesh_normalized.extents}")
    print(f"  Bbox center: {(mesh_normalized.bounds[0] + mesh_normalized.bounds[1]) / 2.0}")
    print(f"  Max extent: {np.max(mesh_normalized.extents):.4f}")
    print(f"  Should span [-0.5, 0.5]: {mesh_normalized.bounds}")


def save_normalized_meshes(dataset_path):
    """
    Save normalized versions of all meshes in the dataset in the same directory as original
    
    Args:
        dataset_path: Path to the original dataset
    """
    # Find all .obj files (both poly.obj and textured.obj)
    poly_files = glob.glob(os.path.join(dataset_path, "**/poly.obj"), recursive=True)
    textured_files = glob.glob(os.path.join(dataset_path, "**/textured.obj"), recursive=True)
    obj_files = poly_files + textured_files
    
    print(f"Normalizing {len(obj_files)} meshes")
    
    success_count = 0
    error_count = 0
    
    for obj_file in tqdm(obj_files, desc="Normalizing meshes"):
        try:
            # Load mesh
            mesh = trimesh.load(obj_file)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # Normalize
            mesh_normalized = normalize_mesh(mesh)
            
            # Save in the same directory as original with name "normalized_mesh.obj"
            output_dir = Path(obj_file).parent
            output_file = os.path.join(output_dir, "normalized_mesh.obj")
            mesh_normalized.export(output_file)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error normalizing {obj_file}: {e}")
            error_count += 1
    
    print(f"Normalization complete!")
    print(f"  - Successfully normalized: {success_count}")
    print(f"  - Errors encountered: {error_count}")
    
    return success_count, error_count


if __name__ == "__main__":
    # Configuration
    dataset_path = "/scratch/cl927/nutritionverse-3d-new"
    
    print("Starting mesh analysis and normalization...")
    
    # 1. Analyze all meshes in the dataset
    df = analyze_dataset(dataset_path)
    
    # 2. Save analysis results
    df.to_csv("mesh_analysis_results.csv", index=False)
    print(f"Analysis results saved to mesh_analysis_results.csv")
    
    # 3. Create visualizations
    visualize_analysis(df, save_path="mesh_analysis_plots.png")
    
    # 4. Demonstrate normalization on a sample mesh
    if len(df) > 0:
        sample_mesh = df.iloc[0]['path']  # Use first mesh as example
        demonstrate_normalization(sample_mesh)
    
    # 5. Save normalized meshes
    print("\n" + "="*60)
    print("CREATING NORMALIZED MESHES...")
    print("="*60)
    success_count, error_count = save_normalized_meshes(dataset_path)
    
    print("\nAnalysis and normalization complete!")
    print(f"Normalized meshes saved as 'normalized_mesh.obj' in each food item directory")
    print("Normalization: Bounding box centered at origin, longest dimension spans [-0.5, 0.5]")
    print("\nFor benchmarking with Chamfer distance, you can:")
    print("1. Load the normalized_mesh.obj files as ground truth")
    print("2. Use sample_points_from_mesh() to sample points for comparison")
    print("3. Compare with SAM3D predictions using Chamfer distance")