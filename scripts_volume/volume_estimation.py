import trimesh
import os
import numpy as np
import json
import open3d as o3d
import matplotlib.pyplot as plt

class RescalingVolumeEstimator:
    """
    Perform volume estimation by rescaling the generated 3D mesh to match scale of real object. 
    """

    def __init__(self, gen_method, folder_name, scene_dir, target_dimensions, gt_volume, view_blacklist, with_pointmaps=False):
        self.gen_method = gen_method
        self.folder_name = folder_name
        self.scene_dir = scene_dir
        self.target_dimensions = target_dimensions
        self.gt_volume = gt_volume
        self.view_blacklist = view_blacklist
        self.with_pointmaps = with_pointmaps
        if with_pointmaps:
            self.gen_folder = os.path.join(self.scene_dir, "generations_with_pointmaps")
        else:
            self.gen_folder = os.path.join(self.scene_dir, "generations_no_pointmaps")
    
    def find_target_dimension_permutation(self, mesh_span, raw_target_dimensions):
        """
        Find the permutation of the target dimensions that best matches the mesh span
        i.e. relative ordering of target dimensions should match mesh span, 
        e.g. both should be a > b > c, or a > c > b, etc. 
        """

        # Get the sorted indices of the mesh span and target dimensions
        mesh_span_indices = np.argsort(mesh_span)
        target_dim_indices = np.argsort(raw_target_dimensions)

        # Check if the relative ordering matches
        if np.array_equal(mesh_span_indices, target_dim_indices):
            return raw_target_dimensions  # No permutation needed
        else:
            # Permute the target dimensions to match the mesh span ordering
            permuted_target_dimensions = np.zeros_like(raw_target_dimensions)
            for i in range(len(raw_target_dimensions)):
                permuted_target_dimensions[mesh_span_indices[i]] = raw_target_dimensions[target_dim_indices[i]]
            return permuted_target_dimensions


    def rescale_mesh_and_estimate_volume(self, mesh_path, target_dimensions):
        """
        Args:
        - mesh_path: path to the generated 3D mesh file (e.g., .ply, .obj)
        - target_dimensions: the target dimensions to which the mesh should be scaled, list of 3 floats
        """

            
        mesh = trimesh.load(mesh_path)
        
        # Print the span of the mesh in each dimensions
        mesh_span = mesh.extents
        print(f"Original mesh span: {mesh_span}")

        # Find the correct permutation of target dimensions to match the mesh span ordering
        target_dimensions = self.find_target_dimension_permutation(mesh_span, target_dimensions)
        print(f"Permuted target dimensions: {target_dimensions}")

        # Calulate the scaling factor for each dimension to match the target dimensions
        scaling_factors = target_dimensions / mesh_span
        print(f"Scaling factors: {scaling_factors}")

        # Apply the scaling factors to the mesh
        mesh.apply_scale(scaling_factors)
        print(f"Mesh span after scaling: {mesh.extents}")

        # Voxelize the scaled mesh and fill it
        voxel_pitch = mesh.extents.max() / 100
        voxelized_mesh = mesh.voxelized(pitch=voxel_pitch)
        voxelized_mesh = voxelized_mesh.fill()
       
        return voxelized_mesh.volume

    def batch_estimate_volumes(self):
        """
        Do the volume estimation for all meshes in a folder
        """

        view_folders = sorted([f for f in os.listdir(self.gen_folder) if os.path.isdir(os.path.join(self.gen_folder, f))])
        errors = []
        volumes = []

        for view_folder in view_folders:
            print(f"view_folder: {view_folder}, type {type(view_folder)}")
            if view_folder in self.view_blacklist:
                print(f"Skipping {view_folder} as it is in the blacklist")
                continue

            mesh_path = os.path.join(self.gen_folder, view_folder, f"{view_folder}_food.ply")
            if os.path.exists(mesh_path):
                volume = self.rescale_mesh_and_estimate_volume(mesh_path, target_dimensions=self.target_dimensions)
                volume_error = abs(volume - self.gt_volume) / self.gt_volume * 100
                volumes.append(volume)
                errors.append(volume_error)
                print(f"Volume for {view_folder}: {volume}")
                print(f"% Error: {volume_error}")
        
        mean_error = np.average(errors)
        std_error = np.std(errors)
        print(f"==========================================")
        print(f"Volume error: {mean_error} +/- {std_error}")

        RESULTS[self.folder_name] = {
            "mean_error": mean_error,
            "std_error": std_error,
            "errors": errors,
            "predicted_target_dimensions": self.target_dimensions,
            "volume_predictions": volumes
        }
            

def post_process(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Do any post processing if needed, e.g. print in a nicer format, etc.
    object_mean_errors = []
    object_std_errors = []
    all_errors = []

    for folder_name, result in data.items():
        print(f"{folder_name}: {result['mean_error']} +/- {result['std_error']}") 
        mean_error = result['mean_error']
        std_error = result['std_error']
        errors = result['errors']

        object_mean_errors.append(mean_error)
        object_std_errors.append(std_error)
        all_errors.extend(errors)
    
    print(f"\nOverall mean error: {np.average(object_mean_errors)} +/- {np.std(object_mean_errors)}")
    print(f"Overall mean error (all errors): {np.average(all_errors)} +/- {np.std(all_errors)}")

    # Save in text file 
    output_txt_path = json_file.replace(".json", ".txt")
    with open(output_txt_path, "w") as f:
        f.write(f"Overall mean error: {np.average(object_mean_errors)} +/- {np.std(object_mean_errors)}\n")
        f.write(f"Overall mean error (all errors): {np.average(all_errors)} +/- {np.std(all_errors)}\n")

class VGGTScaleExtractor:
    """
    Extract the target dimensions from the VGGT pointcloud.
    This will then be used to rescale the generated mesh before volume estimation.
    """

    def __init__(self, scene_folder="/scratch/cl927/sam-3d-objects/scripts_volume/testing/potato_plate", plate_diameter=20.5):
        self.scene_folder = scene_folder
        self.plate_diameter = plate_diameter
        self.conversion_factor = None

    def extract_plate_plane(self):
        """
        Load the VGGT pointcloud of the plate. Extract the plane of the plate using RANSAC.
        Returns the plane equation coefficients [a, b, c, d] where ax + by + cz + d = 0
        """
        plate_pointmap_path = os.path.join(self.scene_folder, "sparse_plate_only_sam_unscaled_conf0.0", "points.ply")
        
        if not os.path.exists(plate_pointmap_path):
            raise FileNotFoundError(f"Plate pointcloud not found at {plate_pointmap_path}")
        
        # Load point cloud using Open3D
        pcd = o3d.io.read_point_cloud(plate_pointmap_path)
        print(f"Loaded point cloud with {len(pcd.points)} points")
        
        # Fit plane using RANSAC
        # Returns: plane_model (4 coefficients [a, b, c, d] where ax + by + cz + d = 0)
        #          inliers (list of indices of inlier points)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,  # Maximum distance from plane for a point to be considered an inlier
            ransac_n=3,               # Minimum number of points to fit plane
            num_iterations=1000        # Number of RANSAC iterations
        )
        
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        print(f"Normal vector: [{a:.4f}, {b:.4f}, {c:.4f}]")
        print(f"Number of inliers: {len(inliers)} ({len(inliers)/len(pcd.points)*100:.1f}%)")
        
        # Optionally visualize inliers vs outliers
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        inlier_cloud.paint_uniform_color([0, 1, 0])  # Green for inliers
        outlier_cloud.paint_uniform_color([1, 0, 0])  # Red for outliers
        
        # Save visualization
        vis_path = os.path.join(self.scene_folder, "plate_plane_fit_visualization.ply")
        o3d.io.write_point_cloud(vis_path, inlier_cloud + outlier_cloud)
        print(f"Saved plane fit visualization to {vis_path}")
        
        return plane_model, inliers

    def measure_plate_dimensions(self, plane_model, inliers, inliers_only=False, percentile=95, axis="pca", visualize=True):
        """
        Measure the dimensions of the plate using the inlier points that belong to the plane.
        
        Strategy:
        1. Use ONLY inliers (outliers already filtered by RANSAC)
        2. Project inlier points onto the fitted plane (3D → 3D points on plane)
        3. Convert to 2D coordinates within the plane using basis vectors
        4. Use percentile-based span measurements (more robust than bounding box)
        5. Optionally use PCA to find principal axes
        
        Args:
            plane_model: [a, b, c, d] coefficients of plane equation ax + by + cz + d = 0
            inliers: indices of inlier points
            percentile: percentile to use for span measurement (default 95 to handle remaining outliers)
            axis: "pca" or "coord" - whether to use PCA-based axes or original coordinate axes for measurement
            visualize: whether to save matplotlib visualization of 2D points and axes
        Returns:
            dimensions: dict with 'diameter' (for circular plates) or 'length', 'width' (for rectangular)
        """
        # Load the plate pointcloud again to get the inlier points
        plate_pointmap_path = os.path.join(self.scene_folder, "sparse_plate_only_sam_unscaled_conf0.0", "points.ply")
        pcd = o3d.io.read_point_cloud(plate_pointmap_path)
        
        # Extract inlier points only
        if inliers_only:
            inlier_pcd = pcd.select_by_index(inliers)
            points = np.asarray(inlier_pcd.points)
            print(f"\nMeasuring plate dimensions from {len(points)} inlier points...")
        else:
            points = np.asarray(pcd.points)
            print(f"\nMeasuring plate dimensions from ALL {len(points)} points (including outliers)...")
                
        # Extract plane parameters
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # STEP 1: Project 3D points onto the plane
        # Points are still in 3D, but now they all lie on the plane surface
        distances = (points @ normal + d).reshape(-1, 1)
        projected_points = points - distances * normal
        print(f"Projected {len(points)} points onto plane (still 3D coordinates)")

        # STEP 2: Create a 2D coordinate system ON the plane
        # basis1 and basis2 are two orthogonal vectors that lie IN the plane
        # They act as "x-axis" and "y-axis" for the plate's surface
        
        # First basis vector: perpendicular to normal and z-axis (if possible)
        if abs(normal[2]) < 0.99:  # Normal is not vertical
            basis1 = np.cross(normal, [0, 0, 1])
        else:  # Normal is nearly vertical, use x-axis instead
            basis1 = np.cross(normal, [1, 0, 0])
        basis1 = basis1 / np.linalg.norm(basis1)
        
        # Second basis vector: perpendicular to both normal and basis1
        basis2 = np.cross(normal, basis1)
        basis2 = basis2 / np.linalg.norm(basis2)
        
        print(f"Created 2D coordinate system on plane:")
        print(f"  basis1 (like x-axis on plate): {basis1}")
        print(f"  basis2 (like y-axis on plate): {basis2}")
        
        # STEP 3: Convert 3D projected points to 2D coordinates within the plane
        # This extracts how far each point is along basis1 and basis2 directions
        coords_2d = np.column_stack([
            projected_points @ basis1,  # Distance along basis1 (x-coordinate on plate)
            projected_points @ basis2   # Distance along basis2 (y-coordinate on plate)
        ])
        print(f"Converted to 2D coordinates: {coords_2d.shape}")
        
        if axis == "coord":
            
            # Method 1: Percentile-based measurement (robust to outliers)
            # Measure span along each axis using percentiles
            lower = (100 - percentile) / 2
            upper = percentile + lower
            
            x_min, x_max = np.percentile(coords_2d[:, 0], [lower, upper])
            y_min, y_max = np.percentile(coords_2d[:, 1], [lower, upper])
            
            span_x = x_max - x_min
            span_y = y_max - y_min
            
            print(f"Percentile-based measurements (using {percentile}th percentile):")
            print(f"  Span along basis1: {span_x:.4f}")
            print(f"  Span along basis2: {span_y:.4f}")
            
            results = {
                "span_1": span_x,
                "span_2": span_y,
            }
        
        elif axis == "pca":

            # Method 2: PCA-based measurement (finds principal directions)
            # This is useful for rectangular plates that might be rotated
            lower = (100 - percentile) / 2
            upper = percentile + lower
            
            mean = np.mean(coords_2d, axis=0)
            centered = coords_2d - mean
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            # Sort by eigenvalue (largest first)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Project onto principal components
            pca_coords = centered @ eigenvectors
            
            # Measure along principal axes using percentiles
            pca_x_min, pca_x_max = np.percentile(pca_coords[:, 0], [lower, upper])
            pca_y_min, pca_y_max = np.percentile(pca_coords[:, 1], [lower, upper])
            
            pca_span_1 = pca_x_max - pca_x_min
            pca_span_2 = pca_y_max - pca_y_min
            
            print(f"\nPCA-based measurements (along principal axes):")
            print(f"  Principal span 1: {pca_span_1:.4f}")
            print(f"  Principal span 2: {pca_span_2:.4f}")
            print(f"  Eigenvalue ratio: {eigenvalues[0]/eigenvalues[1]:.2f}")
            
            # Determine if plate is circular or rectangular
            # If spans are similar (ratio close to 1), it's likely circular
            pca_ratio = max(pca_span_1, pca_span_2) / min(pca_span_1, pca_span_2)
            
            results = {
                "pca_span_1": pca_span_1,
                "pca_span_2": pca_span_2,
                "is_likely_circular": pca_ratio < 1.1,
            }
            
            # Visualize if requested
            if visualize:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Plot 1: Points with coordinate axes (basis1, basis2)
                ax1.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.5, s=1, c='blue')
                ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='basis2 axis')
                ax1.axvline(x=0, color='g', linestyle='--', linewidth=2, label='basis1 axis')
                
                # Draw percentile boundaries for coordinate axes
                x_min_coord, x_max_coord = np.percentile(coords_2d[:, 0], [lower, upper])
                y_min_coord, y_max_coord = np.percentile(coords_2d[:, 1], [lower, upper])
                ax1.plot([x_min_coord, x_max_coord, x_max_coord, x_min_coord, x_min_coord],
                        [y_min_coord, y_min_coord, y_max_coord, y_max_coord, y_min_coord],
                        'k-', linewidth=2, label=f'{percentile}% bounds')
                
                ax1.set_xlabel('Coordinate along basis1 (like x on plate)')
                ax1.set_ylabel('Coordinate along basis2 (like y on plate)')
                ax1.set_title('2D Points with Coordinate Axes (basis1, basis2)')
                ax1.axis('equal')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Plot 2: Points with PCA axes
                ax2.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.5, s=1, c='blue')
                
                # Draw PCA principal axes
                origin = mean
                scale = max(pca_span_1, pca_span_2) * 0.6
                for i, (eigvec, eigval) in enumerate(zip(eigenvectors.T, eigenvalues)):
                    ax2.arrow(origin[0], origin[1], 
                             eigvec[0] * scale, eigvec[1] * scale,
                             head_width=0.3, head_length=0.2, 
                             fc=['red', 'green'][i], ec=['red', 'green'][i],
                             linewidth=3, label=f'PC{i+1} (eigenval={eigval:.2f})')
                
                # Draw percentile boundaries along PCA axes (in original coordinate system)
                # Transform PCA bounds back to original coordinates
                corners_pca = np.array([
                    [pca_x_min, pca_y_min],
                    [pca_x_max, pca_y_min],
                    [pca_x_max, pca_y_max],
                    [pca_x_min, pca_y_max],
                    [pca_x_min, pca_y_min]
                ])
                corners_orig = corners_pca @ eigenvectors.T + mean
                ax2.plot(corners_orig[:, 0], corners_orig[:, 1], 
                        'k-', linewidth=2, label=f'{percentile}% bounds (PCA)')
                
                ax2.set_xlabel('Coordinate along basis1 (like x on plate)')
                ax2.set_ylabel('Coordinate along basis2 (like y on plate)')
                ax2.set_title('2D Points with PCA Principal Axes')
                ax2.axis('equal')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                vis_path = os.path.join(self.scene_folder, f"plate_2d_axes_visualization_perc{percentile}.png")
                plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"\nSaved 2D visualization to {vis_path}")
        
        if results.get("is_likely_circular", False):
            # For circular plates, diameter is the average of the two spans
            diameter = (pca_span_1 + pca_span_2) / 2
            results["diameter"] = diameter
            print(f"\nPlate appears CIRCULAR with diameter: {diameter:.4f}")

            conversion_factor = self.plate_diameter / diameter # Conversion factor to map VGGT length to physical length in cm
            self.conversion_factor = conversion_factor
            print(f"Calculated conversion factor from VGGT to real-world scale: {conversion_factor:.4f} cm/unit")
        else:
            # For rectangular plates, use PCA spans as length and width
            results["length"] = max(pca_span_1, pca_span_2)
            results["width"] = min(pca_span_1, pca_span_2)
            print(f"\nPlate appears RECTANGULAR: {results['length']:.4f} x {results['width']:.4f}")
        
        return results


    def measure_food_dimensions(self, plane_model, percentile=95, visualize=True):
        """
        Load the food pointcloud. 
        Measure the span of the food along three axes:
        1. Height: distance along the plate's normal vector (perpendicular to plate)
        2. Length/Width: PCA-based principal axes on the horizontal projection (parallel to plate)
        
        Args:
            plane_model: [a, b, c, d] coefficients of plate plane equation
            percentile: percentile for robust span measurement
            visualize: whether to save matplotlib visualization
        
        Returns:
            dimensions: dict with 'height', 'length', 'width' in VGGT units and real-world cm
        """
        food_pointmap_path = os.path.join(self.scene_folder, "sparse_food_only_sam_unscaled_conf0.0", "points.ply")
        
        if not os.path.exists(food_pointmap_path):
            raise FileNotFoundError(f"Food pointcloud not found at {food_pointmap_path}")
        
        # Load food point cloud
        pcd = o3d.io.read_point_cloud(food_pointmap_path)
        points = np.asarray(pcd.points)
        print(f"\nLoaded food pointcloud with {len(points)} points")
        
        # Extract plane parameters
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        print(f"Using plate normal as height axis: {normal}")
        
        # STEP 1: Measure HEIGHT (distance along normal from plate)
        # Calculate signed distance from each point to the plate
        distances_to_plate = points @ normal + d
        
        # Use percentile for robust measurement
        lower = (100 - percentile) / 2
        upper = percentile + lower
        
        height_min, height_max = np.percentile(distances_to_plate, [lower, upper])
        height_span = height_max - height_min
        
        print(f"\nHeight measurement (along plate normal):")
        print(f"  Height span: {height_span:.4f} VGGT units")
        if self.conversion_factor is not None:
            height_cm = height_span * self.conversion_factor
            print(f"  Height span: {height_cm:.4f} cm")
        
        # STEP 2: Project food points onto plate (horizontal plane)
        projected_points = points - distances_to_plate.reshape(-1, 1) * normal
        
        # Create 2D coordinates on the plate plane
        # We need basis vectors, but we'll derive new ones from food PCA
        # First, create temporary basis vectors (same as plate measurement)
        if abs(normal[2]) < 0.99:
            temp_basis1 = np.cross(normal, [0, 0, 1])
        else:
            temp_basis1 = np.cross(normal, [1, 0, 0])
        temp_basis1 = temp_basis1 / np.linalg.norm(temp_basis1)
        
        temp_basis2 = np.cross(normal, temp_basis1)
        temp_basis2 = temp_basis2 / np.linalg.norm(temp_basis2)
        
        # Convert to 2D coordinates
        coords_2d = np.column_stack([
            projected_points @ temp_basis1,
            projected_points @ temp_basis2
        ])
        
        print(f"\nProjected {len(points)} food points onto plate plane")
        
        # STEP 3: Run PCA on the 2D projected food points
        mean = np.mean(coords_2d, axis=0)
        centered = coords_2d - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort by eigenvalue (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Project onto principal components
        pca_coords = centered @ eigenvectors
        
        # Measure along principal axes using percentiles
        pca_x_min, pca_x_max = np.percentile(pca_coords[:, 0], [lower, upper])
        pca_y_min, pca_y_max = np.percentile(pca_coords[:, 1], [lower, upper])
        
        length_span = pca_x_max - pca_x_min  # Along first principal component
        width_span = pca_y_max - pca_y_min   # Along second principal component
        
        print(f"\nHorizontal dimensions (PCA on food projection):")
        print(f"  Length (PC1): {length_span:.4f} VGGT units")
        print(f"  Width (PC2): {width_span:.4f} VGGT units")
        print(f"  Eigenvalue ratio: {eigenvalues[0]/eigenvalues[1]:.2f}")
        
        results = {
            "height_vggt": height_span,
            "length_vggt": length_span,
            "width_vggt": width_span,
            "eigenvalue_ratio": eigenvalues[0] / eigenvalues[1],
        }
        
        # Convert to real-world dimensions if conversion factor available
        if self.conversion_factor is not None:
            results["height_cm"] = height_span * self.conversion_factor
            results["length_cm"] = length_span * self.conversion_factor
            results["width_cm"] = width_span * self.conversion_factor
            results["dimensions_cm"] = [results["height_cm"], results["length_cm"], results["width_cm"]]
            
            print(f"\nReal-world dimensions (using conversion factor {self.conversion_factor:.4f}):")
            print(f"  Height: {results['height_cm']:.2f} cm")
            print(f"  Length: {results['length_cm']:.2f} cm")
            print(f"  Width: {results['width_cm']:.2f} cm")
        
        # STEP 4: Visualize
        if visualize:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Plot 1: Height distribution
            ax1 = axes[0]
            ax1.hist(distances_to_plate, bins=50, alpha=0.7, edgecolor='black')
            ax1.axvline(height_min, color='r', linestyle='--', linewidth=2, label=f'{lower:.1f}th percentile')
            ax1.axvline(height_max, color='r', linestyle='--', linewidth=2, label=f'{upper:.1f}th percentile')
            ax1.axvline(0, color='g', linestyle='-', linewidth=2, label='Plate plane')
            ax1.set_xlabel('Distance from plate (VGGT units)')
            ax1.set_ylabel('Number of points')
            ax1.set_title(f'Food Height Distribution\nSpan: {height_span:.4f} VGGT units')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: 2D projection with coordinate axes
            ax2 = axes[1]
            ax2.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.5, s=1, c='blue')
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_xlabel('Coordinate along temp_basis1')
            ax2.set_ylabel('Coordinate along temp_basis2')
            ax2.set_title('Food Projected onto Plate Plane\n(Before PCA)')
            ax2.axis('equal')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: 2D projection with PCA axes
            ax3 = axes[2]
            ax3.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.5, s=1, c='blue')
            
            # Draw PCA principal axes
            origin = mean
            scale = max(length_span, width_span) * 0.6
            for i, (eigvec, eigval) in enumerate(zip(eigenvectors.T, eigenvalues)):
                ax3.arrow(origin[0], origin[1], 
                         eigvec[0] * scale, eigvec[1] * scale,
                         head_width=0.3, head_length=0.2, 
                         fc=['red', 'green'][i], ec=['red', 'green'][i],
                         linewidth=3, label=f'PC{i+1} (eigenval={eigval:.2f})')
            
            # Draw percentile boundaries along PCA axes
            corners_pca = np.array([
                [pca_x_min, pca_y_min],
                [pca_x_max, pca_y_min],
                [pca_x_max, pca_y_max],
                [pca_x_min, pca_y_max],
                [pca_x_min, pca_y_min]
            ])
            corners_orig = corners_pca @ eigenvectors.T + mean
            ax3.plot(corners_orig[:, 0], corners_orig[:, 1], 
                    'k-', linewidth=2, label=f'{percentile}% bounds')
            
            ax3.set_xlabel('Coordinate along temp_basis1')
            ax3.set_ylabel('Coordinate along temp_basis2')
            ax3.set_title(f'Food with PCA Axes\nLength: {length_span:.4f}, Width: {width_span:.4f}')
            ax3.axis('equal')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            plt.tight_layout()
            vis_path = os.path.join(self.scene_folder, f"food_dimensions_visualization_perc{percentile}.png")
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\nSaved food visualization to {vis_path}")
        
        return results


def fit():
    extractor = VGGTScaleExtractor(
        scene_folder="/scratch/cl927/sam-3d-objects/scripts_volume/testing/potato_plate",
        plate_diameter=20.5  # Real plate diameter in cm
    )
    
    # Step 1: Extract plate plane and measure plate dimensions
    plane_model, inliers = extractor.extract_plate_plane()
    plate_dimensions = extractor.measure_plate_dimensions(plane_model, inliers, percentile=98)
    print(f"\nPlate measurements: {plate_dimensions}")
    
    # Step 2: Measure food dimensions using the plate plane and conversion factor
    food_dimensions = extractor.measure_food_dimensions(plane_model, percentile=98)
    target_dimensions = food_dimensions["dimensions_cm"]
    print(f"\nFood measurements (cm): {target_dimensions}")

    dataset = {
        "potato_plate": {
            "target_dimensions": [7.3, 5.8, 8.5],
            "gt_volume": 175,
            "black_list_views": ["3"] # Non watertight. Cannot fill
        },
    }

    # Rescale the SAM3D mesh and calculate volume
    volumeEstimator = RescalingVolumeEstimator(
        gen_method="sam3d_singleview_predictions", 
        folder_name="potato_plate",
        target_dimensions=target_dimensions,
        gt_volume = 175,
        view_blacklist=["3"],
        with_pointmaps=True
    )
    volumeEstimator.batch_estimate_volumes()

    print(f"\n\n*****RESULTS*****")
    print(RESULTS)





def main(results_dir=None, with_pointmaps=True, results_json_prefix="", percentile=98):
    # Dataset
    """
    dataset = {
        "potato_plate": {
            # "target_dimensions": [7.3, 5.8, 8.5],
            "gt_volume": 175,
            "black_list_views": ["3"] # Non watertight. Cannot fill
        },
        "potato_bowl": {
            # "target_dimensions": [7.3, 5.8, 8.5],
            "gt_volume": 175,
            "black_list_views": ["3"]
        },
        "egg_plate": {
            # "target_dimensions": [4.7, 4.2, 5.5],
            "gt_volume": 50,
            "black_list_views": []
        },
        "egg_bowl": {
            # "target_dimensions": [4.7, 4.2, 5.5],
            "gt_volume": 50,
            "black_list_views": ["5"]
        },
        # "pepper_plate": {
        #     "target_dimensions": [8, 8.2, 8.9],
        #     "gt_volume": 310,
        #     "black_list_views": [] # Non watertight. Cannot fill
        # },
        # "pepper_bowl": {
        #     "target_dimensions": [8, 8.2, 8.9],
        #     "gt_volume": 310,
        #     "black_list_views": [] # Non watertight. Cannot fill
        # },
        "orange_plate": {
            # "target_dimensions": [7, 7, 4.2],
            "gt_volume": 125,
            "black_list_views": [] # Non watertight. Cannot fill
        },
        "orange_bowl": {
            # "target_dimensions": [7, 7, 4.2],
            "gt_volume": 125,
            "black_list_views": [] # Non watertight. Cannot fill
        },
    }
    """
    dataset = [
        "potato_plate",
        "potato_bowl",
        "egg_plate",
        "egg_bowl",
        "orange_plate",
        "orange_bowl",
        "avocado_plate",
        "strawberry_plate",
        "strawberry_bowl"
    ]

    properties = {
        "food": {
            "potato":{
                "gt_volume": 175,
                "black_list_views": ["3"]
            },
            "egg":{
                "gt_volume": 50,
                "black_list_views": ["5"]
            },
            "orange":{
                "gt_volume": 125,
                "black_list_views": ["0"]
            },
            "avocado":{
                "gt_volume": 150,
                "black_list_views": []
            },
            "strawberry":{
                "gt_volume": 25,
                "black_list_views": []
            }
        },
        "container": {
            "plate":{
                "diameter": 20.5
            },
            "bowl":{
                "diameter": 16.8
            }
        }
        
    }

    # with_pointmaps = True
    # percentile = 98

    for data_index, folder_name in enumerate(dataset):
        print(f"Processing {folder_name}...")

        food_name, container_name = folder_name.split("_")
        scene_dir = f"/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt/{folder_name}"

        # target_dimensions = data["target_dimensions"]
        gt_volume = properties["food"][food_name]["gt_volume"]
        black_list_views = properties["food"][food_name]["black_list_views"]
        container_diameter = properties["container"][container_name]["diameter"]

        # (1) Measure scale from VGGT
        extractor = VGGTScaleExtractor(
            scene_folder=scene_dir,
            plate_diameter=container_diameter  # Real plate diameter in cm
        )
        plane_model, inliers = extractor.extract_plate_plane()
        plate_dimensions = extractor.measure_plate_dimensions(plane_model, inliers=None, inliers_only=False, percentile=percentile)
        food_dimensions = extractor.measure_food_dimensions(plane_model, percentile=percentile)
        target_dimensions = food_dimensions["dimensions_cm"]
        print(f"Measured target dimensions of food in {folder_name}: {target_dimensions} cm")

        volumeEstimator = RescalingVolumeEstimator(
            gen_method="sam3d_singleview_predictions", 
            folder_name=folder_name,
            scene_dir=scene_dir,
            target_dimensions=target_dimensions,
            gt_volume = gt_volume,
            view_blacklist=black_list_views,
            with_pointmaps=with_pointmaps
        )

        volumeEstimator.batch_estimate_volumes()

    print(f"\n\n*****RESULTS*****")
    print(RESULTS)

    json_file = os.path.join(results_dir, f"{results_json_prefix}_rescaling_volume_estimation_summary_{'with_pointmaps' if with_pointmaps else 'no_pointmaps'}_percentile{percentile}.json")
    with open(json_file, "w") as f:
        json.dump(RESULTS, f, indent=4)
    
    # rescaled_volume = volumeEstimator.rescale_mesh_and_estimate_volume(
    #     mesh_path="/scratch/cl927/sam-3d-objects/scripts_volume/testing/potato_plate/generations_with_pointmaps/0/0_food.ply",
    #     target_dimensions=[7.3, 5.8, 8.5]
    # )
    # print(f"Rescaled volume: {rescaled_volume}")

def gt_scale(results_dir=None, with_pointmaps=True, results_json_prefix="", percentile=98):
    # Dataset
    """
    dataset = {
        "potato_plate": {
            # "target_dimensions": [7.3, 5.8, 8.5],
            "gt_volume": 175,
            "black_list_views": ["3"] # Non watertight. Cannot fill
        },
        "potato_bowl": {
            # "target_dimensions": [7.3, 5.8, 8.5],
            "gt_volume": 175,
            "black_list_views": ["3"]
        },
        "egg_plate": {
            # "target_dimensions": [4.7, 4.2, 5.5],
            "gt_volume": 50,
            "black_list_views": []
        },
        "egg_bowl": {
            # "target_dimensions": [4.7, 4.2, 5.5],
            "gt_volume": 50,
            "black_list_views": ["5"]
        },
        # "pepper_plate": {
        #     "target_dimensions": [8, 8.2, 8.9],
        #     "gt_volume": 310,
        #     "black_list_views": [] # Non watertight. Cannot fill
        # },
        # "pepper_bowl": {
        #     "target_dimensions": [8, 8.2, 8.9],
        #     "gt_volume": 310,
        #     "black_list_views": [] # Non watertight. Cannot fill
        # },
        "orange_plate": {
            # "target_dimensions": [7, 7, 4.2],
            "gt_volume": 125,
            "black_list_views": [] # Non watertight. Cannot fill
        },
        "orange_bowl": {
            # "target_dimensions": [7, 7, 4.2],
            "gt_volume": 125,
            "black_list_views": [] # Non watertight. Cannot fill
        },
    }
    """
    dataset = [
        "potato_plate",
        "potato_bowl",
        "egg_plate",
        "egg_bowl",
        "orange_plate",
        "orange_bowl",
        "avocado_plate",
        "strawberry_plate",
        "strawberry_bowl"
    ]

    properties = {
        "food": {
            "potato":{
                "gt_volume": 175,
                "black_list_views": ["3"],
                "target_dimensions": [7.3, 5.8, 8.5]
            },
            "egg":{
                "gt_volume": 50,
                "black_list_views": ["5"],
                "target_dimensions": [4.7, 4.2, 5.5]
            },
            "orange":{
                "gt_volume": 125,
                "black_list_views": ["0"],
                "target_dimensions": [7, 7, 4.2]
            },
            "avocado":{
                "gt_volume": 150,
                "black_list_views": [],
                "target_dimensions": [9.5, 6.2, 6.2]
            },
            "strawberry":{
                "gt_volume": 25,
                "black_list_views": [],
                "target_dimensions": [4.9, 3.3, 3.5]
            }
        },
        "container": {
            "plate":{
                "diameter": 20.5
            },
            "bowl":{
                "diameter": 16.8
            }
        }
        
    }

    # with_pointmaps = True
    # percentile = 98

    for data_index, folder_name in enumerate(dataset):
        print(f"Processing {folder_name}...")

        food_name, container_name = folder_name.split("_")
        scene_dir = f"/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt/{folder_name}"

        # target_dimensions = data["target_dimensions"]
        gt_volume = properties["food"][food_name]["gt_volume"]
        black_list_views = properties["food"][food_name]["black_list_views"]
        container_diameter = properties["container"][container_name]["diameter"]
        target_dimensions = properties["food"][food_name]["target_dimensions"]


        volumeEstimator = RescalingVolumeEstimator(
            gen_method="sam3d_singleview_predictions", 
            folder_name=folder_name,
            scene_dir=scene_dir,
            target_dimensions=target_dimensions,
            gt_volume = gt_volume,
            view_blacklist=black_list_views,
            with_pointmaps=with_pointmaps
        )

        volumeEstimator.batch_estimate_volumes()

    print(f"\n\n*****RESULTS*****")
    print(RESULTS)

    json_file = os.path.join(results_dir, f"gt_scales_rescaling_volume_estimation_summary_{'with_pointmaps' if with_pointmaps else 'no_pointmaps'}_percentile{percentile}.json")
    with open(json_file, "w") as f:
        json.dump(RESULTS, f, indent=4)
    
    # rescaled_volume = volumeEstimator.rescale_mesh_and_estimate_volume(
    #     mesh_path="/scratch/cl927/sam-3d-objects/scripts_volume/testing/potato_plate/generations_with_pointmaps/0/0_food.ply",
    #     target_dimensions=[7.3, 5.8, 8.5]
    # )
    # print(f"Rescaled volume: {rescaled_volume}")
    
if __name__ == "__main__":

    RESULTS = {

    }

    results_dir = "/scratch/cl927/sam-3d-objects/results/20250305"
    results_json_prefix = "gt_scales"
    # SAM3D with pointmaps
    main(
        results_dir = results_dir,
        with_pointmaps=True,
        results_json_prefix = results_json_prefix,
        percentile=98
    )
    # SAM3D without pointmaps
    main(
        results_dir = results_dir,
        with_pointmaps=False,
        results_json_prefix = results_json_prefix,
        percentile=98
    )

    # fit()

    # post_process(
    #     json_file = os.path.join(results_dir, f"{results_json_prefix}_rescaling_volume_estimation_summary_no_pointmaps_percentile98.json")
    # )
    # post_process(
    #     json_file = os.path.join(results_dir, f"{results_json_prefix}_rescaling_volume_estimation_summary_with_pointmaps_percentile98.json")
    # )


    ## ==================================
    ## EXPERIMENTS TO USE GT VECTOR SCALE
    ## ==================================
    # gt_scale(
    #     results_dir = results_dir,
    #     with_pointmaps=True,
    #     results_json_prefix = results_json_prefix,
    #     percentile=98
    # )
    # gt_scale(
    #     results_dir = results_dir,
    #     with_pointmaps=False,
    #     results_json_prefix = results_json_prefix,
    #     percentile=98
    # )

    # post_process(
    #     json_file = os.path.join(results_dir, f"gt_scales_rescaling_volume_estimation_summary_no_pointmaps_percentile98.json")
    # )
    # post_process(
    #     json_file = os.path.join(results_dir, f"gt_scales_rescaling_volume_estimation_summary_with_pointmaps_percentile98.json")
    # )



