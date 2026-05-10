"""
This script contains extra quantitative and qualitative evaluation code for the SAM3D+VGGT method
in addition to volume estimation accuracy.
This includes:
- Visualization of anisotropically scaled meshes overlayed on VGGT pointmaps
- Chamfer distance between scaled meshes and GT pointmaps
- After aligning the meshes to GT pointmaps using ICP, show reprojection of the mesh v.s. pointmaps. 
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
from volume_estimation_anisotropic_scaling import VGGTScaleExtractor


def compute_chamfer_distance():
    # Placeholder for Chamfer distance computation
    raw_mesh_path = "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/strawberry_plate/generations_no_pointmaps/0/0_food.ply"
    scaler_json_path = "/scratch/cl927/sam-3d-objects/results/20250305/all_rescaling_volume_estimation_summary_no_pointmaps_percentile98.json"
    vggt_combined_path = "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/strawberry_plate/sparse_combined_sam_unscaled_conf0.0/points.ply"
    vggt_food_path = "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/strawberry_plate/sparse_food_only_sam_unscaled_conf0.0/points.ply"

    scale_extractor = VGGTScaleExtractor(
        scene_folder="/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/strawberry_plate",
        plate_diameter=20.5
    )
    


    
def find_target_dimension_permutation(mesh_span: np.ndarray, raw_target_dimensions: np.ndarray) -> np.ndarray:
    """
    Match target dimension ordering to mesh span ordering.
    """
    mesh_span_indices = np.argsort(mesh_span)
    target_dim_indices = np.argsort(raw_target_dimensions)

    if np.array_equal(mesh_span_indices, target_dim_indices):
        return raw_target_dimensions

    permuted_target_dimensions = np.zeros_like(raw_target_dimensions)
    for i in range(len(raw_target_dimensions)):
        permuted_target_dimensions[mesh_span_indices[i]] = raw_target_dimensions[target_dim_indices[i]]
    return permuted_target_dimensions


def percentile_bbox_center(points: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    lower = (100.0 - percentile) / 2.0
    upper = percentile + lower
    mins = np.percentile(points, lower, axis=0)
    maxs = np.percentile(points, upper, axis=0)
    return 0.5 * (mins + maxs)


def percentile_spans_along_axes(points: np.ndarray, axes: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Compute robust span of points along each axis in `axes` (columns).
    """
    lower = (100.0 - percentile) / 2.0
    upper = percentile + lower
    projected = points @ axes
    mins = np.percentile(projected, lower, axis=0)
    maxs = np.percentile(projected, upper, axis=0)
    return maxs - mins


def pca_axes(points: np.ndarray) -> np.ndarray:
    """
    Return a right-handed 3x3 axis basis as column vectors.
    """
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # descending order
    idx = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, idx]

    # Enforce right-handed basis
    axis0 = axes[:, 0] / np.linalg.norm(axes[:, 0])
    axis1 = axes[:, 1] - np.dot(axes[:, 1], axis0) * axis0
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = np.cross(axis0, axis1)
    axis2 = axis2 / np.linalg.norm(axis2)

    return np.stack([axis0, axis1, axis2], axis=1)


def downsample_points(points: np.ndarray, max_points: int = 20000, seed: int = 42) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def symmetric_nn_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    Symmetric mean nearest-neighbor distance between two point clouds.
    """
    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(points_a)
    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(points_b)

    d_ab = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b), dtype=np.float64)
    d_ba = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a), dtype=np.float64)
    return float(0.5 * (np.mean(d_ab) + np.mean(d_ba)))


def chamfer_distance_metrics(points_a: np.ndarray, points_b: np.ndarray) -> dict:
    """
    Symmetric Chamfer metrics between two point clouds.
    Returns both mean distance and mean squared distance variants.
    """
    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(points_a)
    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(points_b)

    d_ab = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b), dtype=np.float64)
    d_ba = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a), dtype=np.float64)

    return {
        "a_to_b_mean": float(np.mean(d_ab)),
        "b_to_a_mean": float(np.mean(d_ba)),
        "symmetric_mean": float(0.5 * (np.mean(d_ab) + np.mean(d_ba))),
        "a_to_b_mean_squared": float(np.mean(d_ab ** 2)),
        "b_to_a_mean_squared": float(np.mean(d_ba ** 2)),
        "symmetric_mean_squared": float(0.5 * (np.mean(d_ab ** 2) + np.mean(d_ba ** 2))),
    }


def sample_equal_points(points_a: np.ndarray, points_b: np.ndarray, n_points: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Randomly sample the same number of points from both sets for fair Chamfer comparison.
    """
    max_n = min(len(points_a), len(points_b), int(n_points))
    if max_n <= 0:
        raise RuntimeError("Cannot sample Chamfer points from empty point set.")

    rng = np.random.default_rng(seed)
    idx_a = rng.choice(len(points_a), size=max_n, replace=False)
    idx_b = rng.choice(len(points_b), size=max_n, replace=False)
    return points_a[idx_a], points_b[idx_b], max_n


def longest_pca_axis_span(points: np.ndarray, percentile: float = 99.0) -> float:
    """
    Robust longest span of a point cloud measured along its own PCA axes.
    """
    axes = pca_axes(points)
    spans = percentile_spans_along_axes(points, axes, percentile=percentile)
    return float(max(np.max(spans), 1e-12))


def add_normalized_chamfer_metrics(
    chamfer_metrics: dict,
    normalization_length: float,
    normalization_definition: str = "longest_span_along_sam3d_pca_axes",
) -> dict:
    """
    Add normalized Chamfer fields by dividing distance terms by L and squared-distance terms by L^2.
    """
    L = max(float(normalization_length), 1e-12)
    out = dict(chamfer_metrics)

    mean_keys = ["a_to_b_mean", "b_to_a_mean", "symmetric_mean"]
    sq_keys = ["a_to_b_mean_squared", "b_to_a_mean_squared", "symmetric_mean_squared"]

    for k in mean_keys:
        if k in out:
            out[f"{k}_normalized"] = float(out[k] / L)
    for k in sq_keys:
        if k in out:
            out[f"{k}_normalized"] = float(out[k] / (L ** 2))

    out["normalization_length"] = float(L)
    out["normalization_definition"] = normalization_definition
    return out


def save_pca_plane_reprojection_plot(
    mesh_points: np.ndarray,
    vggt_points: np.ndarray,
    pca_axes: np.ndarray,
    output_path: Path,
    title_prefix: str,
    max_points: int = 30000,
) -> None:
    """
    Reproject onto 3 planes normal to PCA axes and save a 1x3 plot.
    """
    mesh_plot = downsample_points(mesh_points, max_points=max_points, seed=42)
    vggt_plot = downsample_points(vggt_points, max_points=max_points, seed=123)

    # Coordinates in PCA basis; each axis column is a PCA direction.
    mesh_pca = mesh_plot @ pca_axes
    vggt_pca = vggt_plot @ pca_axes

    # For normal axis i, use the other two coordinates.
    plane_pairs = [(1, 2), (0, 2), (0, 1)]
    plane_names = ["normal=PC1", "normal=PC2", "normal=PC3"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (u, v), pname in zip(axes, plane_pairs, plane_names):
        ax.scatter(vggt_pca[:, u], vggt_pca[:, v], s=1, c="green", alpha=0.45, label="VGGT")
        ax.scatter(mesh_pca[:, u], mesh_pca[:, v], s=1, c="blue", alpha=0.45, label="SAM3D mesh")
        ax.set_xlabel(f"PC{u+1}")
        ax.set_ylabel(f"PC{v+1}")
        ax.set_title(f"{title_prefix} | {pname}")
        ax.axis("equal")
        ax.grid(True, alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def rasterize_occupancy(points_2d: np.ndarray, mins: np.ndarray, maxs: np.ndarray, grid_size: int) -> np.ndarray:
    occ = np.zeros((grid_size, grid_size), dtype=bool)
    denom = np.maximum(maxs - mins, 1e-12)
    normalized = (points_2d - mins) / denom
    normalized = np.clip(normalized, 0.0, 1.0)
    ix = (normalized[:, 0] * (grid_size - 1)).astype(np.int32)
    iy = (normalized[:, 1] * (grid_size - 1)).astype(np.int32)
    occ[iy, ix] = True
    return occ


def rasterized_occupancy_iou_on_pca_planes(
    mesh_points: np.ndarray,
    vggt_points: np.ndarray,
    pca_axes: np.ndarray,
    grid_size: int = 512,
    max_points: int = 50000,
    output_plot_path: Path | None = None,
    title_prefix: str = "",
) -> dict:
    """
    Compute 2D rasterized occupancy IoU on 3 PCA-normal planes.
    Optionally save occupancy overlays.
    """
    mesh_plot = downsample_points(mesh_points, max_points=max_points, seed=7)
    vggt_plot = downsample_points(vggt_points, max_points=max_points, seed=11)

    mesh_pca = mesh_plot @ pca_axes
    vggt_pca = vggt_plot @ pca_axes

    plane_pairs = [(1, 2), (0, 2), (0, 1)]
    plane_names = ["normal_pc1", "normal_pc2", "normal_pc3"]

    ious: dict[str, float] = {}
    overlay_images: list[np.ndarray] = []

    for (u, v), pname in zip(plane_pairs, plane_names):
        mesh_2d = mesh_pca[:, [u, v]]
        vggt_2d = vggt_pca[:, [u, v]]

        both = np.vstack([mesh_2d, vggt_2d])
        mins = np.min(both, axis=0)
        maxs = np.max(both, axis=0)

        # small padding to avoid edge clipping
        pad = 0.02 * np.maximum(maxs - mins, 1e-12)
        mins = mins - pad
        maxs = maxs + pad

        occ_mesh = rasterize_occupancy(mesh_2d, mins, maxs, grid_size=grid_size)
        occ_vggt = rasterize_occupancy(vggt_2d, mins, maxs, grid_size=grid_size)

        inter = np.logical_and(occ_mesh, occ_vggt).sum()
        union = np.logical_or(occ_mesh, occ_vggt).sum()
        iou = float(inter / union) if union > 0 else 0.0
        ious[pname] = iou

        # Blue = mesh, Green = vggt, Cyan = overlap
        rgb = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
        rgb[..., 2] = occ_mesh.astype(np.float32)
        rgb[..., 1] = occ_vggt.astype(np.float32)
        overlay_images.append(rgb)

    ious["mean_iou"] = float(np.mean([ious["normal_pc1"], ious["normal_pc2"], ious["normal_pc3"]]))
    ious["grid_size"] = int(grid_size)

    if output_plot_path is not None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for ax, img, pname in zip(axs, overlay_images, ["normal=PC1", "normal=PC2", "normal=PC3"]):
            ax.imshow(img, origin="lower")
            ax.set_title(f"{title_prefix} | {pname}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot_path, dpi=180)
        plt.close(fig)

    return ious


def choose_signs_by_pointcloud_distance(
    base_target_axes: np.ndarray,
    source_points_centered: np.ndarray,
    target_points_centered: np.ndarray,
    allow_reflection: bool = False,
    sample_points: int = 20000,
) -> tuple[np.ndarray, tuple[float, float, float], float, float]:
    """
    Choose axis signs by minimizing symmetric NN distance after transforming source points.

    Returns:
        transform, best_signs, best_score, determinant
    """
    src = downsample_points(source_points_centered, max_points=sample_points, seed=42)
    tgt = downsample_points(target_points_centered, max_points=sample_points, seed=123)

    best_score = np.inf
    best_transform = None
    best_signs = (1.0, 1.0, 1.0)
    best_det = 1.0

    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                candidate_axes = base_target_axes.copy()
                candidate_axes[:, 0] *= sx
                candidate_axes[:, 1] *= sy
                candidate_axes[:, 2] *= sz
                det = float(np.linalg.det(candidate_axes))

                if not allow_reflection and det < 0:
                    continue

                transformed_src = (candidate_axes @ src.T).T
                score = symmetric_nn_distance(transformed_src, tgt)

                if score < best_score:
                    best_score = score
                    best_transform = candidate_axes
                    best_signs = (sx, sy, sz)
                    best_det = det

    if best_transform is None:
        raise RuntimeError("No valid sign combination found for axis alignment.")

    return best_transform, best_signs, float(best_score), best_det


def assign_vggt_axes_to_mesh_xyz_by_span_order(
    source_spans_xyz: np.ndarray,
    target_axes: np.ndarray,
    target_spans_pca: np.ndarray,
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Determine axis correspondence by span rank only:
    - mesh axis with smallest span matches VGGT PCA axis with smallest span
    - ... middle -> middle
    - ... largest -> largest

    No search over axis permutations is performed.

    Returns:
        selected_axes_for_xyz, mesh_rank_order, vggt_rank_order
    """
    source_spans_xyz = np.asarray(source_spans_xyz, dtype=np.float64)
    target_spans_pca = np.asarray(target_spans_pca, dtype=np.float64)

    mesh_rank_order = np.argsort(source_spans_xyz).tolist()
    vggt_rank_order = np.argsort(target_spans_pca).tolist()

    selected_axes_for_xyz = np.zeros((3, 3), dtype=np.float64)
    for rank_idx in range(3):
        mesh_axis = mesh_rank_order[rank_idx]
        vggt_axis = vggt_rank_order[rank_idx]
        selected_axes_for_xyz[:, mesh_axis] = target_axes[:, vggt_axis]

    return selected_axes_for_xyz, mesh_rank_order, vggt_rank_order


def infer_view_key_from_mesh_path(raw_mesh_path: str) -> str | None:
    mesh_path = Path(raw_mesh_path)
    parent_name = mesh_path.parent.name
    if parent_name.isdigit():
        return parent_name

    stem = mesh_path.stem  # e.g., 0_food
    maybe_idx = stem.split("_")[0]
    if maybe_idx.isdigit():
        return maybe_idx
    return None


def load_uniform_scaling_from_results(results_json_path: str, view_key: str | None) -> tuple[float, str, float]:
    payload = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
    raw_results = payload.get("raw_result") or payload.get("raw_results")
    if not isinstance(raw_results, dict):
        raise RuntimeError(
            f"Expected 'raw_result' or 'raw_results' dict in {results_json_path}, but did not find one."
        )

    resolved_key = str(view_key) if view_key is not None else None
    if resolved_key is None:
        if len(raw_results) == 1:
            resolved_key = next(iter(raw_results.keys()))
        else:
            raise RuntimeError(
                "Could not infer view key from mesh path and JSON has multiple views. "
                "Please set uniform_metric_view_key."
            )

    if resolved_key not in raw_results:
        raise RuntimeError(
            f"View key '{resolved_key}' not found in {results_json_path}. "
            f"Available keys: {list(raw_results.keys())[:20]}"
        )

    entry = raw_results[resolved_key]
    factor = entry.get("metric_conversion_factor")
    if factor is None:
        raise RuntimeError(
            f"'metric_conversion_factor' missing for view '{resolved_key}' in {results_json_path}."
        )

    # Optional SAM3D per-object uniform scale (food is index 1)
    # In this pipeline, raw exported mesh extents correspond to "raw_spans".
    # To recover vanilla SAM3D object size, we should apply this food scale.
    food_uniform_scale = 1.0
    scales = entry.get("scales")
    if isinstance(scales, (list, tuple)) and len(scales) > 1:
        food_scale_vec = scales[1]
        if isinstance(food_scale_vec, (list, tuple)) and len(food_scale_vec) >= 3:
            food_uniform_scale = float(np.mean(np.asarray(food_scale_vec, dtype=np.float64)))

    return float(factor), resolved_key, float(food_uniform_scale)


def align_scaled_mesh_with_vggt_pointcloud(
    raw_mesh_path: str,
    scene_folder: str,
    plate_diameter_cm: float,
    output_dir: str,
    percentile: float = 99.0,
    scaling_space: str = "vggt",
    normalize_longest_axis: bool = False,
    allow_reflection: bool = False,
    uniform_metric_json_path: str | None = None,
    uniform_metric_view_key: str | None = None,
    chamfer_sample_points: int = 10000,
    reprojection_max_points: int = 30000,
    raster_grid_size: int = 512,
    raster_max_points: int = 50000,
    save_visualization: bool = True,
) -> dict:
    """
    1) Scale raw mesh anisotropically using VGGT-extracted food dimensions.
    2) Rotate scaled mesh so canonical XYZ aligns to VGGT food PCA axes.
    3) Center both mesh and VGGT point cloud via percentile bbox center at origin.
    4) Save artifacts for overlay visualization.
    5) Compute Chamfer distance between aligned mesh and aligned food VGGT points.
    6) Optionally run vanilla SAM3D uniform scaling overlay using metric_conversion_factor from JSON.

    Args:
        scaling_space:
            - "vggt": use VGGT food dimensions in VGGT units (best for direct overlay with unscaled VGGT points)
            - "cm": use centimeter dimensions and also scale VGGT points to cm before overlay
        normalize_longest_axis:
            If True, after alignment both clouds are scaled so longest robust axis is 1.
        allow_reflection:
            If True, also allow improper transform (det < 0) when choosing signs.
        uniform_metric_json_path:
            Optional path to voxelize results JSON containing raw_result/raw_results[*].metric_conversion_factor.
            If provided, an additional uniform-scaling overlay branch is generated.
        uniform_metric_view_key:
            Optional view key in JSON (e.g., "0"). If omitted, inferred from raw_mesh_path.
        chamfer_sample_points:
            Number of random points sampled from each cloud for Chamfer distance.
        reprojection_max_points:
            Max points to draw per cloud in reprojection visualization.
        raster_grid_size:
            Grid size for rasterized occupancy IoU on 2D projections.
        raster_max_points:
            Max points per cloud used before rasterizing occupancy.
        save_visualization:
            If False, skip saving PLY visualization artifacts. PNG and JSON outputs are still saved.
    """
    scene_folder_path = Path(scene_folder)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    view_key_for_naming = str(uniform_metric_view_key or infer_view_key_from_mesh_path(raw_mesh_path) or "na")

    def output_path(stem: str, ext: str) -> Path:
        return output_dir_path / f"{stem}_view{view_key_for_naming}.{ext}"

    # --- Extract dimensions from VGGT point cloud ---
    extractor = VGGTScaleExtractor(scene_folder=str(scene_folder_path), plate_diameter=plate_diameter_cm)
    plane_model, inliers = extractor.extract_plate_plane(visualize=False)
    extractor.measure_plate_dimensions(
        plane_model,
        inliers=inliers,
        inliers_only=False,
        percentile=percentile,
        visualize=False,
    )
    food_dimensions = extractor.measure_food_dimensions(plane_model, percentile=percentile, visualize=False)
    if scaling_space not in {"vggt", "cm"}:
        raise ValueError(f"Unsupported scaling_space={scaling_space}. Use 'vggt' or 'cm'.")

    # NOTE: for direct overlay with sparse_*_unscaled VGGT points, use VGGT units.
    target_dimensions_vggt = np.array(
        [food_dimensions["height_vggt"], food_dimensions["length_vggt"], food_dimensions["width_vggt"]],
        dtype=np.float64,
    )
    target_dimensions_cm = np.array(food_dimensions["dimensions_cm"], dtype=np.float64)
    target_dimensions_for_scaling = target_dimensions_vggt if scaling_space == "vggt" else target_dimensions_cm

    # --- Load and scale mesh ---
    mesh = trimesh.load(raw_mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Expected Trimesh at {raw_mesh_path}, got {type(mesh)}")

    mesh_span = mesh.extents.astype(np.float64)
    target_dims_permuted = find_target_dimension_permutation(mesh_span, target_dimensions_for_scaling)
    scaling_factors = target_dims_permuted / np.maximum(mesh_span, 1e-12)
    mesh.apply_scale(scaling_factors)

    # --- Load VGGT food point cloud ---
    vggt_food_path = scene_folder_path / "sparse_food_only_sam_unscaled_conf0.0" / "points.ply"
    if not vggt_food_path.exists():
        raise FileNotFoundError(f"VGGT food point cloud not found at {vggt_food_path}")

    vggt_pcd = o3d.io.read_point_cloud(str(vggt_food_path))
    vggt_points = np.asarray(vggt_pcd.points)
    if vggt_points.size == 0:
        raise RuntimeError(f"VGGT food point cloud is empty: {vggt_food_path}")

    # --- Load VGGT combined point cloud (food + plate) ---
    vggt_combined_path = scene_folder_path / "sparse_combined_sam_unscaled_conf0.0" / "points.ply"
    has_combined_cloud = vggt_combined_path.exists()
    vggt_combined_pcd = None
    vggt_combined_points = None
    if has_combined_cloud:
        vggt_combined_pcd = o3d.io.read_point_cloud(str(vggt_combined_path))
        vggt_combined_points = np.asarray(vggt_combined_pcd.points)
        if vggt_combined_points.size == 0:
            has_combined_cloud = False

    # If mesh is scaled in cm, convert VGGT points to cm as well for consistent overlay units.
    conversion_factor = float(extractor.conversion_factor) if extractor.conversion_factor is not None else None
    if scaling_space == "cm":
        if conversion_factor is None:
            raise RuntimeError("VGGT conversion_factor is not available, cannot convert points to cm.")
        vggt_points = vggt_points * conversion_factor
        if has_combined_cloud and vggt_combined_points is not None:
            vggt_combined_points = vggt_combined_points * conversion_factor

    # --- Translation: center 99% bbox at origin (both clouds) ---
    mesh_vertices = mesh.vertices.copy()
    mesh_center = percentile_bbox_center(mesh_vertices, percentile=percentile)
    vggt_center = percentile_bbox_center(vggt_points, percentile=percentile)

    mesh_vertices_centered = mesh_vertices - mesh_center
    vggt_points_centered = vggt_points - vggt_center
    if has_combined_cloud and vggt_combined_points is not None:
        # Apply EXACT same VGGT-side transform: unit conversion + same translation center.
        vggt_combined_points_centered = vggt_combined_points - vggt_center
    else:
        vggt_combined_points_centered = None

    # --- Rotation: align canonical mesh XYZ axes to VGGT PCA axes ---
    # Axis correspondence is determined by span rank (small/mid/large).
    # Then we only resolve axis sign flips.
    vggt_axes = pca_axes(vggt_points_centered)
    mesh_spans_xyz = percentile_spans_along_axes(mesh_vertices_centered, np.eye(3), percentile=percentile)
    vggt_spans_pca = percentile_spans_along_axes(vggt_points_centered, vggt_axes, percentile=percentile)
    selected_axes, mesh_rank_order, vggt_rank_order = assign_vggt_axes_to_mesh_xyz_by_span_order(
        source_spans_xyz=mesh_spans_xyz,
        target_axes=vggt_axes,
        target_spans_pca=vggt_spans_pca,
    )
    rotation, best_signs, sign_distance_score, transform_det = choose_signs_by_pointcloud_distance(
        base_target_axes=selected_axes,
        source_points_centered=mesh_vertices_centered,
        target_points_centered=vggt_points_centered,
        allow_reflection=allow_reflection,
    )

    mesh_vertices_aligned = (rotation @ mesh_vertices_centered.T).T

    # Optional post-process normalization for visualization-only comparison.
    normalization_factor = 1.0
    if normalize_longest_axis:
        mesh_aligned_spans = percentile_spans_along_axes(mesh_vertices_aligned, np.eye(3), percentile=percentile)
        vggt_aligned_spans = percentile_spans_along_axes(vggt_points_centered, np.eye(3), percentile=percentile)
        longest_axis = max(float(np.max(mesh_aligned_spans)), float(np.max(vggt_aligned_spans)), 1e-12)
        normalization_factor = 1.0 / longest_axis
        mesh_vertices_aligned = mesh_vertices_aligned * normalization_factor
        vggt_points_centered = vggt_points_centered * normalization_factor
        if vggt_combined_points_centered is not None:
            vggt_combined_points_centered = vggt_combined_points_centered * normalization_factor

    # Build transformed mesh
    aligned_mesh = mesh.copy()
    aligned_mesh.vertices = mesh_vertices_aligned

    # Build centered VGGT point cloud for overlay
    centered_vggt_pcd = o3d.geometry.PointCloud()
    centered_vggt_pcd.points = o3d.utility.Vector3dVector(vggt_points_centered)
    if vggt_pcd.has_colors():
        centered_vggt_pcd.colors = vggt_pcd.colors

    # Save outputs for visualization
    aligned_mesh_path = output_path("aligned_mesh", "ply")
    centered_vggt_path = output_path("centered_vggt_food_points", "ply")
    overlay_scene_path = output_path("overlay_mesh_plus_vggt_points", "ply")
    centered_vggt_combined_path = output_path("centered_vggt_combined_points", "ply")
    overlay_combined_scene_path = output_path("overlay_mesh_plus_vggt_combined_points", "ply")

    if save_visualization:
        aligned_mesh.export(aligned_mesh_path)
        o3d.io.write_point_cloud(str(centered_vggt_path), centered_vggt_pcd)

    # Save an overlay point cloud (mesh sampled points + vggt points)
    mesh_points = aligned_mesh.sample(min(200000, max(50000, len(vggt_points_centered))))
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(mesh_points)
    mesh_colors = np.zeros((mesh_points.shape[0], 3), dtype=np.float64)
    mesh_colors[:, 2] = 1.0  # blue mesh points
    mesh_pcd.colors = o3d.utility.Vector3dVector(mesh_colors)

    vggt_overlay = o3d.geometry.PointCloud()
    vggt_overlay.points = o3d.utility.Vector3dVector(vggt_points_centered)
    vggt_colors = np.zeros((vggt_points_centered.shape[0], 3), dtype=np.float64)
    vggt_colors[:, 1] = 1.0  # green vggt points
    vggt_overlay.colors = o3d.utility.Vector3dVector(vggt_colors)

    merged = mesh_pcd + vggt_overlay
    if save_visualization:
        o3d.io.write_point_cloud(str(overlay_scene_path), merged)

    # Save transformed combined cloud and mesh+combined overlay (keeping original combined colors if available)
    combined_outputs = {
        "centered_vggt_combined_points": None,
        "overlay_mesh_plus_vggt_combined_points": None,
        "combined_cloud_has_color": False,
    }
    if has_combined_cloud and vggt_combined_points_centered is not None and vggt_combined_pcd is not None:
        centered_vggt_combined_pcd = o3d.geometry.PointCloud()
        centered_vggt_combined_pcd.points = o3d.utility.Vector3dVector(vggt_combined_points_centered)
        if vggt_combined_pcd.has_colors():
            centered_vggt_combined_pcd.colors = vggt_combined_pcd.colors
            combined_outputs["combined_cloud_has_color"] = True

        if save_visualization:
            o3d.io.write_point_cloud(str(centered_vggt_combined_path), centered_vggt_combined_pcd)
            combined_outputs["centered_vggt_combined_points"] = str(centered_vggt_combined_path)

        # Overlay: keep mesh in red; keep combined cloud original colors if present
        mesh_overlay_for_combined = o3d.geometry.PointCloud()
        mesh_overlay_for_combined.points = o3d.utility.Vector3dVector(mesh_points)
        mesh_overlay_colors = np.zeros((mesh_points.shape[0], 3), dtype=np.float64)
        mesh_overlay_colors[:, 2] = 1.0
        mesh_overlay_for_combined.colors = o3d.utility.Vector3dVector(mesh_overlay_colors)

        if centered_vggt_combined_pcd.has_colors():
            merged_combined = mesh_overlay_for_combined + centered_vggt_combined_pcd
        else:
            combined_vis = o3d.geometry.PointCloud()
            combined_vis.points = o3d.utility.Vector3dVector(vggt_combined_points_centered)
            combined_colors = np.zeros((vggt_combined_points_centered.shape[0], 3), dtype=np.float64)
            combined_colors[:, 1] = 1.0
            combined_vis.colors = o3d.utility.Vector3dVector(combined_colors)
            merged_combined = mesh_overlay_for_combined + combined_vis

        if save_visualization:
            o3d.io.write_point_cloud(str(overlay_combined_scene_path), merged_combined)
            combined_outputs["overlay_mesh_plus_vggt_combined_points"] = str(overlay_combined_scene_path)

    # Chamfer distance between aligned mesh and aligned food VGGT cloud (anisotropic branch)
    mesh_ch, vggt_ch, chamfer_n = sample_equal_points(
        mesh_points,
        vggt_points_centered,
        n_points=chamfer_sample_points,
        seed=42,
    )
    chamfer_metrics = chamfer_distance_metrics(mesh_ch, vggt_ch)
    sam3d_longest_span_aniso = longest_pca_axis_span(mesh_points, percentile=percentile)
    chamfer_metrics = add_normalized_chamfer_metrics(
        chamfer_metrics,
        normalization_length=sam3d_longest_span_aniso,
        normalization_definition="longest_span_along_sam3d_pca_axes",
    )
    chamfer_metrics["n_points_per_set"] = chamfer_n

    reprojection_aniso_path = output_path("reprojection_anisotropic_pca_planes", "png")
    save_pca_plane_reprojection_plot(
        mesh_points=mesh_points,
        vggt_points=vggt_points_centered,
        pca_axes=vggt_axes,
        output_path=reprojection_aniso_path,
        title_prefix="Anisotropic",
        max_points=reprojection_max_points,
    )

    raster_aniso_plot_path = output_path("rasterized_occupancy_anisotropic_pca_planes", "png")
    raster_iou_aniso = rasterized_occupancy_iou_on_pca_planes(
        mesh_points=mesh_points,
        vggt_points=vggt_points_centered,
        pca_axes=vggt_axes,
        grid_size=raster_grid_size,
        max_points=raster_max_points,
        output_plot_path=raster_aniso_plot_path,
        title_prefix="Anisotropic occupancy",
    )

    # ==============================
    # Optional uniform scaling branch
    # ==============================
    uniform_outputs = {
        "enabled": False,
        "metric_json_path": uniform_metric_json_path,
        "metric_json_view_key": None,
        "metric_conversion_factor": None,
        "mesh_center": None,
        "mesh_spans_xyz_before_rotation": None,
        "selected_pca_signs_for_xyz": None,
        "sign_selection_symmetric_nn_distance": None,
        "final_transform_determinant": None,
        "rotation_matrix": None,
        "chamfer_distance": None,
        "rasterized_occupancy_iou_2d": None,
        "outputs": {
            "aligned_mesh_uniform": None,
            "overlay_mesh_uniform_plus_vggt_points": None,
            "overlay_mesh_uniform_plus_vggt_combined_points": None,
            "reprojection_uniform_pca_planes": None,
            "rasterized_occupancy_uniform_pca_planes": None,
        },
    }

    if uniform_metric_json_path is not None:
        inferred_key = uniform_metric_view_key or infer_view_key_from_mesh_path(raw_mesh_path)
        metric_conversion_factor, resolved_key, food_uniform_scale = load_uniform_scaling_from_results(
            uniform_metric_json_path,
            inferred_key,
        )

        # Load raw mesh again for independent uniform-scaling comparison.
        uniform_mesh = trimesh.load(raw_mesh_path, process=False)
        if not isinstance(uniform_mesh, trimesh.Trimesh):
            raise RuntimeError(f"Expected Trimesh at {raw_mesh_path}, got {type(uniform_mesh)}")

        # Vanilla SAM3D uniform scaling:
        # raw mesh -> apply SAM3D food uniform scale -> convert units.
        # - cm space: multiply by metric_conversion_factor (cm / mesh-unit)
        # - vggt space: first to cm, then cm->vggt via extractor conversion_factor
        if scaling_space == "cm":
            unit_scale = metric_conversion_factor
        else:
            if conversion_factor is None:
                raise RuntimeError("VGGT conversion_factor missing; cannot convert uniform branch to VGGT units.")
            unit_scale = metric_conversion_factor / conversion_factor

        total_uniform_scale = food_uniform_scale * unit_scale
        uniform_mesh.apply_scale(total_uniform_scale)

        uniform_vertices = uniform_mesh.vertices.copy()
        uniform_mesh_center = percentile_bbox_center(uniform_vertices, percentile=percentile)
        uniform_vertices_centered = uniform_vertices - uniform_mesh_center

        uniform_mesh_spans_xyz = percentile_spans_along_axes(
            uniform_vertices_centered,
            np.eye(3),
            percentile=percentile,
        )
        selected_axes_uniform, _, _ = assign_vggt_axes_to_mesh_xyz_by_span_order(
            source_spans_xyz=uniform_mesh_spans_xyz,
            target_axes=vggt_axes,
            target_spans_pca=vggt_spans_pca,
        )

        rotation_uniform, best_signs_uniform, sign_score_uniform, det_uniform = choose_signs_by_pointcloud_distance(
            base_target_axes=selected_axes_uniform,
            source_points_centered=uniform_vertices_centered,
            target_points_centered=vggt_points_centered,
            allow_reflection=allow_reflection,
        )
        uniform_vertices_aligned = (rotation_uniform @ uniform_vertices_centered.T).T

        if normalize_longest_axis:
            uniform_vertices_aligned = uniform_vertices_aligned * normalization_factor

        aligned_uniform_mesh = uniform_mesh.copy()
        aligned_uniform_mesh.vertices = uniform_vertices_aligned

        aligned_mesh_uniform_path = output_path("aligned_mesh_uniform", "ply")
        overlay_uniform_path = output_path("overlay_mesh_uniform_plus_vggt_points", "ply")
        overlay_uniform_combined_path = output_path("overlay_mesh_uniform_plus_vggt_combined_points", "ply")
        if save_visualization:
            aligned_uniform_mesh.export(aligned_mesh_uniform_path)

        uniform_mesh_points = aligned_uniform_mesh.sample(min(200000, max(50000, len(vggt_points_centered))))
        uniform_mesh_pcd = o3d.geometry.PointCloud()
        uniform_mesh_pcd.points = o3d.utility.Vector3dVector(uniform_mesh_points)
        uniform_mesh_colors = np.zeros((uniform_mesh_points.shape[0], 3), dtype=np.float64)
        uniform_mesh_colors[:, 2] = 1.0  # blue mesh points
        uniform_mesh_pcd.colors = o3d.utility.Vector3dVector(uniform_mesh_colors)

        vggt_overlay_uniform = o3d.geometry.PointCloud()
        vggt_overlay_uniform.points = o3d.utility.Vector3dVector(vggt_points_centered)
        vggt_overlay_uniform_colors = np.zeros((vggt_points_centered.shape[0], 3), dtype=np.float64)
        vggt_overlay_uniform_colors[:, 1] = 1.0  # green vggt points
        vggt_overlay_uniform.colors = o3d.utility.Vector3dVector(vggt_overlay_uniform_colors)

        merged_uniform = uniform_mesh_pcd + vggt_overlay_uniform
        if save_visualization:
            o3d.io.write_point_cloud(str(overlay_uniform_path), merged_uniform)

        # Combined cloud overlay for uniform branch
        if has_combined_cloud and vggt_combined_points_centered is not None:
            if combined_outputs["centered_vggt_combined_points"] is not None and centered_vggt_combined_pcd is not None:
                if centered_vggt_combined_pcd.has_colors():
                    merged_uniform_combined = uniform_mesh_pcd + centered_vggt_combined_pcd
                else:
                    combined_vis_uniform = o3d.geometry.PointCloud()
                    combined_vis_uniform.points = o3d.utility.Vector3dVector(vggt_combined_points_centered)
                    combined_vis_uniform_colors = np.zeros((vggt_combined_points_centered.shape[0], 3), dtype=np.float64)
                    combined_vis_uniform_colors[:, 1] = 1.0
                    combined_vis_uniform.colors = o3d.utility.Vector3dVector(combined_vis_uniform_colors)
                    merged_uniform_combined = uniform_mesh_pcd + combined_vis_uniform

                if save_visualization:
                    o3d.io.write_point_cloud(str(overlay_uniform_combined_path), merged_uniform_combined)

        uniform_mesh_ch, uniform_vggt_ch, uniform_ch_n = sample_equal_points(
            uniform_mesh_points,
            vggt_points_centered,
            n_points=chamfer_sample_points,
            seed=123,
        )
        chamfer_uniform = chamfer_distance_metrics(uniform_mesh_ch, uniform_vggt_ch)
        sam3d_longest_span_uniform = longest_pca_axis_span(uniform_mesh_points, percentile=percentile)
        chamfer_uniform = add_normalized_chamfer_metrics(
            chamfer_uniform,
            normalization_length=sam3d_longest_span_uniform,
            normalization_definition="longest_span_along_sam3d_pca_axes",
        )
        chamfer_uniform["n_points_per_set"] = uniform_ch_n

        reprojection_uniform_path = output_path("reprojection_uniform_pca_planes", "png")
        save_pca_plane_reprojection_plot(
            mesh_points=uniform_mesh_points,
            vggt_points=vggt_points_centered,
            pca_axes=vggt_axes,
            output_path=reprojection_uniform_path,
            title_prefix="Uniform",
            max_points=reprojection_max_points,
        )

        raster_uniform_plot_path = output_path("rasterized_occupancy_uniform_pca_planes", "png")
        raster_iou_uniform = rasterized_occupancy_iou_on_pca_planes(
            mesh_points=uniform_mesh_points,
            vggt_points=vggt_points_centered,
            pca_axes=vggt_axes,
            grid_size=raster_grid_size,
            max_points=raster_max_points,
            output_plot_path=raster_uniform_plot_path,
            title_prefix="Uniform occupancy",
        )

        uniform_outputs = {
            "enabled": True,
            "metric_json_path": uniform_metric_json_path,
            "metric_json_view_key": resolved_key,
            "metric_conversion_factor": metric_conversion_factor,
            "food_uniform_scale": food_uniform_scale,
            "unit_scale": unit_scale,
            "total_uniform_scale": total_uniform_scale,
            "mesh_center": uniform_mesh_center.tolist(),
            "mesh_spans_xyz_before_rotation": uniform_mesh_spans_xyz.tolist(),
            "selected_pca_signs_for_xyz": list(best_signs_uniform),
            "sign_selection_symmetric_nn_distance": sign_score_uniform,
            "final_transform_determinant": det_uniform,
            "rotation_matrix": rotation_uniform.tolist(),
            "chamfer_distance": chamfer_uniform,
            "rasterized_occupancy_iou_2d": raster_iou_uniform,
            "outputs": {
                "aligned_mesh_uniform": str(aligned_mesh_uniform_path) if save_visualization else None,
                "overlay_mesh_uniform_plus_vggt_points": str(overlay_uniform_path) if save_visualization else None,
                "overlay_mesh_uniform_plus_vggt_combined_points": (
                    str(overlay_uniform_combined_path)
                    if (save_visualization and overlay_uniform_combined_path.exists())
                    else None
                ),
                "reprojection_uniform_pca_planes": str(reprojection_uniform_path),
                "rasterized_occupancy_uniform_pca_planes": str(raster_uniform_plot_path),
            },
        }

    # Save minimal Chamfer summary JSON
    chamfer_summary = {
        "anisotropic_chamfer": chamfer_metrics,
        "uniform_chamfer": uniform_outputs["chamfer_distance"] if uniform_outputs.get("enabled") else None,
        "anisotropic_rasterized_occupancy_iou_2d": raster_iou_aniso,
        "uniform_rasterized_occupancy_iou_2d": uniform_outputs["rasterized_occupancy_iou_2d"] if uniform_outputs.get("enabled") else None,
    }
    chamfer_summary_path = output_path("chamfer_summary", "json")
    chamfer_summary_path.write_text(json.dumps(chamfer_summary, indent=2), encoding="utf-8")
    print(f"Saved Chamfer summary JSON to {chamfer_summary_path}")
    
    return {
        "raw_mesh_path": str(raw_mesh_path),
        "scene_folder": str(scene_folder_path),
        "scaling_space": scaling_space,
        "normalize_longest_axis": normalize_longest_axis,
        "allow_reflection": allow_reflection,
        "conversion_factor_vggt_to_cm": conversion_factor,
        "target_dimensions_vggt": target_dimensions_vggt.tolist(),
        "target_dimensions_cm": target_dimensions_cm.tolist(),
        "target_dimensions_used_for_scaling": target_dimensions_for_scaling.tolist(),
        "mesh_span_before_scaling": mesh_span.tolist(),
        "target_dimensions_permuted_for_scaling": target_dims_permuted.tolist(),
        "scaling_factors": scaling_factors.tolist(),
        "percentile": percentile,
        "mesh_center": mesh_center.tolist(),
        "vggt_center": vggt_center.tolist(),
        "mesh_spans_xyz_before_rotation": mesh_spans_xyz.tolist(),
        "vggt_spans_pca": vggt_spans_pca.tolist(),
        "mesh_axis_rank_order_small_to_large": mesh_rank_order,
        "vggt_pca_axis_rank_order_small_to_large": vggt_rank_order,
        "selected_pca_signs_for_xyz": list(best_signs),
        "sign_selection_symmetric_nn_distance": sign_distance_score,
        "final_transform_determinant": transform_det,
        "chamfer_distance": chamfer_metrics,
        "rasterized_occupancy_iou_2d": raster_iou_aniso,
        "uniform_scaling": uniform_outputs,
        "chamfer_summary_json": str(chamfer_summary_path),
        "normalization_factor": normalization_factor,
        "rotation_matrix": rotation.tolist(),
        "outputs": {
            "aligned_mesh": str(aligned_mesh_path) if save_visualization else None,
            "centered_vggt_food_points": str(centered_vggt_path) if save_visualization else None,
            "overlay_mesh_plus_vggt_points": str(overlay_scene_path) if save_visualization else None,
            "reprojection_anisotropic_pca_planes": str(reprojection_aniso_path),
            "rasterized_occupancy_anisotropic_pca_planes": str(raster_aniso_plot_path),
            "centered_vggt_combined_points": combined_outputs["centered_vggt_combined_points"],
            "overlay_mesh_plus_vggt_combined_points": combined_outputs["overlay_mesh_plus_vggt_combined_points"],
            "combined_cloud_has_color": combined_outputs["combined_cloud_has_color"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale and align raw mesh to VGGT point cloud for overlay visualization")
    parser.add_argument("--raw_mesh_path", required=True, help="Path to raw food mesh (e.g. .../0_food.ply)")
    parser.add_argument("--scene_folder", required=True, help="Scene folder containing sparse_food_only_sam_unscaled_conf0.0")
    parser.add_argument("--plate_diameter_cm", type=float, required=True, help="Real container diameter in cm")
    parser.add_argument("--output_dir", required=True, help="Directory to save aligned outputs")
    parser.add_argument("--percentile", type=float, default=99.0, help="Percentile for bbox centering (default: 99)")
    parser.add_argument(
        "--scaling_space",
        choices=["vggt", "cm"],
        default="vggt",
        help="Scaling unit space. Use 'vggt' for direct overlay with unscaled VGGT points (default).",
    )
    parser.add_argument(
        "--normalize_longest_axis",
        action="store_true",
        help="Normalize both aligned mesh and VGGT cloud by longest robust axis (visualization-only).",
    )
    parser.add_argument(
        "--allow_reflection",
        action="store_true",
        help="Allow det<0 transform during sign selection (can fix mirrored reconstructions).",
    )
    parser.add_argument(
        "--uniform_metric_json_path",
        default=None,
        help="Optional JSON path with raw_result/raw_results metric_conversion_factor for vanilla uniform scaling overlay.",
    )
    parser.add_argument(
        "--uniform_metric_view_key",
        default=None,
        help="Optional view key in uniform metric JSON (e.g., '0'). If omitted, inferred from raw mesh path.",
    )
    parser.add_argument(
        "--chamfer_sample_points",
        type=int,
        default=10000,
        help="Number of random points sampled from each cloud for Chamfer distance (default: 10000).",
    )
    parser.add_argument(
        "--reprojection_max_points",
        type=int,
        default=30000,
        help="Max points per cloud to visualize in PCA-plane reprojections (default: 30000).",
    )
    parser.add_argument(
        "--raster_grid_size",
        type=int,
        default=512,
        help="Grid size for rasterized occupancy IoU in PCA-plane reprojections (default: 512).",
    )
    parser.add_argument(
        "--raster_max_points",
        type=int,
        default=50000,
        help="Max points per cloud used for rasterized occupancy IoU (default: 50000).",
    )
    parser.add_argument(
        "--save_visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save PLY visualization artifacts (default: True). Use --no-save-visualization to disable.",
    )
    return parser.parse_args()


def run_evaluation_on_dataset():

    DATASET_ROOT = "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt"
    SAM3D_GEN_WITH_POINTMAP = False
    if SAM3D_GEN_WITH_POINTMAP:
        sam3d_gen_folder_name = "generations_with_pointmaps"
        json_file_name = "voxelize_volume_estimation_results_with_pointmaps.json"
        output_subfolder = "sam3d_with_pointmaps"
    else:
        sam3d_gen_folder_name = "generations_no_pointmaps"
        json_file_name = "voxelize_volume_estimation_results.json"
        output_subfolder = "sam3d_no_pointmaps"

    FOOD_FOLDER_NAMES = [
        "avocado_plate",
        "egg_plate", "egg_bowl",
        "strawberry_plate", "strawberry_bowl",
        "potato_plate", "potato_bowl",
        "orange_plate", "orange_bowl",
    ]
    
    for food_folder in FOOD_FOLDER_NAMES:
        for view_idx in range(6):
            raw_mesh_path = f"{DATASET_ROOT}/{food_folder}/{sam3d_gen_folder_name}/{view_idx}/{view_idx}_food.ply"
            scene_folder = f"{DATASET_ROOT}/{food_folder}"
            output_dir = f"/scratch/cl927/sam-3d-objects/sam3d+vggt_method/alignment_outputs/{output_subfolder}/{food_folder}_percentile99_view{view_idx}"
            uniform_metric_json_path = f"{DATASET_ROOT}/{food_folder}/{json_file_name}"

            if not(os.path.exists(raw_mesh_path)):
                print(f"Raw mesh not found at {raw_mesh_path}, skipping {food_folder} view {view_idx}.")
                continue
            
            print(f"Processing {food_folder} view {view_idx}...")
            result = align_scaled_mesh_with_vggt_pointcloud(
                raw_mesh_path=raw_mesh_path,
                scene_folder=scene_folder,
                plate_diameter_cm=20.5,
                output_dir=output_dir,
                percentile=99.0,
                scaling_space="vggt",
                normalize_longest_axis=False,
                allow_reflection=True,
                uniform_metric_json_path=uniform_metric_json_path,
                uniform_metric_view_key=str(view_idx),
                chamfer_sample_points=10000,
                reprojection_max_points=10000,
                raster_grid_size=64,
                raster_max_points=10000,
                save_visualization=False,
            )
            print(f"Finished processing {food_folder} view {view_idx}.")
            print(result)

def main():
    # args = parse_args()
    # Test with hardcoded arguments for now

    FOOD_FOLDER_NAME = "avocado_plate"
    VIEW_INDEX = "0"

    args = argparse.Namespace(
        raw_mesh_path=f"/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/{FOOD_FOLDER_NAME}/generations_no_pointmaps/{VIEW_INDEX}/{VIEW_INDEX}_food.ply",
        scene_folder=f"/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/{FOOD_FOLDER_NAME}",
        plate_diameter_cm=20.5,
        output_dir=f"/scratch/cl927/sam-3d-objects/sam3d+vggt_method/alignment_outputs/{FOOD_FOLDER_NAME}_percentile99",
        percentile=99.0,
        scaling_space="vggt",
        normalize_longest_axis=False,
        allow_reflection=True,
        uniform_metric_json_path=f"/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/{FOOD_FOLDER_NAME}/voxelize_volume_estimation_results.json",
        uniform_metric_view_key=None,
        chamfer_sample_points=10000,
        reprojection_max_points=10000,
        raster_grid_size=64,
        raster_max_points=10000,
        save_visualization=True,
    )

    result = align_scaled_mesh_with_vggt_pointcloud(
        raw_mesh_path=args.raw_mesh_path,
        scene_folder=args.scene_folder,
        plate_diameter_cm=args.plate_diameter_cm,
        output_dir=args.output_dir,
        percentile=args.percentile,
        scaling_space=args.scaling_space,
        normalize_longest_axis=args.normalize_longest_axis,
        allow_reflection=args.allow_reflection,
        uniform_metric_json_path=args.uniform_metric_json_path,
        uniform_metric_view_key=args.uniform_metric_view_key,
        chamfer_sample_points=args.chamfer_sample_points,
        reprojection_max_points=args.reprojection_max_points,
        raster_grid_size=args.raster_grid_size,
        raster_max_points=args.raster_max_points,
        save_visualization=args.save_visualization,
    )
    print("Alignment finished.")
    print(result)


if __name__ == "__main__":
    run_evaluation_on_dataset()
    # main()
