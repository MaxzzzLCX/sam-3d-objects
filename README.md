# Robust 3D Food Volume Estimation using Generative and Multiview Computer Vision

**Chenxu (Max) Lyu**, Christ's College, University of Cambridge  
Supervisor: Prof. Roberto Cipolla  
May 2026

> This repository is forked from [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) (Meta). See [SAM3D_README.md](SAM3D_README.md) for the original documentation.

---

## Overview

Dietary tracking is a highly effective behavioral intervention for nutritional assessments and health monitoring, yet manual portion size estimation remains fundamentally unreliable. While computer vision has largely solved 2D food recognition, **3D metric volume estimation** remains a critical challenge.

This project addresses the inherent trade-off between the strict reliability of explicit multi-view reconstruction and the robust, watertight shape priors of implicit 3D generative models. The core contributions are three novel, lightweight strategies for injecting sparse multi-view geometric constraints into single-view 3D generation pipelines:

1. **VGGT Point Map Conditioning** — replaces the default monocular depth prior (MoGe) in SAM3D with multi-view VGGT point maps.
2. **Anisotropic Scaling** — corrects aspect-ratio distortions in generated meshes by independently scaling along orthogonal PCA-derived axes using sparse-view VGGT point clouds. Reduced SAM3D volume estimation error from **24.58% → 15.21%**.
3. **Alternating Conditions** — alternates reference viewpoints during the rectified flow sampling process in TRELLIS. Reduced volume estimation error on Toys4k from **37.78% → 30.27%** and nearly halved Chamfer Distance.

Baselines evaluated include Apple ObjectCapture (dense multiview), VGGT + Poisson Surface Reconstruction (sparse multiview), and Gemini 2.5 Pro (zero-shot VLM reasoning, >42% error).

---

## Strategies

### 1. VGGT Point Map Conditioning

SAM3D's default pipeline uses MoGe — a monocular depth estimator — to produce a per-pixel 3D point map from a single image, which conditions the diffusion process. The relevant section of [`checkpoints/hf/pipeline.yaml`](checkpoints/hf/pipeline.yaml) is:

```yaml
depth_model:
  _target_: sam3d_objects.pipeline.depth_models.moge.MoGe
  model:
    _target_: moge.model.v1.MoGeModel.from_pretrained
    pretrained_model_name_or_path: Ruicheng/moge-vitl
```

This strategy replaces MoGe with point maps produced by VGGT from sparse multiview images. VGGT runs over a small set of input views ([`sam3d+vggt_method/vggt_inference.py`](sam3d+vggt_method/vggt_inference.py)) and outputs per-pixel 3D point maps (`point_map_{i}.npy`, shape `(518, 518, 3)`). These are then passed directly into SAM3D's conditioning pipeline by switching to [`checkpoints/hf/pipeline_no_depth.yaml`](checkpoints/hf/pipeline_no_depth.yaml), which sets:

```yaml
depth_model: null
```

This disables MoGe entirely; the externally-computed VGGT point map is fed in its place. The motivation is that VGGT's multi-view geometry should provide a more metrically-grounded depth prior than single-image MoGe.

**To run:**
```bash
# Step 1: Run VGGT on multiview images to produce per-view point maps
bash sam3d+vggt_method/vggt_reconstruction.sh   # calls vggt_inference.py --generation

# Step 2: Generate SAM3D meshes conditioned on VGGT point maps
#         (uses pipeline_no_depth.yaml; see volume_estimation_vanilla_sam3d.py)
python sam3d+vggt_method/volume_estimation_vanilla_sam3d.py  # set with_pointmaps=True
```

---

### 2. Anisotropic Scaling

Single-view generative models like SAM3D produce watertight meshes but frequently distort the aspect ratio of the reconstructed object — the mesh may be correctly shaped but squashed or stretched along one or more axes. This strategy corrects those distortions post-generation without any model retraining.

The approach ([`sam3d+vggt_method/volume_estimation_anisotropic_scaling.py`](sam3d+vggt_method/volume_estimation_anisotropic_scaling.py)):

1. Run VGGT on sparse multiview images to obtain a metric point cloud of the target object.
2. Compute the object's physical extents along each axis by projecting the point cloud onto PCA-derived orthogonal axes.
3. Compute per-axis scaling factors as the ratio between the VGGT-derived physical dimensions and the corresponding span of the generated mesh (`scaling_factors = target_dimensions / mesh_span`).
4. Apply independent scale factors along each axis (`mesh.apply_scale(scaling_factors)`), then voxelize the corrected mesh to estimate volume.

A permutation step aligns the ordering of the VGGT dimensions to the mesh axes before scaling, so that the largest VGGT extent is matched to the largest mesh axis, etc. This reduced the baseline SAM3D volume estimation error from **24.58% → 15.21%** on RealFoodScenes.

**To run (4-step pipeline):**
```bash
# Step 1: Run VGGT per-view inference to produce dense point maps
bash sam3d+vggt_method/vggt_reconstruction.sh       # vggt_inference.py --generation
#   Output: <scene_dir>/pointmaps/point_map_{i}.npy

# Step 2: Construct sparse COLMAP scenes from point maps (separate food + plate point clouds)
bash sam3d+vggt_method/vggt_construct_scene.sh      # vggt_inference.py --mask --construct_scenes
#   Output: <scene_dir>/sparse_food_only_sam_unscaled_conf0.0/points.ply
#           <scene_dir>/sparse_plate_only_sam_unscaled_conf0.0/points.ply

# Step 3: Generate SAM3D meshes for each view (vanilla or with VGGT pointmap conditioning)
python sam3d+vggt_method/volume_estimation_vanilla_sam3d.py
#   Output: <scene_dir>/generations_no_pointmaps/  or  generations_with_pointmaps/

# Step 4: Apply anisotropic scaling and estimate volume
python sam3d+vggt_method/volume_estimation_anisotropic_scaling.py
#   - VGGTScaleExtractor reads the food + plate point clouds from Step 2
#   - Uses the known plate diameter to convert VGGT units → real-world cm
#   - RescalingVolumeEstimator applies per-axis scaling to SAM3D meshes from Step 3
#   Output: results JSON with per-scene volume predictions and errors
```

---

### 3. Alternating Conditions (TRELLIS)

This strategy targets TRELLIS, a rectified flow-based 3D generative model. TRELLIS conditions its flow on a single reference image, which means geometric information from other viewpoints is ignored. The Alternating Conditions mechanism injects sparse multi-view constraints directly into the sampling process without any retraining.

During flow sampling, the reference conditioning image is alternated across different viewpoints at each denoising step rather than being fixed to a single view. This forces the flow trajectory to remain consistent with geometry observed from multiple angles, improving cross-view structural coherence. The result on the synthetic Toys4k dataset was a volume estimation error reduction from **37.78% → 30.27%** and nearly halved Chamfer Distance.

This strategy is implemented in the [`trellis/`](trellis/) submodule (mirrored from [github.com/MaxzzzLCX/TRELLIS](https://github.com/MaxzzzLCX/TRELLIS)). After cloning this repo, run:

```bash
git submodule update --init
```

Note: `trellis/scripts_generation/` imports utilities from `scripts_evaluation/` in this repo. Both must be present at the same parent path for cross-repo imports to resolve correctly.

---

## Repository Structure

The active code lives in four directories. `scripts/` and `deprecated_scripts_benchmarking/` are legacy and no longer in use.

```
sam-3d-objects/
│
├── sam3d+vggt_method/              # Core method implementations (start here)
│   ├── vggt_inference.py               # ENTRY POINT: run VGGT on multiview images to produce point maps
│   ├── vggt_reconstruction.sh          # Shell wrapper: runs vggt_inference.py (generation step)
│   ├── vggt_construct_scene.sh         # Shell wrapper: runs vggt_inference.py (scene construction step)
│   ├── volume_estimation_vanilla_sam3d.py       # ENTRY POINT: baseline SAM3D volume estimation (no scaling)
│   ├── volume_estimation_anisotropic_scaling.py # ENTRY POINT: anisotropic scaling volume estimation
│   ├── evaluation.py                   # ENTRY POINT: Chamfer distance + ICP alignment evaluation
│   ├── evaluation_utils.py             # Aggregates JSON evaluation results across scenes
│   └── segmentation.py                 # SAM-based object segmentation utilities
│
├── scripts_evaluation/             # Batch generation and evaluation pipeline
│   ├── batch_generation_and_evaluation.py  # ENTRY POINT: end-to-end SAM3D generation + evaluation
│   ├── batch_evaluation.py              # ENTRY POINT: evaluate pre-generated SAM3D outputs
│   ├── volume_evaluation.py             # ENTRY POINT: compute volume metrics from meshes
│   ├── vggt_runner.py                   # VGGTRunner class — isolated subprocess wrapper for VGGT
│   ├── vggt_preprocessing.py            # Resize images to 518×518 before VGGT inference
│   ├── generate_sam3d_multiview.py      # generate_sam3d_outputs() — called by batch scripts
│   ├── chamfer_distance_evaluation.py   # Chamfer distance metric (PyTorch3D)
│   └── align_without_vggt.py            # two_view_fusion() — called by batch_fusion_and_evaluation.py
│
├── scripts_volume/                 # Volume and reprojection analysis
│   ├── render_from_poses.py             # ENTRY POINT: render mesh from camera poses, compute reprojection IoU
│   ├── multiview_consistency.py         # ENTRY POINT: measure cross-view consistency of SAM3D outputs
│   └── evaluate_projection.py           # IoU computation — imported by render_from_poses.py
│
├── vlm-baseline/                   # Gemini 2.5 Pro zero-shot baseline
│   └── gemini_minimal_multimodal.py     # ENTRY POINT: query Gemini for volume estimates
│
├── real_dataset/                   # RealFoodScenes — see dataset section below
│   └── real_data_multiview_volume_vggt/ # Multiview captures with ground-truth volumes
│
├── checkpoints/hf/                 # SAM3D model weights and pipeline configs
│   ├── pipeline.yaml                    # Default config: uses MoGe as depth model
│   └── pipeline_no_depth.yaml           # Config for VGGT conditioning: depth_model set to null
│
├── notebook/                       # SAM3D demo notebooks (upstream + food-specific)
│   └── multi_object_food.ipynb          # Multi-object food reconstruction demo
│
├── sam3d_objects/                  # Upstream SAM3D model source (with patches from patching/)
│
├── vggt/                           # Submodule: MaxzzzLCX/vggt (editable VGGT package)
│
└── trellis/                        # Submodule: MaxzzzLCX/TRELLIS (Alternating Conditions)
```

---

## Dataset

**RealFoodScenes** is a custom dataset of real-world food scenes captured for this project, with ground-truth volumes measured by water displacement. It contains multiview RGB images of 9 food items (potato, egg, orange, avocado, strawberry) each photographed on a standard plate and bowl.

The dataset is hosted on Hugging Face: [huggingface.co/datasets/Max-Lyu/RealFoodScenes](https://huggingface.co/datasets/Max-Lyu/RealFoodScenes)

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Max-Lyu/RealFoodScenes", repo_type="dataset", local_dir="real_dataset/RealFoodScenes")
```

---

## Setup

Follow the upstream SAM3D setup instructions in [doc/setup.md](doc/setup.md) to install dependencies and download model checkpoints.

After cloning, initialise the submodules to pull in the VGGT and TRELLIS repos:

```bash
git submodule update --init
```

Then install VGGT as an editable package into the `vggt` conda environment:

```bash
conda activate vggt
pip install -e vggt/
```

---

## Citing

If you use this work, please also cite the upstream SAM 3D Objects paper:

```bibtex
@article{sam3dteam2025sam3d3dfyimages,
  title={SAM 3D: 3Dfy Anything in Images},
  author={SAM 3D Team and Xingyu Chen and Fu-Jen Chu and Pierre Gleize and Kevin J Liang and Alexander Sax and Hao Tang and Weiyao Wang and others},
  year={2025},
  eprint={2511.16624},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
}
```
