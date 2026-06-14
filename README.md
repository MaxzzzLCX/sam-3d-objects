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

## Repository Structure

```
sam-3d-objects/
│
├── sam3d+vggt_method/          # Core method: Anisotropic Scaling + VGGT integration
│   ├── vggt_inference.py           # Run VGGT to produce point clouds from multiview images
│   ├── vggt_reconstruction.sh      # Shell script for VGGT reconstruction pipeline
│   ├── vggt_construct_scene.sh     # Scene construction from VGGT outputs
│   ├── segmentation.py             # Object segmentation utilities
│   ├── volume_estimation_vanilla_sam3d.py      # Baseline SAM3D volume estimation
│   ├── volume_estimation_anisotropic_scaling.py # Anisotropic Scaling volume estimation
│   └── evaluation.py / evaluation_utils.py     # Evaluation metrics
│
├── scripts_evaluation/         # Batch evaluation pipeline
│   ├── batch_generation_and_evaluation.py  # End-to-end batch generation + eval
│   ├── batch_fusion_and_evaluation.py      # Mesh fusion and evaluation
│   ├── volume_evaluation.py                # Volume metric computation
│   ├── chamfer_distance_evaluation.py      # Chamfer Distance metric
│   ├── vggt_preprocessing.py               # Preprocessing for VGGT inputs
│   ├── vggt_runner.py                      # VGGT inference runner
│   └── align.py / align_without_vggt.py   # Mesh alignment utilities
│
├── scripts_volume/             # Volume and projection evaluation utilities
│   ├── multiview_consistency.py    # Cross-view consistency measurement
│   ├── evaluate_projection.py      # Reprojection IoU evaluation
│   └── render_from_poses.py        # Render meshes from camera poses
│
├── vlm-baseline/               # Gemini 2.5 Pro VLM baseline
│   ├── gemini_minimal_multimodal.py    # Zero-shot volume estimation via Gemini
│   └── utils.py
│
├── real_dataset/               # RealFoodScenes dataset
│   └── real_data_multiview_volume_vggt/    # Multiview captures with GT volumes
│
├── results/                    # Evaluation results and outputs
│
├── notebook/                   # Original SAM3D demo notebooks
│   ├── demo_single_object.ipynb
│   ├── demo_multi_object.ipynb
│   └── multi_object_food.ipynb     # Food-specific multi-object demo
│
├── sam3d_objects/              # SAM3D model source (upstream, with patches)
├── patching/                   # Patches applied to upstream SAM3D
└── deprecated_scripts_benchmarking/  # Early-stage scripts (superseded)
```

---

## Setup

Follow the upstream SAM3D setup instructions in [doc/setup.md](doc/setup.md) to install dependencies and download model checkpoints.

Additional dependencies for the VGGT integration are managed in the [`/scratch/cl927/vggt`](../vggt/) directory.

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
