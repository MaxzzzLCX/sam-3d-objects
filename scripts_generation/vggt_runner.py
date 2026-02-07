#!/usr/bin/env python3
"""
VGGT Runner for SAM3D Integration

This script provides a clean interface to run VGGT inference from the SAM3D environment.
It handles environment switching, batch processing, and provides optimized integration
with the SAM3D pipeline.
"""

import subprocess
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union


class VGGTRunner:
    """
    Interface to run VGGT inference with environment isolation and batch processing
    """
    
    def __init__(self, 
                 vggt_root: str = "/scratch/cl927/vggt",
                 conda_path: str = "/scratch/cl927/miniconda3",
                 vggt_env: str = "vggt"):
        """
        Initialize VGGT runner
        
        Args:
            vggt_root: Path to VGGT repository root
            conda_path: Path to miniconda/conda installation
            vggt_env: Name of VGGT conda environment
        """
        self.vggt_root = Path(vggt_root)
        self.conda_path = Path(conda_path)
        self.vggt_env = vggt_env
        self.vggt_script = self.vggt_root / "test_scripts" / "VGGT_interface_SAM3D.py"
        
        # Validate paths
        if not self.vggt_root.exists():
            raise FileNotFoundError(f"VGGT root not found: {self.vggt_root}")
        if not self.vggt_script.exists():
            raise FileNotFoundError(f"VGGT script not found: {self.vggt_script}")
        
        print(f"VGGT Runner initialized:")
        print(f"  VGGT root: {self.vggt_root}")
        print(f"  VGGT script: {self.vggt_script}")
        print(f"  Conda environment: {self.vggt_env}")
    
    def _build_vggt_command(self, scene_dir: str, **kwargs) -> List[str]:
        """
        Build the command to run VGGT_COLMAP.py with proper environment setup
        
        Args:
            scene_dir: Path to scene directory
            **kwargs: Additional parameters for VGGT
            
        Returns:
            List of command components
        """
        # Default parameters
        default_params = {
            'use_masks': True,
            'mask_dir': 'masks',
            'image_dir': 'resized_images',
            'self_mask': False,
            'select_images': None,
            'conf_thres_value': 0.0,
            'seed': 42,
            'scale_pointcloud': False
        }
        
        # Update with user parameters
        params = {**default_params, **kwargs}
        
        # Build conda activation and VGGT command
        conda_activate = f"source {self.conda_path}/etc/profile.d/conda.sh && conda activate {self.vggt_env}"
        vggt_setup = f"cd {self.vggt_root} && export PYTHONPATH=\"{self.vggt_root}:${{PYTHONPATH:-}}\""
        
        # Build VGGT arguments
        vggt_args = [
            "python", str(self.vggt_script),
            "--scene_dir", str(scene_dir),
            "--image_dir", params['image_dir'],
            "--conf_thres_value", str(params['conf_thres_value']),
            "--seed", str(params['seed'])
        ]
        
        if params['use_masks']:
            vggt_args.extend(["--mask", "--mask_dir", params['mask_dir']])
            if params['self_mask']:
                vggt_args.append("--self_mask")
        
        if params['select_images']:
            vggt_args.extend(["--select_images", params['select_images']])
        
        # Combine into full command
        full_command = f"{conda_activate} && {vggt_setup} && {' '.join(vggt_args)}"
        
        return ['bash', '-c', full_command]
    
    def run_single_scene(self, scene_dir: Union[str, Path], **kwargs) -> Dict:
        """
        Run VGGT inference on a single scene
        
        Args:
            scene_dir: Path to scene directory containing images and optionally masks
            **kwargs: Additional parameters for VGGT (image_dir, mask_dir, etc.)
            
        Returns:
            Dict with status, outputs, and timing information
        """
        scene_dir = Path(scene_dir)
        
        if not scene_dir.exists():
            return {
                'success': False,
                'error': f"Scene directory not found: {scene_dir}",
                'scene_dir': str(scene_dir),
                'outputs': {}
            }
        
        # Check for required inputs
        image_dir_name = kwargs.get('image_dir', 'resized_images')
        images_dir = scene_dir / image_dir_name
        if not images_dir.exists():
            return {
                'success': False,
                'error': f"Images directory not found: {images_dir} (looking for '{image_dir_name}')",
                'scene_dir': str(scene_dir),
                'outputs': {}
            }
        
        print(f"Running VGGT on scene: {scene_dir.name}")
        
        # Build command
        command = self._build_vggt_command(str(scene_dir), **kwargs)
        
        # Run VGGT
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            runtime = time.time() - start_time
            
            if result.returncode == 0:
                # Check for expected outputs
                expected_outputs = self._check_vggt_outputs(scene_dir, **kwargs)
                
                return {
                    'success': True,
                    'scene_dir': str(scene_dir),
                    'runtime_seconds': runtime,
                    'outputs': expected_outputs,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'success': False,
                    'error': f"VGGT failed with return code {result.returncode}",
                    'scene_dir': str(scene_dir),
                    'runtime_seconds': runtime,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'outputs': {}
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': "VGGT timed out after 10 minutes",
                'scene_dir': str(scene_dir),
                'runtime_seconds': time.time() - start_time,
                'outputs': {}
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'scene_dir': str(scene_dir),
                'runtime_seconds': time.time() - start_time,
                'outputs': {}
            }
    
    def _check_vggt_outputs(self, scene_dir: Path, **kwargs) -> Dict:
        """
        Check for expected VGGT output files and return their info
        
        Args:
            scene_dir: Scene directory path
            **kwargs: VGGT parameters
            
        Returns:
            Dict with output file information
        """
        outputs = {}
        
        # Check for camera poses file (always generated) - now under sam3d_multiview_prediction
        output_base_dir = scene_dir / "sam3d_multiview_prediction"
        camera_poses_file = output_base_dir / "vggt_camera_poses.npz"
        if camera_poses_file.exists():
            outputs['camera_poses'] = str(camera_poses_file)
            
            # Load and get basic info
            try:
                data = np.load(camera_poses_file)
                outputs['camera_poses_info'] = {
                    'num_images': len(data['image_names']),
                    'image_names': data['image_names'].tolist(),
                    'extrinsic_shape': data['extrinsic'].shape,
                    'intrinsic_shape': data['intrinsic'].shape,
                    'points_3d_shape': data['points_3d'].shape
                }
            except Exception as e:
                outputs['camera_poses_error'] = str(e)
        
        # Check for COLMAP sparse reconstruction - now under sam3d_multiview_prediction
        # Build folder name based on parameters (matching VGGT_interface_SAM3D.py logic)
        folder_name = "VGGT"
        
        # Add view information if specific images were selected
        if kwargs.get('select_images'):
            selections = [s.strip() for s in kwargs['select_images'].split(',')]
            view_indices = []
            for selection in selections:
                if selection.isdigit():
                    view_indices.append(selection)
                else:
                    # For runner, we can't easily resolve names to indices here
                    # But we can use the selection as-is for folder name
                    view_indices.append(selection.replace('render_', ''))
            if view_indices:
                folder_name += f"_view_{'-'.join(view_indices)}"
        
        if kwargs.get('use_masks', True) or kwargs.get('self_mask', False):
            folder_name += "_masked"
        else:
            folder_name += "_nomask"
        
        if kwargs.get('scale_pointcloud', False):
            folder_name += "_scaled"
        else:
            folder_name += "_unscaled"
        
        folder_name += f"_conf{kwargs.get('conf_thres_value', 0.0)}"
        
        sparse_dir = output_base_dir / folder_name
        if sparse_dir.exists():
            outputs['sparse_reconstruction'] = str(sparse_dir)
            
            # Check for key files
            colmap_files = {
                'cameras.bin': sparse_dir / 'cameras.bin',
                'images.bin': sparse_dir / 'images.bin', 
                'points3D.bin': sparse_dir / 'points3D.bin',
                'points.ply': sparse_dir / 'points.ply'
            }
            
            outputs['sparse_files'] = {}
            for name, file_path in colmap_files.items():
                outputs['sparse_files'][name] = {
                    'exists': file_path.exists(),
                    'path': str(file_path) if file_path.exists() else None,
                    'size_bytes': file_path.stat().st_size if file_path.exists() else 0
                }
        
        return outputs
    
    def run_batch_scenes(self, 
                        scene_list: List[Union[str, Path]], 
                        max_scenes: Optional[int] = None,
                        start_index: int = 0,
                        save_progress: bool = True,
                        progress_file: str = "vggt_batch_progress.json",
                        **kwargs) -> Dict:
        """
        Run VGGT inference on multiple scenes
        
        Args:
            scene_list: List of scene directory paths
            max_scenes: Maximum number of scenes to process (None = all)
            start_index: Index to start processing from
            save_progress: Whether to save intermediate progress
            progress_file: File to save progress to
            **kwargs: Additional parameters for VGGT
            
        Returns:
            Dict with batch processing results
        """
        print(f"Starting VGGT batch processing")
        print(f"Total scenes available: {len(scene_list)}")
        
        # Apply filtering
        scene_list = scene_list[start_index:]
        if max_scenes:
            scene_list = scene_list[:max_scenes]
        
        print(f"Processing {len(scene_list)} scenes (starting from index {start_index})")
        
        # Initialize results
        batch_results = {
            'start_time': time.time(),
            'parameters': kwargs,
            'scenes_processed': 0,
            'scenes_successful': 0,
            'scenes_failed': 0,
            'results': [],
            'summary': {}
        }
        
        # Process scenes
        for i, scene_path in enumerate(scene_list):
            scene_path = Path(scene_path)
            actual_index = start_index + i
            
            print(f"\nProgress: {i+1}/{len(scene_list)} (global index {actual_index}) - {scene_path.name}")
            
            # Run VGGT on single scene
            result = self.run_single_scene(scene_path, **kwargs)
            
            # Update results
            batch_results['results'].append(result)
            batch_results['scenes_processed'] += 1
            
            if result['success']:
                batch_results['scenes_successful'] += 1
                print(f"  ✓ Success ({result['runtime_seconds']:.1f}s)")
            else:
                batch_results['scenes_failed'] += 1
                print(f"  ✗ Failed: {result['error']}")
            
            # Save intermediate progress
            if save_progress and (i + 1) % 5 == 0:
                self._save_batch_progress(batch_results, progress_file)
                print(f"  Progress saved to {progress_file}")
        
        # Finalize results
        batch_results['end_time'] = time.time()
        batch_results['total_runtime_seconds'] = batch_results['end_time'] - batch_results['start_time']
        
        # Create summary
        batch_results['summary'] = {
            'success_rate': batch_results['scenes_successful'] / batch_results['scenes_processed'] * 100,
            'average_runtime_seconds': np.mean([r.get('runtime_seconds', 0) for r in batch_results['results'] if r['success']]),
            'total_scenes': len(scene_list),
            'successful_scenes': batch_results['scenes_successful'],
            'failed_scenes': batch_results['scenes_failed']
        }
        
        # Save final results
        if save_progress:
            self._save_batch_progress(batch_results, progress_file)
        
        # Print summary
        self._print_batch_summary(batch_results)
        
        return batch_results
    
    def _save_batch_progress(self, batch_results: Dict, progress_file: str):
        """Save batch progress to JSON file"""
        # Make results JSON serializable
        json_results = self._make_json_serializable(batch_results.copy())
        
        with open(progress_file, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-compatible types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _print_batch_summary(self, batch_results: Dict):
        """Print comprehensive batch processing summary"""
        summary = batch_results['summary']
        
        print(f"\n{'='*70}")
        print("VGGT BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        
        print(f"Scenes processed: {batch_results['scenes_processed']}")
        print(f"Successful: {batch_results['scenes_successful']} ({summary['success_rate']:.1f}%)")
        print(f"Failed: {batch_results['scenes_failed']}")
        print(f"Total runtime: {batch_results['total_runtime_seconds']:.1f} seconds")
        print(f"Average runtime per scene: {summary['average_runtime_seconds']:.1f} seconds")
        
        # Show failed scenes
        if batch_results['scenes_failed'] > 0:
            print(f"\nFailed scenes:")
            for result in batch_results['results']:
                if not result['success']:
                    scene_name = Path(result['scene_dir']).name
                    print(f"  {scene_name}: {result['error']}")
    
    def find_dataset_scenes(self, dataset_path: Union[str, Path], 
                           pattern: str = "**/resized_images") -> List[Path]:
        """
        Find all scene directories in a dataset
        
        Args:
            dataset_path: Path to dataset root
            pattern: Glob pattern to find scene directories (default looks for resized_images)
            
        Returns:
            List of scene directory paths
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Find directories containing resized_images
        scene_dirs = []
        for resized_images_dir in dataset_path.glob(pattern):
            if resized_images_dir.is_dir():
                scene_dir = resized_images_dir.parent
                scene_dirs.append(scene_dir)
        
        # Sort for consistent ordering
        scene_dirs.sort()
        
        print(f"Found {len(scene_dirs)} scenes in {dataset_path}")
        
        return scene_dirs


def main():
    """Command line interface for VGGT runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run VGGT inference from SAM3D environment")
    parser.add_argument("--scene_dir", type=str, help="Single scene directory to process")
    parser.add_argument("--dataset_path", type=str, help="Dataset path for batch processing")
    parser.add_argument("--max_scenes", type=int, help="Maximum number of scenes to process")
    parser.add_argument("--start_index", type=int, default=0, help="Index to start processing from")
    parser.add_argument("--image_dir", type=str, default="resized_images", help="Image directory name within scene directory")
    parser.add_argument("--mask_dir", type=str, default="masks", help="Mask directory name within scene directory")
    parser.add_argument("--select_images", type=str, default=None, help="Comma-separated list of image names or indices (e.g., 'render_000,render_001' or '0,1,5')")
    parser.add_argument("--no_masks", action="store_true", help="Disable mask usage")
    parser.add_argument("--self_mask", action="store_true", help="Generate masks from images themselves (for rendered images)")
    parser.add_argument("--conf_thres_value", type=float, default=0.0, help="Confidence threshold")
    parser.add_argument("--scale_pointcloud", action="store_true", help="Scale point cloud to real units")
    parser.add_argument("--progress_file", type=str, default="vggt_batch_progress.json", help="Progress file name")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = VGGTRunner()
    
    # Prepare VGGT parameters
    vggt_params = {
        'use_masks': not args.no_masks,
        'image_dir': args.image_dir,
        'mask_dir': args.mask_dir,
        'self_mask': args.self_mask,
        'select_images': args.select_images,
        'conf_thres_value': args.conf_thres_value,
        'scale_pointcloud': args.scale_pointcloud
    }
    
    if args.scene_dir:
        # Single scene processing
        result = runner.run_single_scene(args.scene_dir, **vggt_params)
        
        if result['success']:
            print(f"✓ VGGT completed successfully for {args.scene_dir}")
            print(f"  Runtime: {result['runtime_seconds']:.1f} seconds")
            print(f"  Outputs: {list(result['outputs'].keys())}")
        else:
            print(f"✗ VGGT failed for {args.scene_dir}: {result['error']}")
            if 'stdout' in result and result['stdout'].strip():
                print("STDOUT:")
                print(result['stdout'])
            if 'stderr' in result and result['stderr'].strip():
                print("STDERR:")
                print(result['stderr'])
            sys.exit(1)
    
    elif args.dataset_path:
        # Batch processing
        pattern = f"**/{args.image_dir}"
        scene_list = runner.find_dataset_scenes(args.dataset_path, pattern=pattern)
        
        if not scene_list:
            print(f"No scenes found in {args.dataset_path}")
            sys.exit(1)
        
        batch_results = runner.run_batch_scenes(
            scene_list,
            max_scenes=args.max_scenes,
            start_index=args.start_index,
            progress_file=args.progress_file,
            **vggt_params
        )
        
        if batch_results['scenes_failed'] > 0:
            sys.exit(1)
    
    else:
        print("Error: Must specify either --scene_dir or --dataset_path")
        sys.exit(1)


if __name__ == "__main__":
    main()
