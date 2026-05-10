import argparse
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

import numpy as np
import random

np.random.seed(42)
random.seed(42)


PROMPT_TEMPLATE = """
You are estimating the volume of food from images.

The plate has a known diameter of {} cm which should be used as a scale reference.

Use all provided images jointly to estimate the volume. They are taken from different viewpoints on the same scene.

Return ONLY the following JSON:

{{
    "reasoning": <string>,
    "volume_cm3": <float>
}}

The value must be a single number in cubic centimeters.
Do not include units.
Do not output any text outside the JSON.
""".strip()


class GeminiVolumeEstimator:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_output_tokens: int | None = None,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY in your environment before running.")

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.client = genai.Client(api_key=api_key)

    def list_image_paths(self, image_folder_dir: str) -> list[Path]:
        folder = Path(image_folder_dir)
        if not folder.is_dir():
            raise RuntimeError(f"Image folder does not exist: {image_folder_dir}")

        return sorted(
            [
                path
                for path in folder.iterdir()
                if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
            ]
        )

    def parse_model_json(self, response_text: str) -> dict[str, Any]:
        text = (response_text or "").strip()
        if not text:
            raise RuntimeError("Model returned empty response text.")

        fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
        if fenced_match:
            text = fenced_match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            object_match = re.search(r"\{[\s\S]*\}", text)
            if not object_match:
                raise
            return json.loads(object_match.group(0))

    def load_images(
        self,
        image_folder_dir: str,
        num: int,
        random_order: bool = False,
        selected_image_paths: list[Path] | None = None,
    ) -> list[types.Part]:
        if num <= 0:
            raise ValueError("num must be greater than 0")

        image_paths = list(selected_image_paths) if selected_image_paths is not None else self.list_image_paths(image_folder_dir)

        # If random order is requested, shuffle the list of image paths to introduce variability
        if random_order:
            random.shuffle(image_paths)


        if len(image_paths) < num:
            print(
                f"Warning: Requested {num} images but only found {len(image_paths)} in {image_folder_dir}. "
                f"Using all available images."
            )

        image_parts: list[types.Part] = []
        for image_path in image_paths[:num]:
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type or not mime_type.startswith("image/"):
                raise RuntimeError(f"Unsupported or unknown image type: {image_path.name}")

            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

            image_parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

        return image_parts

    def query_once(
        self,
        image_folder_dir: str,
        num: int,
        prompt: str,
        random_order: bool = False,
        selected_image_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        
        image_parts = self.load_images(
            image_folder_dir,
            num,
            random_order=random_order,
            selected_image_paths=selected_image_paths,
        )

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            response_mime_type="application/json",
        )
        if self.max_output_tokens is not None:
            config.max_output_tokens = self.max_output_tokens

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, *image_parts],
            config=config,
        )

        return self.parse_model_json(response.text)


class VolumeExperimentRunner:
    def __init__(self, estimator: GeminiVolumeEstimator, random_order: bool = False):
        self.estimator = estimator
        self.random_order = random_order

    def make_prompt(self, prerun_info: dict[str, Any]) -> str:

        return PROMPT_TEMPLATE.format(prerun_info['plate_diameter_cm'])

    def run(
        self,
        image_folder_dirs: list[str],
        num_values: list[int],
        repeats: int,
        output_dir: str | None = None,
    ) -> None:
        if repeats <= 0:
            raise ValueError("repeats must be greater than 0")

        print(f"Running: folders={image_folder_dirs}, num_values={num_values}, repeats={repeats}")

        # all_results: list[dict[str, Any]] = []
        for folder_dir in image_folder_dirs:
            for num in num_values:

                # Fetch prerun info
                prerun_info = self.prerun_info(folder_dir)

                prompt = self.make_prompt(prerun_info=prerun_info)

                if num == 1:
                    results = self.run_single_view_all_views_once(
                        folder_dir=folder_dir,
                        prompt=prompt,
                        prerun_info=prerun_info,
                    )
                else:
                    results = self.run_multiple_times(
                        folder_dir=folder_dir,
                        num=num,
                        repeats=repeats,
                        prompt=prompt,
                        prerun_info=prerun_info,
                    )

                # Save experiment results
                self.save_results(
                    image_folder_dir=folder_dir,
                    num=num,
                    results=results,
                    output_dir=f"{output_dir}/{self.estimator.model}",
                )
                
    def prerun_info(self, folder_dir: str) -> dict[str, Any]:
        """
        Fetch relevant information about plate and food before run. 
        """

        # If the /image folder is supplied, find the json in the parent directory
        if folder_dir.endswith("/images"):
            folder_dir = str(Path(folder_dir).parent)

        info_path = Path(folder_dir) / "plate_diameters.json"
        if not info_path.is_file():
            raise RuntimeError(f"Info file not found for folder {folder_dir}: expected at {info_path}")

        try:
            info_data = json.loads(info_path.read_text(encoding="utf-8"))
            info_data_0 = info_data.get("0")
            return {
                "plate_diameter_cm": info_data_0.get("diameter"),
                "volume_gt_cm3": info_data_0.get("gt_volume_ml"),
            }
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON from {info_path}: {e}") from e
        

    def save_results(self, image_folder_dir: str, num: int, results: dict[str, Any], output_dir: str | None = None) -> None:
        if output_dir is None:
            raise ValueError("output_dir must be provided to save results")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        image_name = Path(image_folder_dir).parent.name
        output_path = output_dir / f"{image_name}-num{num}-results.json"
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved results to {output_path.resolve()}")

    def _compute_volume_stats(self, raw_results: dict[str, dict[str, Any]], gt_volume: float | None) -> dict[str, Any]:
        volume_estimations = []
        for _, raw_result in raw_results.items():
            value = raw_result.get("volume_cm3")
            try:
                volume_estimations.append(float(value))
            except (TypeError, ValueError):
                continue

        volume_estimations_np = np.array(volume_estimations, dtype=np.float32)
        volume_estimations_error = (
            np.abs(volume_estimations_np - gt_volume) / gt_volume
            if gt_volume not in [None, 0]
            else None
        )

        return {
            "volume_estimations": volume_estimations,
            "volume_mean": float(np.mean(volume_estimations_np)) if volume_estimations_np.size > 0 else None,
            "volume_std": float(np.std(volume_estimations_np)) if volume_estimations_np.size > 0 else None,
            "volume_error_mean": float(np.mean(volume_estimations_error)) if volume_estimations_error is not None and volume_estimations_error.size > 0 else None,
            "volume_error_std": float(np.std(volume_estimations_error)) if volume_estimations_error is not None and volume_estimations_error.size > 0 else None,
        }

    def run_single_view_all_views_once(
        self,
        folder_dir: str,
        prompt: str,
        prerun_info: dict[str, Any],
    ) -> dict[str, Any]:
        gt_volume = prerun_info.get("volume_gt_cm3")
        image_paths = self.estimator.list_image_paths(folder_dir)
        if not image_paths:
            raise RuntimeError(f"No valid images found in {folder_dir}")

        all_results: dict[str, Any] = {
            "image_folder_dir": folder_dir,
            "num_views": 1,
            "mode": "single_view_all_views_once",
            "n_views_used": len(image_paths),
            "model": self.estimator.model,
            "temperature": self.estimator.temperature,
            "top_p": self.estimator.top_p,
            "results": {},
            "raw_results": {},
        }

        for index, image_path in enumerate(image_paths):
            result = self.estimator.query_once(
                image_folder_dir=folder_dir,
                num=1,
                prompt=prompt,
                random_order=False,
                selected_image_paths=[image_path],
            )
            result["selected_image"] = image_path.name
            all_results["raw_results"][f"view_{index}"] = result

        all_results["results"] = self._compute_volume_stats(all_results["raw_results"], gt_volume)
        return all_results

    def run_multiple_times(
        self,
        folder_dir: str,
        num: int,
        repeats: int,
        prompt: str,
        prerun_info: dict[str, Any],
    ) -> dict[str, Any]:
        
        print("=="*20)
        print(f"Querying with folder={folder_dir}, num={num}, repeats={repeats}, random_order={self.random_order}")
        print("=="*20)
        
        # Extract ground truth volume
        gt_volume = prerun_info.get("volume_gt_cm3")
        
        # Record configs
        all_results = {
            "image_folder_dir": folder_dir,
            "num_views": num,
            "mode": "multi_view_repeats",
            "model": self.estimator.model,
            "temperature": self.estimator.temperature,
            "top_p": self.estimator.top_p,
            "repeats": repeats,
            "results": {},
            "raw_results": {}
        }
        

        for trial in range(repeats):
            result = self.estimator.query_once(
                image_folder_dir=folder_dir,
                num=num,
                prompt=prompt,
                random_order=self.random_order,
            )
            all_results["raw_results"][f"trial_{trial}"] = result

        all_results["results"] = self._compute_volume_stats(all_results["raw_results"], gt_volume)
        return all_results

def parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_csv_ints(raw: str | None) -> list[int]:
    if not raw:
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def save_results(output_json: str, results: list[dict[str, Any]]) -> Path:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_results": len(results),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini volume estimation experiment runner")
    # parser.add_argument(
    #     "--image_folder_dirs",
    #     help="Comma-separated image folder dirs, e.g. /path/a,/path/b",
    # )
    # parser.add_argument(
    #     "--num_values",
    #     help="Comma-separated num views, e.g. 1,2,3",
    # )
    parser.add_argument("--repeats", type=int, default=3, help="Runs per configuration")
    parser.add_argument("--random_order", action="store_true", help="Randomize order of selecting image views")
    parser.add_argument("--prompt", help="Prompt to send")
    # parser.add_argument("--output_json", help="Path to save experiment results JSON")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--max_output_tokens", type=int, default=None, help="Optional max output tokens")
    args = parser.parse_args()

    # if args.image_folder_dirs is None:
    #     args.image_folder_dirs = "/Users/maxlyu/Desktop/IIB_Project_Temp/dataset/avocado_plate"
    # if args.num_values is None:
    #     args.num_values = "1"
    # if args.prompt is None:
    #     args.prompt = DEFAULT_PROMPT
    # if args.output_json is None:
    #     args.output_json = "gemini_experiment_results.json"

    # image_folder_dirs = parse_csv_list(args.image_folder_dirs)
    # num_values = parse_csv_ints(args.num_values)

    ### Manually set arguments for now
   
    image_folder_dirs = [
        # "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/potato_bowl/images",
        # "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/potato_plate/images",
        # "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/orange_bowl/images",
        # "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/orange_plate/images",
        # "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/egg_bowl/images",
        # "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/egg_plate/images",
        "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/avocado_plate/images",
        "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/strawberry_bowl/images",
        "/scratch/cl927/sam-3d-objects/real_dataset/real_data_multiview_volume_vggt/strawberry_plate/images"
    ]
    num_values = [1, 6]
    args.random_order = True
    args.repeats = 1
    args.model = "gemini-3.1-pro-preview" # "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-3-flash-preview"


    prompt = args.prompt or PROMPT_TEMPLATE

    if not image_folder_dirs or not num_values:
        parser.error("Provide non-empty image folders and num values.")

    estimator = GeminiVolumeEstimator(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )
    runner = VolumeExperimentRunner(estimator, random_order=args.random_order)

    runner.run(
        image_folder_dirs=image_folder_dirs,
        num_values=num_values,
        repeats=args.repeats,
        output_dir="/scratch/cl927/sam-3d-objects/vlm-baseline/outputs",
    )

    # output_path = save_results(args.output_json, results)
    # print(json.dumps({"n_results": len(results), "output_json": str(output_path.resolve())}, indent=2))
    # print(f"Saved JSON to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
