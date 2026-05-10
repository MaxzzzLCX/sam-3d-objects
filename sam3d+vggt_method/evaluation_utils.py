"""
Utilities for aggregating SAM3D+VGGT quantitative evaluation outputs.

Expected per-run input file format: chamfer_summary*.json
with keys such as:
  - anisotropic_chamfer.symmetric_mean
  - uniform_chamfer.symmetric_mean
  - anisotropic_rasterized_occupancy_iou_2d.mean_iou
  - uniform_rasterized_occupancy_iou_2d.mean_iou
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _safe_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
	cur: Any = data
	for key in keys:
		if not isinstance(cur, dict) or key not in cur:
			return default
		cur = cur[key]
	return cur


def _parse_folder_metadata(folder_name: str) -> tuple[str, int | None, int | None]:
	"""
	Parse folder names like: avocado_plate_percentile99_view3
	Returns: (food_folder_name, percentile, view_idx)
	"""
	match = re.match(r"^(.*)_percentile(\d+)_view(\d+)$", folder_name)
	if not match:
		return folder_name, None, None
	return match.group(1), int(match.group(2)), int(match.group(3))


def find_summary_files(results_root: str) -> list[Path]:
	root = Path(results_root)
	if not root.exists():
		raise FileNotFoundError(f"results_root does not exist: {results_root}")

	# supports chamfer_summary.json and chamfer_summary_viewX.json
	files = sorted(root.glob("**/chamfer_summary*.json"))
	return [f for f in files if f.is_file()]


def collect_records(results_root: str) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []
	for summary_file in find_summary_files(results_root):
		payload = json.loads(summary_file.read_text(encoding="utf-8"))

		folder_name = summary_file.parent.name
		food_name, percentile, view_idx = _parse_folder_metadata(folder_name)

		record = {
			"file": str(summary_file),
			"folder": folder_name,
			"food_folder": food_name,
			"percentile": percentile,
			"view_idx": view_idx,
			"anisotropic_chamfer": _safe_get(payload, ["anisotropic_chamfer", "symmetric_mean"]),
			"uniform_chamfer": _safe_get(payload, ["uniform_chamfer", "symmetric_mean"]),
			"anisotropic_iou": _safe_get(payload, ["anisotropic_rasterized_occupancy_iou_2d", "mean_iou"]),
			"uniform_iou": _safe_get(payload, ["uniform_rasterized_occupancy_iou_2d", "mean_iou"]),
		}
		records.append(record)

	return records


def _stats(values: list[float | None]) -> dict[str, Any]:
	clean = np.array([v for v in values if v is not None], dtype=np.float64)
	if clean.size == 0:
		return {"n": 0, "mean": None, "std": None}
	return {
		"n": int(clean.size),
		"mean": float(np.mean(clean)),
		"std": float(np.std(clean)),
	}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
	anis_ch = [r["anisotropic_chamfer"] for r in records]
	uni_ch = [r["uniform_chamfer"] for r in records]
	anis_iou = [r["anisotropic_iou"] for r in records]
	uni_iou = [r["uniform_iou"] for r in records]

	summary = {
		"n_records": len(records),
		"overall": {
			"anisotropic_chamfer": _stats(anis_ch),
			"uniform_chamfer": _stats(uni_ch),
			"anisotropic_iou": _stats(anis_iou),
			"uniform_iou": _stats(uni_iou),
		},
		"by_food_folder": {},
	}

	grouped: dict[str, list[dict[str, Any]]] = {}
	for rec in records:
		grouped.setdefault(rec["food_folder"], []).append(rec)

	for food, recs in grouped.items():
		summary["by_food_folder"][food] = {
			"n_records": len(recs),
			"anisotropic_chamfer": _stats([r["anisotropic_chamfer"] for r in recs]),
			"uniform_chamfer": _stats([r["uniform_chamfer"] for r in recs]),
			"anisotropic_iou": _stats([r["anisotropic_iou"] for r in recs]),
			"uniform_iou": _stats([r["uniform_iou"] for r in recs]),
		}

	return summary


def save_records_csv(records: list[dict[str, Any]], out_csv: Path) -> None:
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"file",
		"folder",
		"food_folder",
		"percentile",
		"view_idx",
		"anisotropic_chamfer",
		"uniform_chamfer",
		"anisotropic_iou",
		"uniform_iou",
	]
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for r in records:
			writer.writerow(r)


def plot_distributions(records: list[dict[str, Any]], out_dir: Path) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)

	anis_ch = np.array([r["anisotropic_chamfer"] for r in records if r["anisotropic_chamfer"] is not None], dtype=np.float64)
	uni_ch = np.array([r["uniform_chamfer"] for r in records if r["uniform_chamfer"] is not None], dtype=np.float64)
	anis_iou = np.array([r["anisotropic_iou"] for r in records if r["anisotropic_iou"] is not None], dtype=np.float64)
	uni_iou = np.array([r["uniform_iou"] for r in records if r["uniform_iou"] is not None], dtype=np.float64)

	# Histogram (Chamfer)
	fig, ax = plt.subplots(figsize=(8, 5))
	all_ch = np.concatenate([arr for arr in [anis_ch, uni_ch] if arr.size > 0]) if (anis_ch.size > 0 or uni_ch.size > 0) else np.array([])
	if all_ch.size > 0:
		bin_edges = np.linspace(float(np.min(all_ch)), float(np.max(all_ch)), 21)
		# Guard for degenerate case where all values are identical
		if np.allclose(bin_edges[0], bin_edges[-1]):
			bin_edges = np.linspace(bin_edges[0] - 1e-6, bin_edges[-1] + 1e-6, 21)
	else:
		bin_edges = 20
	if anis_ch.size > 0:
		ax.hist(anis_ch, bins=bin_edges, alpha=0.7, label="anisotropic", color="blue", edgecolor="black")
	if uni_ch.size > 0:
		ax.hist(uni_ch, bins=bin_edges, alpha=0.7, label="uniform", color="orange", edgecolor="black")
	ax.set_title("Chamfer distribution")
	ax.set_xlabel("symmetric_mean")
	ax.set_ylabel("count")
	ax.legend()
	fig.tight_layout()
	fig.savefig(out_dir / "hist_chamfer.png", dpi=180)
	plt.close(fig)

	# Boxplot (Chamfer + IoU)
	fig, axs = plt.subplots(1, 2, figsize=(12, 5))
	chamfer_data, chamfer_labels = [], []
	if anis_ch.size > 0:
		chamfer_data.append(anis_ch)
		chamfer_labels.append("anisotropic")
	if uni_ch.size > 0:
		chamfer_data.append(uni_ch)
		chamfer_labels.append("uniform")
	if chamfer_data:
		axs[0].boxplot(chamfer_data, tick_labels=chamfer_labels)
	axs[0].set_title("Chamfer boxplot")
	axs[0].set_ylabel("symmetric_mean")

	iou_data, iou_labels = [], []
	if anis_iou.size > 0:
		iou_data.append(anis_iou)
		iou_labels.append("anisotropic")
	if uni_iou.size > 0:
		iou_data.append(uni_iou)
		iou_labels.append("uniform")
	if iou_data:
		axs[1].boxplot(iou_data, tick_labels=iou_labels)
	axs[1].set_title("Rasterized occupancy IoU boxplot")
	axs[1].set_ylabel("mean_iou")

	fig.tight_layout()
	fig.savefig(out_dir / "boxplots_chamfer_iou.png", dpi=180)
	plt.close(fig)


def evaluate_results(results_root: str, output_dir: str | None = None) -> dict[str, Any]:
	root = Path(results_root)
	out_dir = Path(output_dir) if output_dir else root / "_aggregate"
	out_dir.mkdir(parents=True, exist_ok=True)

	records = collect_records(results_root)
	summary = summarize_records(records)

	save_records_csv(records, out_dir / "records.csv")
	(out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
	if records:
		plot_distributions(records, out_dir)

	print(f"Processed {len(records)} result files.")
	print(json.dumps(summary["overall"], indent=2))
	print(f"Saved aggregate outputs to {out_dir}")

	return summary


def main() -> None:
	parser = argparse.ArgumentParser(description="Aggregate Chamfer/IoU metrics across per-view evaluation outputs")
	parser.add_argument(
		"--results_root",
		default="/scratch/cl927/sam-3d-objects/sam3d+vggt_method/alignment_outputs/sam3d_with_pointmaps",
		help="Root directory containing per-view folders with chamfer_summary*.json",
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		help="Optional output directory for aggregated csv/json/plots (default: <results_root>/_aggregate)",
	)
	args = parser.parse_args()

	evaluate_results(args.results_root, args.output_dir)


if __name__ == "__main__":
	main()

