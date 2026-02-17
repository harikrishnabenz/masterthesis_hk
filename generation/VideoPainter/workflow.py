"""VideoPainter edit workflow for HLX.

Runs the VideoPainter edit/inpaint pipeline (infer/edit_bench.py) inside the
container.

This workflow is designed to consume preprocessed per-video folders produced by
generation/VideoPainter/data_preprocessing.py and stored in GCS under
VP_DATA_PREFIX.
"""

import csv
import itertools
import json
import logging
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import gcsfs

from hlx.wf import DedicatedNode, Node, fuse_prefetch_metadata, task, workflow
from hlx.wf.mounts import MOUNTPOINT, FuseBucket

logger = logging.getLogger(__name__)


def _parse_instruction_list(instructions: str | None) -> list[str]:
	"""Parse a list of editing instructions from a single CLI string.

	Supported formats:
	- "instr1 || instr2 || instr3" (recommended)
	- newline-separated (useful in configs)
	"""
	if not instructions:
		return []
	s = str(instructions).strip()
	if not s:
		return []
	parts = s.split("||") if "||" in s else s.splitlines()
	return [p.strip().strip('"').strip() for p in parts if p.strip().strip('"').strip()]


def _sanitize_folder_component(text: str, *, max_len: int = 60) -> str:
	"""Make a safe folder name component for GCS/local paths."""
	base = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip()).strip("_") or "instruction"
	return base[:max_len].rstrip("_") if len(base) > max_len else base


def _lane_spec_to_instruction(c: str, col: str, pat: str) -> str:
	"""Convert lane specification to editing instruction."""
	c2 = (c or "").strip().lower()
	col2 = (col or "").strip().lower()
	pat2 = (pat or "").strip().lower()
	return f"lane {c2} {col2} {pat2}".strip() if (c2 and col2 and pat2) else ""


# ----------------------------------------------------------------------------------
# RUN SUFFIX (set via VP_RUN_SUFFIX env var from scripts/build_and_run.sh)
# ----------------------------------------------------------------------------------
X = (os.environ.get("VP_RUN_SUFFIX") or "").strip()


# ----------------------------------------------------------------------------------
# VLM (Qwen2.5-VL-7B) MOUNT
# ----------------------------------------------------------------------------------
# We only support mounting the 7B VLM checkpoint via a dedicated FuseBucket and
# symlinking it into the expected local path:
#   /workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct -> <mounted_folder>
#
# This avoids relying on HuggingFace Hub downloads at runtime and avoids needing
# the VLM checkpoint to live inside the main ckpt mount.
LLM_MODEL_SIZE = "7B"

USE_QWEN2_5_VL_7B = True

# Fixed to 1 GPU for 7B model
COMPUTE_NODE = Node.A100_80GB_1GPU
# To dedicate the second GPU to Qwen (VLM) when available, keep Flux on the same
# GPU as CogVideoX by default.
VP_FLUX_DEVICE_DEFAULT = "cuda:0"


VP_RUN_SUFFIX = X

# Allow the runner script to pin an exact image tag (avoids stale ':latest' pulls).
REMOTE_IMAGE = (
	f"europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_vp{VP_RUN_SUFFIX}"
	if VP_RUN_SUFFIX
	else "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_vp"
)
CONTAINER_IMAGE_DEFAULT = f"{REMOTE_IMAGE}:latest"
CONTAINER_IMAGE = os.environ.get("VP_CONTAINER_IMAGE", CONTAINER_IMAGE_DEFAULT)

# ----------------------------------------------------------------------------------
# PATHS (inside container) and GCS mount
# ----------------------------------------------------------------------------------
BASE_WORKDIR = "/workspace/VideoPainter"
DEFAULT_DATA_DIR = os.path.join(BASE_WORKDIR, "data")
DEFAULT_CKPT_DIR = os.path.join(BASE_WORKDIR, "ckpt")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_WORKDIR, "output_vp")

OUTPUT_SUBDIR = "output/vp"
SCRATCH_BASE = "/tmp/videopainter_output"
SCRATCH_DATA_BASE = "/tmp/videopainter_data"

# GCS bucket paths for checkpoints
VP_BUCKET = "mbadas-sandbox-research-9bb9c7f"
VP_BUCKET_PREFIX = "workspace/user/hbaskar/Video_inpainting/videopainter"

# Optional VLM checkpoint folder (mounted separately so it doesn't interfere with
# existing ckpt or data mounts).
# NOTE: In our bucket layout, VLM checkpoints live under `ckpt/vlm/...`.
VLM_7B_GCS_PREFIX = os.path.join(VP_BUCKET_PREFIX, "ckpt", "vlm", "Qwen2.5-VL-7B-Instruct")

# Trained FluxFill checkpoint folder (configurable via environment variable)
TRAINED_FLUXFILL_GCS_PREFIX = os.environ.get(
	"TRAINED_FLUXFILL_GCS_PATH",
	"workspace/user/hbaskar/Video_inpainting/videopainter/training/trained_checkpoint/fluxfill_single_white_solid_clearroad_20260212_144012",
)

# GCS bucket path for SAM2 preprocessed data.
# IMPORTANT: we mount the *base* prefix so `data_run_id` can be chosen dynamically.
DEFAULT_DATA_RUN_ID = os.environ.get("DATA_RUN_ID", "10")
VP_DATA_PREFIX = "workspace/user/hbaskar/outputs/preprocessed_data_vp"

# FuseBucket mount paths
VP_FUSE_MOUNT_NAME = "vp-bucket"
VP_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, VP_FUSE_MOUNT_NAME)
MOUNTED_CKPT_PATH = os.path.join(VP_FUSE_MOUNT_ROOT, "ckpt")

VP_VLM_7B_FUSE_MOUNT_NAME = "vp-vlm-7b"
VP_VLM_7B_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, VP_VLM_7B_FUSE_MOUNT_NAME)

VLM_7B_DEST_PATH = os.path.join(DEFAULT_CKPT_DIR, "vlm", "Qwen2.5-VL-7B-Instruct")

TRAINED_FLUXFILL_FUSE_MOUNT_NAME = "vp-trained-fluxfill"
TRAINED_FLUXFILL_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, TRAINED_FLUXFILL_FUSE_MOUNT_NAME)
TRAINED_FLUXFILL_DEST_PATH = os.path.join(DEFAULT_CKPT_DIR, "trained_fluxfill_lora")

VP_DATA_FUSE_MOUNT_NAME = "data"
VP_DATA_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, VP_DATA_FUSE_MOUNT_NAME)

# Override via VP_OUTPUT_BASE env var in build_and_run.sh
GCS_OUTPUT_BASE = os.environ.get(
	"VP_OUTPUT_BASE",
	f"gs://{VP_BUCKET}/workspace/user/hbaskar/outputs/vp",
)

DEFAULT_MODEL_PATH = os.path.join(DEFAULT_CKPT_DIR, "CogVideoX-5b-I2V")
DEFAULT_BRANCH_PATH = os.path.join(DEFAULT_CKPT_DIR, "VideoPainter/checkpoints/branch")
DEFAULT_IMG_INPAINT_PATH = os.path.join(DEFAULT_CKPT_DIR, "flux_inp")


DEFAULT_META = os.path.join(DEFAULT_DATA_DIR, "meta.csv")
DEFAULT_VIDEO_ROOT = os.path.join(DEFAULT_DATA_DIR, "raw_videos")


def _resolve_preprocessed_video_paths(*, data_run_id: str, data_video_id: str) -> tuple[str, str]:
	"""Get paths to meta.csv and raw_videos for a specific run_id + video ID."""
	meta = os.path.join(VP_DATA_FUSE_MOUNT_ROOT, data_run_id, data_video_id, "meta.csv")
	raw_root = os.path.join(VP_DATA_FUSE_MOUNT_ROOT, data_run_id, data_video_id, "raw_videos")
	return meta, raw_root


def _video_base_name_from_meta(*, meta_csv_path: str, inpainting_sample_id: int) -> str:
	"""Extract video base name from meta.csv for the given sample ID."""
	if inpainting_sample_id < 0:
		raise ValueError("inpainting_sample_id must be >= 0")
	with open(meta_csv_path, newline="") as f:
		row = next(itertools.islice(csv.DictReader(f), inpainting_sample_id, inpainting_sample_id + 1), None)
		if row:
			path = (row.get("path") or "").strip()
			if not path:
				raise KeyError(
					f"meta.csv row {inpainting_sample_id} is missing required 'path' column/value"
				)
			return path.split(".")[0]
	raise IndexError(
		f"meta.csv does not have row index {inpainting_sample_id} (file={meta_csv_path})"
	)


def _stage_preprocessed_inputs(*, data_run_id: str, data_video_id: str, inpainting_sample_id: int) -> tuple[str, str]:
	"""Stage meta.csv, masks, and video files to local scratch for processing."""
	meta_src, raw_root = _resolve_preprocessed_video_paths(data_run_id=data_run_id, data_video_id=data_video_id)
	if not os.path.exists(meta_src):
		raise FileNotFoundError(
			f"Missing meta.csv for run_id={data_run_id} video_id={data_video_id}: {meta_src}"
		)

	stage_dir = os.path.join(SCRATCH_DATA_BASE, data_video_id)
	Path(stage_dir).mkdir(parents=True, exist_ok=True)
	meta_dst = os.path.join(stage_dir, "meta.csv")
	shutil.copy2(meta_src, meta_dst)

	# IMPORTANT: the name `edit_bench.py` uses for mask lookup comes from the meta.csv row
	# (`video_base_name = meta_data['path'].split('.')[0]`). Do not assume it equals
	# `data_video_id`.
	video_base_name = _video_base_name_from_meta(
		meta_csv_path=meta_dst,
		inpainting_sample_id=inpainting_sample_id,
	)

	# Prefer a 'masks/' layout if it already exists in the mounted data, else fall back
	# to our preprocessing 'mask_root/' layout.
	mounted_video_dir = os.path.join(VP_DATA_FUSE_MOUNT_ROOT, data_run_id, data_video_id)
	mounted_masks_preferred = os.path.join(mounted_video_dir, "masks", data_video_id, "all_masks.npz")
	mounted_masks_legacy = os.path.join(mounted_video_dir, "mask_root", data_video_id, "all_masks.npz")

	if os.path.exists(mounted_masks_preferred):
		masks_src = mounted_masks_preferred
	elif os.path.exists(mounted_masks_legacy):
		masks_src = mounted_masks_legacy
	else:
		raise FileNotFoundError(
			f"Missing all_masks.npz for video_id={data_video_id}. "
			f"Looked for: {mounted_masks_preferred} and {mounted_masks_legacy}"
		)

	# Create the path edit_bench.py looks for: <data_dir>/masks/<video_base_name>/all_masks.npz
	staged_masks_dst = os.path.join(stage_dir, "masks", video_base_name, "all_masks.npz")
	_copy_file(masks_src, staged_masks_dst)

	# Also create the legacy repo-layout location that edit_bench.py may fall back to:
	#   ../data/video_inpainting/videovo/<video_id>/all_masks.npz
	# When the working directory is /workspace/VideoPainter, that resolves to:
	#   /workspace/data/video_inpainting/videovo/<video_id>/all_masks.npz
	legacy_masks_dst = os.path.join(
		"/workspace",
		"data",
		"video_inpainting",
		"videovo",
		video_base_name,
		"all_masks.npz",
	)
	_copy_file(masks_src, legacy_masks_dst)

	# Sanity check: edit_bench.py decides between preferred and legacy using os.path.exists,
	# so ensure both exist as real files.
	if not os.path.exists(staged_masks_dst):
		raise RuntimeError(f"Failed to stage masks to preferred path: {staged_masks_dst}")
	if not os.path.exists(legacy_masks_dst):
		raise RuntimeError(f"Failed to stage masks to legacy path: {legacy_masks_dst}")
	logger.info(
		"Staged masks for video_id=%s (video_base_name=%s): preferred=%s legacy=%s (src=%s)",
		data_video_id,
		video_base_name,
		staged_masks_dst,
		legacy_masks_dst,
		masks_src,
	)


	# Stage the raw video file into the exact layout edit_bench.py expects.
	# In edit_bench.py (for '.0.mp4' rows):
	#   video_path = os.path.join(image_or_video_path, video_base_name[:-3], f"{video_base_name}.0.mp4")
	# So we make image_or_video_path point at our staged 'raw_videos' root.
	staged_video_root = os.path.join(stage_dir, "raw_videos")
	expected_filename = f"{video_base_name}.0.mp4"
	video_src = _select_video_file_under_root(
		raw_root=raw_root,
		video_base_name=video_base_name,
		data_video_id=data_video_id,
		expected_filename=expected_filename,
	)
	staged_video_dir = os.path.join(staged_video_root, video_base_name[:-3])
	Path(staged_video_dir).mkdir(parents=True, exist_ok=True)
	staged_video_dst = os.path.join(staged_video_dir, expected_filename)
	ensure_symlink(video_src, staged_video_dst)
	if not os.path.exists(staged_video_dst):
		raise RuntimeError(f"Failed to stage video to expected path: {staged_video_dst}")
	logger.info(
		"Staged video for video_id=%s (video_base_name=%s): %s -> %s",
		data_video_id,
		video_base_name,
		video_src,
		staged_video_dst,
	)

	return meta_dst, staged_video_root


def _list_files_under_root(*, raw_root: str, suffix: str, max_depth: int = 6, max_results: int = 200) -> list[str]:
	"""List files ending with suffix under raw_root (bounded depth)."""
	raw_root_abs = os.path.abspath(raw_root)
	results: list[str] = []
	if not os.path.isdir(raw_root_abs):
		return results
	for root, dirs, files in os.walk(raw_root_abs):
		rel = os.path.relpath(root, raw_root_abs)
		depth = 0 if rel == "." else rel.count(os.sep) + 1
		if depth > max_depth:
			dirs[:] = []
			continue
		for fn in files:
			if fn.endswith(suffix):
				results.append(os.path.join(root, fn))
				if len(results) >= max_results:
					return results
	return results


def _select_video_file_under_root(
	*,
	raw_root: str,
	video_base_name: str,
	data_video_id: str,
	expected_filename: str,
	max_depth: int = 6,
) -> str:
	"""Find the video file matching the expected filename under raw_root."""
	# 1) Exact fast-paths
	exact = os.path.join(raw_root, video_base_name[:-3], expected_filename)
	if os.path.exists(exact):
		return exact
	exact2 = os.path.join(raw_root, expected_filename)
	if os.path.exists(exact2):
		return exact2

	# 2) Scan for candidates
	candidates = _list_files_under_root(raw_root=raw_root, suffix=".mp4", max_depth=max_depth)
	if not candidates:
		raise FileNotFoundError(
			f"No .mp4 files found under raw_root={os.path.abspath(raw_root)} (max_depth={max_depth}). "
			f"Expected something like '{expected_filename}'."
		)
	if len(candidates) == 1:
		return candidates[0]

	# 3) Score candidates
	def score(p: str) -> tuple[int, int]:
		fn = os.path.basename(p)
		# Higher is better.
		s = 0
		if fn == expected_filename:
			s += 100
		elif fn == f"{video_base_name}.mp4":
			s += 90
		elif video_base_name in fn:
			s += 70
		elif data_video_id in fn:
			s += 60
		if fn.endswith(".0.mp4"):
			s += 10
		# Prefer shallower paths
		depth_penalty = p.count(os.sep)
		return (s, -depth_penalty)

	scored = sorted(((score(p), p) for p in candidates), reverse=True)
	best_score, best_path = scored[0]

	# If multiple share the same top score, it's ambiguous; show a short list.
	ties = [p for (sc, p) in scored if sc == best_score]
	if len(ties) > 1:
		preview = "\n".join(ties[:20])
		raise FileNotFoundError(
			f"Ambiguous video selection under raw_root={os.path.abspath(raw_root)}\n"
			f"Expected filename={expected_filename}\n"
			f"Multiple mp4 candidates match equally well. Top candidates (first 20):\n{preview}"
		)

	logger.info(
		"Selected video source for %s (video_base_name=%s): %s (score=%s)",
		data_video_id,
		video_base_name,
		best_path,
		best_score,
	)
	return best_path



def _list_preprocessed_video_ids(*, data_run_id: str) -> list[str]:
	"""List candidate video IDs for a given run_id under the mounted dataset."""
	try:
		entries = os.listdir(os.path.join(VP_DATA_FUSE_MOUNT_ROOT, data_run_id))
	except FileNotFoundError:
		return []
	video_ids = []
	for name in sorted(entries):
		p = os.path.join(VP_DATA_FUSE_MOUNT_ROOT, data_run_id, name)
		if os.path.isdir(p) and (os.path.exists(os.path.join(p, "meta.csv")) or os.path.exists(os.path.join(p, "raw_videos"))):
			video_ids.append(name)
	return video_ids


def ensure_symlink(src: str, dest: str) -> None:
	"""Create a symlink from dest -> src if not already present."""
	dest_parent = Path(dest).parent
	dest_parent.mkdir(parents=True, exist_ok=True)
	if os.path.islink(dest):
		if os.readlink(dest) == src:
			return
		os.unlink(dest)
	elif os.path.exists(dest):
		# Replace existing directory (empty or non-empty) with symlink
		if os.path.isdir(dest):
			try:
				shutil.rmtree(dest)
				logger.info("Removed existing directory at %s to create symlink.", dest)
			except OSError as e:
				logger.warning("Failed to remove directory at %s: %s", dest, e)
				return
		else:
			logger.info("Path %s already exists and is not a symlink; leaving as-is.", dest)
			return
	os.symlink(src, dest)
	logger.info("Created symlink %s -> %s", dest, src)


def ensure_writable_dir(path: str) -> None:
	"""Ensure `path` is a real, writable directory (not a symlink).

	We need this when we want to create additional symlinks *inside* the directory.
	If `path` is a symlink to a read-only FUSE mount, creating new entries under it
	will fail.
	"""
	if os.path.islink(path):
		os.unlink(path)
	if os.path.exists(path):
		if os.path.isdir(path):
			return
		raise RuntimeError(f"Path exists and is not a directory: {path}")
	Path(path).mkdir(parents=True, exist_ok=True)



def _copy_file(src: str, dest: str) -> None:
	"""Copy a file to dest, replacing symlinks/empty dirs if needed.

	We use copies (not symlinks) for masks because `edit_bench.py` uses `os.path.exists`
	to pick a path; with FUSE-backed symlinks that can still be flaky.
	"""
	dest_parent = Path(dest).parent
	dest_parent.mkdir(parents=True, exist_ok=True)

	# If a symlink exists at destination, replace it.
	if os.path.islink(dest):
		os.unlink(dest)
	elif os.path.exists(dest):
		# If an empty directory exists at destination, replace it.
		if os.path.isdir(dest):
			try:
				if not any(Path(dest).iterdir()):
					shutil.rmtree(dest)
				else:
					logger.info("Path %s exists as non-empty directory; leaving as-is.", dest)
					return
			except OSError:
				logger.info("Path %s exists and cannot be replaced safely; leaving as-is.", dest)
				return
		else:
			# File already exists; keep it.
			return

	shutil.copy2(src, dest)
	logger.info("Copied file %s -> %s", src, dest)


def upload_directory_to_gcs(local_dir: str, gcs_prefix: str) -> None:
	"""Recursively upload a local directory to a GCS prefix using gcsfs."""

	fs = gcsfs.GCSFileSystem(token="google_default")
	base = Path(local_dir)
	for path in base.rglob("*"):
		if path.is_dir():
			continue
		rel = path.relative_to(base).as_posix()
		remote = f"{gcs_prefix.rstrip('/')}/{rel}"
		remote_parent = os.path.dirname(remote)
		if remote_parent:
			fs.makedirs(remote_parent, exist_ok=True)
		fs.put(path.as_posix(), remote)


def upload_file_to_gcs(local_path: str, gcs_path: str) -> None:
	"""Upload a single file to a full gs:// path using gcsfs."""
	fs = gcsfs.GCSFileSystem(token="google_default")
	remote_parent = os.path.dirname(gcs_path)
	if remote_parent:
		fs.makedirs(remote_parent, exist_ok=True)
	fs.put(local_path, gcs_path)


def _load_video_frames(video_path: str) -> tuple[list[np.ndarray], float]:
	"""Load video frames from a video file."""
	cap = cv2.VideoCapture(video_path)
	frames = []
	fps = cap.get(cv2.CAP_PROP_FPS)
	
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frames.append(frame_rgb)
	
	cap.release()
	return frames, fps


def _frames_to_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
	"""Convert list of numpy frames to tensor [T, C, H, W]."""
	frames_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
	frames_tensor = frames_tensor.permute(0, 3, 1, 2)
	return frames_tensor.to(device)


def _load_mask_from_npz(mask_path: str, num_frames: int, frame_height: int, frame_width: int) -> np.ndarray:
	"""Load mask frames from npz file."""
	if os.path.exists(mask_path):
		data = np.load(mask_path)
		if 'masks' in data:
			masks = data['masks'][:num_frames]
			if masks.shape[1:3] != (frame_height, frame_width):
				resized_masks = []
				for mask in masks:
					resized = cv2.resize(mask, (frame_width, frame_height))
					resized_masks.append(resized)
				return np.stack(resized_masks)
			return masks
	return np.ones((num_frames, frame_height, frame_width), dtype=np.uint8) * 255


def _evaluate_video(
	*,
	original_path: str,
	generated_path: str,
	mask_path: str,
	caption: str,
	output_dir: str,
	video_id: str,
) -> dict:
	"""Evaluate generated video against original using numerical metrics."""
	try:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# Import metrics calculator
		sys.path.insert(0, os.path.join(BASE_WORKDIR, "evaluate"))
		from metrics import MetricsCalculator
		
		logger.info("[EVAL][%s] Loading videos for evaluation", video_id)
		original_frames, original_fps = _load_video_frames(original_path)
		generated_frames, _ = _load_video_frames(generated_path)
		
		min_frames = min(len(original_frames), len(generated_frames))
		original_frames = original_frames[:min_frames]
		generated_frames = generated_frames[:min_frames]
		
		orig_h, orig_w = original_frames[0].shape[:2]
		gen_h, gen_w = generated_frames[0].shape[:2]
		
		if (orig_h != gen_h) or (orig_w != gen_w):
			generated_frames = [cv2.resize(f, (orig_w, orig_h)) for f in generated_frames]
		
		masks = _load_mask_from_npz(mask_path, min_frames, orig_h, orig_w)
		mask_ratio = (masks > 127).sum() / masks.size
		
		original_tensor = _frames_to_tensor(original_frames, device)
		generated_tensor = _frames_to_tensor(generated_frames, device)
		mask_tensor = torch.from_numpy(masks).float().to(device) / 255.0
		mask_tensor = mask_tensor.unsqueeze(1)
		
		metrics_calculator = MetricsCalculator(device)
		
		# Full-frame metrics
		logger.info("[EVAL][%s] Calculating full-frame metrics", video_id)
		full_metrics = {}
		full_metrics['PSNR'] = metrics_calculator.calculate_psnr(original_tensor, generated_tensor)
		full_metrics['SSIM'] = metrics_calculator.calculate_ssim(original_tensor, generated_tensor)
		full_metrics['LPIPS'] = metrics_calculator.calculate_lpips(original_tensor, generated_tensor)
		full_metrics['MSE'] = metrics_calculator.calculate_mse(original_tensor, generated_tensor)
		full_metrics['MAE'] = metrics_calculator.calculate_mae(original_tensor, generated_tensor)
		full_metrics['Temporal_Consistency'] = metrics_calculator.calculate_temporal_consistency(generated_tensor)
		
		if caption:
			full_metrics['CLIP_Score'] = metrics_calculator.calculate_clip_score(generated_tensor, [caption])
		
		# Masked region metrics
		logger.info("[EVAL][%s] Calculating masked-region metrics", video_id)
		masked_metrics = {}
		mask_binary = (mask_tensor > 0.5).float()
		original_masked = original_tensor * mask_binary
		generated_masked = generated_tensor * mask_binary
		
		masked_metrics['PSNR'] = metrics_calculator.calculate_psnr(original_masked, generated_masked, mask=mask_binary)
		masked_metrics['SSIM'] = metrics_calculator.calculate_ssim(original_masked, generated_masked)
		masked_metrics['LPIPS'] = metrics_calculator.calculate_lpips(original_masked, generated_masked)
		masked_metrics['MSE'] = metrics_calculator.calculate_mse(original_masked, generated_masked)
		masked_metrics['MAE'] = metrics_calculator.calculate_mae(original_masked, generated_masked)
		masked_metrics['Temporal_Consistency'] = metrics_calculator.calculate_temporal_consistency(generated_masked)
		
		if caption:
			masked_metrics['CLIP_Score'] = metrics_calculator.calculate_clip_score(generated_masked, [caption])
		
		# Save results
		Path(output_dir).mkdir(parents=True, exist_ok=True)
		output_path = os.path.join(output_dir, f"eval_{video_id}.txt")
		
		with open(output_path, 'w') as f:
			f.write("="*80 + "\n")
			f.write("VIDEO EVALUATION SUMMARY\n")
			f.write("="*80 + "\n\n")
			f.write(f"Video ID: {video_id}\n")
			f.write(f"Evaluation Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
			f.write(f"Original Video:  {original_path}\n")
			f.write(f"Generated Video: {generated_path}\n")
			f.write(f"Mask:            {mask_path}\n\n")
			f.write(f"Frames:          {min_frames}\n")
			f.write(f"Resolution:      {orig_w} x {orig_h}\n")
			f.write(f"FPS:             {original_fps:.2f}\n")
			f.write(f"Mask Coverage:   {mask_ratio*100:.2f}%\n\n")
			f.write("="*80 + "\n")
			f.write("FULL-FRAME METRICS\n")
			f.write("="*80 + "\n")
			f.write(f"  PSNR:                 {full_metrics['PSNR']:>10.4f} dB\n")
			f.write(f"  SSIM:                 {full_metrics['SSIM']:>10.4f}\n")
			f.write(f"  LPIPS:                {full_metrics['LPIPS']:>10.4f}\n")
			f.write(f"  MSE:                  {full_metrics['MSE']:>10.6f}\n")
			f.write(f"  MAE:                  {full_metrics['MAE']:>10.6f}\n")
			f.write(f"  Temporal Consistency: {full_metrics['Temporal_Consistency']:>10.4f}\n")
			if caption:
				f.write(f"  CLIP Score:           {full_metrics['CLIP_Score']:>10.4f}\n")
			f.write("\n" + "="*80 + "\n")
			f.write("MASKED-REGION METRICS\n")
			f.write("="*80 + "\n")
			f.write(f"  PSNR:                 {masked_metrics['PSNR']:>10.4f} dB\n")
			f.write(f"  SSIM:                 {masked_metrics['SSIM']:>10.4f}\n")
			f.write(f"  LPIPS:                {masked_metrics['LPIPS']:>10.4f}\n")
			f.write(f"  MSE:                  {masked_metrics['MSE']:>10.6f}\n")
			f.write(f"  MAE:                  {masked_metrics['MAE']:>10.6f}\n")
			f.write(f"  Temporal Consistency: {masked_metrics['Temporal_Consistency']:>10.4f}\n")
			if caption:
				f.write(f"  CLIP Score:           {masked_metrics['CLIP_Score']:>10.4f}\n")
			f.write("\n" + "="*80 + "\n")
			
			quality_score = 0
			if full_metrics['PSNR'] > 40:
				quality_score += 1
			if full_metrics['SSIM'] > 0.95:
				quality_score += 1
			if full_metrics['LPIPS'] < 0.1:
				quality_score += 1
			
			f.write("QUALITY ASSESSMENT: ")
			if quality_score >= 2:
				f.write("EXCELLENT\n")
			elif quality_score == 1:
				f.write("GOOD\n")
			else:
				f.write("MODERATE\n")
			f.write("="*80 + "\n")
		
		logger.info("[EVAL][%s] Evaluation complete: PSNR=%.2f SSIM=%.4f LPIPS=%.4f", 
				   video_id, full_metrics['PSNR'], full_metrics['SSIM'], full_metrics['LPIPS'])
		
		return {
			'output_path': output_path,
			'full_metrics': full_metrics,
			'masked_metrics': masked_metrics,
			'mask_coverage': float(mask_ratio),
		}
		
	except Exception as e:
		logger.error("[EVAL][%s] Evaluation failed: %s", video_id, e, exc_info=True)
		return {'error': str(e)}


def _upload_run_source_files(*, gcs_save_path: str) -> list[str]:
	"""Upload key source files for reproducibility.

	Uploads into: <gcs_save_path>/sources/
	"""
	uploaded: list[str] = []
	sources_prefix = os.path.join(gcs_save_path, "sources")

	# These paths are inside the container image.
	candidates: list[tuple[str, str]] = [
		(os.path.join(BASE_WORKDIR, "workflow.py"), "workflow.py"),
		(os.path.join(BASE_WORKDIR, "infer", "edit_bench.py"), "infer/edit_bench.py"),
		# The user-facing launcher script is typically a .sh; accept a .py name too.
		(os.path.join(BASE_WORKDIR, "scripts", "build_and_run.sh"), "scripts/build_and_run.sh"),
		(os.path.join(BASE_WORKDIR, "scripts", "build_and_run.py"), "scripts/build_and_run.py"),
	]

	for local_path, rel_name in candidates:
		if not os.path.exists(local_path):
			logger.info("Source file not found (skipping): %s", local_path)
			continue
		remote_path = os.path.join(sources_prefix, rel_name)
		try:
			upload_file_to_gcs(local_path, remote_path)
			uploaded.append(remote_path)
		except Exception as e:
			logger.info("Failed to upload source file %s -> %s (non-fatal): %s", local_path, remote_path, e)

	if uploaded:
		logger.info("Uploaded %d source file(s) under: %s", len(uploaded), sources_prefix)
	return uploaded


@dataclass
class VPVideoMetrics:
	video_id: str
	output_name: str
	generation_s: float
	upload_s: float
	device: str
	gpu_name: str
	gpu_compute_capability: str
	peak_gpu_mem_allocated_mb: float
	peak_gpu_mem_reserved_mb: float
	rss_mb: float


def _reset_torch_cuda_peaks() -> None:
	try:
		import torch  # type: ignore
		if torch.cuda.is_available():
			torch.cuda.reset_peak_memory_stats(0)
	except Exception:
		return


def _get_torch_cuda_metrics() -> tuple[str, str, str, float, float]:
	"""Return (device, gpu_name, compute_capability, peak_alloc_mb, peak_reserved_mb)."""
	try:
		import torch  # type: ignore
	except Exception:
		return ("unknown", "unknown", "unknown", 0.0, 0.0)

	if not torch.cuda.is_available():
		return ("cpu", "cpu", "n/a", 0.0, 0.0)

	try:
		props = torch.cuda.get_device_properties(0)
		cc = f"{props.major}.{props.minor}"
		name = torch.cuda.get_device_name(0)
		peak_alloc = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
		peak_reserved = torch.cuda.max_memory_reserved(0) / (1024 * 1024)
		return ("cuda", name, cc, float(peak_alloc), float(peak_reserved))
	except Exception:
		return ("cuda", "unknown", "unknown", 0.0, 0.0)


def _get_rss_mb() -> float:
	"""Best-effort RSS (resident set size) in MB."""
	try:
		import psutil  # type: ignore
		p = psutil.Process(os.getpid())
		return float(p.memory_info().rss) / (1024 * 1024)
	except Exception:
		pass
	try:
		# ru_maxrss is KB on Linux.
		rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
		return float(rss_kb) / 1024.0
	except Exception:
		return 0.0


def _write_videopainter_run_report(
	*,
	run_id: str,
	report_path: str,
	per_video: list[VPVideoMetrics],
	input_data_run_id: Optional[str] = None,
) -> None:
	lines: list[str] = []
	lines.append(f"run_id: {run_id}")
	if input_data_run_id is not None and input_data_run_id != run_id:
		lines.append(f"input_data_run_id: {input_data_run_id}")
	lines.append(f"timestamp_utc: {datetime.utcnow().isoformat()}Z")
	lines.append(f"platform: {platform.platform()}")
	lines.append("")
	lines.append("per_video:")
	for m in per_video:
		lines.append(f"- video_id: {m.video_id}")
		lines.append(f"  output_name: {m.output_name}")
		lines.append(f"  generation_s: {m.generation_s:.3f}")
		lines.append(f"  upload_s: {m.upload_s:.3f}")
		lines.append(f"  device: {m.device}")
		lines.append(f"  gpu_name: {m.gpu_name}")
		lines.append(f"  gpu_compute_capability: {m.gpu_compute_capability}")
		lines.append(f"  peak_gpu_mem_allocated_mb: {m.peak_gpu_mem_allocated_mb:.1f}")
		lines.append(f"  peak_gpu_mem_reserved_mb: {m.peak_gpu_mem_reserved_mb:.1f}")
		lines.append(f"  rss_mb: {m.rss_mb:.1f}")

	Path(os.path.dirname(report_path)).mkdir(parents=True, exist_ok=True)
	Path(report_path).write_text("\n".join(lines) + "\n")


def _run_edit_bench(
	*,
	output_path: str,
	inpainting_mask_meta: str,
	image_or_video_path: str,
	prompt: str,
	model_path: str,
	inpainting_branch: str,
	img_inpainting_model: str,
	img_inpainting_lora_path: str,
	img_inpainting_lora_scale: float,
	num_inference_steps: int,
	guidance_scale: float,
	num_videos_per_prompt: int,
	dtype: str,
	inpainting_sample_id: int,
	inpainting_frames: int,
	down_sample_fps: int,
	overlap_frames: int,
	prev_clip_weight: float,
	strength: float,
	video_editing_instruction: str,
	llm_model: str,
	dilate_size: int,
	mask_feather: int,
	caption_refine_iters: int,
	caption_refine_temperature: float,
	keep_masked_pixels: bool,
	seed: int,
) -> None:
	cog_device = (os.environ.get("VP_COG_DEVICE") or "").strip()
	flux_device = (os.environ.get("VP_FLUX_DEVICE") or "").strip()
	qwen_device = (os.environ.get("VP_QWEN_DEVICE") or "").strip()
	unload_qwen = (os.environ.get("VP_UNLOAD_QWEN_AFTER_USE") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
	cmd = [
		sys.executable,
		"infer/edit_bench.py",
		"--model_path",
		model_path,
		"--inpainting_branch",
		inpainting_branch,
		"--output_path",
		output_path,
		"--num_inference_steps",
		str(num_inference_steps),
		"--guidance_scale",
		str(guidance_scale),
		"--num_videos_per_prompt",
		str(num_videos_per_prompt),
		"--dtype",
		dtype,
		"--generate_type",
		"i2v_inpainting",
		"--inpainting_mask_meta",
		inpainting_mask_meta,
		"--inpainting_sample_id",
		str(inpainting_sample_id),
		"--inpainting_frames",
		str(inpainting_frames),
		"--image_or_video_path",
		image_or_video_path,
		"--down_sample_fps",
		str(down_sample_fps),
		"--overlap_frames",
		str(overlap_frames),
		"--prev_clip_weight",
		str(prev_clip_weight),
		"--strength",
		str(float(strength)),
		"--img_inpainting_model",
		img_inpainting_model,
		"--img_inpainting_lora_path",
		img_inpainting_lora_path,
		"--img_inpainting_lora_scale",
		str(float(img_inpainting_lora_scale)),
		"--video_editing_instruction",
		video_editing_instruction,
		"--llm_model",
		llm_model,
		"--dilate_size",
		str(dilate_size),
		"--mask_feather",
		str(mask_feather),
		"--caption_refine_iters",
		str(int(caption_refine_iters or 0)),
		"--caption_refine_temperature",
		str(float(caption_refine_temperature or 0.2)),
	]
	if qwen_device:
		cmd.extend(["--qwen_device", qwen_device])
	if unload_qwen:
		cmd.append("--unload_qwen_after_caption")
	if keep_masked_pixels:
		cmd.append("--keep_masked_pixels")
	cmd.extend([
		"--seed",
		str(seed),
	])
	if cog_device:
		cmd.extend(["--cog_device", cog_device])
	if flux_device:
		cmd.extend(["--flux_device", flux_device])
	cmd.extend([
		"--first_frame_gt",
		"--replace_gt",
		"--mask_add",
	])
	if prompt:
		cmd.extend(["--prompt", prompt])

	logger.info("Running command: %s", " ".join(cmd))
	result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_WORKDIR)
	if result.stdout:
		logger.info(result.stdout)
	if result.stderr:
		# tqdm/progress bars often go to stderr even on success.
		logger.info(result.stderr)
	if result.returncode != 0:
		stdout_tail = (result.stdout or "")[-4000:]
		stderr_tail = (result.stderr or "")[-4000:]
		raise RuntimeError(
			f"VideoPainter edit failed with code {result.returncode}\n"
			f"--- stdout (tail) ---\n{stdout_tail}\n"
			f"--- stderr (tail) ---\n{stderr_tail}"
		)


def _upload_outputs(
	*,
	output_path: str,
	remote_output_path: str,
) -> list[str]:
	"""Upload main and optional sidecar outputs; return uploaded gs:// paths."""
	uploaded: list[str] = []
	upload_file_to_gcs(output_path, remote_output_path)
	uploaded.append(remote_output_path)

	# Upload common sidecar artifacts produced by infer/edit_bench.py (when enabled).
	# These are very helpful for debugging prompt/mask issues.
	sidecars = [
		output_path.replace(".mp4", ".json"),
		output_path + "_flux_i_img.png",
		output_path + "_flux_i_mask.png",
		output_path + "_flux_o_img.png",
		output_path + "_gt_o_img.png",
	]

	# Optional iterative-refinement artifacts (only when caption_refine_iters > 0).
	# Stored under a dedicated per-video folder: <output_basename>_caption_refine/
	try:
		from glob import glob
		base_no_ext = os.path.splitext(os.path.basename(output_path))[0]
		refine_dir = os.path.join(os.path.dirname(output_path), f"{base_no_ext}_caption_refine")
		for p in sorted(glob(os.path.join(refine_dir, "*"))):
			sidecars.append(p)
	except Exception:
		pass
	for local_path in sidecars:
		if not os.path.exists(local_path):
			continue

		# Preserve folder structure for refinement artifacts.
		if local_path.endswith(".json") and os.path.basename(local_path) == os.path.basename(output_path).replace(".mp4", ".json"):
			remote_path = remote_output_path.replace(".mp4", ".json")
		else:
			remote_dir = os.path.dirname(remote_output_path)
			refine_dir_basename = f"{os.path.splitext(os.path.basename(output_path))[0]}_caption_refine"
			if os.path.basename(os.path.dirname(local_path)) == refine_dir_basename:
				remote_path = os.path.join(remote_dir, refine_dir_basename, os.path.basename(local_path))
			elif local_path.startswith(output_path):
				suffix = local_path[len(output_path):]
				remote_path = remote_output_path + suffix
			else:
				remote_path = os.path.join(remote_dir, os.path.basename(local_path))

		upload_file_to_gcs(local_path, remote_path)
		uploaded.append(remote_path)

	generated_only = output_path.replace(".mp4", "_generated.mp4")
	if os.path.exists(generated_only):
		remote_generated = remote_output_path.replace(".mp4", "_generated.mp4")
		upload_file_to_gcs(generated_only, remote_generated)
		uploaded.append(remote_generated)
	return uploaded


@task(
	compute=DedicatedNode(
		node=COMPUTE_NODE,
		ephemeral_storage="max",
		max_duration="3d",
	),
	container_image=CONTAINER_IMAGE,
	environment={
		"PYTHONUNBUFFERED": "1",
		"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
		"VP_COG_DEVICE": "cuda:0",
		"VP_FLUX_DEVICE": VP_FLUX_DEVICE_DEFAULT,
		# Let infer/edit_bench.py auto-pick a different GPU for Qwen when available.
		"VP_QWEN_DEVICE": "auto",
		"VP_UNLOAD_QWEN_AFTER_USE": "1",
	},
	mounts=[
		FuseBucket(
			bucket=VP_BUCKET,
			name=VP_FUSE_MOUNT_NAME,
			prefix=VP_BUCKET_PREFIX,
		),
		FuseBucket(
			bucket=VP_BUCKET,
			name=VP_DATA_FUSE_MOUNT_NAME,
			prefix=VP_DATA_PREFIX,
		),
		FuseBucket(
			bucket=VP_BUCKET,
			name=TRAINED_FLUXFILL_FUSE_MOUNT_NAME,
			prefix=TRAINED_FLUXFILL_GCS_PREFIX,
		),
		*([
			FuseBucket(
				bucket=VP_BUCKET,
				name=VP_VLM_7B_FUSE_MOUNT_NAME,
				prefix=VLM_7B_GCS_PREFIX,
			),
		] if USE_QWEN2_5_VL_7B else []),
	],
)
def run_videopainter_edit_many(
	*,
	data_run_id: str = DEFAULT_DATA_RUN_ID,
	output_run_id: Optional[str] = None,
	data_video_ids: str = "auto",
	data_video_id: Optional[str] = None,
	inpainting_sample_id: int = 0,
	prompt: str = "",
	model_path: str = DEFAULT_MODEL_PATH,
	inpainting_branch: str = DEFAULT_BRANCH_PATH,
	img_inpainting_model: str = DEFAULT_IMG_INPAINT_PATH,
	img_inpainting_lora_path: str = "",
	img_inpainting_lora_scale: float = 1.0,
	output_name_suffix: str = "vp_edit.mp4",
	num_inference_steps: int = 50,
	guidance_scale: float = 6.0,
	num_videos_per_prompt: int = 1,
	dtype: str = "bfloat16",
	inpainting_frames: int = 49,
	down_sample_fps: int = 8,
	overlap_frames: int = 0,
	prev_clip_weight: float = 0.0,
	strength: float = 1.0,
	# Lane-line spec options (optional):
	# - Provide a single spec via lane_count/lane_color/lane_pattern, OR
	# - Provide many specs via lane_specs (same delimiter rules as video_editing_instructions).
	lane_count: str = "",
	lane_color: str = "",
	lane_pattern: str = "",
	lane_specs: str = "",
	video_editing_instruction: str = "auto",
	video_editing_instructions: str = "",
	# Default to disabling the LLM-based prompt editing/captioning so the workflow
	# does not require `transformers` / `qwen-vl-utils` inside the container.
	# To enable, pass a non-disabled string and ensure dependencies are installed.
	llm_model: str = "disabled",
	dilate_size: int = 0,
	mask_feather: int = 0,
	caption_refine_iters: int = 0,
	caption_refine_temperature: float = 0.2,
	keep_masked_pixels: bool = False,
	seed: int = 42,
) -> str:
	"""Process multiple videos ('auto' for all, or comma-separated video IDs)."""

	logger.info("MOUNTPOINT=%s", MOUNTPOINT)
	logger.info("VP_FUSE_MOUNT_ROOT=%s", VP_FUSE_MOUNT_ROOT)
	logger.info("MOUNTED_CKPT_PATH=%s", MOUNTED_CKPT_PATH)
	logger.info("VP_DATA_FUSE_MOUNT_ROOT=%s", VP_DATA_FUSE_MOUNT_ROOT)
	logger.info("DEFAULT_CKPT_DIR=%s", DEFAULT_CKPT_DIR)
	logger.info("data_run_id=%s", data_run_id)
	effective_output_run_id = (output_run_id or data_run_id).strip()
	if not effective_output_run_id:
		raise ValueError("output_run_id must be non-empty when provided")
	logger.info("output_run_id=%s", effective_output_run_id)
	logger.info("USE_QWEN2_5_VL_7B=%s", USE_QWEN2_5_VL_7B)

	# Prefetch checkpoints from GCS.
	fuse_prefetch_metadata(MOUNTED_CKPT_PATH)

	# If we mount a dedicated VLM folder (7B), keep ckpt as a writable dir and
	# symlink required subfolders so we can add ckpt/vlm/Qwen2.5-VL-7B-Instruct.
	if not USE_QWEN2_5_VL_7B:
		ensure_symlink(MOUNTED_CKPT_PATH, DEFAULT_CKPT_DIR)
	else:
		ensure_writable_dir(DEFAULT_CKPT_DIR)
		for rel in ("CogVideoX-5b-I2V", "VideoPainter", "flux_inp"):
			src = os.path.join(MOUNTED_CKPT_PATH, rel)
			dest = os.path.join(DEFAULT_CKPT_DIR, rel)
			if os.path.exists(src):
				ensure_symlink(src, dest)
		ensure_writable_dir(os.path.join(DEFAULT_CKPT_DIR, "vlm"))

		try:
			fuse_prefetch_metadata(VP_VLM_7B_FUSE_MOUNT_ROOT)
		except Exception:
			logger.info("VLM 7B prefetch skipped/failed; continuing.")

		# Create symlink for VLM 7B mount
		ensure_symlink(VP_VLM_7B_FUSE_MOUNT_ROOT, VLM_7B_DEST_PATH)

	# Mount and symlink the trained FluxFill checkpoint
	try:
		fuse_prefetch_metadata(TRAINED_FLUXFILL_FUSE_MOUNT_ROOT)
	except Exception:
		logger.info("Trained FluxFill prefetch skipped/failed; continuing.")
	ensure_symlink(TRAINED_FLUXFILL_FUSE_MOUNT_ROOT, TRAINED_FLUXFILL_DEST_PATH)
	logger.info("Trained FluxFill checkpoint mounted at: %s", TRAINED_FLUXFILL_DEST_PATH)

	# Keep /workspace/VideoPainter/data present and pointing at the mounted dataset.
	ensure_symlink(VP_DATA_FUSE_MOUNT_ROOT, DEFAULT_DATA_DIR)

	try:
		fuse_prefetch_metadata(VP_DATA_FUSE_MOUNT_ROOT)
	except Exception:
		logger.info("Data prefetch skipped/failed; continuing.")

	# Resolve which video IDs to run
	video_ids: list[str]
	if data_video_id and data_video_id.strip().lower() != "auto":
		video_ids = [data_video_id.strip()]
	elif not data_video_ids or data_video_ids.strip().lower() == "auto":
		video_ids = _list_preprocessed_video_ids(data_run_id=data_run_id)
	else:
		normalized = data_video_ids.replace(",", " ")
		video_ids = [v.strip() for v in normalized.split() if v.strip()]

	if not video_ids:
		raise RuntimeError(f"No videos found at gs://{VP_BUCKET}/{VP_DATA_PREFIX}/{data_run_id}")

	# Validate model inputs exist before running the heavy command.
	paths_to_check = {
		"model_path": model_path,
		"inpainting_branch": inpainting_branch,
		"img_inpainting_model": img_inpainting_model,
	}
	missing = {k: v for k, v in paths_to_check.items() if v and not os.path.exists(v)}
	if missing:
		for k, v in missing.items():
			logger.error("Missing path (%s): %s", k, v)
		raise RuntimeError(f"Required paths missing: {', '.join(missing.keys())}")

	# Stage outputs to scratch then upload once to GCS.
	staged_output_dir = os.path.join(SCRATCH_BASE, effective_output_run_id)
	gcs_save_path = os.path.join(GCS_OUTPUT_BASE, effective_output_run_id)
	Path(staged_output_dir).mkdir(parents=True, exist_ok=True)

	logger.info("GCS output folder for this run: %s", gcs_save_path)
	logger.info("Running %d videos: %s", len(video_ids), ", ".join(video_ids[:50]))

	# Build instruction(s) from explicit lane specs if provided.
	# This lets callers pass only the 3 fields we care about (single/double, white/yellow, continuous/intermittent).
	# The final strings are still passed as --video_editing_instruction to infer/edit_bench.py.
	instruction_list: list[str] = []
	if lane_specs and str(lane_specs).strip():
		instruction_list = _parse_instruction_list(lane_specs)
	elif (lane_count or lane_color or lane_pattern) and _lane_spec_to_instruction(lane_count, lane_color, lane_pattern):
		instruction_list = [_lane_spec_to_instruction(lane_count, lane_color, lane_pattern)]
	else:
		instruction_list = _parse_instruction_list(video_editing_instructions)
		if not instruction_list:
			instruction_list = [video_editing_instruction]

	logger.info("Running %d instruction(s)", len(instruction_list))

	# Run each instruction into its own subfolder.
	used_instr_dirs: set[str] = set()
	for instr_idx, instruction in enumerate(instruction_list, start=1):
		instr_dir = _sanitize_folder_component(instruction, max_len=120)
		if instr_dir in used_instr_dirs:
			instr_dir = f"{instr_dir}_{instr_idx:02d}"
		used_instr_dirs.add(instr_dir)
		logger.info("Instruction %d/%d -> %s", instr_idx, len(instruction_list), instr_dir)

		per_video_metrics: list[VPVideoMetrics] = []
		for vid in video_ids:
			logger.info("[%s][%s] Starting", instr_dir, vid)
			inpainting_mask_meta, video_root = _stage_preprocessed_inputs(
				data_run_id=data_run_id,
				data_video_id=vid,
				inpainting_sample_id=inpainting_sample_id,
			)

			output_name = f"{vid}_{output_name_suffix}" if output_name_suffix else f"{vid}_vp_edit.mp4"
			output_path = os.path.join(staged_output_dir, instr_dir, vid, output_name)
			Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

			_reset_torch_cuda_peaks()
			gen_start = time.perf_counter()
			_run_edit_bench(
				output_path=output_path,
				inpainting_mask_meta=inpainting_mask_meta,
				image_or_video_path=video_root,
				prompt=prompt,
				model_path=model_path,
				inpainting_branch=inpainting_branch,
				img_inpainting_model=img_inpainting_model,
				img_inpainting_lora_path=img_inpainting_lora_path,
				img_inpainting_lora_scale=img_inpainting_lora_scale,
				num_inference_steps=num_inference_steps,
				guidance_scale=guidance_scale,
				num_videos_per_prompt=num_videos_per_prompt,
				dtype=dtype,
				inpainting_sample_id=inpainting_sample_id,
				inpainting_frames=inpainting_frames,
				down_sample_fps=down_sample_fps,
				overlap_frames=overlap_frames,
				prev_clip_weight=prev_clip_weight,
				strength=strength,
				video_editing_instruction=instruction,
				llm_model=llm_model,
				dilate_size=dilate_size,
				mask_feather=mask_feather,
				caption_refine_iters=caption_refine_iters,
				caption_refine_temperature=caption_refine_temperature,
				keep_masked_pixels=keep_masked_pixels,
				seed=seed,
			)
			gen_s = time.perf_counter() - gen_start
			device, gpu_name, gpu_cc, peak_alloc_mb, peak_reserved_mb = _get_torch_cuda_metrics()
			rss_mb = _get_rss_mb()

			remote_output = os.path.join(gcs_save_path, instr_dir, vid, output_name)
			logger.info("[%s][%s] Uploading to %s", instr_dir, vid, remote_output)
			upload_start = time.perf_counter()
			uploaded = _upload_outputs(output_path=output_path, remote_output_path=remote_output)
			upload_s = time.perf_counter() - upload_start
			logger.info("[%s][%s] Uploaded: %s", instr_dir, vid, ", ".join(uploaded))
			
			# Run evaluation on the generated video
			logger.info("[%s][%s] Running evaluation", instr_dir, vid)
			eval_start = time.perf_counter()
			try:
				# Find original video and mask
				meta_path, raw_root = _resolve_preprocessed_video_paths(
					data_run_id=data_run_id,
					data_video_id=vid,
				)
				video_base_name = _video_base_name_from_meta(
					meta_csv_path=inpainting_mask_meta,
					inpainting_sample_id=inpainting_sample_id,
				)
				
				# Find original video file
				expected_filename = f"{video_base_name}.0.mp4"
				original_video_path = _select_video_file_under_root(
					raw_root=raw_root,
					video_base_name=video_base_name,
					data_video_id=vid,
					expected_filename=expected_filename,
				)
				
				# Find mask file
				mounted_video_dir = os.path.join(VP_DATA_FUSE_MOUNT_ROOT, data_run_id, vid)
				mounted_masks_preferred = os.path.join(mounted_video_dir, "masks", vid, "all_masks.npz")
				mounted_masks_legacy = os.path.join(mounted_video_dir, "mask_root", vid, "all_masks.npz")
				
				mask_path = mounted_masks_preferred if os.path.exists(mounted_masks_preferred) else mounted_masks_legacy
				
				# Run evaluation - save in same directory as video
				eval_output_dir = os.path.join(staged_output_dir, instr_dir, vid)
				eval_result = _evaluate_video(
					original_path=original_video_path,
					generated_path=output_path,
					mask_path=mask_path,
					caption=instruction,
					output_dir=eval_output_dir,
					video_id=vid,
				)
				
				if 'error' not in eval_result:
					# Upload evaluation results to GCS in same directory as video
					eval_remote_path = os.path.join(gcs_save_path, instr_dir, vid, f"eval_{vid}.txt")
					upload_file_to_gcs(eval_result['output_path'], eval_remote_path)
					logger.info("[%s][%s] Evaluation uploaded: %s", instr_dir, vid, eval_remote_path)
				else:
					logger.warning("[%s][%s] Evaluation failed: %s", instr_dir, vid, eval_result['error'])
				
			except Exception as e:
				logger.warning("[%s][%s] Evaluation failed: %s", instr_dir, vid, e)
			
			eval_s = time.perf_counter() - eval_start
			logger.info("[%s][%s] Evaluation time: %.2fs", instr_dir, vid, eval_s)
			logger.info("[%s][%s] Done", instr_dir, vid)

			per_video_metrics.append(
				VPVideoMetrics(
					video_id=vid,
					output_name=os.path.join(instr_dir, vid, output_name),
					generation_s=float(gen_s),
					upload_s=float(upload_s),
					device=device,
					gpu_name=gpu_name,
					gpu_compute_capability=gpu_cc,
					peak_gpu_mem_allocated_mb=float(peak_alloc_mb),
					peak_gpu_mem_reserved_mb=float(peak_reserved_mb),
					rss_mb=float(rss_mb),
				)
			)

		# Write report inside the instruction folder.
		try:
			report_local = os.path.join(staged_output_dir, instr_dir, f"{effective_output_run_id}.txt")
			_write_videopainter_run_report(
				run_id=effective_output_run_id,
				report_path=report_local,
				input_data_run_id=data_run_id,
				per_video=per_video_metrics,
			)
			report_remote = os.path.join(gcs_save_path, instr_dir, f"{effective_output_run_id}.txt")
			upload_file_to_gcs(report_local, report_remote)
			logger.info("Uploaded VideoPainter run report: %s", report_remote)
		except Exception as e:
			logger.info("Failed to write/upload VideoPainter run report (non-fatal): %s", e)

	logger.info("All videos completed. Outputs saved under: %s", gcs_save_path)

	# Upload workflow + inference entrypoints used for this run (best-effort)
	try:
		_upload_run_source_files(gcs_save_path=gcs_save_path)
	except Exception as e:
		logger.info("Failed to upload source files (non-fatal): %s", e)
	return gcs_save_path


@workflow
def videopainter_many_wf(
	data_run_id: str = DEFAULT_DATA_RUN_ID,
	output_run_id: Optional[str] = None,
	data_video_ids: str = "auto",
	inpainting_sample_id: int = 0,
	prompt: str = "",
	model_path: str = DEFAULT_MODEL_PATH,
	inpainting_branch: str = DEFAULT_BRANCH_PATH,
	img_inpainting_model: str = DEFAULT_IMG_INPAINT_PATH,
	img_inpainting_lora_path: str = "",
	img_inpainting_lora_scale: float = 1.0,
	output_name_suffix: str = "vp_edit.mp4",
	num_inference_steps: int = 50,
	guidance_scale: float = 6.0,
	num_videos_per_prompt: int = 1,
	dtype: str = "bfloat16",
	inpainting_frames: int = 49,
	down_sample_fps: int = 8,
	overlap_frames: int = 0,
	prev_clip_weight: float = 0.0,
	strength: float = 1.0,
	video_editing_instruction: str = "auto",
	video_editing_instructions: str = "",
	llm_model: str = "disabled",
	dilate_size: int = 0,
	mask_feather: int = 0,
	caption_refine_iters: int = 0,
	caption_refine_temperature: float = 0.2,
	keep_masked_pixels: bool = False,
	seed: int = 42,
) -> str:
	"""Process multiple videos (comma-separated IDs or 'auto' for all)."""
	return run_videopainter_edit_many(
		data_run_id=data_run_id,
		output_run_id=output_run_id,
		data_video_ids=data_video_ids,
		inpainting_sample_id=inpainting_sample_id,
		prompt=prompt,
		model_path=model_path,
		inpainting_branch=inpainting_branch,
		img_inpainting_model=img_inpainting_model,
		img_inpainting_lora_path=img_inpainting_lora_path,
		img_inpainting_lora_scale=img_inpainting_lora_scale,
		output_name_suffix=output_name_suffix,
		num_inference_steps=num_inference_steps,
		guidance_scale=guidance_scale,
		num_videos_per_prompt=num_videos_per_prompt,
		dtype=dtype,
		inpainting_frames=inpainting_frames,
		down_sample_fps=down_sample_fps,
		overlap_frames=overlap_frames,
		prev_clip_weight=prev_clip_weight,
		strength=strength,
		video_editing_instruction=video_editing_instruction,
		video_editing_instructions=video_editing_instructions,
		llm_model=llm_model,
		dilate_size=dilate_size,
		mask_feather=mask_feather,
		caption_refine_iters=caption_refine_iters,
		caption_refine_temperature=caption_refine_temperature,
		keep_masked_pixels=keep_masked_pixels,
		seed=seed,
	)
