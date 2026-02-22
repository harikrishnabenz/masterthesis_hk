"""Generate FluxFill LoRA training data from a GCS folder of .mp4 files.

This script lives under `segmentation/sam2/` so it can directly reuse the SAM2
installation + configs in this repo.

Pipeline:
    1) List + download a slice of mp4s from GCS
    2) Mount Qwen2.5-VL-* model (local HF folder, default: Qwen2.5-VL-7B-Instruct)
  3) Mount SAM2 checkpoint
  4) For N selected mp4s:
    - extract selected frames (by frame number)
     - generate mask using SAM2 (image predictor + fixed road-style points)
    - generate caption using Qwen2.5-VL model
  5) Write FluxFill training structure:
       images/*.png
       masks/*.png
       train.csv (image,mask,prompt,prompt_2) with *relative* paths
  6) Upload to:
       gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data/<run_id>/

CSV produced is compatible with VideoPainter FluxFill trainer:
  generation/VideoPainter/train/train_fluxfill_inpaint_lora.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Iterable, Optional

import gcsfs
from PIL import Image

from hlx.wf import DedicatedNode, Node, fuse_prefetch_metadata, task, workflow
from hlx.wf.mounts import MOUNTPOINT, FuseBucket

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# IMAGES / CONTAINER
# --------------------------------------------------------------------------------------
# This data-generation step needs: torch+transformers+qwen-vl-utils+ffmpeg + sam2.
# If your sam2 container already includes transformers/qwen, set SAM2_CONTAINER_IMAGE
# accordingly. Otherwise, point this env var at a container that has them.
CONTAINER_IMAGE_DEFAULT = os.environ.get("FLUXFILL_DATA_CONTAINER_IMAGE") or os.environ.get(
    "SAM2_CONTAINER_IMAGE", "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2"
)
CONTAINER_IMAGE = os.environ.get("FLUXFILL_DATA_CONTAINER_IMAGE", CONTAINER_IMAGE_DEFAULT)


# --------------------------------------------------------------------------------------
# GCS CONFIG
# --------------------------------------------------------------------------------------
BUCKET = "mbadas-sandbox-research-9bb9c7f"

SOURCE_GCS_PREFIX = "workspace/user/hbaskar/Input/data_physical_ai/camera_front_tele_30fov"
DEST_GCS_PREFIX_BASE = "workspace/user/hbaskar/Video_inpainting/videopainter/training/data"

QWEN_MODEL_GCS_PREFIX = "workspace/user/hbaskar/Video_inpainting/videopainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct"
QWEN_MODEL_FUSE_NAME = "vlm-7b"
QWEN_MODEL_FUSE_ROOT = os.path.join(MOUNTPOINT, QWEN_MODEL_FUSE_NAME)

SAM2_CHECKPOINT_GCS_PREFIX = "workspace/user/hbaskar/Video_inpainting/sam2_checkpoint"
SAM2_FUSE_NAME = "sam2-checkpoints"
SAM2_FUSE_ROOT = os.path.join(MOUNTPOINT, SAM2_FUSE_NAME)
SAM2_CHECKPOINT_MOUNTED_PATH = os.path.join(SAM2_FUSE_ROOT, "checkpoints", "sam2.1_hiera_large.pt")


# --------------------------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------------------------

def _stable_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]


def _extract_frame_ffmpeg(video_path: str, *, frame_number: int, out_png_path: str) -> None:
    """Extract a specific frame (1-based) from a video into a PNG.

    Uses ffmpeg's frame-index selector (0-based internally).
    """
    if int(frame_number) <= 0:
        raise ValueError(f"frame_number must be >= 1, got {frame_number}")

    frame0 = int(frame_number) - 1
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"select=eq(n\\,{frame0})",
        "-vsync",
        "vfr",
        "-frames:v",
        "1",
        out_png_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:  # pragma: no cover
        raise RuntimeError("ffmpeg is not available in this environment") from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed for {video_path}: {err}")


def _extract_frames_ffmpeg_batch(
    video_path: str,
    *,
    frame_numbers: list[int],
    out_png_paths: list[str],
) -> None:
    """Extract multiple frames in a single ffmpeg invocation.

    Builds a select filter like  select='eq(n,0)+eq(n,249)+eq(n,499)'
    and outputs each selected frame to a sequentially-numbered file,
    then renames them to the desired paths.
    """
    if not frame_numbers or len(frame_numbers) != len(out_png_paths):
        raise ValueError("frame_numbers and out_png_paths must have equal non-zero length")

    # Build a single select expression for all requested frames
    selects = "+".join(f"eq(n\\,{int(fn) - 1})" for fn in frame_numbers)

    # ffmpeg will output frame_001.png, frame_002.png, … in order of appearance
    out_dir = os.path.dirname(out_png_paths[0]) or "."
    tmp_pattern = os.path.join(out_dir, f"_tmp_{_stable_id(video_path)}_%03d.png")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", f"select='{selects}'",
        "-vsync", "vfr",
        tmp_pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg is not available in this environment") from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip()
        raise RuntimeError(f"ffmpeg batch extract failed for {video_path}: {err}")

    # Rename sequential outputs to the requested paths
    for i, dest in enumerate(out_png_paths):
        src = tmp_pattern % (i + 1)
        if os.path.exists(src):
            os.rename(src, dest)
        else:
            # Fallback: frame may not exist (e.g. video shorter than requested)
            logger.warning("Frame %d not extracted from %s (video too short?)", frame_numbers[i], video_path)


def _parse_frame_numbers_csv(frame_numbers: str | None) -> list[int]:
    """Parse a comma-separated list of 1-based frame numbers.

    Empty/None => [1].
    """
    s = (frame_numbers or "").strip()
    if not s:
        return [1]
    out: list[int] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        n = int(p)
        if n <= 0:
            raise ValueError(f"Invalid frame number '{p}' (must be >= 1)")
        out.append(n)
    # deterministic order + avoid duplicates
    return sorted(set(out))


def _upload_directory_to_gcs(local_dir: str, gcs_prefix: str) -> None:
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


def _gcs_strip_scheme(uri: str) -> str:
    return uri[len("gs://") :] if uri.startswith("gs://") else uri


def _upload_file_to_gcs(fs: gcsfs.GCSFileSystem, local_path: str, gcs_uri: str, log_upload: bool = True) -> None:
    remote = _gcs_strip_scheme(gcs_uri)
    remote_parent = os.path.dirname(remote)
    if remote_parent:
        fs.makedirs(remote_parent, exist_ok=True)
    fs.put(local_path, remote)
    if log_upload:
        logger.info("Uploaded: %s -> gs://%s", os.path.basename(local_path), remote)


def _list_mp4s_in_gcs_prefix(
    *,
    bucket: str,
    prefix: str,
    start_index: int,
    limit: int,
    max_list_files: int,
    sort_results: bool,
    chunk_start: int = 0,
    chunk_end: int = 200,
) -> list[str]:
    """List mp4 objects under gs://bucket/prefix and return a sliced list.

    When chunk_start/chunk_end are provided, only searches chunk_XXXX folders
    in that range (inclusive).

    Returns fully-qualified gs:// URIs.
    """
    fs = gcsfs.GCSFileSystem(token="google_default")
    root = f"{bucket}/{prefix.strip('/')}"

    # Check if we're using chunk-based structure
    if chunk_start is not None and chunk_end is not None:
        logger.info("Using chunk-based listing: chunk_%04d to chunk_%04d", chunk_start, chunk_end)
        return _list_mp4s_in_chunks(
            fs=fs,
            root=root,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            start_index=start_index,
            limit=limit,
            sort_results=sort_results,
        )

    # NOTE: Listing large prefixes in GCS can take a long time.
    # By default we *do not* sort, and we stop once we have enough results to
    # cover the requested slice. This makes the job start processing quickly.
    # If you need deterministic, lexicographically-sorted selection, set
    # sort_results=True (slower; requires scanning up to max_list_files).

    needed = int(start_index) + int(limit)
    if needed <= 0:
        return []

    all_paths: list[str] = []
    scanned = 0
    max_list = int(max_list_files)
    if max_list <= 0:
        max_list = 0  # 0 means unlimited
    for i, p in enumerate(fs.find(root)):
        scanned = i + 1
        if max_list and scanned > max_list:
            logger.warning("Reached max_list_files=%d while listing %s", max_list, root)
            break
        if not p.lower().endswith(".mp4"):
            continue
        all_paths.append(p)

        # Fast path: once we have enough mp4s to cover the slice, stop.
        if not sort_results and len(all_paths) >= needed:
            break

        if scanned % 50000 == 0:
            logger.info("Listing %s: scanned=%d, mp4s_found=%d", root, scanned, len(all_paths))

    if sort_results:
        all_paths.sort()

    sliced = all_paths[int(start_index) : int(start_index) + int(limit)]
    return [f"gs://{p}" for p in sliced]


def _download_gcs_files(
    *,
    uris: list[str],
    out_dir: str,
) -> list[str]:
    """Download gs:// URIs to out_dir and return local paths."""
    fs = gcsfs.GCSFileSystem(token="google_default")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    local_paths: list[str] = []
    for i, uri in enumerate(uris):
        if not uri.startswith("gs://"):
            raise ValueError(f"Expected gs:// URI, got: {uri}")

        # Convert gs://bucket/key -> bucket/key for gcsfs
        remote = uri[len("gs://") :]
        base = os.path.basename(remote)
        if not base:
            base = f"video_{i:06d}.mp4"
        local = os.path.join(out_dir, base)
        fs.get(remote, local)
        local_paths.append(local)

    return local_paths


def _list_mp4s_in_chunks(
    *,
    fs: gcsfs.GCSFileSystem,
    root: str,
    chunk_start: int,
    chunk_end: int,
    start_index: int,
    limit: int,
    sort_results: bool,
) -> list[str]:
    """List mp4 files from chunk_XXXX folders in the specified range."""
    needed = int(start_index) + int(limit)
    if needed <= 0:
        return []

    all_paths: list[str] = []
    
    # Process chunks in order
    for chunk_num in range(int(chunk_start), int(chunk_end) + 1):
        chunk_folder = f"{root}/chunk_{chunk_num:04d}"
        logger.info("Scanning chunk folder: %s", chunk_folder)
        
        try:
            chunk_files = fs.find(chunk_folder)
            mp4s = [p for p in chunk_files if p.lower().endswith(".mp4")]
            all_paths.extend(mp4s)
            logger.info("Found %d mp4s in %s (total so far: %d)", len(mp4s), chunk_folder, len(all_paths))
            
            # Early stop if we have enough files and not sorting
            if not sort_results and len(all_paths) >= needed:
                logger.info("Early stop: collected enough mp4s (%d >= %d)", len(all_paths), needed)
                break
        except FileNotFoundError:
            logger.warning("Chunk folder not found: %s", chunk_folder)
            continue

    if sort_results:
        all_paths.sort()

    sliced = all_paths[int(start_index) : int(start_index) + int(limit)]
    return [f"gs://{p}" for p in sliced]


def _local_mp4_name_for_uri(uri: str, index: int) -> str:
    # Avoid collisions when many objects share the same basename.
    # Use a stable hash so re-runs produce predictable local names.
    sid = _stable_id(uri)
    return f"{int(index):06d}_{sid}.mp4"


# --------------------------------------------------------------------------------------
# QWEN CAPTIONING
# --------------------------------------------------------------------------------------
# Store models per device to support multiple GPUs
_qwen_models = {}  # device -> (model, processor, model_path)
_process_vision_info = None


def _get_qwen_model(model_path: str, *, qwen_device: str = "cuda:1"):
    global _qwen_models
    
    # Check if model already loaded for this device
    if qwen_device in _qwen_models:
        cached_model, cached_processor, cached_path = _qwen_models[qwen_device]
        if cached_path == model_path:
            return cached_model, cached_processor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _Qwen, AutoProcessor
    except Exception as e:  # pragma: no cover
        raise ImportError("transformers is required for Qwen captioning") from e

    import torch

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Qwen model folder not found: {model_path}")

    load_kwargs: dict = {"local_files_only": True, "low_cpu_mem_usage": True, "device_map": "auto"}

    # Bias placement to a specific GPU (keeps cuda:0 free for SAM2)
    device_key = (qwen_device or "auto").strip().lower()
    if torch.cuda.is_available() and device_key.startswith("cuda"):
        gpu_count = torch.cuda.device_count()
        target = 1
        if device_key == "cuda":
            target = 0
        elif device_key.startswith("cuda:"):
            try:
                target = int(device_key.split(":", 1)[1])
            except Exception:
                target = 1
        if 0 <= target < gpu_count:
            total_bytes = int(torch.cuda.get_device_properties(target).total_memory)
            total_gib = max(1, int(total_bytes / (1024**3)))
            # Reserve some VRAM for SAM2 + misc allocations when sharing a single GPU.
            reserve_gib = int(os.environ.get("QWEN_VRAM_RESERVE_GIB", "12"))
            # Keep a healthy buffer while still allowing offload to CPU when needed.
            budget_gib = max(8, min(total_gib - reserve_gib, int(total_gib * 0.9)))
            max_memory: dict = {target: f"{budget_gib}GiB", "cpu": "192GiB"}
            for i in range(gpu_count):
                if i != target:
                    max_memory[i] = "1GiB"
            load_kwargs["max_memory"] = max_memory

    load_kwargs["offload_folder"] = os.path.join(tempfile.gettempdir(), f"qwen_offload_{qwen_device.replace(':', '_')}")
    load_kwargs["offload_state_dict"] = True

    model = _Qwen.from_pretrained(model_path, torch_dtype=torch.bfloat16, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    
    # Cache for this device
    _qwen_models[qwen_device] = (model, processor, model_path)
    
    return model, processor


def _caption_qwen(
    image: Image.Image,
    *,
    model_path: str,
    qwen_device: str,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 96,
) -> str:
    global _process_vision_info

    qwen_model, qwen_processor = _get_qwen_model(model_path, qwen_device=qwen_device)

    if _process_vision_info is None:
        import importlib

        _process_vision_info = importlib.import_module("qwen_vl_utils").process_vision_info

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
    ]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = _process_vision_info(messages)
    inputs = qwen_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(qwen_model.device)

    import torch

    with torch.inference_mode():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=int(max_new_tokens))

    out_ids = generated_ids[0]
    in_ids = inputs.input_ids[0]
    trimmed = out_ids[len(in_ids) :]
    return qwen_processor.decode(trimmed, skip_special_tokens=True).strip()


def _lane_prompt_from_qwen_output(text: str) -> str:
    """Normalize Qwen output into a prompt that always includes lane attributes."""
    import re

    t = (text or "").strip()
    if not t:
        return "no clear road with lane line markings"

    def _norm(val: str, allowed: set[str]) -> str:
        val = (val or "").strip().lower()
        return val if val in allowed else "unknown"

    def _format(count: str, color: str, pattern: str) -> str:
        count = _norm(count, {"single", "double", "unknown"})
        pattern = _norm(pattern, {"solid", "dashed", "mixed", "unknown"})
        color = _norm(color, {"white", "yellow", "mixed", "unknown"})
        if count == "unknown" and color == "unknown" and pattern == "unknown":
            return "no clear road with lane line markings"
        return f"road with {count} {color} {pattern} lane markings"

    def _strip_markdown_fences(s: str) -> str:
        # Handles e.g. ```json\n{...}\n``` or ```\n{...}\n```
        s = s.strip()
        if "```" not in s:
            return s
        # Remove all backtick fences but keep inner content.
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        return s.strip()

    def _extract_json_object(s: str) -> str | None:
        # Extract the first {...} block if the model emitted surrounding text.
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return s[start : end + 1].strip()

    cleaned = _strip_markdown_fences(t)
    json_candidate = _extract_json_object(cleaned) or cleaned

    # Prefer strict JSON outputs.
    try:
        payload = json.loads(json_candidate)
        if isinstance(payload, dict):
            return _format(
                str(payload.get("lane_marking_count", "unknown")),
                str(payload.get("lane_marking_color", "unknown")),
                str(payload.get("lane_marking_pattern", "unknown")),
            )
    except Exception:
        pass

    # Regex fallback: works even if the model returns a pseudo-JSON / prose blob.
    # Example matches: lane_marking_count: single / "lane_marking_pattern" = "dashed".
    blob = cleaned
    m_count = re.search(r"lane_marking_count\s*[:=]\s*['\"]?(single|double|unknown)", blob, flags=re.IGNORECASE)
    m_pattern = re.search(r"lane_marking_pattern\s*[:=]\s*['\"]?(solid|dashed|mixed|unknown)", blob, flags=re.IGNORECASE)
    m_color = re.search(r"lane_marking_color\s*[:=]\s*['\"]?(white|yellow|mixed|unknown)", blob, flags=re.IGNORECASE)
    if m_count or m_pattern or m_color:
        return _format(
            (m_count.group(1) if m_count else "unknown"),
            (m_color.group(1) if m_color else "unknown"),
            (m_pattern.group(1) if m_pattern else "unknown"),
        )

    # Final fallback: unknown, but keep the output clean (no code fences).
    return "no clear road with lane line markings"


# --------------------------------------------------------------------------------------
# SAM2 MASKING (FIRST FRAME)
# --------------------------------------------------------------------------------------
# Store SAM2 models per device
_sam2_predictors = {}  # device -> (predictor, key)
_sam2_predictors_lock = __import__("threading").Lock()


def _get_sam2_image_predictor(*, ckpt_path: str, config_name: str, device: str):
    global _sam2_predictors
    
    key = f"{ckpt_path}|{config_name}|{device}"
    
    with _sam2_predictors_lock:
        # Check if predictor already loaded for this device
        if device in _sam2_predictors:
            cached_predictor, cached_key = _sam2_predictors[device]
            if cached_key == key:
                return cached_predictor

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")

    # Load SAM2 without dtype conversion - the model handles mixed precision internally
    # Apply_image in SAM2ImagePredictor automatically handles dtype conversion
    sam2_model = build_sam2(config_name, ckpt_path=ckpt_path, device=device, mode="eval", apply_postprocessing=False)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    # Cache for this device
    with _sam2_predictors_lock:
        _sam2_predictors[device] = (sam2_predictor, key)
    
    return sam2_predictor


def _reinit_sam2_predictor(*, ckpt_path: str, config_name: str, device: str):
    """Force-recreate the SAM2 predictor on a device after CUDA error."""
    global _sam2_predictors
    with _sam2_predictors_lock:
        _sam2_predictors.pop(device, None)
    return _get_sam2_image_predictor(ckpt_path=ckpt_path, config_name=config_name, device=device)


def _fill_binary_mask_holes(mask_255):
    """Fill internal holes in a binary mask.

    Mirrors `process_videos_sam2._fill_binary_mask_holes`.
    """
    try:
        import cv2
        import numpy as np
    except Exception:  # pragma: no cover
        return mask_255

    if mask_255 is None:
        return mask_255
    if getattr(mask_255, "dtype", None) != np.uint8:
        mask_255 = mask_255.astype(np.uint8)

    h, w = mask_255.shape[:2]
    if h == 0 or w == 0:
        return mask_255

    inv = cv2.bitwise_not(mask_255)
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(inv, ff_mask, seedPoint=(0, 0), newVal=0)

    holes = inv
    return cv2.bitwise_or(mask_255, holes)


def _postprocess_road_mask(mask_01, *, width: int, height: int):
    """Post-process a road mask to match `process_videos_sam2.py`.

    Steps:
      1) Morphological opening (disconnect leaks)
      2) Keep connected component containing a reference point (bottom center)
      3) Morphological closing + hole fill
    """
    try:
        import cv2
        import numpy as np
    except Exception:  # pragma: no cover
        return (mask_01.astype("uint8") * 255)

    if mask_01 is None:
        return mask_01

    mask_uint8 = (mask_01.astype(bool).astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)

    ref_x = int(width // 2)
    ref_y = int(height * 0.85)
    ref_x = max(0, min(int(width - 1), ref_x))
    ref_y = max(0, min(int(height - 1), ref_y))

    if num_labels > 1:
        label_at_point = int(labels[ref_y, ref_x])
        if label_at_point > 0:
            filtered = (labels == label_at_point).astype(np.uint8) * 255
        else:
            largest_component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            filtered = (labels == largest_component).astype(np.uint8) * 255
    else:
        filtered = opened

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    filtered = _fill_binary_mask_holes(filtered)
    return filtered


def _sam2_road_mask(image_rgb: Image.Image, *, ckpt_path: str, config_name: str, device: str) -> Image.Image:
    import torch
    import numpy as np

    predictor = _get_sam2_image_predictor(ckpt_path=ckpt_path, config_name=config_name, device=device)

    with torch.inference_mode():
        predictor.set_image(image_rgb)
        w, h = image_rgb.size

        points = np.array(
            [
                [w * 0.25, h * 0.80],
                [w * 0.50, h * 0.85],
                [w * 0.75, h * 0.80],
                [w * 0.50, h * 0.65],
                [w * 0.35, h * 0.75],
                [w * 0.65, h * 0.75],
            ],
            dtype=np.float32,
        )
        labels = np.ones((points.shape[0],), dtype=np.int32)

        masks, ious, _low_res = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
            return_logits=False,
            normalize_coords=True,
        )

    if ious is not None and len(ious) > 0:
        best = int(np.argmax(ious))
    else:
        best = 0
    m = masks[best]

    # Explicitly release predictor state (GPU feature tensors) to avoid
    # gradual VRAM fragmentation over thousands of invocations.
    predictor.reset_predictor()

    # Post-process to match `process_videos_sam2.py` behavior more closely.
    # `m` may be bool or float; treat non-zero as foreground.
    m01 = m.astype(bool) if getattr(m, "dtype", None) == np.bool_ else (m > 0.0)
    filtered_255 = _postprocess_road_mask(m01, width=w, height=h)
    return Image.fromarray(filtered_255.astype(np.uint8), mode="L")


def _focus_image_to_mask(image_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """Return an image where only the masked region is visible.

    Outside-mask pixels are set to black so captioning can focus on the segmented region.
    """
    import numpy as np

    if image_rgb.mode != "RGB":
        image_rgb = image_rgb.convert("RGB")
    if mask_l.mode != "L":
        mask_l = mask_l.convert("L")

    img = np.array(image_rgb, dtype=np.uint8)
    m = np.array(mask_l, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        return image_rgb

    keep = m > 0
    if not bool(np.any(keep)):
        return image_rgb

    out = np.zeros_like(img)
    out[keep] = img[keep]
    return Image.fromarray(out, mode="RGB")


def _is_cuda_error(exc: BaseException) -> bool:
    """Return True if the exception is (or wraps) a CUDA error."""
    msg = str(exc).lower()
    return "cuda error" in msg or "illegal memory access" in msg or "device-side assert" in msg


# ---------------------------------------------------------------------------
# GPU WORKER (runs in its own subprocess with isolated CUDA context)
# ---------------------------------------------------------------------------

# Sentinel value workers put on result_queue to signal a fatal CUDA error.
_CUDA_FATAL_SENTINEL = "__CUDA_FATAL__"


def _gpu_worker(
    physical_gpu_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    sam2_checkpoint_path: str,
    sam2_config_name: str,
    qwen_model_path: str,
    system_prompt: str,
    user_prompt: str,
):
    """Run in a dedicated subprocess.  Owns exactly one GPU.

    - ``CUDA_VISIBLE_DEVICES`` is set *before* importing torch so that
      ``cuda:0`` inside this process maps to the assigned physical GPU.
    - If a CUDA fatal error occurs the subprocess signals the parent via
      ``_CUDA_FATAL_SENTINEL`` on the result queue and exits immediately.
      The parent will then abort the entire job.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

    import gc
    import torch

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    _logger = logging.getLogger(f"{__name__}.gpu{physical_gpu_id}")
    _logger.info("GPU worker started — physical GPU %d (CUDA_VISIBLE_DEVICES=%s)",
                 physical_gpu_id, os.environ["CUDA_VISIBLE_DEVICES"])

    # Device is always cuda:0 inside this process
    device = "cuda:0"

    # Load models once
    try:
        _logger.info("Loading SAM2 on %s (physical GPU %d) …", device, physical_gpu_id)
        _get_sam2_image_predictor(
            ckpt_path=sam2_checkpoint_path,
            config_name=sam2_config_name,
            device=device,
        )
        _logger.info("Loading Qwen on %s (physical GPU %d) …", device, physical_gpu_id)
        _get_qwen_model(qwen_model_path, qwen_device=device)
        _logger.info("Models loaded on physical GPU %d", physical_gpu_id)
    except Exception as e:
        _logger.error("Failed to load models on GPU %d: %s", physical_gpu_id, e)
        result_queue.put((physical_gpu_id, _CUDA_FATAL_SENTINEL))
        return

    while True:
        try:
            item = task_queue.get(timeout=5)
        except Exception:
            # Queue empty or timed out — check if sentinel was missed
            continue
        if item is None:
            # Sentinel: no more work
            _logger.info("GPU %d worker received shutdown sentinel", physical_gpu_id)
            break

        (
            global_index, uri, mp4_path, frames,
            images_dir, masks_dir,
            chunk_num,
        ) = item

        rows = []
        sid = _stable_id(uri)

        _logger.info("[GPU %d] Processing %s (chunk_%04d, %d frames)",
                     physical_gpu_id, os.path.basename(mp4_path), chunk_num, len(frames))

        # --- Step 1: Batch-extract ALL frames (CPU only — ffmpeg) ---
        chunk_tag = f"chunk_{chunk_num:04d}"
        chunk_images_dir = os.path.join(images_dir, chunk_tag)
        chunk_masks_dir = os.path.join(masks_dir, chunk_tag)
        Path(chunk_images_dir).mkdir(parents=True, exist_ok=True)
        Path(chunk_masks_dir).mkdir(parents=True, exist_ok=True)

        out_bases = [f"{global_index:06d}_f{int(fn):04d}_{sid}.png" for fn in frames]
        out_imgs = [os.path.join(chunk_images_dir, b) for b in out_bases]
        out_masks = [os.path.join(chunk_masks_dir, b) for b in out_bases]

        try:
            _extract_frames_ffmpeg_batch(mp4_path, frame_numbers=frames, out_png_paths=out_imgs)
        except Exception as e:
            _logger.error("[GPU %d] ffmpeg failed for %s: %s", physical_gpu_id, uri, e)
            result_queue.put((physical_gpu_id, rows))
            continue

        # --- Step 2: For each extracted frame, mask + caption (GPU) ---
        for frame_number, out_img, out_mask in zip(frames, out_imgs, out_masks):
            if not os.path.exists(out_img):
                _logger.warning("[GPU %d] Frame %d not extracted from %s",
                                physical_gpu_id, frame_number, uri)
                continue

            try:
                pil_img = Image.open(out_img).convert("RGB")

                mask = _sam2_road_mask(
                    pil_img,
                    ckpt_path=sam2_checkpoint_path,
                    config_name=sam2_config_name,
                    device=device,
                )
                mask.save(out_mask)

                focus_img = _focus_image_to_mask(pil_img, mask)
                prompt_raw = _caption_qwen(
                    focus_img,
                    model_path=qwen_model_path,
                    qwen_device=device,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=64,
                )
                prompt = _lane_prompt_from_qwen_output(prompt_raw)

                rows.append({
                    "image": f"{chunk_tag}/images/{os.path.basename(out_img)}",
                    "mask": f"{chunk_tag}/masks/{os.path.basename(out_mask)}",
                    "prompt": prompt,
                    "prompt_2": prompt,
                })

            except Exception as e:
                _logger.error("[GPU %d] Failed frame %d of %s: %s",
                              physical_gpu_id, frame_number, uri, e)
                for p in (out_img, out_mask):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
                if _is_cuda_error(e):
                    _logger.error("[GPU %d] FATAL CUDA error — aborting worker",
                                  physical_gpu_id)
                    result_queue.put((physical_gpu_id, _CUDA_FATAL_SENTINEL))
                    return  # exit subprocess immediately
                continue

        # Periodic housekeeping
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        _logger.info("[GPU %d] Completed %s: %d/%d frames OK",
                     physical_gpu_id, os.path.basename(mp4_path), len(rows), len(frames))
        result_queue.put((physical_gpu_id, rows))


# _preload_models_on_gpus is no longer needed — each subprocess loads its own
# models inside _gpu_worker.  Kept as a no-op stub in case anything references it.
def _preload_models_on_gpus(**_kwargs) -> None:
    pass


@task(
    compute=DedicatedNode(
        node=Node.A100_80GB_4GPU,
        ephemeral_storage="max",
        max_duration="3d",
    ),
    container_image=CONTAINER_IMAGE,
    environment={
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_LAUNCH_BLOCKING": "0",
    },
    mounts=[
        FuseBucket(bucket=BUCKET, name=QWEN_MODEL_FUSE_NAME, prefix=QWEN_MODEL_GCS_PREFIX),
        FuseBucket(bucket=BUCKET, name=SAM2_FUSE_NAME, prefix=SAM2_CHECKPOINT_GCS_PREFIX),
    ],
)
def generate_fluxfill_training_data(
    *,
    num_videos: int = 200,
    start_index: int = 0,
    seed: int = 42,
    source_gcs_prefix: str = SOURCE_GCS_PREFIX,
    # Single-GPU defaults (A100 80GB): run both on the same GPU.
    # Qwen may still offload some weights/states to CPU depending on available VRAM.
    qwen_device: str = "cuda:0",
    sam2_device: str = "cuda:0",
    sam2_checkpoint_path: str = SAM2_CHECKPOINT_MOUNTED_PATH,
    sam2_config_name: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    # 0 means unlimited walk; we still early-stop once enough .mp4s are found.
    max_walk_files: int = 0,
    sort_gcs_listing: bool = False,
    download_batch_size: int = 100,
    frame_numbers: str = "1,100,200,300,400,500",
    chunk_start: int = 0,
    chunk_end: int = 0,
    output_run_id: Optional[str] = None,
    num_gpus: int = 4,
    num_workers: int = 4,
) -> str:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # num_videos <= 0 means "process everything found".
    if num_videos <= 0:
        num_videos = float("inf")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if int(download_batch_size) <= 0:
        raise ValueError("download_batch_size must be > 0")

    frames = _parse_frame_numbers_csv(frame_numbers)
    if not frames:
        raise ValueError("frame_numbers resolved to empty")

    run_id = (output_run_id or datetime.utcnow().strftime("fluxfill_%Y%m%d_%H%M%S")).strip()
    if not run_id:
        raise ValueError("output_run_id resolved to empty")

    # Prefetch metadata best-effort
    for p in (QWEN_MODEL_FUSE_ROOT, SAM2_FUSE_ROOT):
        try:
            fuse_prefetch_metadata(p)
        except Exception as e:
            logger.warning("Prefetch failed (non-fatal) for %s: %s", p, e)

    if not os.path.isdir(QWEN_MODEL_FUSE_ROOT):
        raise FileNotFoundError(f"Qwen model mount missing: {QWEN_MODEL_FUSE_ROOT}")
    if not os.path.exists(sam2_checkpoint_path):
        raise FileNotFoundError(f"SAM2 checkpoint missing: {sam2_checkpoint_path}")

    tmp_root = os.path.join(tempfile.gettempdir(), "fluxfill_training_data", run_id)
    videos_dir = os.path.join(tmp_root, "videos")
    images_dir = os.path.join(tmp_root, "images")
    masks_dir = os.path.join(tmp_root, "masks")
    Path(videos_dir).mkdir(parents=True, exist_ok=True)
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    Path(masks_dir).mkdir(parents=True, exist_ok=True)

    system_prompt = (
        "You are a driving scene labeling assistant. "
        "You only describe lane line markings on the road surface. "
        "First, check whether the segmented region shows a CLEAR road surface with lane line markings clearly visible. "
        "If the road is not clear or the lane line markings are not clearly visible, output 'unknown' for ALL fields. "
        "Return ONLY valid JSON with keys: lane_marking_count, lane_marking_pattern, lane_marking_color. "
        "Do NOT wrap the JSON in Markdown, code fences, or backticks. "
        "Use this fixed vocabulary: "
        "lane_marking_count={single|double|unknown}, "
        "lane_marking_pattern={solid|dashed|mixed|unknown}, "
        "lane_marking_color={white|yellow|mixed|unknown}."
    )
    user_prompt = (
        "Look only at the segmented road region. "
        "Only answer if it is a clear road with lane line markings visible. "
        "Classify lane marking count (single vs double line), pattern (solid vs dashed), and color (white vs yellow)."
    )

    dest_prefix = f"gs://{BUCKET}/{DEST_GCS_PREFIX_BASE}/{run_id}".rstrip("/")
    remote_csv_uri = f"{dest_prefix}/train.csv"

    fs = gcsfs.GCSFileSystem(token="google_default")

    csv_path = os.path.join(tmp_root, "train.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=["image", "mask", "prompt", "prompt_2"])
        wri.writeheader()

    # List + download only the requested slice of mp4s.
    num_videos_display = "ALL" if num_videos == float("inf") else int(num_videos)
    logger.info(
        "Listing mp4s in gs://%s/%s (start_index=%d, num_videos=%s, chunk_start=%d, chunk_end=%d)",
        BUCKET,
        source_gcs_prefix,
        int(start_index),
        num_videos_display,
        int(chunk_start),
        int(chunk_end),
    )
    
    # Process chunk-by-chunk: download chunk → process → upload → delete → next chunk
    fs = gcsfs.GCSFileSystem(token="google_default")
    root = f"{BUCKET}/{source_gcs_prefix.strip('/')}"
    
    total_processed = 0
    start_idx = int(start_index)
    limit = int(num_videos) if num_videos != float("inf") else float("inf")

    # ---- Spawn one subprocess per GPU.  Each subprocess has its own CUDA
    #      context, its own copy of SAM2 + Qwen, and is fully isolated.
    #      If one GPU hits a fatal CUDA error the other 3 continue. ----
    mp_ctx = multiprocessing.get_context("spawn")  # fork is unsafe with CUDA
    task_queue = mp_ctx.Queue()       # parent → workers: video tasks
    result_queue = mp_ctx.Queue()     # workers → parent: per-video rows

    workers: list[multiprocessing.Process] = []
    for gpu_id in range(num_gpus):
        p = mp_ctx.Process(
            target=_gpu_worker,
            args=(
                gpu_id,
                task_queue,
                result_queue,
                sam2_checkpoint_path,
                sam2_config_name,
                QWEN_MODEL_FUSE_ROOT,
                system_prompt,
                user_prompt,
            ),
            name=f"gpu-worker-{gpu_id}",
            daemon=True,
        )
        p.start()
        workers.append(p)
    logger.info("Spawned %d GPU worker subprocesses", len(workers))

    # Separate I/O thread pool for parallel uploads (doesn't touch GPUs)
    upload_pool = ThreadPool(processes=8)

    def _upload_one(args_tuple):
        """Upload a single image+mask pair to GCS. Called by upload_pool."""
        _fs, img_path, mask_path, remote_img, remote_mask = args_tuple
        uploaded = 0
        if os.path.exists(img_path):
            _upload_file_to_gcs(_fs, img_path, remote_img, log_upload=False)
            os.remove(img_path)
            uploaded += 1
        if os.path.exists(mask_path):
            _upload_file_to_gcs(_fs, mask_path, remote_mask, log_upload=False)
            os.remove(mask_path)
            uploaded += 1
        return uploaded

    def _build_upload_args_for_chunk(chunk_rows, chunk_num):
        """Build upload args with per-chunk GCS paths."""
        chunk_tag = f"chunk_{chunk_num:04d}"
        remote_chunk_prefix = f"{dest_prefix}/{chunk_tag}"
        args = []
        for row in chunk_rows:
            img_bn = os.path.basename(row["image"])
            mask_bn = os.path.basename(row["mask"])
            args.append((
                fs,
                os.path.join(images_dir, chunk_tag, img_bn),
                os.path.join(masks_dir, chunk_tag, mask_bn),
                f"{remote_chunk_prefix}/images/{img_bn}",
                f"{remote_chunk_prefix}/masks/{mask_bn}",
            ))
        return args

    def _download_chunk_videos(
        _fs, chunk_mp4_uris, chunk_dir, base_index
    ) -> list[tuple[int, str, str]]:
        """Download all videos for a chunk."""
        Path(chunk_dir).mkdir(parents=True, exist_ok=True)
        local_vids = []
        for j, uri in enumerate(chunk_mp4_uris):
            global_idx = base_index + j
            remote = uri[len("gs://"):] if uri.startswith("gs://") else uri
            local_mp4 = os.path.join(chunk_dir, _local_mp4_name_for_uri(uri, global_idx))
            _fs.get(remote, local_mp4)
            local_vids.append((global_idx, uri, local_mp4))
        return local_vids

    def _collect_results(expected_count: int) -> list[dict]:
        """Drain result_queue for *expected_count* video results.

        Raises RuntimeError immediately if any worker signals a fatal CUDA error.
        """
        all_rows: list[dict] = []
        collected = 0
        while collected < expected_count:
            try:
                _gpu_id, rows = result_queue.get(timeout=120)
            except Exception:
                alive = sum(1 for w in workers if w.is_alive())
                if alive < num_gpus:
                    raise RuntimeError(
                        f"GPU worker died (only {alive}/{num_gpus} alive). "
                        "Aborting job — all GPUs must be healthy."
                    )
                logger.warning("Timeout waiting for result (%d/%d collected)",
                               collected, expected_count)
                collected += 1
                continue

            # Check for fatal CUDA signal
            if rows is _CUDA_FATAL_SENTINEL:
                raise RuntimeError(
                    f"GPU {_gpu_id} hit a fatal CUDA error. "
                    "Aborting entire job — all GPUs must be healthy."
                )

            all_rows.extend(rows)
            collected += 1
        return all_rows

    # Track pending upload futures so we can overlap upload(N) with process(N+1)
    pending_upload_result = None
    pending_upload_chunk_num = None

    try:
      for chunk_num in range(int(chunk_start), int(chunk_end) + 1):
        if limit != float("inf") and total_processed >= limit:
            logger.info("Reached video limit (%s), stopping chunk iteration", limit)
            break

        # Abort if ANY worker is dead — do not continue with degraded GPU count
        alive_workers = sum(1 for w in workers if w.is_alive())
        if alive_workers < num_gpus:
            raise RuntimeError(
                f"Only {alive_workers}/{num_gpus} GPU workers alive. "
                "Aborting job — all GPUs must be healthy."
            )

        chunk_folder = f"{root}/chunk_{chunk_num:04d}"
        logger.info("=" * 80)
        logger.info("Processing chunk: %s (chunk %d/%d)", chunk_folder, chunk_num - chunk_start + 1, chunk_end - chunk_start + 1)
        logger.info("=" * 80)
        
        try:
            chunk_files = fs.find(chunk_folder)
            chunk_mp4s = [f"gs://{p}" for p in chunk_files if p.lower().endswith(".mp4")]
            logger.info("Found %d mp4s in chunk_%04d", len(chunk_mp4s), chunk_num)
            
            if not chunk_mp4s:
                logger.warning("No mp4s in chunk_%04d, skipping", chunk_num)
                continue
                
            if limit == float("inf"):
                chunk_to_process = chunk_mp4s[start_idx:] if start_idx > 0 else chunk_mp4s
            else:
                remaining = int(limit) - total_processed
                chunk_to_process = chunk_mp4s[start_idx:start_idx + remaining] if start_idx > 0 else chunk_mp4s[:remaining]
            start_idx = max(0, start_idx - len(chunk_mp4s))
            
            if not chunk_to_process:
                logger.info("No videos to process from chunk_%04d after applying start_index/limit", chunk_num)
                continue
                
        except FileNotFoundError:
            logger.warning("Chunk folder not found: %s", chunk_folder)
            continue
        
        # ---- Download this chunk's videos ----
        chunk_dir = os.path.join(videos_dir, f"chunk_{chunk_num:04d}")
        logger.info("Downloading %d videos from chunk_%04d …", len(chunk_to_process), chunk_num)
        local_videos = _download_chunk_videos(fs, chunk_to_process, chunk_dir, total_processed)
        
        logger.info("Processing %d videos from chunk_%04d (frames: %s) — %d GPU workers",
                     len(local_videos), chunk_num,
                     ",".join(str(f) for f in frames),
                     num_gpus)
        
        # Enqueue each video as a task for the GPU workers
        for vid_idx, (global_index, uri, mp4_path) in enumerate(local_videos):
            task_queue.put((
                global_index, uri, mp4_path, frames,
                images_dir, masks_dir,
                chunk_num,
            ))
        
        # ---- Collect results for all videos in this chunk ----
        chunk_rows = _collect_results(len(local_videos))
        logger.info("chunk_%04d: %d image/mask pairs generated", chunk_num, len(chunk_rows))
        
        # Delete downloaded mp4s immediately (free disk for next chunk)
        for _, _, mp4_path in local_videos:
            try:
                os.remove(mp4_path)
            except Exception:
                pass
        shutil.rmtree(chunk_dir, ignore_errors=True)
        
        # ---- Wait for previous chunk's upload to finish before starting ours ----
        if pending_upload_result is not None:
            upload_counts = pending_upload_result.get()  # blocks until done
            logger.info("Previous chunk_%04d upload finished (%d files)",
                        pending_upload_chunk_num, sum(upload_counts))
            pending_upload_result = None
        
        # ---- Kick off parallel upload for THIS chunk (non-blocking) ----
        upload_args = _build_upload_args_for_chunk(chunk_rows, chunk_num)
        pending_upload_result = upload_pool.map_async(_upload_one, upload_args)
        pending_upload_chunk_num = chunk_num
        logger.info("chunk_%04d: upload dispatched (%d pairs) — processing next chunk …",
                     chunk_num, len(upload_args))
        
        # Append rows to CSV and upload it (small file, fast)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=["image", "mask", "prompt", "prompt_2"])
            for row in chunk_rows:
                wri.writerow(row)
        _upload_file_to_gcs(fs, csv_path, remote_csv_uri)
        
        total_processed += len(chunk_to_process)
        logger.info("Finished chunk_%04d: %d videos (%d total so far)",
                     chunk_num, len(chunk_to_process), total_processed)

      # ---- Wait for last chunk's upload ----
      if pending_upload_result is not None:
          upload_counts = pending_upload_result.get()
          logger.info("Final chunk_%04d upload finished (%d files)",
                      pending_upload_chunk_num, sum(upload_counts))

    finally:
        # Send shutdown sentinels to all workers
        for _ in workers:
            try:
                task_queue.put(None)
            except Exception:
                pass
        # Wait for workers to exit (with timeout)
        for w in workers:
            w.join(timeout=30)
            if w.is_alive():
                logger.warning("Worker %s did not exit cleanly, terminating", w.name)
                w.terminate()
        upload_pool.close()
        upload_pool.join()
        logger.info("All workers and pools shut down")

    if total_processed > 0:
        _upload_file_to_gcs(fs, csv_path, remote_csv_uri)
        logger.info("Upload complete (incremental): %s (processed %d videos)", dest_prefix, total_processed)
    else:
        raise RuntimeError("No videos were processed from any chunks")
        
    return dest_prefix + "/"


@workflow
def fluxfill_data_generation_wf(
    num_videos: int = 200,
    start_index: int = 0,
    seed: int = 42,
    source_gcs_prefix: str = SOURCE_GCS_PREFIX,
    qwen_device: str = "cuda:0",
    sam2_device: str = "cuda:0",
    sam2_checkpoint_path: str = SAM2_CHECKPOINT_MOUNTED_PATH,
    sam2_config_name: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    max_walk_files: int = 0,
    sort_gcs_listing: bool = False,
    download_batch_size: int = 100,
    frame_numbers: str = "1,100,200,300,400,500",
    chunk_start: int = 0,
    chunk_end: int = 0,
    output_run_id: Optional[str] = None,
    num_gpus: int = 4,
    num_workers: int = 4,
) -> str:
    return generate_fluxfill_training_data(
        num_videos=num_videos,
        start_index=start_index,
        seed=seed,
        source_gcs_prefix=source_gcs_prefix,
        qwen_device=qwen_device,
        sam2_device=sam2_device,
        sam2_checkpoint_path=sam2_checkpoint_path,
        sam2_config_name=sam2_config_name,
        max_walk_files=max_walk_files,
        sort_gcs_listing=sort_gcs_listing,
        download_batch_size=download_batch_size,
        frame_numbers=frame_numbers,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        output_run_id=output_run_id,
        num_gpus=num_gpus,
        num_workers=num_workers,
    )
