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
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
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

SOURCE_GCS_PREFIX = "workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov"
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


def _upload_file_to_gcs(fs: gcsfs.GCSFileSystem, local_path: str, gcs_uri: str) -> None:
    remote = _gcs_strip_scheme(gcs_uri)
    remote_parent = os.path.dirname(remote)
    if remote_parent:
        fs.makedirs(remote_parent, exist_ok=True)
    fs.put(local_path, remote)


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
_qwen_model = None
_qwen_processor = None
_qwen_model_id = None
_process_vision_info = None


def _get_qwen_model(model_path: str, *, qwen_device: str = "cuda:1"):
    global _qwen_model, _qwen_processor, _qwen_model_id
    if _qwen_model is not None and _qwen_processor is not None and _qwen_model_id == model_path:
        return _qwen_model, _qwen_processor

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

    load_kwargs["offload_folder"] = os.path.join(tempfile.gettempdir(), "qwen_offload")
    load_kwargs["offload_state_dict"] = True

    _qwen_model = _Qwen.from_pretrained(model_path, torch_dtype=torch.bfloat16, **load_kwargs)
    _qwen_processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    _qwen_model_id = model_path
    return _qwen_model, _qwen_processor


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
_sam2_model = None
_sam2_predictor = None
_sam2_key = None


def _get_sam2_image_predictor(*, ckpt_path: str, config_name: str, device: str):
    global _sam2_model, _sam2_predictor, _sam2_key
    key = f"{ckpt_path}|{config_name}|{device}"
    if _sam2_predictor is not None and _sam2_key == key:
        return _sam2_predictor

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")

    _sam2_model = build_sam2(config_name, ckpt_path=ckpt_path, device=device, mode="eval")
    _sam2_predictor = SAM2ImagePredictor(_sam2_model)
    _sam2_key = key
    return _sam2_predictor


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
    import numpy as np

    predictor = _get_sam2_image_predictor(ckpt_path=ckpt_path, config_name=config_name, device=device)
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


@task(
    compute=DedicatedNode(
        node=Node.A100_80GB_1GPU,
        ephemeral_storage="max",
        max_duration="3d",
    ),
    container_image=CONTAINER_IMAGE,
    environment={
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
    chunk_end: int = 200,
    output_run_id: Optional[str] = None,
) -> str:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if num_videos <= 0:
        raise ValueError("num_videos must be > 0")
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
    remote_images_prefix = f"{dest_prefix}/images"
    remote_masks_prefix = f"{dest_prefix}/masks"
    remote_csv_uri = f"{dest_prefix}/train.csv"

    fs = gcsfs.GCSFileSystem(token="google_default")

    csv_path = os.path.join(tmp_root, "train.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=["image", "mask", "prompt", "prompt_2"])
        wri.writeheader()

    # List + download only the requested slice of mp4s.
    logger.info(
        "Listing mp4s in gs://%s/%s (start_index=%d, num_videos=%d, chunk_start=%d, chunk_end=%d)",
        BUCKET,
        source_gcs_prefix,
        int(start_index),
        int(num_videos),
        int(chunk_start),
        int(chunk_end),
    )
    
    # Process chunk-by-chunk: download chunk → process → upload → delete → next chunk
    fs = gcsfs.GCSFileSystem(token="google_default")
    root = f"{BUCKET}/{source_gcs_prefix.strip('/')}"
    
    total_processed = 0
    start_idx = int(start_index)
    limit = int(num_videos)
    
    for chunk_num in range(int(chunk_start), int(chunk_end) + 1):
        if total_processed >= limit:
            logger.info("Reached video limit (%d), stopping chunk iteration", limit)
            break
            
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
                
            # Apply start_index and limit within this chunk
            remaining = limit - total_processed
            chunk_to_process = chunk_mp4s[start_idx:start_idx + remaining] if start_idx > 0 else chunk_mp4s[:remaining]
            start_idx = max(0, start_idx - len(chunk_mp4s))  # Decrement for next chunk
            
            if not chunk_to_process:
                logger.info("No videos to process from chunk_%04d after applying start_index/limit", chunk_num)
                continue
                
        except FileNotFoundError:
            logger.warning("Chunk folder not found: %s", chunk_folder)
            continue
        
        # Download this chunk's videos
        chunk_dir = os.path.join(videos_dir, f"chunk_{chunk_num:04d}")
        Path(chunk_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading %d videos from chunk_%04d to %s", len(chunk_to_process), chunk_num, chunk_dir)
        
        local_videos: list[tuple[int, str, str]] = []
        for j, uri in enumerate(chunk_to_process):
            global_index = total_processed + j
            remote = uri[len("gs://"):] if uri.startswith("gs://") else uri
            local_mp4 = os.path.join(chunk_dir, _local_mp4_name_for_uri(uri, global_index))
            fs.get(remote, local_mp4)
            local_videos.append((global_index, uri, local_mp4))
        
        logger.info("Processing %d videos from chunk_%04d (frames: %s)", len(local_videos), chunk_num, ",".join(str(f) for f in frames))
        
        # Process each video in this chunk (keep files locally)
        chunk_rows = []
        for vid_idx, (global_index, uri, mp4_path) in enumerate(local_videos, 1):
            logger.info("Processing video %d/%d in chunk_%04d: %s", vid_idx, len(local_videos), chunk_num, os.path.basename(mp4_path))
            sid = _stable_id(uri)

            for frame_idx, frame_number in enumerate(frames, 1):
                logger.info("  Extracting frame %d/%d (frame#%d)", frame_idx, len(frames), frame_number)
                out_base = f"{global_index:06d}_f{int(frame_number):04d}_{sid}.png"
                out_img = os.path.join(images_dir, out_base)
                out_mask = os.path.join(masks_dir, out_base)

                try:
                    _extract_frame_ffmpeg(mp4_path, frame_number=int(frame_number), out_png_path=out_img)
                except Exception as e:
                    # Short videos may not have all requested frames.
                    logger.warning("Skipping %s frame %s: %s", uri, frame_number, e)
                    try:
                        if os.path.exists(out_img):
                            os.remove(out_img)
                    except Exception:
                        pass
                    continue

                pil_img = Image.open(out_img).convert("RGB")
                logger.info("  Generating SAM2 mask...")

                mask = _sam2_road_mask(
                    pil_img,
                    ckpt_path=sam2_checkpoint_path,
                    config_name=sam2_config_name,
                    device=sam2_device,
                )
                mask.save(out_mask)
                logger.info("  Generating caption...")

                focus_img = _focus_image_to_mask(pil_img, mask)

                prompt_raw = _caption_qwen(
                    focus_img,
                    model_path=QWEN_MODEL_FUSE_ROOT,
                    qwen_device=qwen_device,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=64,
                )
                prompt = _lane_prompt_from_qwen_output(prompt_raw)

                row = {
                    "image": f"images/{os.path.basename(out_img)}",
                    "mask": f"masks/{os.path.basename(out_mask)}",
                    "prompt": prompt,
                    "prompt_2": prompt,
                }
                chunk_rows.append(row)

            # After all frames for this mp4, delete the local video.
            try:
                os.remove(mp4_path)
            except Exception:
                pass

            if len(chunk_rows) and (len(chunk_rows) % 50 == 0):
                logger.info("Processed %d rows in chunk_%04d so far", len(chunk_rows), chunk_num)
        
        # Batch upload all images and masks for this chunk
        logger.info("Uploading %d images and masks for chunk_%04d...", len(chunk_rows), chunk_num)
        for row in chunk_rows:
            img_basename = row["image"].split("/")[-1]
            mask_basename = row["mask"].split("/")[-1]
            local_img = os.path.join(images_dir, img_basename)
            local_mask = os.path.join(masks_dir, mask_basename)
            
            if os.path.exists(local_img):
                _upload_file_to_gcs(fs, local_img, f"{remote_images_prefix}/{img_basename}")
                os.remove(local_img)
            if os.path.exists(local_mask):
                _upload_file_to_gcs(fs, local_mask, f"{remote_masks_prefix}/{mask_basename}")
                os.remove(local_mask)
        
        # Append all rows to CSV and upload once per chunk
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=["image", "mask", "prompt", "prompt_2"])
            for row in chunk_rows:
                wri.writerow(row)
        
        _upload_file_to_gcs(fs, csv_path, remote_csv_uri)
        logger.info("Uploaded CSV for chunk_%04d (%d total rows)", chunk_num, len(chunk_rows))

        # Delete the whole chunk directory (should already be empty, but robust).
        shutil.rmtree(chunk_dir, ignore_errors=True)
        
        total_processed += len(chunk_to_process)
        logger.info("Finished chunk_%04d: processed %d videos (%d total so far)", chunk_num, len(chunk_to_process), total_processed)

    # If we processed at least one file, ensure train.csv is present remotely.
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
    chunk_end: int = 200,
    output_run_id: Optional[str] = None,
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
    )
