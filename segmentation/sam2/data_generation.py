"""Generate FluxFill LoRA training data from a GCS folder of .mp4 files.

This script lives under `segmentation/sam2/` so it can directly reuse the SAM2
installation + configs in this repo.

Pipeline:
    1) List + download a slice of mp4s from GCS
  2) Mount Qwen2.5-VL-72B-Instruct (local HF folder)
  3) Mount SAM2 checkpoint
  4) For N selected mp4s:
     - extract first frame
     - generate mask using SAM2 (image predictor + fixed road-style points)
     - generate caption using Qwen2.5-VL-72B
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
import logging
import os
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

SOURCE_GCS_PREFIX = "datasets/public/physical_ai_av/camera/camera_front_tele_30fov"
DEST_GCS_PREFIX_BASE = "workspace/user/hbaskar/Video_inpainting/videopainter/training/data"

VLM_72B_GCS_PREFIX = "workspace/user/hbaskar/Video_inpainting/videopainter/vlm/Qwen2.5-VL-72B-Instruct"
VLM_72B_FUSE_NAME = "vlm-72b"
VLM_72B_FUSE_ROOT = os.path.join(MOUNTPOINT, VLM_72B_FUSE_NAME)

SAM2_CHECKPOINT_GCS_PREFIX = "workspace/user/hbaskar/Video_inpainting/sam2_checkpoint"
SAM2_FUSE_NAME = "sam2-checkpoints"
SAM2_FUSE_ROOT = os.path.join(MOUNTPOINT, SAM2_FUSE_NAME)
SAM2_CHECKPOINT_MOUNTED_PATH = os.path.join(SAM2_FUSE_ROOT, "checkpoints", "sam2.1_hiera_large.pt")


# --------------------------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------------------------

def _stable_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]


def _extract_first_frame_ffmpeg(video_path: str, out_png_path: str) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
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
) -> list[str]:
    """List mp4 objects under gs://bucket/prefix and return a sliced list.

    Returns fully-qualified gs:// URIs.
    """
    fs = gcsfs.GCSFileSystem(token="google_default")
    root = f"{bucket}/{prefix.strip('/')}"

    # fs.find can return a lot; cap the scan for safety.
    all_paths: list[str] = []
    for i, p in enumerate(fs.find(root)):
        if i >= int(max_list_files):
            logger.warning("Reached max_list_files=%d while listing %s", int(max_list_files), root)
            break
        if p.lower().endswith(".mp4"):
            all_paths.append(p)

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
        FuseBucket(bucket=BUCKET, name=VLM_72B_FUSE_NAME, prefix=VLM_72B_GCS_PREFIX),
        FuseBucket(bucket=BUCKET, name=SAM2_FUSE_NAME, prefix=SAM2_CHECKPOINT_GCS_PREFIX),
    ],
)
def generate_fluxfill_training_data(
    *,
    num_videos: int = 200,
    start_index: int = 0,
    seed: int = 42,
    # Single-GPU defaults (A100 80GB): run both on the same GPU.
    # Qwen may still offload some weights/states to CPU depending on available VRAM.
    qwen_device: str = "cuda:0",
    sam2_device: str = "cuda:0",
    sam2_checkpoint_path: str = SAM2_CHECKPOINT_MOUNTED_PATH,
    sam2_config_name: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    max_walk_files: int = 500000,
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

    run_id = (output_run_id or datetime.utcnow().strftime("fluxfill_%Y%m%d_%H%M%S")).strip()
    if not run_id:
        raise ValueError("output_run_id resolved to empty")

    # Prefetch metadata best-effort
    for p in (VLM_72B_FUSE_ROOT, SAM2_FUSE_ROOT):
        try:
            fuse_prefetch_metadata(p)
        except Exception as e:
            logger.warning("Prefetch failed (non-fatal) for %s: %s", p, e)

    if not os.path.isdir(VLM_72B_FUSE_ROOT):
        raise FileNotFoundError(f"Qwen model mount missing: {VLM_72B_FUSE_ROOT}")
    if not os.path.exists(sam2_checkpoint_path):
        raise FileNotFoundError(f"SAM2 checkpoint missing: {sam2_checkpoint_path}")

    tmp_root = os.path.join(tempfile.gettempdir(), "fluxfill_training_data", run_id)
    videos_dir = os.path.join(tmp_root, "videos")
    images_dir = os.path.join(tmp_root, "images")
    masks_dir = os.path.join(tmp_root, "masks")
    Path(videos_dir).mkdir(parents=True, exist_ok=True)
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    Path(masks_dir).mkdir(parents=True, exist_ok=True)

    system_prompt = "You write short, literal captions. Avoid speculation."
    user_prompt = (
        "Describe only the road and lane markings visible in the image. "
        "Do not mention cars, buildings, sky, trees, or anything outside the road. "
        "Write 1 short sentence (<= 20 words)."
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
        "Listing mp4s in gs://%s/%s (start_index=%d, num_videos=%d)",
        BUCKET,
        SOURCE_GCS_PREFIX,
        int(start_index),
        int(num_videos),
    )
    mp4_uris = _list_mp4s_in_gcs_prefix(
        bucket=BUCKET,
        prefix=SOURCE_GCS_PREFIX,
        start_index=start_index,
        limit=num_videos,
        max_list_files=max_walk_files,
    )
    if not mp4_uris:
        raise RuntimeError("No mp4 files found for the requested slice")

    logger.info("Downloading %d mp4s to %s", len(mp4_uris), videos_dir)
    local_mp4s = _download_gcs_files(uris=mp4_uris, out_dir=videos_dir)

    for i, mp4_path in enumerate(local_mp4s):
        sid = _stable_id(mp4_path)
        out_img = os.path.join(images_dir, f"{i:06d}_{sid}.png")
        out_mask = os.path.join(masks_dir, f"{i:06d}_{sid}.png")

        _extract_first_frame_ffmpeg(mp4_path, out_img)
        pil_img = Image.open(out_img).convert("RGB")

        mask = _sam2_road_mask(
            pil_img,
            ckpt_path=sam2_checkpoint_path,
            config_name=sam2_config_name,
            device=sam2_device,
        )
        mask.save(out_mask)

        focus_img = _focus_image_to_mask(pil_img, mask)

        prompt = _caption_qwen(
            focus_img,
            model_path=VLM_72B_FUSE_ROOT,
            qwen_device=qwen_device,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        if not prompt:
            prompt = "a front-facing driving camera frame"

        row = {
            "image": f"images/{os.path.basename(out_img)}",
            "mask": f"masks/{os.path.basename(out_mask)}",
            "prompt": prompt,
            "prompt_2": prompt,
        }

        # Upload image+mask immediately, then update train.csv on GCS.
        _upload_file_to_gcs(fs, out_img, f"{remote_images_prefix}/{os.path.basename(out_img)}")
        _upload_file_to_gcs(fs, out_mask, f"{remote_masks_prefix}/{os.path.basename(out_mask)}")

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=["image", "mask", "prompt", "prompt_2"])
            wri.writerow(row)

        # Overwrite remote CSV each time (simple + robust for resumability).
        _upload_file_to_gcs(fs, csv_path, remote_csv_uri)

        if (i + 1) % 25 == 0:
            logger.info("Processed %d/%d", i + 1, num_videos)

    # If we processed at least one file, ensure train.csv is present remotely.
    _upload_file_to_gcs(fs, csv_path, remote_csv_uri)
    logger.info("Upload complete (incremental): %s", dest_prefix)
    return dest_prefix + "/"


@workflow
def fluxfill_data_generation_wf(
    num_videos: int = 200,
    start_index: int = 0,
    seed: int = 42,
    qwen_device: str = "cuda:0",
    sam2_device: str = "cuda:0",
    sam2_checkpoint_path: str = SAM2_CHECKPOINT_MOUNTED_PATH,
    sam2_config_name: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    max_walk_files: int = 500000,
    output_run_id: Optional[str] = None,
) -> str:
    return generate_fluxfill_training_data(
        num_videos=num_videos,
        start_index=start_index,
        seed=seed,
        qwen_device=qwen_device,
        sam2_device=sam2_device,
        sam2_checkpoint_path=sam2_checkpoint_path,
        sam2_config_name=sam2_config_name,
        max_walk_files=max_walk_files,
        output_run_id=output_run_id,
    )
