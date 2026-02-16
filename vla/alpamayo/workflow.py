"""Alpamayo VLA inference workflow for HLX.

Runs the Alpamayo-R1-10B model inference on video data and produces
trajectory predictions with reasoning traces.
"""
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import gcsfs
import psutil

from hlx.wf import DedicatedNode, Node, task, workflow
from hlx.wf.mounts import MOUNTPOINT, FuseBucket

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------------
VP_BUCKET = "mbadas-sandbox-research-9bb9c7f"
VLA_BASE_PREFIX = "workspace/user/hbaskar/Video_inpainting/vla"

# Checkpoint paths in GCS
ALPAMAYO_CKPT_PREFIX = os.path.join(VLA_BASE_PREFIX, "alpamayo", "checkpoints")

# Output path in GCS
# Override via ALPAMAYO_OUTPUT_BASE env var in build_and_run.sh
VLA_OUTPUT_PREFIX = os.environ.get(
    "ALPAMAYO_OUTPUT_BASE",
    "workspace/user/hbaskar/outputs/alpamayo",
)

# Container image (set by build_and_run.sh)
REMOTE_IMAGE_DEFAULT = "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/alpamayo_vla"
CONTAINER_IMAGE = os.environ.get("ALPAMAYO_CONTAINER_IMAGE", f"{REMOTE_IMAGE_DEFAULT}:latest")

# Compute node - A100_40GB no longer offered; use A100_80GB instead
COMPUTE_NODE = Node.A100_80GB_1GPU

# FuseBucket mount paths
CKPT_FUSE_MOUNT_NAME = "alpamayo-ckpt"
CKPT_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, CKPT_FUSE_MOUNT_NAME)

# FuseBucket for video data input
# Override: ALPAMAYO_VIDEO_DATA_PREFIX="workspace/..." bash scripts/build_and_run.sh
VIDEO_DATA_GCS_PREFIX = os.environ.get(
    "ALPAMAYO_VIDEO_DATA_PREFIX",
    "workspace/user/hbaskar/outputs/vp",
)
VIDEO_DATA_FUSE_MOUNT_NAME = "alpamayo-video-data"
VIDEO_DATA_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, VIDEO_DATA_FUSE_MOUNT_NAME)

# Local paths inside container
BASE_WORKDIR = "/workspace/alpamayo"
CKPT_LOCAL_PATH = os.path.join(BASE_WORKDIR, "checkpoints")
SCRATCH_OUTPUT_BASE = "/tmp/alpamayo_output"


@dataclass
class VLAVideoMetrics:
    """Metrics for a single video inference."""
    video_id: str
    video_path: str
    inference_time_seconds: float
    gpu_memory_used_gb: float
    gpu_memory_peak_gb: float
    ram_used_mb: float
    ram_peak_mb: float
    num_trajectories: int
    success: bool
    min_ade_meters: Optional[float] = None
    clip_id: Optional[str] = None
    camera_name: Optional[str] = None
    error_message: Optional[str] = None


def _sanitize_path_component(text: str, max_len: int = 100) -> str:
    """Make a safe path component for GCS/local paths."""
    base = re.sub(r"[^a-zA-Z0-9_\-]", "_", (text or "").strip()).strip("_") or "video"
    return base[:max_len].rstrip("_") if len(base) > max_len else base


def _get_gpu_memory_gb() -> tuple[float, float]:
    """Get current and peak GPU memory usage in GB."""
    import torch
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    current = torch.cuda.memory_allocated() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return current, peak


def _reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _get_ram_mb() -> float:
    """Get current RAM usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def ensure_symlink(src: str, dest: str) -> None:
    """Create symlink from src to dest, handling existing files/links."""
    dest_path = Path(dest)
    
    if dest_path.exists() or dest_path.is_symlink():
        if dest_path.is_symlink():
            existing_target = os.readlink(dest)
            if existing_target == src:
                logger.info(f"Symlink already exists: {dest} -> {src}")
                return
            dest_path.unlink()
        elif dest_path.is_dir():
            shutil.rmtree(dest)
        else:
            dest_path.unlink()
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dest)
    logger.info(f"Created symlink: {dest} -> {src}")


def _resolve_fuse_data_path(video_data_gcs_path: str) -> str:
    """Resolve a GCS URI to the corresponding FuseBucket local path.

    The video data GCS path is expected to fall under VIDEO_DATA_GCS_PREFIX.
    e.g. gs://bucket/<prefix>/subdir  ->  <VIDEO_DATA_FUSE_MOUNT_ROOT>/subdir
    """
    expected_prefix = f"gs://{VP_BUCKET}/{VIDEO_DATA_GCS_PREFIX}"
    stripped = video_data_gcs_path.rstrip("/")
    if not stripped.startswith(expected_prefix):
        raise ValueError(
            f"video_data_gcs_path={video_data_gcs_path!r} does not start with "
            f"expected prefix {expected_prefix!r}. "
            f"Set ALPAMAYO_VIDEO_DATA_PREFIX env var to the correct GCS prefix."
        )
    relative = stripped[len(expected_prefix):].lstrip("/")
    return os.path.join(VIDEO_DATA_FUSE_MOUNT_ROOT, relative) if relative else VIDEO_DATA_FUSE_MOUNT_ROOT


def _stage_video_data(video_gcs_path: str) -> list[str]:
    """Discover video files via the FuseBucket mount (no gsutil needed)."""
    fuse_path = _resolve_fuse_data_path(video_gcs_path)

    if not os.path.isdir(fuse_path):
        raise FileNotFoundError(
            f"Video data directory not found at FuseBucket path: {fuse_path} "
            f"(resolved from {video_gcs_path})"
        )

    logger.info(f"Scanning video data at FuseBucket path: {fuse_path}")

    # Find all video files
    video_files = []
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        video_files.extend(list(Path(fuse_path).rglob(f"*{ext}")))

    video_paths = sorted(str(p) for p in video_files)
    logger.info(f"Found {len(video_paths)} video files")

    return video_paths


def _load_model(model_id: str, device: str = "cuda"):
    """Load Alpamayo model, processor, and helper once.

    Returns (model, processor, helper_mod, device_str).
    """
    import torch
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper as helper_mod

    resolved_device = device if device != "auto" else "cuda"
    logger.info(f"Loading Alpamayo model from {model_id} …")
    model = AlpamayoR1.from_pretrained(model_id, dtype=torch.bfloat16).to(resolved_device)
    processor = helper_mod.get_processor(model.tokenizer)
    logger.info("Model loaded successfully")
    return model, processor, helper_mod, resolved_device


def _run_alpamayo_inference(
    video_path: str,
    output_dir: str,
    model,
    processor,
    helper_mod,
    device: str = "cuda",
    num_traj_samples: int = 1,
) -> VLAVideoMetrics:
    """Run Alpamayo inference on a single video *in-process* (no subprocess)."""
    from run_inference import run_inference_on_video

    video_id = Path(video_path).stem
    logger.info(f"Running inference on video: {video_id}")

    # Create output directory for this video
    video_output_dir = os.path.join(output_dir, video_id)
    Path(video_output_dir).mkdir(parents=True, exist_ok=True)

    try:
        result = run_inference_on_video(
            video_path=video_path,
            model=model,
            processor=processor,
            helper_mod=helper_mod,
            output_dir=video_output_dir,
            num_traj_samples=num_traj_samples,
            device=device,
        )

        return VLAVideoMetrics(
            video_id=result.get("video_id", video_id),
            video_path=video_path,
            inference_time_seconds=result["metrics"]["inference_time_seconds"],
            gpu_memory_used_gb=result["metrics"]["gpu_memory_used_gb"],
            gpu_memory_peak_gb=result["metrics"]["gpu_memory_peak_gb"],
            ram_used_mb=result["metrics"]["ram_used_mb"],
            ram_peak_mb=result["metrics"]["ram_peak_mb"],
            num_trajectories=result.get("num_trajectories", 0),
            success=result["success"],
            min_ade_meters=result.get("min_ade_meters"),
            clip_id=result.get("clip_id"),
            camera_name=result.get("camera_name"),
            error_message=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Error during inference on {video_id}: {e}", exc_info=True)

        gpu_current, gpu_peak = _get_gpu_memory_gb()

        return VLAVideoMetrics(
            video_id=video_id,
            video_path=video_path,
            inference_time_seconds=0.0,
            gpu_memory_used_gb=gpu_current,
            gpu_memory_peak_gb=gpu_peak,
            ram_used_mb=_get_ram_mb(),
            ram_peak_mb=_get_ram_mb(),
            num_trajectories=0,
            success=False,
            error_message=str(e),
        )


def _write_report(
    output_dir: str,
    run_id: str,
    metrics: list[VLAVideoMetrics],
    video_data_source: str,
) -> str:
    """Write comprehensive report with all metrics."""
    report_path = os.path.join(output_dir, f"{run_id}_report.txt")
    
    import numpy as np

    total_time = sum(m.inference_time_seconds for m in metrics)
    successful = sum(1 for m in metrics if m.success)
    failed = len(metrics) - successful
    
    avg_gpu_peak = np.mean([m.gpu_memory_peak_gb for m in metrics if m.success])
    max_gpu_peak = max([m.gpu_memory_peak_gb for m in metrics], default=0.0)
    
    avg_ram_peak = np.mean([m.ram_peak_mb for m in metrics if m.success])
    max_ram_peak = max([m.ram_peak_mb for m in metrics], default=0.0)
    
    avg_time = np.mean([m.inference_time_seconds for m in metrics if m.success])

    ade_values = [m.min_ade_meters for m in metrics if m.success and m.min_ade_meters is not None]
    avg_min_ade = np.mean(ade_values) if ade_values else float("nan")
    best_min_ade = min(ade_values) if ade_values else float("nan")
    worst_min_ade = max(ade_values) if ade_values else float("nan")
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ALPAMAYO VLA INFERENCE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Video Data Source: {video_data_source}\n")
        f.write(f"Total Videos: {len(metrics)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Total Inference Time: {total_time:.2f}s ({total_time/60:.2f}min)\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("AGGREGATE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Inference Time per Video: {avg_time:.2f}s\n")
        f.write(f"Average GPU Memory Peak: {avg_gpu_peak:.2f} GB\n")
        f.write(f"Maximum GPU Memory Peak: {max_gpu_peak:.2f} GB\n")
        f.write(f"Average RAM Peak: {avg_ram_peak:.2f} MB\n")
        f.write(f"Maximum RAM Peak: {max_ram_peak:.2f} MB\n")
        f.write(f"Average minADE: {avg_min_ade:.4f} m\n")
        f.write(f"Best minADE: {best_min_ade:.4f} m\n")
        f.write(f"Worst minADE: {worst_min_ade:.4f} m\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("PER-VIDEO METRICS\n")
        f.write("-" * 80 + "\n\n")
        
        for m in metrics:
            f.write(f"Video ID: {m.video_id}\n")
            f.write(f"  Path: {m.video_path}\n")
            f.write(f"  Status: {'SUCCESS' if m.success else 'FAILED'}\n")
            if not m.success and m.error_message:
                f.write(f"  Error: {m.error_message}\n")
            f.write(f"  Inference Time: {m.inference_time_seconds:.2f}s\n")
            f.write(f"  GPU Memory (current/peak): {m.gpu_memory_used_gb:.2f} / {m.gpu_memory_peak_gb:.2f} GB\n")
            f.write(f"  RAM (current/peak): {m.ram_used_mb:.2f} / {m.ram_peak_mb:.2f} MB\n")
            f.write(f"  Num Trajectories: {m.num_trajectories}\n")
            if m.min_ade_meters is not None:
                f.write(f"  minADE: {m.min_ade_meters:.4f} m\n")
            if m.clip_id:
                f.write(f"  Clip ID: {m.clip_id}\n")
            if m.camera_name:
                f.write(f"  Camera: {m.camera_name}\n")
            f.write("\n")
    
    logger.info(f"Report written to: {report_path}")
    return report_path


def _upload_directory_to_gcs(local_dir: str, gcs_prefix: str) -> None:
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


def _upload_outputs(local_dir: str, gcs_base: str, run_id: str) -> str:
    """Upload outputs to GCS using gcsfs (reliable write path)."""
    gcs_dest = f"{VP_BUCKET}/{gcs_base}/{run_id}"
    logger.info(f"Uploading outputs from {local_dir} to gs://{gcs_dest}")

    try:
        _upload_directory_to_gcs(local_dir, gcs_dest)
        logger.info(f"Outputs available at gs://{gcs_dest}")
    except Exception as e:
        logger.error(f"Failed to upload outputs to GCS: {e}", exc_info=True)
        raise

    return f"gs://{gcs_dest}"


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
        "HF_HOME": "/root/.cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    },
    mounts=[
        FuseBucket(
            bucket=VP_BUCKET,
            name=CKPT_FUSE_MOUNT_NAME,
            prefix=ALPAMAYO_CKPT_PREFIX,
        ),
        FuseBucket(
            bucket=VP_BUCKET,
            name=VIDEO_DATA_FUSE_MOUNT_NAME,
            prefix=VIDEO_DATA_GCS_PREFIX,
        ),
    ],
)
def run_alpamayo_inference_task(
    video_data_gcs_path: str,
    output_run_id: str,
    model_id: str = "nvidia/Alpamayo-R1-10B",
    num_traj_samples: int = 1,
) -> dict:
    """
    Run Alpamayo VLA inference on video data.
    
    Args:
        video_data_gcs_path: GCS path to video data (e.g., gs://bucket/path/to/videos)
        output_run_id: Unique identifier for this run
        model_id: HuggingFace model ID or local path
        num_traj_samples: Number of trajectory samples per video
    
    Returns:
        Dictionary with output GCS path and metrics
    """
    logger.info("=" * 80)
    logger.info("ALPAMAYO VLA INFERENCE TASK")
    logger.info("=" * 80)
    logger.info(f"Video Data Source: {video_data_gcs_path}")
    logger.info(f"Output Run ID: {output_run_id}")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Container Image: {CONTAINER_IMAGE}")
    
    # Setup checkpoint symlink
    logger.info(f"Setting up checkpoint symlink from {CKPT_FUSE_MOUNT_ROOT} to {CKPT_LOCAL_PATH}")
    ensure_symlink(CKPT_FUSE_MOUNT_ROOT, CKPT_LOCAL_PATH)
    
    # Discover video files via FuseBucket mount
    video_paths = _stage_video_data(video_data_gcs_path)
    
    if not video_paths:
        raise ValueError(f"No videos found in {video_data_gcs_path}")
    
    logger.info(f"Processing {len(video_paths)} videos")
    
    # Load model ONCE for all videos
    model, processor, helper_mod, device = _load_model(model_id)
    
    # Create output directory
    local_output_dir = os.path.join(SCRATCH_OUTPUT_BASE, output_run_id)
    Path(local_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run inference on each video (in-process, reusing loaded model)
    all_metrics = []
    for i, video_path in enumerate(video_paths, 1):
        logger.info(f"Processing video {i}/{len(video_paths)}: {video_path}")
        
        video_output_dir = os.path.join(local_output_dir, Path(video_path).stem)
        metrics = _run_alpamayo_inference(
            video_path=video_path,
            output_dir=video_output_dir,
            model=model,
            processor=processor,
            helper_mod=helper_mod,
            device=device,
            num_traj_samples=num_traj_samples,
        )
        all_metrics.append(metrics)
    
    # Write report
    report_path = _write_report(
        output_dir=local_output_dir,
        run_id=output_run_id,
        metrics=all_metrics,
        video_data_source=video_data_gcs_path,
    )
    
    # Upload outputs to GCS
    gcs_output_path = _upload_outputs(local_output_dir, VLA_OUTPUT_PREFIX, output_run_id)
    
    logger.info("=" * 80)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"Output Location: {gcs_output_path}")
    logger.info(f"Report: {gcs_output_path}/{output_run_id}_report.txt")
    logger.info("=" * 80)
    
    return {
        "output_gcs_path": gcs_output_path,
        "report_path": f"{gcs_output_path}/{output_run_id}_report.txt",
        "num_videos": len(video_paths),
        "num_successful": sum(1 for m in all_metrics if m.success),
        "num_failed": sum(1 for m in all_metrics if not m.success),
    }


@workflow
def alpamayo_vla_inference_wf(
    video_data_gcs_path: str,
    output_run_id: str,
    model_id: str = "nvidia/Alpamayo-R1-10B",
    num_traj_samples: int = 1,
) -> dict:
    """
    Alpamayo VLA inference workflow.
    
    Args:
        video_data_gcs_path: GCS path to video data
        output_run_id: Unique identifier for this run
        model_id: Model identifier
        num_traj_samples: Number of trajectory samples
    
    Returns:
        Dictionary with results
    """
    return run_alpamayo_inference_task(
        video_data_gcs_path=video_data_gcs_path,
        output_run_id=output_run_id,
        model_id=model_id,
        num_traj_samples=num_traj_samples,
    )
