"""Alpamayo VLA inference workflow for HLX.

Runs the Alpamayo-R1-10B model inference on video data and produces
trajectory predictions with reasoning traces.
"""
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch

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
    "workspace/user/hbaskar/Video_inpainting/videopainter/training/output/alpamayo",
)

# Container image (set by build_and_run.sh)
REMOTE_IMAGE_DEFAULT = "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/alpamayo_vla"
CONTAINER_IMAGE = os.environ.get("ALPAMAYO_CONTAINER_IMAGE", f"{REMOTE_IMAGE_DEFAULT}:latest")

# Compute node - requires ≥24 GB VRAM
COMPUTE_NODE = Node.A100_40GB_1GPU

# FuseBucket mount paths
CKPT_FUSE_MOUNT_NAME = "alpamayo-ckpt"
CKPT_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, CKPT_FUSE_MOUNT_NAME)

# Local paths inside container
BASE_WORKDIR = "/workspace/alpamayo"
CKPT_LOCAL_PATH = os.path.join(BASE_WORKDIR, "checkpoints")
SCRATCH_DATA_BASE = "/tmp/alpamayo_data"
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
    error_message: Optional[str] = None


def _sanitize_path_component(text: str, max_len: int = 100) -> str:
    """Make a safe path component for GCS/local paths."""
    base = re.sub(r"[^a-zA-Z0-9_\-]", "_", (text or "").strip()).strip("_") or "video"
    return base[:max_len].rstrip("_") if len(base) > max_len else base


def _get_gpu_memory_gb() -> tuple[float, float]:
    """Get current and peak GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    current = torch.cuda.memory_allocated() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return current, peak


def _reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
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


def _stage_video_data(video_gcs_path: str, local_dir: str) -> list[str]:
    """Download video data from GCS to local directory."""
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading video data from {video_gcs_path} to {local_dir}")
    
    # Use gsutil to download the data
    cmd = f"gsutil -m cp -r {video_gcs_path}/* {local_dir}/"
    ret = os.system(cmd)
    
    if ret != 0:
        raise RuntimeError(f"Failed to download video data from {video_gcs_path}")
    
    # Find all video files
    video_files = []
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        video_files.extend(list(Path(local_dir).rglob(f"*{ext}")))
    
    video_paths = [str(p) for p in video_files]
    logger.info(f"Found {len(video_paths)} video files in {local_dir}")
    
    return video_paths


def _run_alpamayo_inference(
    video_path: str,
    output_dir: str,
    model_id: str = "nvidia/Alpamayo-R1-10B",
    num_traj_samples: int = 1,
) -> VLAVideoMetrics:
    """Run Alpamayo inference on a single video using the run_inference.py script."""
    video_id = Path(video_path).stem
    logger.info(f"Running inference on video: {video_id}")
    
    # Reset metrics
    _reset_gpu_memory_stats()
    ram_start = _get_ram_mb()
    
    start_time = time.time()
    
    # Create output directory for this video
    video_output_dir = os.path.join(output_dir, video_id)
    Path(video_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run the inference script
    inference_script = os.path.join(BASE_WORKDIR, "run_inference.py")
    cmd = [
        "python",
        inference_script,
        "--video_path", video_path,
        "--output_dir", video_output_dir,
        "--model_id", model_id,
        "--num_traj_samples", str(num_traj_samples),
        "--device", "auto",
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        import subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        logger.info(f"Inference stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Inference stderr: {result.stderr}")
        
        # Load the inference results
        inference_result_file = os.path.join(video_output_dir, f"{video_id}_inference.json")
        if os.path.exists(inference_result_file):
            import json
            with open(inference_result_file, "r") as f:
                inference_data = json.load(f)
            
            inference_time = time.time() - start_time
            
            return VLAVideoMetrics(
                video_id=video_id,
                video_path=video_path,
                inference_time_seconds=inference_data["metrics"]["inference_time_seconds"],
                gpu_memory_used_gb=inference_data["metrics"]["gpu_memory_used_gb"],
                gpu_memory_peak_gb=inference_data["metrics"]["gpu_memory_peak_gb"],
                ram_used_mb=inference_data["metrics"]["ram_used_mb"],
                ram_peak_mb=inference_data["metrics"]["ram_peak_mb"],
                num_trajectories=inference_data["num_trajectories"],
                success=inference_data["success"],
                error_message=inference_data.get("error"),
            )
        else:
            raise FileNotFoundError(f"Inference result file not found: {inference_result_file}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference script failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        
        inference_time = time.time() - start_time
        gpu_current, gpu_peak = _get_gpu_memory_gb()
        
        return VLAVideoMetrics(
            video_id=video_id,
            video_path=video_path,
            inference_time_seconds=inference_time,
            gpu_memory_used_gb=gpu_current,
            gpu_memory_peak_gb=gpu_peak,
            ram_used_mb=_get_ram_mb(),
            ram_peak_mb=_get_ram_mb(),
            num_trajectories=0,
            success=False,
            error_message=f"Script failed: {e.stderr}",
        )
    except Exception as e:
        logger.error(f"Error during inference on {video_id}: {e}")
        
        inference_time = time.time() - start_time
        gpu_current, gpu_peak = _get_gpu_memory_gb()
        
        return VLAVideoMetrics(
            video_id=video_id,
            video_path=video_path,
            inference_time_seconds=inference_time,
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
    
    total_time = sum(m.inference_time_seconds for m in metrics)
    successful = sum(1 for m in metrics if m.success)
    failed = len(metrics) - successful
    
    avg_gpu_peak = np.mean([m.gpu_memory_peak_gb for m in metrics if m.success])
    max_gpu_peak = max([m.gpu_memory_peak_gb for m in metrics], default=0.0)
    
    avg_ram_peak = np.mean([m.ram_peak_mb for m in metrics if m.success])
    max_ram_peak = max([m.ram_peak_mb for m in metrics], default=0.0)
    
    avg_time = np.mean([m.inference_time_seconds for m in metrics if m.success])
    
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
            f.write("\n")
    
    logger.info(f"Report written to: {report_path}")
    return report_path


def _upload_outputs(local_dir: str, gcs_base: str, run_id: str) -> str:
    """Upload output directory to GCS."""
    gcs_dest = os.path.join(gcs_base, run_id)
    logger.info(f"Uploading outputs from {local_dir} to gs://{VP_BUCKET}/{gcs_dest}")
    
    cmd = f"gsutil -m cp -r {local_dir}/* gs://{VP_BUCKET}/{gcs_dest}/"
    ret = os.system(cmd)
    
    if ret != 0:
        logger.error(f"Failed to upload outputs to GCS")
    else:
        logger.info(f"Outputs uploaded to gs://{VP_BUCKET}/{gcs_dest}")
    
    return f"gs://{VP_BUCKET}/{gcs_dest}"


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
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
    },
    mounts=[
        FuseBucket(
            bucket=VP_BUCKET,
            name=CKPT_FUSE_MOUNT_NAME,
            prefix=ALPAMAYO_CKPT_PREFIX,
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
    
    # Stage video data
    local_data_dir = os.path.join(SCRATCH_DATA_BASE, output_run_id)
    video_paths = _stage_video_data(video_data_gcs_path, local_data_dir)
    
    if not video_paths:
        raise ValueError(f"No videos found in {video_data_gcs_path}")
    
    logger.info(f"Processing {len(video_paths)} videos")
    
    # Create output directory
    local_output_dir = os.path.join(SCRATCH_OUTPUT_BASE, output_run_id)
    Path(local_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run inference on each video
    all_metrics = []
    for i, video_path in enumerate(video_paths, 1):
        logger.info(f"Processing video {i}/{len(video_paths)}: {video_path}")
        
        video_output_dir = os.path.join(local_output_dir, Path(video_path).stem)
        metrics = _run_alpamayo_inference(
            video_path=video_path,
            output_dir=video_output_dir,
            model_id=model_id,
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
