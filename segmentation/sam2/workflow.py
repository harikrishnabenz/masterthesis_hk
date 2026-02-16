"""SAM2 segmentation workflow for HLX.

Runs the SAM2 video segmentation pipeline (process_videos_sam2.py) inside a
container for road segmentation on autonomous driving videos.
"""
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from hlx.wf import DedicatedNode, Node, fuse_prefetch_metadata, task, workflow
from hlx.wf.mounts import MOUNTPOINT, FuseBucket

logger = logging.getLogger(__name__)


# Allow the runner script to pin an exact image tag
CONTAINER_IMAGE_DEFAULT = "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2"
CONTAINER_IMAGE = os.environ.get("SAM2_CONTAINER_IMAGE", CONTAINER_IMAGE_DEFAULT)

# ----------------------------------------------------------------------------------
# PATHS (inside container)
# ----------------------------------------------------------------------------------
BASE_WORKDIR = "/workspace/sam2"
DEFAULT_CHECKPOINT = os.path.join(BASE_WORKDIR, "checkpoints", "sam2.1_hiera_large.pt")
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# GCS bucket paths for outputs (run_id will be passed as parameter)
# Override via SAM2_OUTPUT_BASE env var in build_and_run.sh
SAM2_OUTPUT_BUCKET_BASE = os.environ.get(
    "SAM2_OUTPUT_BASE",
    "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/output/sam2",
)
SAM2_PREPROCESSED_BUCKET_BASE = "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/preprocessed_data_vp"

# GCS bucket paths for checkpoints (mounted via FuseBucket)
SAM2_BUCKET = "mbadas-sandbox-research-9bb9c7f"
SAM2_CHECKPOINT_PREFIX = "workspace/user/hbaskar/Video_inpainting/sam2_checkpoint"

# GCS bucket paths for input videos (mounted via FuseBucket)
INPUT_VIDEOS_PREFIX = "datasets/public/physical_ai_av/camera/camera_front_tele_30fov"

# IMPORTANT: FuseBucket mounts at /mnt/{name}, not /mnt/{bucket}.
# Because we mount with a prefix, /mnt/{name} corresponds to gs://{bucket}/{prefix}.
SAM2_FUSE_MOUNT_NAME = "sam2-checkpoints"
SAM2_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, SAM2_FUSE_MOUNT_NAME)
MOUNTED_CHECKPOINT_PATH = os.path.join(SAM2_FUSE_MOUNT_ROOT, "checkpoints", "sam2.1_hiera_large.pt")

INPUT_VIDEOS_MOUNT_NAME = "input-videos"
INPUT_VIDEOS_MOUNT_ROOT = os.path.join(MOUNTPOINT, INPUT_VIDEOS_MOUNT_NAME)

# Default video URIs (can be overridden)
DEFAULT_VIDEO_URIS = [
    "https://storage.googleapis.com/mbadas-sandbox-research-9bb9c7f/datasets/public/physical_ai_av/camera/camera_front_tele_30fov/25534c8d-4d02-463a-84c9-dad015f320ac.camera_front_tele_30fov.mp4",
    "https://storage.googleapis.com/mbadas-sandbox-research-9bb9c7f/datasets/public/physical_ai_av/camera/camera_front_tele_30fov/13e7d364-6476-4f2f-b94e-060440bf1a36.camera_front_tele_30fov.mp4",
    "https://storage.googleapis.com/mbadas-sandbox-research-9bb9c7f/datasets/public/physical_ai_av/camera/camera_front_tele_30fov/025887fd-9f6a-4ba4-aa30-60d7ab1e137f.camera_front_tele_30fov.mp4",
    "https://storage.googleapis.com/mbadas-sandbox-research-9bb9c7f/datasets/public/physical_ai_av/camera/camera_front_tele_30fov/ee680a47-b981-468f-b817-8712af5953d5.camera_front_tele_30fov.mp4",
    "https://storage.googleapis.com/mbadas-sandbox-research-9bb9c7f/datasets/public/physical_ai_av/camera/camera_front_tele_30fov/d3317bae-0c7e-4e34-8975-50b6bd715dc3.camera_front_tele_30fov.mp4",
    "https://storage.googleapis.com/mbadas-sandbox-research-9bb9c7f/datasets/public/physical_ai_av/camera/camera_front_tele_30fov/4809025e-0cef-414c-bf59-8c86a5177ef7.camera_front_tele_30fov.mp4",
    "https://storage.googleapis.com/mbadas-sandbox-research-9bb9c7f/datasets/public/physical_ai_av/camera/camera_front_tele_30fov/83563feb-695f-4152-a0bf-346d56f89373.camera_front_tele_30fov.mp4",
    "https://storage.googleapis.com/mbadas-sandbox-research-9bb9c7f/datasets/public/physical_ai_av/camera/camera_front_tele_30fov/99b474d6-9ea5-4e17-87a2-84267728763d.camera_front_tele_30fov.mp4",
    "https://storage.googleapis.com/mbadas-sandbox-research-9bb9c7f/datasets/public/physical_ai_av/camera/camera_front_tele_30fov/d2cadb4e-585e-4b7f-890f-2fa198713203.camera_front_tele_30fov.mp4",
    "https://storage.googleapis.com/mbadas-sandbox-research-9bb9c7f/datasets/public/physical_ai_av/camera/camera_front_tele_30fov/6e08b4de-9282-409f-be26-2d24e066baac.camera_front_tele_30fov.mp4",
]


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
                import shutil
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


@task(
    compute=DedicatedNode(
        node=Node.A100_80GB_1GPU,
        ephemeral_storage="max",
        max_duration="3d",
    ),
    container_image=CONTAINER_IMAGE,
    environment={"PYTHONUNBUFFERED": "1"},
    mounts=[
        FuseBucket(
            bucket=SAM2_BUCKET,
            name=SAM2_FUSE_MOUNT_NAME,
            prefix=SAM2_CHECKPOINT_PREFIX,
        ),
        FuseBucket(
            bucket=SAM2_BUCKET,
            name=INPUT_VIDEOS_MOUNT_NAME,
            prefix=INPUT_VIDEOS_PREFIX,
        ),
    ],
)
def run_sam2_segmentation(
    run_id: str,
    video_uris: Optional[List[str]] = None,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    model_config: str = DEFAULT_CONFIG,
    upload_to_gcp: bool = True,
    upload_to_local: bool = False,
    max_frames: int = 150,
) -> str:
    """Run SAM2 video segmentation on a list of video URIs.
    
    Args:
        run_id: Unique identifier for this run (used in output paths)
        video_uris: List of video URIs to process (HTTP/HTTPS or gs:// URIs)
        checkpoint_path: Path to SAM2 model checkpoint
        model_config: Model configuration file path
        upload_to_gcp: Whether to upload results to GCS
        upload_to_local: Whether to keep local copies
        max_frames: Maximum frames to extract per video
    
    Returns:
        Summary message with GCS paths
    """
    if video_uris is None:
        video_uris = DEFAULT_VIDEO_URIS
    
    # Construct output paths with run_id
    output_bucket = f"{SAM2_OUTPUT_BUCKET_BASE}/{run_id}"
    preprocessed_bucket = f"{SAM2_PREPROCESSED_BUCKET_BASE}/{run_id}"
    
    logger.info("="*60)
    logger.info(f"SAM2 Segmentation Workflow")
    logger.info(f"Processing {len(video_uris)} videos")
    logger.info(f"RUN_ID: {run_id}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Model config: {model_config}")
    logger.info(f"Output bucket: {output_bucket}")
    logger.info(f"Preprocessed bucket: {preprocessed_bucket}")
    logger.info("")
    logger.info("NOTE: To use this SAM2 output in VideoPainter, run:")
    logger.info(f"  hlx wf run workflow.videopainter_wf --data_run_id {run_id}")
    logger.info("="*60)
    
    # Prefetch mounted checkpoints if using default path
    if checkpoint_path == DEFAULT_CHECKPOINT:
        logger.info("Using mounted checkpoint from GCS")
        logger.info(f"MOUNTPOINT={MOUNTPOINT}")
        logger.info(f"SAM2_FUSE_MOUNT_ROOT={SAM2_FUSE_MOUNT_ROOT}")
        logger.info(f"MOUNTED_CHECKPOINT_PATH={MOUNTED_CHECKPOINT_PATH}")
        
        # Prefetch checkpoint metadata for faster access
        try:
            fuse_prefetch_metadata(SAM2_FUSE_MOUNT_ROOT)
        except Exception as e:
            logger.warning(f"Checkpoint prefetch failed (non-fatal): {e}")
        
        # Create symlink from mounted checkpoints to expected local path
        local_checkpoint_dir = os.path.join(BASE_WORKDIR, "checkpoints")
        mounted_checkpoint_dir = os.path.join(SAM2_FUSE_MOUNT_ROOT, "checkpoints")
        
        if os.path.exists(mounted_checkpoint_dir):
            ensure_symlink(mounted_checkpoint_dir, local_checkpoint_dir)
            logger.info(f"Linked mounted checkpoints: {mounted_checkpoint_dir} -> {local_checkpoint_dir}")
            
            # Verify the checkpoint file is accessible through the symlink
            if os.path.exists(checkpoint_path):
                logger.info(f"Using checkpoint at: {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path} after linking")
        else:
            logger.warning(f"Mounted checkpoint directory not found at {mounted_checkpoint_dir}, using baked-in checkpoint")
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint_path}. "
            f"Please ensure the checkpoint is available in the container."
        )
    
    # Setup input videos - create symlinks from mounted videos
    logger.info("Setting up input videos from mounted GCS path...")
    logger.info(f"INPUT_VIDEOS_MOUNT_ROOT={INPUT_VIDEOS_MOUNT_ROOT}")
    
    # Create a local cache directory for symlinks
    video_cache_dir = Path("/tmp/sam2_video_cache")
    video_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Prefetch video metadata for faster access
    try:
        fuse_prefetch_metadata(INPUT_VIDEOS_MOUNT_ROOT)
    except Exception as e:
        logger.warning(f"Video mount prefetch failed (non-fatal): {e}")
    
    # For each video URI, extract the filename and create a symlink from mounted path
    local_video_paths = []
    for uri in video_uris:
        # Extract filename from URI (last part after /)
        video_filename = uri.split("/")[-1]
        mounted_video_path = os.path.join(INPUT_VIDEOS_MOUNT_ROOT, video_filename)
        local_video_path = video_cache_dir / video_filename
        
        if os.path.exists(mounted_video_path):
            # Create symlink to mounted video
            if not local_video_path.exists():
                os.symlink(mounted_video_path, local_video_path)
                logger.info(f"Linked video: {mounted_video_path} -> {local_video_path}")
            local_video_paths.append(str(local_video_path))
        else:
            logger.warning(f"Video not found in mounted path: {mounted_video_path}")
            # Fallback to original URI (will be downloaded)
            local_video_paths.append(uri)
    
    # Update video_uris to use local paths
    video_uris = local_video_paths
    logger.info(f"Prepared {len(local_video_paths)} video paths")
    
    # Build command to run the processing script
    script_path = os.path.join(BASE_WORKDIR, "process_vide_sam2_hlxwf.py")
    cmd = [
        sys.executable,
        script_path,
        "--video-uris", *video_uris,
        "--checkpoint", checkpoint_path,
        "--model-cfg", model_config,
        "--output-bucket", output_bucket,
        "--preprocessed-bucket", preprocessed_bucket,
        "--max-frames", str(max_frames),
        "--run-id", run_id,
    ]
    
    if upload_to_gcp:
        cmd.append("--upload-gcp")
    if upload_to_local:
        cmd.append("--upload-local")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=BASE_WORKDIR,
    )
    
    if result.stdout:
        logger.info("STDOUT:")
        logger.info(result.stdout)
    
    if result.stderr:
        logger.warning("STDERR:")
        logger.warning(result.stderr)
    
    if result.returncode != 0:
        stdout_tail = (result.stdout or "")[-4000:]
        stderr_tail = (result.stderr or "")[-4000:]
        raise RuntimeError(
            f"SAM2 processing failed with exit code {result.returncode}.\n"
            f"--- stdout (tail) ---\n{stdout_tail}\n"
            f"--- stderr (tail) ---\n{stderr_tail}"
        )
    
    # Generate summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = f"""
SAM2 Segmentation Complete!
===========================
Videos processed: {len(video_uris)}
Timestamp: {timestamp}

GCS Outputs:
- Raw outputs: {output_bucket}/
- VideoPainter format: {preprocessed_bucket}/

Timing report:
- {output_bucket}/{run_id}.txt

Each video has:
  - Binary masks (PNG)
  - Visualization overlays (JPG + MP4)
  - VideoPainter preprocessing (MP4 + NPZ masks + meta.csv)
"""
    logger.info(summary)
    return summary


@task(
    compute=DedicatedNode(
        node=Node.A100_80GB_1GPU,
        ephemeral_storage="max",
        max_duration="3d",
    ),
    container_image=CONTAINER_IMAGE,
    environment={"PYTHONUNBUFFERED": "1"},
    mounts=[
        FuseBucket(
            bucket=SAM2_BUCKET,
            name=SAM2_FUSE_MOUNT_NAME,
            prefix=SAM2_CHECKPOINT_PREFIX,
        ),
        FuseBucket(
            bucket=SAM2_BUCKET,
            name=INPUT_VIDEOS_MOUNT_NAME,
            prefix=INPUT_VIDEOS_PREFIX,
        ),
    ],
)
def run_sam2_single_video(
    run_id: str,
    video_uri: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    model_config: str = DEFAULT_CONFIG,
    upload_to_gcp: bool = True,
    upload_to_local: bool = False,
    max_frames: int = 150,
) -> str:
    """Run SAM2 segmentation on a single video.
    
    Args:
        run_id: Unique identifier for this run
        video_uri: Single video URI to process
        checkpoint_path: Path to SAM2 model checkpoint
        model_config: Model configuration file path
        upload_to_gcp: Whether to upload results to GCS
        upload_to_local: Whether to keep local copies
        max_frames: Maximum frames to extract
    
    Returns:
        Summary message with GCS paths
    """
    return run_sam2_segmentation(
        run_id=run_id,
        video_uris=[video_uri],
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        upload_to_gcp=upload_to_gcp,
        upload_to_local=upload_to_local,
        max_frames=max_frames,
    )


@workflow
def sam2_segmentation_wf(
    run_id: str,
    video_uris: Optional[List[str]] = None,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    model_config: str = DEFAULT_CONFIG,
    upload_to_gcp: bool = True,
    upload_to_local: bool = False,
    max_frames: int = 150,
) -> str:
    """Workflow: Process multiple videos with SAM2 for road segmentation.
    
    This workflow:
    1. Downloads videos from GCS/HTTP URIs
    2. Extracts frames (up to max_frames per video)
    3. Runs SAM2 segmentation with road-focused initialization
    4. Applies morphological filtering for clean masks
    5. Generates visualization overlays
    6. Uploads results to GCS in two formats:
       - Raw outputs (masks + visualizations + segmented video)
       - VideoPainter preprocessed format (for video inpainting)
    7. Cleans up local files after upload
    
    Args:
        video_uris: List of video URIs (defaults to 10 sample videos)
        checkpoint_path: Path to SAM2.1 Large checkpoint
        model_config: SAM2 model config (sam2.1_hiera_l.yaml)
        output_bucket: GCS bucket for raw segmentation outputs
        preprocessed_bucket: GCS bucket for VideoPainter-formatted data
        upload_to_gcp: Upload results to GCS (default: True)
        upload_to_local: Keep local copies (default: False)
        max_frames: Max frames per video (default: 150)
    
    Returns:
        Summary of processed videos and GCS output locations
    """
    return run_sam2_segmentation(
        run_id=run_id,
        video_uris=video_uris,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        upload_to_gcp=upload_to_gcp,
        upload_to_local=upload_to_local,
        max_frames=max_frames,
    )


@workflow
def sam2_single_video_wf(
    run_id: str,
    video_uri: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    model_config: str = DEFAULT_CONFIG,
    upload_to_gcp: bool = True,
    upload_to_local: bool = False,
    max_frames: int = 150,
) -> str:
    """Workflow: Process a single video with SAM2 for road segmentation.
    
    Args:
        run_id: Unique identifier for this run
        video_uri: Single video URI to process
        checkpoint_path: Path to SAM2.1 Large checkpoint
        model_config: SAM2 model config
        upload_to_gcp: Upload to GCS (default: True)
        upload_to_local: Keep local copy (default: False)
        max_frames: Max frames to extract (default: 150)
    
    Returns:
        Summary with GCS output locations
    """
    return run_sam2_single_video(
        run_id=run_id,
        video_uri=video_uri,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        upload_to_gcp=upload_to_gcp,
        upload_to_local=upload_to_local,
        max_frames=max_frames,
    )


if __name__ == "__main__":
    # Example usage for local testing (requires HLX environment)
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM2 Segmentation Workflow")
    parser.add_argument(
        "--video-uri",
        type=str,
        help="Single video URI to process (for single video mode)",
    )
    parser.add_argument(
        "--video-uris",
        type=str,
        nargs="+",
        help="List of video URIs to process (for batch mode)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Maximum frames to extract per video (default: 150)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip GCS upload (keep local only)",
    )
    
    args = parser.parse_args()
    
    if args.video_uri:
        result = sam2_single_video_wf(
            video_uri=args.video_uri,
            upload_to_gcp=not args.no_upload,
            upload_to_local=args.no_upload,
            max_frames=args.max_frames,
        )
    else:
        result = sam2_segmentation_wf(
            video_uris=args.video_uris,
            upload_to_gcp=not args.no_upload,
            upload_to_local=args.no_upload,
            max_frames=args.max_frames,
        )
    
    print(result)
