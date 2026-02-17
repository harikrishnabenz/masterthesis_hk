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
    "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/sam2",
)
SAM2_PREPROCESSED_BUCKET_BASE = os.environ.get(
    "SAM2_PREPROCESSED_OUTPUT_BASE",
    "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/preprocessed_data_vp",
)

# GCS bucket paths for checkpoints (mounted via FuseBucket)
SAM2_BUCKET = "mbadas-sandbox-research-9bb9c7f"
SAM2_CHECKPOINT_PREFIX = "workspace/user/hbaskar/Video_inpainting/sam2_checkpoint"

# GCS bucket paths for input videos (mounted via FuseBucket)
INPUT_VIDEOS_PREFIX = os.environ.get(
    "SAM2_INPUT_PREFIX",
    "workspace/user/hbaskar/Input/data_physical_ai",
)
# Camera subfolder within the input parent (used for resolving chunk URIs)
INPUT_CAMERA_SUBFOLDER = os.environ.get("SAM2_CAMERA_SUBFOLDER", "camera_front_tele_30fov")

# IMPORTANT: FuseBucket mounts at /mnt/{name}, not /mnt/{bucket}.
# Because we mount with a prefix, /mnt/{name} corresponds to gs://{bucket}/{prefix}.
SAM2_FUSE_MOUNT_NAME = "sam2-checkpoints"
SAM2_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, SAM2_FUSE_MOUNT_NAME)
MOUNTED_CHECKPOINT_PATH = os.path.join(SAM2_FUSE_MOUNT_ROOT, "checkpoints", "sam2.1_hiera_large.pt")


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
    ],
)
def run_sam2_segmentation(
    run_id: str,
    sam2_video_uris: str = "default",
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    model_config: str = DEFAULT_CONFIG,
    upload_to_gcp: bool = True,
    upload_to_local: bool = False,
    max_frames: int = 150,
) -> str:
    """Run SAM2 video segmentation on a list of video URIs.
    
    Args:
        run_id: Unique identifier for this run (used in output paths)
        sam2_video_uris: Video input specification. Accepts:
            - "default"           → 10 built-in sample videos
            - "chunks://..."      → chunk-based URI (resolved inside container)
            - "gs://bucket/folder/" → single GCS folder
            - "gs://bucket/v.mp4" → single GCS file
            - "gs://b/v1.mp4,gs://b/v2.mp4" → comma-separated URIs
        checkpoint_path: Path to SAM2 model checkpoint
        model_config: Model configuration file path
        upload_to_gcp: Whether to upload results to GCS
        upload_to_local: Whether to keep local copies
        max_frames: Maximum frames to extract per video
    
    Returns:
        Summary message with GCS paths
    """
    # -- parse video URIs: chunks://, folder, comma-separated, or "default" --
    video_uris: Optional[List[str]] = None
    if sam2_video_uris and sam2_video_uris != "default":
        if sam2_video_uris.startswith("chunks://"):
            # Chunk spec — pass as single-element list; task resolves it
            video_uris = [sam2_video_uris]
        elif sam2_video_uris.rstrip("/").startswith("gs://") and "," not in sam2_video_uris:
            # Single GCS folder or single file — pass as-is
            video_uris = [sam2_video_uris]
        else:
            # Comma-separated individual URIs
            video_uris = [u.strip() for u in sam2_video_uris.split(",") if u.strip()]

    if video_uris is None:
        raise ValueError(
            "No video URIs provided. Pass --sam2_video_uris with a chunks:// URI, "
            "gs:// path, or comma-separated list of URIs."
        )
    
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
    
    # Download input videos from GCS to a local temp folder (skip for chunks:// URIs
    # which are resolved by the processing script itself via gcsfs at runtime).
    import gcsfs

    has_chunk_uris = any(u.startswith("chunks://") for u in video_uris)

    if has_chunk_uris:
        # chunks:// URIs are resolved by process_vide_sam2_hlxwf.py — pass through as-is
        logger.info("Detected chunks:// URI — skipping download; processing script will resolve.")
    else:
        video_cache_dir = Path("/tmp/sam2_video_cache")
        video_cache_dir.mkdir(parents=True, exist_ok=True)

        fs = gcsfs.GCSFileSystem()
        local_video_paths = []
        for uri in video_uris:
            video_filename = uri.split("/")[-1]
            local_video_path = video_cache_dir / video_filename

            if local_video_path.exists():
                logger.info(f"Already downloaded: {local_video_path}")
                local_video_paths.append(str(local_video_path))
                continue

            # Normalise URI to a bare GCS path for gcsfs
            gcs_path = uri
            if gcs_path.startswith("gs://"):
                gcs_path = gcs_path[len("gs://"):]
            elif gcs_path.startswith("https://storage.googleapis.com/"):
                gcs_path = gcs_path[len("https://storage.googleapis.com/"):]

            logger.info(f"Downloading gs://{gcs_path} -> {local_video_path}")
            try:
                fs.get(gcs_path, str(local_video_path))
                local_video_paths.append(str(local_video_path))
            except Exception as e:
                logger.error(f"Failed to download gs://{gcs_path}: {e}")
                raise

        logger.info(f"Downloaded {len(local_video_paths)} videos to {video_cache_dir}")
        
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
        sam2_video_uris=video_uri,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        upload_to_gcp=upload_to_gcp,
        upload_to_local=upload_to_local,
        max_frames=max_frames,
    )


@workflow
def sam2_segmentation_wf(
    run_id: str,
    sam2_video_uris: str = "default",
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
        run_id: Unique identifier for this run (used in output paths)
        sam2_video_uris: Video input specification. Accepts:
            - "default"           → 10 built-in sample videos
            - "chunks://..."      → chunk-based URI (resolved inside container)
            - "gs://bucket/folder/" → single GCS folder
            - "gs://bucket/v.mp4" → single GCS file
            - "gs://b/v1.mp4,gs://b/v2.mp4" → comma-separated URIs
        checkpoint_path: Path to SAM2.1 Large checkpoint
        model_config: SAM2 model config (sam2.1_hiera_l.yaml)
        upload_to_gcp: Upload results to GCS (default: True)
        upload_to_local: Keep local copies (default: False)
        max_frames: Max frames per video (default: 150)
    
    Returns:
        Summary of processed videos and GCS output locations
    """
    return run_sam2_segmentation(
        run_id=run_id,
        sam2_video_uris=sam2_video_uris,
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
            sam2_video_uris=",".join(args.video_uris) if args.video_uris else "default",
            upload_to_gcp=not args.no_upload,
            upload_to_local=args.no_upload,
            max_frames=args.max_frames,
        )
    
    print(result)
