"""
SAM 2.1 Video Processing Script for Road Segmentation
Uses SAM2VideoPredictorVOS for best quality output
"""
import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
import urllib.request
import tempfile
import shutil
from tqdm import tqdm
from datetime import datetime
import subprocess
import gcsfs
import time
import json
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# sam2 is already installed via pip install -e .
from sam2.build_sam import build_sam2_video_predictor

# Configuration
# Video URIs are always passed at runtime via build_and_run.sh / workflow.py.
# No hardcoded defaults — use --video-uris when running this script directly.

# Model configuration - Using the LARGEST model for best quality
# Use paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
SAM2_CHECKPOINT = str(SCRIPT_DIR / "checkpoints" / "sam2.1_hiera_large.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Upload configuration
UPLOAD_TO_GCP = True
UPLOAD_TO_LOCAL = False

# GCS bucket base paths (timestamp will be passed from workflow or generated at runtime)
GCP_OUTPUT_BUCKET_BASE = "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/sam2"
GCP_PREPROCESSED_BUCKET_BASE = "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/preprocessed_data_vp"

# Timestamp/run_id - set to "10" to match workflow, or use env var
TIMESTAMP = os.environ.get("SAM2_OUTPUT_TIMESTAMP", "10")

# Construct full GCS paths with run_id (no "output_" prefix to match workflow)
GCP_OUTPUT_BUCKET = f"{GCP_OUTPUT_BUCKET_BASE}/{TIMESTAMP}"
GCP_PREPROCESSED_BUCKET = f"{GCP_PREPROCESSED_BUCKET_BASE}/{TIMESTAMP}"

# Local output directories (use /tmp for workflow compatibility)
BASE_DATA_DIR = Path("/tmp/sam2_data")
OUTPUT_DIR = BASE_DATA_DIR / f"output_{TIMESTAMP}"
FRAMES_DIR = BASE_DATA_DIR / f"frames_{TIMESTAMP}"

# Segmentation parameters
MAX_FRAMES = 100

# Frame extraction parameters
FRAMES_PER_SECOND = 8  # Target content rate for VP (8 fps works best for VideoPainter)

# VideoPainter preprocessing FPS (controls the fps written into raw_videos/*.mp4 and meta.csv)
# Can be overridden per-run without editing code:
#   export VP_PREPROCESS_FPS=8
VP_PREPROCESS_FPS = os.environ.get("VP_PREPROCESS_FPS", "")


def _parse_ffmpeg_fraction(rate: str) -> float | None:
    rate = (rate or "").strip()
    if not rate or rate in {"0/0", "N/A"}:
        return None
    if "/" in rate:
        num_s, den_s = rate.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
        except ValueError:
            return None
        if den == 0:
            return None
        val = num / den
    else:
        try:
            val = float(rate)
        except ValueError:
            return None
    if not (val > 0 and np.isfinite(val)):
        return None
    return float(val)


def detect_video_fps(video_path: Path) -> float:
    """Best-effort FPS detection.

    Order:
      1) ffprobe avg_frame_rate (most accurate)
      2) ffprobe r_frame_rate
      3) OpenCV CAP_PROP_FPS
      4) Fallback to FRAMES_PER_SECOND
    """
    fallback = float(FRAMES_PER_SECOND)

    try:
        for field in ("avg_frame_rate", "r_frame_rate"):
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    f"stream={field}",
                    "-of",
                    "default=nw=1:nk=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                fps = _parse_ffmpeg_fraction(proc.stdout)
                if fps is not None:
                    return fps
    except FileNotFoundError:
        # ffprobe not installed in the container/environment
        pass
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if fps > 0 and np.isfinite(fps):
            return fps
    except Exception:
        pass

    return fallback


def _sync_frame_folder_to_max_frames(frames_dir: Path, max_frames: int) -> None:
    """Ensure frames_dir contains exactly max_frames frames when max_frames > 0.

    This makes changing MAX_FRAMES deterministic even when cached frames already exist.
    If max_frames <= 0, this is a no-op (meaning: keep all extracted frames).
    """
    if int(max_frames) <= 0:
        return

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if len(frame_paths) <= max_frames:
        return

    for p in frame_paths[max_frames:]:
        try:
            p.unlink()
        except Exception:
            pass


_gcs_fs = None
_gcs_fs_lock = threading.Lock()


def get_gcs_filesystem():
    """Get GCS filesystem with default authentication (cached singleton)."""
    global _gcs_fs
    if _gcs_fs is None:
        with _gcs_fs_lock:
            if _gcs_fs is None:
                _gcs_fs = gcsfs.GCSFileSystem(token="google_default")
    return _gcs_fs


def upload_directory_to_gcs(local_dir: str, gcs_path: str, max_workers: int = 8) -> None:
    """Recursively upload a local directory to GCS using gcsfs (parallel).
    
    Args:
        local_dir: Local directory path
        gcs_path: Full gs:// destination path
        max_workers: Number of parallel upload threads
    """
    fs = get_gcs_filesystem()
    base = Path(local_dir)
    
    # Strip gs:// prefix if present
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]
    
    # Collect all files to upload
    files_to_upload = []
    for path in base.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(base).as_posix()
        remote = f"{gcs_path.rstrip('/')}/{rel}"
        files_to_upload.append((path.as_posix(), remote))
    
    if not files_to_upload:
        return
    
    # Pre-create all remote parent directories
    remote_parents = set(os.path.dirname(r) for _, r in files_to_upload)
    for rp in remote_parents:
        if rp:
            fs.makedirs(rp, exist_ok=True)
    
    def _upload_one(local_remote):
        local, remote = local_remote
        fs.put(local, remote)
        return os.path.basename(local)
    
    # Upload in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_upload_one, item): item for item in files_to_upload}
        for future in as_completed(futures):
            try:
                name = future.result()
                print(f"  Uploaded: {name}")
            except Exception as e:
                _, remote = futures[future]
                print(f"  ⚠ Upload failed for {remote}: {e}")
                raise


def upload_file_to_gcs(local_path: str, gcs_path: str) -> None:
    """Upload a single file to GCS using gcsfs.

    Args:
        local_path: Local file path
        gcs_path: Full gs:// destination path (or bucket/path without scheme)
    """
    fs = get_gcs_filesystem()
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]
    remote_parent = os.path.dirname(gcs_path)
    if remote_parent:
        fs.makedirs(remote_parent, exist_ok=True)
    fs.put(local_path, gcs_path)


@contextmanager
def _timer() -> Iterator[float]:
    start = time.perf_counter()
    try:
        yield start
    finally:
        pass


def _elapsed_s(start: float) -> float:
    return time.perf_counter() - start


def _format_seconds(seconds: float) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds:.3f}s"


def _fill_binary_mask_holes(mask_255: np.ndarray) -> np.ndarray:
    """Fill internal holes in a binary mask.

    Args:
        mask_255: HxW uint8 mask with values {0, 255}.

    Returns:
        HxW uint8 mask with holes filled (values {0, 255}).
    """
    if mask_255 is None:
        return mask_255
    if mask_255.dtype != np.uint8:
        mask_255 = mask_255.astype(np.uint8)

    h, w = mask_255.shape[:2]
    if h == 0 or w == 0:
        return mask_255

    # Invert so background becomes 255 and foreground becomes 0.
    inv = cv2.bitwise_not(mask_255)

    # Flood-fill the *external* background (connected to border) to 0.
    # Remaining 255 pixels in `inv` correspond to holes inside the foreground.
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    seed = (0, 0)
    cv2.floodFill(inv, ff_mask, seedPoint=seed, newVal=0)

    holes = inv  # 255 where holes were
    filled = cv2.bitwise_or(mask_255, holes)
    return filled


@dataclass
class VideoTiming:
    video_name: str
    download_s: float = 0.0
    extract_frames_s: float = 0.0
    load_frames_ram_s: float = 0.0
    segment_first_100_s: float = 0.0
    segment_total_s: float = 0.0
    postprocess_write_s: float = 0.0
    upload_raw_s: float = 0.0
    vp_load_frames_masks_ram_s: float = 0.0
    vp_build_artifacts_s: float = 0.0
    vp_upload_s: float = 0.0


def _write_run_report_text(
    run_id: str,
    report_path: Path,
    model_load_s: float,
    per_video: List[VideoTiming],
    total_s: float,
    gcs_output_bucket: str,
    gcs_preprocessed_bucket: str,
    segment_timed_frames: int,
) -> None:
    # Aggregate totals
    totals = defaultdict(float)
    for vt in per_video:
        for k, v in asdict(vt).items():
            if k == "video_name":
                continue
            totals[k] += float(v or 0.0)

    lines: List[str] = []
    lines.append(f"run_id: {run_id}")
    lines.append(f"timestamp_utc: {datetime.utcnow().isoformat()}Z")
    lines.append(f"segment_timed_frames: {segment_timed_frames}")
    lines.append("")
    lines.append("buckets:")
    lines.append(f"  raw_outputs: {gcs_output_bucket}")
    lines.append(f"  preprocessed_outputs: {gcs_preprocessed_bucket}")
    lines.append("")
    lines.append("summary_times_seconds:")
    lines.append(f"  model_load_s: {model_load_s:.3f}")
    lines.append(f"  total_run_s: {total_s:.3f}")
    lines.append("")
    lines.append("totals_across_videos_seconds:")
    lines.append(f"  load_files_ram_s: {totals['load_frames_ram_s']:.3f}")
    lines.append(f"  segment_first_{segment_timed_frames}_s: {totals['segment_first_100_s']:.3f}")
    lines.append(f"  upload_raw_s: {totals['upload_raw_s']:.3f}")
    lines.append(f"  postprocess_segmented_output_s: {totals['postprocess_write_s']:.3f}")
    lines.append(f"  upload_postprocessed_output_s: {totals['vp_upload_s']:.3f}")
    lines.append("")
    lines.append("per_video_seconds:")
    for vt in per_video:
        lines.append(f"- video: {vt.video_name}")
        lines.append(f"  load_files_ram_s: {vt.load_frames_ram_s:.3f}")
        lines.append(f"  segment_first_{segment_timed_frames}_s: {vt.segment_first_100_s:.3f}")
        lines.append(f"  segment_total_s: {vt.segment_total_s:.3f}")
        lines.append(f"  upload_raw_s: {vt.upload_raw_s:.3f}")
        lines.append(f"  postprocess_segmented_output_s: {vt.postprocess_write_s:.3f}")
        lines.append(f"  upload_postprocessed_output_s: {vt.vp_upload_s:.3f}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def _write_run_report_json(report_path: Path, payload: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def setup_device():
    """Setup optimal device with quality-focused settings"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        
        # Enable bfloat16 only on Ampere+ GPUs (compute capability >= 8.0)
        # Tesla T4 is Turing (7.5) and doesn't support bfloat16 natively
        compute_capability = torch.cuda.get_device_properties(0).major
        if compute_capability >= 8:
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"Ampere+ GPU detected (compute {compute_capability}.x): bfloat16 + TF32 enabled")
        else:
            # Use float16 for older GPUs like T4
            torch.autocast("cuda", dtype=torch.float16).__enter__()
            print(f"Pre-Ampere GPU detected (compute {compute_capability}.x): float16 enabled")
    else:
        device = torch.device("cpu")
        print("WARNING: Using CPU - this will be very slow!")
    
    return device


def download_video(uri: str, output_path: Path) -> Path:
    """Download video from GCS URI using gsutil (supports authentication)"""
    
    # If URI is already a local file path, just return it (no download needed)
    if os.path.exists(uri) and not uri.startswith(("http://", "https://", "gs://")):
        print(f"Using existing local file: {uri}")
        return Path(uri)
    
    print(f"Downloading {uri}...")
    
    # Use gcsfs for GCS URIs (handles authentication via google_default token)
    if uri.startswith("gs://") or "storage.googleapis.com" in uri:
        # Convert https URL to gs:// URI if needed
        if "storage.googleapis.com" in uri:
            # Extract bucket and path from URL
            parts = uri.split("storage.googleapis.com/")[1].split("/", 1)
            gs_uri = f"gs://{parts[0]}/{parts[1]}"
        else:
            gs_uri = uri
        
        # Strip gs:// prefix for gcsfs
        gcs_path = gs_uri[5:] if gs_uri.startswith("gs://") else gs_uri
        
        print(f"Downloading from GCS: {gs_uri}")
        try:
            fs = get_gcs_filesystem()
            fs.get(gcs_path, str(output_path))
        except Exception as e:
            raise RuntimeError(f"Failed to download from GCS: {e}. Make sure you have proper GCS permissions.") from e
    else:
        # Use urllib for regular HTTP URLs
        urllib.request.urlretrieve(uri, output_path)
    
    print(f"Downloaded to {output_path}")
    return output_path


def extract_frames(video_path: Path, frames_dir: Path, max_frames: int | None = None) -> List[Path]:
    """Extract high-quality JPEG frames from video at FRAMES_PER_SECOND rate.

    Uses ffmpeg's fps filter to temporally subsample the source video so
    that each extracted frame represents exactly 1/FRAMES_PER_SECOND seconds
    of real time.  This ensures downstream consumers (SAM2 segmentation,
    VideoPainter) work with correctly-spaced frames.

    For example, a 20 s source at 30 fps yields 160 frames at 8 fps
    (capped to *max_frames* if set).

    Args:
        video_path: Input video path.
        frames_dir: Directory to write frames into.
        max_frames: If > 0, limit to first max_frames frames. If <= 0, extract all frames.
            If None, uses the global MAX_FRAMES.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    if max_frames is None:
        max_frames = int(MAX_FRAMES)
    
    target_fps = int(FRAMES_PER_SECOND)
    # Use ffmpeg fps filter to extract at the target content rate.
    vframes_flag = f'-vframes {int(max_frames)}' if int(max_frames) > 0 else ''
    cmd = (
        f'ffmpeg -i "{video_path}" '
        f'-vf fps={target_fps} '
        f'{vframes_flag} '
        f'-q:v 2 -start_number 0 "{frames_dir}/%05d.jpg" -y'
    )
    print(f"Extracting frames at {target_fps} fps (max {max_frames}): {cmd}")
    os.system(cmd)
    
    # Get frame paths
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    duration_s = len(frame_paths) / target_fps if target_fps > 0 else 0
    print(f"Extracted {len(frame_paths)} frames at {target_fps} fps = {duration_s:.1f}s of content")
    return frame_paths


def _postprocess_single_frame(
    frame_idx: int,
    frame_path: Path,
    raw_mask: np.ndarray,
    ref_point: Tuple[int, int],
    output_masks_dir: Path,
    output_vis_dir: Path,
) -> np.ndarray:
    """Post-process and save a single frame's mask + visualization.

    Returns the final binary mask (uint8, values {0, 255}) so callers can
    reuse it without re-reading from disk.
    """
    frame = cv2.imread(str(frame_path))
    mask_uint8 = (raw_mask * 255).astype(np.uint8)

    # Morphological opening to disconnect leaking regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    # Connected-component filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        opened_mask, connectivity=8
    )

    if num_labels > 1:
        label_at_point = labels[ref_point[1], ref_point[0]]
        if label_at_point > 0:
            filtered_mask = (labels == label_at_point).astype(np.uint8) * 255
        else:
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            filtered_mask = (labels == largest_component).astype(np.uint8) * 255
    else:
        filtered_mask = opened_mask

    # Closing + hole-fill
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    filtered_mask = cv2.morphologyEx(
        filtered_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2
    )
    filtered_mask = _fill_binary_mask_holes(filtered_mask)

    # Save binary mask
    cv2.imwrite(str(output_masks_dir / f"{frame_idx:05d}.png"), filtered_mask)

    # Save visualization overlay
    overlay = frame.copy()
    mask_bool = filtered_mask > 0
    overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5
    cv2.imwrite(str(output_vis_dir / f"{frame_idx:05d}.jpg"), overlay)

    return filtered_mask


def segment_road_in_video(
    predictor,
    video_frames_dir: Path,
    output_dir: Path,
    video_name: str,
    timed_frames: int = 100,
    output_fps: float | None = None,
    timings: VideoTiming | None = None,
) -> List[np.ndarray]:
    """
    Segment road in video using SAM2.

    Returns a list of post-processed binary masks (uint8, {0,255}) aligned
    with the sorted frame files, so downstream VP preprocessing can reuse
    them without re-reading from disk.
    
    For road segmentation, we'll use a simple strategy:
    1. Initialize on first frame with points in lower portion (road area)
    2. Propagate through video
    """
    print(f"\n{'='*60}")
    print(f"Processing video: {video_name}")
    print(f"{'='*60}\n")
    
    # Reset CUDA graphs state before each video to prevent tensor overwriting
    torch.compiler.cudagraph_mark_step_begin()
    
    # Initialize inference state
    print("Initializing inference state...")
    inference_state = predictor.init_state(video_path=str(video_frames_dir))
    
    # Get video dimensions from first frame
    first_frame = cv2.imread(str(sorted(video_frames_dir.glob("*.jpg"))[0]))
    height, width = first_frame.shape[:2]
    print(f"Video dimensions: {width}x{height}")
    
    # Define road region points (bottom half only)
    # For autonomous driving videos, road is in the bottom half of frame
    ann_frame_idx = 0
    ann_obj_id = 1  # Road object
    
    # Add multiple points in road region for robust initialization
    # Points distributed across bottom half of frame only
    points = np.array([
        [width * 0.25, height * 0.80],  # Left road
        [width * 0.50, height * 0.85],  # Center-bottom road  
        [width * 0.75, height * 0.80],  # Right road
        [width * 0.50, height * 0.65],  # Upper road boundary
        [width * 0.35, height * 0.75],  # Additional left coverage
        [width * 0.65, height * 0.75],  # Additional right coverage
    ], dtype=np.float32)
    
    labels = np.array([1, 1, 1, 1, 1, 1], np.int32)  # All positive clicks
    
    print(f"Adding {len(points)} road region points (bottom half) on frame {ann_frame_idx}...")
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    
    # Propagate through entire video
    print("Propagating through video...")
    video_segments = {}

    seg_start = time.perf_counter()
    first_n_recorded = False
    produced = 0
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        produced += 1
        if not first_n_recorded and produced >= timed_frames:
            first_n_recorded = True
            if timings is not None:
                timings.segment_first_100_s = _elapsed_s(seg_start)

    if timings is not None:
        timings.segment_total_s = _elapsed_s(seg_start)
    
    # Save segmentation results - create video-specific folder
    # Clear any stale outputs so changing MAX_FRAMES always changes the MP4 duration.
    video_output_dir = output_dir / video_name
    if video_output_dir.exists():
        shutil.rmtree(video_output_dir, ignore_errors=True)
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_masks_dir = video_output_dir / "masks"
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    output_vis_dir = video_output_dir / "visualizations"
    output_vis_dir.mkdir(parents=True, exist_ok=True)
    
    frame_files = sorted(video_frames_dir.glob("*.jpg"))
    
    # Reference point for filtering (bottom center)
    ref_point = (width // 2, int(height * 0.85))
    
    print(f"Saving results to {output_dir}...")
    post_start = time.perf_counter()

    # Build work items for parallel post-processing
    work_items = []
    for frame_idx, frame_path in enumerate(frame_files):
        if frame_idx in video_segments:
            raw_mask = video_segments[frame_idx][ann_obj_id][0]
            work_items.append((frame_idx, frame_path, raw_mask))

    # Process frames in parallel (OpenCV releases the GIL)
    num_workers = min(os.cpu_count() or 4, len(work_items), 8)
    processed_masks: dict[int, np.ndarray] = {}

    def _process_frame(item):
        fidx, fpath, raw = item
        mask = _postprocess_single_frame(
            fidx, fpath, raw, ref_point, output_masks_dir, output_vis_dir,
        )
        return fidx, mask

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_process_frame, w) for w in work_items]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Saving masks"):
            fidx, mask = future.result()
            processed_masks[fidx] = mask

    if timings is not None:
        timings.postprocess_write_s = _elapsed_s(post_start)
    
    print(f"✓ Saved {len(video_segments)} frames")
    print(f"  - Masks: {output_masks_dir}")
    print(f"  - Visualizations: {output_vis_dir}")
    
    # Create output video from visualizations
    output_video_path = video_output_dir / f"{video_name}_segmented.mp4"
    print(f"Creating output video: {output_video_path.name}...")
    fps = float(output_fps) if output_fps is not None else float(FRAMES_PER_SECOND)
    if not (fps > 0 and np.isfinite(fps)):
        fps = float(FRAMES_PER_SECOND)
    cmd = (
        f'ffmpeg -framerate {fps:.6f} -i "{output_vis_dir}/%05d.jpg" '
        f'-c:v libx264 -pix_fmt yuv420p -crf 18 -r {fps:.6f} "{output_video_path}" -y'
    )
    result = os.system(cmd + " 2>&1 | grep -v 'deprecated' || true")
    if result == 0:
        print(f"✓ Video saved: {output_video_path}")
    else:
        print(f"⚠ Warning: Video creation may have failed")

    # Return ordered list of processed masks for VP preprocessing (avoids re-reading)
    ordered_masks = [processed_masks[i] for i in sorted(processed_masks.keys())]
    return ordered_masks


def preprocess_and_upload_video(
    video_name: str,
    frames_dir: Path,
    masks_dir: Path,
    timestamp: str,
    timings: VideoTiming | None = None,
    in_memory_masks: List[np.ndarray] | None = None,
) -> None:
    """Preprocess a video into VideoPainter format and upload to GCS.

    Frames in *frames_dir* are assumed to already be at the target content
    rate (FRAMES_PER_SECOND) because extract_frames() uses the ffmpeg fps
    filter.  No additional temporal subsampling is performed here.

    Args:
        in_memory_masks: If provided, skip re-reading masks from disk.
            Each element is a uint8 mask with values {0, 255}.
    """
    print(f"\n{'='*60}")
    print(f"Preprocessing {video_name} for VideoPainter...")
    print(f"{'='*60}\n")
    
    with tempfile.TemporaryDirectory(prefix=f"vp_pre_{video_name}_") as tmp:
        out_dir = Path(tmp) / "out"
        raw_video_root = out_dir / "raw_videos"
        mask_root = out_dir / "mask_root"
        meta_csv = out_dir / "meta.csv"
        
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_video_root.mkdir(parents=True, exist_ok=True)
        mask_root.mkdir(parents=True, exist_ok=True)
        
        # Load frames
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            print(f"⚠ No frames found for {video_name}")
            return

        load_start = time.perf_counter()
        frames = []
        for p in frame_paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not frames:
            print(f"⚠ Failed to load frames for {video_name}")
            return
        
        # Use in-memory masks if provided; otherwise read from disk
        if in_memory_masks is not None and len(in_memory_masks) > 0:
            masks = [(m > 127).astype(np.uint8) for m in in_memory_masks]
            print(f"  Using {len(masks)} in-memory masks (skipped disk read)")
        else:
            # Fallback: Load and combine masks from disk
            mask_paths = sorted(masks_dir.glob("*.png"))
            if not mask_paths:
                print(f"⚠ No masks found for {video_name}")
                return
            
            masks = []
            for p in mask_paths:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    binary = (img > 127).astype(np.uint8)
                    masks.append(binary)

        if timings is not None:
            timings.vp_load_frames_masks_ram_s = _elapsed_s(load_start)
        
        if len(masks) != len(frames):
            print(f"⚠ Frame/mask count mismatch: {len(frames)} frames, {len(masks)} masks")
            return
        
        build_start = time.perf_counter()

        # Resolve target content FPS for VideoPainter output.
        preprocess_fps = int(FRAMES_PER_SECOND)      # default = 8 fps
        if VP_PREPROCESS_FPS:
            try:
                preprocess_fps = int(float(VP_PREPROCESS_FPS))
            except Exception:
                preprocess_fps = int(FRAMES_PER_SECOND)
        if preprocess_fps <= 0:
            preprocess_fps = int(FRAMES_PER_SECOND)

        # Frames were already extracted at FRAMES_PER_SECOND rate by
        # extract_frames() (ffmpeg fps filter), so no temporal
        # subsampling is needed here — each frame already represents
        # 1/preprocess_fps seconds of real time.

        masks_array = np.stack(masks, axis=0)

        # Create video
        prefix = video_name[:-3] if len(video_name) > 3 else video_name
        video_filename = f"{video_name}.0.mp4"
        video_path = raw_video_root / prefix / video_filename
        video_path.parent.mkdir(parents=True, exist_ok=True)
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, preprocess_fps, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        
        # Save masks (subsampled to match frames)
        mask_out_dir = mask_root / video_name
        mask_out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(mask_out_dir / "all_masks.npz"), masks_array)
        
        # Create meta.csv
        import csv
        with open(meta_csv, 'w', newline='') as f:
            writer_csv = csv.DictWriter(f, fieldnames=["path", "mask_id", "start_frame", "end_frame", "fps", "caption"])
            writer_csv.writeheader()
            writer_csv.writerow({
                "path": video_filename,
                "mask_id": 1,
                "start_frame": 0,
                "end_frame": len(frames),
                "fps": preprocess_fps,
                "caption": (
                    "Front camera video of an autonomous driving car on the road."
                )
            })
        print(
            f"  VP data: {len(frames)} frames, meta.csv fps={preprocess_fps}, "
            f"duration={len(frames)/preprocess_fps:.1f}s"
        )

        if timings is not None:
            timings.vp_build_artifacts_s = _elapsed_s(build_start)
        
        print(f"✓ Preprocessed video: {video_path}")
        print(f"✓ Preprocessed masks: {mask_out_dir / 'all_masks.npz'}")
        print(f"✓ Created meta.csv")
        
        # Upload to GCS using gcsfs
        gcs_destination = f"{GCP_PREPROCESSED_BUCKET}/{video_name}"
        print(f"\nUploading preprocessed data to: {gcs_destination}")
        try:
            up_start = time.perf_counter()
            upload_directory_to_gcs(str(out_dir), gcs_destination)
            if timings is not None:
                timings.vp_upload_s = _elapsed_s(up_start)
            print(f"✓ Preprocessed data uploaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Upload failed: {e}")


def process_all_videos(video_uris: List[str]):
    """Process all videos from GCS"""
    run_start = time.perf_counter()

    effective_max_frames = int(MAX_FRAMES)

    # How many frames we time segmentation for (does not change MAX_FRAMES)
    SEGMENT_TIMED_FRAMES = min(100, effective_max_frames) if effective_max_frames > 0 else 100

    # Setup
    device = setup_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Output base directory: {OUTPUT_DIR.absolute()}")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Max frames per video: {effective_max_frames} ({'no limit' if effective_max_frames <= 0 else 'limited'})")
    print(f"Output video FPS fallback: {float(FRAMES_PER_SECOND)}")
    print(f"{'='*60}\n")
    
    # Load SAM2 model - Using VOS predictor for best quality
    print(f"\nLoading SAM 2.1 (Large) model from {SAM2_CHECKPOINT}...")
    print("Using SAM2VideoPredictorVOS for maximum quality")
    print("First compilation will be VERY SLOW (5-15 min) but ensures best results\n")
    
    model_load_start = time.perf_counter()
    predictor = build_sam2_video_predictor(
        MODEL_CFG, 
        SAM2_CHECKPOINT,
        device=device,
        apply_postprocessing=True,  # Better quality
        vos_optimized=True  # Use VOS-optimized predictor
    )

    model_load_s = _elapsed_s(model_load_start)
    
    print("✓ Model loaded successfully\n")
    
    # Process each video
    per_video_timings: List[VideoTiming] = []
    for idx, uri in enumerate(video_uris, 1):
        print(f"\n{'#'*60}")
        print(f"Video {idx}/{len(video_uris)}")
        print(f"{'#'*60}")
        
        video_name = Path(uri).stem
        video_path = FRAMES_DIR / f"{video_name}.mp4"
        frames_dir = FRAMES_DIR / video_name
        video_output_dir = OUTPUT_DIR / video_name

        vt = VideoTiming(video_name=video_name)
        
        try:
            # Download video (or use existing local file/symlink)
            if not video_path.exists():
                dl_start = time.perf_counter()
                actual_video_path = download_video(uri, video_path)
                vt.download_s = _elapsed_s(dl_start)
            else:
                print(f"Using cached video: {video_path}")
                actual_video_path = video_path

            # Detect source FPS for correct real-time duration in outputs
            source_fps = detect_video_fps(Path(actual_video_path))
            print(f"Detected source FPS: {source_fps:.3f}")
            
            # Extract/sync frames
            desired = int(MAX_FRAMES)
            existing = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
            need_extract = (not frames_dir.exists()) or (len(existing) == 0)
            if not need_extract and desired > 0 and len(existing) < desired:
                # Cached frames are fewer than requested; re-run extraction to extend.
                need_extract = True

            if need_extract:
                ex_start = time.perf_counter()
                extract_frames(actual_video_path, frames_dir, max_frames=desired)
                _sync_frame_folder_to_max_frames(frames_dir, desired)
                vt.extract_frames_s = _elapsed_s(ex_start)
            else:
                # If user reduced max_frames, trim cached frames so downstream sees the new limit.
                _sync_frame_folder_to_max_frames(frames_dir, desired)
                print(f"Using cached frames: {frames_dir}")

            # Time "load all files into RAM" as reading the first N extracted frames into memory.
            # This does not affect the SAM2 predictor (which reads from directory), but matches the user's metric.
            frame_paths_for_ram = sorted(frames_dir.glob("*.jpg"))[:SEGMENT_TIMED_FRAMES]
            ram_start = time.perf_counter()
            _ram_buf = []
            for p in frame_paths_for_ram:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    _ram_buf.append(img)
            vt.load_frames_ram_s = _elapsed_s(ram_start)
            del _ram_buf
            
            # Segment road (returns processed masks for reuse)
            processed_masks = segment_road_in_video(
                predictor,
                frames_dir,
                OUTPUT_DIR,
                video_name,
                timed_frames=SEGMENT_TIMED_FRAMES,
                output_fps=source_fps,
                timings=vt,
            )
            
            # Upload this video's results to GCP if enabled
            if UPLOAD_TO_GCP:
                print(f"\n{'='*60}")
                print(f"Uploading {video_name} results to GCS...")
                print(f"{'='*60}\n")
                
                gcs_destination = f"{GCP_OUTPUT_BUCKET}/{video_name}"
                print(f"Uploading to: {gcs_destination}")
                try:
                    up_start = time.perf_counter()
                    upload_directory_to_gcs(str(video_output_dir), gcs_destination)
                    vt.upload_raw_s = _elapsed_s(up_start)
                    print(f"✓ Upload successful: {gcs_destination}")
                    
                    # Preprocess and upload for VideoPainter (reuse in-memory masks)
                    preprocess_and_upload_video(
                        video_name=video_name,
                        frames_dir=frames_dir,
                        masks_dir=video_output_dir / "masks",
                        timestamp=TIMESTAMP,
                        timings=vt,
                        in_memory_masks=processed_masks,
                    )
                except Exception as e:
                    print(f"⚠ Warning: Upload failed: {e}")
                    print(f"   Skipping preprocessing upload for this video")
            
            # Free masks from memory now that uploads are done
            del processed_masks
            
            # Clean up video-specific files after processing/upload
            print(f"\n{'='*60}")
            print(f"Cleaning up {video_name} files...")
            print(f"{'='*60}\n")
            
            # Delete downloaded video (only if it's not a symlink to mounted storage)
            # Never delete anything from mounted read-only directories (/mnt/*)
            try:
                if os.path.islink(actual_video_path):
                    # Symlink pointing to mounted storage - safe to delete the symlink itself
                    symlink_target = os.readlink(actual_video_path)
                    if not symlink_target.startswith('/mnt/'):
                        # Symlink not pointing to mounted storage, safe to delete
                        actual_video_path.unlink() if isinstance(actual_video_path, Path) else Path(actual_video_path).unlink()
                        print(f"✓ Deleted symlink: {actual_video_path}")
                    else:
                        print(f"✓ Preserved symlink to mounted storage: {actual_video_path} -> {symlink_target}")
                elif video_path.exists() and not str(video_path).startswith('/mnt/'):
                    # Regular file, not in mounted directory
                    video_path.unlink()
                    print(f"✓ Deleted video: {video_path}")
                else:
                    print(f"✓ No video file to clean up (using mounted storage)")
            except Exception as e:
                print(f"⚠ Warning: Could not delete video file: {e}")
            
            # Delete extracted frames (always safe - in /tmp)
            try:
                if frames_dir.exists():
                    shutil.rmtree(frames_dir)
                    print(f"✓ Deleted frames: {frames_dir}")
            except Exception as e:
                print(f"⚠ Warning: Could not delete frames: {e}")
            
            # Delete local output if not keeping local copy (always safe - in /tmp)
            try:
                if not UPLOAD_TO_LOCAL and video_output_dir.exists():
                    shutil.rmtree(video_output_dir)
                    print(f"✓ Deleted output: {video_output_dir}")
            except Exception as e:
                print(f"⚠ Warning: Could not delete output: {e}")
            
            print(f"✓ Cleanup complete for {video_name}\n")

            per_video_timings.append(vt)
            
        except Exception as e:
            print(f"ERROR processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up on error too
            try:
                if video_path.exists():
                    video_path.unlink()
                if frames_dir.exists():
                    shutil.rmtree(frames_dir)
                if not UPLOAD_TO_LOCAL and video_output_dir.exists():
                    shutil.rmtree(video_output_dir)
            except:
                pass
            
            continue

    # Write a timing report into the run folder and upload it
    report_local_dir = BASE_DATA_DIR / f"report_{TIMESTAMP}"
    report_txt = report_local_dir / f"{TIMESTAMP}.txt"
    report_json = report_local_dir / f"{TIMESTAMP}.json"

    total_run_s = _elapsed_s(run_start)

    try:
        _write_run_report_text(
            run_id=TIMESTAMP,
            report_path=report_txt,
            model_load_s=model_load_s,
            per_video=per_video_timings,
            total_s=total_run_s,
            gcs_output_bucket=GCP_OUTPUT_BUCKET,
            gcs_preprocessed_bucket=GCP_PREPROCESSED_BUCKET,
            segment_timed_frames=SEGMENT_TIMED_FRAMES,
        )

        _write_run_report_json(
            report_json,
            {
                "run_id": TIMESTAMP,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "segment_timed_frames": SEGMENT_TIMED_FRAMES,
                "buckets": {
                    "raw_outputs": GCP_OUTPUT_BUCKET,
                    "preprocessed_outputs": GCP_PREPROCESSED_BUCKET,
                },
                "model_load_s": model_load_s,
                "total_run_s": total_run_s,
                "per_video": [asdict(vt) for vt in per_video_timings],
            },
        )

        if UPLOAD_TO_GCP:
            # Put the report inside the run folder, named as the run id
            report_gcs_txt = f"{GCP_OUTPUT_BUCKET}/{TIMESTAMP}.txt"
            report_gcs_json = f"{GCP_OUTPUT_BUCKET}/{TIMESTAMP}.json"
            print(f"\nUploading timing report to: {report_gcs_txt}")
            upload_file_to_gcs(report_txt.as_posix(), report_gcs_txt)
            upload_file_to_gcs(report_json.as_posix(), report_gcs_json)
            print("✓ Timing report uploaded")
        else:
            print(f"\nTiming report written locally at: {report_txt}")
    except Exception as e:
        print(f"⚠ Warning: Failed to write/upload timing report: {e}")
    
    # Final cleanup
    print(f"\n{'='*60}")
    print("All videos processed successfully!")
    print(f"{'='*60}")
    
    if UPLOAD_TO_GCP:
        print(f"✓ All results uploaded to: {GCP_OUTPUT_BUCKET}")
        print(f"✓ Preprocessed data uploaded to: {GCP_PREPROCESSED_BUCKET}")
    
    if UPLOAD_TO_LOCAL:
        print(f"✓ Local output preserved at: {OUTPUT_DIR.absolute()}")
    else:
        # Remove empty base directories
        if OUTPUT_DIR.exists() and not any(OUTPUT_DIR.iterdir()):
            OUTPUT_DIR.rmdir()
        if FRAMES_DIR.exists() and not any(FRAMES_DIR.iterdir()):
            FRAMES_DIR.rmdir()
        print(f"✓ All local files cleaned up")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM2 Video Segmentation")
    parser.add_argument(
        "--video-uris", nargs="+", required=True,
        help="One or more gs:// or local video paths to process",
    )
    args = parser.parse_args()
    process_all_videos(args.video_uris)
