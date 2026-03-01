"""
SAM 3 Video Processing Script for Road Segmentation
Uses SAM3's text-based prompting for concept-driven segmentation.

SAM3 key advantage over SAM2: text-based prompting instead of point-based,
enabling semantic understanding of what to segment (e.g. "road surface").
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

# sam3 is already installed via pip install -e .
from sam3.model_builder import build_sam3_video_predictor

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()

# Upload configuration
UPLOAD_TO_GCP = True
UPLOAD_TO_LOCAL = False

# GCS bucket base paths
GCP_OUTPUT_BUCKET_BASE = "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/sam3"
GCP_PREPROCESSED_BUCKET_BASE = "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/preprocessed_data_vp"

# Timestamp/run_id
TIMESTAMP = os.environ.get("SAM3_OUTPUT_TIMESTAMP", "10")

# Construct full GCS paths with run_id
GCP_OUTPUT_BUCKET = f"{GCP_OUTPUT_BUCKET_BASE}/{TIMESTAMP}"
GCP_PREPROCESSED_BUCKET = f"{GCP_PREPROCESSED_BUCKET_BASE}/{TIMESTAMP}"

# Local output directories
BASE_DATA_DIR = Path("/tmp/sam3_data")
OUTPUT_DIR = BASE_DATA_DIR / f"output_{TIMESTAMP}"
FRAMES_DIR = BASE_DATA_DIR / f"frames_{TIMESTAMP}"

# Segmentation parameters
MAX_FRAMES = 100

# Frame extraction parameters
FRAMES_PER_SECOND = 8

# VideoPainter preprocessing FPS
VP_PREPROCESS_FPS = os.environ.get("VP_PREPROCESS_FPS", "")

# SAM3 text prompt for road segmentation
# SAM3's key advantage: text-based prompting instead of point-based
SAM3_TEXT_PROMPT = os.environ.get(
    "SAM3_TEXT_PROMPT",
    "road surface",
)


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
    """Best-effort FPS detection."""
    fallback = float(FRAMES_PER_SECOND)

    try:
        for field in ("avg_frame_rate", "r_frame_rate"):
            proc = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", f"stream={field}",
                    "-of", "default=nw=1:nk=1",
                    str(video_path),
                ],
                capture_output=True, text=True, check=False,
            )
            if proc.returncode == 0:
                fps = _parse_ffmpeg_fraction(proc.stdout)
                if fps is not None:
                    return fps
    except (FileNotFoundError, Exception):
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
    global _gcs_fs
    if _gcs_fs is None:
        with _gcs_fs_lock:
            if _gcs_fs is None:
                _gcs_fs = gcsfs.GCSFileSystem(token="google_default")
    return _gcs_fs


def upload_directory_to_gcs(local_dir: str, gcs_path: str, max_workers: int = 8) -> None:
    fs = get_gcs_filesystem()
    base = Path(local_dir)
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]

    files_to_upload = []
    for path in base.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(base).as_posix()
        remote = f"{gcs_path.rstrip('/')}/{rel}"
        files_to_upload.append((path.as_posix(), remote))

    if not files_to_upload:
        return

    remote_parents = set(os.path.dirname(r) for _, r in files_to_upload)
    for rp in remote_parents:
        if rp:
            fs.makedirs(rp, exist_ok=True)

    def _upload_one(local_remote):
        local, remote = local_remote
        fs.put(local, remote)
        return os.path.basename(local)

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
    fs = get_gcs_filesystem()
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]
    remote_parent = os.path.dirname(gcs_path)
    if remote_parent:
        fs.makedirs(remote_parent, exist_ok=True)
    fs.put(local_path, gcs_path)


def _elapsed_s(start: float) -> float:
    return time.perf_counter() - start


def _format_seconds(seconds: float) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds:.3f}s"


def _fill_binary_mask_holes(mask_255: np.ndarray) -> np.ndarray:
    if mask_255 is None:
        return mask_255
    if mask_255.dtype != np.uint8:
        mask_255 = mask_255.astype(np.uint8)
    h, w = mask_255.shape[:2]
    if h == 0 or w == 0:
        return mask_255
    inv = cv2.bitwise_not(mask_255)
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(inv, ff_mask, seedPoint=(0, 0), newVal=0)
    holes = inv
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
    totals = defaultdict(float)
    for vt in per_video:
        for k, v in asdict(vt).items():
            if k == "video_name":
                continue
            totals[k] += float(v or 0.0)

    lines: List[str] = []
    lines.append(f"run_id: {run_id}")
    lines.append(f"model: SAM3")
    lines.append(f"text_prompt: {SAM3_TEXT_PROMPT}")
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
    """Setup optimal device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        compute_capability = torch.cuda.get_device_properties(0).major
        if compute_capability >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"Ampere+ GPU (compute {compute_capability}.x): TF32 enabled")
        else:
            print(f"Pre-Ampere GPU (compute {compute_capability}.x)")
    else:
        device = torch.device("cpu")
        print("WARNING: Using CPU - this will be very slow!")
    return device


def download_video(uri: str, output_path: Path) -> Path:
    """Download video from GCS URI."""
    if os.path.exists(uri) and not uri.startswith(("http://", "https://", "gs://")):
        print(f"Using existing local file: {uri}")
        return Path(uri)

    print(f"Downloading {uri}...")

    if uri.startswith("gs://") or "storage.googleapis.com" in uri:
        if "storage.googleapis.com" in uri:
            parts = uri.split("storage.googleapis.com/")[1].split("/", 1)
            gs_uri = f"gs://{parts[0]}/{parts[1]}"
        else:
            gs_uri = uri
        gcs_path = gs_uri[5:] if gs_uri.startswith("gs://") else gs_uri
        print(f"Downloading from GCS: {gs_uri}")
        try:
            fs = get_gcs_filesystem()
            fs.get(gcs_path, str(output_path))
        except Exception as e:
            raise RuntimeError(f"GCS download failed: {e}") from e
    else:
        urllib.request.urlretrieve(uri, output_path)

    print(f"Downloaded to {output_path}")
    return output_path


def extract_frames(video_path: Path, frames_dir: Path, max_frames: int | None = None) -> List[Path]:
    """Extract frames from video at target FPS."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    if max_frames is None:
        max_frames = int(MAX_FRAMES)

    target_fps = int(FRAMES_PER_SECOND)
    vframes_flag = f'-vframes {int(max_frames)}' if int(max_frames) > 0 else ''
    cmd = (
        f'ffmpeg -i "{video_path}" '
        f'-vf fps={target_fps} '
        f'{vframes_flag} '
        f'-q:v 2 -start_number 0 "{frames_dir}/%05d.jpg" -y'
    )
    print(f"Extracting frames at {target_fps} fps (max {max_frames}): {cmd}")
    os.system(cmd)

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
    """Post-process and save a single frame's mask + visualization."""
    frame = cv2.imread(str(frame_path))
    mask_uint8 = (raw_mask * 255).astype(np.uint8) if raw_mask.max() <= 1 else raw_mask.astype(np.uint8)

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


def segment_road_in_video_sam3(
    predictor,
    video_frames_dir: Path,
    output_dir: Path,
    video_name: str,
    text_prompt: str = "road surface",
    timed_frames: int = 100,
    output_fps: float | None = None,
    timings: VideoTiming | None = None,
) -> List[np.ndarray]:
    """
    Segment road in video using SAM3's text-based prompting.

    SAM3 uses text prompts instead of point prompts, enabling semantic
    understanding of what to segment. This is a major improvement over
    SAM2's point-based approach for road segmentation.

    Returns a list of post-processed binary masks (uint8, {0,255}) aligned
    with the sorted frame files.
    """
    print(f"\n{'='*60}")
    print(f"Processing video: {video_name}")
    print(f"Text prompt: '{text_prompt}'")
    print(f"{'='*60}\n")

    # Get video dimensions from first frame
    first_frame_path = sorted(video_frames_dir.glob("*.jpg"))[0]
    first_frame = cv2.imread(str(first_frame_path))
    height, width = first_frame.shape[:2]
    print(f"Video dimensions: {width}x{height}")

    # Reference point for filtering (bottom center)
    ref_point = (width // 2, int(height * 0.85))

    # Start SAM3 video session
    print("Starting SAM3 video session...")
    video_path = str(video_frames_dir)

    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    print(f"Session started: {session_id}")

    # Add text prompt on first frame
    # SAM3's key feature: text-based prompting
    print(f"Adding text prompt '{text_prompt}' on frame 0...")
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=text_prompt,
        )
    )

    # Propagate through entire video
    print("Propagating through video...")
    video_segments = {}
    frame_files = sorted(video_frames_dir.glob("*.jpg"))

    seg_start = time.perf_counter()
    first_n_recorded = False
    produced = 0

    for frame_output in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        frame_idx = frame_output.get("frame_index", produced)
        outputs = frame_output.get("outputs", None)
        if outputs is not None:
            # SAM3 returns {"out_binary_masks": (N_objs, H, W) bool array, ...}
            binary_masks = outputs.get("out_binary_masks", None)
            if binary_masks is not None and len(binary_masks) > 0:
                # Combine all detected object masks into a single road mask
                if isinstance(binary_masks, torch.Tensor):
                    combined_mask = (binary_masks.sum(dim=0) > 0).cpu().numpy()
                elif isinstance(binary_masks, np.ndarray):
                    combined_mask = (binary_masks.sum(axis=0) > 0)
                else:
                    combined_mask = binary_masks
                video_segments[frame_idx] = combined_mask
        produced += 1
        if not first_n_recorded and produced >= timed_frames:
            first_n_recorded = True
            if timings is not None:
                timings.segment_first_100_s = _elapsed_s(seg_start)

    if timings is not None:
        timings.segment_total_s = _elapsed_s(seg_start)

    # Close session
    predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )

    print(f"Segmented {len(video_segments)} frames")

    # Save segmentation results
    video_output_dir = output_dir / video_name
    if video_output_dir.exists():
        shutil.rmtree(video_output_dir, ignore_errors=True)
    video_output_dir.mkdir(parents=True, exist_ok=True)

    output_masks_dir = video_output_dir / "masks"
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    output_vis_dir = video_output_dir / "visualizations"
    output_vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to {output_dir}...")
    post_start = time.perf_counter()

    # Build work items for parallel post-processing
    work_items = []
    for frame_idx, frame_path in enumerate(frame_files):
        if frame_idx in video_segments:
            raw_mask = video_segments[frame_idx]
            work_items.append((frame_idx, frame_path, raw_mask))

    # Process frames in parallel
    num_workers = max(1, min(os.cpu_count() or 4, len(work_items), 8))
    processed_masks: dict[int, np.ndarray] = {}

    if not work_items:
        print("⚠ No segmented frames to post-process")
    else:
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
    """Preprocess a video into VideoPainter format and upload to GCS."""
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

        # Use in-memory masks if provided
        if in_memory_masks is not None and len(in_memory_masks) > 0:
            masks = [(m > 127).astype(np.uint8) for m in in_memory_masks]
            print(f"  Using {len(masks)} in-memory masks (skipped disk read)")
        else:
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

        preprocess_fps = int(FRAMES_PER_SECOND)
        if VP_PREPROCESS_FPS:
            try:
                preprocess_fps = int(float(VP_PREPROCESS_FPS))
            except Exception:
                preprocess_fps = int(FRAMES_PER_SECOND)
        if preprocess_fps <= 0:
            preprocess_fps = int(FRAMES_PER_SECOND)

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

        # Save masks
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

        # Upload to GCS
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
    """Process all videos from GCS using SAM3."""
    run_start = time.perf_counter()

    effective_max_frames = int(MAX_FRAMES)
    SEGMENT_TIMED_FRAMES = min(100, effective_max_frames) if effective_max_frames > 0 else 100

    # Setup
    device = setup_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SAM3 Video Segmentation (Text-Prompted)")
    print(f"Output base directory: {OUTPUT_DIR.absolute()}")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Text prompt: '{SAM3_TEXT_PROMPT}'")
    print(f"Max frames per video: {effective_max_frames}")
    print(f"Output video FPS fallback: {float(FRAMES_PER_SECOND)}")
    print(f"{'='*60}\n")

    # Load SAM3 model
    print(f"\nLoading SAM3 video predictor...")
    print("SAM3 uses text-based prompting for concept-driven segmentation")

    # Use FuseBucket-mounted checkpoint if available, otherwise download from HF
    mounted_ckpt = "/mnt/sam3-checkpoints/checkpoints/sam3.pt"
    if os.path.exists(mounted_ckpt):
        print(f"Using mounted checkpoint: {mounted_ckpt}\n")
        model_kwargs = {"checkpoint_path": mounted_ckpt}
    else:
        print("Mounted checkpoint not found, will download from HuggingFace\n")
        model_kwargs = {}

    model_load_start = time.perf_counter()
    predictor = build_sam3_video_predictor(**model_kwargs)
    model_load_s = _elapsed_s(model_load_start)

    print("✓ SAM3 model loaded successfully\n")

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
            # Download video
            if not video_path.exists():
                dl_start = time.perf_counter()
                actual_video_path = download_video(uri, video_path)
                vt.download_s = _elapsed_s(dl_start)
            else:
                print(f"Using cached video: {video_path}")
                actual_video_path = video_path

            source_fps = detect_video_fps(Path(actual_video_path))
            print(f"Detected source FPS: {source_fps:.3f}")

            # Extract frames
            desired = int(MAX_FRAMES)
            existing = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
            need_extract = (not frames_dir.exists()) or (len(existing) == 0)
            if not need_extract and desired > 0 and len(existing) < desired:
                need_extract = True

            if need_extract:
                ex_start = time.perf_counter()
                extract_frames(actual_video_path, frames_dir, max_frames=desired)
                _sync_frame_folder_to_max_frames(frames_dir, desired)
                vt.extract_frames_s = _elapsed_s(ex_start)
            else:
                _sync_frame_folder_to_max_frames(frames_dir, desired)
                print(f"Using cached frames: {frames_dir}")

            # Load frames into RAM for timing
            frame_paths_for_ram = sorted(frames_dir.glob("*.jpg"))[:SEGMENT_TIMED_FRAMES]
            ram_start = time.perf_counter()
            _ram_buf = []
            for p in frame_paths_for_ram:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    _ram_buf.append(img)
            vt.load_frames_ram_s = _elapsed_s(ram_start)
            del _ram_buf

            # Segment road using SAM3 text prompting
            processed_masks = segment_road_in_video_sam3(
                predictor,
                frames_dir,
                OUTPUT_DIR,
                video_name,
                text_prompt=SAM3_TEXT_PROMPT,
                timed_frames=SEGMENT_TIMED_FRAMES,
                output_fps=float(FRAMES_PER_SECOND),
                timings=vt,
            )

            # Upload results
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

            del processed_masks

            # Clean up
            print(f"\nCleaning up {video_name} files...")
            try:
                if video_path.exists() and not str(video_path).startswith('/mnt/'):
                    video_path.unlink()
                    print(f"✓ Deleted video: {video_path}")
            except Exception as e:
                print(f"⚠ Cleanup warning: {e}")

            try:
                if frames_dir.exists():
                    shutil.rmtree(frames_dir, ignore_errors=True)
                    print(f"✓ Deleted frames: {frames_dir}")
            except Exception as e:
                print(f"⚠ Cleanup warning: {e}")

            try:
                if video_output_dir.exists():
                    shutil.rmtree(video_output_dir, ignore_errors=True)
                    print(f"✓ Deleted output: {video_output_dir}")
            except Exception as e:
                print(f"⚠ Cleanup warning: {e}")

            vt_total = sum(
                v for k, v in asdict(vt).items() if k != "video_name" and isinstance(v, (int, float))
            )
            print(f"\n✓ Video {idx} complete ({_format_seconds(vt_total)})")

        except Exception as e:
            print(f"\n⚠ Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        per_video_timings.append(vt)

    # Write run report
    total_s = _elapsed_s(run_start)
    report_path = OUTPUT_DIR / f"{TIMESTAMP}.txt"
    _write_run_report_text(
        run_id=TIMESTAMP,
        report_path=report_path,
        model_load_s=model_load_s,
        per_video=per_video_timings,
        total_s=total_s,
        gcs_output_bucket=GCP_OUTPUT_BUCKET,
        gcs_preprocessed_bucket=GCP_PREPROCESSED_BUCKET,
        segment_timed_frames=SEGMENT_TIMED_FRAMES,
    )

    # Upload report
    if UPLOAD_TO_GCP:
        try:
            gcs_report = f"{GCP_OUTPUT_BUCKET}/{TIMESTAMP}.txt"
            upload_file_to_gcs(str(report_path), gcs_report)
            print(f"✓ Report uploaded: {gcs_report}")
        except Exception as e:
            print(f"⚠ Report upload failed: {e}")

    # Print final summary
    print(f"\n{'='*60}")
    print(f"SAM3 PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Model: SAM3 (Segment Anything with Concepts)")
    print(f"Text prompt: '{SAM3_TEXT_PROMPT}'")
    print(f"Videos processed: {len(per_video_timings)}/{len(video_uris)}")
    print(f"Model load time: {_format_seconds(model_load_s)}")
    print(f"Total time: {_format_seconds(total_s)}")
    print(f"Output: {GCP_OUTPUT_BUCKET}")
    print(f"Preprocessed: {GCP_PREPROCESSED_BUCKET}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Video Segmentation")
    parser.add_argument(
        "--video-uris", nargs="+",
        help="Video URIs/paths to process",
    )
    parser.add_argument(
        "--text-prompt", type=str, default="road surface",
        help="Text prompt for SAM3 segmentation (default: 'road surface')",
    )
    parser.add_argument(
        "--max-frames", type=int, default=100,
        help="Maximum frames per video",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip GCS upload",
    )

    args = parser.parse_args()

    if args.text_prompt:
        SAM3_TEXT_PROMPT = args.text_prompt

    if args.max_frames:
        MAX_FRAMES = args.max_frames

    if args.no_upload:
        UPLOAD_TO_GCP = False
        UPLOAD_TO_LOCAL = True

    if args.video_uris:
        process_all_videos(args.video_uris)
    else:
        print("No video URIs provided. Use --video-uris to specify videos.")
