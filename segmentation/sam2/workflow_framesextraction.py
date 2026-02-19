"""Frame extraction workflow for HLX.

Extracts specific frames (1, 100, 200, 300, 400, 500) from driving videos
across multiple GCS chunks and uploads them as PNGs for FluxFill training.

Processes chunk-by-chunk: download videos → extract frames → upload frames → cleanup.
This avoids mounting and keeps disk usage bounded.

Output naming: {video_idx:06d}_f{frame_num:04d}_{video_hash_12chars}.png
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import gcsfs

from hlx.wf import DedicatedNode, Node, task, workflow

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CONTAINER_IMAGE_DEFAULT = "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2"
CONTAINER_IMAGE = os.environ.get("SAM2_CONTAINER_IMAGE", f"{CONTAINER_IMAGE_DEFAULT}:latest")

BUCKET = "mbadas-sandbox-research-9bb9c7f"

# Input: GCS prefix for raw driving videos
INPUT_GCS_PREFIX = "workspace/user/hbaskar/Input/data_physical_ai/camera_front_tele_30fov"

# Output: GCS destination for extracted frames
OUTPUT_GCS_PREFIX = (
    "workspace/user/hbaskar/Video_inpainting/videopainter/training/data"
)

# Frames to extract (1-based frame numbers)
DEFAULT_FRAME_NUMBERS = [1, 100, 200, 300, 400, 500]

# Workers — CPU_32 node gives 32 cores; leave a few for OS overhead
DEFAULT_NUM_WORKERS = 28


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _video_hash(path: str) -> str:
    """Deterministic 12-char hash from the video file path."""
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]


def _extract_frame_ffmpeg(
    video_path: str, *, frame_number: int, out_png_path: str
) -> None:
    """Extract a single 1-based frame from a video to a PNG using ffmpeg."""
    if frame_number <= 0:
        raise ValueError(f"frame_number must be >= 1, got {frame_number}")

    frame0 = frame_number - 1
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", video_path,
        "-vf", f"select=eq(n\\,{frame0})",
        "-vsync", "vfr",
        "-frames:v", "1",
        out_png_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not available in this environment") from exc
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed for {video_path} frame {frame_number}: {err}")


def _process_single_video(args: tuple) -> dict:
    """Worker function: extract all requested frames from one video.

    Args is a tuple of:
        (video_path, video_idx, frame_numbers, out_dir, gcs_video_path)

    Returns a dict with status info.
    """
    video_path, video_idx, frame_numbers, out_dir, gcs_video_path = args
    vhash = _video_hash(gcs_video_path)
    results = {"video_path": video_path, "video_idx": video_idx, "frames": [], "errors": []}

    for fnum in frame_numbers:
        fname = f"{video_idx:06d}_f{fnum:04d}_{vhash}.png"
        out_path = os.path.join(out_dir, fname)
        try:
            _extract_frame_ffmpeg(video_path, frame_number=fnum, out_png_path=out_path)
            results["frames"].append(out_path)
        except Exception as exc:
            results["errors"].append(f"frame {fnum}: {exc}")

    return results


def _upload_directory_to_gcs(fs: gcsfs.GCSFileSystem, local_dir: str, gcs_prefix: str) -> int:
    """Recursively upload a local directory to GCS. Returns file count."""
    base = Path(local_dir)
    count = 0
    for path in sorted(base.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(base).as_posix()
        remote = f"{gcs_prefix.rstrip('/')}/{rel}"
        remote_parent = os.path.dirname(remote)
        if remote_parent:
            fs.makedirs(remote_parent, exist_ok=True)
        fs.put(path.as_posix(), remote)
        count += 1
    return count


def _download_chunk_videos(
    fs: gcsfs.GCSFileSystem,
    chunk_idx: int,
    local_dir: str,
) -> list[tuple[str, str]]:
    """Download all .mp4 files for one chunk. Returns [(local_path, gcs_path), ...]."""
    gcs_chunk = f"{BUCKET}/{INPUT_GCS_PREFIX}/chunk_{chunk_idx:04d}"
    try:
        all_files = fs.ls(gcs_chunk, detail=False)
    except FileNotFoundError:
        return []

    mp4s = sorted(f for f in all_files if f.endswith(".mp4"))
    if not mp4s:
        return []

    downloaded: list[tuple[str, str]] = []
    for remote_path in mp4s:
        fname = os.path.basename(remote_path)
        local_path = os.path.join(local_dir, fname)
        fs.get(remote_path, local_path)
        downloaded.append((local_path, remote_path))

    return downloaded


# ──────────────────────────────────────────────────────────────────────────────
# HLX TASK
# ──────────────────────────────────────────────────────────────────────────────
@task(
    compute=DedicatedNode(
        node=Node.CPU_32,
        ephemeral_storage="max",
    ),
    container_image=CONTAINER_IMAGE,
    environment={"PYTHONUNBUFFERED": "1"},
)
def extract_frames_task(
    output_folder_name: str = "trainingdata_chunk00_25",
    chunk_start: int = 0,
    chunk_end: int = 25,
    frame_numbers: str = "1,100,200,300,400,500",
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict:
    """Extract specific frames from driving videos across GCS chunks.

    Downloads one chunk at a time, extracts frames, uploads, then cleans up
    before moving to the next chunk. This keeps local disk usage bounded.

    Args:
        output_folder_name: Name of the output folder under the training data prefix.
        chunk_start: First chunk index (inclusive).
        chunk_end: Last chunk index (inclusive).
        frame_numbers: Comma-separated 1-based frame numbers to extract.
        num_workers: Number of parallel worker processes.

    Returns:
        Summary dict with counts and GCS output path.
    """
    frames = sorted(set(int(f.strip()) for f in frame_numbers.split(",")))
    logger.info("=" * 70)
    logger.info("FRAME EXTRACTION TASK")
    logger.info("=" * 70)
    logger.info(f"  Chunks:        {chunk_start:04d} – {chunk_end:04d}")
    logger.info(f"  Frame numbers: {frames}")
    logger.info(f"  Workers:       {num_workers}")
    logger.info(f"  Output folder: {output_folder_name}")

    fs = gcsfs.GCSFileSystem(token="google_default")
    gcs_dest = f"{BUCKET}/{OUTPUT_GCS_PREFIX}/{output_folder_name}/images"

    global_video_idx = 0
    total_videos = 0
    total_frames_ok = 0
    total_errors = 0
    total_uploaded = 0
    t0 = datetime.now()

    total_chunks = chunk_end - chunk_start + 1

    for chunk_idx in range(chunk_start, chunk_end + 1):
        chunk_num = chunk_idx - chunk_start + 1
        chunk_label = f"chunk_{chunk_idx:04d}"
        logger.info(f"\n{'─' * 60}")
        logger.info(f"[{chunk_num}/{total_chunks}] Processing {chunk_label} …")

        # ── 1. Download videos for this chunk ────────────────────────────
        chunk_tmp = tempfile.mkdtemp(prefix=f"chunk_{chunk_idx:04d}_")
        videos_dir = os.path.join(chunk_tmp, "videos")
        images_dir = os.path.join(chunk_tmp, "images")
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        logger.info(f"  [{chunk_num}/{total_chunks}] Downloading {chunk_label} from GCS …")
        t_dl = datetime.now()
        downloaded = _download_chunk_videos(fs, chunk_idx, videos_dir)
        dl_time = (datetime.now() - t_dl).total_seconds()

        if not downloaded:
            logger.warning(f"  [{chunk_num}/{total_chunks}] {chunk_label}: no .mp4 files found — skipping")
            shutil.rmtree(chunk_tmp, ignore_errors=True)
            continue

        logger.info(
            f"  [{chunk_num}/{total_chunks}] ✓ DOWNLOADED {chunk_label}: "
            f"{len(downloaded)} videos in {dl_time:.1f}s"
        )

        # ── 2. Extract frames in parallel ────────────────────────────────
        logger.info(f"  [{chunk_num}/{total_chunks}] Extracting frames from {chunk_label} …")
        work_items = [
            (local_path, global_video_idx + vi, frames, images_dir, gcs_path)
            for vi, (local_path, gcs_path) in enumerate(downloaded)
        ]

        chunk_frames_ok = 0
        chunk_errors = 0
        t_ext = datetime.now()

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_process_single_video, item): item for item in work_items}
            for future in as_completed(futures):
                res = future.result()
                chunk_frames_ok += len(res["frames"])
                chunk_errors += len(res["errors"])
                if res["errors"]:
                    for e in res["errors"]:
                        logger.warning(f"  Video {res['video_idx']:06d}: {e}")

        ext_time = (datetime.now() - t_ext).total_seconds()
        logger.info(
            f"  [{chunk_num}/{total_chunks}] ✓ EXTRACTED {chunk_label}: "
            f"{chunk_frames_ok} frames in {ext_time:.1f}s ({chunk_errors} errors)"
        )

        # ── 3. Upload extracted frames ───────────────────────────────────
        logger.info(f"  [{chunk_num}/{total_chunks}] Uploading {chunk_label} frames to GCS …")
        t_up = datetime.now()
        uploaded = _upload_directory_to_gcs(fs, images_dir, gcs_dest)
        up_time = (datetime.now() - t_up).total_seconds()
        logger.info(
            f"  [{chunk_num}/{total_chunks}] ✓ UPLOADED {chunk_label}: "
            f"{uploaded} files in {up_time:.1f}s"
        )

        # ── 4. Cleanup local temp ────────────────────────────────────────
        shutil.rmtree(chunk_tmp, ignore_errors=True)
        logger.info(f"  [{chunk_num}/{total_chunks}] ✓ CLEANUP {chunk_label}: temp dir removed")

        global_video_idx += len(downloaded)
        total_videos += len(downloaded)
        total_frames_ok += chunk_frames_ok
        total_errors += chunk_errors
        total_uploaded += uploaded

        chunk_total_time = dl_time + ext_time + up_time
        logger.info(
            f"  [{chunk_num}/{total_chunks}] ✓ {chunk_label} COMPLETE in {chunk_total_time:.1f}s — "
            f"running totals: {total_videos} videos, {total_frames_ok} frames, {total_uploaded} uploads"
        )

    elapsed_total = (datetime.now() - t0).total_seconds()
    output_gcs_uri = f"gs://{gcs_dest}"

    logger.info("\n" + "=" * 70)
    logger.info("DONE — SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total videos:  {total_videos}")
    logger.info(f"  Total frames:  {total_frames_ok}")
    logger.info(f"  Total errors:  {total_errors}")
    logger.info(f"  Total uploads: {total_uploaded}")
    logger.info(f"  Total time:    {elapsed_total:.1f}s")
    logger.info(f"  Output:        {output_gcs_uri}")
    logger.info("=" * 70)

    return {
        "output_gcs_path": output_gcs_uri,
        "total_videos": total_videos,
        "total_frames_extracted": total_frames_ok,
        "total_errors": total_errors,
        "total_uploaded": total_uploaded,
        "total_time_seconds": elapsed_total,
    }


# ──────────────────────────────────────────────────────────────────────────────
# HLX WORKFLOW
# ──────────────────────────────────────────────────────────────────────────────
@workflow
def extract_frames_wf(
    output_folder_name: str = "trainingdata_chunk00_25",
    chunk_start: int = 0,
    chunk_end: int = 25,
    frame_numbers: str = "1,100,200,300,400,500",
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict:
    """Workflow: extract frames from driving videos for FluxFill training.

    Args:
        output_folder_name: Destination folder name under training data prefix.
        chunk_start: First chunk index (inclusive, 0-based).
        chunk_end: Last chunk index (inclusive, 0-based).
        frame_numbers: Comma-separated 1-based frame numbers.
        num_workers: Parallel extraction workers.

    Returns:
        Summary dict.
    """
    return extract_frames_task(
        output_folder_name=output_folder_name,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        frame_numbers=frame_numbers,
        num_workers=num_workers,
    )
