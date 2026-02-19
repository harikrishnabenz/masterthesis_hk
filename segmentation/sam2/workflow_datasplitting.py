"""Data-splitting workflow for preprocessed VP data.

Reads a preprocessed-data folder from GCS (output of SAM2 pipeline),
splits the videos into N equal parts, and copies each part to its own
folder whose name encodes the run-id, part number, and video count:

    gs://<bucket>/.../preprocessed_data_vp/<source_folder>/
        → .../preprocessed_data_vp/<run_id>_<M>videos_part1of3/
        → .../preprocessed_data_vp/<run_id>_<M>videos_part2of3/
        → .../preprocessed_data_vp/<run_id>_<M>videos_part3of3/

If num_splits=1 (default), the whole dataset is copied to a single folder:
        → .../preprocessed_data_vp/<run_id>_<N>videos/

This workflow reuses the SAM2 container image (no GPU needed).
"""

import logging
import math
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple

from hlx.wf import DedicatedNode, Node, task, workflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Container image — reuse the SAM2 image (set by build_and_run_splitdata.sh)
# ---------------------------------------------------------------------------
CONTAINER_IMAGE_DEFAULT = (
    "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2"
)
CONTAINER_IMAGE = os.environ.get(
    "SAM2_CONTAINER_IMAGE", f"{CONTAINER_IMAGE_DEFAULT}:latest"
)

# ---------------------------------------------------------------------------
# GCS paths
# ---------------------------------------------------------------------------
GCS_BUCKET = "mbadas-sandbox-research-9bb9c7f"
PREPROCESSED_BASE = os.environ.get(
    "PREPROCESSED_BASE",
    f"gs://{GCS_BUCKET}/workspace/user/hbaskar/outputs/preprocessed_data_vp",
)


def _compute_splits(total: int, num_splits: int) -> List[Tuple[int, int]]:
    """Divide *total* items into *num_splits* groups.

    Returns a list of ``(start_index, end_index)`` tuples (end exclusive).
    If the total is not evenly divisible, earlier parts get the extra items.

    Examples
    --------
    >>> _compute_splits(10, 3)   # 4 + 3 + 3
    [(0, 4), (4, 7), (7, 10)]
    >>> _compute_splits(9, 3)    # 3 + 3 + 3
    [(0, 3), (3, 6), (6, 9)]
    >>> _compute_splits(7, 3)    # 3 + 2 + 2
    [(0, 3), (3, 5), (5, 7)]
    """
    base_size = total // num_splits
    remainder = total % num_splits
    splits: List[Tuple[int, int]] = []
    start = 0
    for i in range(num_splits):
        size = base_size + (1 if i < remainder else 0)
        splits.append((start, start + size))
        start += size
    return splits


def _discover_video_folders(fs, src_gcs_path: str) -> List[str]:
    """Return a sorted list of video sub-folder paths under *src_gcs_path*.

    Each video produced by SAM2 is stored in its own sub-folder containing
    ``raw_videos/``, ``mask_root/``, and ``meta.csv``.
    """
    try:
        entries = fs.ls(src_gcs_path, detail=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Source folder not found on GCS: gs://{src_gcs_path}\n"
            f"Please check that the folder exists."
        )

    folders = sorted(
        e["name"] for e in entries if e["type"] == "directory"
    )
    return folders


def _copy_video_folder(fs, src_folder: str, dest_base: str) -> None:
    """Copy a single video sub-folder into *dest_base*."""
    folder_name = src_folder.rstrip("/").split("/")[-1]
    dest = f"{dest_base}/{folder_name}"
    logger.info("  Copying %s → %s", src_folder, dest)
    fs.copy(src_folder, dest, recursive=True)


@task(
    compute=DedicatedNode(
        node=Node.CPU_8,
        ephemeral_storage="max",
        max_duration="6h",
    ),
    container_image=CONTAINER_IMAGE,
    environment={"PYTHONUNBUFFERED": "1"},
)
def split_data(
    run_id: str,
    source_folder: str = "006_20260219_135038",
    num_splits: int = 1,
    preprocessed_base: str = PREPROCESSED_BASE,
) -> str:
    """Split preprocessed VP data into *num_splits* parts.

    Steps
    -----
    1. Discover all video sub-folders in the source.
    2. Divide them into *num_splits* groups (remainder distributed to earlier parts).
    3. Copy each group to its own destination folder:
       - 1 split  → ``<run_id>_<N>videos``
       - N splits → ``<run_id>_<M>videos_part1ofN`` … ``<run_id>_<M>videos_partNofN``

    Args:
        run_id: Identifier for this run (used in the destination folder name).
        source_folder: Name of the existing folder under *preprocessed_base*.
        num_splits: Number of parts to split the dataset into (default 1 = copy all).
        preprocessed_base: GCS base path for preprocessed VP data.

    Returns:
        Summary string with the destination paths.
    """
    import gcsfs

    src_uri = f"{preprocessed_base}/{source_folder}"
    logger.info("=" * 60)
    logger.info("Data-Splitting Workflow")
    logger.info("  Source:      %s", src_uri)
    logger.info("  Run ID:      %s", run_id)
    logger.info("  Num splits:  %d", num_splits)
    logger.info("=" * 60)

    if num_splits < 1:
        raise ValueError(f"num_splits must be >= 1, got {num_splits}")

    # ------------------------------------------------------------------
    # 1. Connect to GCS and discover video sub-folders
    # ------------------------------------------------------------------
    fs = gcsfs.GCSFileSystem()
    src_gcs_path = src_uri.replace("gs://", "")

    video_folders = _discover_video_folders(fs, src_gcs_path)
    num_videos = len(video_folders)

    logger.info("Found %d video folder(s) in %s", num_videos, src_uri)
    for f in video_folders[:20]:
        logger.info("  %s", f.split("/")[-1])
    if num_videos > 20:
        logger.info("  … and %d more", num_videos - 20)

    if num_videos == 0:
        raise ValueError(
            f"No video sub-folders found in {src_uri}. Nothing to split."
        )

    if num_splits > num_videos:
        logger.warning(
            "num_splits (%d) > num_videos (%d); clamping to %d",
            num_splits, num_videos, num_videos,
        )
        num_splits = num_videos

    # ------------------------------------------------------------------
    # 2. Compute split ranges
    # ------------------------------------------------------------------
    split_ranges = _compute_splits(num_videos, num_splits)
    logger.info("Split plan (%d parts):", num_splits)
    for i, (s, e) in enumerate(split_ranges, 1):
        logger.info("  Part %d: videos %d–%d  (%d videos)", i, s, e - 1, e - s)

    # ------------------------------------------------------------------
    # 3. Copy each part to its destination
    # ------------------------------------------------------------------
    dest_uris = []
    for part_idx, (start, end) in enumerate(split_ranges, 1):
        part_videos = video_folders[start:end]
        part_count = len(part_videos)

        if num_splits == 1:
            dest_folder_name = f"{run_id}_{part_count}videos"
        else:
            dest_folder_name = (
                f"{run_id}_{part_count}videos_part{part_idx}of{num_splits}"
            )

        dest_uri = f"{preprocessed_base}/{dest_folder_name}"
        dest_gcs_path = dest_uri.replace("gs://", "")
        dest_uris.append(dest_uri)

        logger.info(
            "\nPart %d/%d: copying %d videos → %s",
            part_idx, num_splits, part_count, dest_uri,
        )

        for vf in part_videos:
            _copy_video_folder(fs, vf, dest_gcs_path)

        # Quick verification
        dest_entries = fs.ls(dest_gcs_path, detail=True)
        dest_dirs = [e for e in dest_entries if e["type"] == "directory"]
        logger.info(
            "  ✓ Verified %d video folder(s) in destination", len(dest_dirs)
        )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_list = "\n".join(f"  - {u}" for u in dest_uris)
    summary = f"""
Data-Splitting Complete!
========================
Source folder:   {src_uri}
Total videos:    {num_videos}
Num splits:      {num_splits}
Timestamp:       {timestamp}

Destination folders:
{dest_list}

Split breakdown:
"""
    for i, (s, e) in enumerate(split_ranges, 1):
        summary += f"  Part {i}: {e - s} videos (indices {s}–{e - 1})\n"

    logger.info(summary)
    return summary


@workflow
def datasplitting_wf(
    run_id: str,
    source_folder: str = "006_20260219_135038",
    num_splits: int = 1,
    preprocessed_base: str = PREPROCESSED_BASE,
) -> str:
    """Workflow: split preprocessed VP data into N equal parts.

    Args:
        run_id: Run identifier (used in the output folder names).
        source_folder: Existing folder name under the preprocessed base path
                       (e.g. ``006_20260219_135038``).
        num_splits: Number of parts to split into (default 1 = just copy/rename).
        preprocessed_base: GCS base URI for preprocessed data.

    Returns:
        Summary of the operation.
    """
    return split_data(
        run_id=run_id,
        source_folder=source_folder,
        num_splits=num_splits,
        preprocessed_base=preprocessed_base,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split preprocessed VP data into N equal parts"
    )
    parser.add_argument(
        "--source-folder",
        type=str,
        default="006_20260219_135038",
        help="Source folder name under the preprocessed base path",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=1,
        help="Number of parts to split the dataset into (default: 1 = no split)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID prefix for the destination folder (default: same as source)",
    )

    args = parser.parse_args()
    run_id = args.run_id or args.source_folder.split("_")[0]

    result = datasplitting_wf(
        run_id=run_id,
        source_folder=args.source_folder,
        num_splits=args.num_splits,
    )
    print(result)
