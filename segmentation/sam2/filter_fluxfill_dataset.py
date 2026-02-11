#!/usr/bin/env python3
"""Filter a generated FluxFill dataset by lane-marking prompt and copy to a new folder.

This script expects an input folder that looks like:
  <input_dir>/train.csv
  <input_dir>/images/<name>.png
  <input_dir>/masks/<name>.png

It writes:
  <output_dir>/train.csv
  <output_dir>/images/...
  <output_dir>/masks/...

Filtering is done based on the normalized prompt format produced by
`segmentation/sam2/data_generation.py`:
  "road with {count} {color} {pattern} lane markings"

If your prompts are not in this exact format, the row will be skipped.

This script supports both local folders and GCS prefixes:
    - Local: /abs/path/to/run
    - GCS:   gs://<bucket>/<prefix>/.../test_data_100_7b
    - GCS:   <bucket>/<prefix>/.../test_data_100_7b   (scheme-less)

If you run it with no CLI args, it will use the CONFIG block at the top of the file.

For GCS input/output, the script stages the filtered dataset into a local temporary
folder and then uploads the full folder structure to the output prefix (similar to
`segmentation/sam2/data_generation.py`).

Examples (CLI):
    python filter_fluxfill_dataset.py \
        --input_dir gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data/test_data_100_7b \
        --suffix single_white_dashed \
        --count single --color white --pattern dashed

  # Keep everything with single lanes regardless of color/pattern
  python filter_fluxfill_dataset.py \
    --input_dir /path/to/run \
    --output_dir /path/to/run_single_any \
    --count single
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------------------
# HLX WORKFLOW CONFIG (optional)
# --------------------------------------------------------------------------------------
# These are only used when running via `hlx wf run ...`.

CONTAINER_IMAGE_DEFAULT = os.environ.get("FLUXFILL_DATA_CONTAINER_IMAGE") or os.environ.get(
    "SAM2_CONTAINER_IMAGE", "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2"
)
CONTAINER_IMAGE = os.environ.get("FLUXFILL_DATA_CONTAINER_IMAGE", CONTAINER_IMAGE_DEFAULT)


# --------------------------------------------------------------------------------------
# CONFIG (optional)
# --------------------------------------------------------------------------------------
# If you prefer to "set the sorting instruction at the start of the code",
# edit these values and run:
#   python filter_fluxfill_dataset.py

CONFIG_INPUT_DIR = ""

# Filtering instruction (use "any" to not filter on that field)
CONFIG_COUNT = "any"  # single|double|unknown|any
CONFIG_COLOR = "any"  # white|yellow|mixed|unknown|any
CONFIG_PATTERN = "any"  # solid|dashed|mixed|unknown|any

# Output folder behavior
# If CONFIG_OUTPUT_DIR is empty, it will auto-create a sibling folder next to input_dir
# by appending "__<suffix>".
CONFIG_SUFFIX = ""
CONFIG_OUTPUT_DIR = ""

# Copy mode and CSV sort
CONFIG_COPY_MODE = "copy"  # copy|hardlink|symlink (GCS always uses copy)
CONFIG_SORT = "prompt"  # prompt|image
CONFIG_LIMIT = 0

# Keep only *clear* lane markings (i.e. reject any prompts with unknown attributes).
# This also implicitly excludes negative prompts like "no clear road with lane line markings"
# because they don't match the normalized prompt regex.
CONFIG_REQUIRE_CLEAR_ROAD = 1


@dataclass(frozen=True)
class LaneAttrs:
    count: str
    color: str
    pattern: str


_ALLOWED = {
    "count": {"single", "double", "unknown"},
    "color": {"white", "yellow", "mixed", "unknown"},
    "pattern": {"solid", "dashed", "mixed", "unknown"},
}


_PROMPT_RE = re.compile(
    r"^\s*road\s+with\s+(?P<count>single|double|unknown)\s+"
    r"(?P<color>white|yellow|mixed|unknown)\s+"
    r"(?P<pattern>solid|dashed|mixed|unknown)\s+lane\s+markings\s*$",
    flags=re.IGNORECASE,
)


def _parse_lane_attrs(prompt: str) -> Optional[LaneAttrs]:
    m = _PROMPT_RE.match((prompt or "").strip())
    if not m:
        return None
    return LaneAttrs(
        count=m.group("count").lower(),
        color=m.group("color").lower(),
        pattern=m.group("pattern").lower(),
    )


def _norm_choice(val: Optional[str], kind: str) -> Optional[str]:
    if val is None:
        return None
    v = val.strip().lower()
    if v in {"any", "*"}:
        return None
    if v not in _ALLOWED[kind]:
        raise ValueError(f"Invalid --{kind} '{val}'. Allowed: {sorted(_ALLOWED[kind])} or 'any'.")
    return v


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path, *, mode: str) -> None:
    _ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")


def _is_gcs_path(p: str) -> bool:
    p = (p or "").strip()
    return p.startswith("gs://") or re.match(r"^[a-z0-9][a-z0-9\-_.]{1,62}/", p) is not None


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key_prefix) from either gs://bucket/key or bucket/key."""
    u = (uri or "").strip()
    if u.startswith("gs://"):
        u = u[len("gs://") :]
    if "/" not in u:
        raise ValueError(f"Invalid GCS path (expected bucket/prefix): {uri}")
    bucket, key = u.split("/", 1)
    return bucket, key.rstrip("/")


def _join_gcs(bucket: str, prefix: str, rel: str) -> str:
    rel = (rel or "").lstrip("/")
    base = f"{bucket}/{prefix.strip('/')}" if prefix else bucket
    return f"{base.rstrip('/')}/{rel}" if rel else base.rstrip("/")


def _derive_suffix(count: Optional[str], color: Optional[str], pattern: Optional[str]) -> str:
    # Use 'any' placeholders to keep names readable.
    c = count or "any"
    col = color or "any"
    pat = pattern or "any"
    return f"{c}_{col}_{pat}"


def _auto_output_dir(input_dir: str, suffix: str) -> str:
    sfx = (suffix or "").strip() or "filtered"
    if _is_gcs_path(input_dir):
        bucket, key = _split_gcs_uri(input_dir)
        return f"gs://{bucket}/{key}__{sfx}"
    p = Path(input_dir).expanduser().resolve()
    return str(p.parent / f"{p.name}__{sfx}")


def _upload_directory_to_gcs(*, fs, local_dir: Path, gcs_prefix: str) -> None:
    """Upload a local directory to a GCS prefix (bucket/key style).

    gcs_prefix must be in the form "bucket/some/prefix" (no gs://).
    """
    base = Path(local_dir)
    for path in base.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(base).as_posix()
        remote = f"{gcs_prefix.rstrip('/')}/{rel}"
        remote_parent = os.path.dirname(remote)
        if remote_parent:
            fs.makedirs(remote_parent, exist_ok=True)
        fs.put(str(path), remote)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Filter a FluxFill dataset by lane prompt and copy it")
    ap.add_argument(
        "--input_dir",
        default="",
        help="Folder or GCS prefix containing train.csv + images/ + masks/",
    )
    ap.add_argument(
        "--output_dir",
        default="",
        help="Folder or GCS prefix to write filtered dataset (default: derived from --suffix)",
    )

    ap.add_argument(
        "--suffix",
        default="",
        help="Suffix for auto-derived output_dir (output becomes <input>__<suffix>)",
    )

    ap.add_argument("--count", default="any", help="single|double|unknown|any")
    ap.add_argument("--color", default="any", help="white|yellow|mixed|unknown|any")
    ap.add_argument("--pattern", default="any", help="solid|dashed|mixed|unknown|any")

    ap.add_argument(
        "--copy_mode",
        default="copy",
        choices=["copy", "hardlink", "symlink"],
        help="How to materialize files into output_dir",
    )
    ap.add_argument(
        "--sort",
        default="prompt",
        choices=["prompt", "image"],
        help="Sort order for the output CSV",
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, keep at most this many rows (after filtering)",
    )

    ap.add_argument(
        "--clean_output",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1 (default), delete existing output_dir/prefix before writing",
    )

    ap.add_argument(
        "--require_clear_road",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "If 1 (default), keep only prompts with no 'unknown' lane attributes. "
            "This yields only clear lane-marking samples and excludes negative prompts."
        ),
    )

    return ap.parse_args()


def _resolve_effective_args(args: argparse.Namespace) -> argparse.Namespace:
    """Allow running with no CLI args by using the CONFIG_* block."""
    # If user provided at least one explicit arg, use argparse values.
    # Otherwise, fall back to CONFIG_*.
    if any(
        [
            bool((args.input_dir or "").strip()),
            bool((args.output_dir or "").strip()),
            bool((args.suffix or "").strip()),
            (args.count != "any"),
            (args.color != "any"),
            (args.pattern != "any"),
            (args.copy_mode != "copy"),
            (args.sort != "prompt"),
            bool(args.limit),
            (int(getattr(args, "require_clear_road", 1) or 0) != 1),
        ]
    ):
        return args

    args.input_dir = CONFIG_INPUT_DIR
    args.output_dir = CONFIG_OUTPUT_DIR
    args.suffix = CONFIG_SUFFIX
    args.count = CONFIG_COUNT
    args.color = CONFIG_COLOR
    args.pattern = CONFIG_PATTERN
    args.copy_mode = CONFIG_COPY_MODE
    args.sort = CONFIG_SORT
    args.limit = int(CONFIG_LIMIT)
    args.require_clear_road = int(CONFIG_REQUIRE_CLEAR_ROAD)
    return args


def _attrs_are_clear(a: LaneAttrs) -> bool:
    return (a.count != "unknown") and (a.color != "unknown") and (a.pattern != "unknown")


def run_filter(
    *,
    input_dir: str,
    output_dir: str = "",
    suffix: str = "",
    count: str = "any",
    color: str = "any",
    pattern: str = "any",
    copy_mode: str = "copy",
    sort: str = "prompt",
    limit: int = 0,
    clean_output: bool = True,
    require_clear_road: bool = True,
    verbose: bool = True,
) -> str:
    """Filter a FluxFill dataset and write a new dataset folder.

    Supports local paths and GCS prefixes. Returns the resolved output_dir.
    """

    if not (input_dir or "").strip():
        raise ValueError("Missing input_dir")

    wanted_count = _norm_choice(count, "count")
    wanted_color = _norm_choice(color, "color")
    wanted_pattern = _norm_choice(pattern, "pattern")

    sfx = (suffix or "").strip()
    if not sfx:
        sfx = _derive_suffix(wanted_count, wanted_color, wanted_pattern)

    output_dir_str = (output_dir or "").strip() or _auto_output_dir(input_dir, sfx)

    input_is_gcs = _is_gcs_path(input_dir)
    output_is_gcs = _is_gcs_path(output_dir_str)
    if input_is_gcs != output_is_gcs:
        raise ValueError("input_dir and output_dir must both be local OR both be GCS paths")

    if input_is_gcs and copy_mode != "copy":
        raise ValueError("For GCS paths, copy_mode must be 'copy'")

    if input_is_gcs:
        bucket_in, key_in = _split_gcs_uri(input_dir)
        bucket_out, key_out = _split_gcs_uri(output_dir_str)
        if bucket_in != bucket_out:
            raise ValueError("Cross-bucket copy is not supported by this script")

        try:
            import gcsfs
        except Exception as e:  # pragma: no cover
            raise ImportError("gcsfs is required for GCS input/output") from e

        fs = gcsfs.GCSFileSystem(token="google_default")
        in_csv = _join_gcs(bucket_in, key_in, "train.csv")
        if not fs.exists(in_csv):
            raise FileNotFoundError(f"Missing gs://{in_csv}")

        rows: list[dict[str, str]] = []
        skipped_bad_prompt = 0
        skipped_missing_files = 0

        with fs.open(in_csv, "r") as f:
            reader = csv.DictReader(f)
            required = {"image", "mask", "prompt"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"train.csv must have columns {sorted(required)}; got {reader.fieldnames}")

            for row in reader:
                image_rel = (row.get("image") or "").strip()
                mask_rel = (row.get("mask") or "").strip()
                prompt = (row.get("prompt") or "").strip()
                prompt_2 = (row.get("prompt_2") or "").strip() or prompt

                attrs = _parse_lane_attrs(prompt)
                if attrs is None:
                    skipped_bad_prompt += 1
                    continue

                if require_clear_road and not _attrs_are_clear(attrs):
                    skipped_bad_prompt += 1
                    continue

                if wanted_count is not None and attrs.count != wanted_count:
                    continue
                if wanted_color is not None and attrs.color != wanted_color:
                    continue
                if wanted_pattern is not None and attrs.pattern != wanted_pattern:
                    continue

                src_img = _join_gcs(bucket_in, key_in, image_rel)
                src_msk = _join_gcs(bucket_in, key_in, mask_rel)
                if not fs.exists(src_img) or not fs.exists(src_msk):
                    skipped_missing_files += 1
                    continue

                rows.append(
                    {
                        "image": image_rel,
                        "mask": mask_rel,
                        "prompt": prompt,
                        "prompt_2": prompt_2,
                    }
                )

                if limit and len(rows) >= int(limit):
                    break

        if sort == "prompt":
            rows.sort(key=lambda r: (r.get("prompt", ""), r.get("image", "")))
        else:
            rows.sort(key=lambda r: r.get("image", ""))

        with tempfile.TemporaryDirectory(prefix="fluxfill_filter_") as td:
            stage_root = Path(td) / "out"
            stage_images = stage_root / "images"
            stage_masks = stage_root / "masks"
            stage_images.mkdir(parents=True, exist_ok=True)
            stage_masks.mkdir(parents=True, exist_ok=True)

            staged_rows: list[dict[str, str]] = []
            for r in rows:
                img_rel = (r.get("image") or "").strip()
                msk_rel = (r.get("mask") or "").strip()
                if not img_rel or not msk_rel:
                    skipped_missing_files += 1
                    continue

                src_img = _join_gcs(bucket_in, key_in, img_rel)
                src_msk = _join_gcs(bucket_in, key_in, msk_rel)

                dst_img = stage_images / Path(img_rel).name
                dst_msk = stage_masks / Path(msk_rel).name

                try:
                    fs.get(src_img, str(dst_img))
                    fs.get(src_msk, str(dst_msk))
                except FileNotFoundError:
                    skipped_missing_files += 1
                    continue

                staged_rows.append(
                    {
                        "image": f"images/{dst_img.name}",
                        "mask": f"masks/{dst_msk.name}",
                        "prompt": (r.get("prompt") or "").strip(),
                        "prompt_2": (r.get("prompt_2") or "").strip(),
                    }
                )

            stage_csv = stage_root / "train.csv"
            with stage_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["image", "mask", "prompt", "prompt_2"])
                w.writeheader()
                w.writerows(staged_rows)

            out_prefix = _join_gcs(bucket_out, key_out, "")

            # Make the output prefix contain ONLY this run's selected subset.
            if clean_output and fs.exists(out_prefix):
                fs.rm(out_prefix, recursive=True)

            _upload_directory_to_gcs(fs=fs, local_dir=stage_root, gcs_prefix=out_prefix)

        if verbose:
            print(
                "\n".join(
                    [
                        f"Input: gs://{bucket_in}/{key_in}",
                        f"Output: gs://{bucket_out}/{key_out}",
                        f"Wrote: gs://{bucket_out}/{key_out}/train.csv",
                        f"Rows kept: {len(staged_rows)}",
                        f"Skipped (bad prompt format): {skipped_bad_prompt}",
                        f"Skipped (missing files): {skipped_missing_files}",
                    ]
                )
            )
        return output_dir_str

    # Local filesystem mode
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir_str).expanduser().resolve()

    if clean_output and output_path.exists():
        shutil.rmtree(output_path)

    in_csv = input_path / "train.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing {in_csv}")

    rows: list[dict[str, str]] = []
    skipped_bad_prompt = 0
    skipped_missing_files = 0

    with in_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image", "mask", "prompt"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"train.csv must have columns {sorted(required)}; got {reader.fieldnames}")

        for row in reader:
            image_rel = (row.get("image") or "").strip()
            mask_rel = (row.get("mask") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            prompt_2 = (row.get("prompt_2") or "").strip() or prompt

            attrs = _parse_lane_attrs(prompt)
            if attrs is None:
                skipped_bad_prompt += 1
                continue

            if require_clear_road and not _attrs_are_clear(attrs):
                skipped_bad_prompt += 1
                continue

            if wanted_count is not None and attrs.count != wanted_count:
                continue
            if wanted_color is not None and attrs.color != wanted_color:
                continue
            if wanted_pattern is not None and attrs.pattern != wanted_pattern:
                continue

            src_img = (input_path / image_rel).resolve()
            src_msk = (input_path / mask_rel).resolve()
            if not src_img.exists() or not src_msk.exists():
                skipped_missing_files += 1
                continue

            rows.append(
                {
                    "image": image_rel,
                    "mask": mask_rel,
                    "prompt": prompt,
                    "prompt_2": prompt_2,
                }
            )

            if limit and len(rows) >= int(limit):
                break

    if sort == "prompt":
        rows.sort(key=lambda r: (r.get("prompt", ""), r.get("image", "")))
    else:
        rows.sort(key=lambda r: r.get("image", ""))

    out_images = output_path / "images"
    out_masks = output_path / "masks"
    _ensure_dir(out_images)
    _ensure_dir(out_masks)

    staged_rows: list[dict[str, str]] = []
    for r in rows:
        image_rel = (r.get("image") or "").strip()
        mask_rel = (r.get("mask") or "").strip()
        if not image_rel or not mask_rel:
            continue
        src_img = (input_path / image_rel).resolve()
        src_msk = (input_path / mask_rel).resolve()
        if not src_img.exists() or not src_msk.exists():
            skipped_missing_files += 1
            continue

        dst_img = out_images / Path(image_rel).name
        dst_msk = out_masks / Path(mask_rel).name
        _copy_file(src_img, dst_img, mode=copy_mode)
        _copy_file(src_msk, dst_msk, mode=copy_mode)

        staged_rows.append(
            {
                "image": f"images/{dst_img.name}",
                "mask": f"masks/{dst_msk.name}",
                "prompt": (r.get("prompt") or "").strip(),
                "prompt_2": (r.get("prompt_2") or "").strip(),
            }
        )

    out_csv = output_path / "train.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "mask", "prompt", "prompt_2"])
        w.writeheader()
        w.writerows(staged_rows)

    if verbose:
        print(
            "\n".join(
                [
                    f"Wrote: {out_csv}",
                    f"Rows kept: {len(staged_rows)}",
                    f"Skipped (bad prompt format): {skipped_bad_prompt}",
                    f"Skipped (missing files): {skipped_missing_files}",
                ]
            )
        )

    return output_dir_str


def main() -> None:
    args = _resolve_effective_args(parse_args())
    run_filter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        suffix=args.suffix,
        count=args.count,
        color=args.color,
        pattern=args.pattern,
        copy_mode=args.copy_mode,
        sort=args.sort,
        limit=int(args.limit or 0),
        clean_output=bool(int(getattr(args, "clean_output", 1) or 0)),
        require_clear_road=bool(int(getattr(args, "require_clear_road", 1) or 0)),
        verbose=True,
    )


# --------------------------------------------------------------------------------------
# HLX TASK + WORKFLOW
# --------------------------------------------------------------------------------------

try:
    from hlx.wf import DedicatedNode, Node, task, workflow
except Exception:  # pragma: no cover
    DedicatedNode = None  # type: ignore
    Node = None  # type: ignore
    task = None  # type: ignore
    workflow = None  # type: ignore


if task is not None and workflow is not None and DedicatedNode is not None and Node is not None:

    @task(
        compute=DedicatedNode(
            node=Node.CPU_32,
            ephemeral_storage="max",
            max_duration="1d",
        ),
        container_image=CONTAINER_IMAGE,
        environment={
            "PYTHONUNBUFFERED": "1",
        },
    )
    def filter_fluxfill_dataset_task(
        *,
        input_dir: str,
        suffix: str = "",
        output_dir: str = "",
        count: str = "any",
        color: str = "any",
        pattern: str = "any",
        sort: str = "prompt",
        limit: int = 0,
        clean_output: int = 1,
        require_clear_road: int = 1,
    ) -> str:
        """HLX task wrapper around `run_filter`.

        Use GCS paths for input/output for best results.
        Returns the resolved output_dir.
        """

        return run_filter(
            input_dir=input_dir,
            output_dir=output_dir,
            suffix=suffix,
            count=count,
            color=color,
            pattern=pattern,
            copy_mode="copy",
            sort=sort,
            limit=int(limit or 0),
            clean_output=bool(int(clean_output or 0)),
            require_clear_road=bool(int(require_clear_road or 0)),
            verbose=True,
        )


    @workflow
    def filter_fluxfill_dataset_wf(
        input_dir: str,
        suffix: str = "",
        output_dir: str = "",
        count: str = "any",
        color: str = "any",
        pattern: str = "any",
        sort: str = "prompt",
        limit: int = 0,
        clean_output: int = 1,
        require_clear_road: int = 1,
    ) -> str:
        return filter_fluxfill_dataset_task(
            input_dir=input_dir,
            suffix=suffix,
            output_dir=output_dir,
            count=count,
            color=color,
            pattern=pattern,
            sort=sort,
            limit=limit,
            clean_output=clean_output,
            require_clear_road=require_clear_road,
        )


if __name__ == "__main__":
    main()
