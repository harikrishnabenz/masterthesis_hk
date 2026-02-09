#!/usr/bin/env python3
"""Download required checkpoints into the local `ckpt/` folder.

This is intended for runs where you want all models available as local paths
(e.g. inside the container under `/workspace/VideoPainter/ckpt/...`) rather than
pulling from HuggingFace at runtime.

By default it downloads:
- SDXL inpainting: stabilityai/stable-diffusion-xl-1.0-inpainting-0.1
- Qwen2.5-VL 7B:  Qwen/Qwen2.5-VL-7B-Instruct

Notes:
- Some repos require accepting terms on HuggingFace and/or an access token.
  Set `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) in your environment.
- Downloads are large; ensure you have sufficient disk.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


# Note: the StabilityAI repo name is often gated (401/404). The official Diffusers
# SDXL-inpainting weights are published under this repo id.
DEFAULT_SDXL_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_QWEN_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"


def _snapshot_download(repo_id: str, local_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required. Install it with: pip install huggingface_hub"
        ) from e

    token = (
        os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HF_ACCESS_TOKEN")
    )

    local_dir.mkdir(parents=True, exist_ok=True)

    # Use the local_dir directly so downstream code can use the folder path.
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Download checkpoints into ckpt/")
    p.add_argument(
        "--ckpt_dir",
        default=str(Path(__file__).resolve().parents[1] / "ckpt"),
        help="Destination ckpt directory (default: <repo>/ckpt)",
    )
    p.add_argument(
        "--sdxl_repo",
        default=DEFAULT_SDXL_REPO,
        help=f"SDXL inpainting repo id (default: {DEFAULT_SDXL_REPO})",
    )
    p.add_argument(
        "--qwen_repo",
        default=DEFAULT_QWEN_REPO,
        help=f"Qwen repo id (default: {DEFAULT_QWEN_REPO})",
    )
    p.add_argument(
        "--sdxl_only",
        action="store_true",
        help="Download SDXL inpainting only",
    )
    p.add_argument(
        "--qwen_only",
        action="store_true",
        help="Download Qwen only",
    )

    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()

    download_sdxl = not args.qwen_only
    download_qwen = not args.sdxl_only
    if args.sdxl_only and args.qwen_only:
        raise SystemExit("Choose at most one of --sdxl_only or --qwen_only")

    if download_sdxl:
        sdxl_dest = ckpt_dir / "sdxl_inpaint"
        print(f"Downloading SDXL inpainting -> {sdxl_dest}")
        _snapshot_download(args.sdxl_repo, sdxl_dest)

    if download_qwen:
        qwen_dest = ckpt_dir / "vlm" / "Qwen2.5-VL-7B-Instruct"
        print(f"Downloading Qwen 7B -> {qwen_dest}")
        _snapshot_download(args.qwen_repo, qwen_dest)

    print("Done.")
    print("Local paths you can use inside the container:")
    print("- SDXL inpaint: /workspace/VideoPainter/ckpt/sdxl_inpaint")
    print("- Qwen 7B:      /workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
