#!/usr/bin/env python3
"""Overlay predicted & ground-truth trajectories on the original video.

Usage (standalone):
    python visualize_video.py \
        --video_path /path/to/original_video.mp4 \
        --npz_path   /path/to/<video_id>_vis_data.npz \
        --output_path /path/to/output_overlay.mp4

    # Optional: also pass the JSON for extra info in the HUD
    python visualize_video.py \
        --video_path /path/to/original_video.mp4 \
        --npz_path   /path/to/<video_id>_vis_data.npz \
        --json_path  /path/to/<video_id>_inference.json \
        --output_path /path/to/output_overlay.mp4

The script can also be imported:
    from visualize_video import render_trajectory_video
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Projection helpers ───────────────────────────────────────────────────

def project_trajectory_to_image(
    traj_xyz: np.ndarray,
    img_h: int,
    img_w: int,
    fov_deg: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D ego-frame trajectory onto a front-facing camera image.

    Ego frame: x=forward, y=left, z=up.
    Camera frame: z=forward, x=right, y=down.

    Returns (u, v, mask) — pixel coordinates and validity mask.
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = (img_w / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    cx, cy = img_w / 2.0, img_h / 2.0

    x_cam = -traj_xyz[:, 1]   # right
    y_cam = -traj_xyz[:, 2]   # down
    z_cam = traj_xyz[:, 0]    # forward

    in_front = z_cam > 0.5
    u = np.full_like(z_cam, -1.0)
    v = np.full_like(z_cam, -1.0)
    u[in_front] = fx * x_cam[in_front] / z_cam[in_front] + cx
    v[in_front] = fy * y_cam[in_front] / z_cam[in_front] + cy
    in_bounds = in_front & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return u, v, in_bounds


def draw_trajectory_on_frame(
    frame: np.ndarray,
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    fov_deg: float = 120.0,
    history_xyz: Optional[np.ndarray] = None,
    t_reveal: Optional[int] = None,
    draw_info_text: Optional[str] = None,
) -> np.ndarray:
    """Draw projected trajectories on a single BGR frame (in-place safe copy).

    Args:
        frame: (H, W, 3) BGR image.
        pred_xyz: (S, 64, 3) predicted trajectories.
        gt_xyz: (64, 3) ground truth trajectory.
        fov_deg: horizontal FOV of the camera.
        history_xyz: (T, 3) optional ego history path.
        t_reveal: if set, only draw waypoints 0..t_reveal (for progressive reveal).
        draw_info_text: optional text to render in the top-left corner.

    Returns:
        Annotated BGR frame (copy of input).
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # --- Ground truth (red) ---
    gt = gt_xyz if t_reveal is None else gt_xyz[: t_reveal + 1]
    u_gt, v_gt, mask_gt = project_trajectory_to_image(gt, h, w, fov_deg)
    _draw_path(out, u_gt, v_gt, mask_gt, color=(0, 0, 255), thickness=3, circle_radius=5,
               label="GT")

    # --- Predicted trajectories (cyan / green shades) ---
    sample_colors = [
        (255, 255, 0),   # cyan
        (0, 255, 0),     # green
        (255, 165, 0),   # orange-ish in BGR → (0, 165, 255) but let's keep readable
        (0, 255, 255),   # yellow
        (255, 0, 255),   # magenta
    ]
    num_samples = pred_xyz.shape[0]
    for s in range(num_samples):
        pred = pred_xyz[s] if t_reveal is None else pred_xyz[s, : t_reveal + 1]
        u_p, v_p, mask_p = project_trajectory_to_image(pred, h, w, fov_deg)
        color = sample_colors[s % len(sample_colors)]
        _draw_path(out, u_p, v_p, mask_p, color=color, thickness=2, circle_radius=4,
                   label=f"Pred#{s + 1}")

    # --- Ego history (gray) ---
    if history_xyz is not None:
        u_h, v_h, mask_h = project_trajectory_to_image(history_xyz, h, w, fov_deg)
        _draw_path(out, u_h, v_h, mask_h, color=(180, 180, 180), thickness=1,
                   circle_radius=3, label=None)

    # --- Legend ---
    _draw_legend(out, num_samples, sample_colors)

    # --- Optional info text ---
    if draw_info_text:
        for i, line in enumerate(draw_info_text.split("\n")):
            cv2.putText(out, line, (10, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

    return out


def _draw_path(
    img: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    color: tuple,
    thickness: int = 2,
    circle_radius: int = 4,
    label: Optional[str] = None,
):
    """Draw a polyline + circles for valid projected points."""
    pts = np.column_stack([u[mask].astype(int), v[mask].astype(int)])
    if len(pts) < 2:
        return

    # Draw anti-aliased polyline
    for i in range(len(pts) - 1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color, thickness, cv2.LINE_AA)

    # Draw circles at waypoints (every few to avoid clutter)
    step = max(1, len(pts) // 12)
    for i in range(0, len(pts), step):
        cv2.circle(img, tuple(pts[i]), circle_radius, color, -1, cv2.LINE_AA)

    # Mark the endpoint with a larger circle
    cv2.circle(img, tuple(pts[-1]), circle_radius + 2, color, 2, cv2.LINE_AA)


def _draw_legend(img: np.ndarray, num_samples: int, sample_colors: list):
    """Draw a small legend box in the top-right corner."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    line_h = 22
    pad = 8

    entries = [("Ground Truth", (0, 0, 255))]
    for s in range(num_samples):
        entries.append((f"Predicted #{s + 1}", sample_colors[s % len(sample_colors)]))

    box_w = 180
    box_h = len(entries) * line_h + 2 * pad
    x0 = w - box_w - 10
    y0 = 10

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    for i, (label, color) in enumerate(entries):
        y = y0 + pad + (i + 1) * line_h - 5
        cv2.circle(img, (x0 + 12, y - 4), 5, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 24, y), font, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)


# ── BEV inset helper ────────────────────────────────────────────────────

def render_bev_inset(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    history_xyz: Optional[np.ndarray] = None,
    size: int = 250,
    t_reveal: Optional[int] = None,
) -> np.ndarray:
    """Render a small BEV (bird's-eye-view) map as a BGR image.

    Returns an (size, size, 3) uint8 BGR image.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # dark gray background

    # Collect all points to determine scale
    all_pts = [gt_xyz[:, :2]]
    for s in range(pred_xyz.shape[0]):
        all_pts.append(pred_xyz[s, :, :2])
    if history_xyz is not None:
        all_pts.append(history_xyz[:, :2])
    all_pts = np.concatenate(all_pts, axis=0)

    # Display frame: dx = -y, dy = x  (forward = up)
    all_dx = -all_pts[:, 1]
    all_dy = all_pts[:, 0]

    margin = 5
    x_range = max(all_dx.max() - all_dx.min(), 1) + 2 * margin
    y_range = max(all_dy.max() - all_dy.min(), 1) + 2 * margin
    scale = (size - 20) / max(x_range, y_range)
    cx = size / 2 - (all_dx.min() + all_dx.max()) / 2 * scale
    cy = size / 2 - (all_dy.min() + all_dy.max()) / 2 * scale

    def to_px(dx, dy):
        return int(dx * scale + cx), int(size - (dy * scale + cy))

    # History (gray)
    if history_xyz is not None:
        hdx, hdy = -history_xyz[:, 1], history_xyz[:, 0]
        for i in range(len(hdx) - 1):
            p1 = to_px(hdx[i], hdy[i])
            p2 = to_px(hdx[i + 1], hdy[i + 1])
            cv2.line(img, p1, p2, (150, 150, 150), 1, cv2.LINE_AA)

    # Ground truth (red)
    gt_vis = gt_xyz if t_reveal is None else gt_xyz[: t_reveal + 1]
    gdx, gdy = -gt_vis[:, 1], gt_vis[:, 0]
    for i in range(len(gdx) - 1):
        p1 = to_px(gdx[i], gdy[i])
        p2 = to_px(gdx[i + 1], gdy[i + 1])
        cv2.line(img, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)

    # Predictions (cyan)
    pred_colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255)]
    for s in range(pred_xyz.shape[0]):
        pred = pred_xyz[s] if t_reveal is None else pred_xyz[s, : t_reveal + 1]
        pdx, pdy = -pred[:, 1], pred[:, 0]
        color = pred_colors[s % len(pred_colors)]
        for i in range(len(pdx) - 1):
            p1 = to_px(pdx[i], pdy[i])
            p2 = to_px(pdx[i + 1], pdy[i + 1])
            cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)

    # Ego origin (white dot)
    ego = to_px(0, 0)
    cv2.circle(img, ego, 4, (255, 255, 255), -1, cv2.LINE_AA)

    # Border
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (200, 200, 200), 1)

    # Label
    cv2.putText(img, "BEV", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)

    return img


# ── Main video rendering ────────────────────────────────────────────────

def render_trajectory_video(
    video_path: str,
    npz_path: str,
    output_path: str,
    json_path: Optional[str] = None,
    fov_deg: float = 120.0,
    progressive_reveal: bool = True,
    bev_inset: bool = True,
    bev_size: int = 250,
) -> str:
    """Render an output video with trajectory overlay.

    Args:
        video_path: path to the original .mp4 video.
        npz_path: path to the *_vis_data.npz file.
        output_path: where to write the annotated .mp4.
        json_path: optional *_inference.json for HUD text.
        fov_deg: camera horizontal FOV in degrees.
        progressive_reveal: if True, trajectories are revealed gradually
            across the video frames (first frame = start, last = full path).
        bev_inset: if True, draw a small BEV map in the bottom-left corner.
        bev_size: pixel size of the BEV inset.

    Returns:
        The output_path.
    """
    # Load npz data
    vis = dict(np.load(npz_path, allow_pickle=True))
    pred_xyz = vis["pred_xyz"]         # (S, 64, 3)
    gt_xyz = vis["gt_future_xyz"]      # (64, 3)
    history_xyz = vis.get("ego_history_xyz")  # (16, 3) or None

    num_waypoints = gt_xyz.shape[0]  # 64

    # Load optional JSON metadata
    info_text = None
    if json_path and os.path.isfile(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        ade = meta.get("min_ade_meters")
        info_text = f"minADE: {ade:.4f} m" if ade is not None else None
        clip_id = meta.get("clip_id", "")
        camera = meta.get("camera_name", "")
        if clip_id:
            info_text = f"clip: {clip_id[:20]}  cam: {camera}\n{info_text}"

    # Decode all frames from input video
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 8.0
    all_frames = []
    for frame in container.decode(stream):
        all_frames.append(frame.to_ndarray(format="bgr24"))  # BGR for OpenCV
    container.close()

    if not all_frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    num_frames = len(all_frames)
    h, w = all_frames[0].shape[:2]
    logger.info(f"Input video: {num_frames} frames, {w}x{h}, {fps:.1f} fps")
    logger.info(f"Trajectories: {pred_xyz.shape[0]} samples, {num_waypoints} waypoints")

    # Write output video using PyAV with H.264 codec for universal playback
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    out_container = av.open(output_path, mode="w")
    from fractions import Fraction
    out_stream = out_container.add_stream("libx264", rate=Fraction(fps).limit_denominator(10000))
    out_stream.width = w
    out_stream.height = h
    out_stream.pix_fmt = "yuv420p"
    # Use a reasonable CRF for quality/size balance
    out_stream.options = {"crf": "18", "preset": "medium"}

    for i, frame in enumerate(all_frames):
        # Progressive reveal: linearly map frame index to waypoint count
        if progressive_reveal:
            t_reveal = int((i / max(num_frames - 1, 1)) * (num_waypoints - 1))
        else:
            t_reveal = None

        annotated = draw_trajectory_on_frame(
            frame=frame,
            pred_xyz=pred_xyz,
            gt_xyz=gt_xyz,
            fov_deg=fov_deg,
            history_xyz=history_xyz,
            t_reveal=t_reveal,
            draw_info_text=info_text,
        )

        # BEV inset in bottom-left
        if bev_inset:
            bev_img = render_bev_inset(
                pred_xyz=pred_xyz,
                gt_xyz=gt_xyz,
                history_xyz=history_xyz,
                size=bev_size,
                t_reveal=t_reveal,
            )
            y_off = h - bev_size - 10
            x_off = 10
            annotated[y_off : y_off + bev_size, x_off : x_off + bev_size] = bev_img

        # Convert BGR (OpenCV) → RGB (PyAV) and encode
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
        for packet in out_stream.encode(av_frame):
            out_container.mux(packet)

    # Flush remaining packets
    for packet in out_stream.encode():
        out_container.mux(packet)
    out_container.close()
    logger.info(f"Output video saved: {output_path} ({num_frames} frames, H.264/mp4)")
    return output_path


# ── Batch mode: process a whole output directory ─────────────────────────

def render_all_in_directory(
    results_dir: str,
    video_source_dir: str,
    output_dir: Optional[str] = None,
    fov_deg: float = 120.0,
):
    """Find all *_vis_data.npz files under results_dir and render overlay videos.

    For each npz, it looks for the original video in video_source_dir by matching
    the video_id from the companion JSON, or by matching the npz stem.

    Args:
        results_dir: directory containing inference outputs (JSON + npz).
        video_source_dir: directory containing the original .mp4 files.
        output_dir: where to write overlay videos. Defaults to results_dir/overlays/.
    """
    import glob

    if output_dir is None:
        output_dir = os.path.join(results_dir, "overlays")
    os.makedirs(output_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(results_dir, "**", "*_vis_data.npz"),
                                  recursive=True))
    if not npz_files:
        logger.warning(f"No *_vis_data.npz files found in {results_dir}")
        return

    logger.info(f"Found {len(npz_files)} npz file(s) to process")

    # Build a map of available source videos
    source_videos = {}
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        for p in Path(video_source_dir).rglob(ext):
            source_videos[p.stem] = str(p)

    for npz_path in npz_files:
        npz_stem = Path(npz_path).stem.replace("_vis_data", "")
        json_path = npz_path.replace("_vis_data.npz", "_inference.json")

        # Try to find the original video
        video_id = npz_stem
        if os.path.isfile(json_path):
            with open(json_path) as f:
                meta = json.load(f)
            video_id = meta.get("video_id", npz_stem)

        original_video = source_videos.get(video_id)
        if original_video is None:
            logger.warning(f"Original video not found for {video_id}, skipping")
            continue

        out_path = os.path.join(output_dir, f"{video_id}_overlay.mp4")
        logger.info(f"Rendering overlay: {video_id}")
        try:
            render_trajectory_video(
                video_path=original_video,
                npz_path=npz_path,
                output_path=out_path,
                json_path=json_path if os.path.isfile(json_path) else None,
                fov_deg=fov_deg,
            )
        except Exception as e:
            logger.error(f"Failed to render {video_id}: {e}", exc_info=True)

    logger.info(f"All overlays written to: {output_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Overlay Alpamayo trajectory predictions on the original video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video
  python visualize_video.py \\
      --video_path data/clip.camera_front_wide_120fov_vp_edit_sample0.mp4 \\
      --npz_path   output/clip_vis_data.npz \\
      --output_path output/clip_overlay.mp4

  # Batch: process all results in a directory
  python visualize_video.py \\
      --results_dir /tmp/alpamayo_output/run_20260217 \\
      --video_source_dir /path/to/original_videos \\
      --output_dir /tmp/alpamayo_output/run_20260217/overlays
        """,
    )
    sub = parser.add_subparsers(dest="mode")

    # Single video mode
    single = sub.add_parser("single", help="Overlay on a single video")
    single.add_argument("--video_path", required=True, help="Path to original .mp4")
    single.add_argument("--npz_path", required=True, help="Path to *_vis_data.npz")
    single.add_argument("--json_path", default=None, help="Path to *_inference.json (optional)")
    single.add_argument("--output_path", required=True, help="Output .mp4 path")
    single.add_argument("--fov", type=float, default=120.0, help="Camera FOV in degrees")
    single.add_argument("--no-progressive", action="store_true",
                        help="Show full trajectory on every frame")
    single.add_argument("--no-bev", action="store_true", help="Disable BEV inset")

    # Batch mode
    batch = sub.add_parser("batch", help="Batch overlay all videos in a results directory")
    batch.add_argument("--results_dir", required=True, help="Directory with *_vis_data.npz files")
    batch.add_argument("--video_source_dir", required=True,
                       help="Directory with original .mp4 files")
    batch.add_argument("--output_dir", default=None, help="Output directory (default: results_dir/overlays)")
    batch.add_argument("--fov", type=float, default=120.0, help="Camera FOV in degrees")

    args = parser.parse_args()

    if args.mode == "single":
        render_trajectory_video(
            video_path=args.video_path,
            npz_path=args.npz_path,
            output_path=args.output_path,
            json_path=args.json_path,
            fov_deg=args.fov,
            progressive_reveal=not args.no_progressive,
            bev_inset=not args.no_bev,
        )
    elif args.mode == "batch":
        render_all_in_directory(
            results_dir=args.results_dir,
            video_source_dir=args.video_source_dir,
            output_dir=args.output_dir,
            fov_deg=args.fov,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
