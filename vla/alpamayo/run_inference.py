#!/usr/bin/env python3
"""
Alpamayo VLA Inference Runner

Runs Alpamayo-R1-10B inference on driving videos produced by VideoPainter
and generates trajectory predictions with reasoning traces.

The script:
  1. Extracts clip_id and camera name from the VideoPainter mp4 filename.
  2. Loads the original driving clip from physical_ai_av for ego history,
     multi-camera frames, and ground-truth future trajectory.
  3. Replaces the inpainted camera's frames with frames decoded from the
     VideoPainter mp4.
  4. Runs Alpamayo inference to get predicted trajectories + chain-of-thought.
  5. Computes minADE vs. ground truth and saves results as JSON.
"""
import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import av
import numpy as np
import psutil
import torch
from einops import rearrange

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Camera index map (must match load_physical_aiavdataset.py) ───────────
CAMERA_NAME_TO_INDEX = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}


# ── Helpers ──────────────────────────────────────────────────────────────
def get_gpu_memory_gb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / (1024**3),
        torch.cuda.max_memory_allocated() / (1024**3),
    )


def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_ram_mb() -> float:
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def parse_video_filename(video_path: str) -> tuple[str, str]:
    """Extract clip_id and camera_name from a VideoPainter output filename.

    Expected patterns:
        <clip_id>.<camera_name>_vp_edit_sample0.mp4
        <clip_id>.<camera_name>_vp_edit_sample0_generated.mp4
    """
    stem = Path(video_path).stem  # drop .mp4
    # Strip common suffixes from VideoPainter
    stem = re.sub(r"_generated$", "", stem)
    stem = re.sub(r"_vp_edit_sample\d+$", "", stem)

    # Split on first '.' — uuid part . camera_name
    parts = stem.split(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Cannot parse clip_id.camera_name from filename: {Path(video_path).name}"
        )
    clip_id, camera_name = parts
    if camera_name not in CAMERA_NAME_TO_INDEX:
        raise ValueError(
            f"Unknown camera name '{camera_name}' extracted from {Path(video_path).name}. "
            f"Expected one of {list(CAMERA_NAME_TO_INDEX.keys())}"
        )
    return clip_id, camera_name


def decode_video_frames(
    video_path: str,
    num_frames: int = 4,
) -> tuple[torch.Tensor, float]:
    """Decode the last *num_frames* consecutive frames from an mp4.

    Because we tell load_physical_aiavdataset to use the same time_step as the
    VP video's frame interval, we simply take the last N consecutive frames
    (no temporal resampling).  This keeps everything temporally aligned:
    - VP frames: last N at native fps
    - Other cameras from dataset: sampled at the same fps
    - Ego-motion: sampled at the same fps

    Returns
    -------
    frames : torch.Tensor of shape (num_frames, 3, H, W), uint8.
    fps    : float – detected fps of the video.
    total_frames : int – total number of frames in the video.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]

    # Detect fps from container metadata
    detected_fps = float(stream.average_rate) if stream.average_rate else 8.0
    fps = detected_fps if detected_fps > 0 else 8.0

    # Decode all frames
    all_frames = []
    for frame in container.decode(stream):
        all_frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    if not all_frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    total = len(all_frames)
    logger.info(
        f"  Video stats: {total} frames, {fps:.1f} fps "
        f"({total / fps:.2f}s duration)"
    )

    if total < num_frames:
        logger.warning(
            f"Video has only {total} frames but {num_frames} requested; "
            f"duplicating last frame."
        )
        while len(all_frames) < num_frames:
            all_frames.append(all_frames[-1])
        total = len(all_frames)

    # Take the last num_frames consecutive frames (native fps spacing)
    start_idx = total - num_frames
    indices = list(range(start_idx, total))
    logger.info(f"  Selected frame indices: {indices} (last {num_frames} consecutive at {fps:.1f} fps)")
    selected = [all_frames[i] for i in indices]

    frames_np = np.stack(selected)  # (num_frames, H, W, 3)
    frames_t = torch.from_numpy(frames_np)
    frames_t = rearrange(frames_t, "t h w c -> t c h w")
    return frames_t, fps, total


# ── Main inference logic ─────────────────────────────────────────────────
def run_inference_on_video(
    video_path: str,
    model,
    processor,
    helper_mod,
    output_dir: str,
    num_traj_samples: int = 1,
    device: str = "cuda",
) -> dict:
    """Run Alpamayo inference on one VideoPainter-edited video."""
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    video_id = Path(video_path).stem
    logger.info(f"Processing video: {video_id}")

    reset_gpu_memory_stats()
    start_time = time.time()

    try:
        # 1. Parse clip_id and camera from filename
        clip_id, camera_name = parse_video_filename(video_path)
        logger.info(f"  clip_id={clip_id}  camera={camera_name}")

        # 2. Decode VP frames first so we know the video fps and length
        num_frames = 4  # Alpamayo always uses 4 frames per camera
        vp_frames, vp_fps, vp_total_frames = decode_video_frames(video_path, num_frames=num_frames)
        # vp_frames: (num_frames, 3, H_vp, W_vp)

        # 3. Compute t0_us from VP video metadata for temporal alignment.
        #
        #    The VP video is generated from the FIRST N native frames of
        #    the original clip (extracted by ffmpeg from frame 0, no seek
        #    offset).  The VP output is at 8 fps but each frame maps 1:1
        #    to a native frame in the source clip.
        #
        #    We take the last 4 frames of the VP video as the observation
        #    window, so the "present" moment t0 corresponds to the last
        #    VP frame.  The source clip's native data in physical_ai_av
        #    starts at timestamp 0, so:
        #
        #      t0_us = (vp_total_frames - 1) * time_step_us
        #
        #    where time_step matches the VP video's frame interval.
        #    This ensures the ego history, ego future, and all other
        #    camera frames downloaded from HuggingFace are temporally
        #    aligned with the generated VP frames.
        vp_time_step = 1.0 / vp_fps  # e.g. 1/8 = 0.125s for 8fps
        t0_us = int((vp_total_frames - 1) * vp_time_step * 1_000_000)
        logger.info(
            f"  VP video: {vp_total_frames} frames at {vp_fps:.1f} fps → "
            f"time_step={vp_time_step:.4f}s"
        )
        logger.info(
            f"  Computed t0_us={t0_us} ({t0_us / 1_000_000:.3f}s) from "
            f"last VP frame index ({vp_total_frames - 1}) × "
            f"time_step ({vp_time_step:.4f}s)  "
            f"[dataset default would be 5_100_000 = 5.1s]"
        )
        logger.info("  Loading original clip data from physical_ai_av …")
        data = load_physical_aiavdataset(
            clip_id,
            t0_us=t0_us,
            time_step=vp_time_step,
            num_frames=num_frames,
        )
        # data["image_frames"]: (N_cameras, num_frames, 3, H, W)

        # 4. Determine which camera index to replace
        cam_idx_value = CAMERA_NAME_TO_INDEX[camera_name]
        # data["camera_indices"] is sorted; find position
        positions = (data["camera_indices"] == cam_idx_value).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            logger.warning(
                f"  Camera {camera_name} (idx {cam_idx_value}) not in loaded data; "
                f"skipping replacement."
            )
        else:
            pos = positions[0].item()
            orig_h, orig_w = data["image_frames"].shape[-2:]
            vp_h, vp_w = vp_frames.shape[-2:]
            if (vp_h, vp_w) != (orig_h, orig_w):
                logger.info(
                    f"  Resizing VP frames from ({vp_h},{vp_w}) -> ({orig_h},{orig_w})"
                )
                import torch.nn.functional as F

                vp_frames_float = vp_frames.float()
                vp_frames_float = F.interpolate(
                    vp_frames_float, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                )
                vp_frames = vp_frames_float.to(data["image_frames"].dtype)
            data["image_frames"][pos] = vp_frames
            logger.info(f"  Replaced camera {camera_name} frames with VideoPainter output")

        # 5. Build model inputs (following test_inference.py pattern)
        messages = helper_mod.create_message(data["image_frames"].flatten(0, 1))
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        model_inputs = helper_mod.to_device(model_inputs, device)

        # 6. Run inference
        logger.info(f"  Running inference with {num_traj_samples} trajectory sample(s) …")
        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=num_traj_samples,
                max_generation_length=256,
                return_extra=True,
            )

        # 7. Compute minADE
        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()  # (2, T)
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)  # (S, 2, T)
        diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)  # (S,)
        min_ade = float(diff.min())

        inference_time = time.time() - start_time
        gpu_current, gpu_peak = get_gpu_memory_gb()

        # 8. Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_dir, f"{video_id}_inference.json")

        cot_texts = extra.get("cot", [[[""]]])[0]  # shape [ns, nj]
        cot_flat = []
        for traj_set in cot_texts:
            for text in traj_set:
                cot_flat.append(str(text))

        results = {
            "video_id": video_id,
            "video_path": video_path,
            "clip_id": clip_id,
            "camera_name": camera_name,
            "num_trajectories": num_traj_samples,
            "min_ade_meters": min_ade,
            "reasoning_traces": cot_flat,
            "temporal_config": {
                "vp_video_fps": vp_fps,
                "vp_total_frames": vp_total_frames,
                "time_step_seconds": vp_time_step,
                "t0_us": t0_us,
                "t0_seconds": t0_us / 1_000_000,
                "num_frames_per_camera": num_frames,
                "frame_window_seconds": (num_frames - 1) * vp_time_step,
                "image_frame_timestamps_s": [
                    (t0_us - (num_frames - 1 - i) * int(vp_time_step * 1_000_000)) / 1_000_000
                    for i in range(num_frames)
                ],
            },
            "metrics": {
                "inference_time_seconds": inference_time,
                "gpu_memory_used_gb": gpu_current,
                "gpu_memory_peak_gb": gpu_peak,
                "ram_used_mb": get_ram_mb(),
                "ram_peak_mb": get_ram_mb(),
            },
            "success": True,
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # 9. Save visualization data (tensors) for offline plotting
        vis_file = os.path.join(output_dir, f"{video_id}_vis_data.npz")
        try:
            # image_frames: (N_cameras, num_frames, 3, H, W) -> (N_cams*num_frames, H, W, 3) uint8
            image_frames_np = (
                data["image_frames"]
                .cpu()
                .numpy()
                .transpose(0, 1, 3, 4, 2)          # (N_cam, T, H, W, 3)
                .reshape(-1, *data["image_frames"].shape[-2:], 3)  # (N_cam*T, H, W, 3)
            )
            vis_data = {
                "pred_xyz": pred_xyz.cpu().numpy()[0, 0],       # (S, 64, 3)
                "pred_rot": pred_rot.cpu().numpy()[0, 0],       # (S, 64, 3, 3)
                "gt_future_xyz": data["ego_future_xyz"].cpu().numpy()[0, 0],   # (64, 3)
                "gt_future_rot": data["ego_future_rot"].cpu().numpy()[0, 0],   # (64, 3, 3)
                "ego_history_xyz": data["ego_history_xyz"].cpu().numpy()[0, 0],  # (16, 3)
                "ego_history_rot": data["ego_history_rot"].cpu().numpy()[0, 0],  # (16, 3, 3)
                "image_frames": image_frames_np,                # (N_cam*T, H, W, 3)
                "camera_indices": data["camera_indices"].cpu().numpy(),  # (N_cam,)
            }
            np.savez_compressed(vis_file, **vis_data)
            logger.info(f"  Visualization data saved → {vis_file}")
        except Exception as vis_err:
            logger.warning(f"  Failed to save visualization data: {vis_err}")

        # 10. Render overlay video (trajectories on original video)
        overlay_file = os.path.join(output_dir, f"{video_id}_overlay.mp4")
        try:
            from visualize_video import render_trajectory_video

            fov = 120.0 if "120fov" in camera_name else (30.0 if "30fov" in camera_name else 70.0)
            render_trajectory_video(
                video_path=video_path,
                npz_path=vis_file,
                output_path=overlay_file,
                json_path=output_file,
                fov_deg=fov,
                progressive_reveal=True,
                bev_inset=True,
            )
            results["overlay_video_path"] = overlay_file
            logger.info(f"  Overlay video saved → {overlay_file}")
        except Exception as overlay_err:
            logger.error(f"  Failed to render overlay video: {overlay_err}", exc_info=True)

        logger.info(f"  minADE = {min_ade:.4f} m | time = {inference_time:.1f}s | saved → {output_file}")
        return results

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        inference_time = time.time() - start_time
        gpu_current, gpu_peak = get_gpu_memory_gb()
        return {
            "video_id": video_id,
            "video_path": video_path,
            "num_trajectories": 0,
            "metrics": {
                "inference_time_seconds": inference_time,
                "gpu_memory_used_gb": gpu_current,
                "gpu_memory_peak_gb": gpu_peak,
                "ram_used_mb": get_ram_mb(),
                "ram_peak_mb": get_ram_mb(),
            },
            "success": False,
            "error": str(e),
        }


# ── CLI entry-point ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Alpamayo VLA inference on VideoPainter outputs")
    parser.add_argument("--video_path", type=str, required=True, help="Path to mp4 or directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--model_id",
        type=str,
        default="nvidia/Alpamayo-R1-10B",
        help="HuggingFace repo or local checkpoint path",
    )
    parser.add_argument("--num_traj_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ALPAMAYO VLA INFERENCE")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Input: {args.video_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Num trajectory samples: {args.num_traj_samples}")
    logger.info(f"Device: {args.device}")

    # ── Load model ────────────────────────────────────────────────────
    logger.info("Loading model …")
    try:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        from alpamayo_r1 import helper

        device = args.device if args.device != "auto" else "cuda"
        model = AlpamayoR1.from_pretrained(
            args.model_id, dtype=torch.bfloat16
        ).to(device)
        processor = helper.get_processor(model.tokenizer)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # ── Discover video files ──────────────────────────────────────────
    vp = Path(args.video_path)
    if vp.is_file():
        video_files = [str(vp)]
    elif vp.is_dir():
        video_files = sorted(str(p) for p in vp.rglob("*.mp4"))
    else:
        logger.error(f"Invalid video path: {args.video_path}")
        return 1

    if not video_files:
        logger.error(f"No video files found at: {args.video_path}")
        return 1

    logger.info(f"Found {len(video_files)} video(s) to process")

    # ── Run inference ─────────────────────────────────────────────────
    # Each video gets its own subdirectory so the visualise_results.ipynb
    # notebook can discover them via:
    #   RESULTS_DIR/<video_stem>/<video_stem>_inference.json
    #   RESULTS_DIR/<video_stem>/<video_stem>_vis_data.npz
    all_results = []
    for i, vf in enumerate(video_files, 1):
        logger.info(f"\n[{i}/{len(video_files)}] {vf}")
        video_output_dir = os.path.join(args.output_dir, Path(vf).stem)
        result = run_inference_on_video(
            video_path=vf,
            model=model,
            processor=processor,
            helper_mod=helper,
            output_dir=video_output_dir,
            num_traj_samples=args.num_traj_samples,
            device=device,
        )
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────
    summary_file = os.path.join(args.output_dir, "inference_summary.json")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("=" * 80)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"Processed {len(video_files)} video(s)")
    logger.info(f"Summary: {summary_file}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    exit(main())
