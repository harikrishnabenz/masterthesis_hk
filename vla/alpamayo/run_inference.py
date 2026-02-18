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

# The Alpamayo model and physical_ai_av dataset both operate at 10 Hz.
# VideoPainter downsamples source clips (typically 30 fps) with
#   stride = source_fps // down_sample_fps = 30 // 8 = 3
# giving an effective content rate of 30/3 = 10 fps.
# However, VP encodes its output at 8 fps (hardcoded), so playing the
# VP video at face value yields 20-25 % slow motion.
# We always use the true content rate (0.1 s = 10 Hz) for both the
# dataset ego-trajectory alignment AND the output video frame rate.
CONTENT_TIME_STEP = 0.1      # seconds  (10 Hz — dataset / model native rate)
CONTENT_FPS = 1.0 / CONTENT_TIME_STEP  # 10.0 fps


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


def _run_model_inference(data, model, processor, helper_mod, device, num_traj_samples):
    """Run Alpamayo forward pass and return (pred_xyz, pred_rot, extra).

    This is a stateless helper so we can call it on both the original and
    the VP-modified data without duplicating the model-input construction.
    """
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
    return pred_xyz, pred_rot, extra


def _compute_min_ade(pred_xyz, gt_future_xyz):
    """Compute minADE (meters) between predicted and GT trajectories."""
    gt_xy = gt_future_xyz.cpu()[0, 0, :, :2].T.numpy()       # (2, T)
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)  # (S, 2, T)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)   # (S,)
    return float(diff.min())


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
    """Run Alpamayo inference on one VideoPainter-edited video.

    Runs inference **twice** — once on the original (unmodified) dataset
    frames and once on the VP-modified frames — so that the two predictions
    can be compared side-by-side.
    """
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
        vp_h_native, vp_w_native = vp_frames.shape[-2], vp_frames.shape[-1]  # save before resize

        # 3. Compute t0_us from VP video for temporal alignment.
        #
        #    VP downsamples the source clip (typically 30 fps) with
        #    stride = source_fps // 8 = 3, giving an effective content
        #    rate of 10 fps.  The output is encoded at 8 fps, but each
        #    frame actually represents 0.1 s of real time.
        #
        #    The model / dataset operate at 10 Hz (time_step = 0.1 s),
        #    so we use CONTENT_TIME_STEP for both t0 and the dataset
        #    load.  This also avoids the ~25 % slow-motion artefact
        #    caused by encoding 10 fps content at 8 fps.
        #
        #    t0_us = (vp_total_frames − 1) × 0.1 s × 1e6
        t0_us = int((vp_total_frames - 1) * CONTENT_TIME_STEP * 1_000_000)
        logger.info(
            f"  VP video: {vp_total_frames} frames, encoded at {vp_fps:.1f} fps, "
            f"content rate = {CONTENT_FPS:.0f} fps (time_step={CONTENT_TIME_STEP}s)"
        )
        logger.info(
            f"  Computed t0_us={t0_us} ({t0_us / 1_000_000:.3f}s) from "
            f"last VP frame index ({vp_total_frames - 1}) × "
            f"content_time_step ({CONTENT_TIME_STEP}s)"
        )

        # 3b. Load the FULL frame sequence from physical_ai_av so that
        #     the comparison video has 1-to-1 original frames for every
        #     VP frame.  The model only needs the last 4 frames per
        #     camera, but we load all vp_total_frames so the target
        #     camera's original footage is available for visualisation.
        logger.info(
            f"  Loading original clip data from physical_ai_av "
            f"({vp_total_frames} frames @ {CONTENT_TIME_STEP}s) …"
        )
        data = load_physical_aiavdataset(
            clip_id,
            t0_us=t0_us,
            time_step=CONTENT_TIME_STEP,
            num_frames=vp_total_frames,  # full sequence for comparison
        )
        # data["image_frames"]: (N_cameras, vp_total_frames, 3, H, W)

        # 4. Determine which camera index to replace
        cam_idx_value = CAMERA_NAME_TO_INDEX[camera_name]
        # data["camera_indices"] is sorted; find position
        positions = (data["camera_indices"] == cam_idx_value).nonzero(as_tuple=True)[0]

        # ── Save full original frames for the target camera ──────────
        orig_camera_full_frames = None   # (vp_total_frames, 3, H, W)
        if len(positions) > 0:
            pos = positions[0].item()
            orig_camera_full_frames = data["image_frames"][pos].clone()

        # ── Trim to last `num_frames` for inference ──────────────────
        #    The model expects exactly 4 frames per camera.
        data["image_frames"] = data["image_frames"][:, -num_frames:]
        if "relative_timestamps" in data:
            data["relative_timestamps"] = data["relative_timestamps"][:, -num_frames:]
        if "absolute_timestamps" in data:
            data["absolute_timestamps"] = data["absolute_timestamps"][:, -num_frames:]
        # data["image_frames"]: (N_cameras, 4, 3, H, W) — inference window

        # ── 4a. ORIGINAL inference (before VP replacement) ───────────
        logger.info("  ── Running inference on ORIGINAL frames ──")
        reset_gpu_memory_stats()
        orig_pred_xyz, orig_pred_rot, orig_extra = _run_model_inference(
            data, model, processor, helper_mod, device, num_traj_samples,
        )
        orig_min_ade = _compute_min_ade(orig_pred_xyz, data["ego_future_xyz"])
        logger.info(f"  Original minADE = {orig_min_ade:.4f} m")

        # ── 4b. Replace camera frames with VP output ─────────────────
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
            data["image_frames"][pos] = vp_frames  # replaces last-4 slice
            logger.info(f"  Replaced camera {camera_name} frames with VideoPainter output")

        # ── 4c. GENERATED inference (after VP replacement) ───────────
        logger.info("  ── Running inference on GENERATED (VP) frames ──")
        reset_gpu_memory_stats()
        pred_xyz, pred_rot, extra = _run_model_inference(
            data, model, processor, helper_mod, device, num_traj_samples,
        )

        # 7. Compute minADE
        min_ade = _compute_min_ade(pred_xyz, data["ego_future_xyz"])

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

        orig_cot_texts = orig_extra.get("cot", [[[""]]])[0]
        orig_cot_flat = []
        for traj_set in orig_cot_texts:
            for text in traj_set:
                orig_cot_flat.append(str(text))

        results = {
            "video_id": video_id,
            "video_path": video_path,
            "clip_id": clip_id,
            "camera_name": camera_name,
            "num_trajectories": num_traj_samples,
            "min_ade_meters": min_ade,
            "original_min_ade_meters": orig_min_ade,
            "reasoning_traces": cot_flat,
            "original_reasoning_traces": orig_cot_flat,
            "temporal_config": {
                "vp_video_encoded_fps": vp_fps,
                "content_fps": CONTENT_FPS,
                "content_time_step_seconds": CONTENT_TIME_STEP,
                "vp_total_frames": vp_total_frames,
                "t0_us": t0_us,
                "t0_seconds": t0_us / 1_000_000,
                "num_frames_per_camera": num_frames,
                "frame_window_seconds": (num_frames - 1) * CONTENT_TIME_STEP,
                "image_frame_timestamps_s": [
                    (t0_us - (num_frames - 1 - i) * int(CONTENT_TIME_STEP * 1_000_000)) / 1_000_000
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
                "pred_xyz": pred_xyz.cpu().numpy()[0, 0],       # (S, 64, 3)  — VP inference
                "pred_rot": pred_rot.cpu().numpy()[0, 0],       # (S, 64, 3, 3)
                "orig_pred_xyz": orig_pred_xyz.cpu().numpy()[0, 0],  # (S, 64, 3) — original inference
                "orig_pred_rot": orig_pred_rot.cpu().numpy()[0, 0],  # (S, 64, 3, 3)
                "gt_future_xyz": data["ego_future_xyz"].cpu().numpy()[0, 0],   # (64, 3)
                "gt_future_rot": data["ego_future_rot"].cpu().numpy()[0, 0],   # (64, 3, 3)
                "ego_history_xyz": data["ego_history_xyz"].cpu().numpy()[0, 0],  # (16, 3)
                "ego_history_rot": data["ego_history_rot"].cpu().numpy()[0, 0],  # (16, 3, 3)
                "image_frames": image_frames_np,                # (N_cam*T, H, W, 3)
                "camera_indices": data["camera_indices"].cpu().numpy(),  # (N_cam,)
            }
            # Save full original camera frames for the target camera
            # (all vp_total_frames, resized to VP resolution to keep NPZ small)
            if orig_camera_full_frames is not None:
                orig_np = orig_camera_full_frames.cpu().numpy()  # (vp_total_frames, 3, H, W)
                orig_np = orig_np.transpose(0, 2, 3, 1)         # (vp_total_frames, H, W, 3)
                # Resize to VP video resolution to avoid huge NPZ files
                dataset_h, dataset_w = orig_np.shape[1], orig_np.shape[2]
                if (dataset_h, dataset_w) != (vp_h_native, vp_w_native):
                    import cv2 as _cv2
                    resized = np.empty((orig_np.shape[0], vp_h_native, vp_w_native, 3), dtype=np.uint8)
                    for fi in range(orig_np.shape[0]):
                        resized[fi] = _cv2.resize(orig_np[fi], (vp_w_native, vp_h_native))
                    orig_np = resized
                vis_data["orig_camera_frames"] = orig_np  # (vp_total_frames, H_vp, W_vp, 3)
            np.savez_compressed(vis_file, **vis_data)
            logger.info(f"  Visualization data saved → {vis_file}")
        except Exception as vis_err:
            logger.warning(f"  Failed to save visualization data: {vis_err}")

        # 10. Render overlay video (trajectories on generated video)
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
                camera_name=camera_name,
                output_fps=CONTENT_FPS,   # 10 fps (true content rate)
            )
            results["overlay_video_path"] = overlay_file
            logger.info(f"  Overlay video saved → {overlay_file}")
        except Exception as overlay_err:
            logger.error(f"  Failed to render overlay video: {overlay_err}", exc_info=True)

        # 11. Render side-by-side comparison video (original vs generated)
        comparison_file = os.path.join(output_dir, f"{video_id}_comparison.mp4")
        try:
            from visualize_video import render_comparison_video

            fov = 120.0 if "120fov" in camera_name else (30.0 if "30fov" in camera_name else 70.0)
            render_comparison_video(
                generated_video_path=video_path,
                npz_path=vis_file,
                output_path=comparison_file,
                json_path=output_file,
                fov_deg=fov,
                camera_name=camera_name,
                output_fps=CONTENT_FPS,   # 10 fps (true content rate)
            )
            results["comparison_video_path"] = comparison_file
            logger.info(f"  Comparison video saved → {comparison_file}")
        except Exception as comp_err:
            logger.error(f"  Failed to render comparison video: {comp_err}", exc_info=True)

        logger.info(
            f"  Generated minADE = {min_ade:.4f} m | "
            f"Original minADE = {orig_min_ade:.4f} m | "
            f"time = {inference_time:.1f}s | saved → {output_file}"
        )
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
