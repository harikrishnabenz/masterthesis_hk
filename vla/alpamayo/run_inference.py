#!/usr/bin/env python3
"""
Alpamayo VLA Inference Runner

This script runs Alpamayo-R1-10B inference on driving videos and generates
trajectory predictions with reasoning traces.
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_gpu_memory_gb() -> tuple[float, float]:
    """Get current and peak GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    current = torch.cuda.memory_allocated() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return current, peak


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_ram_mb() -> float:
    """Get current RAM usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def run_inference_on_video(
    video_path: str,
    model,
    tokenizer,
    output_dir: str,
    num_traj_samples: int = 1,
) -> dict:
    """
    Run Alpamayo inference on a single video.
    
    Args:
        video_path: Path to input video
        model: Loaded Alpamayo model
        tokenizer: Tokenizer
        output_dir: Output directory for results
        num_traj_samples: Number of trajectory samples to generate
    
    Returns:
        Dictionary with inference results and metrics
    """
    video_id = Path(video_path).stem
    logger.info(f"Processing video: {video_id}")
    
    # Reset GPU stats
    reset_gpu_memory_stats()
    ram_start = get_ram_mb()
    
    start_time = time.time()
    
    try:
        # TODO: Implement actual Alpamayo inference
        # This is a placeholder - you'll need to adapt based on the actual
        # Alpamayo API from src/alpamayo_r1/test_inference.py
        
        logger.info(f"Running inference with {num_traj_samples} trajectory samples")
        
        # Placeholder for actual inference
        # In reality, you would:
        # 1. Load video frames
        # 2. Process with Alpamayo model
        # 3. Get trajectory predictions and reasoning traces
        
        trajectories = []
        reasoning_traces = []
        
        for i in range(num_traj_samples):
            # Placeholder trajectory (64 waypoints)
            traj = np.random.randn(64, 2).tolist()
            trajectories.append(traj)
            
            # Placeholder reasoning
            reasoning = f"Sample {i+1}: Vehicle maintains lane, no obstacles detected."
            reasoning_traces.append(reasoning)
        
        inference_time = time.time() - start_time
        gpu_current, gpu_peak = get_gpu_memory_gb()
        ram_current = get_ram_mb()
        
        # Save results
        output_file = os.path.join(output_dir, f"{video_id}_inference.json")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {
            "video_id": video_id,
            "video_path": video_path,
            "num_trajectories": len(trajectories),
            "trajectories": trajectories,
            "reasoning_traces": reasoning_traces,
            "metrics": {
                "inference_time_seconds": inference_time,
                "gpu_memory_used_gb": gpu_current,
                "gpu_memory_peak_gb": gpu_peak,
                "ram_used_mb": ram_current,
                "ram_peak_mb": ram_current,  # Approximate
            },
            "success": True,
        }
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Inference time: {inference_time:.2f}s")
        logger.info(f"GPU memory peak: {gpu_peak:.2f} GB")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        
        inference_time = time.time() - start_time
        gpu_current, gpu_peak = get_gpu_memory_gb()
        
        return {
            "video_id": video_id,
            "video_path": video_path,
            "num_trajectories": 0,
            "trajectories": [],
            "reasoning_traces": [],
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


def main():
    parser = argparse.ArgumentParser(
        description="Run Alpamayo VLA inference on driving videos"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video or directory of videos"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="nvidia/Alpamayo-R1-10B",
        help="Model ID (HuggingFace or local path)"
    )
    parser.add_argument(
        "--num_traj_samples",
        type=int,
        default=1,
        help="Number of trajectory samples per video"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference (cuda:0, cpu, or auto)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("ALPAMAYO VLA INFERENCE")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Input: {args.video_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Num trajectory samples: {args.num_traj_samples}")
    logger.info(f"Device: {args.device}")
    
    # Load model
    logger.info("Loading model...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=args.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=True
        )
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Find video files
    video_path = Path(args.video_path)
    if video_path.is_file():
        video_files = [str(video_path)]
    elif video_path.is_dir():
        video_files = []
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            video_files.extend([str(p) for p in video_path.rglob(f"*{ext}")])
    else:
        logger.error(f"Invalid video path: {args.video_path}")
        return 1
    
    if not video_files:
        logger.error(f"No video files found at: {args.video_path}")
        return 1
    
    logger.info(f"Found {len(video_files)} video(s) to process")
    
    # Run inference on each video
    all_results = []
    for i, video_file in enumerate(video_files, 1):
        logger.info(f"\nProcessing {i}/{len(video_files)}: {video_file}")
        
        result = run_inference_on_video(
            video_path=video_file,
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            num_traj_samples=args.num_traj_samples,
        )
        all_results.append(result)
    
    # Save summary
    summary_file = os.path.join(args.output_dir, "inference_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"Processed {len(video_files)} video(s)")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Summary: {summary_file}")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
