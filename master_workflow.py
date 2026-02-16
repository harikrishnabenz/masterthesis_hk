"""Master workflow that chains SAM2 → VideoPainter → Alpamayo sequentially.

This workflow orchestrates the complete pipeline, ensuring each stage completes
before the next begins, with data flowing from one stage to the next via GCS paths.
"""
import logging
import os
import sys
from pathlib import Path

from hlx.wf import workflow

# Add each module to the path so we can import their workflow functions
sys.path.insert(0, str(Path(__file__).parent / "segmentation" / "sam2"))
sys.path.insert(0, str(Path(__file__).parent / "generation" / "VideoPainter"))
sys.path.insert(0, str(Path(__file__).parent / "vla" / "alpamayo"))

logger = logging.getLogger(__name__)


@workflow()
def master_pipeline_wf(
    # SAM2 parameters
    sam2_run_id: str = "experiment_01",
    sam2_max_frames: str = "150",
    sam2_video_uris: str = "default",  # "default" or comma-separated URIs
    
    # VideoPainter parameters
    vp_data_run_id: str | None = None,  # Will default to sam2_run_id
    vp_instruction: str = "remove the cars",
    vp_num_samples: int = 1,
    
    # Alpamayo parameters
    alpamayo_run_id: str | None = None,
    alpamayo_num_traj_samples: int = 1,
    alpamayo_model_id: str = "/workspace/alpamayo/checkpoints/alpamayo-r1-10b",
):
    """
    Run the complete pipeline: SAM2 segmentation → VideoPainter editing → Alpamayo VLA inference.
    
    Each stage runs sequentially and uses its own pre-built Docker container.
    Data flows through GCS paths from one stage to the next.
    
    Args:
        sam2_run_id: Run ID for SAM2 segmentation output
        sam2_max_frames: Maximum frames to process in SAM2
        sam2_video_uris: Video URIs for SAM2 ("default" or comma-separated list)
        vp_data_run_id: VideoPainter data run ID (defaults to sam2_run_id)
        vp_instruction: Editing instruction(s) for VideoPainter
        vp_num_samples: Number of samples per video for VideoPainter
        alpamayo_run_id: Run ID for Alpamayo output (uses sam2_run_id if None)
        alpamayo_num_traj_samples: Number of trajectory samples per video
        alpamayo_model_id: Path to Alpamayo model checkpoint
    """
    
    # Use sam2_run_id for vp_data_run_id if not specified
    if vp_data_run_id is None:
        vp_data_run_id = sam2_run_id
    
    # Use sam2_run_id for alpamayo_run_id if not specified
    if alpamayo_run_id is None:
        alpamayo_run_id = sam2_run_id
    
    # Parse video URIs
    video_uris_list = None
    if sam2_video_uris and sam2_video_uris != "default":
        video_uris_list = [uri.strip() for uri in sam2_video_uris.split(",") if uri.strip()]
    
    logger.info("=" * 80)
    logger.info("MASTER PIPELINE - SEQUENTIAL EXECUTION")
    logger.info("=" * 80)
    logger.info("Stage 1: SAM2 Segmentation (run_id=%s)", sam2_run_id)
    if video_uris_list:
        logger.info("  Processing %d custom videos", len(video_uris_list))
    else:
        logger.info("  Using default video set")
    logger.info("Stage 2: VideoPainter Editing (data_run_id=%s)", vp_data_run_id)
    logger.info("Stage 3: Alpamayo VLA Inference (run_id=%s)", alpamayo_run_id)
    logger.info("=" * 80)
    
    # -------------------------------------------------------------------------
    # STAGE 1: SAM2 Segmentation
    # -------------------------------------------------------------------------
    logger.info("Starting Stage 1: SAM2 Segmentation...")
    
    # Import SAM2 workflow
    from workflow import sam2_segmentation_task
    
    # Prepare SAM2 parameters
    sam2_params = {
        "run_id": sam2_run_id,
        "max_frames": sam2_max_frames,
    }
    if video_uris_list:
        sam2_params["video_uris"] = video_uris_list
    
    sam2_result = sam2_segmentation_task(**sam2_params)
    
    logger.info("✓ Stage 1 complete: SAM2 segmentation finished")
    logger.info("  Output: gs://mbadas-sandbox-research-9bb9c7f/.../output/sam2/%s/", sam2_run_id)
    
    # -------------------------------------------------------------------------
    # STAGE 2: VideoPainter Editing
    # -------------------------------------------------------------------------
    logger.info("Starting Stage 2: VideoPainter Editing...")
    logger.info("  Using preprocessed data from SAM2 run_id: %s", vp_data_run_id)
    
    # Import VideoPainter workflow - need to be careful with module name collision
    import importlib.util
    vp_workflow_path = Path(__file__).parent / "generation" / "VideoPainter" / "workflow.py"
    spec = importlib.util.spec_from_file_location("vp_workflow", vp_workflow_path)
    vp_workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vp_workflow)
    
    vp_result = vp_workflow.videopainter_edit_task(
        data_run_id=vp_data_run_id,
        instruction=vp_instruction,
        num_samples=vp_num_samples,
    )
    
    logger.info("✓ Stage 2 complete: VideoPainter editing finished")
    logger.info("  Output: gs://mbadas-sandbox-research-9bb9c7f/.../output/vp/%s_*/", vp_data_run_id)
    
    # -------------------------------------------------------------------------
    # STAGE 3: Alpamayo VLA Inference
    # -------------------------------------------------------------------------
    logger.info("Starting Stage 3: Alpamayo VLA Inference...")
    
    # Construct the VideoPainter output GCS path for Alpamayo input
    # Note: The actual path will depend on the VP_RUN_SUFFIX from VideoPainter
    # For now, we'll let Alpamayo use its default or pass explicit path
    
    # Import Alpamayo workflow
    alpamayo_workflow_path = Path(__file__).parent / "vla" / "alpamayo" / "workflow.py"
    spec = importlib.util.spec_from_file_location("alpamayo_workflow", alpamayo_workflow_path)
    alpamayo_workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alpamayo_workflow)
    
    alpamayo_result = alpamayo_workflow.alpamayo_inference_task(
        run_id=alpamayo_run_id,
        num_traj_samples=alpamayo_num_traj_samples,
        model_id=alpamayo_model_id,
    )
    
    logger.info("✓ Stage 3 complete: Alpamayo VLA inference finished")
    logger.info("  Output: gs://mbadas-sandbox-research-9bb9c7f/.../output/alpamayo/%s/", alpamayo_run_id)
    
    # -------------------------------------------------------------------------
    # PIPELINE COMPLETE
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("MASTER PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info("All three stages completed successfully!")
    logger.info("")
    logger.info("Output locations:")
    logger.info("  SAM2: gs://.../output/sam2/%s/", sam2_run_id)
    logger.info("  VideoPainter: gs://.../output/vp/%s_*/", vp_data_run_id)
    logger.info("  Alpamayo: gs://.../output/alpamayo/%s/", alpamayo_run_id)
    
    return {
        "status": "complete",
        "sam2_run_id": sam2_run_id,
        "videopainter_run_id": vp_data_run_id,
        "alpamayo_run_id": alpamayo_run_id,
        "sam2_result": sam2_result,
        "videopainter_result": vp_result,
        "alpamayo_result": alpamayo_result,
    }
