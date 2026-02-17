"""Master workflow that chains SAM2 → VideoPainter → Alpamayo sequentially.

This workflow orchestrates the complete pipeline, ensuring each stage completes
before the next begins, with data flowing from one stage to the next via GCS paths.

All task definitions (container images, FuseBuckets, compute nodes) live in
_generated_tasks.py.  This file only wires them together with barriers so
Flytekit respects the SAM2 → VP → Alpamayo ordering.
"""
import logging
from typing import List, Optional

from hlx.wf import ContainerImage, SharedNode, dynamic, task, workflow

# Import pre-built tasks from _generated_tasks – no stub injection needed
from _generated_tasks import (
    task_sam2_segmentation,
    task_videopainter_edit_many,
    task_alpamayo_inference,
    OUTPUT_BASE,
    ALPAMAYO_DEFAULT_MODEL_ID,
    VP_DEFAULT_DATA_RUN_ID,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GCS path constants
# ---------------------------------------------------------------------------
_OUTPUT_BASE = OUTPUT_BASE


# ---------------------------------------------------------------------------
# Barrier tasks – enforce sequential execution via data dependencies.
# Flytekit schedules tasks in parallel unless there is a data dependency.
# These tiny tasks create the chain: SAM2 → VP → Alpamayo.
# ---------------------------------------------------------------------------
@task(compute=SharedNode(), container_image=ContainerImage.PYTHON_3_10.value)
def _barrier(prev_result: str, pass_through: str) -> str:
    """Forces sequential execution: returns pass_through after prev_result completes."""
    return pass_through


@task(compute=SharedNode(), container_image=ContainerImage.PYTHON_3_10.value)
def _build_gcs_path(prev_result: str, base: str, suffix: str) -> str:
    """Build a GCS path after the previous stage completes (enforces ordering)."""
    return f"{base}/{suffix}"


@task(compute=SharedNode(), container_image=ContainerImage.PYTHON_3_10.value)
def _finalize(result: dict) -> str:
    """Convert final stage output to a string.

    Returning this from the @dynamic ensures Flyte tracks the *entire*
    child-task DAG (SAM2 → VP → Alpamayo) as a dependency of the
    dynamic's output.  Without this, `return "Pipeline complete"` is a
    literal that resolves immediately and the engine may never schedule
    the child tasks.
    """
    return f"Pipeline complete: {result}"


# ---------------------------------------------------------------------------
# @dynamic – can use plain Python on its arguments
# ---------------------------------------------------------------------------
@dynamic(
    compute=SharedNode(),
    container_image=ContainerImage.PYTHON_3_10.value,
)
def _master_pipeline_dynamic(
    # SAM2 parameters
    sam2_run_id: str = "001",
    sam2_max_frames: int = 150,
    sam2_video_uris: str = "default",
    # VideoPainter parameters
    vp_data_run_id: str = "",
    vp_output_run_id: str = "",
    vp_instruction: str = "",
    vp_num_samples: int = 1,
    # Alpamayo parameters
    alpamayo_run_id: str = "",
    alpamayo_num_traj_samples: int = 1,
    alpamayo_model_id: str = ALPAMAYO_DEFAULT_MODEL_ID,
) -> str:
    """Chain SAM2 → VideoPainter → Alpamayo sequentially.

    All three stages run as separate HLX tasks in their own GPU containers.
    Data flows between stages via GCS paths.
    """
    # -- resolve defaults that depend on other args ---------------------------
    effective_vp_data_run_id = vp_data_run_id if vp_data_run_id else sam2_run_id
    effective_vp_output_run_id = vp_output_run_id if vp_output_run_id else sam2_run_id
    effective_alpamayo_run_id = alpamayo_run_id if alpamayo_run_id else sam2_run_id

    # -- parse video URIs: chunks://, folder, comma-separated, or "default" --
    video_uris: Optional[List[str]] = None
    if sam2_video_uris and sam2_video_uris != "default":
        if sam2_video_uris.startswith("chunks://"):
            # Chunk spec — pass as single-element list; task resolves it
            video_uris = [sam2_video_uris]
        elif sam2_video_uris.rstrip("/").startswith("gs://") and "," not in sam2_video_uris:
            # Single GCS folder or single file — pass as-is
            video_uris = [sam2_video_uris]
        else:
            # Comma-separated individual URIs
            video_uris = [u.strip() for u in sam2_video_uris.split(",") if u.strip()]

    logger.info("=" * 80)
    logger.info("MASTER PIPELINE – SEQUENTIAL EXECUTION")
    logger.info("=" * 80)
    logger.info("Stage 1: SAM2 Segmentation  (run_id=%s)", sam2_run_id)
    logger.info("Stage 2: VideoPainter Editing (data_run_id=%s, output_run_id=%s)", effective_vp_data_run_id, effective_vp_output_run_id)
    logger.info("Stage 3: Alpamayo VLA Inference (run_id=%s)", effective_alpamayo_run_id)
    logger.info("=" * 80)

    # ── STAGE 1 ── SAM2 Segmentation ────────────────────────────────────────
    sam2_result = task_sam2_segmentation(
        run_id=sam2_run_id,
        video_uris=video_uris,
        max_frames=sam2_max_frames,
    )

    # ── barrier: VP waits for SAM2 ──────────────────────────────────────────
    vp_data_run_id_after_sam2 = _barrier(
        prev_result=sam2_result,
        pass_through=effective_vp_data_run_id,
    )

    # ── STAGE 2 ── VideoPainter Editing ─────────────────────────────────────
    vp_result = task_videopainter_edit_many(
        data_run_id=vp_data_run_id_after_sam2,
        video_editing_instructions=vp_instruction,
        num_videos_per_prompt=vp_num_samples,
    )

    # ── barrier: Alpamayo waits for VP ──────────────────────────────────────
    vp_output_gcs = _build_gcs_path(
        prev_result=vp_result,
        base=f"{_OUTPUT_BASE}/vp",
        suffix=effective_vp_output_run_id,
    )
    alpamayo_run_id_after_vp = _barrier(
        prev_result=vp_result,
        pass_through=effective_alpamayo_run_id,
    )

    # ── STAGE 3 ── Alpamayo VLA Inference ───────────────────────────────────
    alpamayo_result = task_alpamayo_inference(
        video_data_gcs_path=vp_output_gcs,
        output_run_id=alpamayo_run_id_after_vp,
        model_id=alpamayo_model_id,
        num_traj_samples=alpamayo_num_traj_samples,
    )

    # ── Finalize: return a Promise so Flyte tracks the full child-task DAG ──
    return _finalize(result=alpamayo_result)


# ---------------------------------------------------------------------------
# @workflow – the proper entry-point for `hlx wf run`
# A @dynamic cannot be the top-level entry; it must be called from a @workflow.
# ---------------------------------------------------------------------------
@workflow
def master_pipeline_wf(
    # SAM2 parameters
    sam2_run_id: str = "001",
    sam2_max_frames: int = 150,
    sam2_video_uris: str = "default",
    # VideoPainter parameters
    vp_data_run_id: str = "",
    vp_output_run_id: str = "",
    vp_instruction: str = "",
    vp_num_samples: int = 1,
    # Alpamayo parameters
    alpamayo_run_id: str = "",
    alpamayo_num_traj_samples: int = 1,
    alpamayo_model_id: str = ALPAMAYO_DEFAULT_MODEL_ID,
) -> str:
    """Master pipeline workflow: SAM2 → VideoPainter → Alpamayo."""
    return _master_pipeline_dynamic(
        sam2_run_id=sam2_run_id,
        sam2_max_frames=sam2_max_frames,
        sam2_video_uris=sam2_video_uris,
        vp_data_run_id=vp_data_run_id,
        vp_output_run_id=vp_output_run_id,
        vp_instruction=vp_instruction,
        vp_num_samples=vp_num_samples,
        alpamayo_run_id=alpamayo_run_id,
        alpamayo_num_traj_samples=alpamayo_num_traj_samples,
        alpamayo_model_id=alpamayo_model_id,
    )
