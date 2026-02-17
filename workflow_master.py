"""Master Pipeline Orchestrator: SAM2 → VideoPainter → Alpamayo.

Chains three GPU stages sequentially inside a single HLX workflow:

  Stage 1 — SAM2 Segmentation
      Input : raw driving videos (GCS / chunks:// URI)
      Output: binary masks + VideoPainter-preprocessed data
              gs://<bucket>/.../outputs/preprocessed_data_vp/<run_id>/

  Stage 2 — VideoPainter Editing
      Input : SAM2 preprocessed data (masks + raw videos)
      Output: edited / inpainted videos
              gs://<bucket>/.../outputs/vp/<run_id>/

  Stage 3 — Alpamayo VLA Inference
      Input : VideoPainter edited videos
      Output: trajectory predictions + reasoning traces
              gs://<bucket>/.../outputs/alpamayo/<run_id>/

All three stages share a single ``run_id`` for end-to-end traceability.
Each stage runs in its **own** container image (heavy ML deps are isolated)
while the workflow graph itself is serialised from this lightweight
orchestrator container.

Usage (via the master build_and_run.sh):
    bash scripts/build_and_run.sh
"""

import logging
import os
from typing import Optional

from hlx.wf import DedicatedNode, Node, task, workflow
from hlx.wf.mounts import MOUNTPOINT, FuseBucket

logger = logging.getLogger(__name__)

# ==============================================================================
# GCS BUCKET (shared across all stages)
# ==============================================================================
GCS_BUCKET = "mbadas-sandbox-research-9bb9c7f"

# ==============================================================================
# CONTAINER IMAGES  — set by scripts/build_and_run.sh before `hlx wf run`
# Image names include the run ID: <base>_<run_id>:<run_id>
# ==============================================================================
SAM2_CONTAINER_IMAGE = os.environ.get(
    "SAM2_CONTAINER_IMAGE",
    "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2:latest",
)
VP_CONTAINER_IMAGE = os.environ.get(
    "VP_CONTAINER_IMAGE",
    "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_vp:latest",
)
ALPAMAYO_CONTAINER_IMAGE = os.environ.get(
    "ALPAMAYO_CONTAINER_IMAGE",
    "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/alpamayo_vla:latest",
)

# ==============================================================================
# STAGE 1 — SAM2 GCS / mount configuration
# ==============================================================================
SAM2_CKPT_PREFIX = "workspace/user/hbaskar/Video_inpainting/sam2_checkpoint"
SAM2_FUSE_NAME   = "sam2-checkpoints"

SAM2_OUTPUT_BASE = os.environ.get(
    "SAM2_OUTPUT_BASE",
    f"gs://{GCS_BUCKET}/workspace/user/hbaskar/outputs/sam2",
)
SAM2_PREPROCESSED_OUTPUT_BASE = os.environ.get(
    "SAM2_PREPROCESSED_OUTPUT_BASE",
    f"gs://{GCS_BUCKET}/workspace/user/hbaskar/outputs/preprocessed_data_vp",
)

# ==============================================================================
# STAGE 2 — VideoPainter GCS / mount configuration
# ==============================================================================
VP_BUCKET_PREFIX      = "workspace/user/hbaskar/Video_inpainting/videopainter"
VP_DATA_PREFIX        = "workspace/user/hbaskar/outputs/preprocessed_data_vp"
VP_FUSE_NAME          = "vp-bucket"
VP_DATA_FUSE_NAME     = "data"
VP_VLM_7B_FUSE_NAME   = "vp-vlm-7b"
VLM_7B_GCS_PREFIX     = os.path.join(VP_BUCKET_PREFIX, "ckpt", "vlm", "Qwen2.5-VL-7B-Instruct")

TRAINED_FLUXFILL_FUSE_NAME = "vp-trained-fluxfill"
TRAINED_FLUXFILL_GCS_PREFIX = os.environ.get(
    "TRAINED_FLUXFILL_GCS_PATH",
    "workspace/user/hbaskar/Video_inpainting/videopainter/training/"
    "trained_checkpoint/fluxfill_single_white_solid_clearroad_20260212_151908",
)

VP_OUTPUT_BASE = os.environ.get(
    "VP_OUTPUT_BASE",
    f"gs://{GCS_BUCKET}/workspace/user/hbaskar/outputs/vp",
)

# ==============================================================================
# STAGE 3 — Alpamayo GCS / mount configuration
# ==============================================================================
VLA_BASE_PREFIX        = "workspace/user/hbaskar/Video_inpainting/vla"
ALPAMAYO_CKPT_PREFIX   = os.path.join(VLA_BASE_PREFIX, "alpamayo", "checkpoints")
ALPAMAYO_CKPT_FUSE     = "alpamayo-ckpt"
ALPAMAYO_DATA_FUSE     = "alpamayo-video-data"
ALPAMAYO_DATA_PREFIX   = "workspace/user/hbaskar/outputs/vp"

ALPAMAYO_OUTPUT_BASE = os.environ.get(
    "ALPAMAYO_OUTPUT_BASE",
    "workspace/user/hbaskar/outputs/alpamayo",
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1  —  SAM2 Segmentation
# ══════════════════════════════════════════════════════════════════════════════
@task(
    compute=DedicatedNode(
        node=Node.A100_80GB_1GPU,
        ephemeral_storage="max",
        max_duration="3d",
    ),
    container_image=SAM2_CONTAINER_IMAGE,
    environment={
        "PYTHONUNBUFFERED": "1",
        "SAM2_OUTPUT_BASE": SAM2_OUTPUT_BASE,
        "SAM2_PREPROCESSED_OUTPUT_BASE": SAM2_PREPROCESSED_OUTPUT_BASE,
    },
    mounts=[
        FuseBucket(
            bucket=GCS_BUCKET,
            name=SAM2_FUSE_NAME,
            prefix=SAM2_CKPT_PREFIX,
        ),
    ],
)
def sam2_stage(
    run_id: str,
    sam2_video_uris: str,
    max_frames: int = 150,
) -> str:
    """Run SAM2 video segmentation (Stage 1 of 3).

    Produces binary masks and VideoPainter-preprocessed data under
    ``gs://…/outputs/preprocessed_data_vp/<run_id>/``.

    Returns
    -------
    str
        The *run_id* — passed to Stage 2 as ``data_run_id``.
    """
    import sys
    sys.path.insert(0, "/workspace/sam2")
    os.chdir("/workspace/sam2")

    # Import the SAM2 task function from the container's baked-in workflow module.
    # Calling a @task-decorated function directly (outside a @workflow context)
    # executes the raw Python function — no Flyte magic.
    from workflow_sam2 import run_sam2_segmentation  # type: ignore[import-not-found]

    logger.info("=" * 80)
    logger.info("MASTER PIPELINE — STAGE 1: SAM2 SEGMENTATION")
    logger.info("  run_id            = %s", run_id)
    logger.info("  sam2_video_uris   = %s", sam2_video_uris)
    logger.info("  max_frames        = %d", max_frames)
    logger.info("=" * 80)

    result = run_sam2_segmentation(
        run_id=run_id,
        sam2_video_uris=sam2_video_uris,
        max_frames=max_frames,
    )

    logger.info("SAM2 stage completed.")
    logger.info("  Summary (first 500 chars): %s", str(result)[:500])
    return run_id


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2  —  VideoPainter Editing
# ══════════════════════════════════════════════════════════════════════════════
@task(
    compute=DedicatedNode(
        node=Node.A100_80GB_1GPU,
        ephemeral_storage="max",
        max_duration="3d",
    ),
    container_image=VP_CONTAINER_IMAGE,
    environment={
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VP_COG_DEVICE": "cuda:0",
        "VP_FLUX_DEVICE": "cuda:0",
        "VP_QWEN_DEVICE": "auto",
        "VP_UNLOAD_QWEN_AFTER_USE": "1",
        "TRAINED_FLUXFILL_GCS_PATH": TRAINED_FLUXFILL_GCS_PREFIX,
        "VP_OUTPUT_BASE": VP_OUTPUT_BASE,
    },
    mounts=[
        FuseBucket(bucket=GCS_BUCKET, name=VP_FUSE_NAME,          prefix=VP_BUCKET_PREFIX),
        FuseBucket(bucket=GCS_BUCKET, name=VP_DATA_FUSE_NAME,     prefix=VP_DATA_PREFIX),
        FuseBucket(bucket=GCS_BUCKET, name=TRAINED_FLUXFILL_FUSE_NAME,
                   prefix=TRAINED_FLUXFILL_GCS_PREFIX),
        FuseBucket(bucket=GCS_BUCKET, name=VP_VLM_7B_FUSE_NAME,  prefix=VLM_7B_GCS_PREFIX),
    ],
)
def vp_stage(
    data_run_id: str,
    output_run_id: str,
    video_editing_instructions: str = "",
    llm_model: str = "/workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct",
    num_inference_steps: int = 70,
    guidance_scale: float = 6.0,
    strength: float = 1.0,
    caption_refine_iters: int = 10,
    caption_refine_temperature: float = 0.1,
    dilate_size: int = 24,
    mask_feather: int = 8,
    keep_masked_pixels: bool = True,
    img_inpainting_lora_scale: float = 0.0,
    seed: int = 42,
) -> str:
    """Run VideoPainter video editing / inpainting (Stage 2 of 3).

    Reads SAM2-preprocessed data (``data_run_id``) and produces edited videos
    under ``gs://…/outputs/vp/<output_run_id>/``.

    Returns
    -------
    str
        GCS path to the edited videos (input for Stage 3).
    """
    import sys
    sys.path.insert(0, "/workspace/VideoPainter")
    os.chdir("/workspace/VideoPainter")

    from workflow_vp import run_videopainter_edit_many  # type: ignore[import-not-found]

    logger.info("=" * 80)
    logger.info("MASTER PIPELINE — STAGE 2: VIDEOPAINTER EDITING")
    logger.info("  data_run_id       = %s", data_run_id)
    logger.info("  output_run_id     = %s", output_run_id)
    logger.info("  instructions      = %s", video_editing_instructions[:200])
    logger.info("=" * 80)

    gcs_output_path = run_videopainter_edit_many(
        data_run_id=data_run_id,
        output_run_id=output_run_id,
        data_video_ids="auto",
        inpainting_sample_id=0,
        model_path="/workspace/VideoPainter/ckpt/CogVideoX-5b-I2V",
        inpainting_branch="/workspace/VideoPainter/ckpt/VideoPainter/checkpoints/branch",
        img_inpainting_model="/workspace/VideoPainter/ckpt/flux_inp",
        img_inpainting_lora_path="/workspace/VideoPainter/ckpt/trained_fluxfill_lora",
        img_inpainting_lora_scale=img_inpainting_lora_scale,
        output_name_suffix="vp_edit_sample0.mp4",
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        down_sample_fps=8,
        inpainting_frames=49,
        video_editing_instructions=video_editing_instructions,
        llm_model=llm_model,
        caption_refine_iters=caption_refine_iters,
        caption_refine_temperature=caption_refine_temperature,
        dilate_size=dilate_size,
        mask_feather=mask_feather,
        keep_masked_pixels=keep_masked_pixels,
        seed=seed,
    )

    logger.info("VideoPainter stage completed.  Output: %s", gcs_output_path)
    return gcs_output_path


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3  —  Alpamayo VLA Inference
# ══════════════════════════════════════════════════════════════════════════════
@task(
    compute=DedicatedNode(
        node=Node.A100_80GB_1GPU,
        ephemeral_storage="max",
        max_duration="3d",
    ),
    container_image=ALPAMAYO_CONTAINER_IMAGE,
    environment={
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/root/.cache/huggingface",
        "HF_TOKEN": HF_TOKEN,
    },
    mounts=[
        FuseBucket(bucket=GCS_BUCKET, name=ALPAMAYO_CKPT_FUSE,  prefix=ALPAMAYO_CKPT_PREFIX),
        FuseBucket(bucket=GCS_BUCKET, name=ALPAMAYO_DATA_FUSE,  prefix=ALPAMAYO_DATA_PREFIX),
    ],
)
def alpamayo_stage(
    video_data_gcs_path: str,
    output_run_id: str,
    model_id: str = "/workspace/alpamayo/checkpoints/alpamayo-r1-10b",
    num_traj_samples: int = 1,
    video_name: str = "auto",
) -> str:
    """Run Alpamayo VLA inference (Stage 3 of 3).

    Reads VideoPainter output videos and produces trajectory predictions
    under ``gs://…/outputs/alpamayo/<output_run_id>/``.

    Returns
    -------
    str
        GCS path to the Alpamayo output directory.
    """
    import sys
    sys.path.insert(0, "/workspace/alpamayo")
    os.chdir("/workspace/alpamayo")

    from workflow_alpamayo import run_alpamayo_inference_task  # type: ignore[import-not-found]

    logger.info("=" * 80)
    logger.info("MASTER PIPELINE — STAGE 3: ALPAMAYO VLA INFERENCE")
    logger.info("  video_data_gcs_path = %s", video_data_gcs_path)
    logger.info("  output_run_id       = %s", output_run_id)
    logger.info("  model_id            = %s", model_id)
    logger.info("  video_name          = %s", video_name)
    logger.info("=" * 80)

    result = run_alpamayo_inference_task(
        video_data_gcs_path=video_data_gcs_path,
        output_run_id=output_run_id,
        model_id=model_id,
        num_traj_samples=num_traj_samples,
        video_name=video_name,
    )

    gcs_path = result.get("output_gcs_path", "")
    logger.info("Alpamayo stage completed.  Output: %s", gcs_path)
    return gcs_path


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════
@workflow
def master_pipeline_wf(
    run_id: str,
    # ── Stage 1: SAM2 ────────────────────────────────────────────────────────
    sam2_video_uris: str,
    sam2_max_frames: int = 150,
    # ── Stage 2: VideoPainter ─────────────────────────────────────────────────
    vp_video_editing_instructions: str = (
        "Single solid white continuous line, aligned exactly to the original "
        "lane positions and perspective; keep road texture, lighting, and "
        "shadows unchanged\n"
        "Double solid white continuous line, aligned exactly to the original "
        "lane positions and perspective; keep road texture, lighting, and "
        "shadows unchanged\n"
        "Single solid yellow continuous line, aligned exactly to the original "
        "lane positions and perspective; keep road texture, lighting, and "
        "shadows unchanged\n"
        "Double solid yellow continuous line, aligned exactly to the original "
        "lane positions and perspective; keep road texture, lighting, and "
        "shadows unchanged\n"
        "Single dashed white intermitted line, aligned exactly to the original "
        "lane positions and perspective; keep road texture, lighting, and "
        "shadows unchanged"
    ),
    vp_llm_model: str = "/workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct",
    vp_num_inference_steps: int = 70,
    vp_guidance_scale: float = 6.0,
    vp_strength: float = 1.0,
    vp_caption_refine_iters: int = 10,
    vp_caption_refine_temperature: float = 0.1,
    vp_dilate_size: int = 24,
    vp_mask_feather: int = 8,
    vp_keep_masked_pixels: bool = True,
    vp_img_inpainting_lora_scale: float = 0.0,
    vp_seed: int = 42,
    # ── Stage 3: Alpamayo ─────────────────────────────────────────────────────
    alp_model_id: str = "/workspace/alpamayo/checkpoints/alpamayo-r1-10b",
    alp_num_traj_samples: int = 1,
    alp_video_name: str = "auto",
) -> str:
    """End-to-end pipeline: SAM2 → VideoPainter → Alpamayo.

    A single ``run_id`` propagates through every stage so that GCS artefacts
    are co-located and easy to trace:

    =========  ====================================================
    Stage      GCS output
    =========  ====================================================
    SAM2       ``gs://…/outputs/sam2/<run_id>/``
    SAM2 (VP)  ``gs://…/outputs/preprocessed_data_vp/<run_id>/``
    VP         ``gs://…/outputs/vp/<run_id>/``
    Alpamayo   ``gs://…/outputs/alpamayo/<run_id>/``
    =========  ====================================================

    Data-dependency edges ensure strict sequential execution:

      SAM2 ──(run_id)──▶ VP ──(gcs_path)──▶ Alpamayo
    """
    # Stage 1 — SAM2 Segmentation
    # Returns the run_id; the promise creates the dependency for Stage 2.
    sam2_run_id = sam2_stage(
        run_id=run_id,
        sam2_video_uris=sam2_video_uris,
        max_frames=sam2_max_frames,
    )

    # Stage 2 — VideoPainter Editing
    # data_run_id=sam2_run_id ensures VP waits for SAM2 to finish.
    vp_output_path = vp_stage(
        data_run_id=sam2_run_id,
        output_run_id=run_id,
        video_editing_instructions=vp_video_editing_instructions,
        llm_model=vp_llm_model,
        num_inference_steps=vp_num_inference_steps,
        guidance_scale=vp_guidance_scale,
        strength=vp_strength,
        caption_refine_iters=vp_caption_refine_iters,
        caption_refine_temperature=vp_caption_refine_temperature,
        dilate_size=vp_dilate_size,
        mask_feather=vp_mask_feather,
        keep_masked_pixels=vp_keep_masked_pixels,
        img_inpainting_lora_scale=vp_img_inpainting_lora_scale,
        seed=vp_seed,
    )

    # Stage 3 — Alpamayo VLA Inference
    # video_data_gcs_path=vp_output_path ensures Alpamayo waits for VP.
    alp_output_path = alpamayo_stage(
        video_data_gcs_path=vp_output_path,
        output_run_id=run_id,
        model_id=alp_model_id,
        num_traj_samples=alp_num_traj_samples,
        video_name=alp_video_name,
    )

    return alp_output_path
