#!/bin/bash
# ==================================================================================
# MASTER PIPELINE — Build and Run Script
# ==================================================================================
#
# Orchestrates the full three-stage pipeline in a single invocation:
#
#   Stage 1  SAM2 Segmentation
#            Input : raw driving videos (GCS / chunks:// URI)
#            Output: masks + VP-preprocessed data
#
#   Stage 2  VideoPainter Editing
#            Input : SAM2 preprocessed data
#            Output: inpainted / edited videos
#
#   Stage 3  Alpamayo VLA Inference
#            Input : VP edited videos
#            Output: trajectory predictions + reasoning
#
# A single RUN_ID + RUN_TIMESTAMP is shared across all stages.
#
# Usage:
#   bash scripts/build_and_run.sh                      # defaults
#   RUN_ID=002 bash scripts/build_and_run.sh           # custom run id
#   SAM2_CHUNK_END=9 bash scripts/build_and_run.sh     # fewer chunks
#   SKIP_BUILD=1 bash scripts/build_and_run.sh         # reuse images
# ==================================================================================

set -euo pipefail

# ── Resolve repo root (this script lives in scripts/) ────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ==============================================================================
# SHARED CONFIGURATION
# ==============================================================================
GCS_BUCKET="mbadas-sandbox-research-9bb9c7f"

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${RUN_ID:-002}"
MASTER_RUN_ID="${RUN_ID}_${RUN_TIMESTAMP}"

# Skip docker builds (reuse previously pushed images)
SKIP_BUILD="${SKIP_BUILD:-0}"

# ==============================================================================
# STAGE 1: SAM2 CONFIGURATION
# ==============================================================================
SAM2_INPUT_PARENT="${SAM2_INPUT_PARENT:-gs://${GCS_BUCKET}/workspace/user/hbaskar/Input/data_physical_ai}"
SAM2_CAMERA_SUBFOLDER="${SAM2_CAMERA_SUBFOLDER:-camera_front_tele_30fov}"
SAM2_INPUT_BASE="${SAM2_INPUT_BASE:-${SAM2_INPUT_PARENT}/${SAM2_CAMERA_SUBFOLDER}}"
SAM2_MAX_FRAMES="${SAM2_MAX_FRAMES:-150}"

# Chunk-based input selection
SAM2_CHUNK_START="${SAM2_CHUNK_START:-0}"
SAM2_CHUNK_END="${SAM2_CHUNK_END:-19}"
SAM2_FILES_PER_CHUNK="${SAM2_FILES_PER_CHUNK:-1}"

SAM2_VIDEO_URIS="${SAM2_VIDEO_URIS:-chunks://${SAM2_INPUT_BASE#gs://}?start=${SAM2_CHUNK_START}&end=${SAM2_CHUNK_END}&per_chunk=${SAM2_FILES_PER_CHUNK}}"

SAM2_OUTPUT_BASE="${SAM2_OUTPUT_BASE:-gs://${GCS_BUCKET}/workspace/user/hbaskar/outputs/sam2}"
SAM2_PREPROCESSED_OUTPUT_BASE="${SAM2_PREPROCESSED_OUTPUT_BASE:-gs://${GCS_BUCKET}/workspace/user/hbaskar/outputs/preprocessed_data_vp}"

# Calculate expected video count for display
SAM2_TOTAL_CHUNKS=$(( SAM2_CHUNK_END - SAM2_CHUNK_START + 1 ))
SAM2_EXPECTED_VIDEOS=$(( SAM2_TOTAL_CHUNKS * SAM2_FILES_PER_CHUNK ))

# ==============================================================================
# STAGE 2: VIDEOPAINTER CONFIGURATION
# ==============================================================================
VP_OUTPUT_BASE="${VP_OUTPUT_BASE:-gs://${GCS_BUCKET}/workspace/user/hbaskar/outputs/vp}"

# Trained FluxFill LoRA checkpoint in GCS
TRAINED_FLUXFILL_GCS_PATH="${TRAINED_FLUXFILL_GCS_PATH:-workspace/user/hbaskar/Video_inpainting/videopainter/training/trained_checkpoint/fluxfill_single_white_solid_clearroad_20260212_151908}"

# VP run suffix (used in Docker image naming)
CHECKPOINT_TIMESTAMP=$(basename "$TRAINED_FLUXFILL_GCS_PATH" | grep -oE '[0-9]{8}_[0-9]{6}' | head -1 || true)
VP_RUN_SUFFIX="${VP_RUN_SUFFIX:-withoutlora_5prompt_${CHECKPOINT_TIMESTAMP}}"

# LLM model path inside VP container
VP_LLM_MODEL="${VP_LLM_MODEL:-/workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct}"

# Editing instructions (newline-separated; use || to delimit in env vars)
VIDEO_EDITING_INSTRUCTIONS="${VIDEO_EDITING_INSTRUCTIONS:-Single solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged
Double solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged
Single solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged
Double solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged
Single dashed white intermitted line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged}"

# VP inference parameters
VP_NUM_INFERENCE_STEPS="${VP_NUM_INFERENCE_STEPS:-70}"
VP_GUIDANCE_SCALE="${VP_GUIDANCE_SCALE:-6.0}"
VP_STRENGTH="${VP_STRENGTH:-1.0}"
VP_CAPTION_REFINE_ITERS="${VP_CAPTION_REFINE_ITERS:-10}"
VP_CAPTION_REFINE_TEMPERATURE="${VP_CAPTION_REFINE_TEMPERATURE:-0.1}"
VP_DILATE_SIZE="${VP_DILATE_SIZE:-24}"
VP_MASK_FEATHER="${VP_MASK_FEATHER:-8}"
VP_KEEP_MASKED_PIXELS="${VP_KEEP_MASKED_PIXELS:-True}"
VP_IMG_INPAINTING_LORA_SCALE="${VP_IMG_INPAINTING_LORA_SCALE:-0.0}"
VP_SEED="${VP_SEED:-42}"

# ==============================================================================
# STAGE 3: ALPAMAYO CONFIGURATION
# ==============================================================================
ALPAMAYO_OUTPUT_BASE="${ALPAMAYO_OUTPUT_BASE:-workspace/user/hbaskar/outputs/alpamayo}"
ALPAMAYO_MODEL_ID="${ALPAMAYO_MODEL_ID:-/workspace/alpamayo/checkpoints/alpamayo-r1-10b}"
ALPAMAYO_NUM_TRAJ_SAMPLES="${ALPAMAYO_NUM_TRAJ_SAMPLES:-1}"
ALPAMAYO_VIDEO_NAME="${ALPAMAYO_VIDEO_NAME:-auto}"

# HuggingFace token (needed by Alpamayo for ego-motion data)
HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "${HF_TOKEN}" ]]; then
    echo "WARNING: HF_TOKEN not set. Alpamayo may need it for ego-motion data."
    echo "         Set it with: HF_TOKEN=hf_xxx bash scripts/build_and_run.sh"
fi

# ==============================================================================
# DOCKER IMAGE REGISTRY
# ==============================================================================
REGISTRY="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research"

SAM2_LOCAL_IMAGE="sam2/frontend"
SAM2_REMOTE_IMAGE="${REGISTRY}/harimt_sam2"

VP_LOCAL_IMAGE="videopainter:latest"
VP_REMOTE_IMAGE="${REGISTRY}/harimt_vp${VP_RUN_SUFFIX}"

ALP_LOCAL_IMAGE="alpamayo:latest"
ALP_REMOTE_IMAGE="${REGISTRY}/alpamayo_vla_${MASTER_RUN_ID}"

MASTER_LOCAL_IMAGE="master-pipeline:latest"
MASTER_REMOTE_IMAGE="${REGISTRY}/master_pipeline"

# Tagged with the shared run id
SAM2_TAGGED="${SAM2_REMOTE_IMAGE}:${MASTER_RUN_ID}"
VP_TAGGED="${VP_REMOTE_IMAGE}:${MASTER_RUN_ID}"
ALP_TAGGED="${ALP_REMOTE_IMAGE}:${MASTER_RUN_ID}"
MASTER_TAGGED="${MASTER_REMOTE_IMAGE}:${MASTER_RUN_ID}"

# ==============================================================================
# PRINT CONFIGURATION SUMMARY
# ==============================================================================
echo "================================================================================"
echo " MASTER PIPELINE — BUILD AND RUN"
echo "================================================================================"
echo ""
echo "  RUN_ID:             ${RUN_ID}"
echo "  MASTER_RUN_ID:      ${MASTER_RUN_ID}"
echo "  TIMESTAMP:          ${RUN_TIMESTAMP}"
echo ""
echo " ── Stage 1: SAM2 Segmentation ──────────────────────────────────────────────"
echo "  Input:              chunks ${SAM2_CHUNK_START}–${SAM2_CHUNK_END} (${SAM2_TOTAL_CHUNKS} chunks × ${SAM2_FILES_PER_CHUNK} files ≈ ${SAM2_EXPECTED_VIDEOS} videos)"
echo "  Input base:         ${SAM2_INPUT_BASE}"
echo "  Max frames:         ${SAM2_MAX_FRAMES}"
echo "  SAM2 output:        ${SAM2_OUTPUT_BASE}/${MASTER_RUN_ID}/"
echo "  Preprocessed (→VP): ${SAM2_PREPROCESSED_OUTPUT_BASE}/${MASTER_RUN_ID}/"
echo "  Docker image:       ${SAM2_TAGGED}"
echo ""
echo " ── Stage 2: VideoPainter Editing ────────────────────────────────────────────"
echo "  Input (from SAM2):  ${SAM2_PREPROCESSED_OUTPUT_BASE}/${MASTER_RUN_ID}/"
echo "  VP output (→Alp):   ${VP_OUTPUT_BASE}/${MASTER_RUN_ID}/"
echo "  FluxFill ckpt:      ${TRAINED_FLUXFILL_GCS_PATH}"
echo "  Inference steps:    ${VP_NUM_INFERENCE_STEPS}"
echo "  Guidance scale:     ${VP_GUIDANCE_SCALE}"
echo "  Strength:           ${VP_STRENGTH}"
echo "  Refine iters:       ${VP_CAPTION_REFINE_ITERS}"
echo "  Docker image:       ${VP_TAGGED}"
echo ""
echo " ── Stage 3: Alpamayo VLA Inference ──────────────────────────────────────────"
echo "  Input (from VP):    ${VP_OUTPUT_BASE}/${MASTER_RUN_ID}/"
echo "  Output:             gs://${GCS_BUCKET}/${ALPAMAYO_OUTPUT_BASE}/${MASTER_RUN_ID}/"
echo "  Model:              ${ALPAMAYO_MODEL_ID}"
echo "  Traj samples:       ${ALPAMAYO_NUM_TRAJ_SAMPLES}"
echo "  Video name filter:  ${ALPAMAYO_VIDEO_NAME}"
echo "  Docker image:       ${ALP_TAGGED}"
echo ""
echo "================================================================================"

# ==============================================================================
# BUILD & PUSH DOCKER IMAGES
# ==============================================================================
if [[ "${SKIP_BUILD}" == "1" ]]; then
    echo ""
    echo "SKIP_BUILD=1 — reusing previously pushed images."
    echo ""
else
    echo ""
    echo "Building and pushing Docker images …"
    echo ""

    # ── Stage 1: SAM2 ────────────────────────────────────────────────────────
    echo "▸ Building SAM2 image …"
    pushd segmentation/sam2 > /dev/null
    docker compose build
    docker tag "${SAM2_LOCAL_IMAGE}" "${SAM2_TAGGED}"
    docker tag "${SAM2_LOCAL_IMAGE}" "${SAM2_REMOTE_IMAGE}:latest"
    docker push "${SAM2_TAGGED}"
    docker push "${SAM2_REMOTE_IMAGE}:latest"
    popd > /dev/null
    echo "  ✓ SAM2 image pushed: ${SAM2_TAGGED}"

    # ── Stage 2: VideoPainter ─────────────────────────────────────────────────
    echo "▸ Building VideoPainter image …"
    pushd generation/VideoPainter > /dev/null
    docker compose build
    docker tag "${VP_LOCAL_IMAGE}" "${VP_TAGGED}"
    docker tag "${VP_LOCAL_IMAGE}" "${VP_REMOTE_IMAGE}:latest"
    docker push "${VP_TAGGED}"
    docker push "${VP_REMOTE_IMAGE}:latest"
    popd > /dev/null
    echo "  ✓ VP image pushed: ${VP_TAGGED}"

    # ── Stage 3: Alpamayo ─────────────────────────────────────────────────────
    echo "▸ Building Alpamayo image …"
    pushd vla/alpamayo > /dev/null
    docker compose build
    docker tag "${ALP_LOCAL_IMAGE}" "${ALP_TAGGED}"
    docker tag "${ALP_LOCAL_IMAGE}" "${ALP_REMOTE_IMAGE}:latest"
    docker push "${ALP_TAGGED}"
    docker push "${ALP_REMOTE_IMAGE}:latest"
    popd > /dev/null
    echo "  ✓ Alpamayo image pushed: ${ALP_TAGGED}"

    # ── Master orchestrator ───────────────────────────────────────────────────
    echo "▸ Building Master orchestrator image …"
    docker compose build
    docker tag "${MASTER_LOCAL_IMAGE}" "${MASTER_TAGGED}"
    docker tag "${MASTER_LOCAL_IMAGE}" "${MASTER_REMOTE_IMAGE}:latest"
    docker push "${MASTER_TAGGED}"
    docker push "${MASTER_REMOTE_IMAGE}:latest"
    echo "  ✓ Master image pushed: ${MASTER_TAGGED}"
fi

# ==============================================================================
# EXPORT ENV VARS FOR WORKFLOW SERIALISATION
# ==============================================================================
export SAM2_CONTAINER_IMAGE="${SAM2_TAGGED}"
export VP_CONTAINER_IMAGE="${VP_TAGGED}"
export ALPAMAYO_CONTAINER_IMAGE="${ALP_TAGGED}"
export SAM2_OUTPUT_BASE
export SAM2_PREPROCESSED_OUTPUT_BASE
export VP_OUTPUT_BASE
export TRAINED_FLUXFILL_GCS_PATH
export ALPAMAYO_OUTPUT_BASE
export HF_TOKEN

# ==============================================================================
# SUBMIT MASTER WORKFLOW
# ==============================================================================
echo ""
echo "================================================================================"
echo " LAUNCHING MASTER WORKFLOW"
echo "================================================================================"
echo ""

hlx wf run \
  --team-space research \
  --domain prod \
  --execution-name "master-${MASTER_RUN_ID//_/-}" \
  workflow_master.master_pipeline_wf \
  --run_id "${MASTER_RUN_ID}" \
  --sam2_video_uris "${SAM2_VIDEO_URIS}" \
  --sam2_max_frames "${SAM2_MAX_FRAMES}" \
  --vp_video_editing_instructions "${VIDEO_EDITING_INSTRUCTIONS}" \
  --vp_llm_model "${VP_LLM_MODEL}" \
  --vp_num_inference_steps "${VP_NUM_INFERENCE_STEPS}" \
  --vp_guidance_scale "${VP_GUIDANCE_SCALE}" \
  --vp_strength "${VP_STRENGTH}" \
  --vp_caption_refine_iters "${VP_CAPTION_REFINE_ITERS}" \
  --vp_caption_refine_temperature "${VP_CAPTION_REFINE_TEMPERATURE}" \
  --vp_dilate_size "${VP_DILATE_SIZE}" \
  --vp_mask_feather "${VP_MASK_FEATHER}" \
  --vp_keep_masked_pixels \
  --vp_img_inpainting_lora_scale "${VP_IMG_INPAINTING_LORA_SCALE}" \
  --vp_seed "${VP_SEED}" \
  --alp_model_id "${ALPAMAYO_MODEL_ID}" \
  --alp_num_traj_samples "${ALPAMAYO_NUM_TRAJ_SAMPLES}" \
  --alp_video_name "${ALPAMAYO_VIDEO_NAME}"

echo ""
echo "================================================================================"
echo " WORKFLOW SUBMITTED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "  Shared RUN_ID:        ${MASTER_RUN_ID}"
echo ""
echo "  Stage 1 — SAM2:"
echo "    Raw output:         ${SAM2_OUTPUT_BASE}/${MASTER_RUN_ID}/"
echo "    Preprocessed (→VP): ${SAM2_PREPROCESSED_OUTPUT_BASE}/${MASTER_RUN_ID}/"
echo ""
echo "  Stage 2 — VideoPainter:"
echo "    Edited videos:      ${VP_OUTPUT_BASE}/${MASTER_RUN_ID}/"
echo ""
echo "  Stage 3 — Alpamayo:"
echo "    Predictions:        gs://${GCS_BUCKET}/${ALPAMAYO_OUTPUT_BASE}/${MASTER_RUN_ID}/"
echo ""
echo "================================================================================"
