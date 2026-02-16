#!/bin/bash
# ==================================================================================
# VideoPainter Editing - Build and Run Script
# ==================================================================================
#
# PIPELINE POSITION: Stage 2 of 3  (SAM2 --> VP --> Alpamayo)
#
# INPUT:
#   SAM2-preprocessed video data from Stage 1:
#     gs://<bucket>/.../outputs/preprocessed_data_vp/<data_run_id>/<video_id>/
#       ├── meta.csv
#       ├── raw_videos/
#       └── masks/
#
# OUTPUT:
#   Edited videos written to:
#     gs://<bucket>/.../training/output/vp/<output_run_id>/
#   This is the INPUT for Stage 3 (Alpamayo).
# ==================================================================================

set -euo pipefail

# ==============================================================================
# GCS BUCKET
# ==============================================================================
GCS_BUCKET="mbadas-sandbox-research-9bb9c7f"

# ==============================================================================
# RUN ID  (set this to identify your run; combined with timestamp for output folders)
# ==============================================================================
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${RUN_ID:-001}"

# ==============================================================================
# INPUT PATHS
# ==============================================================================
# SAM2 preprocessed data (output of Stage 1).
# The workflow mounts this prefix via FuseBucket and uses data_run_id as subfolder.
# Override: VP_INPUT_BASE="gs://bucket/path" bash scripts/build_and_run.sh
VP_INPUT_BASE="${VP_INPUT_BASE:-gs://${GCS_BUCKET}/workspace/user/hbaskar/outputs/preprocessed_data_vp}"
export VP_INPUT_BASE

# ==============================================================================
# OUTPUT PATHS
# ==============================================================================
# Final edited videos.  Path: ${VP_OUTPUT_BASE}/<output_run_id>/
# This is consumed by Stage 3 (Alpamayo).
# Override: VP_OUTPUT_BASE="gs://..." bash scripts/build_and_run.sh
VP_OUTPUT_BASE="${VP_OUTPUT_BASE:-gs://${GCS_BUCKET}/workspace/user/hbaskar/outputs/vp}"
export VP_OUTPUT_BASE

# ----------------------------------------------------------------------------------
# LLM MODEL SELECTION
# ----------------------------------------------------------------------------------
# This repo is configured to use ONLY the Qwen2.5-VL-7B-Instruct checkpoint
# mounted by workflow.py and exposed under the local path below.
LLM_MODEL_PATH="/workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct"

# ----------------------------------------------------------------------------------
# TRAINED FLUXFILL LORA CHECKPOINT (GCS)
# ----------------------------------------------------------------------------------
# Override with: TRAINED_FLUXFILL_GCS_PATH="workspace/user/.../my_checkpoint" bash scripts/build_and_run.sh
TRAINED_FLUXFILL_GCS_PATH="${TRAINED_FLUXFILL_GCS_PATH:-workspace/user/hbaskar/Video_inpainting/videopainter/training/trained_checkpoint/fluxfill_single_white_solid_clearroad_20260212_151908}"
export TRAINED_FLUXFILL_GCS_PATH

echo "================================================================================"
echo "VIDEOPAINTER EDITING - BUILD AND RUN"
echo "================================================================================"
echo "  RUN_ID:                    ${RUN_ID}"
echo "  RUN_TIMESTAMP:             ${RUN_TIMESTAMP}"
echo "  INPUT (SAM2 preprocessed): ${VP_INPUT_BASE}/<data_run_id>/"
echo "  OUTPUT (edited videos):    ${VP_OUTPUT_BASE}/<output_run_id>/"
echo "  TRAINED_FLUXFILL_GCS_PATH: ${TRAINED_FLUXFILL_GCS_PATH}"
echo "================================================================================"

# Declare a run suffix used by both this script and workflow.py
# Extract timestamp from the trained checkpoint folder name
CHECKPOINT_TIMESTAMP=$(basename "$TRAINED_FLUXFILL_GCS_PATH" | grep -oE '[0-9]{8}_[0-9]{6}' | head -1)
X="withoutlora_5prompt_${CHECKPOINT_TIMESTAMP}"
export VP_RUN_SUFFIX="${X}"

# Build image first (required for the tag below)
docker compose build

# Tag the image for Google Artifact Registry (use a unique tag to avoid stale ':latest' caching)
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_vp${X}"
REMOTE_IMAGE_TAGGED="${REMOTE_IMAGE}:${RUN_TAG}"
docker tag videopainter:latest "${REMOTE_IMAGE_TAGGED}"

# (Optional) Also update :latest for convenience.
docker tag videopainter:latest "${REMOTE_IMAGE}:latest"

# Push the image to the registry
docker push "${REMOTE_IMAGE_TAGGED}"
docker push "${REMOTE_IMAGE}:latest"

# Ensure workflow.py uses this exact image tag when hlx packages the workflow.
export VP_CONTAINER_IMAGE="${REMOTE_IMAGE_TAGGED}"

# Pass data_run_id as parameter to match SAM2 output

# NOTE: workflow names are NOT suffixed (see workflow.py).

# -----------------------------------------------------------------------------
# Video editing instruction(s)
# Choose ONE of the following:
#   - Single instruction: set VIDEO_EDITING_INSTRUCTION and use the singular flag
#   - Multiple instructions: set VIDEO_EDITING_INSTRUCTIONS (newline separated)
# -----------------------------------------------------------------------------

# VIDEO_EDITING_INSTRUCTION="Add a double yellow centerline to the road; realistic worn paint"

# IMPORTANT (edit_bench.py behavior):
# - The string is forwarded to infer/edit_bench.py as --video_editing_instruction.
# - If --llm_model is disabled/none, edit_bench.py uses the instruction text directly
#   for the masked-region caption, taking only the first clause before ';' and
#   trimming to 20 words.
#
# So: put the *core visual change* first, then optional constraints after ';'.
VIDEO_EDITING_INSTRUCTIONS=$'Single solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nDouble solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nSingle solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nDouble solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nSingle dashed white intermitted line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'
# -----------------------------------------------------------------------------
# First-frame caption refinement (Qwen critic loop)
# Set these env vars to override defaults:
#   CAPTION_REFINE_ITERS=5         (number of refinement iterations, default 10)
#   CAPTION_REFINE_TEMPERATURE=0.1 (Qwen temperature, default 0.2)
# Example: CAPTION_REFINE_ITERS=3 bash scripts/build_and_run.sh
# -----------------------------------------------------------------------------
CAPTION_REFINE_ITERS="${CAPTION_REFINE_ITERS:-10}"
CAPTION_REFINE_TEMPERATURE="${CAPTION_REFINE_TEMPERATURE:-0.1}"

# -----------------------------------------------------------------------------
# Inpainting strength
# Forwarded to infer/edit_bench.py as --strength (valid range: [0.0, 1.0]).
# Example: VP_STRENGTH=0.85 bash scripts/build_and_run.sh
# -----------------------------------------------------------------------------
VP_STRENGTH="${VP_STRENGTH:-1.0}"


VP_OUTPUT_RUN_ID="${VP_OUTPUT_RUN_ID:-${RUN_ID}_${RUN_TIMESTAMP}}"
VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-001}"

hlx wf run \
  --team-space research \
  --domain prod \
  --execution-name "vp-${X//_/-}-$(date -u +%Y%m%d-%H%M%S)" \
  workflow.videopainter_many_wf \
  --data_run_id "${VP_DATA_RUN_ID}" \
  --output_run_id "${VP_OUTPUT_RUN_ID}" \
  --data_video_ids "auto" \
  --inpainting_sample_id 0 \
  --model_path "/workspace/VideoPainter/ckpt/CogVideoX-5b-I2V" \
  --inpainting_branch "/workspace/VideoPainter/ckpt/VideoPainter/checkpoints/branch" \
  --img_inpainting_model "/workspace/VideoPainter/ckpt/flux_inp" \
  --img_inpainting_lora_path "/workspace/VideoPainter/ckpt/trained_fluxfill_lora" \
  --img_inpainting_lora_scale 0.0 \
  --output_name_suffix "vp_edit_sample0.mp4" \
  --num_inference_steps 70 \
  --guidance_scale 6.0 \
  --strength "${VP_STRENGTH}" \
  --down_sample_fps 8 \
  --inpainting_frames 49 \
  --video_editing_instructions "${VIDEO_EDITING_INSTRUCTIONS}" \
  --llm_model "${LLM_MODEL_PATH}" \
  --caption_refine_iters "${CAPTION_REFINE_ITERS}" \
  --caption_refine_temperature "${CAPTION_REFINE_TEMPERATURE}" \
  --dilate_size 24 \
  --mask_feather 8 \
  --keep_masked_pixels

echo ""
echo "================================================================================"
echo "WORKFLOW SUBMITTED"
echo "================================================================================"
echo "  Input (SAM2 data):   ${VP_INPUT_BASE}/${VP_DATA_RUN_ID}/"
echo "  Output (→ Alpamayo): ${VP_OUTPUT_BASE}/${VP_OUTPUT_RUN_ID}/"
echo "  Report:              ${VP_OUTPUT_BASE}/${VP_OUTPUT_RUN_ID}/${VP_OUTPUT_RUN_ID}.txt"
echo "================================================================================"

# llm_model options:
# --llm_model "/workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct"  # Local (mounted)
# --llm_model "Qwen/Qwen2.5-VL-7B-Instruct"                             # Hub (requires internet)
# --llm_model "none"