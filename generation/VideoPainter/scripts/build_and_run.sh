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
RUN_TAG="${RUN_ID}_${RUN_TIMESTAMP}"

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

# VP_DATA_RUN_ID: auto-discover the SAM2 output folder matching RUN_ID.
# The folder name is <RUN_ID>_<timestamp>, e.g. 001_20260217_092127.
VP_DATA_RUN_ID=$(gcloud storage ls "${VP_INPUT_BASE}/" 2>/dev/null \
  | grep "/${RUN_ID}_" \
  | head -1 \
  | sed 's|.*/\([^/]*\)/$|\1|')
if [[ -z "${VP_DATA_RUN_ID}" ]]; then
  echo "ERROR: No SAM2 output folder found matching '${RUN_ID}_*' under ${VP_INPUT_BASE}/"
  exit 1
fi
echo "Auto-detected SAM2 data folder: ${VP_DATA_RUN_ID}"

echo "================================================================================"
echo "VIDEOPAINTER EDITING - BUILD AND RUN"
echo "================================================================================"
echo "  RUN_TAG:                   ${RUN_TAG}"
echo "  INPUT (SAM2 preprocessed): ${VP_INPUT_BASE}/${VP_DATA_RUN_ID}/"
echo "  OUTPUT (edited videos):    ${VP_OUTPUT_BASE}/${RUN_TAG}/"
echo "  TRAINED_FLUXFILL_GCS_PATH: ${TRAINED_FLUXFILL_GCS_PATH}"
echo "================================================================================"

# Declare a run suffix used by both this script and workflow.py
# Extract timestamp from the trained checkpoint folder name
CHECKPOINT_TIMESTAMP=$(basename "$TRAINED_FLUXFILL_GCS_PATH" | grep -oE '[0-9]{8}_[0-9]{6}' | head -1)
X="withoutlora_5prompt_${CHECKPOINT_TIMESTAMP}"
export VP_RUN_SUFFIX="${X}"

docker compose build

REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_vp${X}"
REMOTE_IMAGE_TAGGED="${REMOTE_IMAGE}:${RUN_TAG}"
docker tag videopainter:latest "${REMOTE_IMAGE_TAGGED}"
docker tag videopainter:latest "${REMOTE_IMAGE}:latest"
docker push "${REMOTE_IMAGE_TAGGED}"
docker push "${REMOTE_IMAGE}:latest"

export VP_CONTAINER_IMAGE="${REMOTE_IMAGE_TAGGED}"

# Video editing instructions (newline-separated; core change first, constraints after ';')
VIDEO_EDITING_INSTRUCTIONS=$'Single solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nDouble solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nSingle solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nDouble solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nSingle dashed white intermitted line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'
CAPTION_REFINE_ITERS="${CAPTION_REFINE_ITERS:-10}"
CAPTION_REFINE_TEMPERATURE="${CAPTION_REFINE_TEMPERATURE:-0.1}"
VP_STRENGTH="${VP_STRENGTH:-1.0}"


hlx wf run \
  --team-space research \
  --domain prod \
  --execution-name "vp-${X//_/-}-${RUN_TAG//_/-}" \
  workflow.videopainter_many_wf \
  --data_run_id "${VP_DATA_RUN_ID}" \
  --output_run_id "${RUN_TAG}" \
  --data_video_ids "01d3588e-bca7-4a18-8e74-c6cfe9e996db.camera_front_tele_30fov" \
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
echo "  Output (→ Alpamayo): ${VP_OUTPUT_BASE}/${RUN_TAG}/"
echo "  Report:              ${VP_OUTPUT_BASE}/${RUN_TAG}/${RUN_TAG}.txt"
echo "================================================================================"