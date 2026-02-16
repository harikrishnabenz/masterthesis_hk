#!/bin/bash
# ----------------------------------------------------------------------------------
# Alpamayo VLA Inference - Build and Run Script
# ----------------------------------------------------------------------------------

set -e

# ----------------------------------------------------------------------------------
# OUTPUT FOLDER (override: ALPAMAYO_OUTPUT_BASE="workspace/user/..." bash scripts/build_and_run.sh)
# ----------------------------------------------------------------------------------
# NOTE: This is a GCS prefix (no gs:// scheme). The workflow appends gs://<bucket>/ itself.
ALPAMAYO_OUTPUT_BASE="${ALPAMAYO_OUTPUT_BASE:-workspace/user/hbaskar/Video_inpainting/videopainter/training/output/alpamayo}"
export ALPAMAYO_OUTPUT_BASE

# ----------------------------------------------------------------------------------
# VIDEO DATA SOURCE (CONFIGURE THIS)
# ----------------------------------------------------------------------------------
# Set the GCS path to your video data
# Example: gs://bucket/path/to/videos
# Override with: VIDEO_DATA_GCS_PATH="gs://bucket/my/videos" bash scripts/build_and_run.sh
VIDEO_DATA_GCS_PATH="${VIDEO_DATA_GCS_PATH:-gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/output_vp_final/10_withoutlora_5prompt_20260212_151908}"

# ----------------------------------------------------------------------------------
# RUN CONFIGURATION
# ----------------------------------------------------------------------------------
# Unique identifier for this run (default: timestamp)
RUN_ID="${RUN_ID:-alpamayo_1_$(date -u +%Y%m%d_%H%M%S)}"

# Number of trajectory samples per video
NUM_TRAJ_SAMPLES="${NUM_TRAJ_SAMPLES:-1}"

# HuggingFace token for streaming the NVIDIA PhysicalAI-AV dataset
# Required for ego-motion + multi-camera data from HuggingFace
# Override: HF_TOKEN="hf_xxx" bash scripts/build_and_run.sh
HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. The workflow needs it to stream ego-motion"
    echo "         data from HuggingFace (nvidia/PhysicalAI-Autonomous-Vehicles)."
    echo "         Set it with: HF_TOKEN=hf_xxx bash scripts/build_and_run.sh"
fi
export HF_TOKEN

# Model ID (default: use mounted checkpoint from GCS)
# The checkpoint is mounted from GCS and symlinked to /workspace/alpamayo/checkpoints
# Set to HuggingFace ID to download instead: MODEL_ID="nvidia/Alpamayo-R1-10B"
MODEL_ID="${MODEL_ID:-/workspace/alpamayo/checkpoints/alpamayo-r1-10b}"

echo "=" ================================================================
echo "ALPAMAYO VLA INFERENCE - BUILD AND RUN"
echo "=" ================================================================
echo "  VIDEO_DATA_GCS_PATH: $VIDEO_DATA_GCS_PATH"
echo "  RUN_ID: $RUN_ID"
echo "  NUM_TRAJ_SAMPLES: $NUM_TRAJ_SAMPLES"
echo "  MODEL_ID: $MODEL_ID"
echo "  HF_TOKEN: ${HF_TOKEN:+set (hidden)}"
echo "  ALPAMAYO_OUTPUT_BASE: gs://mbadas-sandbox-research-9bb9c7f/$ALPAMAYO_OUTPUT_BASE"
echo "=" ================================================================

# ----------------------------------------------------------------------------------
# DOCKER BUILD AND PUSH
# ----------------------------------------------------------------------------------
echo "Building Docker image..."
cd "$(dirname "$0")/.."
docker compose build

# Tag the image for Google Artifact Registry
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/alpamayo_vla"
REMOTE_IMAGE_TAGGED="${REMOTE_IMAGE}:${RUN_TAG}"

echo "Tagging image: ${REMOTE_IMAGE_TAGGED}"
docker tag alpamayo:latest "${REMOTE_IMAGE_TAGGED}"
docker tag alpamayo:latest "${REMOTE_IMAGE}:latest"

echo "Pushing images to registry..."
docker push "${REMOTE_IMAGE_TAGGED}"
docker push "${REMOTE_IMAGE}:latest"

# Export for workflow to use
export ALPAMAYO_CONTAINER_IMAGE="${REMOTE_IMAGE_TAGGED}"

# ----------------------------------------------------------------------------------
# RUN WORKFLOW
# ----------------------------------------------------------------------------------
echo ""
echo "=" ================================================================
echo "LAUNCHING WORKFLOW"
echo "=" ================================================================

hlx wf run \
  --team-space research \
  --domain prod \
  --execution-name "alpamayo-vla-${RUN_ID//_/-}-$(date -u +%Y%m%d-%H%M%S)" \
  workflow.alpamayo_vla_inference_wf \
  --video_data_gcs_path "${VIDEO_DATA_GCS_PATH}" \
  --output_run_id "${RUN_ID}" \
  --model_id "${MODEL_ID}" \
  --num_traj_samples "${NUM_TRAJ_SAMPLES}"

echo ""
echo "=" ================================================================
echo "WORKFLOW LAUNCHED"
echo "=" ================================================================
echo "Output will be uploaded to:"
echo "  gs://mbadas-sandbox-research-9bb9c7f/${ALPAMAYO_OUTPUT_BASE}/${RUN_ID}/"
echo ""
echo "Report will be available at:"
echo "  gs://mbadas-sandbox-research-9bb9c7f/${ALPAMAYO_OUTPUT_BASE}/${RUN_ID}/${RUN_ID}_report.txt"
echo "=" ================================================================
