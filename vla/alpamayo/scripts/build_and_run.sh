#!/bin/bash
# ==================================================================================
# Alpamayo VLA Inference - Build and Run Script
# ==================================================================================
#
# PIPELINE POSITION: Stage 3 of 3  (SAM2 --> VP --> Alpamayo)
#
# INPUT:
#   VideoPainter-edited videos from Stage 2:
#     gs://<bucket>/.../training/output/vp/<run_id>/
#
# OUTPUT:
#   VLA trajectory predictions + visualizations:
#     gs://<bucket>/.../training/output/alpamayo/<run_id>/
#       ├── <video_id>_inference.json   (trajectory predictions + reasoning)
#       ├── <video_id>_vis_data.npz     (visualization tensors)
#       ├── <video_id>_overlay.mp4      (video with trajectory overlay)
#       └── <run_id>_report.txt         (summary report)
# ==================================================================================

set -e

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
# VideoPainter edited videos (output of Stage 2).
# Override: VP_OUTPUT_BASE="gs://bucket/my/videos" bash scripts/build_and_run.sh
VP_OUTPUT_BASE="${VP_OUTPUT_BASE:-gs://${GCS_BUCKET}/workspace/user/hbaskar/outputs/vp}"
export VP_OUTPUT_BASE

# Auto-discover the VP output folder matching RUN_ID.
# The folder name is <RUN_ID>_<timestamp>, e.g. 001_20260217_095447.
VP_RUN_FOLDER=$(gcloud storage ls "${VP_OUTPUT_BASE}/" 2>/dev/null \
  | grep "/${RUN_ID}_" \
  | head -1 \
  | sed 's|.*/\([^/]*\)/$|\1|')
if [[ -z "${VP_RUN_FOLDER}" ]]; then
  echo "ERROR: No VP output folder found matching '${RUN_ID}_*' under ${VP_OUTPUT_BASE}/"
  exit 1
fi
VIDEO_DATA_GCS_PATH="${VP_OUTPUT_BASE}/${VP_RUN_FOLDER}"
echo "Auto-detected VP data folder: ${VIDEO_DATA_GCS_PATH}"

# ==============================================================================
# OUTPUT PATHS
# ==============================================================================
# NOTE: This is a GCS prefix (no gs:// scheme). The workflow appends gs://<bucket>/ itself.
# Final path: gs://<bucket>/${ALPAMAYO_OUTPUT_BASE}/<run_id>/
ALPAMAYO_OUTPUT_BASE="${ALPAMAYO_OUTPUT_BASE:-workspace/user/hbaskar/outputs/alpamayo}"
export ALPAMAYO_OUTPUT_BASE

# ----------------------------------------------------------------------------------
# RUN CONFIGURATION
# ----------------------------------------------------------------------------------

# Number of trajectory samples per video
NUM_TRAJ_SAMPLES="${NUM_TRAJ_SAMPLES:-1}"

# Optionally restrict to a single video by name (stem, without extension).
# Set to "auto" to process all videos, or specify a video stem.
# Example: VIDEO_NAME="abc123.camera_front_tele_30fov_vp_edit_sample0" bash scripts/build_and_run.sh
VIDEO_NAME="${VIDEO_NAME:-auto}"

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

echo "================================================================================"
echo "ALPAMAYO VLA INFERENCE - BUILD AND RUN"
echo "================================================================================"
echo "  RUN_TAG:               ${RUN_TAG}"
echo "  INPUT  (VP videos):    ${VIDEO_DATA_GCS_PATH}"
echo "  OUTPUT (predictions):  gs://${GCS_BUCKET}/${ALPAMAYO_OUTPUT_BASE}/${RUN_TAG}/"
echo "  NUM_TRAJ_SAMPLES:      ${NUM_TRAJ_SAMPLES}"
echo "  VIDEO_NAME:            ${VIDEO_NAME}"
echo "  MODEL_ID:              ${MODEL_ID}"
echo "  HF_TOKEN:              ${HF_TOKEN:+set (hidden)}"
echo "================================================================================"

# ----------------------------------------------------------------------------------
# DOCKER BUILD AND PUSH
# ----------------------------------------------------------------------------------
echo "Building Docker image..."
cd "$(dirname "$0")/.."
docker compose build

# Tag the image for Google Artifact Registry (include RUN_TAG in image name)
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/alpamayo_vla_${RUN_TAG}"
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
echo "================================================================================"
echo "LAUNCHING WORKFLOW"
echo "================================================================================"

hlx wf run \
  --team-space research \
  --domain prod \
  --execution-name "alpamayo-vla-${RUN_TAG//_/-}" \
  workflow_alpamayo.alpamayo_vla_inference_wf \
  --video_data_gcs_path "${VIDEO_DATA_GCS_PATH}" \
  --output_run_id "${RUN_TAG}" \
  --model_id "${MODEL_ID}" \
  --num_traj_samples "${NUM_TRAJ_SAMPLES}" \
  --video_name "${VIDEO_NAME}"

echo ""
echo "================================================================================"
echo "WORKFLOW SUBMITTED"
echo "================================================================================"
echo "  Input (VP videos):  ${VIDEO_DATA_GCS_PATH}"
echo "  Output:             gs://${GCS_BUCKET}/${ALPAMAYO_OUTPUT_BASE}/${RUN_TAG}/"
echo "  Report:             gs://${GCS_BUCKET}/${ALPAMAYO_OUTPUT_BASE}/${RUN_TAG}/${RUN_TAG}_report.txt"
echo "================================================================================"
