#!/bin/bash
# filepath: build_and_run.sh
# ==================================================================================
# SAM2 Segmentation - Build and Run Script
# ==================================================================================
#
# PIPELINE POSITION: Stage 1 of 3  (SAM2 --> VP --> Alpamayo)
#
# INPUT:
#   Video files from GCS (see SAM2_INPUT_BASE below).
#   Default: camera_front_tele_30fov dataset
#
# OUTPUT (two destinations):
#   1. Raw segmentation masks + visualizations:
#      gs://<bucket>/.../training/output/sam2/<run_id>/
#   2. VideoPainter-preprocessed format (consumed by Stage 2):
#      gs://<bucket>/.../outputs/preprocessed_data_vp/<run_id>/
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
SAM2_RUN_ID="${SAM2_RUN_ID:-${RUN_ID}_${RUN_TIMESTAMP}}"

# ==============================================================================
# INPUT PATHS
# ==============================================================================
# Parent GCS path where input data lives (contains camera subfolders).
# Override: SAM2_INPUT_PARENT="gs://bucket/my/data" bash scripts/build_and_run.sh
SAM2_INPUT_PARENT="${SAM2_INPUT_PARENT:-gs://${GCS_BUCKET}/workspace/user/hbaskar/Input/data_physical_ai}"
SAM2_CAMERA_SUBFOLDER="${SAM2_CAMERA_SUBFOLDER:-camera_front_tele_30fov}"
SAM2_INPUT_BASE="${SAM2_INPUT_BASE:-${SAM2_INPUT_PARENT}/${SAM2_CAMERA_SUBFOLDER}}"
export SAM2_INPUT_BASE
export SAM2_INPUT_PARENT
export SAM2_CAMERA_SUBFOLDER

# ==============================================================================
# OUTPUT PATHS
# ==============================================================================
# 1. Raw segmentation output (masks, visualizations, segmented videos)
#    Final path: ${SAM2_OUTPUT_BASE}/<run_id>/
SAM2_OUTPUT_BASE="${SAM2_OUTPUT_BASE:-gs://${GCS_BUCKET}/workspace/user/hbaskar/outputs/sam2}"
export SAM2_OUTPUT_BASE

# 2. VideoPainter preprocessed data (meta.csv, raw_videos/, masks/)
#    Final path: ${SAM2_PREPROCESSED_OUTPUT_BASE}/<run_id>/<video_id>/
#    This is the INPUT for Stage 2 (VideoPainter).
SAM2_PREPROCESSED_OUTPUT_BASE="${SAM2_PREPROCESSED_OUTPUT_BASE:-gs://${GCS_BUCKET}/workspace/user/hbaskar/outputs/preprocessed_data_vp}"
export SAM2_PREPROCESSED_OUTPUT_BASE

# Generate timestamp for unique image tagging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Build the Docker image
docker compose build

# Tag the image for Google Artifact Registry
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2:${TIMESTAMP}"
docker tag sam2/frontend "${REMOTE_IMAGE}"

# Push the image to the registry
docker push "${REMOTE_IMAGE}"

echo "================================================================================"
echo "SAM2 SEGMENTATION - BUILD AND RUN"
echo "================================================================================"
echo "  RUN_ID:                ${RUN_ID}"
echo "  SAM2_RUN_ID:           ${SAM2_RUN_ID}"
echo "  INPUT (parent):        ${SAM2_INPUT_PARENT}"
echo "  INPUT (camera):        ${SAM2_INPUT_BASE}"
echo "  OUTPUT (raw):          ${SAM2_OUTPUT_BASE}/${SAM2_RUN_ID}/"
echo "  OUTPUT (preprocessed): ${SAM2_PREPROCESSED_OUTPUT_BASE}/${SAM2_RUN_ID}/"
echo "  Docker image:          ${REMOTE_IMAGE}"
echo "================================================================================"

# Ensure workflow.py uses this exact image when hlx packages the workflow
export SAM2_CONTAINER_IMAGE="${REMOTE_IMAGE}"

# Run the HLX workflow for SAM2 segmentation
# Processes videos with SAM2, generates masks, and uploads to GCS in two formats:
#   1. Raw outputs (masks + visualizations + segmented videos)
#   2. VideoPainter preprocessed format (for video inpainting)

hlx wf run \
  --team-space research \
  --domain prod \
  workflow.sam2_segmentation_wf \
  --run_id "${SAM2_RUN_ID}" \
  --max_frames "150"

echo ""
echo "================================================================================"
echo "WORKFLOW SUBMITTED"
echo "================================================================================"
echo "  Raw output:          ${SAM2_OUTPUT_BASE}/${SAM2_RUN_ID}/"
echo "  Preprocessed (→ VP): ${SAM2_PREPROCESSED_OUTPUT_BASE}/${SAM2_RUN_ID}/"
echo "================================================================================"