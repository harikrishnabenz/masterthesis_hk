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

# ── Max frames per video ─────────────────────────────────────────────────────
SAM2_MAX_FRAMES="${SAM2_MAX_FRAMES:-100}"

# ── Chunk-based input selection ──────────────────────────────────────────────
#
# SAM2 input base: parent folder containing camera subfolders.
#   gs://.../Input/data_physical_ai/camera_front_tele_30fov/chunk_0000/<uuid>.mp4
#   gs://.../Input/data_physical_ai/camera_front_tele_30fov/chunk_0001/<uuid>.mp4
#   ...
#
# Configure which chunks and how many files per chunk to process:
SAM2_CHUNK_START="${SAM2_CHUNK_START:-0}"       # first chunk index (inclusive)
SAM2_CHUNK_END="${SAM2_CHUNK_END:-19}"          # last chunk index (inclusive)
SAM2_FILES_PER_CHUNK="${SAM2_FILES_PER_CHUNK:-1}" # number of .mp4 files per chunk

# Encode chunk config into a URI string that the SAM2 task will parse inside
# the GPU container (resolved via gcsfs Python library at runtime).
# Format: chunks://<base_path>?start=N&end=M&per_chunk=K
SAM2_VIDEO_URIS="${SAM2_VIDEO_URIS:-chunks://${SAM2_INPUT_BASE#gs://}?start=${SAM2_CHUNK_START}&end=${SAM2_CHUNK_END}&per_chunk=${SAM2_FILES_PER_CHUNK}}"

# Alternatively, override with:
#   A single folder:   SAM2_VIDEO_URIS="gs://bucket/folder/"
#   A single file:     SAM2_VIDEO_URIS="gs://bucket/video.mp4"
#   Multiple files:    SAM2_VIDEO_URIS="gs://bucket/v1.mp4,gs://bucket/v2.mp4"
#   Built-in defaults: SAM2_VIDEO_URIS="default"

# Calculate total expected videos
SAM2_TOTAL_CHUNKS=$(( SAM2_CHUNK_END - SAM2_CHUNK_START + 1 ))
SAM2_EXPECTED_VIDEOS=$(( SAM2_TOTAL_CHUNKS * SAM2_FILES_PER_CHUNK ))

# Display input info
if [[ "${SAM2_VIDEO_URIS}" == chunks://* ]]; then
  SAM2_INPUT_DISPLAY="chunks ${SAM2_CHUNK_START}-${SAM2_CHUNK_END} (${SAM2_TOTAL_CHUNKS} chunks x ${SAM2_FILES_PER_CHUNK} files = ~${SAM2_EXPECTED_VIDEOS} videos)"
elif [[ "${SAM2_VIDEO_URIS}" == */ ]]; then
  SAM2_INPUT_DISPLAY="GCS folder: ${SAM2_VIDEO_URIS}"
elif [[ "${SAM2_VIDEO_URIS}" == "default" ]]; then
  SAM2_INPUT_DISPLAY="default (10 built-in videos)"
elif [[ "${SAM2_VIDEO_URIS}" == *","* ]]; then
  IFS=',' read -ra _VIDEO_ARRAY <<< "${SAM2_VIDEO_URIS}"
  SAM2_INPUT_DISPLAY="${#_VIDEO_ARRAY[@]} video files"
else
  SAM2_INPUT_DISPLAY="single file: $(basename "${SAM2_VIDEO_URIS}")"
fi

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

# Tag the image for Google Artifact Registry (include SAM2_RUN_ID in image name)
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2_${SAM2_RUN_ID}:${SAM2_RUN_ID}"
docker tag sam2/frontend "${REMOTE_IMAGE}"

# Also update :latest for convenience
REMOTE_IMAGE_LATEST="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2_${SAM2_RUN_ID}:latest"
docker tag sam2/frontend "${REMOTE_IMAGE_LATEST}"

# Push the images to the registry
docker push "${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE_LATEST}"

echo "================================================================================"
echo "SAM2 SEGMENTATION - BUILD AND RUN"
echo "================================================================================"
echo "  RUN_ID:                ${RUN_ID}"
echo "  SAM2_RUN_ID:           ${SAM2_RUN_ID}"
echo "  INPUT:                 ${SAM2_INPUT_DISPLAY}"
echo "  INPUT (parent):        ${SAM2_INPUT_PARENT}"
echo "  INPUT (camera):        ${SAM2_INPUT_BASE}"
echo "  SAM2_MAX_FRAMES:       ${SAM2_MAX_FRAMES}"
echo "  OUTPUT (raw):          ${SAM2_OUTPUT_BASE}/${SAM2_RUN_ID}/"
echo "  OUTPUT (preprocessed): ${SAM2_PREPROCESSED_OUTPUT_BASE}/${SAM2_RUN_ID}/"
echo "  Docker image:          ${REMOTE_IMAGE}"
echo "================================================================================"

# Ensure workflow_sam2.py uses this exact image when hlx packages the workflow
export SAM2_CONTAINER_IMAGE="${REMOTE_IMAGE}"

# Run the HLX workflow for SAM2 segmentation
# Processes videos with SAM2, generates masks, and uploads to GCS in two formats:
#   1. Raw outputs (masks + visualizations + segmented videos)
#   2. VideoPainter preprocessed format (for video inpainting)

hlx wf run \
  --team-space research \
  --domain prod \
  --execution-name "sam2-${SAM2_RUN_ID//_/-}-$(date -u +%Y%m%d-%H%M%S)" \
  workflow_sam2.sam2_segmentation_wf \
  --run_id "${SAM2_RUN_ID}" \
  --sam2_video_uris "${SAM2_VIDEO_URIS}" \
  --max_frames "${SAM2_MAX_FRAMES}"

echo ""
echo "================================================================================"
echo "WORKFLOW SUBMITTED"
echo "================================================================================"
echo "  Raw output:          ${SAM2_OUTPUT_BASE}/${SAM2_RUN_ID}/"
echo "  Preprocessed (→ VP): ${SAM2_PREPROCESSED_OUTPUT_BASE}/${SAM2_RUN_ID}/"
echo "================================================================================"