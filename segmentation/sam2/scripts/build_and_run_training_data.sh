#!/bin/bash

set -euo pipefail

# FluxFill training data generation (mp4 -> first frame -> SAM2 mask -> Qwen caption)
# Hard-coded config: edit these constants when you want a different slice.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2:${TIMESTAMP}"
TEAM_SPACE="research"
DOMAIN="prod"

# Source GCS prefix (chunk-based folder structure)
SOURCE_GCS_PREFIX="workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov"

# Chunk range: process chunk_0026 to chunk_0049 (inclusive)
# Each chunk folder contains 100+ mp4 files
# Override these with command-line args for parallel processing
CHUNK_START="${1:-26}"
CHUNK_END="${2:-49}"
GPU_ID="${3:-0}"
NUM_WORKERS="${4:-8}"

NUM_VIDEOS="10000"
START_INDEX="0"
OUTPUT_RUN_ID="trainingdata_chunk${CHUNK_START}_${CHUNK_END}"

# Frame numbers (1-based) to extract per video. Comma-separated.
FRAME_NUMBERS="1,250,500"

QWEN_DEVICE="cuda:${GPU_ID}"
SAM2_DEVICE="cuda:${GPU_ID}"

echo "Processing chunks ${CHUNK_START} to ${CHUNK_END} with ${NUM_WORKERS} parallel workers"
echo "Output will be: trainingdata_chunk${CHUNK_START}_${CHUNK_END}"

# 0 means: keep walking the GCS prefix until enough .mp4s are found (or the prefix is exhausted).
MAX_WALK_FILES="0"

# Ensure we run from the sam2 repo root (docker-compose.yaml lives there)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Building docker image (sam2/frontend)..."
docker compose build

echo "Tagging and pushing image: ${REMOTE_IMAGE}"
docker tag sam2/frontend "${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE}"

echo "Image built with timestamp: ${TIMESTAMP}"

# Ensure both workflows use the just-pushed image
export SAM2_CONTAINER_IMAGE="${REMOTE_IMAGE}"
export FLUXFILL_DATA_CONTAINER_IMAGE="${REMOTE_IMAGE}"

echo "Running HLX workflow: data_generation.fluxfill_data_generation_wf"
echo "  SOURCE_GCS_PREFIX=${SOURCE_GCS_PREFIX}"
echo "  NUM_VIDEOS=${NUM_VIDEOS} START_INDEX=${START_INDEX} OUTPUT_RUN_ID=${OUTPUT_RUN_ID}"
echo "  FRAME_NUMBERS=${FRAME_NUMBERS}"
echo "  CHUNK_START=${CHUNK_START} CHUNK_END=${CHUNK_END}"
echo "  NUM_WORKERS=${NUM_WORKERS}"

hlx wf run \
	--team-space "${TEAM_SPACE}" \
	--domain "${DOMAIN}" \
	data_generation.fluxfill_data_generation_wf \
	--source_gcs_prefix "${SOURCE_GCS_PREFIX}" \
	--num_videos "${NUM_VIDEOS}" \
	--start_index "${START_INDEX}" \
	--output_run_id "${OUTPUT_RUN_ID}" \
	--qwen_device "${QWEN_DEVICE}" \
	--sam2_device "${SAM2_DEVICE}" \
	--max_walk_files "${MAX_WALK_FILES}" \
	--frame_numbers "${FRAME_NUMBERS}" \
	--chunk_start "${CHUNK_START}" \
	--chunk_end "${CHUNK_END}" \
	--num_workers "${NUM_WORKERS}"

echo "If successful, data is under:"
echo "  gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data/${OUTPUT_RUN_ID}/"

