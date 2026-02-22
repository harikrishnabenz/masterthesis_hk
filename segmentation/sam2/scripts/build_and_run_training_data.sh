#!/bin/bash

set -euo pipefail

# FluxFill training data generation (mp4 -> first frame -> SAM2 mask -> Qwen caption)
# Optimised for A100_80GB_4GPU: 48 CPUs, 680 GB RAM, 4 × A100 80 GB.
# Each GPU runs SAM2 (~3 GB) + Qwen-7B-bf16 (~14 GB) ≈ 17 GB, leaving ~63 GB headroom.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2:${TIMESTAMP}"
TEAM_SPACE="research"
DOMAIN="prod"

# ---------- Source GCS prefix (new Input path, single chunk_0000) ----------
SOURCE_GCS_PREFIX="workspace/user/hbaskar/Input/data_physical_ai/camera_front_tele_30fov"

# Chunk range: only chunk_0000
CHUNK_START="${1:-0}"
CHUNK_END="${2:-99}"

# Multi-GPU / multiprocessing settings (A100_80GB_4GPU → 4 GPUs, 1 worker per GPU)
NUM_GPUS="${3:-4}"
NUM_WORKERS="${4:-4}"

# 0 = process ALL videos found in the chunk(s), no cap.
NUM_VIDEOS="0"
START_INDEX="0"
OUTPUT_RUN_ID="td-chunk${CHUNK_START}-${CHUNK_END}"

# Frame numbers (1-based) to extract per video. Comma-separated.
FRAME_NUMBERS="1,250,500"

# With multi-GPU, each worker auto-selects its GPU via round-robin (vid_idx % num_gpus).
# These defaults are only used when the workflow falls back to single-device mode.
QWEN_DEVICE="cuda:0"
SAM2_DEVICE="cuda:0"

echo "============================================================"
echo "Processing chunks ${CHUNK_START}..${CHUNK_END}"
echo "  GPUs  : ${NUM_GPUS} × A100 80 GB"
echo "  Workers: ${NUM_WORKERS} (1 per GPU recommended)"
echo "  Output : ${OUTPUT_RUN_ID}"
echo "============================================================"

# 0 = keep walking the GCS prefix until enough .mp4s are found (or exhausted).
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
echo "  NUM_GPUS=${NUM_GPUS}  NUM_WORKERS=${NUM_WORKERS}"

hlx wf run \
	--team-space "${TEAM_SPACE}" \
	--domain "${DOMAIN}" \
	--execution-name "datagen-${OUTPUT_RUN_ID}-$(echo ${TIMESTAMP} | tr '_' '-')" \
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
	--num_gpus "${NUM_GPUS}" \
	--num_workers "${NUM_WORKERS}"

echo "If successful, data is under:"
echo "  gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data/${OUTPUT_RUN_ID}/"

