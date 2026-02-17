#!/bin/bash
# Master workflow build and run script
# Builds the orchestrator container, pushes to GCP, and runs the master workflow via HLX
#
# PIPELINE FLOW (strictly sequential via barrier tasks):
#
#   [1] SAM2 Segmentation
#       Input:  SAM2_VIDEO_URIS (GCS video paths)
#       Output: gs://.../output/sam2/<SAM2_RUN_ID>/
#               + preprocessed data for VP at gs://.../preprocessed_data_vp/<SAM2_RUN_ID>/
#                          |
#                     (barrier: VP waits for SAM2)
#                          v
#   [2] VideoPainter Editing
#       Input:  SAM2 preprocessed output (located by VP_DATA_RUN_ID = SAM2_RUN_ID)
#       Output: gs://.../output/vp/<VP_DATA_RUN_ID>_*/
#                          |
#                     (barrier: Alpamayo waits for VP)
#                          v
#   [3] Alpamayo VLA Inference
#       Input:  VP output video path (constructed from VP output)
#       Output: gs://.../output/alpamayo/<ALPAMAYO_RUN_ID>/
#

set -euo pipefail

echo "================================================================================"
echo "MASTER WORKFLOW - BUILD AND RUN"
echo "================================================================================"
echo ""
echo "Pipeline: SAM2 --> VideoPainter --> Alpamayo (sequential)"
echo ""

# ----------------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------------
# Generate a single timestamp for this entire pipeline run
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ----------------------------------------------------------------------------------
# RUN ID  (set this to identify your run; combined with timestamp for output folders)
# ----------------------------------------------------------------------------------
RUN_ID="${RUN_ID:-002}"

# ----------------------------------------------------------------------------------
# OUTPUT PATHS
# ----------------------------------------------------------------------------------
# Base output folder in GCS (all pipeline outputs go here)
OUTPUT_BASE="gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs"

# Per-stage output sub-folders
SAM2_OUTPUT_DIR="${OUTPUT_BASE}/sam2"
VP_OUTPUT_DIR="${OUTPUT_BASE}/vp"
ALPAMAYO_OUTPUT_DIR="${OUTPUT_BASE}/alpamayo"

# ----------------------------------------------------------------------------------
# INTER-STAGE DATA PATHS (outputs → inputs)
# ----------------------------------------------------------------------------------
# SAM2 also writes VP-preprocessed data here (separate from its raw output above).
# This is the INPUT for Stage 2 (VideoPainter).
SAM2_PREPROCESSED_DIR="${OUTPUT_BASE}/preprocessed_data_vp"

# Stage 2 output (${VP_OUTPUT_DIR}/<run_id>/) is the INPUT for Stage 3 (Alpamayo).
# The master workflow constructs this path automatically via barrier tasks.

# ----------------------------------------------------------------------------------
# SAM2 CONFIGURATION  (Stage 1)
# ----------------------------------------------------------------------------------
SAM2_RUN_ID="${SAM2_RUN_ID:-${RUN_ID}_${RUN_TIMESTAMP}}"
SAM2_MAX_FRAMES="${SAM2_MAX_FRAMES:-150}"

# ── Input video source ──────────────────────────────────────────────────────
#
# SAM2 input base: parent folder containing camera subfolders.
#   gs://.../Input/data_physical_ai/camera_front_tele_30fov/chunk_0000/<uuid>.mp4
#   gs://.../Input/data_physical_ai/camera_front_tele_30fov/chunk_0001/<uuid>.mp4
#   ...
# The camera subfolder is appended below.
#
# Configure which chunks and how many files per chunk to process:
SAM2_INPUT_PARENT="${SAM2_INPUT_PARENT:-gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Input/data_physical_ai}"
SAM2_CAMERA_SUBFOLDER="${SAM2_CAMERA_SUBFOLDER:-camera_front_tele_30fov}"
SAM2_INPUT_BASE="${SAM2_INPUT_BASE:-${SAM2_INPUT_PARENT}/${SAM2_CAMERA_SUBFOLDER}}"
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

# ----------------------------------------------------------------------------------
# VIDEOPAINTER CONFIGURATION  (Stage 2 - receives SAM2 output)
# ----------------------------------------------------------------------------------
# VP_DATA_RUN_ID links to SAM2's output: VP looks for preprocessed data under this run_id
VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-${SAM2_RUN_ID}}"
VP_OUTPUT_RUN_ID="${VP_OUTPUT_RUN_ID:-${RUN_ID}_${RUN_TIMESTAMP}}"
VP_INSTRUCTION="${VP_INSTRUCTION:-$'Single solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nDouble solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nSingle solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nDouble solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nSingle dashed white intermitted line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'}"
VP_NUM_SAMPLES="${VP_NUM_SAMPLES:-1}"

# ----------------------------------------------------------------------------------
# ALPAMAYO CONFIGURATION  (Stage 3 - receives VP output)
# ----------------------------------------------------------------------------------
ALPAMAYO_RUN_ID="${ALPAMAYO_RUN_ID:-${RUN_ID}_${RUN_TIMESTAMP}}"
ALPAMAYO_NUM_TRAJ_SAMPLES="${ALPAMAYO_NUM_TRAJ_SAMPLES:-1}"
# Model ID: local mounted checkpoint (default) or HuggingFace ID
ALPAMAYO_MODEL_ID="${ALPAMAYO_MODEL_ID:-/workspace/alpamayo/checkpoints/alpamayo-r1-10b}"

echo "Workflow Configuration:"
echo "  RUN_ID:                    ${RUN_ID}"
echo "  RUN_TIMESTAMP:             ${RUN_TIMESTAMP}"
echo "  OUTPUT_BASE:               ${OUTPUT_BASE}"
echo "  SAM2_PREPROCESSED_DIR:     ${SAM2_PREPROCESSED_DIR}"
echo ""
echo "  [Stage 1] SAM2 Segmentation:"
echo "    Input:                   ${SAM2_INPUT_DISPLAY}"
echo "    Output (raw):            ${SAM2_OUTPUT_DIR}/${SAM2_RUN_ID}/"
echo "    Output (preprocessed):   ${SAM2_PREPROCESSED_DIR}/${SAM2_RUN_ID}/"
echo "    SAM2_RUN_ID:             ${SAM2_RUN_ID}"
echo "    SAM2_MAX_FRAMES:         ${SAM2_MAX_FRAMES}"
echo ""
echo "  [Stage 2] VideoPainter Editing:"
echo "    Input:                   ${SAM2_PREPROCESSED_DIR}/${VP_DATA_RUN_ID}/"
echo "    Output:                  ${VP_OUTPUT_DIR}/${VP_OUTPUT_RUN_ID}/"
echo "    VP_DATA_RUN_ID:          ${VP_DATA_RUN_ID}  (= SAM2_RUN_ID)"
echo "    VP_OUTPUT_RUN_ID:        ${VP_OUTPUT_RUN_ID}"
echo "    VP_NUM_SAMPLES:          ${VP_NUM_SAMPLES}"
echo "    VP_INSTRUCTIONS:         $(echo "${VP_INSTRUCTION}" | wc -l) editing instructions"
echo ""
echo "  [Stage 3] Alpamayo VLA Inference:"
echo "    Input:                   ${VP_OUTPUT_DIR}/${VP_OUTPUT_RUN_ID}/  (auto-resolved)"
echo "    Output:                  ${ALPAMAYO_OUTPUT_DIR}/${ALPAMAYO_RUN_ID}/"
echo "    ALPAMAYO_RUN_ID:         ${ALPAMAYO_RUN_ID}"
echo "    ALPAMAYO_NUM_TRAJ_SAMPLES:${ALPAMAYO_NUM_TRAJ_SAMPLES}"
echo "    ALPAMAYO_MODEL_ID:       ${ALPAMAYO_MODEL_ID}"
echo ""

# ----------------------------------------------------------------------------------
# BUILD AND PUSH MASTER CONTAINER
# ----------------------------------------------------------------------------------
echo "================================================================================"
echo "STEP 1: BUILD AND PUSH DOCKER IMAGE"
echo "================================================================================"

# Generate timestamp for unique image tagging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Building master orchestrator Docker image..."
docker compose build master-pipeline

# Tag the image for Google Artifact Registry
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/master_pipeline:${TIMESTAMP}"
REMOTE_IMAGE_LATEST="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/master_pipeline:latest"

echo "Tagging image with timestamp: ${TIMESTAMP}"
docker tag master-pipeline:latest "${REMOTE_IMAGE}"
docker tag master-pipeline:latest "${REMOTE_IMAGE_LATEST}"

echo "Pushing images to Google Artifact Registry..."
docker push "${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE_LATEST}"

echo ""
echo "✓ Docker image built and pushed successfully"
echo "  Timestamped: ${REMOTE_IMAGE}"
echo "  Latest: ${REMOTE_IMAGE_LATEST}"
echo ""

# Export environment variables for workflow.py and _generated_tasks.py
export MASTER_CONTAINER_IMAGE="${REMOTE_IMAGE}"
export SAM2_OUTPUT_BASE="${SAM2_OUTPUT_DIR}"
export SAM2_PREPROCESSED_OUTPUT_BASE="${SAM2_PREPROCESSED_DIR}"
export VP_OUTPUT_BASE="${VP_OUTPUT_DIR}"
export ALPAMAYO_OUTPUT_BASE="${ALPAMAYO_OUTPUT_DIR}"
export SAM2_INPUT_PARENT
export SAM2_CAMERA_SUBFOLDER

# ----------------------------------------------------------------------------------
# RUN MASTER WORKFLOW
# ----------------------------------------------------------------------------------
echo "================================================================================"
echo "STEP 2: RUN MASTER WORKFLOW"
echo "================================================================================"
echo ""
echo "Starting HLX master workflow that chains all three stages:"
echo "  1. SAM2 Segmentation"
echo "  2. VideoPainter Editing"
echo "  3. Alpamayo VLA Inference"
echo ""

hlx wf run \
  --team-space research \
  --domain prod \
  --execution-name "master-${RUN_ID//_/-}-$(date -u +%Y%m%d-%H%M%S)" \
  workflow.master_pipeline_wf \
  --sam2_run_id "${SAM2_RUN_ID}" \
  --sam2_video_uris "${SAM2_VIDEO_URIS}" \
  --sam2_max_frames "${SAM2_MAX_FRAMES}" \
  --vp_data_run_id "${VP_DATA_RUN_ID}" \
  --vp_output_run_id "${VP_OUTPUT_RUN_ID}" \
  --vp_instruction "${VP_INSTRUCTION}" \
  --vp_num_samples "${VP_NUM_SAMPLES}" \
  --alpamayo_run_id "${ALPAMAYO_RUN_ID}" \
  --alpamayo_num_traj_samples "${ALPAMAYO_NUM_TRAJ_SAMPLES}" \
  --alpamayo_model_id "${ALPAMAYO_MODEL_ID}"

echo ""
echo "================================================================================"
echo "MASTER WORKFLOW SUBMITTED"
echo "================================================================================"
echo ""
echo "The pipeline executes SEQUENTIALLY (each stage waits for the previous):"
echo ""
echo "  [1] SAM2 Segmentation"
echo "      Input:  ${SAM2_INPUT_DISPLAY}"
echo "      Output: ${SAM2_OUTPUT_DIR}/${SAM2_RUN_ID}/"
echo "      Output: ${SAM2_PREPROCESSED_DIR}/${SAM2_RUN_ID}/  (→ VP preprocessed)"
echo "                  |"
echo "                  v  (barrier: VP waits for SAM2 to finish)"
echo "  [2] VideoPainter Editing"
echo "      Input:  ${SAM2_PREPROCESSED_DIR}/${VP_DATA_RUN_ID}/"
echo "      Output: ${VP_OUTPUT_DIR}/${VP_OUTPUT_RUN_ID}/"
echo "                  |"
echo "                  v  (barrier: Alpamayo waits for VP to finish)"
echo "  [3] Alpamayo VLA Inference"
echo "      Input:  ${VP_OUTPUT_DIR}/${VP_OUTPUT_RUN_ID}/"
echo "      Output: ${ALPAMAYO_OUTPUT_DIR}/${ALPAMAYO_RUN_ID}/"
echo ""
echo "Monitor workflow progress:"
echo "  hlx wf logs <workflow-id>"
echo ""
echo "================================================================================"
