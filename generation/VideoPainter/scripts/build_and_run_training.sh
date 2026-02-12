#!/bin/bash

set -euo pipefail

#+#+#+#+--------------------------------------------------------------------------
# VideoPainter FluxFill LoRA training (FluxFill CSV -> LoRA checkpoints)
#
# Keep image naming/tagging consistent with scripts/build_and_run.sh:
# - Image name uses a run suffix (X)
# - Push a unique timestamp tag to avoid stale ':latest'
# - Export VP_CONTAINER_IMAGE to the exact timestamp tag
#+#+#+#+--------------------------------------------------------------------------

TEAM_SPACE="research"
DOMAIN="prod"

# Declare a run suffix used by both this script and the HLX workflow.
# Default matches scripts/build_and_run.sh.
# Override by: TRAIN_RUN_SUFFIX="..." bash scripts/build_and_run_training.sh
X="${TRAIN_RUN_SUFFIX:-5p_10v_p2_s5_11226_1}"
export VP_RUN_SUFFIX="${X}"

REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_vp${X}"

# Input dataset (filtered output from segmentation/sam2/filter_fluxfill_dataset.py)
# Override example:
#   INPUT_DATA_DIR="gs://.../training/data/my_dataset" bash scripts/build_and_run_training.sh
INPUT_DATA_DIR="${INPUT_DATA_DIR:-mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data/trainingdata_chunk199__chunk199_single_white_solid}"

# Output checkpoint base prefix (workflow will create a subfolder per run_id)
# Override example:
#   OUTPUT_CHECKPOINT_DIR="gs://.../training/trained_checkpoint" bash scripts/build_and_run_training.sh
OUTPUT_CHECKPOINT_DIR="${OUTPUT_CHECKPOINT_DIR:-gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/trained_checkpoint}"

# Run naming
# Override example:
#   OUTPUT_RUN_ID="fluxfill_experiment_01" bash scripts/build_and_run_training.sh
OUTPUT_RUN_ID="${OUTPUT_RUN_ID:-fluxfill_single_white_solid_clearroad_$(date -u +%Y%m%d_%H%M%S)}"

# Training hyperparameters (tweak as needed)
MAX_TRAIN_STEPS="1000"
CHECKPOINTING_STEPS="250"
LEARNING_RATE="1e-4"
TRAIN_BATCH_SIZE="1"
GRAD_ACCUM="1"
MIXED_PRECISION="bf16"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Building docker image (VideoPainter)..."
docker compose build

# Tag the image for Google Artifact Registry (use a unique tag to avoid stale ':latest' caching)
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_IMAGE_TAGGED="${REMOTE_IMAGE}:${RUN_TAG}"

echo "Tagging and pushing image: ${REMOTE_IMAGE_TAGGED}"
docker tag videopainter:latest "${REMOTE_IMAGE_TAGGED}"
docker tag videopainter:latest "${REMOTE_IMAGE}:latest"
docker push "${REMOTE_IMAGE_TAGGED}"
docker push "${REMOTE_IMAGE}:latest"

# Ensure workflow uses this exact image tag when hlx packages the workflow.
export VP_CONTAINER_IMAGE="${REMOTE_IMAGE_TAGGED}"

echo "Running HLX workflow: training_workflow.fluxfill_training_wf"
echo "  INPUT_DATA_DIR=${INPUT_DATA_DIR}"
echo "  OUTPUT_CHECKPOINT_DIR=${OUTPUT_CHECKPOINT_DIR}"
echo "  OUTPUT_RUN_ID=${OUTPUT_RUN_ID}"

hlx wf run \
	--team-space "${TEAM_SPACE}" \
	--domain "${DOMAIN}" \
	training_workflow.fluxfill_training_wf \
	--input_data_dir "${INPUT_DATA_DIR}" \
	--output_checkpoint_dir "${OUTPUT_CHECKPOINT_DIR}" \
	--output_run_id "${OUTPUT_RUN_ID}" \
	--mixed_precision "${MIXED_PRECISION}" \
	--train_batch_size "${TRAIN_BATCH_SIZE}" \
	--gradient_accumulation_steps "${GRAD_ACCUM}" \
	--learning_rate "${LEARNING_RATE}" \
	--max_train_steps "${MAX_TRAIN_STEPS}" \
	--checkpointing_steps "${CHECKPOINTING_STEPS}"

echo "If successful, checkpoints are under:"
echo "  ${OUTPUT_CHECKPOINT_DIR}/${OUTPUT_RUN_ID}/"

