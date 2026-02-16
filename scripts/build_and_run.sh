#!/bin/bash
# Master workflow build and run script
# Builds the orchestrator container, pushes to GCP, and runs the master workflow via HLX

set -euo pipefail

echo "================================================================================"
echo "MASTER WORKFLOW - BUILD AND RUN"
echo "================================================================================"

# ----------------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------------
# Generate a single timestamp for this entire pipeline run
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ----------------------------------------------------------------------------------
# OUTPUT PATHS
# ----------------------------------------------------------------------------------
# Base output folder in GCS (all pipeline outputs go here)
OUTPUT_BASE="gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/output"

# Per-stage output sub-folders
SAM2_OUTPUT_DIR="${OUTPUT_BASE}/sam2"
VP_OUTPUT_DIR="${OUTPUT_BASE}/vp"
ALPAMAYO_OUTPUT_DIR="${OUTPUT_BASE}/alpamayo"

# SAM2 configuration
SAM2_RUN_ID="${SAM2_RUN_ID:-experiment_01_${RUN_TIMESTAMP}}"
SAM2_MAX_FRAMES="${SAM2_MAX_FRAMES:-150}"
# Video URIs (comma-separated list or "default" to use default videos)
# Default: First mp4 from each of 20 chunks (chunk_0000 to chunk_0019)
SAM2_VIDEO_URIS="${SAM2_VIDEO_URIS:-gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0000/01d3588e-bca7-4a18-8e74-c6cfe9e996db.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0001/03560c1a-eac6-499d-bd81-90a19b8d66bd.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0002/043d0d6f-f357-43d6-a819-246551190ac0.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0003/02fd3a17-5936-4098-849f-50ae34d07370.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0004/002dec8e-3d95-4cc2-abbe-99b3a2e78618.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0005/014a6337-e5ff-42fc-9487-49e674b4cac7.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0006/01c6b380-30b5-4565-bc15-2607da5b671f.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0007/04bb5e8e-5e8b-447b-87e1-b2911bae99a9.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0008/031bc104-aadb-4a02-9611-9f49be421c0f.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0009/01c46b6b-fe98-4754-98e1-7010d294bff4.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0010/0109a518-69a0-4355-afde-5e617e06415e.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0011/08ff1470-ac35-49ad-8da6-85c30c8c63bf.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0012/0211f97c-d416-487d-b402-eb97c8f39c64.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0013/0019ded5-41bf-48da-9f66-0e46e31e234e.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0014/0064ce6d-5f40-417f-915f-87c6643e0de4.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0015/01b5167c-0fa2-499b-b761-f6ed78897d22.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0016/00d0b3a1-620c-4c51-af8d-2d1681a86249.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0017/00d4efa8-62a5-47ce-b8c7-e2a6b67b8f5c.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0018/02fd6260-73e6-435a-b8a0-a1536840dec6.camera_front_tele_30fov.mp4,gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai/camera_front_tele_30fov/chunk_0019/00475fb6-cb94-4633-a350-2b1136394123.camera_front_tele_30fov.mp4}"

# VideoPainter configuration
VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-${SAM2_RUN_ID}}"
VP_INSTRUCTION="${VP_INSTRUCTION:-$'Single solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nDouble solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nSingle solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nDouble solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nSingle dashed white intermitted line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'}"
VP_NUM_SAMPLES="${VP_NUM_SAMPLES:-1}"

# Alpamayo configuration
ALPAMAYO_RUN_ID="${ALPAMAYO_RUN_ID:-experiment_01_${RUN_TIMESTAMP}}"
ALPAMAYO_NUM_TRAJ_SAMPLES="${ALPAMAYO_NUM_TRAJ_SAMPLES:-1}"

echo ""
echo "Workflow Configuration:"
echo "  OUTPUT_BASE: ${OUTPUT_BASE}"
echo "  SAM2_OUTPUT_DIR: ${SAM2_OUTPUT_DIR}"
echo "  VP_OUTPUT_DIR: ${VP_OUTPUT_DIR}"
echo "  ALPAMAYO_OUTPUT_DIR: ${ALPAMAYO_OUTPUT_DIR}"
echo "  SAM2_RUN_ID: ${SAM2_RUN_ID}"
echo "  SAM2_MAX_FRAMES: ${SAM2_MAX_FRAMES}"
echo "  SAM2_VIDEO_URIS: ${SAM2_VIDEO_URIS}"
echo "  VP_DATA_RUN_ID: ${VP_DATA_RUN_ID}"
echo "  VP_INSTRUCTION: ${VP_INSTRUCTION}"
echo "  VP_NUM_SAMPLES: ${VP_NUM_SAMPLES}"
echo "  ALPAMAYO_RUN_ID: ${ALPAMAYO_RUN_ID}"
echo "  ALPAMAYO_NUM_TRAJ_SAMPLES: ${ALPAMAYO_NUM_TRAJ_SAMPLES}"
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

# Export for workflow to use
export MASTER_CONTAINER_IMAGE="${REMOTE_IMAGE}"

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
  workflow.master_pipeline_wf \
  --sam2_run_id "${SAM2_RUN_ID}" \
  --sam2_video_uris "${SAM2_VIDEO_URIS}" \
  --sam2_max_frames "${SAM2_MAX_FRAMES}" \
  --vp_data_run_id "${VP_DATA_RUN_ID}" \
  --vp_instruction "${VP_INSTRUCTION}" \
  --vp_num_samples "${VP_NUM_SAMPLES}" \
  --alpamayo_run_id "${ALPAMAYO_RUN_ID}" \
  --alpamayo_num_traj_samples "${ALPAMAYO_NUM_TRAJ_SAMPLES}"

echo ""
echo "================================================================================"
echo "MASTER WORKFLOW SUBMITTED"
echo "================================================================================"
echo ""
echo "The workflow has been submitted to HLX and will execute all three stages"
echo "sequentially. Each stage will use its own pre-built Docker image:"
echo ""
echo "  • SAM2: Built via segmentation/sam2/scripts/build_and_run.sh"
echo "  • VideoPainter: Built via generation/VideoPainter/scripts/build_and_run.sh"
echo "  • Alpamayo: Built via vla/alpamayo/scripts/build_and_run.sh"
echo ""
echo "Monitor workflow progress:"
echo "  hlx wf logs <workflow-id>"
echo ""
echo "Output locations will be:"
echo "  • SAM2: ${SAM2_OUTPUT_DIR}/${SAM2_RUN_ID}/"
echo "  • VideoPainter: ${VP_OUTPUT_DIR}/${VP_DATA_RUN_ID}_*/"
echo "  • Alpamayo: ${ALPAMAYO_OUTPUT_DIR}/${ALPAMAYO_RUN_ID}/"
echo ""
echo "================================================================================"
