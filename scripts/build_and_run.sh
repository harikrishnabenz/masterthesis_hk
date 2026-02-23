#!/bin/bash
# ==================================================================================
# MASTER PIPELINE — Build and Run Script
# ==================================================================================
#
# Orchestrates a flexible multi-stage pipeline:
#
#   Stage 1  SAM2 Segmentation
#            Input : raw driving videos (GCS / chunks:// URI)
#            Output: masks + VP-preprocessed data
#
#   Stage 2  VideoPainter Editing
#            Input : SAM2 preprocessed data
#            Output: inpainted / edited videos
#
#   Stage 3  Alpamayo VLA Inference
#            Input : VP edited videos
#            Output: trajectory predictions + reasoning
#
# A single RUN_ID + RUN_TIMESTAMP is shared across all stages.
#
# Usage — select stages with STAGES=<digits>:
#   STAGES=1   bash scripts/build_and_run.sh   # SAM2 only (needs chunk config)
#   STAGES=2   SAM2_DATA_RUN_ID=003_…  bash scripts/build_and_run.sh  # VP only
#   STAGES=3   VP_DATA_RUN_ID=003_…  bash scripts/build_and_run.sh  # Alp only
#   STAGES=12  bash scripts/build_and_run.sh   # SAM2 → VP (needs chunk config)
#   STAGES=23  SAM2_DATA_RUN_ID=003_…  bash scripts/build_and_run.sh  # VP → Alp
#   STAGES=123 bash scripts/build_and_run.sh   # Full pipeline (needs chunk config)
#
# Other options:
#   RUN_ID=002 bash scripts/build_and_run.sh   # custom run id
# ==================================================================================

set -euo pipefail

# ── Resolve repo root (this script lives in scripts/) ────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ==============================================================================
# SHARED CONFIGURATION
# ==============================================================================
GCS_BUCKET="mbadas-sandbox-research-9bb9c7f"

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${RUN_ID:-06_p5_lora_gen_video}"
MASTER_RUN_ID="${RUN_ID}_${RUN_TIMESTAMP}"






















# ==============================================================================
# INPUT CONFIGURATION — set these before running
# ==============================================================================

# ── Stage selection (1=SAM2, 2=VP, 3=Alpamayo; combine: 12, 23, 123) ─────────
STAGES="${STAGES:-123}"

# ── Stage 1 (SAM2) inputs ────────────────────────────────────────────────────
SAM2_CHUNK_START="${SAM2_CHUNK_START:-100}"
SAM2_CHUNK_END="${SAM2_CHUNK_END:-100}"
SAM2_FILES_PER_CHUNK="${SAM2_FILES_PER_CHUNK:-20}"

# ── Stage 2 (VP) input — required when running VP without SAM2 (STAGES=2,23) ─
SAM2_DATA_RUN_ID="${SAM2_DATA_RUN_ID:-}"

# ── Stage 3 (Alpamayo) input — required when running Alp without VP (STAGES=3)
# ── VP output folders (copy-paste the one you need into VP_DATA_RUN_ID) ───────
#VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-06_p1_100v_20260219_165629}"          # Prompt 1 — single solid white (50 steps)
#VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-06_p2_100v_20260219_165801}"          # Prompt 2 — double solid white (50 steps)
#VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-06_p3_100v_20260219_165938}"          # Prompt 3 — single solid yellow (50 steps)
#VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-06_p4_100v_20260219_170051}"          # Prompt 4 — double solid yellow (50 steps)
#VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-06_p5_100v_20260219_170208}"          # Prompt 5 — single dashed white (50 steps)
#VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-06_p1_100v_vps90_20260219_232441}"    # Prompt 1 — single solid white (90 steps)
#VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-06_p2_100v_vps90_20260219_232331}"    # Prompt 2 — double solid white (90 steps)
#VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-06_p3_100v_vps90_20260219_232214}"    # Prompt 3 — single solid yellow (90 steps)
#VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-06_p4_100v_vps90_20260219_232102}"    # Prompt 4 — double solid yellow (90 steps)
#VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-06_p5_100v_vps90_20260219_231936}"    # Prompt 5 — single dashed white (90 steps)


VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-}"



























# Parse which stages are active
RUN_SAM2=0; RUN_VP=0; RUN_ALP=0
[[ "${STAGES}" == *1* ]] && RUN_SAM2=1
[[ "${STAGES}" == *2* ]] && RUN_VP=1
[[ "${STAGES}" == *3* ]] && RUN_ALP=1

# Validate STAGES contains only 1, 2, 3
if [[ ! "${STAGES}" =~ ^[123]+$ ]]; then
    echo "ERROR: STAGES='${STAGES}' is invalid. Use digits 1, 2, 3 (e.g. 1, 12, 23, 123)."
    exit 1
fi

# ── Input folder dependency checks ───────────────────────────────────────────
if [[ ${RUN_VP} -eq 1 && ${RUN_SAM2} -eq 0 && -z "${SAM2_DATA_RUN_ID}" ]]; then
    echo "ERROR: STAGES=${STAGES} includes VP (2) without SAM2 (1)."
    echo "       You must set SAM2_DATA_RUN_ID (the SAM2 output folder name)."
    echo "       Example: SAM2_DATA_RUN_ID=003_20260217_162441 STAGES=${STAGES} bash scripts/build_and_run.sh"
    exit 1
fi

if [[ ${RUN_ALP} -eq 1 && ${RUN_VP} -eq 0 && -z "${VP_DATA_RUN_ID}" ]]; then
    echo "ERROR: STAGES=${STAGES} includes Alpamayo (3) without VP (2)."
    echo "       You must set VP_DATA_RUN_ID (the VP output folder name)."
    echo "       Example: VP_DATA_RUN_ID=003_20260217_162441 STAGES=${STAGES} bash scripts/build_and_run.sh"
    exit 1
fi

# ==============================================================================
# STAGE 1: SAM2 CONFIGURATION
# ==============================================================================
SAM2_INPUT_PARENT="${SAM2_INPUT_PARENT:-gs://${GCS_BUCKET}/workspace/user/hbaskar/Input/data_physical_ai}"
SAM2_CAMERA_SUBFOLDER="${SAM2_CAMERA_SUBFOLDER:-camera_front_tele_30fov}"
SAM2_INPUT_BASE="${SAM2_INPUT_BASE:-${SAM2_INPUT_PARENT}/${SAM2_CAMERA_SUBFOLDER}}"
SAM2_MAX_FRAMES="${SAM2_MAX_FRAMES:-100}"

SAM2_VIDEO_URIS="${SAM2_VIDEO_URIS:-chunks://${SAM2_INPUT_BASE#gs://}?start=${SAM2_CHUNK_START}&end=${SAM2_CHUNK_END}&per_chunk=${SAM2_FILES_PER_CHUNK}}"

SAM2_OUTPUT_BASE="${SAM2_OUTPUT_BASE:-gs://${GCS_BUCKET}/workspace/user/hbaskar/outputs/sam2}"
SAM2_PREPROCESSED_OUTPUT_BASE="${SAM2_PREPROCESSED_OUTPUT_BASE:-gs://${GCS_BUCKET}/workspace/user/hbaskar/outputs/preprocessed_data_vp}"

# Calculate expected video count for display
SAM2_TOTAL_CHUNKS=$(( SAM2_CHUNK_END - SAM2_CHUNK_START + 1 ))
SAM2_EXPECTED_VIDEOS=$(( SAM2_TOTAL_CHUNKS * SAM2_FILES_PER_CHUNK ))

# ==============================================================================
# STAGE 2: VIDEOPAINTER CONFIGURATION
# ==============================================================================
VP_OUTPUT_BASE="${VP_OUTPUT_BASE:-gs://${GCS_BUCKET}/workspace/user/hbaskar/outputs/vp}"































# ── Trained FluxFill LoRA checkpoints (per-prompt) ────────────────────────────
# Base path under the GCS bucket where all checkpoint folders live.
LORA_CKPT_BASE="workspace/user/hbaskar/Video_inpainting/videopainter/training/trained_checkpoint"

# Per-prompt checkpoint folders.  Keys match the PROMPTS array (1-5).
# Override individual paths:
#   LORA_CKPT_1="workspace/user/.../my_ckpt" bash scripts/build_and_run.sh
declare -A LORA_CHECKPOINTS
LORA_CHECKPOINTS[1]="${LORA_CKPT_1:-${LORA_CKPT_BASE}/fluxfill_single_white_solid_20260222_201003}"    # Prompt 1 — single solid white
LORA_CHECKPOINTS[2]="${LORA_CKPT_2:-${LORA_CKPT_BASE}/fluxfill_double_white_solid_20260222_201003}"    # Prompt 2 — double solid white
LORA_CHECKPOINTS[3]="${LORA_CKPT_3:-${LORA_CKPT_BASE}/fluxfill_single_yellow_solid_20260222_201003}"   # Prompt 3 — single solid yellow
LORA_CHECKPOINTS[4]="${LORA_CKPT_4:-${LORA_CKPT_BASE}/fluxfill_double_yellow_solid_20260222_201003}"   # Prompt 4 — double solid yellow
LORA_CHECKPOINTS[5]="${LORA_CKPT_5:-${LORA_CKPT_BASE}/fluxfill_single_white_dashed_20260222_201003}"   # Prompt 5 — single dashed white

# Legacy single-path fallback (used when lora_scale == 0, i.e. no LoRA).
# When lora_scale > 0 the per-prompt LORA_CHECKPOINTS are used instead.
TRAINED_FLUXFILL_GCS_PATH="${TRAINED_FLUXFILL_GCS_PATH:-${LORA_CHECKPOINTS[1]}}"

# VP run suffix (used in Docker image naming) — set after NUM_PROMPTS is computed below
CHECKPOINT_TIMESTAMP=$(basename "${LORA_CHECKPOINTS[1]}" | grep -oE '[0-9]{8}_[0-9]{6}' | head -1 || true)

# LLM model path inside VP container
VP_LLM_MODEL="${VP_LLM_MODEL:-/workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct}"





























# ----------------------------------------------------------------------------------
# PROMPT SELECTION
# ----------------------------------------------------------------------------------
# Pass PROMPT_IDS to choose which editing prompts to run.  Examples:
#   PROMPT_IDS=1        → prompt 1 only
#   PROMPT_IDS=123      → prompts 1, 2, 3
#   PROMPT_IDS=15       → prompts 1 and 5
#   PROMPT_IDS=12345    → all five (default)
PROMPT_IDS="${PROMPT_IDS:-12345}"






























declare -A PROMPTS
PROMPTS[1]='Single solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'
PROMPTS[2]='Double solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'
PROMPTS[3]='Single solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'
PROMPTS[4]='Double solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'
PROMPTS[5]='Single dashed white intermitted line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'

# Build VIDEO_EDITING_INSTRUCTIONS from selected prompt IDs
VIDEO_EDITING_INSTRUCTIONS=""
for (( i=0; i<${#PROMPT_IDS}; i++ )); do
  pid="${PROMPT_IDS:$i:1}"
  if [[ -z "${PROMPTS[$pid]+x}" ]]; then
    echo "ERROR: Invalid prompt ID '${pid}'. Valid IDs are 1-5."
    exit 1
  fi
  if [[ -n "${VIDEO_EDITING_INSTRUCTIONS}" ]]; then
    VIDEO_EDITING_INSTRUCTIONS+=$'\n'
  fi
  VIDEO_EDITING_INSTRUCTIONS+="${PROMPTS[$pid]}"
done
NUM_PROMPTS=${#PROMPT_IDS}

# Determine LoRA mode (lora_scale > 0 ⇒ per-prompt checkpoint runs)
USE_LORA=0
if command -v bc &>/dev/null; then
    [[ $(echo "${VP_IMG_INPAINTING_LORA_SCALE:-0} > 0" | bc -l 2>/dev/null) -eq 1 ]] && USE_LORA=1
else
    # Fallback: treat anything other than "0" / "0.0" / "0.00" as lora-enabled
    [[ ! "${VP_IMG_INPAINTING_LORA_SCALE:-0}" =~ ^0(\.0+)?$ ]] && USE_LORA=1
fi

if [[ ${USE_LORA} -eq 1 ]]; then
    VP_RUN_SUFFIX="${VP_RUN_SUFFIX:-lora_${NUM_PROMPTS}prompt_${CHECKPOINT_TIMESTAMP}}"
else
    VP_RUN_SUFFIX="${VP_RUN_SUFFIX:-withoutlora_${NUM_PROMPTS}prompt_${CHECKPOINT_TIMESTAMP}}"
fi

# VP inference parameters
VP_NUM_INFERENCE_STEPS="${VP_NUM_INFERENCE_STEPS:-50}"
VP_GUIDANCE_SCALE="${VP_GUIDANCE_SCALE:-6.0}"


VP_STRENGTH="${VP_STRENGTH:-1.0}"


VP_CAPTION_REFINE_ITERS="${VP_CAPTION_REFINE_ITERS:-10}"
VP_CAPTION_REFINE_TEMPERATURE="${VP_CAPTION_REFINE_TEMPERATURE:-0.1}"
# Reduced dilate size to minimize border artifacts during camera movement
VP_DILATE_SIZE="${VP_DILATE_SIZE:-8}"
# Reduced feather size for more precise masking
VP_MASK_FEATHER="${VP_MASK_FEATHER:-4}"
# Enable border-aware masking to prevent black borders during camera movement
VP_BORDER_AWARE_MASKING="${VP_BORDER_AWARE_MASKING:-true}"
# Method for border handling: inpaint (best), blur (fast), interpolate (experimental)
VP_BORDER_METHOD="${VP_BORDER_METHOD:-inpaint}"
VP_KEEP_MASKED_PIXELS="${VP_KEEP_MASKED_PIXELS:-False}"
VP_IMG_INPAINTING_LORA_SCALE="${VP_IMG_INPAINTING_LORA_SCALE:-1.0}"















































VP_SEED="${VP_SEED:-42}"

# ==============================================================================
# STAGE 3: ALPAMAYO CONFIGURATION
# ==============================================================================
ALPAMAYO_OUTPUT_BASE="${ALPAMAYO_OUTPUT_BASE:-workspace/user/hbaskar/outputs/alpamayo}"
ALPAMAYO_MODEL_ID="${ALPAMAYO_MODEL_ID:-/workspace/alpamayo/checkpoints/alpamayo-r1-10b}"
ALPAMAYO_NUM_TRAJ_SAMPLES="${ALPAMAYO_NUM_TRAJ_SAMPLES:-1}"
ALPAMAYO_VIDEO_NAME="${ALPAMAYO_VIDEO_NAME:-auto}"

# If true, replace all non-target camera frames with black (zero) frames so
# the model prediction depends ONLY on the generated front camera video.
# Override: ALPAMAYO_BLACK_NON_TARGET_CAMERAS=true STAGES=3 bash scripts/build_and_run.sh
ALPAMAYO_BLACK_NON_TARGET_CAMERAS="${ALPAMAYO_BLACK_NON_TARGET_CAMERAS:-true}"

# HuggingFace token (needed by Alpamayo for ego-motion data)
HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "${HF_TOKEN}" ]]; then
    echo "WARNING: HF_TOKEN not set. Alpamayo may need it for ego-motion data."
    echo "         Set it with: HF_TOKEN=hf_xxx bash scripts/build_and_run.sh"
fi

# ==============================================================================
# DOCKER IMAGE REGISTRY
# ==============================================================================
REGISTRY="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research"

SAM2_LOCAL_IMAGE="sam2/frontend"
SAM2_REMOTE_IMAGE="${REGISTRY}/harimt_sam2_${MASTER_RUN_ID}"

VP_LOCAL_IMAGE="videopainter:latest"
VP_REMOTE_IMAGE="${REGISTRY}/harimt_vp${VP_RUN_SUFFIX}_${MASTER_RUN_ID}"

ALP_LOCAL_IMAGE="alpamayo:latest"
ALP_REMOTE_IMAGE="${REGISTRY}/alpamayo_vla_${MASTER_RUN_ID}"

MASTER_LOCAL_IMAGE="master-pipeline:latest"
MASTER_REMOTE_IMAGE="${REGISTRY}/master_pipeline_${MASTER_RUN_ID}"

# Tagged with the shared run id
SAM2_TAGGED="${SAM2_REMOTE_IMAGE}:${MASTER_RUN_ID}"
VP_TAGGED="${VP_REMOTE_IMAGE}:${MASTER_RUN_ID}"
ALP_TAGGED="${ALP_REMOTE_IMAGE}:${MASTER_RUN_ID}"
MASTER_TAGGED="${MASTER_REMOTE_IMAGE}:${MASTER_RUN_ID}"

# ==============================================================================
# PRINT CONFIGURATION SUMMARY
# ==============================================================================
# Build a human-readable label for the selected stages
STAGES_LABEL=""
[[ ${RUN_SAM2} -eq 1 ]] && STAGES_LABEL="SAM2"
[[ ${RUN_VP} -eq 1 ]]   && STAGES_LABEL="${STAGES_LABEL:+${STAGES_LABEL} → }VideoPainter"
[[ ${RUN_ALP} -eq 1 ]]  && STAGES_LABEL="${STAGES_LABEL:+${STAGES_LABEL} → }Alpamayo"

echo "================================================================================"
echo " MASTER PIPELINE — BUILD AND RUN"
echo "================================================================================"
echo ""
echo "  RUN_ID:             ${RUN_ID}"
echo "  MASTER_RUN_ID:      ${MASTER_RUN_ID}"
echo "  TIMESTAMP:          ${RUN_TIMESTAMP}"
echo "  STAGES:             ${STAGES}  (${STAGES_LABEL})"
if [[ ${RUN_VP} -eq 1 && ${RUN_SAM2} -eq 0 ]]; then
    echo "  SAM2_DATA_RUN_ID:   ${SAM2_DATA_RUN_ID}  (reusing SAM2 output)"
fi
if [[ ${RUN_ALP} -eq 1 && ${RUN_VP} -eq 0 ]]; then
    echo "  VP_DATA_RUN_ID:     ${VP_DATA_RUN_ID}  (reusing VP output)"
fi
echo ""

if [[ ${RUN_SAM2} -eq 1 ]]; then
    echo " ── Stage 1: SAM2 Segmentation ──────────────────────────────────────────────"
    echo "  Input:              chunks ${SAM2_CHUNK_START}–${SAM2_CHUNK_END} (${SAM2_TOTAL_CHUNKS} chunks × ${SAM2_FILES_PER_CHUNK} files ≈ ${SAM2_EXPECTED_VIDEOS} videos)"
    echo "  Input base:         ${SAM2_INPUT_BASE}"
    echo "  Max frames:         ${SAM2_MAX_FRAMES}"
    echo "  SAM2 output:        ${SAM2_OUTPUT_BASE}/${MASTER_RUN_ID}/"
    echo "  Preprocessed (→VP): ${SAM2_PREPROCESSED_OUTPUT_BASE}/${MASTER_RUN_ID}/"
    echo "  Docker image:       ${SAM2_TAGGED}"
    echo ""
fi

if [[ ${RUN_VP} -eq 1 ]]; then
    echo " ── Stage 2: VideoPainter Editing ────────────────────────────────────────────"
    if [[ ${RUN_SAM2} -eq 1 ]]; then
        echo "  Input (from SAM2):  ${SAM2_PREPROCESSED_OUTPUT_BASE}/${MASTER_RUN_ID}/  (auto from Stage 1)"
    else
        echo "  Input (SAM2 data):  ${SAM2_PREPROCESSED_OUTPUT_BASE}/${SAM2_DATA_RUN_ID}/"
    fi
    echo "  VP output:          ${VP_OUTPUT_BASE}/${MASTER_RUN_ID}/"
    echo "  LoRA mode:          $(if [[ ${USE_LORA} -eq 1 ]]; then echo "ON (scale=${VP_IMG_INPAINTING_LORA_SCALE})"; else echo "OFF"; fi)"
    if [[ ${USE_LORA} -eq 1 ]]; then
        echo "  LoRA checkpoints:"
        for (( _i=0; _i<${#PROMPT_IDS}; _i++ )); do
            _pid="${PROMPT_IDS:$_i:1}"
            echo "    Prompt ${_pid}:       ${LORA_CHECKPOINTS[$_pid]}"
        done
    else
        echo "  FluxFill ckpt:      ${TRAINED_FLUXFILL_GCS_PATH}"
    fi
    echo "  Inference steps:    ${VP_NUM_INFERENCE_STEPS}"
    echo "  Guidance scale:     ${VP_GUIDANCE_SCALE}"
    echo "  Strength:           ${VP_STRENGTH}"
    echo "  Refine iters:       ${VP_CAPTION_REFINE_ITERS}"
    echo "  Docker image:       ${VP_TAGGED}"
    echo ""
fi

if [[ ${RUN_ALP} -eq 1 ]]; then
    echo " ── Stage 3: Alpamayo VLA Inference ──────────────────────────────────────────"
    if [[ ${RUN_VP} -eq 1 ]]; then
        echo "  Input (from VP):    ${VP_OUTPUT_BASE}/${MASTER_RUN_ID}/  (auto from Stage 2)"
    else
        echo "  Input (VP data):    ${VP_OUTPUT_BASE}/${VP_DATA_RUN_ID}/"
    fi
    echo "  Output:             gs://${GCS_BUCKET}/${ALPAMAYO_OUTPUT_BASE}/${MASTER_RUN_ID}/"
    echo "  Model:              ${ALPAMAYO_MODEL_ID}"
    echo "  Traj samples:       ${ALPAMAYO_NUM_TRAJ_SAMPLES}"
    echo "  Video name filter:  ${ALPAMAYO_VIDEO_NAME}"
    echo "  Black non-target:   ${ALPAMAYO_BLACK_NON_TARGET_CAMERAS}"
    echo "  Docker image:       ${ALP_TAGGED}"
    echo ""
fi

echo "================================================================================"

# ==============================================================================
# BUILD & PUSH DOCKER IMAGES
# ==============================================================================
echo ""
echo "Building and pushing Docker images …"
echo ""

# ── Stage 1: SAM2 ─────────────────────────────────────────────────────────
if [[ ${RUN_SAM2} -eq 1 ]]; then
    echo "▸ Building SAM2 image …"
    pushd segmentation/sam2 > /dev/null
    docker compose build
    docker tag "${SAM2_LOCAL_IMAGE}" "${SAM2_TAGGED}"
    docker tag "${SAM2_LOCAL_IMAGE}" "${SAM2_REMOTE_IMAGE}:latest"
    docker push "${SAM2_TAGGED}"
    docker push "${SAM2_REMOTE_IMAGE}:latest"
    popd > /dev/null
    echo "  ✓ SAM2 image pushed: ${SAM2_TAGGED}"
else
    echo "  ⏭ SAM2 build skipped (not in STAGES=${STAGES})"
fi

# ── Stage 2: VideoPainter ─────────────────────────────────────────────────
if [[ ${RUN_VP} -eq 1 ]]; then
    echo "▸ Building VideoPainter image …"
    pushd generation/VideoPainter > /dev/null
    docker compose build
    docker tag "${VP_LOCAL_IMAGE}" "${VP_TAGGED}"
    docker tag "${VP_LOCAL_IMAGE}" "${VP_REMOTE_IMAGE}:latest"
    docker push "${VP_TAGGED}"
    docker push "${VP_REMOTE_IMAGE}:latest"
    popd > /dev/null
    echo "  ✓ VP image pushed: ${VP_TAGGED}"
else
    echo "  ⏭ VP build skipped (not in STAGES=${STAGES})"
fi

# ── Stage 3: Alpamayo ─────────────────────────────────────────────────────
if [[ ${RUN_ALP} -eq 1 ]]; then
    echo "▸ Building Alpamayo image …"
    pushd vla/alpamayo > /dev/null
    docker compose build
    docker tag "${ALP_LOCAL_IMAGE}" "${ALP_TAGGED}"
    docker tag "${ALP_LOCAL_IMAGE}" "${ALP_REMOTE_IMAGE}:latest"
    docker push "${ALP_TAGGED}"
    docker push "${ALP_REMOTE_IMAGE}:latest"
    popd > /dev/null
    echo "  ✓ Alpamayo image pushed: ${ALP_TAGGED}"
else
    echo "  ⏭ Alpamayo build skipped (not in STAGES=${STAGES})"
fi

# ── Master orchestrator (only needed for multi-stage pipelines) ───────────
if [[ ${#STAGES} -gt 1 ]]; then
    echo "▸ Building Master orchestrator image …"
    docker compose build
    docker tag "${MASTER_LOCAL_IMAGE}" "${MASTER_TAGGED}"
    docker tag "${MASTER_LOCAL_IMAGE}" "${MASTER_REMOTE_IMAGE}:latest"
    docker push "${MASTER_TAGGED}"
    docker push "${MASTER_REMOTE_IMAGE}:latest"
    echo "  ✓ Master image pushed: ${MASTER_TAGGED}"
else
    echo "  ⏭ Master orchestrator build skipped (single-stage STAGES=${STAGES})"
fi

# ==============================================================================
# EXPORT ENV VARS FOR WORKFLOW SERIALISATION
# ==============================================================================
export SAM2_CONTAINER_IMAGE="${SAM2_TAGGED}"
export VP_CONTAINER_IMAGE="${VP_TAGGED}"
export ALPAMAYO_CONTAINER_IMAGE="${ALP_TAGGED}"
export SAM2_OUTPUT_BASE
export SAM2_PREPROCESSED_OUTPUT_BASE
export VP_OUTPUT_BASE
export TRAINED_FLUXFILL_GCS_PATH
export ALPAMAYO_OUTPUT_BASE
export HF_TOKEN
export STAGES

# ==============================================================================
# SUBMIT MASTER WORKFLOW
# ==============================================================================
echo ""
echo "================================================================================"
echo " LAUNCHING MASTER WORKFLOW"
echo "================================================================================"
echo ""

# ── Helper: build VP_COMMON_ARGS for a given prompt + lora scale ────────────
# Usage: build_vp_args <instructions_text> <lora_scale>
build_vp_args() {
    local _instr="$1"
    local _lora="$2"
    VP_COMMON_ARGS=(
        --vp_video_editing_instructions "${_instr}"
        --vp_llm_model "${VP_LLM_MODEL}"
        --vp_num_inference_steps "${VP_NUM_INFERENCE_STEPS}"
        --vp_guidance_scale "${VP_GUIDANCE_SCALE}"
        --vp_strength "${VP_STRENGTH}"
        --vp_caption_refine_iters "${VP_CAPTION_REFINE_ITERS}"
        --vp_caption_refine_temperature "${VP_CAPTION_REFINE_TEMPERATURE}"
        --vp_dilate_size "${VP_DILATE_SIZE}"
        --vp_mask_feather "${VP_MASK_FEATHER}"
        --vp_img_inpainting_lora_scale "${_lora}"
        --vp_seed "${VP_SEED}"
    )
    if [[ "${VP_KEEP_MASKED_PIXELS}" =~ ^[Tt]rue$ ]]; then
        VP_COMMON_ARGS+=(--vp_keep_masked_pixels)
    fi
}

# ── Helper: common Alpamayo arguments ───────────────────────────────────────
ALP_COMMON_ARGS=(
    --alp_model_id "${ALPAMAYO_MODEL_ID}"
    --alp_num_traj_samples "${ALPAMAYO_NUM_TRAJ_SAMPLES}"
    --alp_video_name "${ALPAMAYO_VIDEO_NAME}"
)
# Only add the boolean flag when true (omitting it = False for HLX bool params)
if [[ "${ALPAMAYO_BLACK_NON_TARGET_CAMERAS}" =~ ^[Tt]rue$ ]]; then
    ALP_COMMON_ARGS+=(--alp_black_non_target_cameras)
fi

# ══════════════════════════════════════════════════════════════════════════════
# submit_workflow — submit a single workflow run
#
# Globals read: STAGES, SAM2_VIDEO_URIS, SAM2_MAX_FRAMES, SAM2_DATA_RUN_ID,
#               VP_DATA_RUN_ID, VP_OUTPUT_BASE, VP_COMMON_ARGS, ALP_COMMON_ARGS
# ══════════════════════════════════════════════════════════════════════════════
submit_workflow() {
    local _run_id="$1"         # unique run id for this submission
    local _prompt_tag="$2"     # short tag for execution-name (e.g. "p3")

    case "${STAGES}" in
        1)
            echo "Submitting SAM2-only pipeline …"
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "sam2-${_run_id//_/-}" \
              workflow_master.sam2_only_wf \
              --run_id "${_run_id}" \
              --sam2_video_uris "${SAM2_VIDEO_URIS}" \
              --sam2_max_frames "${SAM2_MAX_FRAMES}"
            ;;
        2)
            echo "  Using SAM2 data from run: ${SAM2_DATA_RUN_ID}"
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "vp-${_prompt_tag}-${_run_id//_/-}" \
              workflow_master.vp_only_wf \
              --run_id "${_run_id}" \
              --sam2_data_run_id "${SAM2_DATA_RUN_ID}" \
              "${VP_COMMON_ARGS[@]}"
            ;;
        3)
            echo "  Using VP data run: ${VP_DATA_RUN_ID}"
            local _vp_gcs="${VP_OUTPUT_BASE}/${VP_DATA_RUN_ID}/"
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "alp-${_run_id//_/-}" \
              workflow_master.alpamayo_only_wf \
              --run_id "${_run_id}" \
              --vp_output_gcs_path "${_vp_gcs}" \
              "${ALP_COMMON_ARGS[@]}"
            ;;
        12)
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "sam2-vp-${_prompt_tag}-${_run_id//_/-}" \
              workflow_master.sam2_vp_wf \
              --run_id "${_run_id}" \
              --sam2_video_uris "${SAM2_VIDEO_URIS}" \
              --sam2_max_frames "${SAM2_MAX_FRAMES}" \
              "${VP_COMMON_ARGS[@]}"
            ;;
        23)
            echo "  Using SAM2 data from run: ${SAM2_DATA_RUN_ID}"
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "vp-alp-${_prompt_tag}-${_run_id//_/-}" \
              workflow_master.vp_alpamayo_wf \
              --run_id "${_run_id}" \
              --sam2_data_run_id "${SAM2_DATA_RUN_ID}" \
              "${VP_COMMON_ARGS[@]}" \
              "${ALP_COMMON_ARGS[@]}"
            ;;
        123)
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "master-${_prompt_tag}-${_run_id//_/-}" \
              workflow_master.master_pipeline_wf \
              --run_id "${_run_id}" \
              --sam2_video_uris "${SAM2_VIDEO_URIS}" \
              --sam2_max_frames "${SAM2_MAX_FRAMES}" \
              "${VP_COMMON_ARGS[@]}" \
              "${ALP_COMMON_ARGS[@]}"
            ;;
        *)
            echo "ERROR: STAGES='${STAGES}' is not a supported combination."
            echo "       Supported: 1, 2, 3, 12, 23, 123"
            exit 1
            ;;
    esac
}

# ══════════════════════════════════════════════════════════════════════════════
# DISPATCH: LoRA mode (per-prompt separate runs) vs standard mode (single run)
#
# LoRA mode (lora_scale > 0):
#   SAM2 runs ONCE (shared), then VP (+Alp) fans out per prompt/checkpoint.
#   STAGES=123 → 1× sam2_only_wf  +  N× vp_alpamayo_wf
#   STAGES=12  → 1× sam2_only_wf  +  N× vp_only_wf
#   STAGES=23  → N× vp_alpamayo_wf  (SAM2 already done)
#   STAGES=2   → N× vp_only_wf      (SAM2 already done)
#   STAGES=3   → N× alpamayo_only_wf (VP already done per prompt)
#
# Standard mode (lora_scale == 0):
#   All prompts bundled into a single workflow submission.
# ══════════════════════════════════════════════════════════════════════════════
SUBMITTED_RUN_IDS=()
SAM2_LORA_RUN_ID=""           # set when SAM2 is submitted in LoRA mode

if [[ ${USE_LORA} -eq 1 && ${RUN_VP} -eq 1 ]]; then
    # ── LoRA mode: SAM2 once, then VP (+Alp) per prompt ──────────────────────
    echo "LoRA mode ON (scale=${VP_IMG_INPAINTING_LORA_SCALE})"
    echo ""

    # ── Step 1: Submit SAM2 once (if included in STAGES) ─────────────────────
    if [[ ${RUN_SAM2} -eq 1 ]]; then
        echo "▶ Submitting SAM2 once (shared across all ${NUM_PROMPTS} LoRA prompt runs) …"
        SAM2_LORA_RUN_ID="${MASTER_RUN_ID}"
        _saved_stages="${STAGES}"
        STAGES="1"
        submit_workflow "${SAM2_LORA_RUN_ID}" "sam2"
        STAGES="${_saved_stages}"
        SUBMITTED_RUN_IDS+=("${SAM2_LORA_RUN_ID} (SAM2)")
        # Point subsequent VP runs at this SAM2 output
        SAM2_DATA_RUN_ID="${SAM2_LORA_RUN_ID}"
        echo ""
        echo "  ⚠  VP/Alp runs below depend on SAM2 completing first."
        echo "     SAM2 data run ID: ${SAM2_DATA_RUN_ID}"
        echo ""
    fi

    # ── Step 2: Per-prompt VP (+Alp) runs ────────────────────────────────────
    # Remove "1" from STAGES → the per-prompt stages (e.g. 123→23, 12→2, 23→23)
    LORA_PER_PROMPT_STAGES="${STAGES//1/}"

    if [[ -n "${LORA_PER_PROMPT_STAGES}" ]]; then
        _saved_stages="${STAGES}"
        STAGES="${LORA_PER_PROMPT_STAGES}"

        echo "▶ Submitting ${NUM_PROMPTS} per-prompt VP/Alp run(s) (STAGES=${STAGES}) …"
        echo ""

        for (( _i=0; _i<${#PROMPT_IDS}; _i++ )); do
            _pid="${PROMPT_IDS:$_i:1}"
            _ckpt="${LORA_CHECKPOINTS[$_pid]}"
            _prompt_text="${PROMPTS[$_pid]}"
            _per_prompt_run_id="${RUN_ID}_p${_pid}_lora_${RUN_TIMESTAMP}"

            echo "────────────────────────────────────────────────────────────────────────────────"
            echo "  Prompt ${_pid}: ${_prompt_text:0:80}…"
            echo "  Checkpoint: ${_ckpt}"
            echo "  Run ID:     ${_per_prompt_run_id}"
            echo "────────────────────────────────────────────────────────────────────────────────"

            # Re-export checkpoint path so workflow_master.py picks it up
            export TRAINED_FLUXFILL_GCS_PATH="${_ckpt}"

            # Build VP args with this single prompt + its LoRA scale
            build_vp_args "${_prompt_text}" "${VP_IMG_INPAINTING_LORA_SCALE}"

            submit_workflow "${_per_prompt_run_id}" "p${_pid}"
            SUBMITTED_RUN_IDS+=("${_per_prompt_run_id}")
            echo ""
        done

        STAGES="${_saved_stages}"
    fi

else
    # ── Standard mode (no LoRA, or LoRA without VP stage) ────────────────────
    build_vp_args "${VIDEO_EDITING_INSTRUCTIONS}" "${VP_IMG_INPAINTING_LORA_SCALE}"
    submit_workflow "${MASTER_RUN_ID}" "p${PROMPT_IDS}"
    SUBMITTED_RUN_IDS+=("${MASTER_RUN_ID}")
fi

echo ""
echo "================================================================================"
echo " WORKFLOW(S) SUBMITTED SUCCESSFULLY"
echo "================================================================================"
echo ""
if [[ ${USE_LORA} -eq 1 && ${RUN_VP} -eq 1 ]]; then
    echo "  Mode:                 LoRA (scale=${VP_IMG_INPAINTING_LORA_SCALE})"
    echo "  Submissions:          ${#SUBMITTED_RUN_IDS[@]} total"
    echo ""
    for _rid in "${SUBMITTED_RUN_IDS[@]}"; do
        echo "    • ${_rid}"
    done
    echo ""
else
    echo "  Shared RUN_ID:        ${MASTER_RUN_ID}"
fi
echo "  Stages:               ${STAGES}  (${STAGES_LABEL})"
echo ""
if [[ ${RUN_SAM2} -eq 1 ]]; then
    echo "  Stage 1 — SAM2 (runs once):"
    if [[ -n "${SAM2_LORA_RUN_ID}" ]]; then
        echo "    Run ID:             ${SAM2_LORA_RUN_ID}"
    fi
    echo "    Raw output:         ${SAM2_OUTPUT_BASE}/${SAM2_LORA_RUN_ID:-${MASTER_RUN_ID}}/"
    echo "    Preprocessed (→VP): ${SAM2_PREPROCESSED_OUTPUT_BASE}/${SAM2_LORA_RUN_ID:-${MASTER_RUN_ID}}/"
    echo ""
fi
if [[ ${RUN_VP} -eq 1 ]]; then
    echo "  Stage 2 — VideoPainter:"
    if [[ ${USE_LORA} -eq 1 ]]; then
        for _rid in "${SUBMITTED_RUN_IDS[@]}"; do
            # Skip the SAM2-only entry
            [[ "${_rid}" == *"(SAM2)"* ]] && continue
            echo "    Edited videos:      ${VP_OUTPUT_BASE}/${_rid}/"
        done
    else
        echo "    Edited videos:      ${VP_OUTPUT_BASE}/${MASTER_RUN_ID}/"
    fi
    echo ""
fi
if [[ ${RUN_ALP} -eq 1 ]]; then
    echo "  Stage 3 — Alpamayo:"
    if [[ ${USE_LORA} -eq 1 ]]; then
        for _rid in "${SUBMITTED_RUN_IDS[@]}"; do
            [[ "${_rid}" == *"(SAM2)"* ]] && continue
            echo "    Predictions:        gs://${GCS_BUCKET}/${ALPAMAYO_OUTPUT_BASE}/${_rid}/"
        done
    else
        echo "    Predictions:        gs://${GCS_BUCKET}/${ALPAMAYO_OUTPUT_BASE}/${MASTER_RUN_ID}/"
    fi
    echo ""
fi
echo "================================================================================"
