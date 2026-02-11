

# ----------------------------------------------------------------------------------
# LLM MODEL SELECTION
# ----------------------------------------------------------------------------------
# This repo is configured to use ONLY the Qwen2.5-VL-7B-Instruct checkpoint
# mounted by workflow.py and exposed under the local path below.
LLM_MODEL_PATH="/workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct"


echo "  MODEL_PREFIX: $MODEL_PREFIX"

# Declare a run suffix used by both this script and workflow.py
# The model-size prefix is fixed to 7 (we only use Qwen2.5-VL-7B-Instruct).
X="5p_10v_p2_s5_10226_1"
export VP_RUN_SUFFIX="${X}"

# Build image first (required for the tag below)
docker compose build

# Tag the image for Google Artifact Registry (use a unique tag to avoid stale ':latest' caching)
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_vp${X}"
REMOTE_IMAGE_TAGGED="${REMOTE_IMAGE}:${RUN_TAG}"
docker tag videopainter:latest "${REMOTE_IMAGE_TAGGED}"

# (Optional) Also update :latest for convenience.
docker tag videopainter:latest "${REMOTE_IMAGE}:latest"

# Push the image to the registry
docker push "${REMOTE_IMAGE_TAGGED}"
docker push "${REMOTE_IMAGE}:latest"

# Ensure workflow.py uses this exact image tag when hlx packages the workflow.
export VP_CONTAINER_IMAGE="${REMOTE_IMAGE_TAGGED}"

# Pass data_run_id as parameter to match SAM2 output

# NOTE: workflow names are NOT suffixed (see workflow.py).

# -----------------------------------------------------------------------------
# Video editing instruction(s)
# Choose ONE of the following:
#   - Single instruction: set VIDEO_EDITING_INSTRUCTION and use the singular flag
#   - Multiple instructions: set VIDEO_EDITING_INSTRUCTIONS (newline separated)
# -----------------------------------------------------------------------------

# VIDEO_EDITING_INSTRUCTION="Add a double yellow centerline to the road; realistic worn paint"

# IMPORTANT (edit_bench.py behavior):
# - The string is forwarded to infer/edit_bench.py as --video_editing_instruction.
# - If --llm_model is disabled/none, edit_bench.py uses the instruction text directly
#   for the masked-region caption, taking only the first clause before ';' and
#   trimming to 20 words.
#
# So: put the *core visual change* first, then optional constraints after ';'.
VIDEO_EDITING_INSTRUCTIONS=$'Single solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nDouble solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nSingle solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nDouble solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged\nSingle dashed white intermitted line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'

# -----------------------------------------------------------------------------
# First-frame caption refinement (Qwen critic loop)
# Set these env vars to override defaults:
#   CAPTION_REFINE_ITERS=5         (number of refinement iterations, default 10)
#   CAPTION_REFINE_TEMPERATURE=0.1 (Qwen temperature, default 0.2)
# Example: CAPTION_REFINE_ITERS=3 bash scripts/build_and_run.sh
# -----------------------------------------------------------------------------
CAPTION_REFINE_ITERS="${CAPTION_REFINE_ITERS:-10}"
CAPTION_REFINE_TEMPERATURE="${CAPTION_REFINE_TEMPERATURE:-0.1}"

# -----------------------------------------------------------------------------
# Inpainting strength
# Forwarded to infer/edit_bench.py as --strength (valid range: [0.0, 1.0]).
# Example: VP_STRENGTH=0.85 bash scripts/build_and_run.sh
# -----------------------------------------------------------------------------
VP_STRENGTH="${VP_STRENGTH:-1.0}"


hlx wf run \
  --team-space research \
  --domain prod \
  --execution-name "vp-${X//_/-}-$(date -u +%Y%m%d-%H%M%S)" \
  workflow.videopainter_many_wf \
  --data_run_id "10_150f_caption_fps8" \
  --output_run_id "10_${X}" \
  --data_video_ids "auto" \
  --inpainting_sample_id 0 \
  --model_path "/workspace/VideoPainter/ckpt/CogVideoX-5b-I2V" \
  --inpainting_branch "/workspace/VideoPainter/ckpt/VideoPainter/checkpoints/branch" \
  --img_inpainting_model "/workspace/VideoPainter/ckpt/flux_inp" \
  --output_name_suffix "vp_edit_sample0.mp4" \
  --num_inference_steps 70 \
  --guidance_scale 6.0 \
  --strength "${VP_STRENGTH}" \
  --down_sample_fps 8 \
  --inpainting_frames 49 \
  --video_editing_instructions "${VIDEO_EDITING_INSTRUCTIONS}" \
  --llm_model "${LLM_MODEL_PATH}" \
  --caption_refine_iters "${CAPTION_REFINE_ITERS}" \
  --caption_refine_temperature "${CAPTION_REFINE_TEMPERATURE}" \
  --dilate_size 24 \
  --mask_feather 8 \
  --keep_masked_pixels

echo "VideoPainter report will be uploaded as: gs://.../videopainter/output_vp/10_${X}/10_${X}.txt"

# llm_model options:
# --llm_model "/workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct"  # Local (mounted)
# --llm_model "Qwen/Qwen2.5-VL-7B-Instruct"                             # Hub (requires internet)
# --llm_model "none"