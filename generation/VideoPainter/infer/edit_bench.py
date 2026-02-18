from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
import argparse
from typing import Literal
import importlib
import json
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from diffusers import (
    CogVideoXDPMScheduler,
    CogvideoXBranchModel,
    CogVideoXTransformer3DModel,
    CogVideoXI2VDualInpaintPipeline,
    CogVideoXI2VDualInpaintAnyLPipeline,
    FluxFillPipeline
)
import cv2
try:
    # Qwen2.5-VL uses the newer Transformers naming.
    # Prefer the dedicated Qwen2.5-VL class when available.
    from transformers import Qwen2_5_VLForConditionalGeneration as _QwenForConditionalGeneration, AutoProcessor
except Exception:  # pragma: no cover
    _QwenForConditionalGeneration = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]

# Optional dependency (avoid static import errors in editors/linters).
process_vision_info = None  # type: ignore[assignment]
from diffusers.utils import export_to_video, load_image, load_video
from PIL import Image
from io import BytesIO
import re
import base64
import gc

_qwen_model = None
_qwen_processor = None
_qwen_model_id = None
_qwen_device = None


def _unload_qwen_model() -> None:
    """Best-effort free of Qwen model VRAM/RAM.

    Useful for single-GPU runs where Qwen (VLM) is only needed to produce
    captions/instructions, and the remaining steps need VRAM for FluxFill/Cog.
    """

    global _qwen_model, _qwen_processor, _qwen_model_id, _qwen_device

    if _qwen_model is None and _qwen_processor is None:
        return

    _qwen_model = None
    _qwen_processor = None
    _qwen_model_id = None
    _qwen_device = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


# ---------------------------------------------------------------------------
# Preloaded-models API — load all heavy models once, reuse across videos
# ---------------------------------------------------------------------------

def preload_models(
    *,
    model_path: str,
    inpainting_branch: str | None = None,
    img_inpainting_model: str | None = None,
    img_inpainting_lora_path: str | None = None,
    img_inpainting_lora_scale: float = 1.0,
    llm_model: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    cog_device: str = "cuda",
    flux_device: str = "cuda",
    qwen_device: str | None = None,
    id_adapter_resample_learnable_path: str | None = None,
) -> dict:
    """Pre-load heavy models once and return them in a dict.

    The returned dict can be passed as ``preloaded_models`` to
    :func:`generate_video` so that it skips its own model loading.

    **Single-GPU strategy** (Qwen 7B ~14 GiB + FluxFill ~32 GiB ≈ 46 GiB ≤ 80 GiB):
    Load Qwen + FluxFill together — they coexist on one GPU.  CogVideoX is
    deferred and built lazily in ``generate_video`` *after* Qwen + FluxFill
    are unloaded, because CogVideoX alone needs ~35 GiB.

    **Multi-GPU strategy**: Load all three models onto separate GPUs.

    Returns a dict with keys ``cog_pipe``, ``flux_pipe`` (or None), plus metadata.
    """

    _single_gpu = (not torch.cuda.is_available()) or torch.cuda.device_count() <= 1

    result: dict = {
        "cog_pipe": None,
        "flux_pipe": None,
        "cog_device": cog_device,
        "flux_device": flux_device,
        "qwen_device": qwen_device,
        # Store CogVideoX build params so generate_video can construct it lazily.
        "_cog_deferred": _single_gpu,
        "_cog_build_params": {
            "model_path": model_path,
            "inpainting_branch": inpainting_branch,
            "dtype": dtype,
            "cog_device": cog_device,
            "id_adapter_resample_learnable_path": id_adapter_resample_learnable_path,
        },
    }

    # ---- 1. Qwen VLM (into module-level cache) ----
    # Qwen 7B (~14 GiB bf16) is loaded first.  On single-GPU it coexists with
    # FluxFill (~32 GiB).  On multi-GPU it gets its own device.
    if llm_model and not _llm_disabled(llm_model):
        print(f"[preload] Loading Qwen VLM: {llm_model}")
        _get_qwen_model(llm_model, qwen_device=qwen_device)

    # ---- 2. FluxFill pipeline ----
    if img_inpainting_model:
        print(f"[preload] Loading FluxFill pipeline: {img_inpainting_model}")
        flux_pipe = FluxFillPipeline.from_pretrained(
            img_inpainting_model,
            torch_dtype=torch.bfloat16,
        ).to(flux_device)
        if img_inpainting_lora_path:
            flux_pipe.load_lora_weights(img_inpainting_lora_path)
            flux_pipe.set_adapters(["default"], adapter_weights=[float(img_inpainting_lora_scale)])
        result["flux_pipe"] = flux_pipe

    # ---- 3. CogVideoX pipeline ----
    if _single_gpu:
        # Defer CogVideoX on single-GPU: Qwen + FluxFill already occupy ~46 GiB.
        # CogVideoX will be built in generate_video after Qwen+FluxFill are freed.
        print(f"[preload] Deferring CogVideoX on single-GPU — will build after Qwen+FluxFill phase")
    else:
        print(f"[preload] Loading CogVideoX pipeline: {model_path}")
        result["cog_pipe"] = _build_cog_pipeline(
            model_path=model_path,
            inpainting_branch=inpainting_branch,
            dtype=dtype,
            cog_device=cog_device,
            id_adapter_resample_learnable_path=id_adapter_resample_learnable_path,
        )

    print("[preload] All requested models loaded successfully.")
    return result


def _build_cog_pipeline(
    *,
    model_path: str,
    inpainting_branch: str | None,
    dtype: torch.dtype,
    cog_device: str,
    id_adapter_resample_learnable_path: str | None = None,
):
    """Construct and return a fully-configured CogVideoX pipeline on *cog_device*."""
    branch = None
    transformer = None
    pipe = None

    if inpainting_branch:
        branch = CogvideoXBranchModel.from_pretrained(
            inpainting_branch, torch_dtype=dtype
        ).to(dtype=dtype).to(cog_device)
        if id_adapter_resample_learnable_path is None:
            pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
                model_path,
                branch=branch,
                torch_dtype=dtype,
            )
        else:
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=dtype,
                id_pool_resample_learnable=True,
            ).to(dtype=dtype).to(cog_device)
            pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
                model_path,
                branch=branch,
                transformer=transformer,
                torch_dtype=dtype,
            )
            pipe.load_lora_weights(
                id_adapter_resample_learnable_path,
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name="test_1",
                target_modules=["transformer"],
            )
    else:
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype,
        ).to(dtype=dtype).to(cog_device)
        branch = CogvideoXBranchModel.from_transformer(
            transformer=transformer,
            num_layers=1,
            attention_head_dim=transformer.config.attention_head_dim,
            num_attention_heads=transformer.config.num_attention_heads,
            load_weights_from_transformer=True,
        ).to(dtype=dtype).to(cog_device)
        pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
            model_path,
            branch=branch,
            transformer=transformer,
            torch_dtype=dtype,
        )

    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.branch.requires_grad_(False)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.to(cog_device)
    return pipe


def unload_all_models(preloaded: dict | None = None) -> None:
    """Free all model VRAM/RAM — both preloaded dict and Qwen module cache."""
    if preloaded:
        preloaded["cog_pipe"] = None
        preloaded["flux_pipe"] = None
    _unload_qwen_model()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _dilate_and_feather_mask_images(
    masks: list[Image.Image],
    *,
    dilate_size: int = 0,
    feather_radius: int = 0,
    inpaint_is_white: bool = True,
) -> list[Image.Image]:
    """Expand and optionally soften the inpaint region in mask images.

    Assumes the *inpaint* region is white (255) and keep-region is black (0).
    This matches how `read_video_with_mask` produces `binary_masks` when
    `mask_background=False`.

    Implementation matches the reference behavior:
      - dilation uses a ones-kernel of shape (dilate_size, dilate_size)
      - feathering uses Gaussian blur with kernel size (2*feather_radius + 1)

    Note: when `inpaint_is_white=False` (e.g. mask_background=True), we invert
    the mask before/after processing so dilation still expands the inpaint area.
    """

    dilate_size = int(dilate_size or 0)
    feather_radius = int(feather_radius or 0)
    if dilate_size <= 0 and feather_radius <= 0:
        return masks

    processed: list[Image.Image] = []

    kernel = None
    if dilate_size > 0:
        kernel = np.ones((dilate_size, dilate_size), dtype=np.uint8)

    for mask in masks:
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3:
            # RGB -> single channel; masks are grayscale replicated into 3 channels.
            mask_arr = mask_arr[:, :, 0]

        # Work on a single channel with white = inpaint.
        if not inpaint_is_white:
            mask_arr = 255 - mask_arr

        if kernel is not None:
            mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)

        if feather_radius > 0:
            k = 2 * feather_radius + 1
            mask_arr = cv2.GaussianBlur(mask_arr, (k, k), sigmaX=0)

        if not inpaint_is_white:
            mask_arr = 255 - mask_arr

        processed.append(Image.fromarray(mask_arr.astype(np.uint8)).convert("RGB"))
    return processed


def _llm_disabled(llm_model: str | None) -> bool:
    if llm_model is None:
        return True
    if not isinstance(llm_model, str):
        return True
    return llm_model.strip().lower() in {"", "none", "off", "disable", "disabled", "false", "0", "no"}


def _limit_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words).strip()
    return " ".join(words[:max_words]).strip()


def _primary_instruction_clause(instruction: str) -> str:
    """Extract the main edit clause from a possibly long instruction.

    When running without an LLM, we use the instruction text directly as the
    masked-region inpainting caption. Users often append long constraints like
    "keep everything else unchanged"; those should not dominate the caption.
    """
    if not instruction:
        return ""

    # Prefer the first clause before semicolons, which commonly separates the
    # requested edit from constraints.
    parts = [p.strip() for p in instruction.split(";")]
    for part in parts:
        if part:
            return part
    return instruction.strip()


def _encode_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _thumbnail_copy(img: Image.Image, *, max_side: int = 768) -> Image.Image:
    if img is None:
        return img
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    m = max(w, h)
    if m <= max_side:
        return img
    scale = float(max_side) / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), resample=Image.BICUBIC)


def _mask_overlay_image(base: Image.Image, mask_rgb: Image.Image, *, color=(255, 0, 0), alpha: float = 0.45) -> Image.Image:
    """Overlay the inpaint region (white pixels in mask_rgb) onto base."""
    base = base.convert("RGB")
    mask = mask_rgb.convert("RGB")
    b = np.array(base).astype(np.float32)
    m = np.array(mask)
    # Use channel 0 as mask intensity; white=255 indicates inpaint region.
    region = (m[:, :, 0] > 127)
    if not np.any(region):
        return base
    overlay = b.copy()
    overlay[region] = (1.0 - alpha) * overlay[region] + alpha * np.array(color, dtype=np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def _make_qwen_eval_panel(*, original: Image.Image, inpainted: Image.Image, flux_mask: Image.Image) -> bytes:
    """Create a side-by-side panel (orig+mask overlay | mask | inpainted) for VLM evaluation."""
    orig_small = _thumbnail_copy(original, max_side=640)
    inp_small = _thumbnail_copy(inpainted, max_side=640)
    mask_small = _thumbnail_copy(flux_mask, max_side=640)
    # Resize to common height
    h = min(orig_small.size[1], inp_small.size[1], mask_small.size[1])
    def fit_h(img: Image.Image) -> Image.Image:
        w, ih = img.size
        if ih == h:
            return img
        nw = max(1, int(round(w * (float(h) / float(ih)))))
        return img.resize((nw, h), resample=Image.BICUBIC)

    orig_small = fit_h(orig_small)
    inp_small = fit_h(inp_small)
    mask_small = fit_h(mask_small)

    orig_overlay = _mask_overlay_image(orig_small, mask_small, color=(255, 0, 0), alpha=0.45)
    total_w = orig_overlay.size[0] + mask_small.size[0] + inp_small.size[0]
    panel = Image.new("RGB", (total_w, h))
    x = 0
    panel.paste(orig_overlay, (x, 0))
    x += orig_overlay.size[0]
    panel.paste(mask_small.convert("RGB"), (x, 0))
    x += mask_small.size[0]
    panel.paste(inp_small.convert("RGB"), (x, 0))
    return _encode_png_bytes(panel)


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from model text."""
    if not text:
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Try to find the first {...} block.
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _qwen_refine_first_frame_caption(
    *,
    llm_model: str,
    qwen_device: str | None,
    instruction: str,
    current_caption: str,
    original_frame: Image.Image,
    inpainted_frame: Image.Image,
    flux_mask: Image.Image,
    temperature: float = 0.2,
) -> tuple[bool, str, str]:
    """Return (pass_ok, revised_caption, notes)."""
    panel_bytes = _make_qwen_eval_panel(original=original_frame, inpainted=inpainted_frame, flux_mask=flux_mask)
    system_prompt = (
        "You are a strict visual quality evaluator for image inpainting edits in driving scenes. "
        "Judge whether the edited image satisfies the instruction inside the masked region, "
        "and whether it remains visually consistent with the unmasked context. "
        "IMPORTANT: the input videos may have no centerline, a single centerline, or a double centerline. "
        "Evaluate based on the *desired post-edit result* implied by the instruction and the visible scene geometry.\n"
        "CRITICAL VALIDATION:\n"
        "- If instruction says SINGLE, REJECT captions describing DOUBLE (two parallel) lines\n"
        "- If instruction says DOUBLE, REJECT captions describing SINGLE line\n"
        "- If instruction says SOLID/CONTINUOUS, REJECT captions describing DASHED/INTERMITTENT\n"
        "- If instruction says DASHED/INTERMITTENT, REJECT captions describing SOLID/CONTINUOUS\n"
        "- If instruction specifies color (white/yellow), caption color must match exactly\n"
        "- Reject captions that imply existing markings that are not present in the original panel\n"
        "- Reject captions that are too vague (e.g., 'lane line')\n"
        "- Captions must include visual details (paint wear, texture, lighting, perspective alignment, placement)\n"
        "When revising captions, preserve ALL key descriptors from the instruction (single/double, solid/dashed, color, etc.)"
    )
    user_prompt = f"""You will be shown a 3-panel image: (1) original with mask overlay (red indicates inpaint region), (2) the binary mask (white=inpaint), (3) the inpainted result.

Editing instruction:
{instruction}

Current inpainting caption (used for the masked region):
{current_caption}

Return ONLY a JSON object with these keys:
- verdict: "PASS" or "FAIL"
- revised_caption: a better caption to try next if FAIL (<= 20 words). If PASS, you can repeat the current caption.
- notes: short reason, mention any artifacts or mismatches.

Rules for revised_caption (IMPORTANT):
1. ALWAYS include ALL key descriptors from the instruction (single/double, solid/dashed, color, etc.)
2. Add VISUAL DETAILS: paint wear, crispness, asphalt texture blending, lighting/shadow consistency
3. Placement rule: align with the road center and perspective (follow curvature / vanishing point). If there was no centerline originally, describe how the new centerline is placed plausibly at the center of the lane.
4. If there was an existing single/double centerline and the instruction changes it, describe the *final* marking (e.g., replace single->double or double->single) without saying "replace".
5. Describe ONLY the target edited region (what it should look like after the edit)
6. Do NOT mention the mask, inpainting, or editing process
7. Keep perspective/geometry consistent with the original

Good captions have:
- Key descriptors from instruction ✓
- Visual texture details (smooth, worn, painted, etc.) ✓
- Positioning/alignment info ✓
- Lighting/shadow consistency ✓
- Specific visual characteristics ✓

Bad captions are generic:
- "Lane line" (missing: single/double, color, texture)
- "White line" (missing: solid/dashed, perspective, alignment)
- "Replacing markings" (missing: what it should LOOK like visually)
"""

    out = _qwen_generate_text(
        model=llm_model,
        qwen_device=qwen_device,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_bytes=panel_bytes,
        temperature=temperature,
        max_output_tokens=200,
    )
    obj = _extract_json_object(out)
    if not obj:
        return (False, current_caption, (out or "").strip())
    verdict = str(obj.get("verdict", "")).strip().upper()
    revised = str(obj.get("revised_caption", "") or "").strip()
    notes = str(obj.get("notes", "") or "").strip()
    pass_ok = verdict == "PASS"
    if revised:
        revised = _limit_words(revised, 20)
    else:
        revised = current_caption
    return (pass_ok, revised, notes)


def _get_qwen_model(model_id: str, *, qwen_device: str | None = None):
    """Load and cache a Qwen2.5-VL model + processor.

    `model_id` can be a HuggingFace model id (e.g. "Qwen/Qwen2.5-VL-7B-Instruct")
    or a local path to a downloaded snapshot folder.

    Note: Qwen2.5-VL may require a newer Transformers build than some pinned
    releases. If you hit KeyError like 'qwen2_5_vl', install from source:
      pip install git+https://github.com/huggingface/transformers accelerate
    """

    global _qwen_model, _qwen_processor, _qwen_model_id, _qwen_device

    if _QwenForConditionalGeneration is None or AutoProcessor is None:
        raise ImportError(
            "Qwen2.5-VL requires `transformers`, `accelerate`, and `qwen-vl-utils`. "
            "If using Qwen2.5-VL, you may need a newer Transformers build: "
            "`pip install git+https://github.com/huggingface/transformers accelerate` "
            "and `pip install qwen-vl-utils`."
        )

    qwen_device_key = (qwen_device or "auto").strip()

    # Reuse cached model if it's already loaded for the same id/path + device.
    if (
        _qwen_model is not None
        and _qwen_processor is not None
        and _qwen_model_id == model_id
        and (_qwen_device or "auto").strip() == qwen_device_key
    ):
        return _qwen_model, _qwen_processor

    if qwen_device_key.lower() == "auto":
        print(f"Loading Qwen model '{model_id}' with device_map=auto... This may take a few minutes.")
    else:
        print(f"Loading Qwen model '{model_id}' with requested device '{qwen_device_key}'... This may take a few minutes.")
    
    # Determine if model_id is a local path or a HuggingFace Hub ID
    import os
    is_local_path = os.path.isdir(model_id)
    
    import tempfile

    # IMPORTANT:
    # - For large checkpoints (e.g., Qwen2.5-VL-72B), forcing all weights onto a
    #   single CUDA device via device_map={"": "cuda:X"} can easily OOM.
    # - We default to Accelerate sharded loading (device_map="auto") with CPU
    #   offload. When a specific CUDA ordinal is requested, we guide placement
    #   using max_memory so Qwen stays on that GPU (and avoids the Cog/Flux GPU).

    qwen_torch_dtype: torch.dtype = torch.bfloat16
    model_load_kwargs: dict = {
        "local_files_only": is_local_path,
        "low_cpu_mem_usage": True,
    }

    if qwen_device_key.lower() == "cpu":
        qwen_torch_dtype = torch.float32
        model_load_kwargs["device_map"] = {"": "cpu"}
    else:
        model_load_kwargs["device_map"] = "auto"

        # Constrain placement so we don't spill onto other GPUs.
        # - If multiple GPUs and user requested cuda:N, give cuda:N most budget.
        # - If single GPU, keep Qwen mostly offloaded to CPU so Flux/Cog can fit.
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()

            target_ordinal = None
            if qwen_device_key.startswith("cuda:"):
                try:
                    target_ordinal = int(qwen_device_key.split(":", 1)[1])
                except Exception:
                    target_ordinal = None
            elif qwen_device_key == "cuda":
                target_ordinal = 0

            if target_ordinal is not None and 0 <= target_ordinal < gpu_count:
                total_bytes = int(torch.cuda.get_device_properties(target_ordinal).total_memory)
                total_gib = max(1, int(total_bytes / (1024 ** 3)))

                if gpu_count <= 1:
                    # Single-GPU fallback: keep Qwen's GPU footprint small.
                    budget_gib = max(4, min(16, total_gib - 8))
                else:
                    # Multi-GPU: allow Qwen to use most of its dedicated GPU.
                    budget_gib = max(8, min(total_gib - 4, int(total_gib * 0.9)))

                max_memory: dict = {target_ordinal: f"{budget_gib}GiB", "cpu": "192GiB"}
                for i in range(gpu_count):
                    if i != target_ordinal:
                        max_memory[i] = "1GiB"
                model_load_kwargs["max_memory"] = max_memory

        model_load_kwargs["offload_folder"] = os.path.join(tempfile.gettempdir(), "qwen_offload")
        model_load_kwargs["offload_state_dict"] = True

    _qwen_model = _QwenForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=qwen_torch_dtype,
        **model_load_kwargs,
    )
    _qwen_processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=is_local_path,
    )
    _qwen_model_id = model_id
    _qwen_device = qwen_device_key
    print(f"Qwen model '{model_id}' loaded successfully.")
    return _qwen_model, _qwen_processor


def _qwen_generate_text(
    *,
    model: str,
    qwen_device: str | None = None,
    system_prompt: str,
    user_prompt: str,
    image_bytes: bytes | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    if model is None:
        raise ValueError("llm_model is None")

    # `model` is the HuggingFace model id or local path.
    qwen_model, qwen_processor = _get_qwen_model(model, qwen_device=qwen_device)

    global process_vision_info
    if process_vision_info is None:
        try:
            process_vision_info = importlib.import_module("qwen_vl_utils").process_vision_info
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "qwen-vl-utils is required for vision inputs with Qwen2-VL. "
                "Install it with: `pip install qwen-vl-utils`"
            ) from e

    # Build messages in Qwen's chat format
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    
    # Add user message with optional image
    user_content = []
    if image_bytes is not None:
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes))
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": user_prompt})
    
    messages.append({
        "role": "user",
        "content": user_content
    })

    # Prepare inputs
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(qwen_model.device)

    # Generate
    generation_kwargs = {}
    if temperature is not None:
        generation_kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        generation_kwargs["max_new_tokens"] = max_output_tokens
    else:
        generation_kwargs["max_new_tokens"] = 512

    with torch.inference_mode():
        generated_ids = qwen_model.generate(
            **inputs,
            **generation_kwargs
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0].strip()

def _visualize_video(pipe, mask_background, original_video, video, masks):
    
    original_video = pipe.video_processor.preprocess_video(original_video, height=video.shape[1], width=video.shape[2])
    masks = pipe.masked_video_processor.preprocess_video(masks, height=video.shape[1], width=video.shape[2])
    
    if mask_background:
        masked_video = original_video * (masks >= 0.5)
    else:
        masked_video = original_video * (masks < 0.5)
    
    original_video = pipe.video_processor.postprocess_video(video=original_video, output_type="np")[0]
    masked_video = pipe.video_processor.postprocess_video(video=masked_video, output_type="np")[0]
    
    masks = masks.squeeze(0).squeeze(0).numpy()
    masks = masks[..., np.newaxis].repeat(3, axis=-1)

    video_ = concatenate_images_horizontally(
        [original_video, masked_video, masks, video],
    )
    return video_

def concatenate_images_horizontally(images_list, output_type="np"):
    '''
    Concatenate three lists of images horizontally.
    Args:
        images_list: List[List[Image.Image]] or List[List[np.ndarray]]
    Returns:
        List[Image.Image] or List[np.ndarray]
    '''
    concatenated_images = []

    length = len(images_list[0])
    for i in range(length):
        tmp_tuple = ()
        for item in images_list:
            tmp_tuple += (np.array(item[i]), )

        # Concatenate arrays horizontally
        concatenated_img = np.concatenate(tmp_tuple, axis=1)

        # Convert back to PIL Image
        if output_type == "pil":
            concatenated_img = Image.fromarray(concatenated_img)
        elif output_type == "np":
            pass
        else:
            raise NotImplementedError
        concatenated_images.append(concatenated_img)
    return concatenated_images

def read_video_with_mask(
    video_path,
    masks,
    mask_id,
    skip_frames_start=0,
    skip_frames_end=-1,
    mask_background=False,
    fps=0,
):
    '''
    read the video and masks, and return the video, masked video and binary masks
    Args:
        video_path: str, the path of the video
        masks: np.ndarray, the masks of the video
        mask_id: int, the id of the mask
        skip_frames_start: int, the number of frames to skip at the beginning
        skip_frames_end: int, the number of frames to skip at the end
    Returns:
        video: List[Image.Image], the video (RGB)
        masked_video: List[Image.Image], the masked video (RGB)
        binary_masks: List[Image.Image], the binary masks (RGB)
    '''

    video = load_video(video_path)[skip_frames_start:skip_frames_end]
    mask = masks[skip_frames_start:skip_frames_end]
    # read fps
    if fps == 0:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
    masked_video = []
    binary_masks = []
    for frame, frame_mask in zip(video, mask):
        frame_array = np.array(frame)
        
        black_frame = np.zeros_like(frame_array)
        
        binary_mask = (frame_mask == mask_id)
        
        binary_mask_expanded = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
        
        masked_frame = np.where(binary_mask_expanded, black_frame, frame_array)
        
        masked_video.append(Image.fromarray(masked_frame.astype(np.uint8)).convert("RGB"))
        
        if mask_background:
            binary_mask_image = np.where(binary_mask, 0, 255).astype(np.uint8)
        else:
            binary_mask_image = np.where(binary_mask, 255, 0).astype(np.uint8)
        binary_masks.append(Image.fromarray(binary_mask_image).convert("RGB"))
    video = [item.convert("RGB") for item in video]
    return video, masked_video, binary_masks, fps

def video_editing_prompt(prompt, instruction, llm_model, masked_image=None, target_img_caption=True, *, qwen_device: str | None = None):
    if prompt is None or instruction is None:
        raise ValueError("prompt is None or instruction is None")

    # Allow running without an LLM/Gemini.
    if _llm_disabled(llm_model):
        edited_prompt = f"{prompt}. {instruction}".strip()
        # Simple fallback: use the primary edit clause as the inpainting caption.
        masked_image_caption = _limit_words(_primary_instruction_clause(instruction), 20)
        return edited_prompt, masked_image_caption
    # Use LLM to edit the video description
    system_prompt = """You are a video description editing expert for driving scenes. You will edit the original video description based on the user's instruction.

Requirements:
1. Keep the description coherent and natural.
2. Apply ONLY what the instruction requests (do not invent extra objects).
3. Retain important details that are not affected by the instruction.
4. Lane-marking instructions may apply to videos with: no centerline, a single centerline, or a double centerline. Do not assume which one exists; describe the *result after the edit*.
"""
    
    user_prompt = f"""Original video description: {prompt}
    Editing instruction: {instruction}
    Please edit the video description based on the editing instruction. Only return the edited description, no other words."""

    prompt = _qwen_generate_text(
        model=llm_model,
        qwen_device=qwen_device,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


    if target_img_caption:
        if masked_image is None:
            raise ValueError("masked_image is None when target_img_caption=True")
            
        # Convert PIL image to base64
        buffered = BytesIO()
        masked_image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        
        system_prompt ="""You are an expert in precise visual description for image inpainting. Your captions directly control what is generated.

CRITICAL: Your description quality directly impacts output quality. Generate DETAILED, SPECIFIC descriptions that:
1. Preserve ALL key words from the editing instruction (single/double, solid/dashed, white/yellow, etc.)
2. Include VISUAL TEXTURE details (smooth, worn, faded, crisp, weathered paint, etc.)
3. Specify POSITIONING and PERSPECTIVE (aligned with original, centered, offset, perspective-matched, etc.)
4. Include LIGHTING consistency (shadows, reflections, daylight/ambient, matching surroundings, etc.)
5. Describe ONLY the edited result (what it SHOULD look like after the edit), NOT the process

This is NOT a generic labeling task. The more specific and visually detailed, the better the generation.
Analyze the image context and produce descriptions rich enough to guide a diffusion model.

Centerline handling (CRITICAL for road videos):
- The original video may contain no centerline, a single centerline, or a double centerline.
- Infer what is currently present from the image.
- Describe ONLY the desired final centerline marking inside the masked region, following the road geometry and perspective.
- If there is no centerline and the instruction requests one/two: place it at the road center, aligned to the lane direction/vanishing point.
- If there is a single and instruction requests double (or vice versa): describe the final marking as one or two parallel lines with realistic spacing.
"""
        
        user_prompt = f"""IMAGE CONTEXT:
    You are analyzing a road/driving scene with a masked editing region (the mask indicates the region that will be regenerated).

Editing instruction: {instruction}

TASK - Generate a detailed description for inpainting the masked region:

Requirements:
1. MANDATORY ELEMENTS (must include all relevant ones):
   - Count: single, double, triple?
   - Style: solid, dashed, dotted, intermittent?
   - Color: white, yellow, other? (explicit colors)
   - Texture: smooth/worn/faded/crisp/weathered paint?
   - Perspective: centered, offset, perspective-aligned to original markings?
   - Lighting: match surrounding road lighting/shadows?

2. SPECIFIC DETAILS:
   - Describe line width, spacing (if multiple)
   - Road surface integration (asphalt texture match)
   - Any wear patterns or degradation
   - Alignment with original lane geometry

3. CONSTRAINT:
   - Describe ONLY what the edited region should look like
   - Do NOT mention mask, editing, or transformation process
    - Maximum 25 words (prioritize key visual details)
    - Do NOT claim existing markings unless they are visible; focus on the post-edit target result

Example (good):
Instruction: "Add solid yellow double line"
Caption: "Two solid yellow continuous lines, worn paint texture, centered lane position, perspective-aligned"

Example (bad):
Instruction: "Add solid yellow double line"
Caption: "Yellow double line" (missing: solid vs dashed, texture, perspective, alignment)

Generate only the description, nothing else:"""

        masked_image_caption = _qwen_generate_text(
            model=llm_model,
            qwen_device=qwen_device,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_bytes=image_bytes,
            temperature=0.6,  # Slightly lower for consistency
            max_output_tokens=60,  # Slightly higher to allow detail
        )
    else:
        # image inpainting prompt from the new video caption
        system_prompt = "You are an expert in image description. Based on the given video description and editing instruction, please generate a concise description for the first static frame, focusing on the most important visual elements."
        
        user_prompt = f"""Video description (after editing): {prompt}, editing instruction: {instruction}
        Please generate a static description for the first frame. Requirements:
        1. Keep the description concise and precise
        2. Only describe edited visual elements
        3. Avoid using any dynamic or temporal-related words
        4. Must explicitly reflect the visual changes requested in the editing instruction
        5. Emphasize and highlight the edited content as the focus of the description

        Example:
        Editing instruction: "Change the sea water to green"
        Good description: "Green seawater washing against the shore"

        Only return the description, no other words."""

        masked_image_caption = _qwen_generate_text(
            model=llm_model,
            qwen_device=qwen_device,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    return prompt, masked_image_caption

def generate_video_editing_instruction(masked_image, llm_model, *, qwen_device: str | None = None):    
    buffered = BytesIO()
    masked_image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()
    
    '''
       - Add objects (e.g., "Add a flock of birds in the sky", "Add a necklace to the woman's neck")
       - Delete objects (e.g., "Remove the car from the street", "Remove the hat from the man's head")
       - Swap objects (e.g., "Replace the dog with a cat", "Replace the woman's dress with a red dress")
       - Change object attributes (e.g., "Change the color of the house to blue", "Change the material of the wall to marble", "Change the environment to a alien planet")
    '''
    system_prompt = """You are an expert in visual scene understanding and creative editing. Your task is to:
    1. Analyze the visible objects/region in the provided masked image
    2. Generate a creative editing instruction from one of these categories:
       - Delete objects (e.g., "Remove the car from the street", "Remove the hat from the man's head")
    
    Requirements:
    1. The instruction must be relevant to the visible content in the masked region
    2. The instruction should be specific and clear
    3. The instruction should be creative but reasonable
    4. Return only the instruction, no explanations"""

    '''
    - Add objects
    - Delete objects
    - Swap objects
    - Change object attributes (color, material, gender, species, location, etc.)
    '''
    
    user_prompt = """Based on the visible content in this masked image, generate a creative editing instruction.
    The instruction should be randomly selected from these types:
    - Delete objects
    
    Only return the instruction, no other text."""

    instruction = _qwen_generate_text(
        model=llm_model,
        qwen_device=qwen_device,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_bytes=image_bytes,
        temperature=0.8,
        max_output_tokens=30,
    )
    return instruction

def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],
    seed: int = 42,
    # inpainting
    inpainting_mask_meta: str = None,
    inpainting_sample_id: int = None,
    inpainting_branch: str = None,
    inpainting_frames: int = None,
    mask_background: bool = False,
    keep_masked_pixels: bool = False,
    add_first: bool = False,
    first_frame_gt: bool = False,
    replace_gt: bool = False,
    mask_add: bool = False,
    down_sample_fps: int = 8,
    overlap_frames: int = 0,
    prev_clip_weight: float = 0.0,
    img_inpainting_model: str = None,
    img_inpainting_lora_path: str = None,
    img_inpainting_lora_scale: float = 1.0,
    video_editing_instruction: str = None,
    llm_model: str = None,
    qwen_device: str | None = None,
    unload_qwen_after_caption: bool = True,
    cog_device: str = "cuda",
    flux_device: str = "cuda",
    dilate_size: int = -1,
    mask_feather: int = 0,
    caption_refine_iters: int = 0,
    caption_refine_temperature: float = 0.2,
    long_video: bool = False,
    id_adapter_resample_learnable_path: str = None,
    strength: float = 1.0,
    # Pre-loaded models (from preload_models()) — skip loading when provided.
    preloaded_models: dict | None = None,
    # Two-phase execution for single-GPU multi-video:
    #   "first_frame" — Qwen + FluxFill only, save results, return early
    #   "video"       — CogVideoX only, load first frame + prompt from disk
    run_phase: str = "first_frame",
):

    def _normalize_device(dev: str, *, fallback_cuda: str = "cuda:0") -> str:
        dev = (dev or "").strip()
        if not dev:
            dev = "cuda"

        # CPU fallback if CUDA isn't available.
        if not torch.cuda.is_available():
            return "cpu"

        # Accept plain "cuda".
        if dev == "cuda":
            return dev

        # Validate explicit ordinals like "cuda:1".
        if dev.startswith("cuda:"):
            try:
                ordinal = int(dev.split(":", 1)[1])
            except Exception:
                return fallback_cuda

            count = torch.cuda.device_count()
            if count <= 0:
                return "cpu"
            if 0 <= ordinal < count:
                return dev
            # If only one GPU is visible, cuda:1 is invalid -> fall back to cuda:0.
            return fallback_cuda

        # Unknown device string; keep it (torch will raise a clearer error).
        return dev

    def _normalize_device_or_auto(dev: str | None) -> str | None:
        if dev is None:
            return None
        dev = (dev or "").strip()
        if not dev:
            return None
        if dev.lower() in {"auto"}:
            return "auto"
        if dev.lower() in {"cpu"}:
            return "cpu"
        # Respect explicit cuda device ids, but fall back to cpu if cuda isn't present.
        if dev.startswith("cuda") and (not torch.cuda.is_available()):
            return "cpu"
        # Validate ordinals like cuda:1 similarly to the main normalize.
        if dev.startswith("cuda:"):
            try:
                ordinal = int(dev.split(":", 1)[1])
            except Exception:
                return "auto"
            count = torch.cuda.device_count()
            if count <= 0:
                return "cpu"
            if 0 <= ordinal < count:
                return dev
            return "cuda:0" if count > 0 else "cpu"
        return dev

    def _device_ordinal(dev: str) -> int | None:
        d = (dev or "").strip().lower()
        if d == "cuda":
            return 0
        if d.startswith("cuda:"):
            try:
                return int(d.split(":", 1)[1])
            except Exception:
                return None
        return None

    def _auto_pick_qwen_device(
        *,
        requested: str | None,
        cog_dev: str,
        flux_dev: str,
    ) -> str | None:
        # If user explicitly requested a device, honor it.
        if requested is not None and str(requested).strip() and str(requested).strip().lower() != "auto":
            return requested

        if not torch.cuda.is_available():
            return "cpu"
        gpu_count = torch.cuda.device_count()

        # If we have >1 GPU, place Qwen on a different GPU than Cog/Flux.
        if gpu_count > 1:
            used = {
                _device_ordinal(cog_dev) if _device_ordinal(cog_dev) is not None else 0,
                _device_ordinal(flux_dev) if _device_ordinal(flux_dev) is not None else 0,
            }
            for i in range(gpu_count - 1, -1, -1):
                if i not in used:
                    return f"cuda:{i}"

        # Single-GPU fallback: keep Qwen on the same GPU.
        return "cuda:0"

    image = None
    video = None

    strength = float(strength)
    if strength < 0.0 or strength > 1.0:
        raise ValueError(f"strength must be within [0.0, 1.0], got {strength}")

    cog_device = _normalize_device(cog_device, fallback_cuda="cuda:0")
    flux_device = _normalize_device(flux_device, fallback_cuda=cog_device if cog_device.startswith("cuda") else "cuda:0")
    qwen_device = _normalize_device_or_auto(qwen_device)
    qwen_device = _auto_pick_qwen_device(requested=qwen_device, cog_dev=cog_device, flux_dev=flux_device)

    if generate_type == "i2v_inpainting":
        meta_data = pd.read_csv(inpainting_mask_meta).iloc[inpainting_sample_id, :]
        video_base_name = meta_data['path'].split(".")[0]
        if ".0.mp4" in meta_data['path']:
            video_path = os.path.join(image_or_video_path, video_base_name[:-3], f'{video_base_name}.0.mp4')

            # Prefer masks produced by our preprocessing script:
            #   <data_dir>/masks/<video_id>/all_masks.npz
            data_dir = os.path.dirname(inpainting_mask_meta) if inpainting_mask_meta else ""
            preferred_masks = os.path.join(data_dir, "masks", video_base_name, "all_masks.npz")

            # Backward compatible path used by the original repo layout.
            legacy_masks = os.path.join("../data/video_inpainting/videovo", video_base_name, "all_masks.npz")

            mask_frames_path = preferred_masks if os.path.exists(preferred_masks) else legacy_masks
        elif ".mp4" in meta_data['path']:
            video_path = os.path.join(image_or_video_path.replace("videovo", "pexels/pexels"), video_base_name[:9], f'{video_base_name}.mp4')
            mask_frames_path = os.path.join("../data/video_inpainting/pexels", video_base_name, "all_masks.npz")
        else:
            raise NotImplementedError
        fps = meta_data['fps']
        mask_id = meta_data['mask_id']
        start_frame = meta_data['start_frame']
        end_frame = meta_data['end_frame']
        all_masks = np.load(mask_frames_path)["arr_0"]
        prompt = meta_data['caption']

       
        print("-"*100)
        print(f"video_path: {video_path}; mask_id: {mask_id}; start_frame: {start_frame}; end_frame: {end_frame}; mask_shape: {all_masks.shape}")
        print("-"*100)

        video, masked_video_black, binary_masks, fps = read_video_with_mask(
            video_path,
            skip_frames_start=start_frame,
            skip_frames_end=end_frame,
            masks=all_masks,
            mask_id=mask_id,
            mask_background=mask_background,
            fps=fps,
        )

        # Most inpainting pipelines accept (original video, mask) without needing the
        # masked region to be blacked out. When the mask covers the whole road, using
        # the blacked-out input removes the lane geometry, making it hard to place
        # new lanes exactly where the originals were.
        masked_video_for_pipe = video if keep_masked_pixels else masked_video_black

        pipe = preloaded_models["cog_pipe"] if preloaded_models else None

        if img_inpainting_model:
            print(f"Using the provided image inpainting model: {img_inpainting_model}")

            image = video[0]
            mask = binary_masks[0]

            # -------------------------------------------------------
            # Phase "video": skip Qwen+FluxFill, load results from disk.
            # -------------------------------------------------------
            _skip_qwen_flux = (run_phase == "video")
            if _skip_qwen_flux:
                prompt_json_path = output_path.replace(".mp4", ".json")
                flux_img_path = os.path.join(
                    os.path.dirname(output_path),
                    f"{os.path.basename(output_path)}_flux_o_img.png",
                )
                if not os.path.exists(prompt_json_path) or not os.path.exists(flux_img_path):
                    raise FileNotFoundError(
                        f"Phase 'video' requires first_frame outputs. "
                        f"Missing: {prompt_json_path if not os.path.exists(prompt_json_path) else flux_img_path}"
                    )
                with open(prompt_json_path, "r") as f:
                    pj = json.load(f)
                prompt = pj.get("Edited_video_caption", prompt)
                image_inpainting = Image.open(flux_img_path).convert("RGB")
                print(f"[phase=video] Loaded first frame from {flux_img_path}")
                print(f"[phase=video] Loaded edited prompt: {prompt[:120]}...")

                gt_video_first_frame = video[0]
                video[0] = image_inpainting
                masked_video_for_pipe[0] = image_inpainting

            if not _skip_qwen_flux:

                # FluxFill expects white = inpaint region. When mask_background=True, our
                # binary mask is inverted (black=inpaint), so invert for FluxFill.
                flux_mask = mask
                if mask_background:
                    m = np.array(mask)
                    if m.ndim == 3:
                        m = m[:, :, 0]
                    flux_mask = Image.fromarray((255 - m).astype(np.uint8)).convert("RGB")

                # Optional preprocessing on the FluxFill mask as well.
                if (dilate_size and dilate_size > 0) or (mask_feather and mask_feather > 0):
                    flux_mask = _dilate_and_feather_mask_images(
                        [flux_mask],
                        dilate_size=dilate_size if dilate_size and dilate_size > 0 else 0,
                        feather_radius=mask_feather if mask_feather and mask_feather > 0 else 0,
                        inpaint_is_white=True,
                    )[0]

                image_array = np.array(image)
                mask_array = np.array(mask)
                foreground_mask = (mask_array == 255)
                masked_image = np.where(foreground_mask, image_array, 0)
                masked_image = Image.fromarray(masked_image.astype(np.uint8))

                # For the VLM prompt, provide *surrounding context* (unmasked area)
                # by showing the original frame with the inpaint region removed.
                # This matches the instructions in `video_editing_prompt` and helps
                # Qwen generate a caption consistent with scene lighting/perspective.
                # For lane-placement tasks, showing the original frame helps the VLM
                # preserve perspective and align edits with existing lane markings.
                vlm_context_image = video[0] if keep_masked_pixels else masked_video_for_pipe[0]

                if llm_model is None:
                    llm_model = "none"

                if video_editing_instruction == 'auto':
                    if _llm_disabled(llm_model):
                        raise ValueError(
                            "video_editing_instruction='auto' requires an LLM. "
                            "Provide --video_editing_instruction explicitly, or set --llm_model (e.g., 'Qwen/Qwen2.5-VL-72B-Instruct')."
                        )
                    video_editing_instruction = generate_video_editing_instruction(masked_image, llm_model, qwen_device=qwen_device)
                    output_path = output_path.replace("auto", video_editing_instruction).replace("..", ".")

                original_prompt = prompt

                prompt, masked_image_caption = video_editing_prompt(
                    prompt, 
                    video_editing_instruction, 
                    llm_model, 
                    masked_image=vlm_context_image, 
                    target_img_caption=True,
                    qwen_device=qwen_device,
                )

                print("-"*100)
                print(f"Original video caption: {original_prompt}")
                print("-"*100)
                print(f"Edited video caption: {prompt}")
                print("-"*100)
                print(f"Masked image caption: {masked_image_caption}")
                print("-"*100)
                print(f"Video editing instruction: {video_editing_instruction}")
                print("-"*100)

                prompt_json_path = output_path.replace(".mp4", f".json")
                with open(prompt_json_path, "w") as f:
                    prompt_dict = {
                        "path": meta_data['path'],
                        "mask_id": int(mask_id),
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "fps": int(meta_data['fps']),
                        "Original_video_caption": original_prompt,
                        "Edited_video_caption": prompt,
                        "Edited_image_caption": masked_image_caption,
                        "Edited_image_caption_initial": masked_image_caption,
                        "Edited_image_caption_final": None,
                        "Editing_instruction": video_editing_instruction,
                        "Caption_refine_iters_requested": int(caption_refine_iters or 0),
                        "Caption_refine_temperature": float(caption_refine_temperature or 0.2),
                    }
                    json.dump(prompt_dict, f, indent=4, ensure_ascii=False)
            

                pipe_img_inpainting = preloaded_models["flux_pipe"]

                masked_image_caption_initial = masked_image_caption

                def _run_flux(prompt_text: str) -> Image.Image:
                    return pipe_img_inpainting(
                        prompt=prompt_text,
                        image=image,
                        mask_image=flux_mask,
                        height=image.size[1],
                        width=image.size[0],
                        guidance_scale=30,
                        num_inference_steps=50,
                        max_sequence_length=512,
                        generator=torch.Generator("cpu").manual_seed(0),
                    ).images[0]

                image.save(os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}_flux_i_img.png"))
                flux_mask.save(os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}_flux_i_mask.png"))

                iters_used = 0
                last_notes = ""
                image_inpainting = None
                refine_enabled = (int(caption_refine_iters or 0) > 0) and (not _llm_disabled(llm_model))
                if refine_enabled:
                    # Put iterative artifacts under a dedicated folder per video.
                    # Folder name is stable and derived from the output basename.
                    base_no_ext = os.path.splitext(os.path.basename(output_path))[0]
                    refine_dir = os.path.join(os.path.dirname(output_path), f"{base_no_ext}_caption_refine")
                    os.makedirs(refine_dir, exist_ok=True)
                    max_iters = int(caption_refine_iters or 0)
                    for i in range(1, max_iters + 1):
                        iters_used = i
                        image_inpainting = _run_flux(masked_image_caption)
                        image_inpainting.save(
                            os.path.join(refine_dir, f"iter{i:03d}_flux_o_img.png")
                        )
                        passed, revised_caption, notes = _qwen_refine_first_frame_caption(
                            llm_model=llm_model,
                            qwen_device=qwen_device,
                            instruction=video_editing_instruction,
                            current_caption=masked_image_caption,
                            original_frame=image,
                            inpainted_frame=image_inpainting,
                            flux_mask=flux_mask,
                            temperature=float(caption_refine_temperature or 0.2),
                        )
                        last_notes = notes
                        eval_payload = {
                            "iter": i,
                            "verdict": "PASS" if passed else "FAIL",
                            "caption_in": masked_image_caption,
                            "caption_out": revised_caption,
                            "notes": notes,
                        }
                        with open(
                            os.path.join(refine_dir, f"iter{i:03d}_qwen_eval.json"),
                            "w",
                        ) as f:
                            json.dump(eval_payload, f, indent=2, ensure_ascii=False)
                        if passed:
                            break
                        masked_image_caption = revised_caption
                else:
                    image_inpainting = _run_flux(masked_image_caption)

                # Always write the final image to the legacy output name.
                if image_inpainting is not None:
                    image_inpainting.save(os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}_flux_o_img.png"))

                # Update sidecar JSON with refinement results.
                try:
                    with open(prompt_json_path, "r") as f:
                        pj = json.load(f)
                    pj["Edited_image_caption_initial"] = masked_image_caption_initial
                    pj["Edited_image_caption_final"] = masked_image_caption
                    pj["Caption_refine_iterations_used"] = int(iters_used)
                    if last_notes:
                        pj["Caption_refine_last_notes"] = last_notes
                    with open(prompt_json_path, "w") as f:
                        json.dump(pj, f, indent=4, ensure_ascii=False)
                except Exception:
                    pass

                masked_image.save(os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}_gt_o_img.png"))
                gt_video_first_frame = video[0]
                video[0] = image_inpainting
                masked_video_for_pipe[0] = image_inpainting

            # -------------------------------------------------------
            # Phase "first_frame": Qwen + FluxFill done → return early.
            # -------------------------------------------------------
            if run_phase == "first_frame":
                print(f"[phase=first_frame] Saved first frame + prompt. Skipping CogVideoX.")
                return

    if pipe is None:
        raise RuntimeError("CogVideoX pipeline was not constructed.")

    if long_video:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    if generate_type == "i2v_inpainting":
        frames = inpainting_frames if inpainting_frames else 49
        down_sample_fps = fps if down_sample_fps == 0 else down_sample_fps
        print(f"Before downsample: {len(video)}, fps: {fps}, down_sample_fps: {down_sample_fps}, int(fps//down_sample_fps): {int(fps//down_sample_fps)}")
        stride = int(fps // down_sample_fps)
        video = video[::stride]
        masked_video_for_pipe = masked_video_for_pipe[::stride]
        masked_video_black = masked_video_black[::stride]
        binary_masks = binary_masks[::stride]
        print(f"After downsample: {len(video)}")
        if not long_video:
            video = video[:frames]
            masked_video_for_pipe = masked_video_for_pipe[:frames]
            masked_video_black = masked_video_black[:frames]
            binary_masks = binary_masks[:frames]
        if len(video) < frames:
            raise ValueError(f"video length is less than {frames}, using {len(video)} frames...")
            
        if first_frame_gt:
            gt_mask_first_frame = binary_masks[0]
            if mask_background:
                binary_masks[0] = Image.fromarray(np.ones_like(np.array(binary_masks[0])) * 255).convert("RGB")
            else:
                binary_masks[0] = Image.fromarray(np.zeros_like(np.array(binary_masks[0]))).convert("RGB")

        # Optional tolerance around mask boundaries for video inpainting.
        # This expands (and optionally softens) the *inpaint* region so generation can
        # extend slightly beyond the original mask for better edge consistency.
        if dilate_size and dilate_size > 0 or mask_feather and mask_feather > 0:
            binary_masks = _dilate_and_feather_mask_images(
                binary_masks,
                dilate_size=dilate_size if dilate_size and dilate_size > 0 else 0,
                feather_radius=mask_feather if mask_feather and mask_feather > 0 else 0,
                inpaint_is_white=not mask_background,
            )

        image = masked_video_for_pipe[0]
        inpaint_outputs = pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            video=masked_video_for_pipe,
            masks=binary_masks,
            strength=strength,
            replace_gt=replace_gt,
            mask_add=mask_add,
            stride= int(frames - down_sample_fps),
            prev_clip_weight=prev_clip_weight,
            id_pool_resample_learnable=True if id_adapter_resample_learnable_path is not None else False,
            output_type="np"
        ).frames[0]
        video_generate = inpaint_outputs
        binary_masks[0] = gt_mask_first_frame
        video[0] = gt_video_first_frame

        # For visualization, always show the "blacked" masked video.
        round_video = _visualize_video(
            pipe,
            mask_background,
            video[: len(video_generate)],
            video_generate,
            binary_masks[: len(video_generate)],
        )

        # Save both outputs:
        # 1) comparison video (original/masked/mask/generated)
        export_to_video(round_video, output_path, fps=8)
        # 2) generated-only video
        generated_only_path = output_path.replace(".mp4", "_generated.mp4")
        export_to_video(video_generate, generated_only_path, fps=8)
       
        print(f"{inpainting_sample_id} generate completed! Total frames: {len(round_video)}")
        print(f"Saved comparison mp4: {output_path}")
        print(f"Saved generated-only mp4: {generated_only_path}\n")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=False, default="", help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v', 'v2v', 'inpainting')"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--inpainting_branch", type=str, default=None, help="The path of the inpainting branch")
    parser.add_argument("--inpainting_mask_meta", type=str, default=None, help="The path of the inpainting mask meta")
    parser.add_argument("--inpainting_sample_id", type=int, default=None, help="The id of the inpainting sample")
    parser.add_argument("--inpainting_frames", type=int, default=None, help="The number of frames to generate")
    parser.add_argument(
        "--mask_background",
        action='store_true',
        help="Enable mask_background feature. Default is False.",
    )
    parser.add_argument(
        "--keep_masked_pixels",
        action="store_true",
        help=(
            "Keep original pixels inside the masked region as conditioning for the video inpaint pipeline. "
            "Useful when the mask covers a large area (e.g., the full road) and you need edits aligned to existing lane positions."
        ),
    )
    parser.add_argument(
        "--add_first",
        action='store_true',
        help="Enable add_first feature. Default is False.",
    )
    parser.add_argument(
        "--first_frame_gt",
        action='store_true',
        help="Enable first_frame_gt feature. Default is False.",
    )
    parser.add_argument(
        "--replace_gt",
        action='store_true',
        help="Enable replace_gt feature. Default is False.",
    )
    parser.add_argument(
        "--mask_add",
        action='store_true',
        help="Enable mask_add feature. Default is False.",
    )
    parser.add_argument(
        "--down_sample_fps",
        type=int,
        default=0,
        help="The down sample fps for the video. Default is 8.",
    )
    parser.add_argument(
        "--overlap_frames",
        type=int,
        default=0,
        help="The overlap frames for the video. Default is 0.",
    )
    parser.add_argument(
        "--prev_clip_weight",
        type=float,
        default=0.0,
        help="The weight for prev_clip. Default is 0.0.",
    )
    parser.add_argument(
        "--img_inpainting_model",
        type=str,
        default=None,
        help="The path of the image inpainting model. Default is None.",
    )
    parser.add_argument(
        "--img_inpainting_lora_path",
        type=str,
        default=None,
        help="Optional LoRA adapter path to load into FluxFillPipeline for first-frame inpainting.",
    )
    parser.add_argument(
        "--img_inpainting_lora_scale",
        type=float,
        default=1.0,
        help="LoRA scale for FluxFillPipeline adapter (default: 1.0).",
    )
    parser.add_argument(
        "--video_editing_instruction",
        type=str,
        default=None,
        help="The instruction for video editing. Default is None.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="The LLM model for video editing (e.g., 'Qwen/Qwen2.5-VL-72B-Instruct'). Default is None.",
    )
    parser.add_argument(
        "--qwen_device",
        type=str,
        default=None,
        help=(
            "Device placement for Qwen2.5-VL (e.g., 'auto', 'cpu', 'cuda:0', 'cuda:1'). "
            "Use this to pin the VLM to a different GPU than FluxFill/CogVideoX to avoid OOM. "
            "Default is None (same as 'auto')."
        ),
    )
    parser.add_argument(
        "--unload_qwen_after_caption",
        action="store_true",
        help=(
            "Unload Qwen model after generating captions/instructions to free VRAM. "
            "This is useful for single-GPU runs where FluxFill/CogVideoX also need VRAM. "
            "Automatically skipped when --caption_refine_iters > 0 (since Qwen is needed later)."
        ),
    )
    parser.add_argument(
        "--cog_device",
        type=str,
        default="cuda",
        help="Device for the main CogVideoX pipeline (e.g., 'cuda', 'cuda:0', 'cuda:1'). Default is 'cuda'.",
    )
    parser.add_argument(
        "--flux_device",
        type=str,
        default="cuda",
        help="Device for FluxFill (image inpainting) pipeline (e.g., 'cuda', 'cuda:0', 'cuda:1'). Default is 'cuda'.",
    )
    parser.add_argument(
        "--long_video",
        action='store_true',
        help="Enable long_video feature. Default is False.",
    )
    parser.add_argument(
        "--dilate_size",
        type=int,
        default=0,
        help="Mask dilation kernel size in pixels (OpenCV ones-kernel). 0 disables dilation.",
    )
    parser.add_argument(
        "--mask_feather",
        type=int,
        default=0,
        help="Optional feather radius (Gaussian blur) for mask edges, in pixels. Default is 0 (off).",
    )
    parser.add_argument(
        "--caption_refine_iters",
        type=int,
        default=0,
        help=(
            "Optional Qwen-in-the-loop refinement iterations for the first inpainted frame. "
            "If > 0 and --llm_model is set, repeats FluxFill and asks Qwen to PASS/FAIL and revise the masked caption."
        ),
    )
    parser.add_argument(
        "--caption_refine_temperature",
        type=float,
        default=0.2,
        help="Temperature for Qwen evaluation/refinement. Default is 0.2.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Inpainting strength in [0.0, 1.0]. Default is 1.0.",
    )
    parser.add_argument(
        "--id_adapter_resample_learnable_path",
        type=str,
        default=None,
        help="The path of the id_pool_resample_learnable. Default is None.",
    )
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # Allow workflow-driven defaults via environment variables.
    # CLI flags take precedence when set.
    qwen_device = args.qwen_device or (os.environ.get("VP_QWEN_DEVICE") or None)
    unload_qwen_after_caption = bool(args.unload_qwen_after_caption) or _env_flag("VP_UNLOAD_QWEN_AFTER_USE", False)

    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        inpainting_mask_meta=args.inpainting_mask_meta,
        inpainting_sample_id=args.inpainting_sample_id,
        inpainting_branch=args.inpainting_branch,
        inpainting_frames=args.inpainting_frames,
        mask_background=args.mask_background,
        keep_masked_pixels=args.keep_masked_pixels,
        add_first=args.add_first,
        first_frame_gt=args.first_frame_gt,
        replace_gt=args.replace_gt,
        mask_add=args.mask_add,
        down_sample_fps=args.down_sample_fps,
        overlap_frames=args.overlap_frames,
        prev_clip_weight=args.prev_clip_weight,
        img_inpainting_model=args.img_inpainting_model,
        img_inpainting_lora_path=args.img_inpainting_lora_path,
        img_inpainting_lora_scale=args.img_inpainting_lora_scale,
        video_editing_instruction=args.video_editing_instruction,
        llm_model=args.llm_model,
        qwen_device=qwen_device,
        unload_qwen_after_caption=unload_qwen_after_caption,
        cog_device=args.cog_device,
        flux_device=args.flux_device,
        long_video=args.long_video,
        dilate_size=args.dilate_size,
        mask_feather=args.mask_feather,
        caption_refine_iters=args.caption_refine_iters,
        caption_refine_temperature=args.caption_refine_temperature,
        id_adapter_resample_learnable_path=args.id_adapter_resample_learnable_path,
        strength=args.strength,
    )
