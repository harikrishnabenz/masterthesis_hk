#!/usr/bin/env python3

"""Build a FluxFill LoRA training CSV.

Two modes:

1) Auto-caption + random masks (synthetic holes)
    - input: --images_dir
    - output: masks/*.png + CSV columns: image,mask,prompt,prompt_2
    - requires: openai package + OPENAI_API_KEY

2) Use your provided prompt + final image + mask
    - input: --input_csv containing at least: image, mask, prompt
    - output: normalized CSV with columns: image,mask,prompt,prompt_2

The resulting CSV is compatible with: train/train_fluxfill_inpaint_lora.py
"""

import argparse
import base64
import csv
import os
import random
import time
import importlib
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw


_qwen_model = None
_qwen_processor = None
_qwen_model_id = None
_qwen_device = None
_process_vision_info = None


def _iter_images(images_dir: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _encode_image_b64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_rect_mask(w: int, h: int, rng: random.Random, *, min_frac: float, max_frac: float) -> Image.Image:
    mask = Image.new("L", (w, h), 0)
    area = w * h
    target = int(area * rng.uniform(min_frac, max_frac))

    # Simple: pick rectangle dimensions with roughly target area.
    rect_w = max(1, int((target * rng.uniform(0.4, 1.6)) ** 0.5))
    rect_h = max(1, int(target / max(1, rect_w)))
    rect_w = min(rect_w, w)
    rect_h = min(rect_h, h)

    x0 = rng.randint(0, max(0, w - rect_w))
    y0 = rng.randint(0, max(0, h - rect_h))
    x1 = x0 + rect_w
    y1 = y0 + rect_h

    draw = ImageDraw.Draw(mask)
    draw.rectangle([x0, y0, x1, y1], fill=255)
    return mask


def _make_brush_mask(w: int, h: int, rng: random.Random, *, strokes: int = 3) -> Image.Image:
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    max_dim = max(w, h)
    min_width = max(4, max_dim // 20)
    max_width = max(min_width + 1, max_dim // 8)

    for _ in range(strokes):
        width = rng.randint(min_width, max_width)
        n_pts = rng.randint(3, 10)
        pts: List[Tuple[int, int]] = []
        x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
        pts.append((x, y))
        for _k in range(n_pts - 1):
            dx = rng.randint(-w // 3, w // 3)
            dy = rng.randint(-h // 3, h // 3)
            x = max(0, min(w - 1, x + dx))
            y = max(0, min(h - 1, y + dy))
            pts.append((x, y))

        draw.line(pts, fill=255, width=width)
        # thicken endpoints
        for x, y in pts:
            r = width // 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    return mask


def _caption_openai(pil_img: Image.Image, *, model: str, system_prompt: str, user_prompt: str, timeout: float):
    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover
        raise RuntimeError("openai package not available. Install it or run inside the VideoPainter environment.") from e

    client = OpenAI(timeout=timeout)
    img_b64 = _encode_image_b64(pil_img)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            },
        ],
    )

    text = (resp.choices[0].message.content or "").strip()
    return text


def _get_qwen_model(model_id: str, *, qwen_device: str | None = None):
    global _qwen_model, _qwen_processor, _qwen_model_id, _qwen_device

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _QwenForConditionalGeneration, AutoProcessor
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Qwen2.5-VL captioning requires `transformers`, `accelerate`, and `qwen-vl-utils`. "
            "You may need a newer Transformers build: "
            "`pip install git+https://github.com/huggingface/transformers accelerate` and `pip install qwen-vl-utils`."
        ) from e

    qwen_device_key = (qwen_device or "auto").strip()
    if (
        _qwen_model is not None
        and _qwen_processor is not None
        and _qwen_model_id == model_id
        and (_qwen_device or "auto").strip() == qwen_device_key
    ):
        return _qwen_model, _qwen_processor

    is_local_path = os.path.isdir(model_id)

    import torch
    import tempfile

    qwen_torch_dtype: torch.dtype = torch.bfloat16
    model_load_kwargs: dict = {
        "local_files_only": is_local_path,
        "low_cpu_mem_usage": True,
    }

    # For 72B, device_map="auto" with CPU offload is the safest default.
    if qwen_device_key.lower() == "cpu":
        qwen_torch_dtype = torch.float32
        model_load_kwargs["device_map"] = {"": "cpu"}
    else:
        model_load_kwargs["device_map"] = "auto"
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
                total_gib = max(1, int(total_bytes / (1024**3)))
                if gpu_count <= 1:
                    budget_gib = max(4, min(16, total_gib - 8))
                else:
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
    _qwen_processor = AutoProcessor.from_pretrained(model_id, local_files_only=is_local_path)
    _qwen_model_id = model_id
    _qwen_device = qwen_device_key
    return _qwen_model, _qwen_processor


def _caption_qwen(
    pil_img: Image.Image,
    *,
    model_id: str,
    qwen_device: str | None,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None,
    max_output_tokens: int | None,
) -> str:
    global _process_vision_info

    import torch

    qwen_model, qwen_processor = _get_qwen_model(model_id, qwen_device=qwen_device)

    if _process_vision_info is None:
        try:
            _process_vision_info = importlib.import_module("qwen_vl_utils").process_vision_info
        except Exception as e:  # pragma: no cover
            raise ImportError("qwen-vl-utils is required for vision inputs. Install it with: `pip install qwen-vl-utils`") from e

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = _process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(qwen_model.device)

    generation_kwargs = {}
    if temperature is not None:
        generation_kwargs["temperature"] = float(temperature)
    generation_kwargs["max_new_tokens"] = int(max_output_tokens) if max_output_tokens is not None else 128

    with torch.inference_mode():
        generated_ids = qwen_model.generate(**inputs, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (output_text[0] or "").strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-caption images + generate masks + write FluxFill inpaint CSV")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images_dir", type=str, help="Folder containing source images (auto-caption + random masks mode)")
    src.add_argument(
        "--input_csv",
        type=str,
        help="Existing CSV containing your provided columns (at least: image, mask, prompt)",
    )

    p.add_argument("--out_dir", type=str, required=True, help="Output folder for masks + CSV")
    p.add_argument("--csv_name", type=str, default="train.csv", help="CSV filename to write inside out_dir")

    # Only used in --images_dir mode (synthetic holes)
    p.add_argument("--mask_type", type=str, default="brush", choices=["brush", "rect", "mixed"])
    p.add_argument("--rect_min_frac", type=float, default=0.05)
    p.add_argument("--rect_max_frac", type=float, default=0.25)
    p.add_argument("--brush_strokes", type=int, default=3)

    # Captioning backend
    p.add_argument(
        "--caption_backend",
        type=str,
        default="qwen",
        choices=["qwen", "openai"],
        help="Captioning backend. Use 'qwen' for local Qwen2.5-VL, or 'openai' for OpenAI Vision.",
    )

    # Qwen captioning
    p.add_argument(
        "--qwen_model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-72B-Instruct",
        help="HuggingFace model id or local path for Qwen2.5-VL (default: 72B Instruct).",
    )
    p.add_argument(
        "--qwen_device",
        type=str,
        default="auto",
        help="Device placement hint for Qwen (e.g. 'auto', 'cpu', 'cuda', 'cuda:1').",
    )
    p.add_argument("--qwen_temperature", type=float, default=None)
    p.add_argument("--qwen_max_new_tokens", type=int, default=128)

    # OpenAI captioning
    p.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    p.add_argument("--openai_timeout", type=float, default=60.0)

    p.add_argument("--sleep", type=float, default=0.0, help="Sleep between caption requests (seconds)")

    p.add_argument("--max_images", type=int, default=0, help="If >0, limits number of images processed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--relative_paths", action="store_true", help="Write image/mask paths relative to current working dir")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir) if args.images_dir else None
    out_dir = Path(args.out_dir)
    out_masks = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only create mask folder in synthetic-mask mode
    if images_dir is not None:
        out_masks.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    system_prompt = "You are an expert image captioner."
    user_prompt = (
        "Write a short, concrete caption of this image for training a text-guided inpainting model. "
        "Mention the main objects, scene type, materials/colors, and overall style. "
        "Do not mention masks, holes, or editing. Return only the caption."
    )

    csv_path = out_dir / args.csv_name

    rows = []
    n = 0

    if args.input_csv:
        input_csv = Path(args.input_csv)
        if not input_csv.exists():
            raise SystemExit(f"input_csv does not exist: {input_csv}")

        with open(input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image = (row.get("image") or "").strip()
                mask = (row.get("mask") or "").strip()
                prompt = (row.get("prompt") or "").strip()
                prompt_2 = (row.get("prompt_2") or "").strip()

                if not image or not mask:
                    continue

                img_path = Path(image)
                mask_path = Path(mask)
                if not img_path.exists():
                    raise SystemExit(f"Missing image referenced in CSV: {img_path}")
                if not mask_path.exists():
                    raise SystemExit(f"Missing mask referenced in CSV: {mask_path}")

                if not prompt:
                    img_pil = Image.open(img_path)
                    if img_pil.mode != "RGB":
                        img_pil = img_pil.convert("RGB")
                    if args.caption_backend == "qwen":
                        prompt = _caption_qwen(
                            img_pil,
                            model_id=args.qwen_model_id,
                            qwen_device=args.qwen_device,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=args.qwen_temperature,
                            max_output_tokens=args.qwen_max_new_tokens,
                        )
                    else:
                        prompt = _caption_openai(
                            img_pil,
                            model=args.openai_model,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            timeout=args.openai_timeout,
                        )

                if not prompt_2:
                    prompt_2 = prompt

                img_out = img_path
                mask_out = mask_path
                if args.relative_paths:
                    img_out = Path(os.path.relpath(img_path, Path.cwd()))
                    mask_out = Path(os.path.relpath(mask_path, Path.cwd()))

                rows.append({"image": str(img_out), "mask": str(mask_out), "prompt": prompt, "prompt_2": prompt_2})
                n += 1
                if args.max_images and n >= args.max_images:
                    break

        if n == 0:
            raise SystemExit(f"No usable rows found in input_csv: {input_csv}")

    else:
        assert images_dir is not None
        for img_path in _iter_images(images_dir):
            if args.max_images and n >= args.max_images:
                break

            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            w, h = img.size
            if args.mask_type == "rect":
                mask = _make_rect_mask(w, h, rng, min_frac=args.rect_min_frac, max_frac=args.rect_max_frac)
            elif args.mask_type == "brush":
                mask = _make_brush_mask(w, h, rng, strokes=args.brush_strokes)
            else:
                if rng.random() < 0.5:
                    mask = _make_rect_mask(w, h, rng, min_frac=args.rect_min_frac, max_frac=args.rect_max_frac)
                else:
                    mask = _make_brush_mask(w, h, rng, strokes=args.brush_strokes)

            # Always save mask as PNG to preserve 0/255.
            mask_name = img_path.stem + "_mask.png"
            mask_path = out_masks / mask_name
            mask.save(mask_path)

            if args.caption_backend == "qwen":
                caption = _caption_qwen(
                    img,
                    model_id=args.qwen_model_id,
                    qwen_device=args.qwen_device,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.qwen_temperature,
                    max_output_tokens=args.qwen_max_new_tokens,
                )
            else:
                caption = _caption_openai(
                    img,
                    model=args.openai_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    timeout=args.openai_timeout,
                )

            img_out = img_path
            mask_out = mask_path
            if args.relative_paths:
                img_out = Path(os.path.relpath(img_path, Path.cwd()))
                mask_out = Path(os.path.relpath(mask_path, Path.cwd()))

            rows.append({"image": str(img_out), "mask": str(mask_out), "prompt": caption, "prompt_2": caption})

            n += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

        if n == 0:
            raise SystemExit(f"No images found under: {images_dir}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "mask", "prompt", "prompt_2"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {n} rows to: {csv_path}")
    if images_dir is not None:
        print(f"Masks saved under: {out_masks}")


if __name__ == "__main__":
    main()
