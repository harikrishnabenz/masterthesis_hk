"""LoRA fine-tuning for FLUX.1 Fill (FluxFillPipeline) on inpainting data.

Expected CSV columns (configurable):
  - image: path to RGB image
  - mask: path to grayscale mask (white=region to inpaint)
  - prompt: text prompt
  - prompt_2: optional second prompt (defaults to prompt)
"""

import argparse
import csv
import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from diffusers import FluxFillPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module


check_min_version("0.31.0.dev0")
logger = get_logger(__name__)


@dataclass
class InpaintExample:
    image_path: str
    mask_path: str
    prompt: str
    prompt_2: Optional[str] = None


class InpaintCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_column: str = "image",
        mask_column: str = "mask",
        caption_column: str = "prompt",
        caption_2_column: Optional[str] = "prompt_2",
    ):
        self.examples: List[InpaintExample] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = (row.get(image_column) or "").strip()
                mask_path = (row.get(mask_column) or "").strip()
                prompt = (row.get(caption_column) or "").strip()
                prompt_2 = None
                if caption_2_column:
                    prompt_2 = (row.get(caption_2_column) or "").strip() or None
                if not image_path or not mask_path or not prompt:
                    continue
                self.examples.append(
                    InpaintExample(image_path=image_path, mask_path=mask_path, prompt=prompt, prompt_2=prompt_2)
                )

        if len(self.examples) == 0:
            raise ValueError(
                f"No training rows found in {csv_path}. Expected columns: {image_column}, {mask_column}, {caption_column}."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> InpaintExample:
        return self.examples[idx]


def _load_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _load_mask(path: str) -> Image.Image:
    m = Image.open(path)
    if m.mode != "L":
        m = m.convert("L")
    return m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune FLUX.1 Fill (FluxFillPipeline) for domain inpainting")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--mask_column", type=str, default="mask")
    parser.add_argument("--caption_column", type=str, default="prompt")
    parser.add_argument("--caption_2_column", type=str, default="prompt_2")

    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=512)

    parser.add_argument("--invert_mask", action="store_true", help="If set, inverts the mask (expects black=inpaint)")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "cosine", "linear", "cosine_with_restarts"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1000)

    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)

    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard"])

    args = parser.parse_args()

    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be > 0")
    if args.max_sequence_length > 512:
        raise ValueError("FluxFillPipeline only supports max_sequence_length <= 512")

    return args


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("wandb is not installed but --report_to wandb was set")
        import wandb  # noqa: F401

    logger.info(accelerator.state, main_process_only=False)

    pipe = FluxFillPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    pipe.transformer.add_adapter(transformer_lora_config)
    pipe.transformer.train()

    if accelerator.mixed_precision == "fp16":
        cast_training_params([pipe.transformer], dtype=torch.float32)

    trainable_params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    dataset = InpaintCsvDataset(
        csv_path=args.train_csv,
        image_column=args.image_column,
        mask_column=args.mask_column,
        caption_column=args.caption_column,
        caption_2_column=args.caption_2_column,
    )

    def collate(examples: List[InpaintExample]):
        images = [_load_rgb(e.image_path) for e in examples]
        masks = [_load_mask(e.mask_path) for e in examples]
        prompts = [e.prompt for e in examples]
        prompts_2 = [e.prompt_2 for e in examples]
        return images, masks, prompts, prompts_2

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
        pin_memory=True,
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return
        transformer_lora_layers_to_save = None
        for model in models:
            if isinstance(unwrap_model(model), type(unwrap_model(pipe.transformer))):
                transformer_lora_layers_to_save = get_peft_model_state_dict(unwrap_model(model))
            else:
                raise ValueError(f"Unexpected model in save hook: {model.__class__}")
            if weights:
                weights.pop()

        FluxFillPipeline.save_lora_weights(output_dir, transformer_lora_layers=transformer_lora_layers_to_save)

    def load_model_hook(models, input_dir):
        transformer_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(unwrap_model(model), type(unwrap_model(pipe.transformer))):
                transformer_ = model
            else:
                raise ValueError(f"Unexpected model in load hook: {model.__class__}")

        lora_state_dict = FluxFillPipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            k.replace("transformer.", ""): v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        _ = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    pipe.transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        pipe.transformer, optimizer, dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    global_step = 0

    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        # Best-effort step restore from folder name: checkpoint-<global_step>
        ckpt_name = os.path.basename(os.path.normpath(args.resume_from_checkpoint))
        if ckpt_name.startswith("checkpoint-"):
            try:
                global_step = int(ckpt_name.split("checkpoint-")[-1])
            except ValueError:
                global_step = 0
        logger.info(f"Resumed from checkpoint: {args.resume_from_checkpoint} (global_step={global_step})")

    for _epoch in range(num_train_epochs):
        for images, masks, prompts, prompts_2 in dataloader:
            with accelerator.accumulate(pipe.transformer):
                device = accelerator.device

                image = pipe.image_processor.preprocess(images, height=args.height, width=args.width).to(
                    device=device, dtype=weight_dtype
                )
                mask = pipe.mask_processor.preprocess(masks, height=args.height, width=args.width).to(device=device)
                if args.invert_mask:
                    mask = 1.0 - mask

                masked_image = (image * (1.0 - mask)).to(device=device, dtype=weight_dtype)

                batch_size = image.shape[0]
                height_px, width_px = image.shape[-2], image.shape[-1]
                num_channels_latents = pipe.vae.config.latent_channels

                x0 = pipe.vae.encode(image).latent_dist.sample()
                x0 = (x0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                x0 = x0.to(device=device, dtype=weight_dtype)

                height_lat = 2 * (int(height_px) // (pipe.vae_scale_factor * 2))
                width_lat = 2 * (int(width_px) // (pipe.vae_scale_factor * 2))
                x0 = pipe._pack_latents(x0, batch_size, num_channels_latents, height_lat, width_lat)

                packed_mask, masked_image_latents = pipe.prepare_mask_latents(
                    mask,
                    masked_image,
                    batch_size,
                    num_channels_latents,
                    1,
                    height_px,
                    width_px,
                    weight_dtype,
                    device,
                    generator=None,
                )
                masked_image_latents = torch.cat((masked_image_latents, packed_mask), dim=-1)

                prompt_2 = [p2 if p2 is not None else p for p, p2 in zip(prompts, prompts_2)]
                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=prompts,
                    prompt_2=prompt_2,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=args.max_sequence_length,
                )

                latent_image_ids = pipe._prepare_latent_image_ids(
                    batch_size,
                    height_lat // 2,
                    width_lat // 2,
                    device=device,
                    dtype=prompt_embeds.dtype,
                )

                idx = torch.randint(
                    low=0,
                    high=len(pipe.scheduler.timesteps),
                    size=(batch_size,),
                    device=torch.device("cpu"),
                )
                timesteps = pipe.scheduler.timesteps[idx].to(device=device, dtype=torch.float32)

                noise = torch.randn_like(x0)
                noisy_latents = pipe.scheduler.scale_noise(sample=x0, timestep=timesteps, noise=noise)

                target = noise - x0

                guidance = None
                if getattr(pipe.transformer.config, "guidance_embeds", False):
                    guidance = torch.ones((batch_size,), device=device, dtype=torch.float32)

                model_pred = pipe.transformer(
                    hidden_states=torch.cat((noisy_latents, masked_image_latents), dim=2),
                    timestep=(timesteps.to(noisy_latents.dtype) / 1000.0),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process and (global_step % 10 == 0):
                    logger.info(f"step={global_step} loss={loss.detach().item():.6f}")

                if (
                    accelerator.is_main_process
                    and args.checkpointing_steps
                    and (global_step % args.checkpointing_steps == 0)
                ):
                    accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save_state(args.output_dir)
        logger.info(f"Done. LoRA weights saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
