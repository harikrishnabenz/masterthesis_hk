"""LoRA fine-tuning for FLUX.1 Fill (FluxFillPipeline) on inpainting data.

Expected CSV columns (configurable):
  - image: path to RGB image
  - mask: path to grayscale mask (white=region to inpaint)
  - prompt: text prompt
  - prompt_2: optional second prompt (defaults to prompt)
"""

import argparse
import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
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
        base_dir = os.path.dirname(os.path.abspath(csv_path))
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

                if not os.path.isabs(image_path):
                    image_path = os.path.join(base_dir, image_path)
                if not os.path.isabs(mask_path):
                    mask_path = os.path.join(base_dir, mask_path)

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
    parser.add_argument("--max_train_steps", type=int, default=None, help="Max training steps. If None, trains for full epochs to cover all data")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs to train (used if max_train_steps is None)")

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
    
    training_start_time = time.time()
    training_start_datetime = datetime.now()

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
    
    dataset_size = len(dataset)
    logger.info(f"Dataset loaded: {dataset_size} training examples")

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
    
    # Calculate training steps and epochs
    if args.max_train_steps is None:
        # Train for specified epochs to cover all data
        num_train_epochs = args.num_train_epochs
        args.max_train_steps = num_train_epochs * num_update_steps_per_epoch
        logger.info(f"Training for {num_train_epochs} epoch(s) = {args.max_train_steps} steps to cover all {dataset_size} images")
    else:
        # Use specified max_train_steps
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        logger.info(f"Training for {args.max_train_steps} steps (~{num_train_epochs} epoch(s))")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    def unwrap_model(model):
        # Avoid DeepSpeed import issues by manually unwrapping
        # instead of using accelerator.unwrap_model which tries to import DeepSpeed
        from torch.nn.parallel import DistributedDataParallel
        from torch.nn.parallel import DataParallel
        
        # Unwrap DDP/DP wrapper
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        
        # Unwrap compiled module
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
    loss_history = []  # Track loss for summary
    epoch_losses = []  # Track average loss per epoch

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

    logger.info(f"Starting training for {num_train_epochs} epoch(s), {args.max_train_steps} total steps")
    epoch_step_losses = []  # Track losses within current epoch

    for epoch in range(num_train_epochs):
        epoch_start_time = time.time()
        epoch_step_losses = []
        
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
                
                # Track loss
                current_loss = loss.detach().item()
                epoch_step_losses.append(current_loss)

            if accelerator.sync_gradients:
                global_step += 1
                loss_history.append((global_step, current_loss))

                if accelerator.is_main_process and (global_step % 10 == 0):
                    logger.info(f"epoch={epoch+1}/{num_train_epochs} step={global_step}/{args.max_train_steps} loss={current_loss:.6f}")

                if (
                    accelerator.is_main_process
                    and args.checkpointing_steps
                    and (global_step % args.checkpointing_steps == 0)
                ):
                    accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))

                if global_step >= args.max_train_steps:
                    break

        # Log epoch summary
        if epoch_step_losses:
            epoch_avg_loss = sum(epoch_step_losses) / len(epoch_step_losses)
            epoch_losses.append((epoch + 1, epoch_avg_loss))
            epoch_duration = time.time() - epoch_start_time
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch+1}/{num_train_epochs} completed in {epoch_duration:.2f}s - Average loss: {epoch_avg_loss:.6f}")
        
        if global_step >= args.max_train_steps:
            break

    training_duration = time.time() - training_start_time
    training_end_datetime = datetime.now()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save_state(args.output_dir)
        logger.info(f"Done. LoRA weights saved to: {args.output_dir}")
        
        # Generate training summary
        summary_path = os.path.join(args.output_dir, "training_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("FLUX.1 FILL LORA TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("TRAINING INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Training Start:     {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training End:       {training_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration:     {training_duration/3600:.2f} hours ({training_duration/60:.2f} minutes)\n")
            f.write(f"Output Directory:   {args.output_dir}\n\n")
            
            f.write("MODEL INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Model Type:         FLUX.1 Fill (FluxFillPipeline)\n")
            f.write(f"Training Method:    LoRA (Low-Rank Adaptation)\n")
            f.write(f"Base Model Path:    {args.pretrained_model_name_or_path}\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Training CSV:       {args.train_csv}\n")
            f.write(f"Total Images:       {dataset_size}\n")
            f.write(f"Image Column:       {args.image_column}\n")
            f.write(f"Mask Column:        {args.mask_column}\n")
            f.write(f"Caption Column:     {args.caption_column}\n")
            f.write(f"Caption 2 Column:   {args.caption_2_column}\n")
            f.write(f"Image Resolution:   {args.height}x{args.width}\n")
            f.write(f"Mask Inverted:      {args.invert_mask}\n\n")
            
            f.write("TRAINING HYPERPARAMETERS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of Epochs:           {num_train_epochs}\n")
            f.write(f"Total Training Steps:       {global_step} / {args.max_train_steps}\n")
            f.write(f"Steps per Epoch:            {num_update_steps_per_epoch}\n")
            f.write(f"Batch Size:                 {args.train_batch_size}\n")
            f.write(f"Gradient Accumulation:      {args.gradient_accumulation_steps}\n")
            f.write(f"Effective Batch Size:       {args.train_batch_size * args.gradient_accumulation_steps}\n")
            f.write(f"Learning Rate:              {args.learning_rate}\n")
            f.write(f"LR Scheduler:               {args.lr_scheduler}\n")
            f.write(f"LR Warmup Steps:            {args.lr_warmup_steps}\n")
            f.write(f"Mixed Precision:            {args.mixed_precision}\n")
            f.write(f"Gradient Checkpointing:     {args.gradient_checkpointing}\n")
            f.write(f"Random Seed:                {args.seed if args.seed is not None else 'Not set'}\n\n")
            
            f.write("LORA CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"LoRA Rank (r):              {args.rank}\n")
            f.write(f"LoRA Alpha:                 {args.lora_alpha}\n")
            f.write(f"Target Modules:             to_q, to_k, to_v, to_out.0\n")
            f.write(f"Trainable Parameters:       Transformer LoRA layers only\n")
            f.write(f"Frozen Components:          VAE, Text Encoders, Base Transformer\n\n")
            
            f.write("TRAINING PROGRESS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Images Processed:           ~{global_step * args.train_batch_size * args.gradient_accumulation_steps}\n")
            f.write(f"Checkpointing Interval:     Every {args.checkpointing_steps} steps\n")
            if args.resume_from_checkpoint:
                f.write(f"Resumed from:               {args.resume_from_checkpoint}\n")
            f.write("\n")
            
            f.write("LOSS METRICS\n")
            f.write("-" * 80 + "\n")
            if loss_history:
                f.write(f"Loss Function:              MSE (Mean Squared Error)\n")
                f.write(f"Initial Loss (step 1):      {loss_history[0][1]:.6f}\n")
                f.write(f"Final Loss (step {loss_history[-1][0]}):     {loss_history[-1][1]:.6f}\n")
                
                # Calculate average loss in first and last 10% of training
                first_10_percent = max(1, len(loss_history) // 10)
                last_10_percent = max(1, len(loss_history) // 10)
                early_avg = sum(l[1] for l in loss_history[:first_10_percent]) / first_10_percent
                late_avg = sum(l[1] for l in loss_history[-last_10_percent:]) / last_10_percent
                
                f.write(f"Early Training Avg Loss:    {early_avg:.6f} (first 10%)\n")
                f.write(f"Late Training Avg Loss:     {late_avg:.6f} (last 10%)\n")
                f.write(f"Loss Reduction:             {((early_avg - late_avg) / early_avg * 100):.2f}%\n")
                f.write(f"Min Loss:                   {min(l[1] for l in loss_history):.6f}\n")
                f.write(f"Max Loss:                   {max(l[1] for l in loss_history):.6f}\n")
            
            if epoch_losses:
                f.write("\nPer-Epoch Average Loss:\n")
                for epoch_num, avg_loss in epoch_losses:
                    f.write(f"  Epoch {epoch_num}: {avg_loss:.6f}\n")
            f.write("\n")
            
            f.write("SYSTEM CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accelerator State:          {accelerator.state}\n")
            f.write(f"Distributed Training:       {accelerator.num_processes > 1}\n")
            f.write(f"Number of Processes:        {accelerator.num_processes}\n")
            f.write(f"Device:                     {accelerator.device}\n")
            f.write(f"Data Loader Workers:        4\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("DETAILED LOSS HISTORY\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Step':<10} {'Loss':<15}\n")
            f.write("-" * 25 + "\n")
            for step, loss_val in loss_history:
                f.write(f"{step:<10} {loss_val:<15.6f}\n")
            
        logger.info(f"Training summary saved to: {summary_path}")
        
        # Also save loss history as JSON for easier parsing
        json_path = os.path.join(args.output_dir, "loss_history.json")
        with open(json_path, "w") as f:
            json.dump({
                "loss_history": [{"step": s, "loss": l} for s, l in loss_history],
                "epoch_losses": [{"epoch": e, "avg_loss": l} for e, l in epoch_losses],
                "training_info": {
                    "dataset_size": dataset_size,
                    "total_steps": global_step,
                    "num_epochs": num_train_epochs,
                    "batch_size": args.train_batch_size,
                    "learning_rate": args.learning_rate,
                    "duration_seconds": training_duration
                }
            }, f, indent=2)
        logger.info(f"Loss history JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
