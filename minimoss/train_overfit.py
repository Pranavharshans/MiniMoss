#!/usr/bin/env python3
"""Overfit training script for MiniMoss.

Usage:
    python -m minimoss.train_overfit \\
        --manifest data/manifest.jsonl \\
        --token-dir data/tokens \\
        --output-dir checkpoints \\
        --overfit-one-batch
"""

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import MiniMossConfig
from .model import MiniMossModel
from .dataset import MiniMossDataset, collate_fn
from .utils import set_seed, format_metrics


def weighted_loss(group_losses, weights):
    return sum(weight * loss for weight, loss in zip(weights, group_losses)) / sum(weights)


def trainable_state_dict(model):
    """Return a checkpoint without duplicating the frozen Hugging Face backbone."""
    return {key: value for key, value in model.state_dict().items() if not key.startswith("_backbone.")}


def save_checkpoint(path, step, model, optimizer, scaler, config, best_validation_loss):
    torch.save({
        "step": step,
        "model_state_dict": trainable_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "best_validation_loss": best_validation_loss,
    }, path)


@torch.inference_mode()
def validate(model, loader, device, weights, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_groups = [0.0] * len(weights)
    batches = 0
    for text, text_mask, rvq, audio_mask in loader:
        text = text.to(device)
        text_mask = text_mask.to(device)
        rvq = rvq.to(device)
        audio_mask = audio_mask.to(device)
        _, group_losses = model(
            text,
            rvq,
            text_attention_mask=text_mask,
            audio_frame_mask=audio_mask,
        )
        total_loss += weighted_loss(group_losses, weights).item()
        for index, loss in enumerate(group_losses):
            total_groups[index] += loss.item()
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    model.train()
    if batches == 0:
        raise ValueError("Validation loader produced no batches")
    return total_loss / batches, [loss / batches for loss in total_groups]


def main():
    parser = argparse.ArgumentParser(description="Train MiniMoss overfit test")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-manifest", default=None)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--overfit-one-batch", action="store_true",
                        help="Overfit a single batch for sanity check")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=None,
                        help="Limit validation batches; default evaluates the full manifest")
    parser.add_argument("--resume", default=None, help="Resume from a training checkpoint")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--codec", default="OpenMOSS-Team/MOSS-Audio-Tokenizer",
                        help="Tokenizer name/checkpoint used to create the token files")
    parser.add_argument("--n-codebooks", type=int, default=32,
                        help="Number of RVQ codebooks in the token files")
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    config = MiniMossConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        codec_name=args.codec,
        n_codebooks=args.n_codebooks,
        n_groups=args.n_codebooks // 4,
        codebook_size=args.codebook_size,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        token_dir=args.token_dir,
        use_amp=not args.no_amp,
    )
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.validate_every <= 0:
        raise ValueError("--validate-every must be positive")
    if args.validation_batches is not None and args.validation_batches <= 0:
        raise ValueError("--validation-batches must be positive")

    print("=" * 60)
    print("MiniMoss Overfit Training")
    print("=" * 60)
    print(f"  backbone: {config.backbone_name}")
    print(f"  audio tokenizer name: {config.codec_name}")
    print(f"  local layers: {config.local_num_layers}")
    print(f"  local hidden: {config.local_hidden_size}")
    print(f"  n_codebooks: {config.n_codebooks}")
    print(f"  n_groups: {config.n_groups}")
    print("  qwen: frozen")
    print("  trainable: frame conditioner + projection + RVQ embeddings + local decoder + heads")
    print(f"  device: {args.device}")
    print(f"  overfit_one_batch: {args.overfit_one_batch}")
    print(f"  validation_manifest: {args.validation_manifest}")
    print(f"  resume: {args.resume}")

    # Dataset
    dataset = MiniMossDataset(
        manifest_path=args.manifest,
        token_dir=args.token_dir,
        max_frames=config.max_frames,
        n_codebooks=config.n_codebooks,
        codebook_size=config.codebook_size,
    )
    print(f"\nDataset: {len(dataset)} utterances")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=not args.overfit_one_batch,
        collate_fn=collate_fn,
        drop_last=False,
    )
    validation_loader = None
    if args.validation_manifest:
        validation_dataset = MiniMossDataset(
            manifest_path=args.validation_manifest,
            token_dir=args.token_dir,
            max_frames=config.max_frames,
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )
        print(f"Validation dataset: {len(validation_dataset)} utterances")

    # Model
    print("\nLoading model...")
    model = MiniMossModel(config)
    model.to(args.device)
    _ = model.backbone
    model.backbone.to(args.device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if config.use_amp and args.device.startswith("cuda") else None

    start_step = 0
    best_validation_loss = float("inf")
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = int(checkpoint["step"])
        best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        print(f"  resumed at step {start_step}; best validation loss={best_validation_loss:.4f}")

    # Get one batch for overfit mode
    if args.overfit_one_batch:
        text_batch, mask_batch, rvq_batch, audio_mask_batch = next(iter(loader))
        text_batch = text_batch.to(args.device)
        mask_batch = mask_batch.to(args.device)
        rvq_batch = rvq_batch.to(args.device)
        audio_mask_batch = audio_mask_batch.to(args.device)
        print(f"  overfit batch: text={list(text_batch.shape)}, rvq={list(rvq_batch.shape)}")

    # Training loop
    model.train()
    step = 0
    total_steps = config.max_steps
    if args.overfit_one_batch:
        total_steps = 500  # enough to overfit one batch

    if start_step >= total_steps:
        raise ValueError(f"Resume step {start_step} is already at or beyond max steps {total_steps}")
    print(f"\nTraining from step {start_step + 1} through {total_steps}...\n")

    # Create iterator once, recreate on exhaustion
    if not args.overfit_one_batch:
        data_iter = iter(loader)

    for step in range(start_step + 1, total_steps + 1):
        t_start = time.time()

        if args.overfit_one_batch:
            text_input, mask_input = text_batch, mask_batch
            rvq_input, audio_mask_input = rvq_batch, audio_mask_batch
        else:
            try:
                text_input, mask_input, rvq_input, audio_mask_input = next(data_iter)
            except StopIteration:
                data_iter = iter(DataLoader(
                    dataset, batch_size=config.batch_size, shuffle=True,
                    collate_fn=collate_fn, drop_last=False,
                ))
                text_input, mask_input, rvq_input, audio_mask_input = next(data_iter)
            text_input = text_input.to(args.device)
            mask_input = mask_input.to(args.device)
            rvq_input = rvq_input.to(args.device)
            audio_mask_input = audio_mask_input.to(args.device)

        # Forward
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits, group_losses = model(
                    text_input,
                    rvq_input,
                    text_attention_mask=mask_input,
                    audio_frame_mask=audio_mask_input,
                )
        else:
            logits, group_losses = model(
                text_input,
                rvq_input,
                text_attention_mask=mask_input,
                audio_frame_mask=audio_mask_input,
            )

        # Weighted total loss
        weights = config.group_loss_weights
        total_loss = weighted_loss(group_losses, weights)

        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

        step_time = time.time() - t_start

        # Log
        if step % config.log_every == 0 or step == 1:
            g_losses = [gl.item() for gl in group_losses]
            print(format_metrics(step, total_loss.item(), g_losses, config.learning_rate, step_time))

        if validation_loader is not None and step % args.validate_every == 0:
            validation_loss, validation_groups = validate(
                model,
                validation_loader,
                args.device,
                weights,
                args.validation_batches,
            )
            group_text = " | ".join(
                f"val_g{index + 1}={loss:.4f}"
                for index, loss in enumerate(validation_groups)
            )
            print(f"validation step={step} | val_loss={validation_loss:.4f} | {group_text}")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_path = Path(args.output_dir) / "best_validation.pt"
                save_checkpoint(
                    best_path, step, model, optimizer, scaler, config, best_validation_loss
                )
                print(f"  -> new best validation checkpoint: {best_path}")

        # Checkpoint
        if step % config.checkpoint_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"step_{step}.pt")
            save_checkpoint(
                ckpt_path, step, model, optimizer, scaler, config, best_validation_loss
            )
            print(f"  -> saved {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(args.output_dir, "final.pt")
    save_checkpoint(
        final_path, step, model, optimizer, scaler, config, best_validation_loss
    )
    print(f"\nFinal checkpoint: {final_path}")
    print("Done!")


if __name__ == "__main__":
    main()
