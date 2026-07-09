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
import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from .config import MiniMossConfig
from .model import MiniMossModel
from .dataset import MiniMossDataset, collate_fn
from .utils import set_seed, format_metrics


def main():
    parser = argparse.ArgumentParser(description="Train MiniMoss overfit test")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--overfit-one-batch", action="store_true",
                        help="Overfit a single batch for sanity check")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-text", default=None,
                        help="Text to generate during training samples")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    config = MiniMossConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        token_dir=args.token_dir,
        use_amp=not args.no_amp,
    )
    if args.max_steps is not None:
        config.max_steps = args.max_steps

    print("=" * 60)
    print("MiniMoss Overfit Training")
    print("=" * 60)
    print(f"  backbone: {config.backbone_name}")
    print(f"  local layers: {config.local_num_layers}")
    print(f"  local hidden: {config.local_hidden_size}")
    print(f"  n_codebooks: {config.n_codebooks}")
    print(f"  n_groups: {config.n_groups}")
    print(f"  device: {args.device}")
    print(f"  overfit_one_batch: {args.overfit_one_batch}")

    # Dataset
    dataset = MiniMossDataset(
        manifest_path=args.manifest,
        token_dir=args.token_dir,
    )
    print(f"\nDataset: {len(dataset)} utterances")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=not args.overfit_one_batch,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Model
    print("\nLoading model...")
    model = MiniMossModel(config)
    model.to(args.device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if config.use_amp and args.device == "cuda" else None

    # Get one batch for overfit mode
    if args.overfit_one_batch:
        text_batch, mask_batch, rvq_batch = next(iter(loader))
        text_batch = text_batch.to(args.device)
        mask_batch = mask_batch.to(args.device)
        rvq_batch = rvq_batch.to(args.device)
        print(f"  overfit batch: text={list(text_batch.shape)}, rvq={list(rvq_batch.shape)}")

    def _trainable_state_dict(model):
        """Return state_dict excluding the frozen backbone."""
        return {k: v for k, v in model.state_dict().items() if not k.startswith("_backbone.")}

    # Training loop
    model.train()
    step = 0
    total_steps = config.max_steps
    if args.overfit_one_batch:
        total_steps = 500  # enough to overfit one batch

    print(f"\nTraining for {total_steps} steps...\n")

    # Create iterator once, recreate on exhaustion
    if not args.overfit_one_batch:
        data_iter = iter(loader)

    for step in range(1, total_steps + 1):
        t_start = time.time()

        if args.overfit_one_batch:
            text_input, mask_input = text_batch, mask_batch
            rvq_input = rvq_batch
        else:
            try:
                text_input, mask_input, rvq_input = next(data_iter)
            except StopIteration:
                data_iter = iter(DataLoader(
                    dataset, batch_size=config.batch_size, shuffle=True,
                    collate_fn=collate_fn, drop_last=False,
                ))
                text_input, mask_input, rvq_input = next(data_iter)
            text_input = text_input.to(args.device)
            mask_input = mask_input.to(args.device)
            rvq_input = rvq_input.to(args.device)

        # Forward
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits, group_losses = model(text_input, rvq_input, text_attention_mask=mask_input)
        else:
            logits, group_losses = model(text_input, rvq_input, text_attention_mask=mask_input)

        # Weighted total loss
        weights = config.group_loss_weights
        total_loss = sum(w * gl for w, gl in zip(weights, group_losses))

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

        # Checkpoint
        if step % config.checkpoint_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": _trainable_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)
            print(f"  -> saved {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(args.output_dir, "final.pt")
    torch.save({
        "step": step,
        "model_state_dict": _trainable_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, final_path)
    print(f"\nFinal checkpoint: {final_path}")
    print("Done!")


if __name__ == "__main__":
    main()
