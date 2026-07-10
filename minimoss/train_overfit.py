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


def context_dropout_for_step(
    step: int,
    warmup_steps: int,
    decay_steps: int,
    start_probability: float,
    end_probability: float,
) -> float:
    """Hold dropout high, then linearly decay it to the inference-adjacent floor."""
    if step <= warmup_steps:
        return start_probability
    if decay_steps == 0 or step >= warmup_steps + decay_steps:
        return end_probability
    progress = (step - warmup_steps) / decay_steps
    return start_probability + progress * (end_probability - start_probability)


def curriculum_phase_and_weights(step: int, phase_a_end: int, phase_b_end: int):
    """Return the V4 phase and loss weights for eight grouped RVQ steps."""
    if step <= phase_a_end:
        return "A", (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if step <= phase_b_end:
        return "B", (4.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    return "C", (4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def per_codebook_losses(logits, audio_codes, audio_frame_mask, codebook_size):
    losses = []
    for codebook, codebook_logits in enumerate(logits):
        targets = audio_codes[:, :, codebook].masked_fill(
            ~audio_frame_mask.bool(), -100
        )
        losses.append(torch.nn.functional.cross_entropy(
            codebook_logits.reshape(-1, codebook_size),
            targets.reshape(-1),
            ignore_index=-100,
        ))
    return losses


def trainable_state_dict(model):
    """Return a checkpoint without duplicating the frozen Hugging Face backbone."""
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("_backbone.") or key in trainable_names
    }


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
def validate(
    model,
    loader,
    device,
    weights,
    max_batches=None,
    shuffle_text=False,
    audio_context_dropout_prob=0.0,
):
    model.eval()
    total_loss = 0.0
    total_groups = [0.0] * len(weights)
    total_codebooks = None
    batches = 0
    for text, text_mask, rvq, audio_mask in loader:
        if shuffle_text:
            permutation = torch.arange(text.shape[0] - 1, -1, -1)
            text = text[permutation]
            text_mask = text_mask[permutation]
        text = text.to(device)
        text_mask = text_mask.to(device)
        rvq = rvq.to(device)
        audio_mask = audio_mask.to(device)
        logits, group_losses = model(
            text,
            rvq,
            text_attention_mask=text_mask,
            audio_frame_mask=audio_mask,
            audio_context_dropout_prob=audio_context_dropout_prob,
        )
        total_loss += weighted_loss(group_losses, weights).item()
        for index, loss in enumerate(group_losses):
            total_groups[index] += loss.item()
        codebook_losses = per_codebook_losses(
            logits, rvq, audio_mask, model.config.codebook_size
        )
        if total_codebooks is None:
            total_codebooks = [0.0] * len(codebook_losses)
        for index, loss in enumerate(codebook_losses):
            total_codebooks[index] += loss.item()
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    model.train()
    if batches == 0:
        raise ValueError("Validation loader produced no batches")
    return (
        total_loss / batches,
        [loss / batches for loss in total_groups],
        [loss / batches for loss in total_codebooks],
    )


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
    parser.add_argument("--qwen-lora", action="store_true")
    parser.add_argument("--qwen-lora-rank", type=int, default=8)
    parser.add_argument("--qwen-lora-alpha", type=float, default=16.0)
    parser.add_argument("--nonlinear-frame-conditioner", action="store_true")
    parser.add_argument("--frame-position-embedding", action="store_true")
    parser.add_argument("--context-dropout-warmup-steps", type=int, default=1000)
    parser.add_argument("--context-dropout-decay-steps", type=int, default=3000)
    parser.add_argument("--context-dropout-start", type=float, default=0.0)
    parser.add_argument("--context-dropout-end", type=float, default=0.0)
    parser.add_argument("--text-diagnostics", action="store_true",
                        help="Also validate with text reversed across each batch")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Stop after this many validation checks without improvement")
    parser.add_argument("--early-stopping-start-step", type=int, default=0)
    parser.add_argument("--group-curriculum", action="store_true")
    parser.add_argument("--phase-a-end", type=int, default=750)
    parser.add_argument("--phase-b-end", type=int, default=1500)
    parser.add_argument("--phase-gate-min-improvement", type=float, default=0.05)
    parser.add_argument("--phase-gate-min-text-delta", type=float, default=0.005)
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
        use_qwen_lora=args.qwen_lora,
        qwen_lora_rank=args.qwen_lora_rank,
        qwen_lora_alpha=args.qwen_lora_alpha,
        use_nonlinear_frame_conditioner=args.nonlinear_frame_conditioner,
        use_frame_position_embedding=args.frame_position_embedding,
    )
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.validate_every <= 0:
        raise ValueError("--validate-every must be positive")
    if args.validation_batches is not None and args.validation_batches <= 0:
        raise ValueError("--validation-batches must be positive")
    if args.early_stopping_patience is not None and args.early_stopping_patience <= 0:
        raise ValueError("--early-stopping-patience must be positive")
    if args.context_dropout_warmup_steps < 0 or args.context_dropout_decay_steps < 0:
        raise ValueError("Context-dropout schedule steps cannot be negative")
    if not 0.0 <= args.context_dropout_end <= args.context_dropout_start <= 1.0:
        raise ValueError("Context dropout must satisfy 0 <= end <= start <= 1")
    if args.context_dropout_start > 0 and not args.frame_position_embedding:
        raise ValueError("Context dropout requires --frame-position-embedding")
    if args.context_dropout_start > 0 and not args.nonlinear_frame_conditioner:
        raise ValueError("Context dropout requires --nonlinear-frame-conditioner")
    if args.group_curriculum:
        if not 0 < args.phase_a_end < args.phase_b_end < config.max_steps:
            raise ValueError("Group curriculum requires 0 < phase A < phase B < max steps")
        if args.phase_a_end % args.validate_every or args.phase_b_end % args.validate_every:
            raise ValueError("Phase boundaries must be divisible by --validate-every")
        if not args.text_diagnostics:
            raise ValueError("Group curriculum requires --text-diagnostics")

    print("=" * 60)
    print("MiniMoss Overfit Training")
    print("=" * 60)
    print(f"  backbone: {config.backbone_name}")
    print(f"  audio tokenizer name: {config.codec_name}")
    print(f"  local layers: {config.local_num_layers}")
    print(f"  local hidden: {config.local_hidden_size}")
    print(f"  n_codebooks: {config.n_codebooks}")
    print(f"  n_groups: {config.n_groups}")
    qwen_status = "frozen base + trainable LoRA" if config.use_qwen_lora else "frozen"
    print(f"  qwen: {qwen_status}")
    print("  trainable: frame conditioner + projection + RVQ embeddings + local decoder + heads")
    print(f"  device: {args.device}")
    print(f"  overfit_one_batch: {args.overfit_one_batch}")
    print(f"  validation_manifest: {args.validation_manifest}")
    print(f"  resume: {args.resume}")
    print(f"  qwen_lora: {config.use_qwen_lora} (rank={config.qwen_lora_rank})")
    print(f"  nonlinear_frame_conditioner: {config.use_nonlinear_frame_conditioner}")
    print(f"  frame_position_embedding: {config.use_frame_position_embedding}")
    print(
        f"  context_dropout: {args.context_dropout_start:.2f} -> "
        f"{args.context_dropout_end:.2f} after warmup={args.context_dropout_warmup_steps}, "
        f"decay={args.context_dropout_decay_steps}"
    )
    print(
        f"  group_curriculum: {args.group_curriculum} "
        f"(A<= {args.phase_a_end}, B<= {args.phase_b_end})"
    )

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
    validation_checks_without_improvement = 0
    phase_best_losses = {"A": float("inf"), "B": float("inf"), "C": float("inf")}
    phase_initial_losses = {}
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
        context_dropout = context_dropout_for_step(
            step,
            args.context_dropout_warmup_steps,
            args.context_dropout_decay_steps,
            args.context_dropout_start,
            args.context_dropout_end,
        )
        if args.group_curriculum:
            phase, weights = curriculum_phase_and_weights(
                step, args.phase_a_end, args.phase_b_end
            )
        else:
            phase, weights = "standard", config.group_loss_weights

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
                    audio_context_dropout_prob=context_dropout,
                )
        else:
            logits, group_losses = model(
                text_input,
                rvq_input,
                text_attention_mask=mask_input,
                audio_frame_mask=audio_mask_input,
                audio_context_dropout_prob=context_dropout,
            )

        # Weighted total loss
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
            metrics = format_metrics(
                step, total_loss.item(), g_losses, config.learning_rate, step_time
            )
            print(
                f"{metrics} | phase={phase} | context_drop={context_dropout:.3f} | "
                f"weights={','.join(f'{weight:g}' for weight in weights)}"
            )

        if validation_loader is not None and step % args.validate_every == 0:
            validation_context_dropout = (
                1.0 if args.group_curriculum and phase in ("A", "B") else 0.0
            )
            validation_loss, validation_groups, validation_codebooks = validate(
                model,
                validation_loader,
                args.device,
                weights,
                args.validation_batches,
                audio_context_dropout_prob=validation_context_dropout,
            )
            phase_initial_losses.setdefault(phase, validation_loss)
            group_text = " | ".join(
                f"val_g{index + 1}={loss:.4f}"
                for index, loss in enumerate(validation_groups)
            )
            print(
                f"validation step={step} | phase={phase} | "
                f"val_loss={validation_loss:.4f} | {group_text}"
            )
            codebook_text = " | ".join(
                f"cb{index + 1}={loss:.4f}"
                for index, loss in enumerate(validation_codebooks)
            )
            print(f"codebook validation step={step} | {codebook_text}")

            if args.group_curriculum and validation_loss < phase_best_losses[phase]:
                phase_best_losses[phase] = validation_loss
                phase_path = Path(args.output_dir) / f"best_phase_{phase.lower()}.pt"
                save_checkpoint(
                    phase_path, step, model, optimizer, scaler, config, validation_loss
                )
                print(f"  -> new best phase {phase} checkpoint: {phase_path}")

            tracks_final_metric = not args.group_curriculum or phase == "C"
            if tracks_final_metric and validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                validation_checks_without_improvement = 0
                best_path = Path(args.output_dir) / "best_validation.pt"
                save_checkpoint(
                    best_path, step, model, optimizer, scaler, config, best_validation_loss
                )
                print(f"  -> new best validation checkpoint: {best_path}")
            elif step >= args.early_stopping_start_step:
                validation_checks_without_improvement += 1
            if args.text_diagnostics:
                shuffled_loss, shuffled_groups, shuffled_codebooks = validate(
                    model,
                    validation_loader,
                    args.device,
                    weights,
                    args.validation_batches,
                    shuffle_text=True,
                    audio_context_dropout_prob=validation_context_dropout,
                )
                print(
                    f"text diagnostic step={step} | normal={validation_loss:.4f} | "
                    f"shuffled={shuffled_loss:.4f} | delta={shuffled_loss - validation_loss:.4f} | "
                    f"g1_delta={shuffled_groups[0] - validation_groups[0]:.4f}"
                )
                coarse_deltas = " | ".join(
                    f"cb{index + 1}_delta={shuffled_codebooks[index] - validation_codebooks[index]:.4f}"
                    for index in range(min(4, len(validation_codebooks)))
                )
                print(f"coarse text diagnostic step={step} | {coarse_deltas}")
                alternate_context_dropout = 1.0 - validation_context_dropout
                alternate_context_loss, _, _ = validate(
                    model,
                    validation_loader,
                    args.device,
                    weights,
                    args.validation_batches,
                    audio_context_dropout_prob=alternate_context_dropout,
                )
                print(
                    f"context diagnostic step={step} | "
                    f"primary_drop={validation_context_dropout:.1f} "
                    f"primary={validation_loss:.4f} | "
                    f"alternate_drop={alternate_context_dropout:.1f} "
                    f"alternate={alternate_context_loss:.4f} | "
                    f"delta={alternate_context_loss - validation_loss:.4f}"
                )

                if args.group_curriculum and step in (args.phase_a_end, args.phase_b_end):
                    improvement = phase_initial_losses[phase] - validation_loss
                    group1_delta = shuffled_groups[0] - validation_groups[0]
                    passed = (
                        improvement >= args.phase_gate_min_improvement
                        and group1_delta >= args.phase_gate_min_text_delta
                    )
                    print(
                        f"PHASE_{phase}_{'PASS' if passed else 'FAIL'} | "
                        f"improvement={improvement:.4f} "
                        f"(required={args.phase_gate_min_improvement:.4f}) | "
                        f"g1_text_delta={group1_delta:.4f} "
                        f"(required={args.phase_gate_min_text_delta:.4f})"
                    )
                    if not passed:
                        print(f"Stopping because phase {phase} alignment gate failed.")
                        break
            if (
                args.early_stopping_patience is not None
                and step >= args.early_stopping_start_step
                and validation_checks_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"Early stopping after {validation_checks_without_improvement} "
                    "validation checks without improvement."
                )
                break

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
