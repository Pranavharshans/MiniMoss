#!/usr/bin/env python3
"""Train the coarse-first grouped local student from cached MOSS states."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torchaudio

from .codec import AudioCodec
from .grouped_student import GroupedLocalStudent, GroupedStudentConfig
from .utils import set_seed


def load_cache(path: str):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if not cache:
        raise ValueError(f"Empty state cache: {path}")
    return cache


def flatten_cache(cache):
    states = torch.cat([item["states"] for item in cache]).float()
    targets = torch.cat([item["rvq"] for item in cache]).long()
    positions = torch.cat([
        torch.arange(item["rvq"].shape[0]).float() / max(item["rvq"].shape[0] - 1, 1)
        for item in cache
    ])
    return states, targets, positions


@torch.inference_mode()
def validation_loss(model, states, targets, device: str, batch_size: int):
    model.eval()
    total = 0.0
    count = 0
    channel_totals = torch.zeros(model.config.n_codebooks)
    for start in range(0, states.shape[0], batch_size):
        batch_states = states[start:start + batch_size].to(device)
        batch_targets = targets[start:start + batch_size].to(device)
        loss, channel_losses = model.loss(batch_states, batch_targets)
        size = batch_states.shape[0]
        total += float(loss) * size
        channel_totals += torch.tensor([float(value) for value in channel_losses]) * size
        count += size
    return total / count, (channel_totals / count).tolist()


def group_accuracies(predictions, targets, groups):
    matches = predictions.eq(targets)
    return [
        float(matches[:, list(group)].float().mean()) for group in groups
    ]


@torch.inference_mode()
def final_metrics(model, states, targets, positions, device: str, batch_size: int):
    model.eval()
    teacher_parts = []
    free_parts = []
    for start in range(0, states.shape[0], batch_size):
        batch_states = states[start:start + batch_size].to(device)
        batch_targets = targets[start:start + batch_size].to(device)
        teacher_parts.append(model.predict(batch_states, teacher_targets=batch_targets).cpu())
        free_parts.append(model.predict(batch_states).cpu())
    teacher = torch.cat(teacher_parts)
    free = torch.cat(free_parts)
    coarse_matches = free[:, :4].eq(targets[:, :4]).float()
    position_accuracy = []
    for lower in (0.0, 0.25, 0.5, 0.75):
        mask = (positions >= lower) & (positions <= lower + 0.25 if lower == 0.75 else positions < lower + 0.25)
        position_accuracy.append(float(coarse_matches[mask].mean()))
    return {
        "teacher_token_accuracy": float(teacher.eq(targets).float().mean()),
        "free_token_accuracy": float(free.eq(targets).float().mean()),
        "teacher_group_accuracy": group_accuracies(teacher, targets, model.config.groups),
        "free_group_accuracy": group_accuracies(free, targets, model.config.groups),
        "free_coarse_accuracy_by_position_quartile": position_accuracy,
    }


@torch.inference_mode()
def save_audio_examples(model, cache, output_dir: Path, codec_name: str, device: str, limit: int):
    codec = AudioCodec(model_name=codec_name, n_quantizers=32)
    if hasattr(codec.model, "to"):
        codec.model.to(device)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for index, item in enumerate(cache[:limit], start=1):
        states = item["states"].float().to(device)
        targets = item["rvq"].long().to(device)
        teacher = model.predict(states, teacher_targets=targets)
        free = model.predict(states)
        for name, codes in (("ground_truth", targets), ("teacher", teacher), ("free", free)):
            waveform = codec.decode(codes.transpose(0, 1).unsqueeze(0))[0].cpu()
            torchaudio.save(
                str(audio_dir / f"{index:02d}_{name}.wav"),
                waveform,
                codec.sample_rate,
            )
        (audio_dir / f"{index:02d}.txt").write_text(
            f"id: {item['id']}\ntext: {item['text']}\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", default="checkpoints/moss_grouped_hybrid11")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--local-hidden-size", type=int, default=512)
    parser.add_argument("--local-layers", type=int, default=4)
    parser.add_argument("--local-ffn-size", type=int, default=1024)
    parser.add_argument("--local-dropout", type=float, default=0.1)
    parser.add_argument("--audio-limit", type=int, default=10)
    parser.add_argument("--codec", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.max_steps, args.batch_size, args.validate_every, args.patience) <= 0:
        raise ValueError("Training steps, batch size, validation interval, and patience must be positive")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cache = load_cache(args.train_cache)
    validation_cache = load_cache(args.validation_cache)
    train_states, train_targets, _ = flatten_cache(train_cache)
    validation_states, validation_targets, validation_positions = flatten_cache(validation_cache)
    config = GroupedStudentConfig(
        global_hidden_size=train_states.shape[1],
        local_hidden_size=args.local_hidden_size,
        local_num_layers=args.local_layers,
        local_ffn_hidden_size=args.local_ffn_size,
        local_dropout=args.local_dropout,
    )
    model = GroupedLocalStudent(config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda") if args.device.startswith("cuda") else None
    generator = torch.Generator().manual_seed(args.seed)
    trainable = sum(parameter.numel() for parameter in model.parameters())
    print(f"train frames: {train_states.shape[0]} | validation frames: {validation_states.shape[0]}")
    print(f"student parameters: {trainable:,} | local steps/frame: {len(config.groups)}")
    best_loss = float("inf")
    best_step = 0
    best_state = None
    checks_without_improvement = 0
    for step in range(1, args.max_steps + 1):
        started = time.time()
        indices = torch.randint(
            train_states.shape[0], (args.batch_size,), generator=generator
        )
        states = train_states[indices].to(args.device)
        targets = train_targets[indices].to(args.device)
        model.train()
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss, _ = model.loss(states, targets, args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, _ = model.loss(states, targets, args.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step == 1 or step % 20 == 0:
            print(f"step={step} | loss={loss.item():.4f} | t={time.time() - started:.2f}s")
        if step % args.validate_every == 0:
            val_loss, channel_losses = validation_loss(
                model, validation_states, validation_targets, args.device, args.eval_batch_size
            )
            print(
                f"validation step={step} | loss={val_loss:.4f} | "
                f"cb1={channel_losses[0]:.4f} | cb4={channel_losses[3]:.4f} | "
                f"cb32={channel_losses[-1]:.4f}"
            )
            if val_loss < best_loss - args.min_delta:
                best_loss = val_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                checks_without_improvement = 0
                torch.save({
                    "model_state_dict": best_state,
                    "config": config.to_dict(),
                    "step": step,
                    "validation_loss": val_loss,
                }, output_dir / "best.pt")
                print(f"  -> new best checkpoint at step {step}")
            else:
                checks_without_improvement += 1
                if checks_without_improvement >= args.patience:
                    print(f"early stopping at step={step}; best_step={best_step}")
                    break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    model.load_state_dict(best_state)
    metrics = {
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "student_parameters": trainable,
        "local_steps_per_frame": len(config.groups),
        **final_metrics(
            model,
            validation_states,
            validation_targets,
            validation_positions,
            args.device,
            args.eval_batch_size,
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    save_audio_examples(
        model, validation_cache, output_dir, args.codec, args.device, args.audio_limit
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
