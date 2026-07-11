#!/usr/bin/env python3
"""Train a configured grouped local student from cached MOSS states."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torchaudio

from .codec import AudioCodec
from .grouped_student import (
    EXPERIMENT_SPECS,
    GroupedLocalStudent,
    GroupedStudentConfig,
    get_experiment_spec,
)
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
    teacher_indices = None
    teacher_values = None
    teacher_tokens = None
    if "teacher_topk_indices" in cache[0]:
        teacher_indices = torch.cat([item["teacher_topk_indices"] for item in cache]).long()
        teacher_values = torch.cat([item["teacher_topk_values"] for item in cache]).float()
        teacher_tokens = torch.cat([item["teacher_tokens"] for item in cache]).long()
    return states, targets, positions, teacher_indices, teacher_values, teacher_tokens


@torch.inference_mode()
def validation_loss(
    model,
    states,
    targets,
    teacher_indices,
    teacher_values,
    device: str,
    batch_size: int,
    ground_truth_weight: float,
    distillation_weight: float,
    temperature: float,
):
    model.eval()
    total = 0.0
    count = 0
    channel_totals = torch.zeros(model.config.n_codebooks)
    ground_truth_total = 0.0
    distillation_total = 0.0
    for start in range(0, states.shape[0], batch_size):
        batch_states = states[start:start + batch_size].to(device)
        batch_targets = targets[start:start + batch_size].to(device)
        batch_indices = teacher_indices[start:start + batch_size].to(device)
        batch_values = teacher_values[start:start + batch_size].to(device)
        loss, ground_truth_loss, teacher_loss, channel_losses = model.combined_loss(
            batch_states,
            batch_targets,
            batch_indices,
            batch_values,
            ground_truth_weight,
            distillation_weight,
            temperature,
        )
        size = batch_states.shape[0]
        total += float(loss) * size
        ground_truth_total += float(ground_truth_loss) * size
        distillation_total += float(teacher_loss) * size
        channel_totals += torch.tensor([float(value) for value in channel_losses]) * size
        count += size
    return (
        total / count,
        ground_truth_total / count,
        distillation_total / count,
        (channel_totals / count).tolist(),
    )


@torch.inference_mode()
def validation_rollout_loss(
    model,
    states,
    targets,
    device: str,
    batch_size: int,
    teacher_forcing_probability: float,
    label_smoothing: float,
):
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, states.shape[0], batch_size):
        batch_states = states[start:start + batch_size].to(device)
        batch_targets = targets[start:start + batch_size].to(device)
        loss, _ = model.rollout_loss(
            batch_states,
            batch_targets,
            teacher_forcing_probability=teacher_forcing_probability,
            label_smoothing=label_smoothing,
        )
        size = batch_states.shape[0]
        total += float(loss) * size
        count += size
    return total / count


def group_accuracies(predictions, targets, groups):
    matches = predictions.eq(targets)
    return [
        float(matches[:, list(group)].float().mean()) for group in groups
    ]


@torch.inference_mode()
def final_metrics(
    model, states, targets, positions, teacher_tokens, device: str, batch_size: int
):
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
    metrics = {
        "teacher_token_accuracy": float(teacher.eq(targets).float().mean()),
        "free_token_accuracy": float(free.eq(targets).float().mean()),
        "teacher_group_accuracy": group_accuracies(teacher, targets, model.config.groups),
        "free_group_accuracy": group_accuracies(free, targets, model.config.groups),
        "free_coarse_accuracy_by_position_quartile": position_accuracy,
    }
    if teacher_tokens is not None:
        metrics["teacher_original_token_accuracy"] = float(
            teacher_tokens.eq(targets).float().mean()
        )
        metrics["student_teacher_agreement"] = float(
            teacher.eq(teacher_tokens).float().mean()
        )
        metrics["student_free_teacher_agreement"] = float(
            free.eq(teacher_tokens).float().mean()
        )
    return metrics


@torch.inference_mode()
def save_audio_examples(
    model,
    cache,
    output_dir: Path,
    codec_name: str,
    device: str,
    limit: int,
    temperature: float,
    top_k: int,
):
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
        sampled = model.predict(states, temperature=temperature, top_k=top_k)
        outputs = [
            ("ground_truth", targets),
            ("original_teacher_greedy", item["teacher_tokens"].long().to(device)),
            ("student_teacher", teacher),
            ("student_free", free),
            ("student_sampled", sampled),
        ]
        if "teacher_sampled_tokens" in item:
            outputs.insert(
                2,
                (
                    "original_teacher_topk_sampled",
                    item["teacher_sampled_tokens"].long().to(device),
                ),
            )
        for name, codes in outputs:
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
    parser.add_argument(
        "--variant",
        choices=tuple(EXPERIMENT_SPECS),
        default="baseline11",
        help="Controlled experiment preset; explicit numeric flags override its defaults",
    )
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", default="checkpoints/moss_grouped_hybrid11")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--ground-truth-weight", type=float, default=None)
    parser.add_argument("--distillation-weight", type=float, default=None)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--rollout-weight", type=float, default=None)
    parser.add_argument("--rollout-teacher-forcing-start", type=float, default=None)
    parser.add_argument("--rollout-teacher-forcing-end", type=float, default=None)
    parser.add_argument("--rollout-ramp-steps", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--local-hidden-size", type=int, default=None)
    parser.add_argument("--local-layers", type=int, default=None)
    parser.add_argument("--local-ffn-size", type=int, default=None)
    parser.add_argument("--local-dropout", type=float, default=0.1)
    parser.add_argument("--audio-limit", type=int, default=10)
    parser.add_argument("--audio-temperature", type=float, default=1.0)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--codec", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.max_steps, args.batch_size, args.validate_every, args.patience) <= 0:
        raise ValueError("Training steps, batch size, validation interval, and patience must be positive")
    spec = get_experiment_spec(args.variant)
    ground_truth_weight = (
        spec["ground_truth_weight"]
        if args.ground_truth_weight is None
        else args.ground_truth_weight
    )
    distillation_weight = (
        spec["distillation_weight"]
        if args.distillation_weight is None
        else args.distillation_weight
    )
    label_smoothing = (
        spec["label_smoothing"]
        if args.label_smoothing is None
        else args.label_smoothing
    )
    local_hidden_size = (
        spec["local_hidden_size"]
        if args.local_hidden_size is None
        else args.local_hidden_size
    )
    local_layers = spec["local_num_layers"] if args.local_layers is None else args.local_layers
    local_ffn_size = (
        spec["local_ffn_hidden_size"]
        if args.local_ffn_size is None
        else args.local_ffn_size
    )
    rollout_weight = spec.get("rollout_weight", 0.0) if args.rollout_weight is None else args.rollout_weight
    rollout_teacher_forcing_start = spec.get(
        "rollout_teacher_forcing_start", 1.0
    ) if args.rollout_teacher_forcing_start is None else args.rollout_teacher_forcing_start
    rollout_teacher_forcing_end = spec.get(
        "rollout_teacher_forcing_end", 0.0
    ) if args.rollout_teacher_forcing_end is None else args.rollout_teacher_forcing_end
    rollout_ramp_steps = spec.get("rollout_ramp_steps", 0) if args.rollout_ramp_steps is None else args.rollout_ramp_steps
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cache = load_cache(args.train_cache)
    validation_cache = load_cache(args.validation_cache)
    (
        train_states,
        train_targets,
        _,
        train_teacher_indices,
        train_teacher_values,
        _,
    ) = flatten_cache(train_cache)
    (
        validation_states,
        validation_targets,
        validation_positions,
        validation_teacher_indices,
        validation_teacher_values,
        validation_teacher_tokens,
    ) = flatten_cache(validation_cache)
    if train_teacher_indices is None or validation_teacher_indices is None:
        raise ValueError(
            "Distillation caches must contain teacher_topk_indices and teacher_topk_values"
        )
    if ground_truth_weight < 0 or distillation_weight < 0:
        raise ValueError("Loss weights cannot be negative")
    if ground_truth_weight + distillation_weight <= 0:
        raise ValueError("At least one loss weight must be positive")
    if not 0.0 <= rollout_weight <= 1.0:
        raise ValueError("rollout_weight must be in [0, 1]")
    if not 0.0 <= rollout_teacher_forcing_start <= 1.0 or not 0.0 <= rollout_teacher_forcing_end <= 1.0:
        raise ValueError("rollout teacher-forcing probabilities must be in [0, 1]")
    if rollout_ramp_steps < 0:
        raise ValueError("rollout_ramp_steps cannot be negative")
    config = GroupedStudentConfig(
        global_hidden_size=train_states.shape[1],
        local_hidden_size=local_hidden_size,
        local_num_layers=local_layers,
        local_ffn_hidden_size=local_ffn_size,
        local_dropout=args.local_dropout,
        groups=spec["groups"],
    )
    model = GroupedLocalStudent(config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda") if args.device.startswith("cuda") else None
    generator = torch.Generator().manual_seed(args.seed)
    trainable = sum(parameter.numel() for parameter in model.parameters())
    print(f"experiment: {args.variant} | {spec['description']}")
    print(f"train frames: {train_states.shape[0]} | validation frames: {validation_states.shape[0]}")
    print(
        f"student parameters: {trainable:,} | local steps/frame: {len(config.groups)} | "
        f"gt_weight={ground_truth_weight} | kd_weight={distillation_weight} | "
        f"rollout_weight={rollout_weight}"
    )
    best_loss = float("inf")
    best_step = 0
    best_state = None
    best_validation_loss = None
    best_validation_rollout_loss = None
    best_validation_ground_truth_loss = None
    best_validation_distillation_loss = None
    best_validation_channel_losses = None
    checks_without_improvement = 0
    for step in range(1, args.max_steps + 1):
        started = time.time()
        indices = torch.randint(
            train_states.shape[0], (args.batch_size,), generator=generator
        )
        states = train_states[indices].to(args.device)
        targets = train_targets[indices].to(args.device)
        teacher_indices = train_teacher_indices[indices].to(args.device)
        teacher_values = train_teacher_values[indices].to(args.device)
        if rollout_ramp_steps == 0:
            rollout_probability = rollout_teacher_forcing_end
        else:
            progress = min(step / rollout_ramp_steps, 1.0)
            rollout_probability = (
                rollout_teacher_forcing_start
                + progress
                * (rollout_teacher_forcing_end - rollout_teacher_forcing_start)
            )
        model.train()
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                base_loss, ground_truth_loss, teacher_loss, _ = model.combined_loss(
                    states,
                    targets,
                    teacher_indices,
                    teacher_values,
                    ground_truth_weight,
                    distillation_weight,
                    args.distillation_temperature,
                    label_smoothing,
                )
                if rollout_weight > 0.0:
                    rollout_training_loss, _ = model.rollout_loss(
                        states,
                        targets,
                        teacher_forcing_probability=rollout_probability,
                        label_smoothing=label_smoothing,
                    )
                    loss = (1.0 - rollout_weight) * base_loss + rollout_weight * rollout_training_loss
                else:
                    rollout_training_loss = base_loss.detach() * 0.0
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            base_loss, ground_truth_loss, teacher_loss, _ = model.combined_loss(
                states,
                targets,
                teacher_indices,
                teacher_values,
                ground_truth_weight,
                distillation_weight,
                args.distillation_temperature,
                label_smoothing,
            )
            if rollout_weight > 0.0:
                rollout_training_loss, _ = model.rollout_loss(
                    states,
                    targets,
                    teacher_forcing_probability=rollout_probability,
                    label_smoothing=label_smoothing,
                )
                loss = (1.0 - rollout_weight) * base_loss + rollout_weight * rollout_training_loss
            else:
                rollout_training_loss = base_loss.detach() * 0.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step == 1 or step % 20 == 0:
            print(
                f"step={step} | loss={loss.item():.4f} | "
                f"gt={ground_truth_loss.item():.4f} | "
                f"kd={teacher_loss.item():.4f} | "
                f"rollout={rollout_training_loss.item():.4f} | "
                f"rollout_p={rollout_probability:.3f} | t={time.time() - started:.2f}s"
            )
        if step % args.validate_every == 0:
            val_loss, val_gt, val_kd, channel_losses = validation_loss(
                model,
                validation_states,
                validation_targets,
                validation_teacher_indices,
                validation_teacher_values,
                args.device,
                args.eval_batch_size,
                ground_truth_weight,
                distillation_weight,
                args.distillation_temperature,
            )
            print(
                f"validation step={step} | loss={val_loss:.4f} | "
                f"gt={val_gt:.4f} | kd={val_kd:.4f} | "
                f"cb1={channel_losses[0]:.4f} | cb4={channel_losses[3]:.4f} | "
                f"cb32={channel_losses[-1]:.4f}"
            )
            val_rollout = None
            selection_loss = val_loss
            if rollout_weight > 0.0:
                val_rollout = validation_rollout_loss(
                    model,
                    validation_states,
                    validation_targets,
                    args.device,
                    args.eval_batch_size,
                    teacher_forcing_probability=0.0,
                    label_smoothing=label_smoothing,
                )
                selection_loss = (1.0 - rollout_weight) * val_loss + rollout_weight * val_rollout
                print(
                    f"validation rollout step={step} | loss={val_rollout:.4f} | "
                    f"selection={selection_loss:.4f}"
                )
            if selection_loss < best_loss - args.min_delta:
                best_loss = selection_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                best_validation_loss = val_loss
                best_validation_rollout_loss = val_rollout
                best_validation_ground_truth_loss = val_gt
                best_validation_distillation_loss = val_kd
                best_validation_channel_losses = channel_losses
                checks_without_improvement = 0
                torch.save({
                    "model_state_dict": best_state,
                    "config": config.to_dict(),
                    "variant": args.variant,
                    "step": step,
                    "validation_loss": val_loss,
                    "validation_rollout_loss": val_rollout,
                    "selection_loss": selection_loss,
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
        "experiment": args.variant,
        "description": spec["description"],
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
        "best_validation_selection_loss": best_loss,
        "best_validation_rollout_loss": best_validation_rollout_loss,
        "best_validation_ground_truth_loss": best_validation_ground_truth_loss,
        "best_validation_distillation_loss": best_validation_distillation_loss,
        "best_validation_channel_losses": best_validation_channel_losses,
        "student_parameters": trainable,
        "local_steps_per_frame": len(config.groups),
        "groups": [list(group) for group in config.groups],
        "ground_truth_weight": ground_truth_weight,
        "distillation_weight": distillation_weight,
        "distillation_temperature": args.distillation_temperature,
        "rollout_weight": rollout_weight,
        "rollout_teacher_forcing_start": rollout_teacher_forcing_start,
        "rollout_teacher_forcing_end": rollout_teacher_forcing_end,
        "rollout_ramp_steps": rollout_ramp_steps,
        "label_smoothing": label_smoothing,
        "local_hidden_size": local_hidden_size,
        "local_layers": local_layers,
        "local_ffn_size": local_ffn_size,
        **final_metrics(
            model,
            validation_states,
            validation_targets,
            validation_positions,
            validation_teacher_tokens,
            args.device,
            args.eval_batch_size,
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if not args.skip_audio and args.audio_limit > 0:
        save_audio_examples(
            model,
            validation_cache,
            output_dir,
            args.codec,
            args.device,
            args.audio_limit,
            args.audio_temperature,
            args.audio_top_k,
        )
    else:
        print("Audio generation skipped")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
