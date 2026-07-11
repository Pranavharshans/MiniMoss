#!/usr/bin/env python3
"""Validate official MOSS global states before training a grouped student."""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, AutoProcessor

from .dataset import load_manifest
from .utils import set_seed


DEFAULT_MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
DEFAULT_REVISION = "12aa734e4f11a7b3fdf4eb0ad2aa2029675ffc2e"


class CoarseStateProbe(nn.Module):
    """Small probe for predicting the first four RVQ channels from a global state."""

    def __init__(self, input_size: int, hidden_size: int, codebook_size: int = 1024):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, codebook_size) for _ in range(4)
        ])

    def forward(self, states: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.trunk(states)
        return [head(hidden) for head in self.heads]


def valid_audio_target_mask(targets: torch.Tensor, audio_pad_code: int) -> torch.Tensor:
    """Select positions whose 32 RVQ targets are real codec tokens."""
    audio = targets[..., 1:]
    return ((audio >= 0) & (audio < audio_pad_code)).all(dim=-1)


def probe_loss(logits: list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        F.cross_entropy(channel_logits, targets[:, channel])
        for channel, channel_logits in enumerate(logits)
    ]).mean()


def pack_teacher_forcing(processor, text: str, rvq: torch.Tensor, device: str):
    user = processor.build_user_message(text=text, language="en")
    assistant = processor.build_assistant_message(audio_codes_list=[rvq])
    packed = processor(
        [[user, assistant]], mode="continuation", n_vq=rvq.shape[1]
    )
    full_ids = packed["input_ids"].to(device)
    input_ids = full_ids[:, :-1].contiguous()
    targets = full_ids[:, 1:].contiguous()
    attention_mask = torch.ones(
        input_ids.shape[:2], dtype=torch.bool, device=input_ids.device
    )
    valid_mask = valid_audio_target_mask(
        targets, processor.model_config.audio_pad_code
    )
    return input_ids, attention_mask, targets, valid_mask


@torch.inference_mode()
def extract_split(model, processor, items, token_dir: Path, device: str):
    utterances = []
    for index, item in enumerate(items, start=1):
        token_path = token_dir / f"{item['id']}.pt"
        token_data = torch.load(token_path, map_location="cpu", weights_only=True)
        rvq = token_data["rvq"].long()
        input_ids, attention_mask, targets, valid_mask = pack_teacher_forcing(
            processor, item["text"], rvq, device
        )
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        states = outputs.last_hidden_state[valid_mask]
        audio_targets = targets[..., 1:][valid_mask]
        if states.shape[0] != rvq.shape[0]:
            raise RuntimeError(
                f"{item['id']}: extracted {states.shape[0]} states for "
                f"{rvq.shape[0]} codec frames"
            )
        utterances.append({
            "id": item["id"],
            "text": item["text"],
            "states": states.to(dtype=torch.float16, device="cpu"),
            "rvq": audio_targets.to(dtype=torch.int16, device="cpu"),
        })
        print(f"[{index:03d}/{len(items):03d}] {item['id']} | {states.shape[0]} frames")
    return utterances


def flatten_cache(utterances):
    states = torch.cat([item["states"] for item in utterances]).float()
    targets = torch.cat([item["rvq"] for item in utterances]).long()
    return states, targets


@torch.inference_mode()
def save_control_audio(model, processor, items, output_dir: Path, device: str, max_new_tokens: int):
    control_dir = output_dir / "official_control"
    control_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(processor, "audio_tokenizer") and hasattr(processor.audio_tokenizer, "to"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    for index, item in enumerate(items, start=1):
        user = processor.build_user_message(text=item["text"], language="en")
        batch = processor([[user]], mode="generation")
        outputs = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
        )
        message = processor.decode(outputs)[0]
        if not message.audio_codes_list:
            raise RuntimeError(f"Official MOSS produced no audio for {item['id']}")
        waveform = message.audio_codes_list[0].detach().float().cpu()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(
            str(control_dir / f"{index:02d}_official_moss.wav"),
            waveform,
            processor.model_config.sampling_rate,
        )
        (control_dir / f"{index:02d}.txt").write_text(
            f"id: {item['id']}\ntext: {item['text']}\n"
        )
        print(f"control [{index:02d}/{len(items):02d}] {item['id']}")


def unigram_metrics(train_targets: torch.Tensor, validation_targets: torch.Tensor):
    losses = []
    accuracies = []
    for channel in range(4):
        counts = torch.bincount(train_targets[:, channel], minlength=1024).float() + 1.0
        probabilities = counts / counts.sum()
        losses.append(-probabilities.log()[validation_targets[:, channel]].mean())
        accuracies.append(
            (validation_targets[:, channel] == probabilities.argmax()).float().mean()
        )
    return float(torch.stack(losses).mean()), float(torch.stack(accuracies).mean())


@torch.inference_mode()
def evaluate_probe(probe, states, targets, device: str, batch_size: int):
    losses = []
    correct = 0
    count = 0
    predictions = []
    for start in range(0, states.shape[0], batch_size):
        batch_states = states[start:start + batch_size].to(device)
        batch_targets = targets[start:start + batch_size, :4].to(device)
        logits = probe(batch_states)
        losses.append(probe_loss(logits, batch_targets) * batch_targets.shape[0])
        predicted = torch.stack([channel.argmax(dim=-1) for channel in logits], dim=-1)
        correct += int((predicted == batch_targets).sum())
        count += batch_targets.numel()
        predictions.append(predicted.cpu())
    return (
        float(torch.stack(losses).sum() / states.shape[0]),
        correct / count,
        torch.cat(predictions),
    )


def train_probe(args, train_cache, validation_cache):
    train_states, train_targets = flatten_cache(train_cache)
    validation_states, validation_targets = flatten_cache(validation_cache)
    input_size = train_states.shape[1]
    probe = CoarseStateProbe(input_size, args.probe_hidden_size).to(args.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr)
    generator = torch.Generator().manual_seed(args.seed)
    for step in range(1, args.probe_steps + 1):
        indices = torch.randint(
            train_states.shape[0], (args.probe_batch_size,), generator=generator
        )
        states = train_states[indices].to(args.device)
        targets = train_targets[indices, :4].to(args.device)
        loss = probe_loss(probe(states), targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(f"probe step={step} | loss={loss.item():.4f}")

    probe.eval()
    train_loss, train_accuracy, _ = evaluate_probe(
        probe, train_states, train_targets, args.device, args.eval_batch_size
    )
    validation_loss, validation_accuracy, validation_predictions = evaluate_probe(
        probe, validation_states, validation_targets, args.device, args.eval_batch_size
    )
    unigram_loss, unigram_accuracy = unigram_metrics(train_targets, validation_targets)
    metrics = {
        "train_frames": train_states.shape[0],
        "validation_frames": validation_states.shape[0],
        "train_coarse_loss": train_loss,
        "train_coarse_token_accuracy": train_accuracy,
        "validation_coarse_loss": validation_loss,
        "validation_coarse_token_accuracy": validation_accuracy,
        "unigram_validation_loss": unigram_loss,
        "unigram_validation_token_accuracy": unigram_accuracy,
        "uniform_loss": float(torch.log(torch.tensor(1024.0))),
    }
    return probe, metrics, validation_predictions


def save_hybrid_audio(args, processor, validation_cache, predictions):
    output_dir = Path(args.output_dir) / "hybrid_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    for index, item in enumerate(validation_cache[:args.audio_limit], start=1):
        frames = item["rvq"].shape[0]
        hybrid = item["rvq"].long().clone()
        hybrid[:, :4] = predictions[offset:offset + frames]
        offset += frames
        wav = processor.decode_audio_codes([hybrid.to(args.device)])[0].float().cpu()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        torchaudio.save(
            str(output_dir / f"{index:02d}_predicted_cb01-04.wav"),
            wav,
            processor.model_config.sampling_rate,
        )
        (output_dir / f"{index:02d}.txt").write_text(
            f"id: {item['id']}\ntext: {item['text']}\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--output-dir", default="evaluation/moss_teacher_probe")
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--validation-limit", type=int, default=20)
    parser.add_argument("--probe-hidden-size", type=int, default=768)
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--probe-lr", type=float, default=3e-4)
    parser.add_argument("--audio-limit", type=int, default=10)
    parser.add_argument("--control-limit", type=int, default=3)
    parser.add_argument("--control-max-new-tokens", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(
        args.train_limit,
        args.validation_limit,
        args.probe_steps,
        args.control_limit,
        args.control_max_new_tokens,
    ) <= 0:
        raise ValueError("Limits and probe steps must be positive")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"Loading official MOSS teacher: {args.model}")
    # The custom processor forwards all kwargs to the separate audio-tokenizer
    # repository, so passing the TTS revision here incorrectly applies that
    # commit hash to the codec as well. Pin the TTS model below; the processor
    # resolves the model's declared code and codec revisions independently.
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()
    train_items = load_manifest(args.train_manifest)[:args.train_limit]
    validation_items = load_manifest(args.validation_manifest)[:args.validation_limit]
    token_dir = Path(args.token_dir)
    print("Generating untouched official-MOSS controls...")
    save_control_audio(
        model,
        processor,
        validation_items[:args.control_limit],
        output_dir,
        args.device,
        args.control_max_new_tokens,
    )
    print("Extracting training global states...")
    train_cache = extract_split(model, processor, train_items, token_dir, args.device)
    print("Extracting validation global states...")
    validation_cache = extract_split(
        model, processor, validation_items, token_dir, args.device
    )
    torch.save(train_cache, output_dir / "train_states.pt")
    torch.save(validation_cache, output_dir / "validation_states.pt")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    probe, metrics, predictions = train_probe(args, train_cache, validation_cache)
    torch.save({
        "model_state_dict": probe.state_dict(),
        "input_size": probe.trunk[0].normalized_shape[0],
        "hidden_size": args.probe_hidden_size,
        "metrics": metrics,
    }, output_dir / "coarse_probe.pt")
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    save_hybrid_audio(args, processor, validation_cache, predictions)
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
