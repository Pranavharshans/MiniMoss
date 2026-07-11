#!/usr/bin/env python3
"""Batch evaluation with numbered ground-truth, teacher, and free audio."""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from transformers import AutoTokenizer

from .codec import AudioCodec
from .dataset import load_manifest
from .generate import generate, teacher_forced_generate
from .model import MiniMossModel
from .utils import set_seed


def numbered_prefix(index: int, total: int) -> str:
    width = max(2, len(str(total)))
    return f"{index:0{width}d}"


def token_metrics(prediction: torch.Tensor, target: torch.Tensor, group_size: int = 4):
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction={prediction.shape}, target={target.shape}")
    matches = prediction.cpu().eq(target.cpu())
    n_groups = target.shape[1] // group_size
    return {
        "token_accuracy": matches.float().mean().item(),
        "frame_accuracy": matches.all(dim=1).float().mean().item(),
        "group_accuracy": [
            matches[:, group * group_size:(group + 1) * group_size].float().mean().item()
            for group in range(n_groups)
        ],
    }


def save_codes(codec: AudioCodec, codes: torch.Tensor, path: Path, device: str):
    codes_for_decode = codes.transpose(0, 1).unsqueeze(0).to(device)
    wav = codec.decode(codes_for_decode)[0].cpu()
    torchaudio.save(str(path), wav, codec.sample_rate)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniMoss on a manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-audio-context", action="store_true",
                        help="Use the learned null previous-frame context")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = load_manifest(args.manifest)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        items = items[:args.limit]
    if not items:
        raise ValueError(f"Manifest is empty: {args.manifest}")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = MiniMossModel(config).to(args.device)
    _ = model.backbone
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    print(f"Loading MOSS audio tokenizer: {config.codec_name}")
    codec = AudioCodec(model_name=config.codec_name, n_quantizers=config.n_codebooks)
    if hasattr(codec.model, "to"):
        codec.model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)

    results = []
    for index, item in enumerate(items, start=1):
        prefix = numbered_prefix(index, len(items))
        token_path = Path(args.token_dir) / f"{item['id']}.pt"
        data = torch.load(token_path, map_location="cpu", weights_only=True)
        target = data["rvq"]
        text = data.get("text", item["text"])
        frames = target.shape[0]
        print(f"[{prefix}/{len(items):02d}] {item['id']} | {frames} frames | {text}")

        teacher = teacher_forced_generate(
            model, codec, str(token_path), args.device,
            use_audio_context=not args.no_audio_context,
        )
        free = generate(
            model, tokenizer, codec, text, frames, args.temperature, args.device,
            use_audio_context=not args.no_audio_context,
        )
        save_codes(codec, target, output_dir / f"{prefix}_ground_truth.wav", args.device)
        save_codes(codec, teacher, output_dir / f"{prefix}_teacher.wav", args.device)
        save_codes(codec, free, output_dir / f"{prefix}_free.wav", args.device)

        result = {
            "number": prefix,
            "id": item["id"],
            "text": text,
            "frames": frames,
            "free_uses_reference_frame_count": True,
            "uses_audio_context": not args.no_audio_context,
            "teacher": token_metrics(teacher, target, config.codebooks_per_group),
            "free": token_metrics(free, target, config.codebooks_per_group),
        }
        results.append(result)
        (output_dir / f"{prefix}.txt").write_text(
            f"id: {item['id']}\ntext: {text}\nframes: {frames}\n"
            f"teacher token accuracy: {result['teacher']['token_accuracy']:.6f}\n"
            f"free token accuracy: {result['free']['token_accuracy']:.6f}\n"
            "note: free generation uses the reference frame count\n"
        )

    with (output_dir / "evaluation.jsonl").open("w") as report:
        for result in results:
            report.write(json.dumps(result) + "\n")
    summary = {
        "utterances": len(results),
        "mean_teacher_token_accuracy": sum(r["teacher"]["token_accuracy"] for r in results) / len(results),
        "mean_teacher_frame_accuracy": sum(r["teacher"]["frame_accuracy"] for r in results) / len(results),
        "mean_free_token_accuracy": sum(r["free"]["token_accuracy"] for r in results) / len(results),
        "mean_free_frame_accuracy": sum(r["free"]["frame_accuracy"] for r in results) / len(results),
        "free_uses_reference_frame_count": True,
        "uses_audio_context": not args.no_audio_context,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {output_dir}")
    print("Listen to the numbered *_free.wav files. Compare *_ground_truth.wav only when needed.")
    print("Duration limitation: free generation uses each reference token file's frame count.")


if __name__ == "__main__":
    main()
