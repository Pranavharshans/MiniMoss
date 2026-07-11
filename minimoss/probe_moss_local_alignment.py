#!/usr/bin/env python3
"""Compare fixed-slot and per-frame channel-0 prefixes for the MOSS local decoder."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, AutoProcessor

from .extract_moss_teacher_targets import original_local_logits
from .validate_moss_teacher import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    pack_teacher_forcing,
)
from .dataset import load_manifest


def extract_item(model, processor, item, token_dir: Path, device: str):
    token_data = torch.load(
        token_dir / f"{item['id']}.pt", map_location="cpu", weights_only=True
    )
    rvq = token_data["rvq"].long()
    input_ids, attention_mask, full_targets, valid_mask = pack_teacher_forcing(
        processor, item["text"], rvq, device
    )
    with torch.inference_mode():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_vq_for_inference=processor.model_config.n_vq,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    states = outputs.last_hidden_state[valid_mask]
    targets = full_targets[..., 1:][valid_mask]
    prefix_tokens = full_targets[..., 0][valid_mask]
    if states.shape[0] != rvq.shape[0] or prefix_tokens.shape[0] != rvq.shape[0]:
        raise RuntimeError(
            f"{item['id']}: expected {rvq.shape[0]} frames, got "
            f"states={states.shape[0]}, prefixes={prefix_tokens.shape[0]}"
        )
    return (
        states.float().cpu(),
        targets.long().cpu(),
        prefix_tokens.long().cpu(),
    )


@torch.inference_mode()
def predict_prefix_mode(
    model,
    states: torch.Tensor,
    targets: torch.Tensor,
    prefix_tokens: torch.Tensor,
    batch_size: int,
):
    logits_parts = []
    for start in range(0, states.shape[0], batch_size):
        logits_parts.append(
            torch.stack(
                original_local_logits(
                    model,
                    states[start:start + batch_size].to(
                        device=model.device, dtype=model.dtype
                    ),
                    targets[start:start + batch_size].to(model.device),
                    prefix_tokens[start:start + batch_size].to(model.device),
                ),
                dim=1,
            ).float().cpu()
        )
    logits = torch.cat(logits_parts)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    predictions = logits.argmax(dim=-1)
    return {
        "loss": float(F.cross_entropy(flat_logits, flat_targets)),
        "token_accuracy": float(predictions.eq(targets).float().mean()),
        "per_codebook_accuracy": predictions.eq(targets).float().mean(dim=0).tolist(),
        "predictions": predictions,
    }


def decode(processor, codes: torch.Tensor, device: str):
    waveform = processor.decode_audio_codes([codes.to(device)])[0].float().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--output-dir", default="evaluation/moss_local_alignment")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--audio-limit", type=int, default=3)
    parser.add_argument("--min-loss-improvement", type=float, default=0.01)
    parser.add_argument("--min-accuracy-improvement", type=float, default=0.002)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.limit, args.max_frames, args.batch_size) <= 0:
        raise ValueError("--limit, --max-frames, and --batch-size must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = load_manifest(args.manifest)[:args.limit]
    if not items:
        raise ValueError("The manifest contains no items within --limit")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()
    if hasattr(processor, "audio_tokenizer") and hasattr(processor.audio_tokenizer, "to"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(args.device)

    slot_id = int(model.config.audio_assistant_gen_slot_token_id)
    fixed_stats = []
    observed_stats = []
    utterance_results = []
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(items, start=1):
        states, targets, observed_prefix = extract_item(
            model, processor, item, Path(args.token_dir), args.device
        )
        states = states[:args.max_frames]
        targets = targets[:args.max_frames]
        observed_prefix = observed_prefix[:args.max_frames]
        fixed_prefix = torch.full_like(observed_prefix, slot_id)
        fixed = predict_prefix_mode(
            model, states, targets, fixed_prefix, args.batch_size
        )
        observed = predict_prefix_mode(
            model, states, targets, observed_prefix, args.batch_size
        )
        fixed_stats.append(fixed)
        observed_stats.append(observed)
        loss_improvement = fixed["loss"] - observed["loss"]
        accuracy_improvement = observed["token_accuracy"] - fixed["token_accuracy"]
        result = {
            "id": item["id"],
            "text": item["text"],
            "frames": int(states.shape[0]),
            "fixed_prefix_token": slot_id,
            "observed_prefix_unique_tokens": int(observed_prefix.unique().numel()),
            "fixed_loss": fixed["loss"],
            "observed_loss": observed["loss"],
            "loss_improvement": loss_improvement,
            "fixed_token_accuracy": fixed["token_accuracy"],
            "observed_token_accuracy": observed["token_accuracy"],
            "accuracy_improvement": accuracy_improvement,
            "status": (
                "PASS"
                if loss_improvement >= args.min_loss_improvement
                and accuracy_improvement >= args.min_accuracy_improvement
                else "FAIL_OR_INCONCLUSIVE"
            ),
        }
        utterance_results.append(result)
        print(
            f"[{index:02d}/{len(items):02d}] {item['id']} | "
            f"fixed_loss={fixed['loss']:.4f} observed_loss={observed['loss']:.4f} | "
            f"fixed_acc={fixed['token_accuracy']:.4f} "
            f"observed_acc={observed['token_accuracy']:.4f} | {result['status']}"
        )

        if index <= args.audio_limit:
            audio_outputs = (
                ("ground_truth", targets),
                ("teacher_fixed_prefix", fixed["predictions"]),
                ("teacher_observed_prefix", observed["predictions"]),
            )
            for name, codes in audio_outputs:
                torchaudio.save(
                    str(audio_dir / f"{index:02d}_{name}.wav"),
                    decode(processor, codes, args.device),
                    processor.model_config.sampling_rate,
                )
            (audio_dir / f"{index:02d}.txt").write_text(
                f"id: {item['id']}\ntext: {item['text']}\n"
            )

    def weighted_mean(stats, key):
        weights = [result["frames"] for result in utterance_results]
        return sum(row[key] * weight for row, weight in zip(stats, weights)) / sum(weights)

    summary = {
        "utterances": len(utterance_results),
        "fixed_prefix_token": slot_id,
        "fixed_loss": weighted_mean(fixed_stats, "loss"),
        "observed_loss": weighted_mean(observed_stats, "loss"),
        "fixed_token_accuracy": weighted_mean(fixed_stats, "token_accuracy"),
        "observed_token_accuracy": weighted_mean(observed_stats, "token_accuracy"),
        "loss_improvement": weighted_mean(fixed_stats, "loss")
        - weighted_mean(observed_stats, "loss"),
        "accuracy_improvement": weighted_mean(observed_stats, "token_accuracy")
        - weighted_mean(fixed_stats, "token_accuracy"),
        "results": utterance_results,
        "audio_dir": str(audio_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
