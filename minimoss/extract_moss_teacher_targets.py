#!/usr/bin/env python3
"""Cache top-k logits from the pretrained 32-step MOSS local teacher."""

import argparse
import gc
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModel

from .codec import AudioCodec
from .validate_moss_teacher import DEFAULT_MODEL, DEFAULT_REVISION


def original_local_logits(model, global_states: torch.Tensor, rvq: torch.Tensor):
    """Run the original MOSS local decoder under within-frame teacher forcing."""
    batch = global_states.shape[0]
    text_code = torch.full(
        (batch,),
        model.config.audio_assistant_gen_slot_token_id,
        dtype=torch.long,
        device=global_states.device,
    )
    local_inputs = [global_states, model.model.embedding_list[0](text_code)]
    for codebook in range(model.config.n_vq - 1):
        local_inputs.append(
            model.model.embedding_list[codebook + 1](rvq[:, codebook])
        )
    local_inputs = torch.stack(local_inputs, dim=1)
    local_inputs = model.speech_embedding_to_local_mlp(local_inputs)
    local_hidden = model.local_transformer(
        input_ids=None,
        attention_mask=None,
        inputs_embeds=local_inputs,
        return_dict=True,
    ).last_hidden_state
    logits = []
    for codebook in range(model.config.n_vq):
        channel = codebook + 1
        hidden = model.layer_norm_before_lm_heads[channel](
            model.local_to_speech_embedding_mlps[channel](
                local_hidden[:, channel]
            )
        )
        logits.append(model.lm_heads[channel](hidden)[..., :model.config.audio_vocab_size])
    return logits


@torch.inference_mode()
def sample_topk(indices: torch.Tensor, values: torch.Tensor, temperature: float):
    probabilities = torch.softmax(values.float() / temperature, dim=-1)
    sampled_offsets = torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]), 1
    ).reshape(indices.shape[:-1] + (1,))
    return indices.gather(-1, sampled_offsets).squeeze(-1)


def add_sampled_tokens(cache, temperature: float):
    for item in cache:
        item["teacher_sampled_tokens"] = sample_topk(
            item["teacher_topk_indices"].long(),
            item["teacher_topk_values"].float(),
            temperature,
        ).to(dtype=torch.int16)
    return cache


def enrich_cache(
    model,
    cache,
    device: str,
    batch_size: int,
    top_k: int,
    sampling_temperature: float,
):
    enriched = []
    for item_index, item in enumerate(cache, start=1):
        states = item["states"].to(device=device, dtype=model.dtype)
        targets = item["rvq"].to(device=device, dtype=torch.long)
        indices_parts = []
        values_parts = []
        tokens_parts = []
        sampled_parts = []
        for start in range(0, states.shape[0], batch_size):
            logits = original_local_logits(
                model,
                states[start:start + batch_size],
                targets[start:start + batch_size],
            )
            stacked = torch.stack(logits, dim=1)
            values, indices = stacked.topk(top_k, dim=-1)
            indices_parts.append(indices.to(dtype=torch.int16, device="cpu"))
            values_parts.append(values.to(dtype=torch.float16, device="cpu"))
            tokens_parts.append(stacked.argmax(dim=-1).to(dtype=torch.int16, device="cpu"))
            sampled_parts.append(
                sample_topk(indices, values, sampling_temperature).to(
                    dtype=torch.int16, device="cpu"
                )
            )
        enriched.append({
            **item,
            "teacher_topk_indices": torch.cat(indices_parts),
            "teacher_topk_values": torch.cat(values_parts),
            "teacher_tokens": torch.cat(tokens_parts),
            "teacher_sampled_tokens": torch.cat(sampled_parts),
        })
        print(
            f"[{item_index:04d}/{len(cache):04d}] {item['id']} | "
            f"{states.shape[0]} frames"
        )
    return enriched


def save_teacher_audio(cache, output_dir: Path, codec_name: str, device: str, limit: int):
    codec = AudioCodec(model_name=codec_name, n_quantizers=32)
    if hasattr(codec.model, "to"):
        codec.model.to(device)
    audio_dir = output_dir / "teacher_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(cache[:limit], start=1):
        for name, codes in (
            ("ground_truth", item["rvq"].long()),
            ("teacher_greedy", item["teacher_tokens"].long()),
            ("teacher_topk_sampled", item["teacher_sampled_tokens"].long()),
        ):
            waveform = codec.decode(
                codes.transpose(0, 1).unsqueeze(0).to(device)
            )[0].cpu()
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
    parser.add_argument("--output-dir", default="evaluation/moss_teacher_distillation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--audio-limit", type=int, default=10)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-topk", action="store_true")
    parser.add_argument("--codec", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 < args.top_k <= 1024:
        raise ValueError("--top-k must be in [1, 1024]")
    if args.sampling_temperature <= 0:
        raise ValueError("--sampling-temperature must be positive")
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_output = output_dir / "train_distill.pt"
    validation_output = output_dir / "validation_distill.pt"
    if args.reuse_topk:
        if not train_output.exists() or not validation_output.exists():
            raise FileNotFoundError("--reuse-topk requires existing distillation caches")
        print("Reusing cached top-k teacher logits")
        train_enriched = add_sampled_tokens(
            torch.load(train_output, map_location="cpu", weights_only=True),
            args.sampling_temperature,
        )
        validation_enriched = add_sampled_tokens(
            torch.load(validation_output, map_location="cpu", weights_only=True),
            args.sampling_temperature,
        )
    else:
        train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=True)
        validation_cache = torch.load(
            args.validation_cache, map_location="cpu", weights_only=True
        )
        dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        print(f"Loading original MOSS local teacher: {args.model}@{args.revision}")
        model = AutoModel.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(args.device)
        model.eval()
        print("Extracting training teacher targets...")
        train_enriched = enrich_cache(
            model,
            train_cache,
            args.device,
            args.batch_size,
            args.top_k,
            args.sampling_temperature,
        )
        print("Extracting validation teacher targets...")
        validation_enriched = enrich_cache(
            model,
            validation_cache,
            args.device,
            args.batch_size,
            args.top_k,
            args.sampling_temperature,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    torch.save(train_enriched, train_output)
    torch.save(validation_enriched, validation_output)
    save_teacher_audio(
        validation_enriched,
        output_dir,
        args.codec,
        args.device,
        args.audio_limit,
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
