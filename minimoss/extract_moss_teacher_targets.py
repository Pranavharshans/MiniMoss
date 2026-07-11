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
def enrich_cache(model, cache, device: str, batch_size: int, top_k: int):
    enriched = []
    for item_index, item in enumerate(cache, start=1):
        states = item["states"].to(device=device, dtype=model.dtype)
        targets = item["rvq"].to(device=device, dtype=torch.long)
        indices_parts = []
        values_parts = []
        tokens_parts = []
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
        enriched.append({
            **item,
            "teacher_topk_indices": torch.cat(indices_parts),
            "teacher_topk_values": torch.cat(values_parts),
            "teacher_tokens": torch.cat(tokens_parts),
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
            ("teacher", item["teacher_tokens"].long()),
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
    parser.add_argument("--codec", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 < args.top_k <= 1024:
        raise ValueError("--top-k must be in [1, 1024]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        model, train_cache, args.device, args.batch_size, args.top_k
    )
    torch.save(train_enriched, output_dir / "train_distill.pt")
    print("Extracting validation teacher targets...")
    validation_enriched = enrich_cache(
        model, validation_cache, args.device, args.batch_size, args.top_k
    )
    torch.save(validation_enriched, output_dir / "validation_distill.pt")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
