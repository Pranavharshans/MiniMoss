#!/usr/bin/env python3
"""Generation/inference script for MiniMoss.

Usage:
    # Greedy generation
    python -m minimoss.generate \\
        --checkpoint checkpoints/final.pt \\
        --text "hello world" \\
        --max-frames 200 \\
        --output generated.wav

    # Teacher-forced (decode ground truth)
    python -m minimoss.generate \\
        --checkpoint checkpoints/final.pt \\
        --teacher-forced tokens/utt_0001.pt \\
        --output teacher_forced.wav
"""

import argparse
import torch
import torch.nn.functional as F
import torchaudio

from .model import MiniMossModel
from .codec import AudioCodec
from .utils import set_seed
from transformers import AutoTokenizer


@torch.inference_mode()
def generate(
    model: MiniMossModel,
    tokenizer,
    codec: AudioCodec,
    text: str,
    max_frames: int = 200,
    temperature: float = 0.0,
    device: str = "cuda",
    use_audio_context: bool = True,
) -> torch.Tensor:
    """Free-running autoregressive generation.

    Returns:
        rvq_codes: [T_frames, n_codebooks]
    """
    model.eval()
    config = model.config
    n_groups = config.n_groups
    cbg = config.codebooks_per_group
    n_cb = config.n_codebooks

    # Encode text. Frame states are produced causally inside the frame loop.
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    text_tokens = torch.tensor([text_tokens], dtype=torch.long, device=device)
    text_mask = torch.ones_like(text_tokens)

    all_codes = []

    for frame_idx in range(max_frames):
        # Append a placeholder current frame. encode_frames shifts history so
        # this position can see only previously generated RVQ frames.
        if all_codes:
            history = torch.stack(all_codes, dim=0).unsqueeze(0)
            placeholder = torch.zeros((1, 1, n_cb), dtype=torch.long, device=device)
            frame_context = torch.cat([history, placeholder], dim=1)
        else:
            frame_context = torch.zeros((1, 1, n_cb), dtype=torch.long, device=device)
        frame_h = model.encode_frames(
            text_tokens,
            frame_context,
            text_attention_mask=text_mask,
            audio_context_dropout_prob=0.0 if use_audio_context else 1.0,
        )[:, -1, :]

        # Decoder input starts with frame hidden
        decoder_states = [frame_h]  # list of [1, D_local]
        frame_codes = []

        for g in range(n_groups):
            # Run local decoder on current sequence
            decoder_input = torch.stack(decoder_states, dim=1)  # [1, cur_len, D_local]
            group_pos = torch.arange(len(decoder_states), device=device).unsqueeze(0)
            decoder_out = model.local_decoder(decoder_input, position_ids=group_pos)
            last_hidden = decoder_out[:, -1, :]  # [1, D_local]

            # Predict 4 codebooks from this group
            group_codes = []
            for cb in range(g * cbg, (g + 1) * cbg):
                logits = model.output_heads[cb](last_hidden)  # [1, codebook_size]
                if temperature <= 0:
                    token = torch.argmax(logits, dim=-1)  # [1]
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                group_codes.append(token)
            group_tokens = torch.stack(group_codes, dim=-1)  # [1, cbg]

            # Embed predicted group for next step
            g_emb = 0
            for i in range(cbg):
                g_emb = g_emb + model.codebook_embeddings[g * cbg + i](group_tokens[:, i])
            decoder_states.append(g_emb)  # [1, D_local]
            frame_codes.append(group_tokens)

        frame_tokens = torch.cat(frame_codes, dim=-1)  # [1, n_codebooks]
        all_codes.append(frame_tokens[0])  # [n_codebooks]

    return torch.stack(all_codes, dim=0)  # [T_frames, n_codebooks]


@torch.inference_mode()
def teacher_forced_generate(
    model: MiniMossModel,
    codec: AudioCodec,
    token_path: str,
    device: str = "cuda",
    use_audio_context: bool = True,
) -> torch.Tensor:
    """Teacher-forced: use ground-truth audio codes as decoder input, decode the model's predictions.

    Returns:
        rvq_codes: [T_frames, n_codebooks]
    """
    model.eval()
    config = model.config
    n_groups = config.n_groups
    cbg = config.codebooks_per_group
    n_cb = config.n_codebooks

    data = torch.load(token_path, map_location="cpu", weights_only=True)
    text_tokens = data["text_tokens"].unsqueeze(0).to(device)
    rvq_gt = data["rvq"].unsqueeze(0).to(device)  # [1, T_audio, n_codebooks]

    B, T_audio, _ = rvq_gt.shape

    text_mask = torch.ones_like(text_tokens)
    frame_h = model.encode_frames(
        text_tokens,
        rvq_gt,
        text_attention_mask=text_mask,
        audio_context_dropout_prob=0.0 if use_audio_context else 1.0,
    )

    all_codes = []
    for t in range(T_audio):
        decoder_states = [frame_h[:, t, :]]  # [1, D_local]
        frame_codes = []

        for g in range(n_groups):
            decoder_input = torch.stack(decoder_states, dim=1)  # [1, cur_len, D_local]
            group_pos = torch.arange(len(decoder_states), device=device).unsqueeze(0)
            decoder_out = model.local_decoder(decoder_input, position_ids=group_pos)
            last_hidden = decoder_out[:, -1, :]

            group_codes = []
            for cb in range(g * cbg, (g + 1) * cbg):
                logits = model.output_heads[cb](last_hidden)
                token = torch.argmax(logits, dim=-1)
                group_codes.append(token)

            group_tokens = torch.stack(group_codes, dim=-1)

            # Use ground-truth embedding to keep on track
            gt_group = rvq_gt[:, t, g * cbg:(g + 1) * cbg]
            g_emb = 0
            for i in range(cbg):
                g_emb = g_emb + model.codebook_embeddings[g * cbg + i](gt_group[:, i])
            decoder_states.append(g_emb)
            frame_codes.append(group_tokens)

        frame_tokens = torch.cat(frame_codes, dim=-1)
        all_codes.append(frame_tokens[0])

    return torch.stack(all_codes, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Generate audio with MiniMoss")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--text", default=None, help="Text to synthesize")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--teacher-forced", default=None,
                        help="Path to pre-tokenized .pt file for teacher-forced generation")
    parser.add_argument("--no-audio-context", action="store_true",
                        help="Use the learned null previous-frame context")
    parser.add_argument("--output", default="generated.wav")
    parser.add_argument("--codec-name", default=None,
                        help="Defaults to checkpoint config.codec_name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    # Load model — force backbone materialization first, then load trainable weights
    print("Loading model...")
    model = MiniMossModel(config)
    model.to(args.device)
    _ = model.backbone  # materialize frozen backbone before loading state
    # Load only trainable weights (backbone is loaded from HuggingFace, not checkpoint)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Load codec
    codec_name = args.codec_name or config.codec_name
    print(f"Loading MOSS audio tokenizer: {codec_name}")
    codec = AudioCodec(
        model_name=codec_name,
        n_quantizers=config.n_codebooks,
    )
    if hasattr(codec.model, "to"):
        codec.model.to(args.device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)

    if args.teacher_forced:
        print(f"Teacher-forced generation from: {args.teacher_forced}")
        codes = teacher_forced_generate(
            model, codec, args.teacher_forced, device=args.device,
            use_audio_context=not args.no_audio_context,
        )
    elif args.text:
        print(f"Generating: \"{args.text}\"")
        codes = generate(
            model, tokenizer, codec,
            text=args.text,
            max_frames=args.max_frames,
            temperature=args.temperature,
            device=args.device,
            use_audio_context=not args.no_audio_context,
        )
    else:
        print("ERROR: provide --text or --teacher-forced")
        return

    print(f"Generated codes: {codes.shape}")

    # Decode to audio
    codes_for_decode = codes.transpose(0, 1).unsqueeze(0).to(args.device)  # [1, n_cb, T]
    wav = codec.decode(codes_for_decode)
    wav = wav[0].cpu()  # [1, T_samples]

    torchaudio.save(args.output, wav, codec.sample_rate)
    duration = wav.shape[-1] / codec.sample_rate
    print(f"Saved: {args.output} ({duration:.2f}s)")


if __name__ == "__main__":
    main()
