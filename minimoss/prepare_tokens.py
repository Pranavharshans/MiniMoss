#!/usr/bin/env python3
"""Pre-tokenize audio files through the codec and save RVQ tokens alongside text.

Usage:
    python -m minimoss.prepare_tokens \\
        --manifest data/manifest.jsonl \\
        --token-dir data/tokens \\
        --codec descript/dac_24khz \\
        --n-codebooks 16
"""

import argparse
import json
import os
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

from .codec import AudioCodec


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize audio for MiniMoss")
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest")
    parser.add_argument("--token-dir", required=True, help="Output directory for token files")
    parser.add_argument("--codec", default="descript/dac_24khz")
    parser.add_argument("--n-codebooks", type=int, default=16)
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.token_dir, exist_ok=True)

    # Load codec
    print(f"Loading codec: {args.codec}")
    codec = AudioCodec(model_name=args.codec, n_quantizers=args.n_codebooks)
    codec.model.to(args.device)
    print(f"  sample_rate={codec.sample_rate}")

    # Load tokenizer
    print(f"Loading tokenizer from: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    # Load manifest
    with open(args.manifest) as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"Processing {len(items)} utterances")

    # Verify codec decode quality first
    print("\n--- Codec sanity check (first utterance) ---")
    first = items[0]
    wav, sr = torchaudio.load(first["wav"])
    wav = wav.to(args.device)
    if sr != codec.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, codec.sample_rate)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    codes = codec.encode(wav, n_quantizers=args.n_codebooks)
    print(f"  codes shape: {codes.shape} (B={codes.shape[0]}, nq={codes.shape[1]}, T={codes.shape[2]})")

    decoded = codec.decode(codes)
    check_path = Path(args.token_dir) / "_codec_check.wav"
    torchaudio.save(str(check_path), decoded.cpu(), codec.sample_rate)
    print(f"  saved codec check: {check_path}")
    print("  Listen to this file. If it sounds bad, fix the codec setup before training.\n")

    # Process all utterances
    output_manifest = []
    for item in tqdm(items, desc="Tokenizing"):
        utt_id = item["id"]
        text = item["text"]

        # Load + resample audio
        wav, sr = torchaudio.load(item["wav"])
        wav = wav.to(args.device)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != codec.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, codec.sample_rate)

        # Encode
        codes = codec.encode(wav, n_quantizers=args.n_codebooks)
        rvq = codes[0].transpose(0, 1).cpu()  # [nq, T] -> [T, nq]

        # Tokenize text
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        text_tokens = torch.tensor(text_tokens, dtype=torch.long)

        # Save
        out = {
            "id": utt_id,
            "text": text,
            "text_tokens": text_tokens,
            "rvq": rvq,
        }
        torch.save(out, os.path.join(args.token_dir, f"{utt_id}.pt"))

        output_manifest.append({
            "id": utt_id,
            "wav": item["wav"],
            "text": text,
            "n_frames": rvq.shape[0],
            "duration_s": wav.shape[-1] / codec.sample_rate,
        })

    # Save processed manifest
    out_manifest_path = os.path.join(args.token_dir, "token_manifest.jsonl")
    with open(out_manifest_path, "w") as f:
        for item in output_manifest:
            f.write(json.dumps(item) + "\n")

    print(f"\nDone! {len(output_manifest)} utterances tokenized.")
    print(f"Token manifest: {out_manifest_path}")
    total_frames = sum(item["n_frames"] for item in output_manifest)
    total_dur = sum(item["duration_s"] for item in output_manifest)
    print(f"Total: {total_frames} frames, {total_dur:.1f}s audio")


if __name__ == "__main__":
    main()
