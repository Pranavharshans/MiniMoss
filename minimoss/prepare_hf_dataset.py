#!/usr/bin/env python3
"""Create a tiny single-speaker manifest from LibriTTS-R on Hugging Face."""

import argparse
import json
from pathlib import Path

from datasets import Audio, load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download a tiny LibriTTS-R subset")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--speaker-id", default="3081")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "parler-tts/libritts_r_filtered",
        "clean",
        split="dev.clean",
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))

    records = []
    for row in dataset:
        if str(row["speaker_id"]) != args.speaker_id:
            continue

        utterance_id = str(row["id"])
        wav_path = wav_dir / f"{utterance_id}.wav"
        audio = row["audio"]
        audio_bytes = audio.get("bytes")
        if audio_bytes is None:
            source_path = audio.get("path")
            if source_path is None:
                raise RuntimeError(f"No bytes or path returned for {utterance_id}")
            wav_path.write_bytes(Path(source_path).read_bytes())
        else:
            wav_path.write_bytes(audio_bytes)

        records.append({
            "id": utterance_id,
            "wav": str(wav_path.resolve()),
            "text": row["text_normalized"],
        })
        if len(records) == args.count:
            break

    if len(records) < args.count:
        raise RuntimeError(
            f"Found only {len(records)} clips for speaker {args.speaker_id}; requested {args.count}"
        )

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w") as manifest:
        for record in records:
            manifest.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} single-speaker clips to {wav_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
